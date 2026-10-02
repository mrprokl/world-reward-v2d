// Own cumulative signed shell-volume veto; native libigl QSlim cost is unchanged.
// Reuse the frozen parser/export helpers, never copy or edit their source.
#define main wr_guarded_base_main
#include "/opt/world-reward/guarded-qem/mesh_guarded_qem.cpp"
#undef main
#include <igl/circulation.h>
#include <igl/decimate.h>
#include <igl/edge_flaps.h>
#include <igl/intersection_blocking_collapse_edge_callbacks.h>
#include <igl/max_faces_stopping_condition.h>
#include <igl/per_vertex_point_to_plane_quadrics.h>
#include <igl/qslim_optimal_collapse_edge_callbacks.h>
#include <numeric>

#ifndef WR_VOLUME_SOURCE_SHA256
#error Build must bind the volume wrapper source SHA256
#endif
#ifndef WR_BASE_SOURCE_SHA256
#error Build must bind the unchanged inherited base source SHA256
#endif

namespace volume_qem {
constexpr long double RELATIVE_VOLUME_LIMIT = .05L;
using Vec3 = Eigen::Matrix<long double, 3, 1>;

struct Sum {
  long double sum = 0, correction = 0;
  void add(long double x) {
    const long double y = x - correction, t = sum + y;
    correction = (t - sum) - y; sum = t;
  }
};

struct Shell {
  Vec3 origin = Vec3::Zero();
  long double source = 0, final_volume = 0, magnitude = 0;
  Sum current;
  uint64_t vetoes = 0;
};

struct Volumes {
  std::vector<int> face_shell, vertex_shell;
  std::vector<Shell> shells;
  uint64_t checks = 0, vetoes = 0, committed = 0;
  int pending_shell = -1;
  long double pending_delta = 0;

  Volumes(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F) {
    std::vector<int> parent(V.rows()), used(V.rows(), 0), labels(V.rows(), -1);
    std::iota(parent.begin(), parent.end(), 0);
    const auto root = [&parent](int a) {
      while (parent[a] != a) { parent[a] = parent[parent[a]]; a = parent[a]; }
      return a;
    };
    for (int f = 0; f < F.rows(); ++f) for (int j = 0; j < 3; ++j) {
      const int a = root(F(f,0)), b = root(F(f,j));
      parent[std::max(a,b)] = std::min(a,b); used[F(f,j)] = 1;
    }
    vertex_shell.resize(V.rows(), -1);
    for (int v = 0; v < V.rows(); ++v) if (used[v]) {
      const int r = root(v);
      if (labels[r] < 0) { labels[r] = int(shells.size()); shells.emplace_back(); }
      vertex_shell[v] = labels[r];
    }
    std::vector<int> counts(shells.size(), 0);
    for (int v = 0; v < V.rows(); ++v) if (used[v]) {
      shells[vertex_shell[v]].origin += V.row(v).transpose().cast<long double>();
      ++counts[vertex_shell[v]];
    }
    for (size_t k = 0; k < shells.size(); ++k) shells[k].origin /= counts[k];
    face_shell.resize(F.rows()); std::vector<Sum> sums(shells.size());
    for (int f = 0; f < F.rows(); ++f) {
      const int k = vertex_shell[F(f,0)]; face_shell[f] = k;
      const long double value = face_volume(V, F, f, k);
      sums[k].add(value); shells[k].magnitude += std::abs(value);
    }
    for (size_t k = 0; k < shells.size(); ++k) {
      shells[k].source = sums[k].sum; shells[k].current.add(sums[k].sum);
      if (!std::isfinite(sums[k].sum) || sums[k].sum == 0)
        throw std::runtime_error("Source shell has zero/nonfinite signed volume");
    }
  }

  long double face_volume(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F,
                          int f, int k, int s = -1, int d = -1,
                          const Eigen::RowVectorXd* placement = nullptr) const {
    Vec3 p[3];
    for (int j = 0; j < 3; ++j) {
      const int v = F(f,j);
      if (placement && (v == s || v == d)) p[j] = placement->transpose().cast<long double>();
      else p[j] = V.row(v).transpose().cast<long double>();
      p[j] -= shells[k].origin;
    }
    return p[0].dot(p[1].cross(p[2])) / 6.L;
  }

  bool allowed(int e, const Eigen::MatrixXd& V, const Eigen::MatrixXi& F,
               const Eigen::MatrixXi& E, const Eigen::VectorXi& EMAP,
               const Eigen::MatrixXi& EF, const Eigen::MatrixXi& EI,
               const Eigen::MatrixXd& C) {
    ++checks;
    const int s = E(e,0), d = E(e,1), k = vertex_shell[s];
    if (k < 0 || vertex_shell[d] != k)
      throw std::runtime_error("Native edge crosses original source shells");
    auto faces = igl::circulation(e, true, EMAP, EF, EI);
    const auto other = igl::circulation(e, false, EMAP, EF, EI);
    faces.insert(faces.end(), other.begin(), other.end());
    std::sort(faces.begin(), faces.end()); faces.erase(std::unique(faces.begin(), faces.end()), faces.end());
    const Eigen::RowVectorXd p = C.row(e);
    if (p.size() != 3 || !p.allFinite()) throw std::runtime_error("Nonfinite native placement");
    Sum delta;
    for (int f : faces) {
      // NULL is the ENTIRE row (0,0,0), not any use of real vertex index zero.
      if (f < 0 || f >= F.rows() || (F.row(f).array() == IGL_COLLAPSE_EDGE_NULL).all()
          || face_shell[f] != k) throw std::runtime_error("Invalid live one-ring/source-shell binding");
      delta.add(face_volume(V,F,f,k,s,d,&p) - face_volume(V,F,f,k));
    }
    const long double proposed = shells[k].current.sum + delta.sum;
    if (!std::isfinite(proposed) || proposed * shells[k].source <= 0
        || std::abs(proposed - shells[k].source) / std::abs(shells[k].source) > RELATIVE_VOLUME_LIMIT) {
      ++vetoes; ++shells[k].vetoes; return false;
    }
    pending_shell = k; pending_delta = delta.sum; return true;
  }

  void verify_final(const Eigen::MatrixXd& U, const Eigen::MatrixXi& G,
                    const Eigen::VectorXi& J, const Eigen::VectorXi& I) {
    std::vector<Sum> final(shells.size());
    for (int f = 0; f < G.rows(); ++f) {
      const int k = face_shell[J(f)];
      for (int j = 0; j < 3; ++j) if (vertex_shell[I(G(f,j))] != k)
        throw std::runtime_error("Output birth vertex crosses original source shell");
      const long double term = face_volume(U,G,f,k);
      final[k].add(term); shells[k].magnitude += std::abs(term);
    }
    for (size_t k = 0; k < shells.size(); ++k) shells[k].final_volume = final[k].sum;
    for (size_t k = 0; k < shells.size(); ++k) {
      auto& shell = shells[k];
      const long double allowance = 128.L * std::numeric_limits<long double>::epsilon()
        * (shell.magnitude + std::abs(shell.source));
      if (std::abs(shell.current.sum - shell.final_volume) > allowance)
        throw std::runtime_error("Cumulative and independently recomputed signed volumes disagree");
      if (!std::isfinite(shell.final_volume) || shell.final_volume * shell.source <= 0
          || std::abs(shell.final_volume - shell.source) / std::abs(shell.source) > RELATIVE_VOLUME_LIMIT)
        throw std::runtime_error("Final original shell violates unchanged five-percent volume band");
    }
  }
};

bool simplify(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, Volumes& state,
              Eigen::MatrixXd& U, Eigen::MatrixXi& G, Eigen::VectorXi& J, Eigen::VectorXi& I) {
  // Closed input only: native connect_boundary_to_infinity would add only an
  // unused infinite vertex. Omitting it leaves every active quadric unchanged.
  Eigen::VectorXi EMAP; Eigen::MatrixXi E, EF, EI;
  igl::edge_flaps(F,E,EMAP,EF,EI);
  std::vector<std::tuple<Eigen::MatrixXd,Eigen::RowVectorXd,double>> quadrics;
  igl::per_vertex_point_to_plane_quadrics(V,F,EMAP,EF,EI,quadrics);
  int v1 = -1, v2 = -1;
  igl::decimate_cost_and_placement_callback cost;
  igl::decimate_pre_collapse_callback pre;
  igl::decimate_post_collapse_callback post;
  igl::qslim_optimal_collapse_edge_callbacks(E,quadrics,v1,v2,cost,pre,post);
  auto* tree = new igl::AABB<Eigen::MatrixXd,3>(); tree->init(V,F);
  try {
    igl::intersection_blocking_collapse_edge_callbacks(pre,post,tree,pre,post);
    const auto guarded_pre = pre; const auto guarded_post = post;
    pre = [&state,guarded_pre](const auto& V, const auto& F, const auto& E, const auto& EMAP,
        const auto& EF, const auto& EI, const auto& Q, const auto& EQ, const auto& C, int e) {
      state.pending_shell = -1;
      return guarded_pre(V,F,E,EMAP,EF,EI,Q,EQ,C,e) && state.allowed(e,V,F,E,EMAP,EF,EI,C);
    };
    post = [&state,guarded_post](const auto& V, const auto& F, const auto& E, const auto& EMAP,
        const auto& EF, const auto& EI, const auto& Q, const auto& EQ, const auto& C,
        int e, int e1, int e2, int f1, int f2, bool collapsed) {
      guarded_post(V,F,E,EMAP,EF,EI,Q,EQ,C,e,e1,e2,f1,f2,collapsed);
      if (collapsed) {
        if (state.pending_shell < 0) throw std::runtime_error("Collapse committed without volume approval");
        auto& shell = state.shells[state.pending_shell]; shell.current.add(state.pending_delta);
        shell.magnitude += std::abs(state.pending_delta); ++state.committed;
      }
      state.pending_shell = -1;
    };
    int m = F.rows();
    const bool reached = igl::decimate(V,F,cost,igl::max_faces_stopping_condition(m,F.rows(),TARGET),pre,post,U,G,J,I);
    delete tree; return reached;
  } catch (...) { delete tree; throw; }
}

void mapping_json(FILE* stream, const Eigen::MatrixXd& V, const Eigen::MatrixXi& F,
                  const Eigen::MatrixXd& U, const Eigen::MatrixXi& G,
                  const Eigen::VectorXi& J, const Eigen::VectorXi& I,
                  const Volumes& state, bool reached, bool verified) {
  std::fprintf(stream,"{\"source_faces\":%d,\"source_vertices\":%d,\"output_faces\":%d,\"output_vertices\":%d,"
    "\"target_faces\":4096,\"target_reached\":%s,\"budget_met\":%s,\"mapping_complete\":true,"
    "\"block_intersections\":true,\"native_cost_and_placement_unchanged\":true,\"cost_normalization\":false,"
    "\"volume_relative_limit\":0.05,\"final_shell_volumes_verified\":%s,"
    "\"volume_checks\":%llu,\"volume_vetoes\":%llu,\"committed_collapses\":%llu,\"J\":",
    int(F.rows()),int(V.rows()),int(G.rows()),int(U.rows()),reached?"true":"false",
    U.rows()<=TARGET && G.rows()<=TARGET?"true":"false",verified?"true":"false",
    (unsigned long long)state.checks,(unsigned long long)state.vetoes,(unsigned long long)state.committed);
  vector_json(stream,J,true); std::fputs(",\"I\":",stream); vector_json(stream,I,true);
  std::fputs(",\"shells\":[",stream);
  for (size_t k = 0; k < state.shells.size(); ++k) {
    const auto& shell = state.shells[k];
    std::fprintf(stream,"%s{\"source_component\":%zu,\"source_signed_volume\":%.21Lg,"
      "\"cumulative_signed_volume\":%.21Lg,\"final_signed_volume\":%.21Lg,"
      "\"relative_volume_error\":%.21Lg,\"volume_vetoes\":%llu}",k?",":"",k,shell.source,
      shell.current.sum,shell.final_volume,std::abs(shell.final_volume-shell.source)/std::abs(shell.source),
      (unsigned long long)shell.vetoes);
  }
  std::fputs("]}\n",stream);
}
}

int main(int argc, char** argv) {
  try {
    if (argc == 2 && std::string(argv[1]) == "--build-info") {
      std::cout << "{\"libigl_revision\":\"" << WR_LIBIGL_REVISION << "\",\"eigen_revision\":\""
        << WR_EIGEN_REVISION << "\",\"source_sha256\":\"" << WR_VOLUME_SOURCE_SHA256
        << "\",\"base_source_sha256\":\"" << WR_BASE_SOURCE_SHA256
        << "\",\"target_faces\":4096,\"block_intersections\":true,\"volume_relative_limit\":0.05,"
        "\"native_cost_and_placement_unchanged\":true,\"cost_normalization\":false}\n"; return 0;
    }
    if (argc != 4) throw std::runtime_error("Usage: mesh_volume_qem input.obj output.obj mapping.json");
    require_absent(argv[2]); require_absent(argv[3]);
    if (std::string(argv[2]) == argv[3]) throw std::runtime_error("Output paths must differ");
    Eigen::MatrixXd V,U; Eigen::MatrixXi F,G; Eigen::VectorXi J,I;
    read_obj(argv[1],V,F); volume_qem::Volumes state(V,F);
    const bool reached = volume_qem::simplify(V,F,state,U,G,J,I);
    if (U.cols()!=3 || G.cols()!=3 || !U.allFinite() || U.rows()<4 || G.rows()<4
        || J.size()!=G.rows() || I.size()!=U.rows() || G.minCoeff()<0 || G.maxCoeff()>=U.rows()
        || J.minCoeff()<0 || J.maxCoeff()>=F.rows() || I.minCoeff()<0 || I.maxCoeff()>=V.rows())
      throw std::runtime_error("Invalid native output or birth maps");
    std::string volume_error;
    try { state.verify_final(U,G,J,I); } catch (const std::exception& error) { volume_error=error.what(); }
    auto out=exclusive_file(argv[2]);
    for (int i=0;i<U.rows();++i) std::fprintf(out.get(),"v %.17g %.17g %.17g\n",U(i,0),U(i,1),U(i,2));
    for (int i=0;i<G.rows();++i) std::fprintf(out.get(),"f %d %d %d\n",G(i,0)+1,G(i,1)+1,G(i,2)+1);
    if (std::fflush(out.get()) || std::ferror(out.get())) throw std::runtime_error("OBJ write failed");
    auto map=exclusive_file(argv[3]); volume_qem::mapping_json(map.get(),V,F,U,G,J,I,state,reached,volume_error.empty());
    if (std::fflush(map.get()) || std::ferror(map.get())) throw std::runtime_error("Mapping write failed");
    if (!volume_error.empty()) throw std::runtime_error(volume_error);
    return reached && U.rows()<=TARGET && G.rows()<=TARGET ? 0 : 3;
  } catch (const std::exception& error) {
    std::cerr << "mesh_volume_qem: " << error.what() << '\n'; return 2;
  }
}
