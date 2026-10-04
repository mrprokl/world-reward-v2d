// A fixed numerical chart, not a physical rescale or mesh repair.
// Both historical cores are authenticated byte prefixes; they remain unedited.
#include "wr_serialization_core.hpp"
#include <cstdlib>
#include <iomanip>

#ifndef WR_CONDITIONED_SOURCE_SHA256
#error Bind this new numerical algorithm source
#endif
#ifndef WR_SERIALIZATION_CORE_PREFIX_SHA256
#error Bind the unmodified serialization prefix
#endif
#ifndef WR_CONDITIONED_CACHE
#define WR_CONDITIONED_CACHE 0
#endif
#if WR_CONDITIONED_CACHE != 0 && WR_CONDITIONED_CACHE != 1
#error Physical coordinate cache must be explicitly zero or one
#endif

namespace conditioned_qem {
using serialization_qem::Serialization;
using serialization_qem::Transaction;

struct Chart {
  Eigen::RowVector3d origin;
  double scale=0;
  int exponent=0;
  uint64_t source_vertices=0;
  bool byte_exact=true;

  Eigen::RowVector3d encode(const Eigen::RowVector3d& p) const {
    Eigen::RowVector3d result;
    for(int j=0;j<3;++j) {
      volatile double delta=p[j]-origin[j];
      volatile double x=delta/scale; result[j]=x;
      if(!std::isfinite(delta) || !std::isfinite(x) || (delta!=0 && x==0))
        throw std::runtime_error("Chart encode overflow or nonzero underflow");
    }
    return result;
  }

  Eigen::RowVector3d decode(const Eigen::RowVector3d& p) const {
    Eigen::RowVector3d result;
    for(int j=0;j<3;++j) {
      // The exact same separate multiply/add defines pre, post and export.
      volatile double delta=scale*p[j];
      volatile double x=origin[j]+delta; result[j]=x;
      if(!std::isfinite(delta) || !std::isfinite(x) || (p[j]!=0 && delta==0))
        throw std::runtime_error("Chart decode overflow or nonzero underflow");
    }
    return result;
  }

  Eigen::MatrixXd decode_matrix(const Eigen::MatrixXd& V) const {
    Eigen::MatrixXd physical(V.rows(),3);
    for(int v=0;v<V.rows();++v) physical.row(v)=decode(Eigen::RowVector3d(V.row(v)));
    return physical;
  }

  Chart(const Eigen::MatrixXd& V,const Eigen::MatrixXi& F,Eigen::MatrixXd& canonical) {
    Eigen::RowVector3d lo=V.row(F(0,0)),hi=lo;
    for(int f=0;f<F.rows();++f) for(int j=0;j<3;++j) {
      lo=lo.cwiseMin(V.row(F(f,j))); hi=hi.cwiseMax(V.row(F(f,j)));
    }
    double maximum=0;
    for(int j=0;j<3;++j) {
      volatile double extent=hi[j]-lo[j],half=extent*.5;
      volatile double midpoint=lo[j]+half; origin[j]=midpoint;
      if(!std::isfinite(extent) || !std::isfinite(midpoint) || (extent!=0 && half==0))
        throw std::runtime_error("Chart bbox overflow or midpoint underflow");
      maximum=std::max(maximum,double(extent));
    }
    if(!(maximum>0)) throw std::runtime_error("Chart requires positive source extent");
    const double mantissa=std::frexp(maximum,&exponent);
    exponent-=int(mantissa==.5);
    if(exponent < -1074 || exponent > 1023)
      throw std::runtime_error("Chart covering scale outside finite binary64");
    scale=std::ldexp(1.,exponent);
    canonical.resize(V.rows(),3); source_vertices=V.rows();
    // This native route is deliberately stricter than the diagnostic primitive:
    // all source rows, including orphans, must roundtrip; no removal or snapping.
    for(int v=0;v<V.rows();++v) {
      const Eigen::RowVector3d p=V.row(v),x=encode(p),roundtrip=decode(x);
      for(int j=0;j<3;++j) {
        if(roundtrip[j]!=p[j]) throw std::runtime_error("Source chart roundtrip not numerically exact");
        byte_exact &= std::memcmp(&roundtrip[j],&p[j],sizeof(double))==0;
      }
      canonical.row(v)=x;
    }
  }

  void summary(std::ostream& out) const {
    out<<std::setprecision(17)<<"{\"origin\":["<<origin[0]<<','<<origin[1]<<','<<origin[2]
       <<"],\"scale\":"<<scale<<",\"scale_exponent\":"<<exponent
       <<",\"source_roundtrip_vertices\":"<<source_vertices
       <<",\"chart_scale_positive\":true,\"source_roundtrip_numerically_exact\":true,"
         "\"source_roundtrip_byte_exact\":"<<(byte_exact?"true":"false")
       <<",\"physical_geometry_rescaled\":false,\"new_numeric_algorithm\":true,"
         "\"native_qslim_implementation_reused\":true,\"chart_refitted\":false,\"adopted\":false}";
  }
};

bool simplify(const Eigen::MatrixXd& physical,const Eigen::MatrixXi& F,const Chart& chart,
              const Eigen::MatrixXd& canonical,volume_qem::Volumes& volumes,Serialization& state,
              Eigen::MatrixXd& U,Eigen::MatrixXi& G,Eigen::VectorXi& J,Eigen::VectorXi& I) {
  if(state.budget_safe()) {
    Eigen::VectorXi old_to_new; igl::remove_unreferenced(physical,F,U,G,old_to_new,I);
    J=Eigen::VectorXi::LinSpaced(F.rows(),0,F.rows()-1); return true;
  }
#if WR_CONDITIONED_CACHE
  // Decode, rather than copy physical: even signed-zero bits must match slow pre.
  Eigen::MatrixXd physicalVcache=chart.decode_matrix(canonical);
  Eigen::MatrixXd physicalCcache;
#endif
  Eigen::VectorXi EMAP; Eigen::MatrixXi E,EF,EI; igl::edge_flaps(F,E,EMAP,EF,EI);
  std::vector<std::tuple<Eigen::MatrixXd,Eigen::RowVectorXd,double>> quadrics;
  igl::per_vertex_point_to_plane_quadrics(canonical,F,EMAP,EF,EI,quadrics);
  int v1=-1,v2=-1;
  igl::decimate_cost_and_placement_callback cost;
  igl::decimate_pre_collapse_callback pre; igl::decimate_post_collapse_callback post;
  igl::qslim_optimal_collapse_edge_callbacks(E,quadrics,v1,v2,cost,pre,post);
  auto* tree=new igl::AABB<Eigen::MatrixXd,3>(); tree->init(canonical,F);
  try {
    igl::intersection_blocking_collapse_edge_callbacks(pre,post,tree,pre,post);
    const auto native_pre=pre; const auto native_post=post;
    pre=[&chart,&state,&volumes,native_pre
#if WR_CONDITIONED_CACHE
        ,&physicalVcache,&physicalCcache
#endif
        ](const auto& V,const auto& F,const auto& E,
        const auto& EMAP,const auto& EF,const auto& EI,const auto& Q,const auto& EQ,const auto& C,int e) {
      volumes.pending_shell=-1; state.pending=Transaction{};
      // Do not reject unselected Inf queue rows. Only the offered placement
      // is decoded; the full physical V defines both original safety guards.
      if(!C.row(e).allFinite()) return false;
#if WR_CONDITIONED_CACHE
      if(physicalVcache.rows()!=V.rows() || physicalVcache.cols()!=3)
        throw std::runtime_error("Physical vertex cache row count changed");
      if(physicalCcache.rows()==0) physicalCcache=Eigen::MatrixXd::Zero(C.rows(),3);
      if(physicalCcache.rows()!=C.rows() || physicalCcache.cols()!=3)
        throw std::runtime_error("Physical placement cache row count changed");
      const Eigen::MatrixXd& physicalV=physicalVcache;
      Eigen::MatrixXd& physicalC=physicalCcache;
#else
      const Eigen::MatrixXd physicalV=chart.decode_matrix(V);
      Eigen::MatrixXd physicalC=Eigen::MatrixXd::Zero(C.rows(),3);
#endif
      physicalC.row(e)=chart.decode(Eigen::RowVector3d(C.row(e)));
      return state.allowed(e,physicalV,F,E,EMAP,EF,EI,physicalC)
        && native_pre(V,F,E,EMAP,EF,EI,Q,EQ,C,e)
        && volumes.allowed(e,physicalV,F,E,EMAP,EF,EI,physicalC);
    };
    post=[&chart,&state,&volumes,&physical,native_post
#if WR_CONDITIONED_CACHE
        ,&physicalVcache
#endif
        ](const auto& V,const auto& F,const auto& E,
        const auto& EMAP,const auto& EF,const auto& EI,const auto& Q,const auto& EQ,const auto& C,
        int e,int e1,int e2,int f1,int f2,bool collapsed) {
      native_post(V,F,E,EMAP,EF,EI,Q,EQ,C,e,e1,e2,f1,f2,collapsed);
      if(collapsed) {
        if(volumes.pending_shell<0) throw std::runtime_error("Collapse lacks original physical volume approval");
        auto& shell=volumes.shells[volumes.pending_shell]; shell.current.add(volumes.pending_delta);
        shell.magnitude+=std::abs(volumes.pending_delta); ++volumes.committed;
#if WR_CONDITIONED_CACHE
        const auto& t=state.pending;
        if(!t.approved || t.s<0 || t.d<0 || t.s>=V.rows() || t.d>=V.rows()
            || physicalVcache.rows()!=V.rows())
          throw std::runtime_error("Collapsed endpoints lack approved physical cache transaction");
        const Eigen::RowVector3d s=chart.decode(Eigen::RowVector3d(V.row(t.s)));
        const Eigen::RowVector3d d=chart.decode(Eigen::RowVector3d(V.row(t.d)));
        for(int j=0;j<3;++j) if(s[j]!=t.p[j] || d[j]!=t.p[j])
          throw std::runtime_error("Cached endpoints differ from approved physical placement");
        physicalVcache.row(t.s)=s; physicalVcache.row(t.d)=d;
        state.finish(physicalVcache,F,true,f1,f2);
#else
        state.finish(chart.decode_matrix(V),F,true,f1,f2);
#endif
      } else state.finish(physical,F,false,f1,f2);
      volumes.pending_shell=-1;
    };
    const igl::decimate_stopping_condition_callback stop=[&state](const auto&,const auto&,const auto&,
        const auto&,const auto&,const auto&,const auto&,const auto&,const auto&,int,int,int,int,int) {
      return state.budget_safe();
    };
    Eigen::MatrixXd canonicalU;
    const bool reached=igl::decimate(canonical,F,cost,stop,pre,post,canonicalU,G,J,I);
    U=chart.decode_matrix(canonicalU);
#if WR_CONDITIONED_CACHE
    if(I.size()!=U.rows()) throw std::runtime_error("Terminal cache birth map row count differs");
    for(int v=0;v<U.rows();++v) {
      if(I(v)<0 || I(v)>=physicalVcache.rows()) throw std::runtime_error("Terminal cache birth index invalid");
      for(int j=0;j<3;++j) if(U(v,j)!=physicalVcache(I(v),j))
        throw std::runtime_error("Independent full decode differs from terminal physical cache");
    }
#endif
    delete tree; return reached;
  } catch(...) { delete tree; throw; }
}

std::string volume_mapping(const Eigen::MatrixXd& V,const Eigen::MatrixXi& F,
                          const Eigen::MatrixXd& U,const Eigen::MatrixXi& G,
                          const Eigen::VectorXi& J,const Eigen::VectorXi& I,
                          const volume_qem::Volumes& volumes,bool reached) {
  char* bytes=nullptr; size_t size=0;
  File stream(::open_memstream(&bytes,&size),&std::fclose);
  if(!stream) throw std::runtime_error("Unable to serialize original volume mapping");
  volume_qem::mapping_json(stream.get(),V,F,U,G,J,I,volumes,reached,true);
  if(std::fflush(stream.get()) || std::ferror(stream.get())) {
    stream.reset(); std::free(bytes); throw std::runtime_error("Original mapping serialization failed");
  }
  stream.reset(); std::unique_ptr<char,decltype(&std::free)> owned(bytes,&std::free);
  std::string result(bytes,size);
  // Change only the historical formatter's two algorithm metadata fields.
  // Numeric volumes, every I/J entry and all shell rows remain its exact bytes.
  for(const auto& pair:std::vector<std::pair<std::string,std::string>>{
      {"\"native_cost_and_placement_unchanged\":true","\"native_cost_and_placement_unchanged\":false"},
      {"\"cost_normalization\":false","\"cost_normalization\":true"}}) {
    const auto at=result.find(pair.first);
    if(at==std::string::npos || result.find(pair.first,at+1)!=std::string::npos)
      throw std::runtime_error("Original mapping metadata ABI differs");
    result.replace(at,pair.first.size(),pair.second);
  }
  return result;
}
}

int main(int argc,char** argv) {
  using namespace serialization_qem;
  try {
    arithmetic_contract();
    if(argc==2 && std::string(argv[1])=="--build-info") {
      std::cout<<"{\"source_sha256\":\""<<WR_CONDITIONED_SOURCE_SHA256
        <<"\",\"serialization_source_sha256\":\""<<WR_SERIALIZATION_SOURCE_SHA256
        <<"\",\"serialization_core_prefix_sha256\":\""<<WR_SERIALIZATION_CORE_PREFIX_SHA256
        <<"\",\"volume_core_prefix_sha256\":\""<<WR_VOLUME_CORE_PREFIX_SHA256
        <<"\",\"volume_source_sha256\":\""<<WR_VOLUME_SOURCE_SHA256
        <<"\",\"base_source_sha256\":\""<<WR_BASE_SOURCE_SHA256
        <<"\",\"libigl_revision\":\""<<WR_LIBIGL_REVISION<<"\",\"eigen_revision\":\""<<WR_EIGEN_REVISION
        <<"\",\"weld_digits\":8,\"rounding\":\"ties-to-even\",\"referenced_only\":true,"
          "\"representative\":\"first original vertex index\",\"exact_arithmetic\":\"cpp_int float32 units 2^-149\","
          "\"native_cost_and_placement_unchanged\":false,\"new_numeric_algorithm\":true,"
          "\"native_qslim_implementation_reused\":true,\"cost_normalization\":true,"
          "\"physical_geometry_rescaled\":false,\"block_intersections\":true,"
          "\"volume_relative_limit\":0.05,\"target_faces\":4096,\"prepared_only\":true,\"adopted\":false,"
          "\"physical_coordinate_cache\":"<<(WR_CONDITIONED_CACHE?"true":"false")<<"}\n";
      return 0;
    }
    if(argc==3 && std::string(argv[1])=="--preflight") {
      Eigen::MatrixXd V; Eigen::MatrixXi F; read_obj(argv[2],V,F); Serialization state(V,F);
      summary(std::cout,state); std::cout<<'\n'; return 0;
    }
    if((argc==5 && std::string(argv[1])=="--position-key")
        || (argc==20 && std::string(argv[1])=="--triangle-predicates")) {
      std::array<FloatPosition,3> a,b;
      const bool key_only=argc==5;
      for(int point=0;point<(key_only?1:6);++point) {
        Position p;
        for(int j=0;j<3;++j) {
          std::istringstream token(argv[2+point*3+j]); token.imbue(std::locale::classic()); std::string extra;
          if(!(token>>p[j]) || (token>>extra) || !std::isfinite(p[j]))
            throw std::runtime_error("Scalar controls require finite coordinates");
        }
        (point<3?a[point]:b[point-3])=quantize(p);
      }
      if(key_only) {
        const auto k=key(a[0]); std::cout<<"{\"key\":["<<k[0]<<','<<k[1]<<','<<k[2]<<"],\"adopted\":false}\n";
      } else {
        const auto old_normal=normal(a),new_normal=normal(b);
        std::cout<<"{\"before_float32_exactly_active\":"<<(active(old_normal)?"true":"false")
          <<",\"after_float32_exactly_active\":"<<(active(new_normal)?"true":"false")
          <<",\"exact_normal_dot_positive\":"<<(orientation(old_normal,new_normal)?"true":"false")
          <<",\"adopted\":false}\n";
      }
      return 0;
    }
    if(argc!=4) throw std::runtime_error("Usage: mesh_conditioned_qem input.obj output.obj mapping.json");
    require_absent(argv[2]); require_absent(argv[3]);
    if(std::string(argv[2])==argv[3]) throw std::runtime_error("Output paths must differ");
    Eigen::MatrixXd V,canonical,U; Eigen::MatrixXi F,G; Eigen::VectorXi J,I;
    read_obj(argv[1],V,F); Serialization state(V,F); volume_qem::Volumes volumes(V,F);
    const conditioned_qem::Chart chart(V,F,canonical);
    const bool reached=conditioned_qem::simplify(V,F,chart,canonical,volumes,state,U,G,J,I);
    if(U.cols()!=3 || G.cols()!=3 || !U.allFinite() || U.rows()<4 || G.rows()<4
        || J.size()!=G.rows() || I.size()!=U.rows() || G.minCoeff()<0 || G.maxCoeff()>=U.rows()
        || J.minCoeff()<0 || J.maxCoeff()>=F.rows() || I.minCoeff()<0 || I.maxCoeff()>=V.rows())
      throw std::runtime_error("Invalid conditioned output or full native birth maps");
    volumes.verify_final(U,G,J,I); Serialization independent(U,G);
    if(independent.live_faces!=state.live_faces || independent.live_vertices!=state.live_vertices
        || independent.bad.size()!=state.bad.size()) throw std::runtime_error("Final physical serialization differs");
    if(!reached || !independent.budget_safe()) throw std::runtime_error("Conditioned queue exhausted before budget AND safety");
    const std::string volume=conditioned_qem::volume_mapping(V,F,U,G,J,I,volumes,reached);
    auto out=exclusive_file(argv[2]);
    for(int v=0;v<U.rows();++v) std::fprintf(out.get(),"v %.17g %.17g %.17g\n",U(v,0),U(v,1),U(v,2));
    for(int f=0;f<G.rows();++f) std::fprintf(out.get(),"f %d %d %d\n",G(f,0)+1,G(f,1)+1,G(f,2)+1);
    if(std::fflush(out.get()) || std::ferror(out.get())) throw std::runtime_error("Physical OBJ write failed");
    auto mapping=exclusive_file(argv[3]); std::ostringstream details;
    details<<"{\"conditioning\":"; chart.summary(details); details<<",\"serialization\":";
    summary(details,state); details<<",\"native_volume\":"<<volume<<"}\n";
    std::fputs(details.str().c_str(),mapping.get());
    if(std::fflush(mapping.get()) || std::ferror(mapping.get())) throw std::runtime_error("Physical mapping write failed");
    return 0;
  } catch(const std::exception& error) {
    std::cerr<<"mesh_conditioned_qem: "<<error.what()<<'\n'; return 2;
  }
}
