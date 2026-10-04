// Prepared adapter only. wr_volume_core.hpp is the authenticated original
// mesh_volume_qem.cpp prefix before its unique main, not an edited upstream file.
#include "wr_volume_core.hpp"
#include <boost/multiprecision/cpp_int.hpp>
#include <igl/remove_unreferenced.h>
#include <cfenv>
#include <cstring>
#include <map>
#include <set>

#ifndef WR_SERIALIZATION_SOURCE_SHA256
#error Bind the new serialization adapter source
#endif
#ifndef WR_VOLUME_CORE_PREFIX_SHA256
#error Bind the generated immutable original volume prefix
#endif
#ifdef __FAST_MATH__
#error Exact serialization predicates forbid fast-math
#endif

namespace serialization_qem {
using Integer = boost::multiprecision::cpp_int;
using Position = std::array<double,3>;
using FloatPosition = std::array<float,3>;
using Key = std::array<int64_t,3>;
using Pair = std::pair<int,int>;
using Bucket = std::set<int>;
using Pairs = std::set<Pair>;
using Triangle = std::array<int,3>;

void arithmetic_contract() {
  if (sizeof(float)!=4 || sizeof(double)!=8 || !std::numeric_limits<float>::is_iec559
      || !std::numeric_limits<double>::is_iec559 || std::numeric_limits<float>::digits!=24
      || std::numeric_limits<double>::digits!=53 || std::fegetround()!=FE_TONEAREST)
    throw std::runtime_error("Require IEEE F32/F64 and FE_TONEAREST, no arithmetic waiver");
  volatile float smallest=std::numeric_limits<float>::denorm_min(), one=1.f;
  volatile float preserved=smallest*one;
  if(!(smallest>0.f) || preserved!=smallest)
    throw std::runtime_error("Require gradual float32 underflow, no flush-to-zero");
}

FloatPosition quantize(const Position& p) {
  FloatPosition q;
  for(int j=0;j<3;++j) {
    volatile float stored = static_cast<float>(p[j]); q[j]=stored;
    if(!std::isfinite(p[j]) || !std::isfinite(q[j]))
      throw std::runtime_error("Nonfinite/unrepresentable referenced float32 position");
  }
  return q;
}

Key key(const FloatPosition& q) {
  Key result;
  for(int j=0;j<3;++j) {
    volatile double product=static_cast<double>(q[j])*100000000.;
    const double value=product, lower=std::floor(value), fraction=value-lower;
    double rounded=lower;
    if(fraction>.5 || (fraction==.5 && std::fmod(lower,2.)!=0.)) rounded+=1.;
    if(!std::isfinite(rounded) || rounded < -0x1p63 || rounded >= 0x1p63)
      throw std::runtime_error("Default8 weld key outside supported int64 range");
    result[j]=static_cast<int64_t>(rounded);
  }
  return result;
}

Integer units(float value) {
  // Every finite float32 is an integer times 2^-149, including subnormals.
  uint32_t bits; std::memcpy(&bits,&value,sizeof(bits));
  const uint32_t exponent=(bits>>23)&255, fraction=bits&0x7fffff;
  if(exponent==255) throw std::runtime_error("Nonfinite exact dyadic operand");
  Integer result=exponent ? (fraction|0x800000) : fraction;
  if(exponent) result<<=(exponent-1);
  return (bits>>31) ? -result : result;
}

std::array<Integer,3> normal(const std::array<FloatPosition,3>& p) {
  std::array<Integer,3> a,b,n;
  for(int j=0;j<3;++j) { a[j]=units(p[1][j])-units(p[0][j]); b[j]=units(p[2][j])-units(p[0][j]); }
  n[0]=a[1]*b[2]-a[2]*b[1]; n[1]=a[2]*b[0]-a[0]*b[2]; n[2]=a[0]*b[1]-a[1]*b[0];
  return n;
}
bool active(const std::array<Integer,3>& n) { return n[0]!=0 || n[1]!=0 || n[2]!=0; }
bool orientation(const std::array<Integer,3>& before,const std::array<Integer,3>& after) {
  return active(after) && before[0]*after[0]+before[1]*after[1]+before[2]*after[2]>0;
}
bool null_face(const Eigen::MatrixXi& F,int f) {
  return (F.row(f).array()==IGL_COLLAPSE_EDGE_NULL).all();
}
Position position(const Eigen::MatrixXd& V,int v) { return {V(v,0),V(v,1),V(v,2)}; }
Triangle triangle(const Eigen::MatrixXi& F,int f) { return {F(f,0),F(f,1),F(f,2)}; }

struct Transaction {
  bool approved=false;
  int s=-1,d=-1;
  Position p{}; FloatPosition q{}; Key new_key{};
  std::map<int,Triangle> before_faces,after_faces;
  std::set<int> deleted;
  std::map<int,int64_t> references;
  std::map<Key,Bucket> buckets;
  Pairs before_bad,after_bad;
};

struct Serialization {
  std::vector<Position> positions;
  std::vector<FloatPosition> stored;
  std::vector<Key> keys;
  std::vector<int64_t> references;
  std::map<Key,Bucket> buckets;
  Pairs bad;
  int live_faces=0,live_vertices=0;
  uint64_t checks=0,vetoes=0,committed=0;
  bool source_active=true;
  Transaction pending;

  Pairs bad_pairs(const Bucket& bucket,const Transaction* t=nullptr) const {
    Pairs pairs;
    for(auto a=bucket.begin();a!=bucket.end();++a) {
      auto b=a; for(++b;b!=bucket.end();++b) {
        const auto& pa=t && (*a==t->s || *a==t->d) ? t->p : positions[*a];
        const auto& pb=t && (*b==t->s || *b==t->d) ? t->p : positions[*b];
        // Exact numeric seams include signed zero; do not certify byte equality.
        if(pa!=pb) pairs.emplace(*a,*b);
      }
    }
    return pairs;
  }

  Serialization(const Eigen::MatrixXd& V,const Eigen::MatrixXi& F)
      : positions(V.rows()),stored(V.rows()),keys(V.rows()),references(V.rows(),0) {
    arithmetic_contract();
    for(int v=0;v<V.rows();++v) positions[v]=position(V,v);
    for(int f=0;f<F.rows();++f) {
      if(null_face(F,f)) throw std::runtime_error("Source contains NULL/repeated-index face");
      ++live_faces;
      for(int j=0;j<3;++j) ++references[F(f,j)];
    }
    for(int v=0;v<V.rows();++v) if(references[v]) {
      ++live_vertices; stored[v]=quantize(positions[v]); keys[v]=key(stored[v]); buckets[keys[v]].insert(v);
    }
    for(int f=0;f<F.rows();++f) {
      std::array<FloatPosition,3> p;
      for(int j=0;j<3;++j) p[j]=stored[F(f,j)];
      if(!active(normal(p))) source_active=false;
    }
    if(!source_active) throw std::runtime_error("Source has an exactly inactive float32 triangle");
    for(const auto& item:buckets) { auto pairs=bad_pairs(item.second); bad.insert(pairs.begin(),pairs.end()); }
  }

  bool safe() const { return source_active && bad.empty(); }
  bool budget_safe() const { return safe() && live_faces<=TARGET && live_vertices<=TARGET; }

  bool allowed(int e,const Eigen::MatrixXd& V,const Eigen::MatrixXi& F,
               const Eigen::MatrixXi& E,const Eigen::VectorXi& EMAP,
               const Eigen::MatrixXi& EF,const Eigen::MatrixXi& EI,const Eigen::MatrixXd& C) {
    pending=Transaction{}; ++checks;
    Transaction t; t.s=std::min(E(e,0),E(e,1)); t.d=std::max(E(e,0),E(e,1));
    if(t.s<0 || t.d>=int(references.size()) || t.s==t.d || !references[t.s] || !references[t.d])
      throw std::runtime_error("Invalid live native edge endpoints");
    if(position(V,t.s)!=positions[t.s] || position(V,t.d)!=positions[t.d])
      throw std::runtime_error("Native live endpoints differ from serialized state");
    t.references.emplace(t.s,references[t.s]); t.references.emplace(t.d,references[t.d]);
    t.p=position(C,e); t.q=quantize(t.p); t.new_key=key(t.q);
    t.deleted.insert(EF(e,0)); t.deleted.insert(EF(e,1));
    if(t.deleted.size()!=2) throw std::runtime_error("Closed native collapse must delete two distinct faces");
    auto faces=igl::circulation(e,true,EMAP,EF,EI);
    auto other=igl::circulation(e,false,EMAP,EF,EI); faces.insert(faces.end(),other.begin(),other.end());
    std::sort(faces.begin(),faces.end()); faces.erase(std::unique(faces.begin(),faces.end()),faces.end());
    for(int f:faces) {
      if(f<0 || f>=F.rows() || null_face(F,f)) throw std::runtime_error("Invalid live one-ring face");
      const auto old=triangle(F,f); t.before_faces[f]=old;
      bool incident=false;
      for(int v:old) {
        if(v<0 || v>=int(references.size()) || !references[v] || position(V,v)!=positions[v])
          throw std::runtime_error("Invalid live face/reference/position binding");
        incident|=v==t.s || v==t.d; t.references.emplace(v,references[v]); --t.references[v];
      }
      if(!incident) throw std::runtime_error("Unrelated face in native one-ring");
      if(t.deleted.count(f)) {
        if(std::find(old.begin(),old.end(),t.s)==old.end() || std::find(old.begin(),old.end(),t.d)==old.end())
          throw std::runtime_error("Deleted face does not contain both native endpoints");
        t.after_faces[f]={IGL_COLLAPSE_EDGE_NULL,IGL_COLLAPSE_EDGE_NULL,IGL_COLLAPSE_EDGE_NULL};
        continue;
      }
      Triangle after=old; std::array<FloatPosition,3> a,b;
      for(int j=0;j<3;++j) {
        a[j]=stored[old[j]]; if(after[j]==t.d) after[j]=t.s;
        b[j]=(after[j]==t.s) ? t.q : stored[after[j]];
        t.references.emplace(after[j],references[after[j]]); ++t.references[after[j]];
      }
      if(!orientation(normal(a),normal(b))) { ++vetoes; return false; }
      t.after_faces[f]=after;
    }
    for(int f:t.deleted) if(!t.before_faces.count(f)) throw std::runtime_error("Deleted face absent from one-ring");
    if(t.references[t.d]!=0 || t.references[t.s]<=0) throw std::runtime_error("Native endpoint reference lifecycle differs");
    for(const auto& item:t.references) {
      const int v=item.first;
      if(item.second<0) throw std::runtime_error("Negative staged live reference count");
      t.buckets.emplace(keys[v],buckets.at(keys[v]));
      if(v==t.s || v==t.d) {
        const auto current=buckets.find(t.new_key);
        t.buckets.emplace(t.new_key,current==buckets.end()?Bucket{}:current->second);
      }
    }
    for(const auto& item:t.buckets) { auto pairs=bad_pairs(item.second); t.before_bad.insert(pairs.begin(),pairs.end()); }
    for(const auto& item:t.references) {
      const int v=item.first; t.buckets.at(keys[v]).erase(v);
      if(item.second) t.buckets.at(v==t.s || v==t.d ? t.new_key : keys[v]).insert(v);
    }
    for(const auto& item:t.buckets) {
      auto pairs=bad_pairs(item.second,&t);
      for(const auto& pair:pairs) if(!bad.count(pair)) { ++vetoes; return false; }
      t.after_bad.insert(pairs.begin(),pairs.end());
    }
    if(t.after_bad.size()>t.before_bad.size()) throw std::runtime_error("Collision transaction unexpectedly increases count");
    t.approved=true; pending=std::move(t); return true;
  }

  void finish(const Eigen::MatrixXd& V,const Eigen::MatrixXi& F,bool collapsed,int f1,int f2) {
    if(!collapsed) { pending=Transaction{}; return; }
    auto& t=pending;
    if(!t.approved || t.deleted!=std::set<int>{f1,f2} || position(V,t.s)!=t.p || position(V,t.d)!=t.p)
      throw std::runtime_error("Native collapse differs from approved serialization transaction");
    for(const auto& item:t.after_faces) if(triangle(F,item.first)!=item.second)
      throw std::runtime_error("Native live/NULL faces differ from staged transaction");
    for(const auto& item:t.references) {
      live_vertices+=(item.second>0)-(references[item.first]>0); references[item.first]=item.second;
    }
    for(int v:{t.s,t.d}) { positions[v]=t.p; stored[v]=t.q; keys[v]=t.new_key; }
    for(auto& item:t.buckets) {
      if(item.second.empty()) buckets.erase(item.first); else buckets[item.first]=std::move(item.second);
    }
    for(const auto& pair:t.before_bad) bad.erase(pair);
    bad.insert(t.after_bad.begin(),t.after_bad.end()); live_faces-=int(t.deleted.size()); ++committed;
    pending=Transaction{};
  }
};

bool simplify(const Eigen::MatrixXd& V,const Eigen::MatrixXi& F,volume_qem::Volumes& volumes,
              Serialization& serialization,Eigen::MatrixXd& U,Eigen::MatrixXi& G,
              Eigen::VectorXi& J,Eigen::VectorXi& I) {
  if(serialization.budget_safe()) {
    Eigen::VectorXi old_to_new; igl::remove_unreferenced(V,F,U,G,old_to_new,I);
    J=Eigen::VectorXi::LinSpaced(F.rows(),0,F.rows()-1); return true;
  }
  Eigen::VectorXi EMAP; Eigen::MatrixXi E,EF,EI; igl::edge_flaps(F,E,EMAP,EF,EI);
  std::vector<std::tuple<Eigen::MatrixXd,Eigen::RowVectorXd,double>> quadrics;
  igl::per_vertex_point_to_plane_quadrics(V,F,EMAP,EF,EI,quadrics);
  int v1=-1,v2=-1;
  igl::decimate_cost_and_placement_callback cost;
  igl::decimate_pre_collapse_callback pre; igl::decimate_post_collapse_callback post;
  igl::qslim_optimal_collapse_edge_callbacks(E,quadrics,v1,v2,cost,pre,post);
  auto* tree=new igl::AABB<Eigen::MatrixXd,3>(); tree->init(V,F);
  try {
    igl::intersection_blocking_collapse_edge_callbacks(pre,post,tree,pre,post);
    const auto native_pre=pre; const auto native_post=post;
    pre=[&serialization,&volumes,native_pre](const auto& V,const auto& F,const auto& E,
        const auto& EMAP,const auto& EF,const auto& EI,const auto& Q,const auto& EQ,const auto& C,int e) {
      volumes.pending_shell=-1; serialization.pending=Transaction{};
      return serialization.allowed(e,V,F,E,EMAP,EF,EI,C)
        && native_pre(V,F,E,EMAP,EF,EI,Q,EQ,C,e) && volumes.allowed(e,V,F,E,EMAP,EF,EI,C);
    };
    post=[&serialization,&volumes,native_post](const auto& V,const auto& F,const auto& E,
        const auto& EMAP,const auto& EF,const auto& EI,const auto& Q,const auto& EQ,const auto& C,
        int e,int e1,int e2,int f1,int f2,bool collapsed) {
      native_post(V,F,E,EMAP,EF,EI,Q,EQ,C,e,e1,e2,f1,f2,collapsed);
      if(collapsed) {
        if(volumes.pending_shell<0) throw std::runtime_error("Collapse committed without original volume approval");
        auto& shell=volumes.shells[volumes.pending_shell]; shell.current.add(volumes.pending_delta);
        shell.magnitude+=std::abs(volumes.pending_delta); ++volumes.committed;
      }
      volumes.pending_shell=-1; serialization.finish(V,F,collapsed,f1,f2);
    };
    const igl::decimate_stopping_condition_callback stop=[&serialization](const auto&,const auto&,const auto&,
        const auto&,const auto&,const auto&,const auto&,const auto&,const auto&,int,int,int,int,int) {
      return serialization.budget_safe();
    };
    const bool reached=igl::decimate(V,F,cost,stop,pre,post,U,G,J,I);
    delete tree; return reached;
  } catch(...) { delete tree; throw; }
}

void summary(std::ostream& out,const Serialization& state) {
  uint64_t storage_pairs=0,key_pairs=0;
  for(const auto& pair:state.bad) {
    if(state.stored[pair.first]==state.stored[pair.second]) ++storage_pairs; else ++key_pairs;
  }
  out<<"{\"phase\":\"prepared\",\"source_float32_exactly_active\":"<<(state.source_active?"true":"false")
     <<",\"nonexact_collision_pairs\":"<<state.bad.size()<<",\"nonexact_key_collision_pairs\":"<<key_pairs
     <<",\"float32_collapsed_distinct_position_pairs\":"<<storage_pairs
     <<",\"active_vertices\":"<<state.live_vertices<<",\"active_faces\":"<<state.live_faces
     <<",\"keys_in_int64_range\":true,\"serialization_safe\":"<<(state.safe()?"true":"false")
     <<",\"serialization_checks\":"<<state.checks<<",\"serialization_vetoes\":"<<state.vetoes
     <<",\"committed_collapses\":"<<state.committed
     <<",\"geometry_snapped\":false,\"adopted\":false}";
}
}

int main(int argc,char** argv) {
  using namespace serialization_qem;
  try {
    arithmetic_contract();
    if(argc==2 && std::string(argv[1])=="--build-info") {
      std::cout<<"{\"source_sha256\":\""<<WR_SERIALIZATION_SOURCE_SHA256
        <<"\",\"volume_core_prefix_sha256\":\""<<WR_VOLUME_CORE_PREFIX_SHA256
        <<"\",\"volume_source_sha256\":\""<<WR_VOLUME_SOURCE_SHA256
        <<"\",\"base_source_sha256\":\""<<WR_BASE_SOURCE_SHA256
        <<"\",\"libigl_revision\":\""<<WR_LIBIGL_REVISION<<"\",\"eigen_revision\":\""<<WR_EIGEN_REVISION
        <<"\",\"weld_digits\":8,\"rounding\":\"ties-to-even\",\"referenced_only\":true,"
        "\"representative\":\"first original vertex index\",\"exact_arithmetic\":\"cpp_int float32 units 2^-149\","
        "\"native_cost_and_placement_unchanged\":true,\"block_intersections\":true,"
        "\"volume_relative_limit\":0.05,\"target_faces\":4096,\"prepared_only\":true,\"adopted\":false}\n";
      return 0;
    }
    if(argc==3 && std::string(argv[1])=="--preflight") {
      Eigen::MatrixXd V; Eigen::MatrixXi F; read_obj(argv[2],V,F); Serialization state(V,F);
      summary(std::cout,state); std::cout<<'\n'; return 0;
    }
    if(argc==5 && std::string(argv[1])=="--position-key") {
      Position p;
      for(int j=0;j<3;++j) {
        std::istringstream token(argv[2+j]); token.imbue(std::locale::classic());
        std::string extra;
        if(!(token>>p[j]) || (token>>extra) || !std::isfinite(p[j]))
          throw std::runtime_error("Position key requires three finite coordinates");
      }
      const auto k=key(quantize(p));
      std::cout<<"{\"key\":["<<k[0]<<','<<k[1]<<','<<k[2]<<"],\"adopted\":false}\n";
      return 0;
    }
    if(argc==20 && std::string(argv[1])=="--triangle-predicates") {
      std::array<FloatPosition,3> a,b;
      for(int point=0;point<6;++point) {
        Position p;
        for(int j=0;j<3;++j) {
          std::istringstream token(argv[2+point*3+j]); token.imbue(std::locale::classic());
          std::string extra;
          if(!(token>>p[j]) || (token>>extra) || !std::isfinite(p[j]))
            throw std::runtime_error("Triangle predicate requires eighteen finite coordinates");
        }
        (point<3?a[point]:b[point-3])=quantize(p);
      }
      const auto old_normal=normal(a),new_normal=normal(b);
      std::cout<<"{\"before_float32_exactly_active\":"<<(active(old_normal)?"true":"false")
        <<",\"after_float32_exactly_active\":"<<(active(new_normal)?"true":"false")
        <<",\"exact_normal_dot_positive\":"<<(orientation(old_normal,new_normal)?"true":"false")
        <<",\"adopted\":false}\n";
      return 0;
    }
    if(argc!=4) throw std::runtime_error("Usage: mesh_serialization_qem input.obj output.obj mapping.json");
    require_absent(argv[2]); require_absent(argv[3]);
    if(std::string(argv[2])==argv[3]) throw std::runtime_error("Output paths must differ");
    Eigen::MatrixXd V,U; Eigen::MatrixXi F,G; Eigen::VectorXi J,I;
    read_obj(argv[1],V,F); Serialization state(V,F); volume_qem::Volumes volumes(V,F);
    const bool reached=simplify(V,F,volumes,state,U,G,J,I);
    if(U.cols()!=3 || G.cols()!=3 || !U.allFinite() || U.rows()<4 || G.rows()<4
        || J.size()!=G.rows() || I.size()!=U.rows() || G.minCoeff()<0 || G.maxCoeff()>=U.rows()
        || J.minCoeff()<0 || J.maxCoeff()>=F.rows() || I.minCoeff()<0 || I.maxCoeff()>=V.rows())
      throw std::runtime_error("Invalid native output or complete birth maps");
    volumes.verify_final(U,G,J,I); Serialization independent(U,G);
    if(independent.live_faces!=state.live_faces || independent.live_vertices!=state.live_vertices
        || independent.bad.size()!=state.bad.size()) throw std::runtime_error("Final serialization recomputation differs");
    if(!reached || !independent.budget_safe()) throw std::runtime_error("Native queue exhausted before budget AND serialization safety");
    auto out=exclusive_file(argv[2]);
    for(int v=0;v<U.rows();++v) std::fprintf(out.get(),"v %.17g %.17g %.17g\n",U(v,0),U(v,1),U(v,2));
    for(int f=0;f<G.rows();++f) std::fprintf(out.get(),"f %d %d %d\n",G(f,0)+1,G(f,1)+1,G(f,2)+1);
    if(std::fflush(out.get()) || std::ferror(out.get())) throw std::runtime_error("OBJ write failed");
    auto mapping=exclusive_file(argv[3]);
    std::fputs("{\"serialization\":",mapping.get()); std::ostringstream details; summary(details,state);
    std::fputs(details.str().c_str(),mapping.get()); std::fputs(",\"native_volume\":",mapping.get());
    volume_qem::mapping_json(mapping.get(),V,F,U,G,J,I,volumes,reached,true);
    std::fputs("}\n",mapping.get());
    if(std::fflush(mapping.get()) || std::ferror(mapping.get())) throw std::runtime_error("Mapping write failed");
    return 0;
  } catch(const std::exception& error) {
    std::cerr<<"mesh_serialization_qem: "<<error.what()<<'\n'; return 2;
  }
}
