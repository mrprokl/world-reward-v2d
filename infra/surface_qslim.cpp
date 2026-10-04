// SPDX-License-Identifier: Apache-2.0
// Prepared surface-only adapter; libigl/Eigen retain their MPL-2.0 notices.
// Upstream intersection blocking is floating-point, NOT an embedding certificate.
#include <Eigen/Core>
#include <igl/AABB.h>
#include <igl/connect_boundary_to_infinity.h>
#include <igl/decimate.h>
#include <igl/edge_collapse_is_valid.h>
#include <igl/edge_flaps.h>
#include <igl/intersection_blocking_collapse_edge_callbacks.h>
#include <igl/per_vertex_point_to_plane_quadrics.h>
#include <igl/qslim_optimal_collapse_edge_callbacks.h>
#include <igl/remove_unreferenced.h>
#include <boost/multiprecision/cpp_int.hpp>
#include <algorithm>
#include <array>
#include <cerrno>
#include <cfenv>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <limits>
#include <locale>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

#ifndef WR_SURFACE_QSLIM_SOURCE_SHA256
#error Bind the complete new adapter source SHA256 at compilation
#endif
#ifdef __FAST_MATH__
#error Exact represented predicates require no fast-math
#endif
#define WR_LIBIGL_REVISION "40e7900ccbd767f1f360e0eb10f0f1a6432e0993"
#define WR_EIGEN_REVISION "3147391d946bb4b6c68edd901f2add6ac1f31f8c"
constexpr int TARGET = 4096;
namespace surface {
using VMat = Eigen::MatrixXd;
using FMat = Eigen::MatrixXi;
using Face = std::array<int,3>;
using Edge = std::pair<int,int>;
using Integer = boost::multiprecision::cpp_int;
using Normal = std::array<Integer,3>;
using Queue = igl::min_heap<std::tuple<double,int,int>>;
struct Inapplicable : std::runtime_error { using std::runtime_error::runtime_error; };
struct Exhausted : std::runtime_error { using std::runtime_error::runtime_error; };
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
void arithmetic_contract() {
  require(std::numeric_limits<float>::is_iec559 && sizeof(float)==4 &&
          std::numeric_limits<double>::is_iec559 && sizeof(double)==8 &&
          std::fegetround()==FE_TONEAREST, "Unsupported IEEE arithmetic contract");
  volatile float denormal=std::numeric_limits<float>::denorm_min();
  require(denormal!=0 && denormal*1.0f!=0, "Gradual float32 underflow required");
}
Integer units(double value, bool f32) {
  require(std::isfinite(value), "Nonfinite represented coordinate");
  uint64_t bits=0, fraction=0, exponent=0;
  if (f32) {
    volatile float cast=static_cast<float>(value); float stored=cast;
    require(std::isfinite(stored), "Float32 coordinate overflow");
    uint32_t raw; std::memcpy(&raw,&stored,4); bits=raw;
    exponent=(raw>>23)&255; fraction=raw&0x7fffff;
  } else {
    std::memcpy(&bits,&value,8); exponent=(bits>>52)&2047;
    fraction=bits&0xfffffffffffffULL;
  }
  Integer result=fraction; if(exponent) result+=(Integer(1)<<(f32?23:52));
  if (exponent) result<<=(exponent-1);
  if((bits>>(f32?31:63))&1) result=-result; return result;
}
Normal normal(const VMat& V, const Face& f, bool f32,
              int s=-1, int d=-1, const Eigen::RowVector3d* placement=nullptr) {
  std::array<Normal,3> p;
  for (int k=0;k<3;++k) for (int a=0;a<3;++a)
    p[k][a]=units(placement && (f[k]==s || f[k]==d) ? (*placement)[a] : V(f[k],a), f32);
  Normal a,b,n;
  for (int k=0;k<3;++k) { a[k]=p[1][k]-p[0][k]; b[k]=p[2][k]-p[0][k]; }
  n[0]=a[1]*b[2]-a[2]*b[1]; n[1]=a[2]*b[0]-a[0]*b[2]; n[2]=a[0]*b[1]-a[1]*b[0];
  return n;
}
bool active(const Normal& n) { return n[0]!=0 || n[1]!=0 || n[2]!=0; }
bool same_orientation(const Normal& a, const Normal& b) {
  return active(b) && a[0]*b[0]+a[1]*b[1]+a[2]*b[2]>0;
}
Face face(const FMat& F,int f) { return {F(f,0),F(f,1),F(f,2)}; }
Edge edge(int a,int b) { return std::minmax(a,b); }
bool live(const FMat& F,int f) { return F(f,0)!=0 || F(f,1)!=0 || F(f,2)!=0; }
struct DSU {
  std::vector<int> parent;
  explicit DSU(int n):parent(n) { std::iota(parent.begin(),parent.end(),0); }
  int find(int i) const { while(parent[i]!=i) i=parent[i]; return i; }
  void join(int a,int b) { a=find(a); b=find(b); if(a!=b) parent[std::max(a,b)]=std::min(a,b); }
};
struct Topology {
  std::vector<int> face_component, vertex_component, euler;
  std::vector<bool> boundary, referenced;
  std::set<Edge> boundary_edges;
  int components=0, boundary_vertices=0, unused_vertices=0;
};
Topology topology(const VMat& V,const FMat& F) {
  require(V.cols()==3 && F.cols()==3 && V.rows()>=3 && F.rows()>=1 && V.allFinite(), "Invalid surface arrays");
  struct Use { int count=0, direction=0, first=-1; };
  std::map<Edge,Use> edges;
  std::set<Face> triangles;
  std::vector<std::map<int,std::set<int>>> links(V.rows());
  Topology out; out.boundary.assign(V.rows(),false); out.referenced.assign(V.rows(),false);
  out.vertex_component.assign(V.rows(),-1); DSU components(F.rows());
  for (int f=0;f<F.rows();++f) {
    const Face t=face(F,f); Face unordered=t; std::sort(unordered.begin(),unordered.end());
    require(unordered[0]>=0 && unordered[2]<V.rows() && unordered[0]!=unordered[1] && unordered[1]!=unordered[2], "Invalid/repeated triangle index");
    require(triangles.insert(unordered).second, "Duplicate surface triangle");
    const Normal native_normal=normal(V,t,false), stored_normal=normal(V,t,true);
    require(active(native_normal) && same_orientation(native_normal,stored_normal), "Inactive/reversed represented float32 triangle");
    for (int k=0;k<3;++k) {
      const int a=t[k], b=t[(k+1)%3], c=t[(k+2)%3]; out.referenced[a]=true;
      links[a][b].insert(c); links[a][c].insert(b);
      Use& u=edges[edge(a,b)]; ++u.count; u.direction+=a<b?1:-1;
      if(u.first<0) u.first=f; else components.join(u.first,f);
    }
  }
  for(const auto& item:edges) {
    const auto& u=item.second;
    require(u.count==1 || (u.count==2 && u.direction==0), "Non-manifold or inconsistently oriented edge");
    if(u.count==1) { out.boundary_edges.insert(item.first); out.boundary[item.first.first]=out.boundary[item.first.second]=true; }
  }
  for(int v=0;v<V.rows();++v) {
    if(!out.referenced[v]) { ++out.unused_vertices; continue; }
    int endpoints=0; const auto& link=links[v]; std::set<int> reached; std::vector<int> pending{link.begin()->first};
    while(!pending.empty()) { int n=pending.back(); pending.pop_back(); if(!reached.insert(n).second)continue;
      for(int q:link.at(n)) pending.push_back(q); }
    for(const auto& n:link) { require(n.second.size()==1 || n.second.size()==2, "Invalid vertex link degree"); endpoints+=n.second.size()==1; }
    require(reached.size()==link.size() && endpoints==(out.boundary[v]?2:0), "Non-manifold vertex link");
    out.boundary_vertices+=out.boundary[v];
  }
  std::map<int,int> ids; out.face_component.resize(F.rows());
  for(int f=0;f<F.rows();++f) { int root=components.find(f); auto p=ids.emplace(root,int(ids.size())); out.face_component[f]=p.first->second; }
  out.components=ids.size(); out.euler.assign(out.components,0);
  for(int f=0;f<F.rows();++f) {
    int c=out.face_component[f]; ++out.euler[c];
    for(int k=0;k<3;++k) { int v=F(f,k); require(out.vertex_component[v]<0 || out.vertex_component[v]==c,"Components share a vertex"); out.vertex_component[v]=c; }
  }
  for(int c:out.vertex_component) if(c>=0) ++out.euler[c];
  for(const auto& e:edges) --out.euler[out.vertex_component[e.first.first]];
  return out;
}
void read_obj(const char* path,VMat& V,FMat& F) {
  struct stat st; require(::lstat(path,&st)==0 && S_ISREG(st.st_mode) && st.st_nlink==1 && st.st_size>0 && st.st_size<=268435456,"Input requires bounded regular single-link OBJ");
  std::ifstream in(path); in.imbue(std::locale::classic()); std::vector<Eigen::RowVector3d> vertices; std::vector<Face> faces;
  std::string line;
  while(std::getline(in,line)) {
    require(line.size()<=4096,"OBJ record exceeds bound"); line.resize(line.find('#')==std::string::npos?line.size():line.find('#'));
    std::istringstream row(line); row.imbue(std::locale::classic()); std::string kind,extra; if(!(row>>kind))continue;
    if(kind=="v") { Eigen::RowVector3d p; require(bool(row>>p[0]>>p[1]>>p[2]) && !(row>>extra) && p.allFinite(),"Malformed vertex"); vertices.push_back(p); }
    else if(kind=="f") { Face t; for(int& v:t) { std::string token; require(bool(row>>token),"Nontriangular face");
      auto result=std::from_chars(token.data(),token.data()+token.size(),v);
      require(result.ec==std::errc() && result.ptr==token.data()+token.size() && v>0,"Only positive plain OBJ indices supported"); --v; }
      require(!(row>>extra),"Nontriangular face"); faces.push_back(t); }
    else throw std::runtime_error("Only original v/f records and comments supported");
    require(vertices.size()<=1000000 && faces.size()<=2000000,"Surface count cap exceeded");
  }
  require(in.eof(),"Incomplete OBJ read"); V.resize(vertices.size(),3); F.resize(faces.size(),3);
  for(int i=0;i<V.rows();++i) V.row(i)=vertices[i];
  for(int i=0;i<F.rows();++i) for(int k=0;k<3;++k) F(i,k)=faces[i][k];
}
struct Event { int s,d; Eigen::RowVector3d placement; std::vector<int> removed; };
struct Transaction { int s=-1,d=-1; Eigen::RowVector3d placement; std::map<int,Face> before,after; std::set<int> removed; };
struct State {
  const Topology& source; int real_faces, real_vertices, source_faces, source_vertices;
  std::vector<std::set<int>> incident; std::vector<int> component_faces, reference_count;
  DSU quotient; Transaction pending; std::vector<Event> ledger;
  int attempts=0, rejected_link=0, rejected_geometry=0, rejected_component=0, rejected_intersection=0, native_failed=0;
  State(const Topology& t,const FMat& F,int nv):source(t),real_faces(F.rows()),real_vertices(nv),source_faces(F.rows()),source_vertices(nv),incident(nv),component_faces(t.components,0),reference_count(nv,0),quotient(nv) {
    for(int f=0;f<F.rows();++f) { ++component_faces[t.face_component[f]]; for(int v:face(F,f)) { incident[v].insert(f); ++reference_count[v]; } }
  }
  bool eligible(int a,int b) const { return a>=0 && b>=0 && a<source_vertices && b<source_vertices && !source.boundary[a] && !source.boundary[b]; }
  bool prepare(const VMat& V,const FMat& F,const FMat& E,const Eigen::VectorXi& EMAP,const FMat& EF,const FMat& EI,const VMat& C,int e) {
    pending=Transaction{}; ++attempts;
    int s=std::min(E(e,0),E(e,1)),d=std::max(E(e,0),E(e,1));
    if(!eligible(s,d) || source.vertex_component[s]<0 || source.vertex_component[s]!=source.vertex_component[d]) { ++rejected_component; return false; }
    if(!igl::edge_collapse_is_valid(e,F,E,EMAP,EF,EI)) { ++rejected_link; return false; }
    Transaction next; next.s=s; next.d=d; next.placement=C.row(e);
    if(!next.placement.allFinite()) { ++rejected_geometry; return false; }
    for(int k=0;k<3;++k) { volatile float stored=static_cast<float>(next.placement[k]);
      if(!std::isfinite(stored)) { ++rejected_geometry; return false; } }
    for(int k=0;k<2;++k) { int f=EF(e,k); if(f<0 || f>=source_faces || !live(F,f)) { ++rejected_component; return false; } next.removed.insert(f); }
    if(next.removed.size()!=2 || component_faces[source.vertex_component[s]]<=2) { ++rejected_component; return false; }
    std::set<int> affected=incident[s]; affected.insert(incident[d].begin(),incident[d].end());
    for(int f:affected) {
      require(f>=0 && f<source_faces && live(F,f),"Incident face ledger drift"); const Face before=face(F,f); next.before[f]=before;
      if(next.removed.count(f))continue;
      Face after=before; for(int& v:after) if(v==d)v=s;
      if(after[0]==after[1] || after[1]==after[2] || after[2]==after[0]) { ++rejected_link; return false; }
      for(bool f32:{false,true}) if(!same_orientation(normal(V,before,f32),normal(V,after,f32,s,d,&next.placement))) { ++rejected_geometry; return false; }
      if(!same_orientation(normal(V,after,false,s,d,&next.placement),normal(V,after,true,s,d,&next.placement))) { ++rejected_geometry; return false; }
      next.after[f]=after;
    }
    pending=std::move(next); return true;
  }
  void commit(const VMat& V,const FMat& F,int f1,int f2) {
    require(pending.s>=0 && pending.removed==std::set<int>{f1,f2},"Native deleted faces differ from transaction");
    require((V.row(pending.s).array()==pending.placement.array()).all() && (V.row(pending.d).array()==pending.placement.array()).all(),"Native placement differs from transaction");
    for(const auto& item:pending.before) {
      int f=item.first; for(int v:item.second) { incident[v].erase(f); --reference_count[v]; }
      if(pending.removed.count(f)) { require(!live(F,f),"Removed face is still live"); --component_faces[source.face_component[f]]; --real_faces; }
      else { require(face(F,f)==pending.after.at(f),"Native face remap differs from transaction"); for(int v:pending.after.at(f)) { incident[v].insert(f); ++reference_count[v]; } }
    }
    require(reference_count[pending.d]==0 && reference_count[pending.s]>0,"Collapsed vertex reference census drift");
    --real_vertices; quotient.parent[pending.d]=pending.s;
    ledger.push_back({pending.s,pending.d,pending.placement,std::vector<int>(pending.removed.begin(),pending.removed.end())}); pending=Transaction{};
  }
};
struct Result { VMat V; FMat F; Eigen::VectorXi J,I,vertex_map; std::vector<int> candidate_components; };
Result finalize(const VMat& originalV,const FMat& originalF,const Topology& source,State& state,const VMat& U,const FMat& G,const Eigen::VectorXi& J,const Eigen::VectorXi& I) {
  require(J.size()==G.rows() && I.size()==U.rows(),"Native birth dimensions mismatch");
  std::vector<int> keep; for(int f=0;f<J.size();++f) { require(J[f]>=0 && J[f]<originalF.rows()+int(source.boundary_edges.size()),"Invalid proxy face birth"); if(J[f]<originalF.rows())keep.push_back(f); }
  FMat real(keep.size(),3); Result out; out.J.resize(keep.size());
  for(int f=0;f<int(keep.size());++f) { real.row(f)=G.row(keep[f]); out.J[f]=J[keep[f]]; }
  Eigen::VectorXi old_to_new, retained;
  igl::remove_unreferenced(U,real,out.V,out.F,old_to_new,retained); out.I.resize(retained.size()+source.unused_vertices);
  for(int v=0;v<retained.size();++v) { require(retained[v]>=0 && retained[v]<I.size(),"Invalid compact vertex index"); out.I[v]=I[retained[v]]; }
  // Preserve original unused vertices verbatim; no source component/vertex cleanup.
  int n=retained.size(); out.V.conservativeResize(n+source.unused_vertices,3);
  for(int v=0;v<originalV.rows();++v) if(!source.referenced[v]) { out.V.row(n)=originalV.row(v); out.I[n++]=v; }
  const Topology final=topology(out.V,out.F);
  require(final.components==source.components && out.F.rows()==state.real_faces && out.V.rows()==state.real_vertices,"Final component/count census drift");
  std::vector<int> correspondence(final.components,-1); std::set<int> source_components;
  std::set<int> face_births; std::map<int,int> survivor;
  for(int v=0;v<out.I.size();++v) { int birth=out.I[v]; require(birth>=0 && birth<originalV.rows() && state.quotient.find(birth)==birth && survivor.emplace(birth,v).second,"Invalid surviving vertex birth"); }
  out.vertex_map.resize(originalV.rows());
  for(int v=0;v<originalV.rows();++v) { auto found=survivor.find(state.quotient.find(v)); require(found!=survivor.end(),"Incomplete original vertex quotient replay"); out.vertex_map[v]=found->second; }
  for(int f=0;f<out.F.rows();++f) {
    int birth=out.J[f]; require(face_births.insert(birth).second,"Duplicate surviving face birth"); int fc=final.face_component[f],sc=source.face_component[birth];
    require(correspondence[fc]<0 || correspondence[fc]==sc,"Candidate merges source components"); correspondence[fc]=sc;
    for(int k=0;k<3;++k) require(out.F(f,k)==out.vertex_map[originalF(birth,k)],"Oriented face birth/quotient replay mismatch");
  }
  std::set<Edge> actual_boundary;
  for(const auto& e:final.boundary_edges) actual_boundary.insert(edge(out.I[e.first],out.I[e.second]));
  require(actual_boundary==source.boundary_edges,"Fixed boundary edge geometry/lineage changed");
  for(int v=0;v<out.I.size();++v) if(source.boundary[out.I[v]]) require(std::memcmp(out.V.row(v).eval().data(),originalV.row(out.I[v]).eval().data(),3*sizeof(double))==0,"Fixed boundary vertex bytes changed");
  for(int c=0;c<final.components;++c) { int sc=correspondence[c]; require(sc>=0 && source_components.insert(sc).second && final.euler[c]==source.euler[sc],"Component bijection/Euler mismatch"); }
  for(int v=0;v<out.I.size();++v) if(!source.referenced[out.I[v]]) require(std::memcmp(out.V.row(v).eval().data(),originalV.row(out.I[v]).eval().data(),3*sizeof(double))==0,"Original unused vertex bytes changed");
  out.candidate_components=std::move(correspondence); return out;
}
Result simplify(const VMat& V,const FMat& F,const Topology& source,State& state) {
  if(V.rows()<=TARGET && F.rows()<=TARGET) { Result r; r.V=V; r.F=F; r.J=Eigen::VectorXi::LinSpaced(F.rows(),0,F.rows()-1); r.I=Eigen::VectorXi::LinSpaced(V.rows(),0,V.rows()-1); r.vertex_map=r.I;
    r.candidate_components.resize(source.components); std::iota(r.candidate_components.begin(),r.candidate_components.end(),0); return r; }
  VMat proxyV,U; FMat proxyF,E,EF,EI,G; Eigen::VectorXi EMAP,J,I;
  // Match upstream qslim: AABB ONLY original finite faces; proxy created afterward.
  igl::AABB<VMat,3>* tree=new igl::AABB<VMat,3>(); tree->init(V,F);
  struct TreeOwner { igl::AABB<VMat,3>*& pointer; ~TreeOwner(){delete pointer;} } owner{tree};
  igl::connect_boundary_to_infinity(V,F,proxyV,proxyF); igl::edge_flaps(proxyF,E,EMAP,EF,EI);
  std::vector<std::tuple<Eigen::MatrixXd,Eigen::RowVectorXd,double>> quadrics;
  igl::per_vertex_point_to_plane_quadrics(proxyV,proxyF,EMAP,EF,EI,quadrics);
  int v1=-1,v2=-1; igl::decimate_cost_and_placement_callback native_cost,cost;
  igl::decimate_pre_collapse_callback native_pre,intersection_pre,pre;
  igl::decimate_post_collapse_callback native_post,intersection_post,post;
  igl::qslim_optimal_collapse_edge_callbacks(E,quadrics,v1,v2,native_cost,native_pre,native_post);
  igl::intersection_blocking_collapse_edge_callbacks(native_pre,native_post,tree,intersection_pre,intersection_post);
  cost=[&](int e,const VMat& v,const FMat& f,const FMat& edges,const Eigen::VectorXi& emap,const FMat& ef,const FMat& ei,double& c,Eigen::RowVectorXd& p) {
    // Initial costs can be evaluated in parallel: only immutable boundary reads.
    if(!state.eligible(edges(e,0),edges(e,1))) { c=std::numeric_limits<double>::infinity(); p=Eigen::RowVector3d::Zero(); return; }
    native_cost(e,v,f,edges,emap,ef,ei,c,p);
  };
  pre=[&](const VMat& v,const FMat& f,const FMat& edges,const Eigen::VectorXi& emap,const FMat& ef,const FMat& ei,const Queue& q,const Eigen::VectorXi& eq,const VMat& c,int e) {
    if(!state.prepare(v,f,edges,emap,ef,ei,c,e))return false;
    if(!intersection_pre(v,f,edges,emap,ef,ei,q,eq,c,e)) { ++state.rejected_intersection; state.pending=Transaction{}; return false; } return true;
  };
  post=[&](const VMat& v,const FMat& f,const FMat& edges,const Eigen::VectorXi& emap,const FMat& ef,const FMat& ei,const Queue& q,const Eigen::VectorXi& eq,const VMat& c,int e,int e1,int e2,int f1,int f2,bool collapsed) {
    // Failed f1/f2 may be undefined. Do not inspect or forward them.
    if(!collapsed) { ++state.native_failed; state.pending=Transaction{}; return; }
    intersection_post(v,f,edges,emap,ef,ei,q,eq,c,e,e1,e2,f1,f2,true);
    state.commit(v,f,f1,f2);
  };
  igl::decimate_stopping_condition_callback stop=[&](const VMat&,const FMat&,const FMat&,const Eigen::VectorXi&,const FMat&,const FMat&,const Queue&,const Eigen::VectorXi&,const VMat&,int,int,int,int,int) {
    return state.real_faces<=TARGET && state.real_vertices<=TARGET;
  };
  bool reached=igl::decimate(proxyV,proxyF,cost,stop,pre,post,U,G,J,I);
  if(!reached) throw Exhausted("Native queue exhausted before both real surface budgets");
  Result result=finalize(V,F,source,state,U,G,J,I);
  require(result.V.rows()<=TARGET && result.F.rows()<=TARGET,"Real output exceeds budget"); return result;
}
using File=std::unique_ptr<FILE,decltype(&std::fclose)>;
void absent(const char* p) { struct stat st; require(::lstat(p,&st)!=0 && errno==ENOENT,"Output must be fresh and absent"); }
File exclusive(const char* p) { int fd=::open(p,O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW,0644); require(fd>=0,"Exclusive output creation failed"); FILE* f=::fdopen(fd,"w"); if(!f){::close(fd);throw std::runtime_error("fdopen failed");} return File(f,&std::fclose); }
void finish(FILE* f) { require(std::fflush(f)==0 && !std::ferror(f) && ::fsync(::fileno(f))==0,"Output write/fsync failed"); }
template<class Values> void array(FILE* f,const Values& a) { std::fputc('[',f); for(int i=0;i<int(a.size());++i)std::fprintf(f,"%s%d",i?",":"",int(a[i])); std::fputc(']',f); }
void write(const char* mesh,const char* mapping,const Result& r,const Topology& t,const State& s) {
  auto obj=exclusive(mesh); for(int v=0;v<r.V.rows();++v)std::fprintf(obj.get(),"v %.17g %.17g %.17g\n",r.V(v,0),r.V(v,1),r.V(v,2));
  for(int f=0;f<r.F.rows();++f)std::fprintf(obj.get(),"f %d %d %d\n",r.F(f,0)+1,r.F(f,1)+1,r.F(f,2)+1); finish(obj.get());
  auto out=exclusive(mapping);
  std::fprintf(out.get(),"{\"schema\":\"surface-qslim-mapping-v1\",\"source_sha256\":\"%s\",\"source_vertices\":%d,\"source_faces\":%d,\"output_vertices\":%d,\"output_faces\":%d,\"target_vertices\":4096,\"target_faces\":4096,\"components\":%d,\"boundary_vertices\":%d,\"original_unused_vertices_preserved\":%d,\"native_attempts\":%d,\"committed_collapses\":%zu,\"native_failed\":%d,\"veto_link\":%d,\"veto_geometry\":%d,\"veto_component\":%d,\"veto_intersection\":%d,\"boundary_policy\":\"fixed_original_vertices\",\"intersection_blocking\":\"upstream_floating_point\",\"initial_embedding_certified\":false,\"volume_or_closure_required\":false,\"serialization_qualification_completed\":false,\"adoption\":false,\"J\":",WR_SURFACE_QSLIM_SOURCE_SHA256,s.source_vertices,s.source_faces,int(r.V.rows()),int(r.F.rows()),t.components,t.boundary_vertices,t.unused_vertices,s.attempts,s.ledger.size(),s.native_failed,s.rejected_link,s.rejected_geometry,s.rejected_component,s.rejected_intersection);
  array(out.get(),r.J); std::fputs(",\"I\":",out.get()); array(out.get(),r.I); std::fputs(",\"original_vertex_to_output\":",out.get()); array(out.get(),r.vertex_map);
  std::fputs(",\"candidate_component_to_source\":",out.get()); array(out.get(),r.candidate_components); std::fputs(",\"ledger\":[",out.get());
  for(size_t i=0;i<s.ledger.size();++i) { const auto& e=s.ledger[i]; std::fprintf(out.get(),"%s{\"survivor\":%d,\"removed_vertex\":%d,\"placement\":[%.17g,%.17g,%.17g],\"removed_faces\":",i?",":"",e.s,e.d,e.placement[0],e.placement[1],e.placement[2]);array(out.get(),e.removed);std::fputc('}',out.get()); }
  std::fputs("]}\n",out.get()); finish(out.get());
}
} // namespace surface
int main(int argc,char** argv) {
  using namespace surface;
  try {
    arithmetic_contract();
    if(argc==2 && std::string(argv[1])=="--build-info") {
      std::cout<<"{\"stage\":\"surface_qslim_build_info_v1\",\"source_sha256\":\""<<WR_SURFACE_QSLIM_SOURCE_SHA256<<"\",\"libigl_revision\":\""<<WR_LIBIGL_REVISION<<"\",\"eigen_revision\":\""<<WR_EIGEN_REVISION<<"\",\"target_vertices\":4096,\"target_faces\":4096,\"boundary_policy\":\"fixed_original_vertices\",\"intersection_blocking\":\"upstream_floating_point\",\"prepared_only\":true,\"adoption\":false}\n"; return 0;
    }
    const bool preflight=argc==3 && std::string(argv[1])=="--preflight";
    require(preflight || argc==4,"Usage: surface_qslim --preflight input.obj OR input.obj output.obj mapping.json");
    if(!preflight) { require(std::string(argv[2])!=argv[3],"Distinct output paths required"); absent(argv[2]); absent(argv[3]); }
    VMat V; FMat F; read_obj(argv[preflight?2:1],V,F); Topology source=topology(V,F);
    if(source.boundary_vertices>=TARGET) throw Inapplicable("Fixed referenced boundary vertices reach the phase2 target");
    if(preflight) { std::cout<<"{\"source_vertices\":"<<V.rows()<<",\"source_faces\":"<<F.rows()<<",\"components\":"<<source.components<<",\"boundary_vertices\":"<<source.boundary_vertices<<",\"unused_vertices\":"<<source.unused_vertices<<",\"float32_triangles_exactly_active\":true,\"oriented_vertex_manifold\":true,\"volume_or_closure_required\":false,\"qem_calls\":0,\"adoption\":false}\n"; return 0; }
    State state(source,F,V.rows()); Result result=simplify(V,F,source,state); write(argv[2],argv[3],result,source,state); return 0;
  } catch(const Inapplicable& e) { std::cerr<<"INAPPLICABLE: "<<e.what()<<'\n'; return 4;
  } catch(const Exhausted& e) { std::cerr<<"EXHAUSTED: "<<e.what()<<'\n'; return 3;
  } catch(const std::exception& e) { std::cerr<<"FAIL: "<<e.what()<<'\n'; return 2; }
}
