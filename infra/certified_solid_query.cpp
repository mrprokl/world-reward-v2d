// SPDX-License-Identifier: Apache-2.0
// Standalone CGAL query: certify represented input, never repair or simplify.
// The GPL CGAL combination is not an Apache-only runtime. No QEM is executed.
// This independently written glue retains Apache terms; linked CGAL PMP/Side
// requires GPL-3.0-or-later (or a separately obtained commercial licence).
#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Polygon_mesh_processing/self_intersections.h>
#include <CGAL/Polygon_mesh_processing/intersection.h>
#include <CGAL/Side_of_triangle_mesh.h>
#include <CGAL/boost/graph/helpers.h>
#include <CGAL/version.h>
#include <algorithm>
#include <array>
#include <cerrno>
#include <cfenv>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef __FAST_MATH__
#error Exact filtered predicates cannot use fast-math
#endif
#ifndef WR_SOLID_QUERY_SOURCE_SHA256
#error Bind standalone source identity at compilation
#endif
#if CGAL_VERSION_NR != 1060011000
#error This numerical route requires the audited CGAL 6.0.1 release
#endif

namespace solid_query {
using Kernel = CGAL::Exact_predicates_exact_constructions_kernel;
using Point = Kernel::Point_3;
using Mesh = CGAL::Surface_mesh<Point>;
namespace PMP = CGAL::Polygon_mesh_processing;
// Resource bounds are input/output limits, never geometric tolerances.
constexpr std::size_t MAX_VERTICES = 1000000, MAX_FACES = 2000000, MAX_COMPONENTS = 256;

void require(bool condition, const char* message) {
  if(!condition) throw std::runtime_error(message);
}

bool whitespace(int c) { return c==' ' || c=='\t' || c=='\r' || c=='\n'; }

std::string token(std::istream& input) {
  std::string result;
  int c;
  while((c=input.get())!=EOF && whitespace(c)) {}
  require(c!=EOF, "Missing input token");
  do {
    require(c>=33 && c<=126 && result.size()<128, "Non-ASCII or oversized token");
    result.push_back(static_cast<char>(c));
    c=input.get();
  } while(c!=EOF && !whitespace(c));
  return result;
}

std::size_t integer(std::istream& input, std::size_t maximum) {
  const auto value=token(input);
  std::size_t number=0;
  for(char c:value) {
    require(c>='0' && c<='9', "Unsigned decimal integer required");
    const std::size_t digit=static_cast<std::size_t>(c-'0');
    require(digit<=maximum && number<=(maximum-digit)/10, "Integer outside input bounds");
    number=number*10+digit;
  }
  return number;
}

double coordinate(std::istream& input) {
  const auto value=token(input);
  std::size_t p=0, digits=0;
  bool nonzero=false;
  if(value[p]=='+' || value[p]=='-') ++p;
  while(p<value.size() && value[p]>='0' && value[p]<='9') {
    nonzero=nonzero || value[p]!='0'; ++p; ++digits;
  }
  if(p<value.size() && value[p]=='.') {
    ++p;
    while(p<value.size() && value[p]>='0' && value[p]<='9') {
      nonzero=nonzero || value[p]!='0'; ++p; ++digits;
    }
  }
  require(digits>0, "Finite decimal coordinate required");
  if(p<value.size() && (value[p]=='e' || value[p]=='E')) {
    ++p;
    if(p<value.size() && (value[p]=='+' || value[p]=='-')) ++p;
    const std::size_t first=p;
    while(p<value.size() && value[p]>='0' && value[p]<='9') ++p;
    require(p>first, "Invalid decimal exponent");
  }
  require(p==value.size(), "Invalid decimal coordinate");
  char* end=nullptr;
  errno=0;
  const double result=std::strtod(value.c_str(),&end);
  require(end==value.c_str()+value.size() && std::isfinite(result), "Nonfinite coordinate");
  require(result!=0 || !nonzero, "Nonzero coordinate rounded to zero");
  // ERANGE may also report a valid nonzero subnormal. Never cast it to zero;
  // exact EPECK predicates certify the retained binary64, not the decimal text.
  require(errno==0 || (errno==ERANGE && std::fpclassify(result)==FP_SUBNORMAL),
          "Decimal conversion range error");
  return result;
}

struct Face { std::array<std::size_t,3> vertices; std::size_t component; };
struct Input {
  std::vector<Point> points;
  std::vector<Face> faces;
  std::size_t components;
};

Input read(std::istream& stream) {
  require(token(stream)=="WR_SOLID_QUERY_V1", "Wrong input header");
  const auto nv=integer(stream,MAX_VERTICES), nf=integer(stream,MAX_FACES);
  const auto nk=integer(stream,MAX_COMPONENTS);
  require(nv>=4 && nf>=4 && nk>=1, "Nonempty closed solid required");
  Input result; result.components=nk; result.points.reserve(nv); result.faces.reserve(nf);
  for(std::size_t i=0;i<nv;++i) {
    const double x=coordinate(stream), y=coordinate(stream), z=coordinate(stream);
    result.points.emplace_back(x,y,z); // EPECK represents these original doubles exactly.
  }
  for(std::size_t i=0;i<nf;++i) {
    Face face;
    for(auto& v:face.vertices) v=integer(stream,nv-1);
    face.component=integer(stream,nk-1);
    result.faces.push_back(face);
  }
  int c;
  while((c=stream.get())!=EOF) require(whitespace(c), "Unexpected trailing input");
  require(!stream.bad(), "Input I/O failure");
  return result;
}

struct Component {
  Mesh mesh;
  std::vector<std::size_t> original_vertices;
  std::size_t face_count=0;
  int sign=0;
  double diagnostic_volume=0;
};

std::vector<Component> validate(const Input& input) {
  const auto nv=input.points.size();
  std::vector<std::size_t> owner(nv,input.components), counts(input.components,0);
  std::vector<std::vector<std::pair<std::size_t,std::size_t>>> links(nv);
  std::vector<std::vector<std::size_t>> adjacency(nv);
  std::map<std::pair<std::size_t,std::size_t>,std::pair<int,int>> edges;
  std::set<std::array<std::size_t,3>> triangles;
  for(const auto& face:input.faces) {
    const auto& f=face.vertices;
    require(f[0]!=f[1] && f[1]!=f[2] && f[2]!=f[0], "Repeated face indices");
    require(!CGAL::collinear(input.points[f[0]],input.points[f[1]],input.points[f[2]]),
            "Exact degenerate triangle");
    auto sorted=f; std::sort(sorted.begin(),sorted.end());
    require(triangles.insert(sorted).second, "Duplicate indexed triangle");
    ++counts[face.component];
    for(int j=0;j<3;++j) {
      const auto a=f[j], b=f[(j+1)%3], c=f[(j+2)%3];
      require(owner[a]==input.components || owner[a]==face.component,
              "A vertex belongs to multiple component labels");
      owner[a]=face.component; links[a].emplace_back(b,c);
      adjacency[a].push_back(b); adjacency[b].push_back(a);
      auto& edge=edges[{std::min(a,b),std::max(a,b)}];
      ++edge.first; edge.second+=(a<b ? 1 : -1);
    }
  }
  for(const auto& edge:edges)
    require(edge.second.first==2 && edge.second.second==0, "Closed opposite edge incidences required");
  for(std::size_t v=0;v<nv;++v) {
    require(owner[v]!=input.components, "Unreferenced original vertex forbidden");
    std::map<std::size_t,std::vector<std::size_t>> link;
    for(const auto& edge:links[v]) {
      link[edge.first].push_back(edge.second); link[edge.second].push_back(edge.first);
    }
    for(const auto& row:link) require(row.second.size()==2, "Vertex link is not a cycle");
    std::set<std::size_t> visited;
    std::vector<std::size_t> pending{link.begin()->first};
    while(!pending.empty()) {
      const auto q=pending.back(); pending.pop_back();
      if(visited.insert(q).second) for(auto neighbor:link.at(q)) pending.push_back(neighbor);
    }
    require(visited.size()==link.size(), "Disconnected vertex link");
  }
  std::vector<Component> result(input.components);
  std::vector<Mesh::Vertex_index> local(nv);
  for(std::size_t v=0;v<nv;++v) {
    auto& component=result[owner[v]];
    component.original_vertices.push_back(v);
    local[v]=component.mesh.add_vertex(input.points[v]);
  }
  for(std::size_t i=0;i<input.components;++i) {
    auto& component=result[i];
    require(counts[i]>=4 && component.original_vertices.size()>=4, "Empty or incomplete component label");
    component.face_count=counts[i];
    std::set<std::size_t> visited;
    std::vector<std::size_t> pending{component.original_vertices.front()};
    while(!pending.empty()) {
      const auto q=pending.back(); pending.pop_back();
      require(owner[q]==i, "Component edge crosses labels");
      if(visited.insert(q).second) for(auto neighbor:adjacency[q]) pending.push_back(neighbor);
    }
    require(visited.size()==component.original_vertices.size(), "One label contains disconnected surfaces");
  }
  std::vector<Kernel::FT> volume6(input.components,Kernel::FT(0));
  for(const auto& face:input.faces) {
    auto& component=result[face.component]; const auto& f=face.vertices;
    require(component.mesh.add_face(local[f[0]],local[f[1]],local[f[2]])!=Mesh::null_face(),
            "Surface_mesh rejected an original face");
    const auto& origin=input.points[component.original_vertices.front()];
    const auto a=input.points[f[0]]-origin, b=input.points[f[1]]-origin, c=input.points[f[2]]-origin;
    volume6[face.component]+=a.x()*(b.y()*c.z()-b.z()*c.y())
                           -a.y()*(b.x()*c.z()-b.z()*c.x())
                           +a.z()*(b.x()*c.y()-b.y()*c.x());
  }
  for(std::size_t i=0;i<input.components;++i) {
    auto& component=result[i];
    require(CGAL::is_valid_polygon_mesh(component.mesh) && CGAL::is_closed(component.mesh)
            && CGAL::is_triangle_mesh(component.mesh)
            && component.mesh.number_of_vertices()==component.original_vertices.size()
            && component.mesh.number_of_faces()==counts[i], "Original component construction differs");
    const auto sign=CGAL::sign(volume6[i]);
    require(sign!=CGAL::ZERO, "Exact zero signed component volume");
    component.sign=(sign==CGAL::POSITIVE ? 1 : -1);
    component.diagnostic_volume=CGAL::to_double(volume6[i]/Kernel::FT(6));
  }
  return result;
}

std::string query(const Input& input) {
  auto components=validate(input);
  // All geometric checks complete before any containment is returned.
  for(const auto& component:components)
    require(!PMP::does_self_intersect(component.mesh), "Exact component self-intersection");
  for(std::size_t i=0;i<input.components;++i)
    for(std::size_t j=i+1;j<input.components;++j)
      require(!PMP::do_intersect(components[i].mesh,components[j].mesh,
                  CGAL::parameters::do_overlap_test_of_bounded_sides(false)),
              "Components intersect or touch");
  std::vector<std::vector<bool>> inside(input.components,std::vector<bool>(input.components,false));
  for(std::size_t j=0;j<input.components;++j) {
    CGAL::Side_of_triangle_mesh<Mesh,Kernel> side(components[j].mesh);
    for(std::size_t i=0;i<input.components;++i) if(i!=j) {
      const auto answer=side(input.points[components[i].original_vertices.front()]);
      require(answer!=CGAL::ON_BOUNDARY, "Containment witness is on boundary");
      inside[i][j]=(answer==CGAL::ON_BOUNDED_SIDE);
    }
  }
  std::ostringstream out;
  out<<std::setprecision(std::numeric_limits<double>::max_digits10)
     <<"{\"schema\":\"world_reward.certified_solid_query.v1\",\"status\":\"pass\","
     <<"\"source_sha256\":\""<<WR_SOLID_QUERY_SOURCE_SHA256<<"\",\"cgal_version\":\"6.0.1\","
     <<"\"vertices\":"<<input.points.size()<<",\"faces\":"<<input.faces.size()
     <<",\"component_count\":"<<input.components<<",\"components\":[";
  for(std::size_t i=0;i<input.components;++i) {
    if(i) out<<',';
    const auto& component=components[i];
    out<<"{\"original_component_id\":"<<i<<",\"original_vertices\":"<<component.original_vertices.size()
       <<",\"original_faces\":"<<component.face_count<<",\"witness_original_vertex\":"<<component.original_vertices.front()
       <<",\"exact_volume_sign\":"<<component.sign<<",\"diagnostic_signed_volume\":";
    if(std::isfinite(component.diagnostic_volume)) out<<component.diagnostic_volume; else out<<"null";
    out<<'}';
  }
  out<<"],\"inside\":[";
  for(std::size_t i=0;i<input.components;++i) {
    if(i) out<<','; out<<'[';
    for(std::size_t j=0;j<input.components;++j) { if(j) out<<','; out<<(inside[i][j] ? "true" : "false"); }
    out<<']';
  }
  out<<"],\"represented_coordinates\":\"EPECK exact values of original parsed IEEE754 binary64; no perturbation\","
     <<"\"closed_oriented_vertex_manifold_verified\":true,\"all_original_faces_retained\":true,"
     <<"\"all_original_vertices_referenced\":true,\"exact_nondegenerate_triangles_verified\":true,"
     <<"\"component_self_intersections_absent\":true,\"inter_component_surface_contacts_absent\":true,"
     <<"\"geometry_repaired\":false,\"orientation_changed\":false,\"qem_executed\":false,"
     <<"\"forest_adjudicated\":false,\"reconstruction_accuracy_verified\":false}";
  require(out.str().size()<1048576, "Output exceeds fixed bound");
  return out.str();
}
} // namespace solid_query

int main(int argc,char**) {
  try {
    solid_query::require(argc==1, "No alternate input modes accepted");
    solid_query::require(std::setlocale(LC_NUMERIC,"C")!=nullptr, "Classic numeric locale required");
    solid_query::require(sizeof(double)==8 && std::numeric_limits<double>::is_iec559
                        && std::numeric_limits<double>::digits==53 && std::fegetround()==FE_TONEAREST,
                        "IEEE754 binary64 nearest arithmetic required");
    const auto input=solid_query::read(std::cin);
    const auto output=solid_query::query(input);
    // No partial PASS: stdout is touched only after the complete query succeeds.
    std::cout<<output<<'\n'; std::cout.flush();
    solid_query::require(bool(std::cout), "Output I/O failure");
    return 0;
  } catch(const std::exception& error) {
    std::string message=error.what();
    for(char& c:message) if(c<32 || c>126) c='?';
    std::cerr<<"certified_solid_query FAIL: "<<message.substr(0,300)<<'\n';
    return 1;
  } catch(...) {
    std::cerr<<"certified_solid_query FAIL: unknown exception\n";
    return 1;
  }
}
