// Own CLI wrapper. libigl/Eigen remain external MPL-2.0 sources with notices.
// Intersection blocking is a floating-point upstream guard, not an exact proof.
#include <igl/qslim.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <limits>
#include <locale>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>

#ifndef WR_SOURCE_SHA256
#error Build must bind the original wrapper source SHA256
#endif
#define WR_LIBIGL_REVISION "40e7900ccbd767f1f360e0eb10f0f1a6432e0993"
#define WR_EIGEN_REVISION "3147391d946bb4b6c68edd901f2add6ac1f31f8c"
constexpr int TARGET = 4096;

namespace {
using File = std::unique_ptr<FILE, decltype(&std::fclose)>;
File exclusive_file(const char* path) {
  const int fd = ::open(path, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW, 0644);
  if (fd < 0) throw std::runtime_error("Output is not exclusively creatable");
  FILE* f = ::fdopen(fd, "w");
  if (!f) { ::close(fd); throw std::runtime_error("fdopen failed"); }
  return File(f, &std::fclose);
}

void require_absent(const char* path) {
  struct stat st;
  if (::lstat(path, &st) == 0 || errno != ENOENT)
    throw std::runtime_error("Output already exists or is inaccessible");
}

void read_obj(const char* path, Eigen::MatrixXd& V, Eigen::MatrixXi& F) {
  struct stat st;
  if (::lstat(path, &st) != 0 || !S_ISREG(st.st_mode))
    throw std::runtime_error("Input must be a regular non-symlink OBJ");
  std::ifstream in(path); in.imbue(std::locale::classic());
  if (!in) throw std::runtime_error("Cannot open input OBJ");
  std::vector<std::array<double, 3>> vertices;
  std::vector<std::array<int, 3>> faces;
  std::string line;
  while (std::getline(in, line)) {
    line.resize(line.find('#') == std::string::npos ? line.size() : line.find('#'));
    std::istringstream row(line); row.imbue(std::locale::classic());
    std::string kind, extra; if (!(row >> kind)) continue;
    if (kind == "v") {
      std::array<double, 3> v;
      if (!(row >> v[0] >> v[1] >> v[2]) || (row >> extra))
        throw std::runtime_error("Vertices require exactly three numbers");
      for (double x : v) if (!std::isfinite(x)) throw std::runtime_error("Nonfinite vertex");
      vertices.push_back(v);
    } else if (kind == "f") {
      std::array<int, 3> f;
      for (int& x : f) {
        std::string token; if (!(row >> token)) throw std::runtime_error("Nontriangular face");
        const auto parsed = std::from_chars(token.data(), token.data() + token.size(), x);
        if (parsed.ec != std::errc() || parsed.ptr != token.data() + token.size() || x <= 0)
          throw std::runtime_error("Faces require positive plain integer indices");
        --x;
      }
      if (row >> extra) throw std::runtime_error("Nontriangular face");
      faces.push_back(f);
    } else throw std::runtime_error("Only v/f records and comments are supported");
    if (vertices.size() >= static_cast<size_t>(std::numeric_limits<int>::max()) ||
        faces.size() >= static_cast<size_t>(std::numeric_limits<int>::max()))
      throw std::runtime_error("Input exceeds Eigen integer index range");
  }
  if (!in.eof() || vertices.size() < 4 || faces.size() < 4)
    throw std::runtime_error("Incomplete or empty closed surface");
  V.resize(vertices.size(), 3); F.resize(faces.size(), 3);
  for (int i = 0; i < V.rows(); ++i) for (int j = 0; j < 3; ++j) V(i,j) = vertices[i][j];
  struct Edge { int count = 0, direction = 0; };
  std::unordered_map<uint64_t, Edge> edges;
  for (int i = 0; i < F.rows(); ++i) {
    for (int j = 0; j < 3; ++j) {
      F(i,j) = faces[i][j];
      if (F(i,j) >= V.rows()) throw std::runtime_error("Face index out of bounds");
    }
    if (F(i,0) == F(i,1) || F(i,1) == F(i,2) || F(i,2) == F(i,0))
      throw std::runtime_error("Repeated-index triangle");
    const Eigen::Vector3d a = (V.row(F(i,1)) - V.row(F(i,0))).transpose();
    const Eigen::Vector3d b = (V.row(F(i,2)) - V.row(F(i,0))).transpose();
    if (a.cross(b).squaredNorm() == 0.) throw std::runtime_error("Collapsed triangle");
    for (int j = 0; j < 3; ++j) {
      const uint32_t aidx = F(i,j), bidx = F(i,(j+1)%3);
      const uint64_t key = (uint64_t(std::min(aidx,bidx)) << 32) | std::max(aidx,bidx);
      Edge& e = edges[key]; ++e.count; e.direction += aidx < bidx ? 1 : -1;
    }
  }
  for (const auto& item : edges) if (item.second.count != 2 || item.second.direction != 0)
    throw std::runtime_error("Input must be closed and consistently edge-oriented; no repair");
}

void vector_json(FILE* f, const Eigen::VectorXi& values, bool complete) {
  std::fputc('[', f);
  if (complete) for (int i = 0; i < values.size(); ++i)
    std::fprintf(f, "%s%d", i ? "," : "", values(i));
  std::fputc(']', f);
}
}

int main(int argc, char** argv) {
  try {
    if (argc == 2 && std::string(argv[1]) == "--build-info") {
      std::cout << "{\"libigl_revision\":\"" << WR_LIBIGL_REVISION
        << "\",\"eigen_revision\":\"" << WR_EIGEN_REVISION
        << "\",\"source_sha256\":\"" << WR_SOURCE_SHA256
        << "\",\"target_faces\":4096,\"block_intersections\":true}\n";
      return 0;
    }
    if (argc != 4) throw std::runtime_error("Usage: mesh_guarded_qem input.obj output.obj mapping.json");
    require_absent(argv[2]); require_absent(argv[3]);
    if (std::string(argv[2]) == argv[3]) throw std::runtime_error("Output paths must differ");
    Eigen::MatrixXd V, U; Eigen::MatrixXi F, G; Eigen::VectorXi J, I;
    read_obj(argv[1], V, F);
    const bool reached = igl::qslim(V, F, TARGET, true, U, G, J, I);
    if (U.cols() != 3 || G.cols() != 3 || !U.allFinite() || U.rows() < 4 || G.rows() < 4 ||
        J.size() != G.rows() || I.size() != U.rows() || G.minCoeff() < 0 ||
        G.maxCoeff() >= U.rows() || J.minCoeff() < 0 || J.maxCoeff() >= F.rows() ||
        I.minCoeff() < 0 || I.maxCoeff() >= V.rows())
      throw std::runtime_error("Invalid native output or birth maps");
    const bool budget = U.rows() <= TARGET && G.rows() <= TARGET;
    // No accepted export on failure: these files are diagnostic output only.
    auto out = exclusive_file(argv[2]);
    for (int i = 0; i < U.rows(); ++i) std::fprintf(out.get(), "v %.17g %.17g %.17g\n", U(i,0), U(i,1), U(i,2));
    for (int i = 0; i < G.rows(); ++i) std::fprintf(out.get(), "f %d %d %d\n", G(i,0)+1, G(i,1)+1, G(i,2)+1);
    if (std::fflush(out.get()) || std::ferror(out.get())) throw std::runtime_error("OBJ write failed");
    auto map = exclusive_file(argv[3]);
    std::fprintf(map.get(), "{\"source_faces\":%d,\"source_vertices\":%d,\"output_faces\":%d,\"output_vertices\":%d,\"target_faces\":4096,\"block_intersections\":true,\"target_reached\":%s,\"mapping_complete\":%s,\"J\":", int(F.rows()), int(V.rows()), int(G.rows()), int(U.rows()), reached ? "true" : "false", budget ? "true" : "false");
    vector_json(map.get(), J, budget); std::fputs(",\"I\":", map.get()); vector_json(map.get(), I, budget);
    std::fputs("}\n", map.get());
    if (std::fflush(map.get()) || std::ferror(map.get())) throw std::runtime_error("Mapping write failed");
    return reached && budget ? 0 : 3;
  } catch (const std::exception& error) {
    std::cerr << "mesh_guarded_qem: " << error.what() << '\n'; return 2;
  }
}
