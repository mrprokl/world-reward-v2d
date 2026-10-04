"""Static source contracts only: no compiler, CGAL import or native execution."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "infra/certified_solid_query.cpp"


def section(first, last):
    text = SOURCE.read_text()
    return text[text.index(first):text.index(last, text.index(first))]


def test_standalone_exact_native_headers_and_license_not_a_repair_backend():
    text = SOURCE.read_text()
    assert "SPDX-License-Identifier: GPL-3.0-or-later" in text
    for header in ("Exact_predicates_exact_constructions_kernel.h", "Surface_mesh.h",
                   "Polygon_mesh_processing/self_intersections.h",
                   "Polygon_mesh_processing/intersection.h", "Side_of_triangle_mesh.h"):
        assert f"#include <CGAL/{header}>" in text
    assert "using Kernel = CGAL::Exact_predicates_exact_constructions_kernel;" in text
    assert "using Mesh = CGAL::Surface_mesh<Point>;" in text
    assert "WR_SOLID_QUERY_SOURCE_SHA256" in text and "#ifdef __FAST_MATH__" in text
    for forbidden in ("nlohmann", "qslim(", "decimate(", "repair(", "reverse_face_orientations",
                      "merge_duplicate", "snap(", "nextafter(", "epsilon()"):
        assert forbidden not in text


def test_strict_original_input_and_all_face_vertex_labels_are_retained():
    reader = section("Input read(", "struct Component")
    assert 'token(stream)=="WR_SOLID_QUERY_V1"' in reader
    assert "result.points.emplace_back(x,y,z)" in reader
    assert "face.component=integer(stream,nk-1)" in reader
    assert '"Unexpected trailing input"' in reader
    number = section("double coordinate(", "struct Face")
    assert "std::strtod(value.c_str(),&end)" in number
    assert "std::isfinite(result)" in number
    assert "result!=0 || !nonzero" in number
    assert "errno==ERANGE && std::fpclassify(result)==FP_SUBNORMAL" in number
    token = section("std::string token(", "std::size_t integer(")
    assert "c>=33 && c<=126 && result.size()<128" in token
    integer = section("std::size_t integer(", "double coordinate(")
    assert "number<=(maximum-digit)/10" in integer


def test_full_closed_oriented_vertex_manifold_checked_before_queries():
    validation = section("std::vector<Component> validate(", "std::string query(")
    for required in ("f[0]!=f[1]", "!CGAL::collinear(", "triangles.insert(sorted).second",
                     "edge.second.first==2 && edge.second.second==0",
                     "owner[a]==input.components || owner[a]==face.component",
                     "owner[v]!=input.components", "row.second.size()==2",
                     "visited.size()==link.size()", "visited.size()==component.original_vertices.size()",
                     "CGAL::is_valid_polygon_mesh(component.mesh)", "CGAL::is_closed(component.mesh)",
                     "CGAL::is_triangle_mesh(component.mesh)", "!=Mesh::null_face()",
                     "component.mesh.number_of_faces()==counts[i]"):
        assert required in validation
    assert "continue;" not in validation  # No silent omission of any original face.


def test_volume_sign_is_exact_and_diagnostic_conversion_does_not_choose_orientation():
    validation = section("std::vector<Kernel::FT> volume6", "std::string query(")
    assert "component.original_vertices.front()" in validation
    assert "input.points[f[0]]-origin" in validation
    assert "volume6[face.component]+=" in validation
    assert "CGAL::sign(volume6[i])" in validation
    assert "sign!=CGAL::ZERO" in validation
    assert "component.sign=(sign==CGAL::POSITIVE ? 1 : -1)" in validation
    assert validation.index("CGAL::sign(volume6[i])") < validation.index("CGAL::to_double(")


def test_exact_self_and_pair_contact_checks_precede_per_component_containment():
    query = section("std::string query(", "} // namespace solid_query")
    assert query.index("auto components=validate(input)") < query.index("PMP::does_self_intersect(")
    assert query.index("PMP::does_self_intersect(") < query.index("PMP::do_intersect(")
    assert query.index("PMP::do_intersect(") < query.index("Side_of_triangle_mesh<Mesh,Kernel>")
    assert "CGAL::parameters::do_overlap_test_of_bounded_sides(false)" in query
    assert "side(components[j].mesh)" in query
    assert "side(input.points[components[i].original_vertices.front()])" in query
    assert "answer!=CGAL::ON_BOUNDARY" in query
    assert "answer==CGAL::ON_BOUNDED_SIDE" in query


def test_output_is_single_bounded_result_no_claim_of_forest_or_reconstruction_quality():
    text = SOURCE.read_text()
    main = text[text.index("int main("):]
    assert main.index("solid_query::query(input)") < main.index("std::cout<<output")
    assert 'std::setlocale(LC_NUMERIC,"C")' in main
    assert text.count("std::cout") == 3
    assert "out.str().size()<1048576" in text
    assert "message.substr(0,300)" in main and main.count("return 1;") == 2
    for field in ("original_component_id", "exact_volume_sign", "witness_original_vertex",
                  "represented_coordinates", "inside", "all_original_faces_retained"):
        assert field in text
    for field in ("geometry_repaired", "orientation_changed", "qem_executed",
                  "forest_adjudicated", "reconstruction_accuracy_verified"):
        assert '\\"' + field + '\\":false' in text
