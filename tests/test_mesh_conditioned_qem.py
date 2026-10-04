"""Native source contracts only; actual compilation/geometry stay on Azure."""
import hashlib
import json
from pathlib import Path

import numpy as np

from world_reward.mesh_conditioning import prepare_conditioning

ROOT = Path(__file__).resolve().parents[1]
CPP = ROOT / 'infra/mesh_conditioned_qem.cpp'


def section(first, last):
    text = CPP.read_text()
    return text[text.index(first):text.index(last, text.index(first))]


def test_original_cores_are_unedited_byte_prefixes_not_copied_math():
    text = CPP.read_text()
    assert '#include "wr_serialization_core.hpp"' in text
    assert '#define main' not in text and text.count('int main(') == 1
    assert 'struct Serialization' not in text and 'struct Volumes' not in text
    assert 'void read_obj(' not in text and 'cpp_int' not in section('struct Chart', 'bool simplify(')
    config = json.loads((ROOT/'configs/mesh_conditioned_qem_protocol_v1.json').read_text())
    old = (ROOT/'infra/mesh_serialization_qem.cpp').read_bytes()
    marker = b'\nint main(int argc,char** argv) {'
    assert old.count(marker) == 1
    assert hashlib.sha256(old).hexdigest() == config['source_authentication']['original_serialization_cpp_sha256']
    assert hashlib.sha256(old.split(marker)[0]).hexdigest() == config['source_authentication']['serialization_core_prefix_sha256']


def test_chart_fixed_once_roundtrips_all_source_rows_before_native_call():
    main = CPP.read_text().split('int main(')[1]
    assert main.count('const conditioned_qem::Chart chart(V,F,canonical)') == 1
    assert main.index('Chart chart(V,F,canonical)') < main.index('conditioned_qem::simplify(')
    chart = section('struct Chart', 'bool simplify(')
    assert 'lo.cwiseMin(V.row(F(f,j)))' in chart
    assert 'std::frexp(maximum,&exponent)' in chart and 'std::ldexp(1.,exponent)' in chart
    assert 'roundtrip[j]!=p[j]' in chart and 'for(int v=0;v<V.rows();++v)' in chart
    assert 'Source chart roundtrip not numerically exact' in chart
    assert 'std::memcmp(&roundtrip[j],&p[j],sizeof(double))' in chart
    assert 'volatile double delta=scale*p[j]' in chart and 'volatile double x=origin[j]+delta' in chart
    assert 'std::fma(' not in chart and 'nextafter(' not in chart


def test_original_qslim_computes_canonical_but_physical_guards_and_export_share_inverse():
    simplify = section('bool simplify(', 'std::string volume_mapping(')
    assert 'igl::per_vertex_point_to_plane_quadrics(canonical,F,EMAP,EF,EI,quadrics)' in simplify
    assert 'tree->init(canonical,F)' in simplify
    assert 'igl::qslim_optimal_collapse_edge_callbacks(E,quadrics,v1,v2,cost,pre,post)' in simplify
    assert 'native_pre(V,F,E,EMAP,EF,EI,Q,EQ,C,e)' in simplify
    assert 'state.allowed(e,physicalV,F,E,EMAP,EF,EI,physicalC)' in simplify
    assert 'volumes.allowed(e,physicalV,F,E,EMAP,EF,EI,physicalC)' in simplify
    assert 'physicalC.row(e)=chart.decode(Eigen::RowVector3d(C.row(e)))' in simplify
    assert 'state.finish(chart.decode_matrix(V),F,true,f1,f2)' in simplify
    assert 'U=chart.decode_matrix(canonicalU)' in simplify
    assert 'return state.budget_safe()' in simplify
    assert 'if(!C.row(e).allFinite()) return false' in simplify
    assert 'C.allFinite()' not in simplify and 'Chart(' not in simplify


def test_identity_and_native_births_remain_physical_not_reconstructed():
    simplify = section('bool simplify(', 'std::string volume_mapping(')
    assert simplify.index('igl::remove_unreferenced(physical,F,U,G,old_to_new,I)') < simplify.index('igl::edge_flaps')
    assert 'igl::decimate(canonical,F,cost,stop,pre,post,canonicalU,G,J,I)' in simplify
    main = CPP.read_text().split('int main(')[1]
    assert main.index('volumes.verify_final(U,G,J,I)') < main.index('auto out=exclusive_file')
    assert main.index('!independent.budget_safe()') < main.index('auto out=exclusive_file')
    assert 'conditioned_qem::volume_mapping(V,F,U,G,J,I,volumes,reached)' in main


def test_mapping_reuses_actual_volume_formatter_and_changes_only_algorithm_metadata():
    mapping = section('std::string volume_mapping(', '\nint main(')
    assert 'volume_qem::mapping_json(stream.get(),V,F,U,G,J,I,volumes,reached,true)' in mapping
    assert 'result.find(pair.first,at+1)!=std::string::npos' in mapping
    assert mapping.count('result.replace(') == 1
    assert 'native_cost_and_placement_unchanged' in mapping and 'cost_normalization' in mapping
    assert 'vector_json(' not in mapping and 'relative_volume_error' not in mapping
    info = section('if(argc==2', 'if(argc==3')
    for pair in ('\\"native_cost_and_placement_unchanged\\":false', '\\"new_numeric_algorithm\\":true',
                 '\\"physical_geometry_rescaled\\":false', '\\"adopted\\":false'):
        assert pair in info


def test_manufactured_same_source_chart_inverse_used_for_new_placements():
    v = np.array([[0., 0., 0.], [2., 0., 0.], [0., 1., 0.], [0., 0., .5]]) * 2.**-14
    v += np.array([4., -2., 1.]) * 2.**-14
    f = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    before = v.tobytes(), f.tobytes()
    chart = prepare_conditioning(v, f)
    assert chart.diagnostics['roundtrip_numerically_exact']
    assert chart.decode(chart.encode(v)).tobytes() == v.tobytes()
    proposals = np.array([[1/8, -3/32, 1/16], [-7/64, 5/128, -1/4]])
    physical = chart.decode(proposals)
    assert np.array_equal(physical, chart.origin + chart.scale*proposals)
    assert chart.scale_exponent == -13
    assert (v.tobytes(), f.tobytes()) == before
    assert not chart.diagnostics['metric_fidelity_verified']
