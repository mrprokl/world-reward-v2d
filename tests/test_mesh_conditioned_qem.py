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
    assert 'state.finish(physicalVcache,F,true,f1,f2)' in simplify
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
    assert '\\"physical_coordinate_cache\\":' in info
    assert 'physical_coordinate_cache' not in section('std::string volume_mapping(', '\nint main(')


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


def test_compile_macro_default_slow_and_fast_only_storage_with_original_callback_order():
    text=CPP.read_text();simplify=section('bool simplify(', 'std::string volume_mapping(')
    assert '#ifndef WR_CONDITIONED_CACHE\n#define WR_CONDITIONED_CACHE 0\n#endif' in text
    assert '#if WR_CONDITIONED_CACHE != 0 && WR_CONDITIONED_CACHE != 1' in text
    assert simplify.index('return true;') < simplify.index('physicalVcache=chart.decode_matrix(canonical)')
    assert 'Eigen::MatrixXd physicalCcache;' in simplify
    assert 'if(physicalCcache.rows()==0) physicalCcache=Eigen::MatrixXd::Zero(C.rows(),3);' in simplify
    assert 'physicalCcache.rows()!=C.rows()' in simplify
    assert simplify.index('state.allowed(e,physicalV')<simplify.index('native_pre(V,F')<simplify.index('volumes.allowed(e,physicalV')
    post=simplify[simplify.index('post=['):]
    assert post.index('native_post(V,F')<post.index('if(collapsed)')<post.index('physicalVcache.row(t.s)=s')<post.index('state.finish(physicalVcache')
    assert 'else state.finish(physical,F,false,f1,f2)'in post
    assert 'U(v,j)!=physicalVcache(I(v),j)'in post
    assert 'physicalVcache.row('in post and post.count('physicalVcache.row(')==2
    inherited=(ROOT/'infra/mesh_serialization_qem.cpp').read_text()
    assert '__FAST_MATH__'in inherited and 'arithmetic_contract();'in text


def test_tiny_cache_exact_decode_same_sequence_endpoint_only_updates_and_failed_proposals():
    vertices=np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[-1.,-1.,-1.]])*2.**-14
    vertices+=np.array([4.,-2.,1.])*2.**-14
    faces=np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]],np.int64)
    chart=prepare_conditioning(vertices,faces);canonical=chart.encode(vertices).copy();cache=chart.decode(canonical).copy()
    placements=[(0,1,np.array([.125,-.0625,.03125]),True),(2,3,np.array([-.125,.0625,-.03125]),False),
                (0,2,np.array([.25,.125,-.0625]),True)]
    for source,destination,placement,collapsed in placements:
        before=cache.copy();physical_p=chart.decode(placement[None])[0]
        if collapsed:
            canonical[source]=placement;canonical[destination]=placement
            cache[source]=chart.decode(canonical[source:source+1])[0]
            cache[destination]=chart.decode(canonical[destination:destination+1])[0]
            assert cache[source].tobytes()==physical_p.tobytes()==cache[destination].tobytes()
        else:assert cache.tobytes()==before.tobytes()
        assert cache.tobytes()==chart.decode(canonical).tobytes()
        untouched=[i for i in range(len(cache))if i not in(source,destination)]
        assert cache[untouched].tobytes()==before[untouched].tobytes()


def test_cache_initialization_matches_slow_signed_zero_not_source_bitcopy():
    physical=np.array([[-0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],np.float64)
    # One fixed dyadic inverse can legitimately change a signed-zero source bit.
    scale=2.;origin=np.zeros(3)
    canonical=physical/scale;canonical[0,0]=-0.
    slow=origin+canonical*scale;cache=slow.copy()
    assert np.array_equal(cache,physical)and cache.tobytes()!=physical.tobytes()
    assert cache.tobytes()==slow.tobytes()and not np.signbit(cache[0,0])


def test_persistent_placement_row_only_current_edge_can_be_observed():
    candidate_cache=np.zeros((4,3),np.float64);current=np.array([.25,-.5,1.])
    candidate_cache[2]=current
    previous=candidate_cache.copy();candidate_cache[1]=[2.,3.,4.]
    np.testing.assert_array_equal(candidate_cache[2],previous[2])
    np.testing.assert_array_equal(candidate_cache[[0,3]],np.zeros((2,3)))
    serialization=(ROOT/'infra/mesh_serialization_qem.cpp').read_text()
    volume=(ROOT/'infra/mesh_volume_qem.cpp').read_text()
    assert 't.p=position(C,e)'in serialization and 'const Eigen::RowVectorXd p = C.row(e)'in volume


def test_terminal_birth_map_compares_independent_full_decode_to_cached_actual_rows():
    cache=np.array([[0.,1.,2.],[3.,4.,5.],[6.,7.,8.],[9.,10.,11.]])
    birth=np.array([3,0,2],np.int64);exported=cache[birth].copy()
    assert np.array_equal(exported,cache[birth])
    wrong=cache.copy();wrong[2,0]+=1
    assert not np.array_equal(exported,wrong[birth])
