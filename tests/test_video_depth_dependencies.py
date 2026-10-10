"""Metadata/namespace tests only; never fetch wheels or models in local tests."""
import pytest
from video_depth_dependencies import extract, WHEEL_BYTES, WHEEL_SHA256


def test_exact_pure_python_wheel_identity():
    assert WHEEL_BYTES == 6804
    assert WHEEL_SHA256 == '6b787daf4dcaf6377b4ad9403a5cee5a86adbc0ca9a5bcf5410e9902002aeac2'


def test_unknown_wheel_rejected_without_writes(tmp_path):
    with pytest.raises(ValueError): extract(b'not an official wheel', tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_v2_runtime_and_sensor_namespaces_are_disjoint():
    from pathlib import Path
    root=Path(__file__).parents[1]
    s=(root/'infra/run_video_depth_real_infer_v2.sh').read_text()
    assert 'src=$BASE/inputs,dst=$BASE/inputs,readonly' in s
    assert 'src=$BASE,dst=$BASE' not in s and 'eval_private' not in s
    assert 'video_depth_real_output_v2' in s
    assert 'vda_dependencies_v1' in s
    assert 'infer_v2' in s
    e=(root/'infra/run_video_depth_real_evaluate_v2.sh').read_text()
    assert 'video_depth_real_output_v2' in e and 'evaluate_v2' in e
