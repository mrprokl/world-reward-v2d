"""Authorization changes metadata only; actual historical audits remain required."""
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def test_explicit_one_replay_authorization_only_change():
    path="configs/ycbv_point_protocol_v2.json"
    old=json.loads(subprocess.check_output(["git","show","dc858f5:"+path]))
    current=json.loads((ROOT/path).read_text())
    old_execution=old.pop("execution");actual=current.pop("execution")
    assert current==old
    assert old_execution=={"authorized":False,"status":"prepared_non_executable","engineering_replay":True}
    assert actual=={"authorized":True,"status":"authorized_one_corrected_engineering_replay","engineering_replay":True,
                    "reason":"explicit_user_authorization_2026-10-04"}
    assert current["selection"]["scene_ids"]==[48,49,50]
    assert current["selection"]["frames_per_scene"]==96
    assert current["selection"]["first_source_frame_id"]==1 and current["selection"]["first_frame_position"]==0
    assert current["limits"]["seconds"]==3600 and current["limits"]["cleanup_seconds"]==180


def test_original_configs_and_failed_evidence_unchanged():
    names=("ycbv_point_protocol.json","ycbv_point_failed_acquisition_pins.json","ycbv_point_continuation_pins.json",
           "ycbv_point_inventory_failed_pins.json","ycbv_archive_header_failed_pins.json",
           "ycbv_archive_header_diagnostic_pins.json","ycbv_native_frame_diagnostic.json")
    for name in names:
        relative="configs/"+name
        assert (ROOT/relative).read_bytes()==subprocess.check_output(["git","show","dc858f5:"+relative])


def test_activation_is_engineering_same_cohort_not_new_validation():
    d=json.loads((ROOT/"configs/ycbv_point_protocol_v2.json").read_text())
    assert d["output"]=={"base":"validation/ycbv_point_pose_v2","public_schema":"world-reward-ycbv-point-rgb-v2"}
    assert d["execution"]["engineering_replay"]is True
    assert d["archives"]["ycbv_test_all.zip"]["bytes"]==14969383039
    assert d["archives"]["ycbv_test_all.zip"]["sha256"]=="fea2ab5f18aba1857acd320827cec10d9dbf258e4940ea4b52f5dd51cb2356a7"


def test_acquisition_diff_is_authorization_only():
    import ast
    path="infra/ycbv_point_acquire.py"
    previous=ast.parse(subprocess.check_output(["git","show","dc858f5:"+path]))
    current=ast.parse((ROOT/path).read_text())
    def neutralize(tree):
        for node in tree.body:
            if isinstance(node,ast.Assign)and any(isinstance(t,ast.Name)and t.id=="EXPECTED_PROTOCOL"for t in node.targets):
                assert isinstance(node.value,ast.Dict)
                index=next(i for i,k in enumerate(node.value.keys)if isinstance(k,ast.Constant)and k.value=="execution")
                node.value.values[index]=ast.Constant(value="authorization-only")
            if isinstance(node,ast.FunctionDef)and node.name=="require_execution":node.body=[ast.Pass()]
        return ast.dump(tree,include_attributes=False)
    assert neutralize(current)==neutralize(previous)
