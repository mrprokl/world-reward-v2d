"""Readonly stdlib-only pins for an independently dispatched CPU volume mesh.

All seven artifact identities are measured before strict JSON interpretation.
Expected producer revision/script, video and source-report hashes/scale come
from the caller, never inferred as trusted values from the proposal receipt.
No GLB/NPZ/video/model is decoded; geometry checks remain producer provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import stat
import sys

ROOT = Path("/srv/scenesmith/world-reward")
PRODUCER_SCRIPT = "infra/object_budget_volume.py"
HELPERS = ("infra/volume_mesh_pin_inventory.py","infra/run_volume_mesh_pin_inventory.sh")
SCHEMA = "world-reward-volume-mesh-pins-v1"
STAGE = "world_reward_cpu_volume_constrained_object_mesh"
CONTROL_SHA = "53faaf1c913c4296251680a598414d1b918d36b795d8a44267037e6d26937187"
OFFICIAL_HELPER_SHA = "42ab8ab35f37b806fb1465eadd96abe43eaac04575da47a4855d08eefe6167b0"


def _hex(value, length):
    if type(value) is not str or not re.fullmatch("[0-9a-f]{%d}" % length, value):
        raise ValueError("Exact independently supplied hex identity required")
    return value


def _episode(episode):
    if type(episode) is not int or not 0 <= episode < 30:
        raise ValueError("Explicit Track1 episode integer0..29 required")
    return episode


def paths(episode):
    base = "outputs/episode_%06d" % _episode(episode)
    return dict(report=base+"/object_budget_volume/report.json", geometry=base+"/object_budget_volume/geometry.npz",
        glb=base+"/object_budget_volume/object_fixed_canonical.glb", control="validation/volume_qem_v1/report.json",
        build="results/image-volume-qem.json", object=base+"/object_grounded/report.json", alignment=base+"/scale_smoke/report.json")


def identity(path):
    path = Path(path)
    if not path.is_absolute() or path.resolve() != path or any(item.is_symlink() for item in (path,*path.parents)):
        raise ValueError("Canonical absolute original artifact without symlink ancestors required")
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_size <= 0:
        raise ValueError("Nonempty regular original artifact required")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""): digest.update(chunk)
    after = path.lstat()
    keys = lambda item:(item.st_dev,item.st_ino,item.st_size,item.st_mode,item.st_mtime_ns,item.st_ctime_ns)
    if keys(before) != keys(after): raise ValueError("Original artifact changed while hashing")
    return dict(sha256=digest.hexdigest(),bytes=after.st_size)


def bound_source(root,code,revision,executing=None):
    root,code=Path(root),Path(code);_hex(revision,40)
    if (not root.is_dir() or root.resolve()!=root or not code.is_dir() or code.resolve()!=code
            or code != root/"jobs"/revision/"run_volume_mesh_pin_inventory"/"code"
            or any(path.is_symlink() for path in(code,*code.parents))):
        raise ValueError("Actual immutable volume-inventory dispatch namespace required")
    if executing is not None and Path(executing)!=code/HELPERS[0]:
        raise ValueError("Actual source-bound volume inventory entrypoint required")
    observed={}
    for name in HELPERS+(PRODUCER_SCRIPT,):
        path=code/name;observed[name]=identity(path)
        if path.stat().st_mode&0o222:raise ValueError("Complete original inventory/producer source closure must be readonly")
    return observed


def strict_json(text):
    def pairs(rows):
        result = {}
        for key,value in rows:
            if key in result: raise ValueError("Duplicate report JSON key forbidden")
            result[key] = value
        return result
    def invalid(value): raise ValueError("Nonfinite JSON constants forbidden")
    result = json.loads(text,object_pairs_hook=pairs,parse_constant=invalid)
    def finite(value):
        if type(value) is float and not math.isfinite(value): raise ValueError("Nonfinite JSON values forbidden")
        if type(value) is list:
            for item in value: finite(item)
        elif type(value) is dict:
            for item in value.values(): finite(item)
    finite(result); return result


def _require(record, expected):
    if type(record) is not dict or any(type(record.get(key)) is not type(value) or record[key] != value for key,value in expected.items()):
        raise ValueError("Exact current CPU proposal/source/control fields required")


def _measurement(value, limit):
    if type(value) not in (int,float) or not math.isfinite(value) or not 0 <= value <= limit:
        raise ValueError("Finite unchanged predeclared geometry thresholds required")


def validate_pins(pins,episode,input_sha,object_sha,alignment_sha,scale):
    _episode(episode);_hex(input_sha,64);_hex(object_sha,64);_hex(alignment_sha,64)
    if type(scale) is not float or not math.isfinite(scale) or scale <= 0:
        raise ValueError("Exact positive finite original scalar required")
    if type(pins) is not dict or set(pins) != {"schema","episode_index","input_sha256","metric_scale_baked_once","report","files"}:
        raise ValueError("Exact generic CPU volume pins schema required")
    _require(pins,dict(schema=SCHEMA,episode_index=episode,input_sha256=input_sha,metric_scale_baked_once=scale))
    if type(pins["files"]) is not dict or set(pins["files"]) != set(paths(episode).values()):
        raise ValueError("Exactly seven original proposal/control/source artifact pins required")
    for row in pins["files"].values():
        if type(row) is not dict or set(row) != {"sha256","bytes"} or type(row["bytes"]) is not int or row["bytes"] <= 0:
            raise ValueError("Exact SHA/positive byte identities required")
        _hex(row["sha256"],64)
    report = pins["report"]
    if type(report) is not dict or set(report) != {"sha256","bytes","producer_revision","script_sha256"}:
        raise ValueError("Independent original CPU producer pin required")
    _hex(report["producer_revision"],40);_hex(report["script_sha256"],64)
    _hex(report["sha256"],64)
    if type(report["bytes"]) is not int or report["bytes"]<=0:raise ValueError("Exact positive report byte count required")
    if {key:report[key] for key in ("sha256","bytes")} != pins["files"][paths(episode)["report"]]:
        raise ValueError("Original report identity differs within manifest")
    if (pins["files"][paths(episode)["object"]]["sha256"] != object_sha
            or pins["files"][paths(episode)["alignment"]]["sha256"] != alignment_sha):
        raise ValueError("Independently supplied original source-report hashes differ")


def validate_reports(records,pins):
    episode = pins["episode_index"]; p = paths(episode)
    report,obj,alignment,control,build = (records[key] for key in ("report","object","alignment","control","build"))
    _require(report,dict(stage=STAGE,status="pass",episode_index=episode,input_track="track_1",
        producer_revision=pins["report"]["producer_revision"],script_sha256=pins["report"]["script_sha256"],
        input_sha256=pins["input_sha256"],ground_truth_used=False,hand_labeled_test=False,oracle_modes=[],
        adoption_performed=False,challenge_performance_verified=False,budget_seconds=900,target_faces=4096,target_vertices=4096,
        source_shell_volume_relative_limit=.05,components_deleted=False,holes_filled=False,normals_repaired=False,
        frame_poses_changed=False,native_cost_and_placement_unchanged=True,source_embedding_exact_universal_proof=False,
        source_intersecting_faces=0,independent_candidate_intersecting_faces=0,packed_intersecting_faces=0,
        metric_scale_baked_once=pins["metric_scale_baked_once"],source_arrays_unchanged=True,official_helper_sha256=OFFICIAL_HELPER_SHA,
        geometry_sha256=pins["files"][p["geometry"]]["sha256"],canonical_glb_sha256=pins["files"][p["glb"]]["sha256"]))
    for field in ("candidate_geometry","export_geometry"):
        measurements=report.get(field)
        if type(measurements) is not dict or type(measurements.get("birthface_matched_shells")) is not list or not measurements["birthface_matched_shells"]:
            raise ValueError("Every original birth-face-matched shell measurement required")
        _measurement(measurements.get("sampled_bidirectional_chamfer_diagonal_ratio"),.01)
        _measurement(measurements.get("net_volume_relative_error"),.05)
        for shell in measurements["birthface_matched_shells"]:
            if type(shell) is not dict: raise ValueError("Original shell report required")
            _measurement(shell.get("relative_volume_error"),.05)
    _require(report.get("official_pack_fidelity"),dict(oriented_triangles_exact=True,official_helper_simplification_invoked=False,
        nonexact_merge_or_face_deletion=False))
    for record,stage in ((obj,"sam3d_objects_grounded_fixed_frame"),(alignment,"predicted_human_anchored_moge2_pointmaps")):
        _require(record,dict(stage=stage,status="pass",episode_index=episode,input_track="track_1",
            input_sha256=pins["input_sha256"],ground_truth_used=False,hand_labeled_test=False,oracle_modes=[]))
    _require(obj,dict(frame_index=0,scale_source="already_human_anchored_MoGe2_no_second_scalar"))
    _require(obj.get("pointmap_grounding"),dict(alignment_report_sha256=pins["files"][p["alignment"]]["sha256"]))
    _require(alignment,dict(coordinate_frame="OpenCV_x_right_y_down_z_forward",
        pointmap_scale_application="one_clip_scalar_to_MoGe2_XYZ_already_applied"))
    source = report.get("source_hashes")
    _require(source,dict(object_report=pins["files"][p["object"]]["sha256"],alignment_report=pins["files"][p["alignment"]]["sha256"],video=pins["input_sha256"]))
    for key,field in (("object.glb","object_sha256"),("transform.json","transform_sha256"),("intrinsics.json","intrinsics_sha256")):
        _hex(obj.get(field),64)
        if source.get(key) != obj[field]: raise ValueError("Original object artifact ancestry differs")
    for folder,field in (("body_smoke_report","body_report_sha256"),("depth_smoke_report","depth_report_sha256")):
        _hex(alignment.get(field),64)
        if source.get(folder) != alignment[field]: raise ValueError("Original alignment source ancestry differs")
    for field in ("mask_report","prompts"): _hex(source.get(field),64)
    transform=obj.get("transform")
    if type(transform) is not dict or type(transform.get("scale")) is not list or len(transform["scale"]) != 3:
        raise ValueError("Original uniform grounding scale metadata required")
    scales=transform["scale"]
    if (any(type(value) not in (int,float) or not math.isfinite(value) or value<=0 for value in scales)
            or float(scales[0]) != pins["metric_scale_baked_once"]
            or any(abs(value-scales[0]) > 1e-5*abs(scales[0]) for value in scales)):
        raise ValueError("Original positive isotropic scale cannot be averaged or applied twice")
    _require(control,dict(stage="own_volume_constrained_intersection_qem_geometry",status="pass",script_sha256=CONTROL_SHA,
        challenge_inputs_used=False,adoption_performed=False,target_faces=4096,target_vertices=4096))
    _require(build,dict(stage="world_reward_volume_qem_build",status="pass",image_id=report.get("image_id")))
    if type(report.get("image_id")) is not str or not re.fullmatch("sha256:[0-9a-f]{64}",report["image_id"]):
        raise ValueError("Actual immutable original CPU image required")
    evidence=report.get("control_evidence")
    _require(evidence,dict(volume_gate_sha256=pins["files"][p["control"]]["sha256"],build_report_sha256=pins["files"][p["build"]]["sha256"]))
    if control.get("build") != {key:value for key,value in evidence.items() if key != "volume_gate_sha256"}:
        raise ValueError("Original complete CPU control/build evidence differs")
    return report


def verify_pinned_artifacts(root,pins,episode,input_sha,object_sha,alignment_sha,scale):
    root=Path(root);validate_pins(pins,episode,input_sha,object_sha,alignment_sha,scale)
    observed={name:identity(root/name) for name in sorted(pins["files"])}
    if observed != pins["files"]: raise ValueError("All seven independent artifact SHA/bytes must match before JSON")
    records={role:strict_json((root/relative).read_text()) for role,relative in paths(episode).items() if role not in ("geometry","glb")}
    report=validate_reports(records,pins)
    if {name:identity(root/name) for name in sorted(pins["files"])} != observed:
        raise ValueError("Original artifacts changed during report validation")
    return report,observed


def inventory(root,episode,producer_revision,producer_script_sha256,input_sha,object_sha,alignment_sha,scale):
    _episode(episode);_hex(producer_revision,40);_hex(producer_script_sha256,64)
    _hex(input_sha,64);_hex(object_sha,64);_hex(alignment_sha,64)
    # 0644 producer outputs are legitimate: read-only here means no mutation,
    # not a fabricated chmod/freeze. Consumers use explicit RO bind mounts.
    observed={name:identity(Path(root)/name) for name in sorted(paths(episode).values())}
    pins=dict(schema=SCHEMA,episode_index=episode,input_sha256=input_sha,metric_scale_baked_once=scale,
        report=dict(observed[paths(episode)["report"]],producer_revision=producer_revision,script_sha256=producer_script_sha256),files=observed)
    verify_pinned_artifacts(root,pins,episode,input_sha,object_sha,alignment_sha,scale)
    return pins


def parser():
    class Once(argparse.Action):
        def __call__(self,parser,namespace,value,option_string=None):
            if getattr(namespace,self.dest,None) is not None: parser.error("Each explicit source/clip control must occur once")
            setattr(namespace,self.dest,value)
    result=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    def episode_text(value):
        if not re.fullmatch("0|[1-9]|[12][0-9]",value):raise argparse.ArgumentTypeError("Canonical explicit Track1 episode0..29 required")
        return int(value)
    result.add_argument("--episode",type=episode_text,choices=range(30),required=True,action=Once)
    for field in ("producer-revision","producer-script-sha256","input-sha256","object-report-sha256","alignment-report-sha256"):
        result.add_argument("--"+field,required=True,action=Once)
    result.add_argument("--scale",type=float,required=True,action=Once)
    return result


def main(argv=None):
    args=parser().parse_args(argv)
    if platform.system() != "Linux" or os.environ.get("WR_ROOT") != str(ROOT):
        raise RuntimeError("Canonical Azure Linux control-host inventory only")
    code=Path(os.environ["WR_CODE"]);revision=os.environ["WR_CODE_REVISION"]
    helpers=bound_source(ROOT,code,revision,Path(__file__))
    if helpers[PRODUCER_SCRIPT]["sha256"]!=args.producer_script_sha256:
        raise ValueError("Original producer source code differs from independently supplied SHA")
    pins=inventory(ROOT,args.episode,args.producer_revision,args.producer_script_sha256,args.input_sha256,
        args.object_report_sha256,args.alignment_report_sha256,args.scale)
    if bound_source(ROOT,code,revision,Path(__file__))!=helpers:
        raise ValueError("Original inventory/producer source changed during readonly inventory")
    json.dump(pins,sys.stdout,allow_nan=False,separators=(",",":"));sys.stdout.write("\n")


if __name__ == "__main__": main()
