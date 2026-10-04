"""Native offline BootsTAPIR on public RGB/oracle initial queries, never labels."""
import argparse
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import re
import signal
import stat
import time

import robotap_boots_acquire as source

ROOT = source.ROOT
BASE = "validation/robotap_boots_v1"
PUBLIC = BASE + "/public_v2"
OUT = BASE + "/infer_v2"
JOB = "run_robotap_boots_infer"
STAGE = "public_robotap_native_bootstapir_predictions"
PINS = "configs/robotap_boots_inference_pins.json"
PROTOCOL = "configs/robotap_boots_protocol.json"
IMAGE = "sha256:ef12f589dd270e56be3a2d2e2f33ccd356e5b160a5c6ca03b8a9449ccc10d1e4"
BUDGET = 900
PREVIOUS_FAILURE = dict(bytes=1022,sha256='054bedb32ae379910f18dacd09a749a095a33d3768d2b41ce61cbedbea47c4f1')
PREVIOUS_REVISION = '9dfccb7999c56fe5ff7d262af7299079c467d29b'
RESOLUTION = (256, 256)
QUERY_CHUNK = 32
DEMO = dict(url="https://raw.githubusercontent.com/google-deepmind/tapnet/730cda1c730877cfedbe01bf87fb1cadb78a565d/colabs/torch_tapir_demo.ipynb", bytes=17144,
            sha256="1eae7b199b4c9800e1e1d97295ed9b7b058a53d938e8417bd3c14caf0fe355d3")


def require(value, message):
    if not value: raise ValueError(message)


def strict_json(raw):
    def pairs(rows):
        result = {}
        for key, value in rows:
            require(key not in result, "Duplicate JSON key"); result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))


def pin(value, producer=False):
    keys = {"bytes", "sha256"} | ({"producer_revision", "script_sha256"} if producer else set())
    require(type(value) is dict and set(value) == keys and type(value["bytes"]) is int and value["bytes"] > 0, "Exact positive byte pins required")
    fields = (("sha256", 64),) + ((("producer_revision", 40), ("script_sha256", 64)) if producer else ())
    for key, length in fields:
        require(type(value[key]) is str and re.fullmatch(fr"[0-9a-f]{{{length}}}", value[key]), "Exact hash/revision pins required")


def validate_pins(pins):
    require(type(pins) is dict and set(pins) == {"schema", "public", "runtime", "image_id", "protocol"}
            and pins["schema"] == "world-reward-robotap-boots-inference-pins-v1" and pins["image_id"] == IMAGE, "Frozen inference pins/image required")
    public = pins["public"]
    require(type(public) is dict and set(public) == {"sha256", "bytes", "producer_revision", "script_sha256", "files"}, "Complete public producer pins required")
    pin({k: public[k] for k in public if k != "files"}, True)
    names = {"manifest.json", "video_000.npz", "video_001.npz", "video_002.npz"}
    require(type(public["files"]) is dict and set(public["files"]) == names, "Exact three fullvideo public artifacts required")
    for row in public["files"].values(): pin(row)
    runtime = pins["runtime"]
    require(type(runtime) is dict and set(runtime) == {"sha256", "bytes", "producer_revision", "script_sha256", "report_path"}, "Independent CPU verifier pins required")
    pin({k: runtime[k] for k in runtime if k != "report_path"}, True)
    require(runtime["report_path"] == f"results/bootstapir-runtime-verify-{runtime['producer_revision']}/report.json", "Canonical CPU verifier report required")
    pin(pins["protocol"])
    require(pins["protocol"] == dict(bytes=source.PROTOCOL_BYTES, sha256=source.PROTOCOL_SHA256), "Whole original protocol pin required")


def checked(path, wanted):
    actual = source.identity(path)
    require(actual == {k: wanted[k] for k in ("sha256", "bytes")}, "Pinned artifact changed")
    return actual


def bindings(root, code, revision, pins):
    validate_pins(pins); root, code = map(source.canonical, (root, code))
    require(root == ROOT and code == root / "jobs" / revision / JOB / "code"
            and re.fullmatch(r"[0-9a-f]{40}", revision) and Path(__file__) == code / "infra/robotap_boots_infer.py" and Path(source.__file__) == code / "infra/robotap_boots_acquire.py", "Actual immutable inference source required")
    files = {}
    for path in (code, *sorted(code.rglob("*"))):
        source.canonical(path)
        require(not path.stat().st_mode & 0o222 and (path.is_dir() or path.is_file()), "Complete readonly source closure required")
        if path.is_file(): files[str(path.relative_to(code))] = source.identity(path)
    markers = {name: source.identity(code.parent / name, readonly=False) for name in ("revision", "source-sha256")}
    require((code.parent / "revision").read_bytes() == (revision + "\n").encode()
            and re.fullmatch(b"[0-9a-f]{64}\n", (code.parent / "source-sha256").read_bytes()), "Original dispatch markers required")
    previous=previous_failure(root)
    protocol = source.read_protocol(code / PROTOCOL)
    native = {name: checked(root / BASE / "assets/tapnet_source" / name, row) for name, row in protocol["source"]["files"].items()}
    checkpoint = checked(root / BASE / "assets" / protocol["checkpoint"]["file"], protocol["checkpoint"])
    runtime_path = root / pins["runtime"]["report_path"]; checked(runtime_path, pins["runtime"])
    runtime = strict_json(runtime_path.read_bytes()); cpu = runtime.get("cpu_import", {})
    expected = dict(stage="bootstapir_runtime_verify", status="pass", phase="complete", producer_revision=pins["runtime"]["producer_revision"],
                    child_image_id=IMAGE, native_source_revision=protocol["source"]["revision"], original_build_status="fail",
                    original_build_failure_preserved=True, source_rehashed_after=True, gpu_execution=False, checkpoint_read=False, rgb_or_labels_read=False)
    require(all(type(runtime.get(k)) is type(v) and runtime[k] == v for k, v in expected.items()) and runtime.get("native_sources") == native, "Actual independent native CPU verification required")
    require(runtime.get("source_helpers", {}).get("infra/run_bootstapir_runtime_verify.sh", {}).get("sha256") == pins["runtime"]["script_sha256"], "CPU verifier producer script differs")
    require(cpu.get("versions") == {"dm-tree": "0.1.10", "einshape": "1.0", "absl-py": "2.5.0", "attrs": "26.1.0", "wrapt": "1.17.3"}
            and cpu.get("python") == "3.11" and cpu.get("torch") == "2.5.1+cu124" and cpu.get("numpy") == "1.26.3"
            and all(cpu.get(k) is True for k in ("tree_cpu_verified", "source_native_einshape_cpu_verified", "native_bilinear_cpu_verified", "native_tapir_modules_imported"))
            and cpu.get("models_instantiated") is False and cpu.get("cuda_initialized") is False, "Exact native CPU API/dependencies required")
    report_path = root / PUBLIC / "report.json"; checked(report_path, pins["public"])
    report = strict_json(report_path.read_bytes())
    require(report.get("stage") == "external_robotap_oracle_initial_query_public_adapter" and report.get("status") == "pass"
            and report.get("producer_revision") == pins["public"]["producer_revision"]
            and report.get("source_before", {}).get("files", {}).get("infra/robotap_boots_public.py", {}).get("sha256") == pins["public"]["script_sha256"]
            and report.get("source_after") == report.get("source_before") and report.get("public_files") == pins["public"]["files"]
            and all(report.get(k) is True for k in ("source_after_reverified", "originals_after_reverified", "initial_queries_are_external_oracles"))
            and all(report.get(k) is False for k in ("future_labels_available_to_inference", "inference_performed", "evaluation_performed", "gpu_used", "challenge_inputs_used")), "Completed public-only adapter required")
    public_files = {name: checked(root / PUBLIC / "inputs" / name, row) for name, row in pins["public"]["files"].items()}
    require({p.name for p in (root / PUBLIC / 'inputs').iterdir()} == set(public_files), "Exact public-only input inventory required")
    manifests = public_records(root / PUBLIC / 'inputs')
    require(report.get('actual_full_frame_matrix') == [r['frames'] for r in manifests], "Adapter's original timelines differ")
    return dict(source_files=files, markers=markers, native_sources=native, checkpoint=checkpoint,
                runtime=checked(runtime_path, pins["runtime"]), previous_inference_failure=previous, public_report=checked(report_path, pins["public"]), public_files=public_files)


def public_records(directory):
    manifest = strict_json((directory / "manifest.json").read_bytes())
    expected = {'schema','videos','selection','initial_query_is_external_oracle','query_format','future_tracks_or_visibility_public','frame_crop_or_resize',
                'training_overlap_verified','challenge_overlap_verified','full_hoi_accuracy_verified','tap_query_source','public_namespace','serialization_decoder'}
    require(type(manifest) is dict and set(manifest) == expected and manifest.get("schema") == "world-reward-robotap-boots-public-v1" and manifest.get("initial_query_is_external_oracle") is True
            and manifest.get("query_format") == "t,y,x; normalized_xy multiplied by original W,H; no half-pixel offset"
            and all(manifest.get(k) is False for k in ("future_tracks_or_visibility_public", "frame_crop_or_resize", "training_overlap_verified", "challenge_overlap_verified", "full_hoi_accuracy_verified")), "Original public RGB/oracle query schema required")
    decoder=manifest.get('serialization_decoder')
    require(manifest.get('public_namespace')=='public_v2' and type(decoder) is dict
            and decoder.get('source_sha256')=='279aa5b1c1cf1d5b2e2025f76c8594df6312fdc65e9431636448926271eccca2'
            and decoder.get('source_bytes')==73425 and decoder.get('allowed_global')=='mediapy._VideoArray'
            and decoder.get('mediapy_package_imported') is False, 'Pinned public-v2 ndarray-only serialization decoder required')
    records = manifest.get("videos"); require(type(records) is list and len(records) == 3, "All three selected videos required")
    for i, row in enumerate(records):
        keys={'file','sha256','bytes','source_pickle','video_key','frames','height','width','point_indices','unavailable_original_indices','query_count','all_original_frames_retained'}
        require(type(row) is dict and set(row)==keys and row.get("file") == f"video_{i:03d}.npz" and row.get("all_original_frames_retained") is True
                and all(type(row.get(k)) is int and row[k] > 0 for k in ("frames", "height", "width", "query_count"))
                and 1 <= row["query_count"] <= 32, "Full original frame/query dimensions required")
        pin({k: row[k] for k in ("bytes", "sha256")}); checked(directory / row["file"], row)
    selection=[dict(pickle_file=r['source_pickle'],video_key=r['video_key']) for r in records]
    require(manifest['selection']==selection and [r['source_pickle'] for r in records]==[f'eval_private/pickles/robotap/robotap_split{i}.pkl' for i in range(3)]
            and len({r['video_key'] for r in records})==3 and all(type(r['video_key']) is str and r['video_key'] and r['video_key'].isascii()
                and not any(ord(c)<32 or ord(c)==127 for c in r['video_key']) for r in records), 'Original first lexicographic video identities required')
    return records


def validate_video(data, row):
    import numpy as np
    require(set(data) == {"video", "query_points", "point_indices"} and not any(np.ma.isMaskedArray(v) for v in data.values()), "Public arrays only; no masks or future labels")
    video, queries, indices = (data[k] for k in ("video", "query_points", "point_indices"))
    t, h, w, n = (row[k] for k in ("frames", "height", "width", "query_count"))
    require(video.dtype == np.uint8 and video.shape == (t, h, w, 3) and queries.dtype == np.float64 and queries.shape == (n, 3)
            and indices.dtype == np.int64 and indices.shape == (n,) and indices.tolist() == row["point_indices"]
            and np.all(np.diff(indices) > 0) and np.all((indices >= 0) & (indices < 32)) and np.isfinite(queries).all(), "Exact RGB/original-query arrays required")
    unavailable = row["unavailable_original_indices"]
    require(type(unavailable) is list and all(type(i) is int and 0 <= i < 32 for i in unavailable)
            and unavailable == sorted(set(unavailable)) and not set(unavailable) & set(indices.tolist()), "Original unavailable query indices required")
    require(np.all(queries[:, 0] == np.floor(queries[:, 0])) and np.all((queries[:, 0] >= 0) & (queries[:, 0] < t))
            and np.all((queries[:, 1:] >= 0) & (queries[:, 1:] <= [h, w])), "Native tyx query bounds required")


def native_modules(native_root):
    import sys
    sys.path.insert(0, str(native_root))
    modules = {name: importlib.import_module("tapnet.torch." + name) for name in ("tapir_model", "utils", "nets")}
    for name, module in modules.items():
        require(source.canonical(Path(module.__file__)) == native_root / f"tapnet/torch/{name}.py", "Native module origin differs")
    for name in ('tapnet', 'tapnet.torch'):
        module=importlib.import_module(name)
        require(source.canonical(Path(module.__file__)) == native_root / (name.replace('.', '/')+'/__init__.py'), 'Native namespace origin differs')
    return modules["tapir_model"], modules["utils"]


def load_model(torch, tapir, checkpoint):
    model = tapir.TAPIR(pyramid_level=1)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    expected_state=model.state_dict()
    require(isinstance(state, dict) and set(state) == set(expected_state), "Unwrapped exact native checkpoint state required")
    for name, value in state.items():
        require(isinstance(value, torch.Tensor) and value.shape == expected_state[name].shape
                and value.dtype == expected_state[name].dtype and torch.isfinite(value).all().item(), "Finite exact checkpoint tensor inventory required")
    model.load_state_dict(state, strict=True); del state
    model = model.float().eval().to("cuda")
    require(not model.training and all(not p.is_floating_point() or p.dtype == torch.float32 for p in model.state_dict().values()), "Native evaluation FP32 state required")
    return model


def native_prediction(torch, utils, model, data, report=None):
    import numpy as np
    video, queries = data["video"], data["query_points"]
    t, h, w = video.shape[:3]
    frames = torch.as_tensor(video, device="cuda", dtype=torch.float32)[None]
    frames = utils.bilinear(frames, RESOLUTION) / 255 * 2 - 1
    q = torch.as_tensor(queries, device="cuda", dtype=torch.float32)[None]
    q = utils.convert_grid_coordinates(q, (t, h, w), (t, *RESOLUTION), coordinate_format="tyx")
    if report is not None: report['native_calls_attempted'] += 1
    with torch.no_grad(): result = model(frames, q, is_training=False, query_chunk_size=QUERY_CHUNK)
    if report is not None: report['native_calls_returned'] += 1
    require(type(result) is dict and {"tracks", "occlusion", "expected_dist"} <= set(result), "Native prediction fields required")
    n = len(queries)
    require(result["tracks"].shape == (1, n, t, 2) and result["occlusion"].shape == result["expected_dist"].shape == (1, n, t)
            and all(result[k].dtype == torch.float32 and torch.isfinite(result[k]).all().item() for k in ("tracks", "occlusion", "expected_dist")), "Full native finite FP32 predictions required")
    tracks = utils.convert_grid_coordinates(result["tracks"][0], (256, 256), (w, h))
    visible = (1 - torch.sigmoid(result["occlusion"][0])) * (1 - torch.sigmoid(result["expected_dist"][0])) > .5
    arrays = {k: result[k][0].detach().cpu().numpy().copy() for k in ("occlusion", "expected_dist")}
    arrays.update(tracks=tracks.detach().cpu().numpy().copy(), tracks_256=result["tracks"][0].detach().cpu().numpy().copy(),
                  visible=visible.detach().cpu().numpy().copy(), query_points=queries.copy(), point_indices=data["point_indices"].copy(), frame_index=np.arange(t, dtype=np.int64),
                  static_tracks=np.broadcast_to(queries[:, [2, 1]][:, None], (n, t, 2)).copy(), static_visible=np.ones((n, t), dtype=bool))
    del result, frames, q
    return arrays


def validate_prediction(data, row, queries=None, indices=None):
    import numpy as np
    keys={'tracks','tracks_256','occlusion','expected_dist','visible','query_points','point_indices','frame_index','static_tracks','static_visible'}
    require(set(data)==keys and not any(np.ma.isMaskedArray(v) for v in data.values()),'Exact native prediction/control fields required')
    n,t=row['query_count'],row['frames']
    for name,shape in (('tracks',(n,t,2)),('tracks_256',(n,t,2)),('occlusion',(n,t)),('expected_dist',(n,t))):
        require(data[name].dtype==np.float32 and data[name].shape==shape and np.isfinite(data[name]).all(),'Full finite FP32 native arrays required')
    require(data['visible'].dtype==data['static_visible'].dtype==np.bool_ and data['visible'].shape==data['static_visible'].shape==(n,t)
            and data['static_visible'].all(),'All-frame native and explicit static-control visibility required')
    require(data['query_points'].dtype==np.float64 and data['query_points'].shape==(n,3) and np.isfinite(data['query_points']).all()
            and data['point_indices'].dtype==data['frame_index'].dtype==np.int64 and data['point_indices'].tolist()==row['point_indices']
            and np.array_equal(data['frame_index'],np.arange(t,dtype=np.int64)),'Original query/point/timeline identity required')
    scaled=(data['tracks_256']*np.asarray([row['width'],row['height']],np.float32))/np.asarray([256,256],np.float32)
    require(np.array_equal(scaled,data['tracks']) and data['static_tracks'].dtype==np.float64
            and np.array_equal(data['static_tracks'],np.broadcast_to(data['query_points'][:,[2,1]][:,None],(n,t,2))), 'Native multiply-then-divide coordinates/static control changed')
    if queries is not None:require(np.array_equal(queries,data['query_points']) and np.array_equal(indices,data['point_indices']),'Frozen query arrays changed')


def preflight(root, code, revision):
    pinpath=code/PINS;source.identity(pinpath)
    pins=strict_json(pinpath.read_bytes());bound=bindings(root,code,revision,pins)
    return pins,bound


def previous_failure(root):
    path=root/BASE/'infer_v1/report.json';checked(path,PREVIOUS_FAILURE);report=strict_json(path.read_bytes())
    require(report.get('stage')==STAGE and report.get('status')=='fail' and report.get('phase')=='preflight'
            and report.get('producer_revision')==PREVIOUS_REVISION and report.get('error_type')=='PermissionError'
            and all(type(report.get(k)) is int and report[k]==0 for k in ('native_calls_attempted','native_calls_returned','native_calls_completed'))
            and report.get('videos')==[] and all(report.get(k) is False for k in ('private_pickles_read','future_tracks_or_visibility_read','evaluation_performed','challenge_inputs_used')), 'Original zero-model preflight failure must remain FAIL')
    return checked(path,PREVIOUS_FAILURE)


def runtime_proof_path(root,revision):
    return root/'results'/('robotap-boots-runtime-proof-'+revision)/'report.json'


def verify_runtime_proof(root,revision,pins):
    path=source.canonical(runtime_proof_path(root,revision));meta=path.lstat();parent=path.parent.lstat()
    require(stat.S_ISDIR(parent.st_mode) and parent.st_uid==0 and parent.st_mode&0o777==0o700
            and stat.S_ISREG(meta.st_mode) and meta.st_uid==0 and meta.st_mode&0o777==0o444 and meta.st_nlink==1
            and {p.name for p in path.parent.iterdir()}=={'report.json'}, 'Exclusive root-owned readonly CPU receipt mirror required')
    checked(root/pins['runtime']['report_path'],pins['runtime']);checked(path,pins['runtime'])
    return path


def publish_runtime_proof(root,code,revision,pins,before):
    require(os.geteuid()==0,'Only host root may publish authorized metadata mirror')
    original=root/pins['runtime']['report_path'];checked(original,pins['runtime']);raw=original.read_bytes();checked(original,pins['runtime'])
    path=source.canonical(runtime_proof_path(root,revision));path.parent.mkdir(mode=0o700)
    source.save_bytes(path,raw,0o444);verify_runtime_proof(root,revision,pins)
    require(bindings(root,code,revision,pins)==before,'Originals changed while publishing byte-identical proof')
    return path


def host_mode(mode):
    root,code,revision=Path(os.environ['WR_ROOT']),Path(os.environ['WR_CODE']),os.environ['WR_CODE_REVISION']
    pins,bound=preflight(root,code,revision)
    if mode=='--publish-runtime-proof':
        source.azure_vm02_identity(source.read_protocol(code/PROTOCOL))
        print(publish_runtime_proof(root,code,revision,pins,bound))
    elif mode=='--mounts':
        mirror=verify_runtime_proof(root,revision,pins)
        paths=[code,code.parent/'revision',code.parent/'source-sha256',root/PUBLIC/'report.json',root/BASE/'infer_v1/report.json']
        paths += [root/PUBLIC/'inputs'/name for name in sorted(pins['public']['files'])]
        protocol=source.read_protocol(code/PROTOCOL)
        paths += [root/BASE/'assets/tapnet_source'/name for name in protocol['source']['files']]
        paths += [root/BASE/'assets'/protocol['checkpoint']['file']]
        for path in paths:print(str(source.canonical(path))+'\t'+str(source.canonical(path)))
        print(str(mirror)+'\t'+str(source.canonical(root/pins['runtime']['report_path'])))
    else:
        protocol=source.read_protocol(code/PROTOCOL);source.azure_vm02_identity(protocol)
        if mode=='--verify':verify_runtime_proof(root,revision,pins)
        print(hashlib.sha256(json.dumps(bound,sort_keys=True).encode()).hexdigest())


def main(argv=None):
    argparse.ArgumentParser(description=__doc__, allow_abbrev=False).parse_args(argv)
    root, code, revision = Path(os.environ["WR_ROOT"]), Path(os.environ["WR_CODE"]), os.environ["WR_CODE_REVISION"]
    require(os.uname().sysname == "Linux" and os.environ.get("WR_AZURE_VM02_VERIFIED") == "1" and os.environ.get("WR_IMAGE_ID") == IMAGE and os.environ.get("WR_RUNTIME_PROOF_MIRROR")=="1"
            and {p.name for p in Path('/sys/class/net').iterdir()} == {"lo"}, "Offline host-verified Azure VM02 CUDA required")
    output = source.canonical(root / OUT)
    require(output.is_dir() and {p.name for p in output.iterdir()} == {".container.cid"}, "Fresh reserved inference namespace required")
    cid=output/'.container.cid'
    require(stat.S_ISREG(cid.lstat().st_mode) and re.fullmatch(b'[0-9a-f]{64}\n?',cid.read_bytes()) and os.geteuid()==1000,'Owned container CID and UID1000 required')
    start = time.monotonic(); report = dict(stage=STAGE, status="fail", phase="preflight", producer_revision=revision,
        image_id=IMAGE, runtime_proof_byte_identical_authorized_mirror=True, previous_inference_failure_preserved=False, budget_seconds=BUDGET, native_calls_attempted=0, native_calls_returned=0, native_calls_completed=0, videos=[], network="none",
        future_tracks_or_visibility_read=False, private_pickles_read=False, challenge_inputs_used=False, evaluation_performed=False,
        oracle_initial_queries=True, private_truth_read=False, benchmark_verified=False, quality_verified=False, static_control_is_valid_motion_prediction=False, training_overlap_verified=False, full_hoi_verified=False, adoption_performed=False)
    error = None; before = None
    def expired(*_): raise TimeoutError("Fixed900s whole inference budget exceeded")
    previous = {s: signal.signal(s, expired) for s in (signal.SIGALRM, signal.SIGTERM)}; signal.alarm(BUDGET)
    try:
        pins,before = preflight(root,code,revision)
        rows = public_records(root / PUBLIC / "inputs"); report.update(phase="model_load", input_bindings=before, inference_pins=source.identity(code/PINS),
            previous_inference_failure=before["previous_inference_failure"], previous_inference_failure_preserved=True,
            script_sha256=before["source_files"]["infra/robotap_boots_infer.py"]["sha256"], native_source_revision="730cda1c730877cfedbe01bf87fb1cadb78a565d", demo_evidence=DEMO,
            inference_config=dict(pyramid_level=1, resolution=[256,256], query_chunk_size=32, is_training=False, compute_dtype="float32", AMP=False,
                                  resize="native utils.bilinear align_corners=False", normalize="RGB float32 /255*2-1", frame_offload=False, visibility="(1-sigmoid(occlusion))*(1-sigmoid(expected_dist))>0.5"))
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        import numpy as np
        random.seed(0); np.random.seed(0)
        import torch
        require(torch.cuda.is_available(), "CUDA required; no fallback")
        torch.manual_seed(0); torch.cuda.manual_seed_all(0); torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.use_deterministic_algorithms(False, warn_only=False);torch.backends.cudnn.benchmark=False
        require(torch.__version__=='2.5.1+cu124' and np.__version__=='1.26.3','Exact independently tested numerical runtime required')
        report['numerical_runtime']=dict(torch=torch.__version__,numpy=np.__version__,deterministic_algorithms=False,warn_only=False,TF32=False,seed=0,device='cuda',exact_replay_verified=False)
        tapir, utils = native_modules(root / BASE / "assets/tapnet_source")
        model = load_model(torch, tapir, root / BASE / "assets/bootstapir_checkpoint_v2.pt")
        (output / "predictions").mkdir(mode=0o755)
        for index, row in enumerate(rows):
            report['phase']=f"video_{index:03d}"; torch.cuda.synchronize(); started=time.monotonic()
            with np.load(root / PUBLIC / "inputs" / row["file"], allow_pickle=False) as file: data={k:file[k] for k in file.files}
            validate_video(data,row)
            arrays=native_prediction(torch,utils,model,data,report);torch.cuda.synchronize();report["native_calls_completed"]+=1
            validate_prediction(arrays,row,data['query_points'],data['point_indices'])
            target=output/"predictions"/row["file"]
            with source.private_writer(target) as stream:np.savez(stream,**arrays)
            target.chmod(0o444)
            with np.load(target,allow_pickle=False) as saved:
                require(set(saved.files)==set(arrays) and all(np.array_equal(saved[k],value) for k,value in arrays.items()),"Saved native arrays changed")
            report['videos'].append(dict(file=row['file'],frames=row['frames'],query_count=row['query_count'],point_indices=row['point_indices'],
                elapsed_seconds=time.monotonic()-started,output=source.identity(target),input=before['public_files'][row['file']]))
            del arrays,data;gc.collect()
        require(report['native_calls_attempted']==report['native_calls_returned']==report['native_calls_completed']==3,"All three native calls required")
        report.update(phase="complete",actual_native_inference=True,all_original_frames_retained=True,all_original_selected_queries_retained=True)
    except Exception as caught:error=caught;report.update(error_type=type(caught).__name__,error="Native inference failed closed; partial evidence retained")
    finally:
        try:
            require(before is not None and bindings(root,code,revision,pins)==before,"Input/source/checkpoint/CPU proof changed")
            if (output/'predictions').exists():require({p.name for p in (output/'predictions').iterdir()}=={r['file'] for r in report['videos']}, 'Unexpected/incomplete prediction artifacts')
            for row in report['videos']:require(source.identity(output/'predictions'/row['file'])==row['output'],"Frozen prediction changed")
            report['all_input_assets_sources_rechecked']=True
        except Exception as caught:
            if error is None:error=caught
            report['final_integrity_failed']=True
        report.update(status="pass" if error is None else "fail",elapsed_seconds=time.monotonic()-start)
        source.save_bytes(output/'report.json',(json.dumps(report,indent=2,allow_nan=False)+'\n').encode(),0o444)
        signal.alarm(0)
        for s,handler in previous.items():signal.signal(s,handler)
    if error is not None:raise RuntimeError("Native Boots inference failed; inspect sealed receipt") from None


if __name__ == "__main__":
    import sys
    if sys.argv[1:] in (['--preflight'],['--verify'],['--mounts'],['--publish-runtime-proof']):host_mode(sys.argv[1])
    else:main()
