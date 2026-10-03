"""Inspect frozen MHR limit getter/source metadata; never decode or fit a body.

A completed inspection does not establish the physiological meaning of [0,0]
entries. Source/state evidence is preserved for a separate explicit conclusion.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import signal
import sys
import time

import numpy as np

OUT="results/mhr-limits-semantics-v1"
STAGE="frozen_mhr_limit_getter_and_state_semantics_inspection"
BUDGET=60
IMAGE="sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7"
MODEL_SHA="352e271a6c42729c68554ceaea0c955e866970160c31e35506d782dc0f7377bc"
MODEL_BYTES=696110248
PREVIOUS="results/keypoint-rgb-bounds-audit-v2/report.json"
PREVIOUS_SHA="8e5e725f244a5a122fa5a154ce0ce49334619494d1f5df9a45c58e6b7bdbd21b"
PREVIOUS_REV="3cdef2da0c2e935080410562c826752a1833fcef"
PREVIOUS_SCRIPT="8fce37c4d9aed75328e302f54c7f092e532ab2253d47c7db4c50eecba7130003"
CODE_LIMIT=100000


def regular(path,digest=None,size=None,*,immutable=True):
    path=Path(path)
    if path.resolve()!=path.absolute() or not path.is_file() or (immutable and path.stat().st_mode & 0o222):
        raise ValueError("Canonical immutable regular input required")
    with path.open("rb")as stream:actual=hashlib.file_digest(stream,"sha256").hexdigest()
    row=dict(path=str(path),sha256=actual,bytes=path.stat().st_size)
    if (digest is not None and actual!=digest)or(size is not None and row["bytes"]!=size):raise ValueError("Pinned input differs")
    return row


def readonly_model_mount(text,path):
    matches=[]
    for line in text.splitlines():
        fields=line.split();sep=fields.index("-")if"-"in fields else -1
        if sep<6 or len(fields)<sep+4:raise ValueError("Malformed kernel mountinfo")
        mount=re.sub(r"\\(040|011|012|134)",lambda m:chr(int(m[1],8)),fields[4])
        if re.search(r"\\[0-9]",mount):raise ValueError("Unsupported mount path escape")
        if mount==str(path):
            options=fields[5].split(",")
            if"ro"not in options or"rw"in options:raise ValueError("Exact reference file mount must be read-only")
            matches.append(dict(mount_point=mount,mount_options=options,super_options=fields[sep+3].split(",")))
    if len(matches)!=1:raise ValueError("One exact read-only reference file mount required")
    return matches[0]


def fields(row,expected):
    if not isinstance(row,dict)or any(type(row.get(k))is not type(v)or row[k]!=v for k,v in expected.items()):
        raise ValueError("Exact previous diagnostic fields required")


def previous(root):
    path=Path(root)/PREVIOUS;identity=regular(path,PREVIOUS_SHA);row=json.loads(path.read_text())
    fields(row,dict(stage="public_keypoint_rgb_native_bounds_readonly_audit_v2",status="pass",phase="complete",
        producer_revision=PREVIOUS_REV,script_sha256=PREVIOUS_SCRIPT,image_id=IMAGE,network="none",device="cpu",
        model_sha256=MODEL_SHA,model_forward_calls=0,optimizer_updates=0,frames=15,all_cases_retained=True,
        original_inputs_rehashed=True,raw_root_mapping_mismatch_frames=0,shared_root_mapping_mismatch_frames=0,
        private_truth_read=False,ground_truth_used=False,challenge_inputs_used=False,hand_labeled_test=False,oracle_modes=[],
        bounds_relaxed=False,predictions_modified=False,original_failure_rewritten=False,adoption_authorized=False))
    records=row.get("records")
    if not isinstance(records,list)or len(records)!=15:raise ValueError("All15 original diagnostic cases required")
    for i,record in enumerate(records):
        fields(record,dict(clip_index=i//5,frame_index=i%5))
        for mode in("raw","shared"):
            fields(record.get(mode),dict(root_Euler_byte_equal=True))
    return identity


def safe_values(value):
    if isinstance(value,list):return[safe_values(v)for v in value]
    if isinstance(value,float)and not np.isfinite(value):
        if np.isnan(value):return"NaN"
        return"+infinity"if value>0 else"-infinity"
    return value


def array_metadata(value,*,values=True):
    if np.ma.isMaskedArray(value):raise ValueError("Hidden tensor validity forbidden")
    array=np.asarray(value)
    if array.dtype.kind not in"biuf"or array.dtype.hasobject:raise ValueError("Numeric tensor metadata required")
    row=dict(dtype=str(array.dtype),shape=list(array.shape),entries=int(array.size),
        sha256=hashlib.sha256(array.tobytes(order="C")).hexdigest(),finite=bool(np.isfinite(array).all()))
    if values and array.size<=2048:row["values"]=safe_values(array.tolist())
    return row


def selected_name(name):
    return isinstance(name,str)and("limit"in name.lower()or"minmax"in name.lower())


def getter_source(model):
    methods=list(model._c._method_names())
    if ("get_parameter_limits"not in methods or len(methods)>512
            or any(not isinstance(name,str)or not name for name in methods)):
        raise ValueError("Actual scripted limit getter required")
    method=model._c._get_method("get_parameter_limits");code=method.code
    if not isinstance(code,str)or not code or len(code.encode())>CODE_LIMIT:raise ValueError("Bounded actual getter source required")
    row=dict(method_names=methods,getter_code=code,getter_code_sha256=hashlib.sha256(code.encode()).hexdigest(),getter_code_bytes=len(code.encode()))
    graph=str(method.graph)
    if len(code.encode())+len(graph.encode())<=CODE_LIMIT:
        row.update(getter_graph=graph,getter_graph_sha256=hashlib.sha256(graph.encode()).hexdigest())
    return row


def snapshot(model):
    limits=model.get_parameter_limits().detach().cpu().numpy();names=list(model.get_parameter_names())
    if (limits.shape!=(249,2)or limits.dtype.kind!="f"or np.isnan(limits).any()or np.any(limits[:,0]>limits[:,1])
            or len(names)!=249 or len(set(names))!=249 or any(not isinstance(n,str)or not n for n in names)):
        raise ValueError("Exact native249 names/dense bounds metadata required")
    buffers={name:array_metadata(value.detach().cpu().numpy())for name,value in model.named_buffers()if selected_name(name)}
    state={name:array_metadata(value.detach().cpu().numpy())for name,value in model.state_dict().items()if selected_name(name)}
    modules=[]
    for name,module in model.named_modules():
        if selected_name(name):
            methods=list(module._c._method_names())if hasattr(module,"_c")else[]
            modules.append(dict(path=name,scripted_methods=methods))
    return dict(**getter_source(model),parameter_names=names,dense_parameter_limits=array_metadata(limits),
        zero_zero_columns=np.flatnonzero((limits[:,0]==0)&(limits[:,1]==0)).tolist(),
        selected_named_buffers=buffers,selected_state_dict=state,selected_named_modules=modules)


def perform(root,report,persist):
    prior=previous(root);path=root/"weights/mhr/mhr_model.pt"
    mount=readonly_model_mount(Path("/proc/self/mountinfo").read_text(),path)
    model_identity=regular(path,MODEL_SHA,MODEL_BYTES,immutable=False);source=regular(Path(__file__))
    report.update(previous_diagnostic=prior,model_file=model_identity,model_mount=mount,model_host_write_bits=int(path.stat().st_mode&0o222),
        model_exact_mount_readonly_verified=True,phase="CPU_scripted_metadata_load");persist()
    if"torch"in sys.modules:raise ValueError("Fresh standalone CPU metadata runtime required")
    import torch
    torch.set_num_threads(4);torch.manual_seed(0)
    model=torch.jit.load(str(path),map_location="cpu").float().eval()
    evidence=snapshot(model);report.update(torch_version=str(torch.__version__),evidence=evidence,phase="getter_source_and_state_inspected");persist()
    if snapshot(model)!=evidence:raise ValueError("Getter/source/state metadata changed during inspection")
    if (previous(root)!=prior or regular(Path(__file__))!=source or readonly_model_mount(Path("/proc/self/mountinfo").read_text(),path)!=mount
            or regular(path,MODEL_SHA,MODEL_BYTES,immutable=False)!=model_identity):raise ValueError("Frozen source/input/model evidence changed")
    report.update(status="pass",phase="complete",source_and_inputs_rehashed=True,getter_source_verified=True,
        model_forward_calls=0,metadata_inspection_completed=True)


def main(argv=None):
    argparse.ArgumentParser(description=__doc__,allow_abbrev=False).parse_args(argv)
    root=Path(os.environ.get("WR_ROOT",""));out=root/OUT;revision=os.environ.get("WR_CODE_REVISION","")
    if (platform.system()!="Linux"or root!=Path("/srv/scenesmith/world-reward")or out.resolve()!=out.absolute()or os.geteuid()!=1000
            or {p.name for p in Path("/sys/class/net").iterdir()}!={"lo"}or os.environ.get("WR_IMAGE_ID")!=IMAGE
            or os.environ.get("CUDA_VISIBLE_DEVICES")!=""or not re.fullmatch("[0-9a-f]{40}",revision)or not out.is_dir()or any(out.iterdir())):
        raise ValueError("Fresh canonical reserved offline CPU source inspection required")
    report=dict(stage=STAGE,status="fail",phase="frozen_input_integrity",producer_revision=revision,image_id=IMAGE,device="cpu",network="none",
        budget_seconds=BUDGET,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),model_forward_calls=0,optimizer_updates=0,
        private_truth_read=False,ground_truth_used=False,challenge_inputs_used=False,hand_labeled_test=False,oracle_modes=[],
        predictions_modified=False,bounds_relaxed=False,accuracy_verified=False,adoption_authorized=False,physiological_validity_verified=False,
        zero_zero_semantics_conclusion_verified=False,metadata_inspection_completed=False,source_and_inputs_rehashed=False)
    started=time.perf_counter();path=out/"report.json"
    with path.open("x")as stream:
        def persist():
            report["elapsed_seconds"]=time.perf_counter()-started;stream.seek(0);json.dump(report,stream,allow_nan=False)
            stream.write("\n");stream.truncate();stream.flush();os.fsync(stream.fileno())
        def expired(*_):raise TimeoutError("Whole limit source inspection exceeded60s")
        alarm=signal.signal(signal.SIGALRM,expired);term=signal.signal(signal.SIGTERM,expired);signal.alarm(BUDGET)
        try:persist();perform(root,report,persist)
        except Exception as error:report.update(error_type=type(error).__name__,error=str(error));raise
        finally:
            signal.alarm(0);signal.signal(signal.SIGALRM,alarm);signal.signal(signal.SIGTERM,term);persist();path.chmod(0o444)


if __name__=="__main__":main()
