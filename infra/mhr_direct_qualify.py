"""One standalone MHR component control; no SAM, tracker or accuracy claim."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import re
import signal
import stat
import subprocess
import sys
import time

ENTRY = 'run_mhr_direct_qualify'
IMAGE = 'sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7'
BUDGET = 120
HELPERS = ('infra/mhr_direct_qualify.py', 'infra/run_mhr_direct_qualify.sh',
    'infra/mhr_direct_bridge.py', 'infra/mediapipe_cpu_runtime_verify.py',
    'configs/mhr_official_release_selected_qualification_pins.json',
    'configs/mhr_official_release_protocol_v1.json')


def runtime():
    import mediapipe_cpu_runtime_verify as rt
    rt.require(Path(rt.__file__).resolve() == Path(__file__).parent/'mediapipe_cpu_runtime_verify.py',
        'Actual immutable runtime helper required')
    return rt


def bridge_module():
    import mhr_direct_bridge as bridge
    runtime().require(Path(bridge.__file__).resolve() == Path(__file__).parent/'mhr_direct_bridge.py',
        'Actual immutable direct bridge required')
    return bridge


def control_mode():
    mode=os.environ.get('WR_MHR_DIRECT_CONTROL','named')
    runtime().require(mode in ('named','gradients'),'Explicit direct component mode required')
    return mode


def component_module():
    if control_mode()=='named':return bridge_module()
    import mhr_direct_gradients as gradients
    runtime().require(Path(gradients.__file__).resolve()==Path(__file__).parent/'mhr_direct_gradients.py',
        'Actual immutable gradient component required')
    return gradients


def container_name(revision):
    return 'wr-mhr-direct-'+('gradients-'if control_mode()=='gradients'else'')+revision


def namespaces(root, revision):
    stem='mhr-direct-gradients-'if control_mode()=='gradients'else'mhr-direct-'
    return root/'results'/(stem+'qualify-'+revision), root/'results'/(stem+'control-'+revision)


def source_helpers():
    return (*HELPERS,'infra/mhr_direct_gradients.py')if control_mode()=='gradients'else HELPERS



def host_proof(root, code, revision):
    rt=runtime();bridge=bridge_module()
    rt.require(Path(__file__).resolve()==code/HELPERS[0], 'Actual current entrypoint required')
    before=rt.source(root,code,revision,ENTRY,source_helpers())
    release=bridge.authenticate_release(root,code,rt)  # Must precede any JIT load.
    bridge.recheck_release(release,root,rt)
    rt.require(before==rt.source(root,code,revision,ENTRY,source_helpers()),'Current complete source changed')
    return dict(source=before, release_source=release['source'],
        release_report_identity=release['release_report_identity'],
        frozen={str(p):pin for p,pin in release['frozen'].items()},
        model_path=str(release['model_path']), original_code=str(release['original_code']))


def mounts(proof, code):
    # Exactly two source-only snapshots, selected public notices/receipt, model leaf.
    return (code.parent, Path(proof['original_code']).parent,
        Path(proof['model_path']), Path(next(p for p in proof['frozen']
            if p.endswith('/mhr-official-release-license-v2/report.json'))).parent)


def execute_control(proof, torch, bridge, check, *, event=lambda _:None):
    """One authenticated load and explicit component; no dtype conversion/retry."""
    check()
    with torch.jit.optimized_execution(False):
        event('model_load_attempts')
        model=torch.jit.load(proof['model_path'],map_location='cuda').eval()
        event('model_load_returns')
        if any(v.is_floating_point() and v.dtype!=torch.float32 for _,v in
                (*model.named_parameters(),*model.named_buffers())):
            raise ValueError('Original model must already be float32; no casting permitted')
        check()
        event('control_attempts');result=bridge.run_control(model,torch,check=check)
        event('control_returns');return result


def write_receipt(path, record, check, *, maximum=1<<20, seal_parent=False):
    """Exclusive same-inode failure demotion if sealing exhausts native budget."""
    rt=runtime();fd=os.open(path,os.O_RDWR|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o444)
    with os.fdopen(fd,'w+b')as stream:
        os.fchmod(stream.fileno(),0o444)
        def put():
            raw=(json.dumps(record,sort_keys=True,allow_nan=False)+'\n').encode()
            rt.require(len(raw)<=maximum,'Bounded scalar receipt required')
            a=path.lstat();b=os.fstat(stream.fileno())
            rt.require((a.st_dev,a.st_ino,a.st_nlink)==(b.st_dev,b.st_ino,1)and b.st_nlink==1,'Owned receipt changed')
            stream.seek(0);stream.truncate();stream.write(raw);stream.flush();os.fsync(stream.fileno())
        put()
        try:
            if seal_parent:
                path.parent.chmod(0o555)
                directory=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
                try:os.fsync(directory)
                finally:os.close(directory)
            check()
        except BaseException as error:
            record.update(status='fail',phase='receipt',failure_type=type(error).__name__)
            put();raise


def native(root, code, revision, image):
    rt=runtime();mode=control_mode();out,control=namespaces(root,revision);started=time.monotonic()
    check=lambda:rt.require(time.monotonic()-started<=BUDGET,'Inclusive direct control deadline exhausted')
    rt.require(sys.platform=='linux' and os.geteuid()==1000 and image==IMAGE
        and os.environ.get('WR_MHR_DIRECT_LEASE')=='fd9'
        and {p.name for p in Path('/sys/class/net').iterdir()}=={'lo'},'Exact offline leased native runtime required')
    rt.canonical(out);rt.require(out.is_dir() and not tuple(out.iterdir())
        and out.stat().st_uid==1000 and stat.S_IMODE(out.stat().st_mode)==0o755,'Owned empty native output required')
    record=dict(stage='mhr_direct_gradients_native_v1'if mode=='gradients'else'mhr_direct_qualify_native_v1',status='fail',phase='authentication',producer_revision=revision,
        image_id=image,budget_seconds=BUDGET,control_attempts=0,control_returns=0,model_load_attempts=0,model_load_returns=0,
        source_release_model_rehashed_after=False,models_loaded=False,SAM_checkpoint_used=False,
        dataset_read=False,private_values_read=False,gradients_verified=False,calibration_verified=False,
        native_end_to_end_qualified=False,repeatability_verified=False,reconstruction_accuracy_verified=False,adoption=False)
    def expired(*_):raise TimeoutError('Direct standalone control exceeded120s')
    handlers={s:signal.signal(s,expired)for s in(signal.SIGALRM,signal.SIGTERM,signal.SIGINT)}
    signal.setitimer(signal.ITIMER_REAL,BUDGET);failure=None;proof=None
    try:
        proof=host_proof(root,code,revision);check()
        rt.require(proof==rt.strict((control/'proof.json').read_bytes()),'Host/native complete provenance differs')
        record['source_binding']=proof['source'];record['release_report_identity']=proof['release_report_identity']
        rt.require('torch'not in sys.modules and os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8',
            'Deterministic workspace must precede Torch import')
        import numpy as np
        import torch
        rt.require(torch.cuda.is_available(),'CUDA required; no CPU fallback')
        random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
        torch.set_num_threads(4);torch.use_deterministic_algorithms(True,warn_only=False)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        record.update(phase='direct_control',torch_version=torch.__version__,
            numpy_version=np.__version__,cuda_version=torch.version.cuda,jit_optimized_execution=False,
            execution_dtype='float32',deterministic_algorithms=True,TF32=False,seed=0)
        def event(key):
            record[key]+=1
            if key=='model_load_returns':record['models_loaded']=True
        result=execute_control(proof,torch,component_module(),check,event=event)
        record.update(component=result,phase='posthash',gradients_verified=mode=='gradients')
        rt.require(host_proof(root,code,revision)==proof,'Complete source/release/model changed')
        check();record.update(status='pass',phase='complete',source_release_model_rehashed_after=True)
    except BaseException as error:
        failure=error;record.update(status='fail',failure_type=type(error).__name__,failure=str(error)[:400])
    finally:
        if proof is not None and not record['source_release_model_rehashed_after'] and time.monotonic()-started<BUDGET:
            try:
                rt.require(host_proof(root,code,revision)==proof,'Failed control source/release/model changed')
                check();record['source_release_model_rehashed_after']=True
            except BaseException as error:
                failure=failure or error;record.update(status='fail',posthash_failure_type=type(error).__name__)
        signal.setitimer(signal.ITIMER_REAL,5)  # Failure-only reporting, never extra compute.
        record['elapsed_seconds']=time.monotonic()-started
        try:
            write_receipt(out/'native.json',record,check if failure is None else lambda:None,seal_parent=True)
        except BaseException as error:
            failure=failure or error
        finally:
            signal.setitimer(signal.ITIMER_REAL,0)
            for sig,handler in handlers.items():signal.signal(sig,handler)
    if failure is not None:raise failure


def inspect_container(cid, name, revision, *, run=subprocess.run):
    """Return found/absent; daemon/timeout/ambiguous stream errors are not absence."""
    if not re.fullmatch('[0-9a-f]{64}',cid):raise ValueError('Exact owned CID required')
    r=run(['docker','inspect',cid,'--format','{{.Id}} {{.Name}} {{.Image}} {{index .Config.Labels "world_reward.mhr_direct.owner"}}'],
        capture_output=True,timeout=5)
    if r.returncode==0:
        if r.stdout!=(f'{cid} /{name} {IMAGE} {revision}\n').encode() or r.stderr:raise ValueError('Foreign/ambiguous container')
        return True
    errors=tuple((p+cid).encode()for p in('Error: No such object: ','error: no such object: ',
        'Error: No such container: ','Error response from daemon: No such container: '))
    if r.returncode==1 and r.stdout in(b'',b'\n',b'[]\n') and r.stderr.strip(b'\r\n')in errors:return False
    raise ValueError('Docker error is not owned container absence')


def cleanup(control, name, revision):
    rt=runtime();path=control/'.container.cid';before=rt.identity(path,65,readonly=False)
    rt.require(path.stat().st_uid==os.geteuid(),'Owned CID required')
    raw=path.read_bytes();rt.require(re.fullmatch(b'[0-9a-f]{64}\n?',raw)
        and rt.identity(path,65,readonly=False)==before,'Stable exact CID required');cid=raw.decode().rstrip('\n')
    if inspect_container(cid,name,revision):
        result=subprocess.run(['docker','rm','-f',cid],capture_output=True,timeout=10)
        rt.require(result.returncode==0,'Owned removal failed')
    rt.require(not inspect_container(cid,name,revision),'Owned container remains')


def seal(root, code, revision, status, cleanup_verified):
    rt=runtime();mode=control_mode();out,control=namespaces(root,revision);failure=None;proof=None;rehashed=False;absence=False
    try:
        before=rt.strict((control/'proof.json').read_bytes());proof=host_proof(root,code,revision)
        rt.require(proof==before,'Original complete source/release/model changed')
        rehashed=True
        rt.require(cleanup_verified,'Owned cleanup not verified')
        cleanup(control,container_name(revision),revision)  # Independent final CID absence.
        absence=True
        record=rt.pinned(out/'native.json',rt.identity(out/'native.json',1<<20),1<<20)
        wanted=dict(stage='mhr_direct_gradients_native_v1'if mode=='gradients'else'mhr_direct_qualify_native_v1',status='pass',phase='complete',producer_revision=revision,
            image_id=IMAGE,budget_seconds=120,control_attempts=1,control_returns=1,
            model_load_attempts=1,model_load_returns=1,
            source_release_model_rehashed_after=True,models_loaded=True,SAM_checkpoint_used=False,dataset_read=False,
            private_values_read=False,gradients_verified=mode=='gradients',calibration_verified=False,
            native_end_to_end_qualified=False,repeatability_verified=False,reconstruction_accuracy_verified=False,adoption=False,
            jit_optimized_execution=False,execution_dtype='float32',deterministic_algorithms=True,TF32=False,seed=0)
        rt.require(status==0 and all(type(record.get(k))is type(v)and record[k]==v for k,v in wanted.items())
            and 0<record['elapsed_seconds']<=BUDGET and record['source_binding']==proof['source']
            and record['release_report_identity']==proof['release_report_identity'],
            'Complete actual direct native control required')
        component=record['component'];expected=dict(stage='mhr_direct_named_component_v1',status='pass',
            native_calls=1,decoded_frames=9,metadata_rehashed_after=True,bounds_used_literally=True,bounds_relaxed=False,
            parameter_limit_hard_enforcement_claimed=False,supplied_live_model_load_authenticated_by_this_function=False,
            SAM_checkpoint_used=False,keypoint70_bridge_qualified=False,render_calls=0,tracker_calls=0,optimizer_calls=0,
            contact_verified=False,RGB_only_reconstruction_verified=False,quality_verified=False,adoption=False)
        if mode=='gradients':
            expected=dict(stage='mhr_direct_directional_autograd_component_v1',status='pass',
                native_forward_calls=2,decoded_frames=17,vjp_calls=1,forward_dtype='float32',
                scalar_reduction_dtype='float64',apply_correctives=True,native_inputs_unchanged=True,
                constant_identity_expression_scale=True,weights_version_unchanged=True,weight_gradients_none=True,
                metadata_rehashed_after=True,root_gradients_qualified=False,full_jacobian_qualified=False,
                keypoint70_bridge_qualified=False,quaternion_gradients_qualified=False,render_calls=0,
                tracker_calls=0,optimizer_calls=0,SAM_checkpoint_used=False,contact_verified=False,
                quality_verified=False,backend_adoption=False)
            rt.require(component['parameters']==list(component_module().PARAMETERS)
                and component['centre']==2**-5 and component['finite_difference_steps']==[2**-8,2**-10]
                and component['directional_atol_m_per_unit']==1e-5 and component['directional_rtol']==1e-2
                and len(component['comparisons'])==8 and len(component['material_support_motion'])==4,
                'Complete predeclared directional component required')
        else:
            rt.require(component['frame_index']==list(range(9))and len(component['controls'])==4
                and len(component['named_rows'])==9,'Complete original named component required')
        rt.require(all(type(component.get(k))is type(v)and component[k]==v for k,v in expected.items()),
            'Complete unchanged explicit component required')
        rt.require({p.name for p in out.iterdir()}=={'native.json'},'No generated geometry/predictions retained')
    except BaseException as error:failure=error
    host=dict(stage='mhr_direct_gradients_host_v1'if mode=='gradients'else'mhr_direct_qualify_host_v1',status='fail'if failure else'pass',producer_revision=revision,
        original_exit_status=status,owned_container_absence_verified=absence,
        source_release_model_rehashed_after=rehashed,adoption=False,
        native_report_identity=rt.identity(out/'native.json',1<<20)if(out/'native.json').exists()else None)
    if failure:host.update(failure_type=type(failure).__name__,failure=str(failure)[:400])
    rt.require({p.name for p in control.iterdir()}=={'proof.json','.container.cid'},'Exact pre-seal control inventory required')
    for p in control.iterdir():
        a=p.lstat();rt.require(a.st_uid==os.geteuid()and stat.S_ISREG(a.st_mode)
            and a.st_nlink==1 and not p.is_symlink(),'Only owned control metadata permitted');p.chmod(0o444)
    def sealed():
        rt.require(control.stat().st_mode&0o777==0o555 and {p.name for p in control.iterdir()}
            =={'proof.json','.container.cid','report.json'},'Complete sealed control inventory required')
        for p in control.iterdir():
            a=p.lstat();rt.require(a.st_uid==os.geteuid()and stat.S_ISREG(a.st_mode)and a.st_nlink==1
                and stat.S_IMODE(a.st_mode)==0o444 and not p.is_symlink(),'Exact sealed control metadata required')
    write_receipt(control/'report.json',host,sealed,seal_parent=True)
    if failure:raise failure


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--gradients',action='store_true')
    args=parser.parse_args()
    runtime().require(args.gradients==(control_mode()=='gradients'),'Native/host explicit component mode differs')
    root=Path(os.environ['WR_ROOT']);code=Path(os.environ['WR_CODE']);revision=os.environ['WR_CODE_REVISION']
    runtime().require(root==Path('/srv/scenesmith/world-reward'),'Exact Azure root required')
    native(root,code,revision,os.environ['WR_IMAGE_ID'])


if __name__=='__main__':main()
