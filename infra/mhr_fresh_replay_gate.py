"""Fresh-model replay of the frozen failed216-row fixture, not hand accuracy.

Each optimization/correctives condition loads a new CUDA TorchScript model,
in a separately started Python process (not merely a new cached JIT module),
then executes exactly neutral1, perturb216, replay216. There are no warmups.
Only both unoptimized conditions being raw-bit exact can pass. Optimized
execution is descriptive; this contrast does not identify a causal kernel or
graph pass, nor certify all shapes/batch sizes or real-image inference.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import re
import secrets
import subprocess
import sys
import time

import numpy as np

import mhr_determinism_gate as diagnosis
from world_reward.data import sha256

STAGE = 'reference_mhr_fresh_model_jit_replay_diagnosis'
TOTAL_SECONDS, MAX_CALLS, SEED = 60, 12, 0
CONDITIONS = ((True, False), (True, True), (False, False), (False, True))


def outcome(conditions):
    """Unoptimized both-correctives raw-bit equality only; no numeric tolerance."""
    if len(conditions) != 4 or [(c.get('optimized_execution'),c.get('apply_correctives')) for c in conditions] != list(CONDITIONS):
        return False
    return all(c.get('status') == 'complete' and c.get('fresh_model_loaded') is True
        and c.get('completed_calls') == 3 and c.get('replay',{}).get('bitexact') is True for c in conditions[2:])


def run_conditions(torch, model_path, report, persist, forward_call=None, condition_indices=range(4)):
    """Fixed3call condition; production passes one index per fresh subprocess."""
    inputs = diagnosis.own_inputs()
    report['input_sha256'] = [hashlib.sha256(v.tobytes()).hexdigest() for v in inputs]
    source = report.get('scripted_source_evidence')
    def actual_forward(model, tensors, correctives):
        with torch.inference_mode(): values = model(*tensors,correctives)
        torch.cuda.synchronize()
        if (not isinstance(values,tuple) or len(values) != 2
                or tuple(values[0].shape) != (len(tensors[0]),18439,3)
                or tuple(values[1].shape) != (len(tensors[0]),127,8)):
            raise ValueError('Actual reference forward ABI differs')
        outputs = tuple(v.detach().cpu().numpy().copy() for v in values)
        if any(v.dtype != np.float32 or not np.isfinite(v).all() for v in outputs):
            raise ValueError('Actual reference output must be finite original float32')
        return outputs
    forward_call = actual_forward if forward_call is None else forward_call
    for index in condition_indices:
        optimized,correctives = CONDITIONS[index]
        condition = {'optimized_execution':optimized,'apply_correctives':correctives,'status':'running',
                     'fresh_model_loaded':False,'completed_calls':0,'worker_pid':os.getpid()}
        report['conditions'].append(condition); report['active_condition'] = len(report['conditions'])-1; persist()
        model = tensors = neutral = first = replay = None
        try:
            random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
            with torch.jit.optimized_execution(optimized):
                model = torch.jit.load(str(model_path),map_location='cuda').float().eval()
                condition['fresh_model_loaded'] = True; persist()
                if any(v.is_floating_point() and v.dtype != torch.float32 for _,v in (*model.named_parameters(),*model.named_buffers())):
                    raise ValueError('No original-model dtype mutation is allowed')
                evidence = diagnosis.script_sources(model)
                if source is None:
                    source = evidence; report['scripted_source_evidence'] = source
                    report['forward_schema'] = str(model._c._get_method('forward').schema)
                elif evidence != source: raise ValueError('Fresh model scripted source differs between conditions')
                if (len(model.get_joint_names()),len(model.get_parameter_names()),model.get_num_identity_blendshapes(),
                        model.get_num_face_expression_blendshapes()) != (127,249,45,72):
                    raise ValueError('Frozen external MHRDemo ABI differs')
                tensors = tuple(torch.as_tensor(v.copy(),device='cuda') for v in inputs)
                calls = (tuple(torch.zeros_like(t[:1]) for t in tensors),tensors,tensors)
                outputs = []
                for label,values in zip(('neutral','perturb','replay'),calls):
                    if report['forward_calls'] >= MAX_CALLS: raise RuntimeError('Frozen12call maximum exceeded')
                    report['forward_calls'] += 1; condition['active_call'] = label; persist()
                    outputs.append(forward_call(model,values,correctives));condition['completed_calls'] += 1;persist()
                neutral,first,replay = outputs
                condition['neutral_output_sha256'] = [hashlib.sha256(v.tobytes()).hexdigest() for v in neutral]
                condition['replay'] = diagnosis.bit_comparison(first,replay)
                if any(not np.array_equal(t.detach().cpu().numpy(),v) for t,v in zip(tensors,inputs)):
                    raise ValueError('Model mutated the frozen own input controls')
                condition['status'] = 'complete'
                if not condition['replay']['bitexact'] and report.get('first_replay_failure') is None:
                    report['first_replay_failure'] = {'condition_index':report['active_condition'],
                        'optimized_execution':optimized,'apply_correctives':correctives,'comparison':condition['replay']}
        except (RuntimeError,ValueError) as error:
            condition.update(status='fail',error_type=type(error).__name__,error=str(error)[:2000])
        finally:
            model = tensors = neutral = first = replay = None
            torch.cuda.empty_cache(); persist()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--condition',type=int,choices=range(4),help=argparse.SUPPRESS)
    args=parser.parse_args(argv)
    if platform.system() != 'Linux' or {p.name for p in Path('/sys/class/net').iterdir()} != {'lo'}:
        raise RuntimeError('Require remote Linux CUDA container with network none')
    root = Path(os.environ['WR_ROOT']);revision = os.environ.get('WR_CODE_REVISION','');image = os.environ.get('WR_IMAGE_ID','')
    if (not root.is_absolute() or not re.fullmatch(r'[0-9a-f]{40}',revision)
            or not re.fullmatch(r'sha256:[0-9a-f]{64}',image)):
        raise ValueError('Require absolute root and immutable image/source revision')
    output = root/'results/mhr-fresh-replay.json'
    if output.is_symlink() or output.parent.resolve() != output.parent.absolute():
        raise FileExistsError('Frozen replay report or indirect output path')
    if args.condition is not None:
        if not output.is_file(): raise ValueError('Require parent-created running fresh replay receipt')
        report=json.loads(output.read_text());nonce=os.environ.get('WR_FRESH_REPLAY_NONCE','')
        if (report.get('status')!='running' or report.get('stage')!=STAGE or not nonce
                or report.get('nonce_sha256')!=hashlib.sha256(nonce.encode()).hexdigest()
                or report.get('script_sha256')!=sha256(Path(__file__))
                or report.get('diagnostic_helper_sha256')!=sha256(Path(diagnosis.__file__))
                or report.get('code_revision')!=revision or report.get('pending_condition')!=args.condition
                or report.get('source_image_id')!=image
                or len(report.get('conditions',[]))!=args.condition):
            raise RuntimeError('Require authorized fresh child; frozen reports immutable')
        def persist_child():
            temporary=output.with_suffix('.partial');temporary.write_text(json.dumps(report,allow_nan=False)+'\n');temporary.replace(output)
        model_path=root/'weights/mhr/mhr_model.pt'
        if (model_path.resolve()!=model_path.absolute() or not model_path.is_file() or model_path.stat().st_size!=696110248
                or sha256(model_path)!=diagnosis.MODEL_SHA): raise ValueError('Pinned regular reference model mismatch')
        if 'torch' in sys.modules: raise RuntimeError('Require fresh child before Torch import')
        os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';random.seed(SEED);np.random.seed(SEED)
        import torch
        if not torch.cuda.is_available(): raise RuntimeError('CUDA required, no CPU/local fallback')
        torch.set_num_threads(4);torch.use_deterministic_algorithms(True,warn_only=False)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        report.update(torch=torch.__version__,CUDA=torch.version.cuda)
        run_conditions(torch,model_path,report,persist_child,condition_indices=(args.condition,))
        return
    with output.open('x') as handle:
        started = time.perf_counter()
        nonce=secrets.token_hex(32)
        report = {'stage':STAGE,'status':'running','phase':'integrity','script_sha256':sha256(Path(__file__)),
            'diagnostic_helper_sha256':sha256(Path(diagnosis.__file__)),'code_revision':revision,'source_image_id':image,
            'network':'none','challenge_inputs_used':False,'hand_labeled_test':False,'adoption_performed':False,
            'hand_accuracy_verified':False,'hypothesis_validation':False,'known_failed_fixture':True,
            'kernel_cause_verified':False,'graph_pass_cause_verified':False,'fresh_model_per_condition':True,
            'fresh_process_per_condition':True,'nonce_sha256':hashlib.sha256(nonce.encode()).hexdigest(),
            'warmup_calls':0,'call_order':['neutral1','perturb216','replay216'],'seed':SEED,
            'conditions':[],'forward_calls':0,'budgets':{'total_seconds':TOTAL_SECONDS,'forward_calls':MAX_CALLS,'CPU_threads':4},
            'optimized_true_descriptive_only':True,'raw_bit_exact_required':True,'first_replay_failure':None}
        def persist():
            report['elapsed_seconds'] = time.perf_counter()-started
            with output.open('w') as stream:
                stream.write(json.dumps(report,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
        try:
            persist();model_path=root/'weights/mhr/mhr_model.pt'
            if (not model_path.is_file() or model_path.stat().st_size != 696110248
                    or model_path.resolve() != model_path.absolute() or sha256(model_path) != diagnosis.MODEL_SHA):
                raise ValueError('Pinned regular reference model mismatch')
            report.update(model_sha256=diagnosis.MODEL_SHA,model_bytes=696110248,
                CUBLAS_WORKSPACE_CONFIG=':4096:8',TF32=False,execution_dtype='float32',strict_algorithms=True,phase='fresh_replay')
            for index in range(4):
                remaining=TOTAL_SECONDS-(time.perf_counter()-started)
                if remaining<=0: raise TimeoutError('Fresh-model diagnosis exceeded frozen60s budget')
                report['pending_condition']=index;persist()
                process=subprocess.run([sys.executable,str(Path(__file__)),'--condition',str(index)],check=False,
                    timeout=remaining,env=os.environ|{'WR_FRESH_REPLAY_NONCE':nonce,'CUBLAS_WORKSPACE_CONFIG':':4096:8'})
                report=json.loads(output.read_text());persist()
                if process.returncode: raise RuntimeError('Fresh subprocess failed before a complete condition')
                if len(report['conditions'])!=index+1: raise ValueError('Fresh worker did not emit exactly one condition')
            if len({c.get('worker_pid') for c in report['conditions']}) != 4:
                raise ValueError('Four distinct fresh worker processes were not observed')
            report['unoptimized_fresh_replay_bitexact']=outcome(report['conditions'])
            if not report['unoptimized_fresh_replay_bitexact']:
                raise ValueError('Both fresh unoptimized correctives conditions must replay bit-exactly')
            report.update(status='pass',phase='complete');persist()
        except BaseException as error:
            report=json.loads(output.read_text())
            report.update(status='fail',error_type=type(error).__name__,error=str(error)[:2000]);persist();raise
    print(json.dumps({k:report[k] for k in ('stage','status','forward_calls','elapsed_seconds')}),flush=True)


if __name__=='__main__':main()
