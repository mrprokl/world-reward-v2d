"""Bounded actual full-T native SQP; A/B sealed unchanged, no GT or new detector.

Full pinned native stage181 objective (PEN/silhouette/prior/temporal) plus exact
B RGB extension coefficients. Sparse same-ID anatomical VJPs, not dropped frames
or surfaces. Witness bounds freeze on fresh native A before any pose proposal.
Results remain internal automatic QA, not physical/held-out/leaderboard accuracy.
"""
from dataclasses import asdict
import gc
import json
import os
from pathlib import Path
import signal
import sys
import time

import numpy as np

import native_contact_continuation_real as replay
from mediapipe_cpu_runtime_verify import source
from world_reward.native_pose_sqp import PoseSQPConfig, optimize_native_pose, _BudgetExhausted
from world_reward.native_pose_sqp_adapter import NativeSQPCallbacks, LAYER_PIN, OPTIMIZER_PIN, OBJECTIVE_STAGE, prime_native_autograd
from world_reward.native_joint_refinement import native_joint_optimizer_class
from world_reward.shared_identity import NATIVE_PARAMETER_DIMS

native=replay.native;saved=replay.saved;real=replay.real;ROOT=replay.ROOT
ENTRY='run_native_pose_sqp_real'
BUDGET=900
HELPERS=tuple(dict.fromkeys(('infra/native_pose_sqp_real.py','infra/run_native_pose_sqp_real.sh',
    'src/world_reward/native_pose_sqp.py','src/world_reward/native_pose_sqp_adapter.py',*replay.HELPERS)))


def native_instance(torch,layer,ledger,src,bank,b,code):
    """Original raw bundle/config defines priors/activation; never rebase on B."""
    location,_=native.native_assets(ledger,src)
    sys.path.insert(0,str(location))
    from learning.training import mhr_opt_refineout as optimizer
    if Path(optimizer.__file__).resolve()!=location/native.full.contract.OPTIMIZER_RELATIVE_PATH:
        raise ValueError('Unchanged authenticated native optimizer import required')
    raw=torch.load(src['raw'],map_location='cpu',weights_only=False)
    native.full.validate_source_bundle(raw,src['mesh'],len(bank['frame_index']))
    fingerprint=native.full.fingerprint(raw)
    vertices,faces=optimizer._load_object_vertices(src['mesh'])
    if not np.array_equal(vertices,bank['vertices']) or not np.array_equal(faces,bank['faces']):
        raise ValueError('Native complete immutable object topology/scale differs')
    cfg=optimizer.MHRParityPostOptConfig(
        penetration_collision_proxy_path=str(ROOT/'weights/cari4d/refinement/mhr_collision_proxy_4000v.npz'),
        hand_surface_spec_path=str(ROOT/'weights/cari4d/refinement/mhr_hand_surface_spec.npz'),report_every=100)
    if asdict(cfg)!=b['report']['native_config']:
        raise ValueError('Same B full native physics/silhouette config required')
    _,extension=native.settings(code)
    instance=native_joint_optimizer_class(optimizer,b['evidence'],extension)(raw,vertices,faces,cfg,mhr_layer=layer)
    if not np.array_equal(instance.object_rotation_initial.detach().cpu().numpy(),bank['rotations']):
        raise ValueError('Original native fixed object rotations differ')
    if not np.array_equal(instance.hand_surface_vertex_indices.detach().cpu().numpy(),src['QA_hand_ids']):
        raise ValueError('Original native anatomical IDs differ')
    if OBJECTIVE_STAGE<=cfg.penetration_start_fraction*cfg.num_steps or cfg.w_penetration<=0 or cfg.w_silhouette<=0:
        raise ValueError('Stage181 must include original scheduled PEN and silhouette')
    return instance,raw,fingerprint,dict(native_config=asdict(cfg),extension=asdict(extension),
        objective_stage=OBJECTIVE_STAGE,source_binding=dict(layer=LAYER_PIN,optimizer=OPTIMIZER_PIN),
        raw_priors_rebased_on_baseline_or_candidate=False,scheduled_PEN_and_silhouette_retained=True,
        native_objective_FP32_object_translation_cast=True)


def control_gauge_receipt(parameters):
    from world_reward.native_contact_continuation import _so3
    raw=parameters['mhr_body_pose_cont'][:,:138].astype(float).reshape(-1,23,6);r=_so3(raw)
    first=np.linalg.norm(raw[...,:3],axis=-1);parallel=np.sum(raw[...,3:]*r[..., :,0],axis=-1)
    second=np.linalg.norm(raw[...,3:]-parallel[...,None]*r[..., :,0],axis=-1)
    so2=np.linalg.norm(parameters['mhr_body_pose_cont'][:,138:254].astype(float).reshape(-1,58,2),axis=-1)
    return dict(SO3_first_norm_min=float(first.min()),SO3_first_norm_max=float(first.max()),
        SO3_second_orthogonal_norm_min=float(second.min()),SO3_second_orthogonal_norm_max=float(second.max()),
        SO3_parallel_abs_max=float(np.abs(parallel).max()),SO2_radius_min=float(so2.min()),SO2_radius_max=float(so2.max()),
        raw_encoding_gauge_preserved_under_retraction=True,zero_tangent_blocks_byte_unchanged=True)


def budget_for_next_step(deadline,calls,decode_calls,QA_seconds):
    """Measured minimum next complete step; never change work quality to fit."""
    jac=[r['seconds'] for r in calls if r['kind']=='full_native_gradient_and_same_ID_VJP']
    objective=[r['seconds'] for r in calls if r['kind']=='full_native_objective']
    if jac and objective and decode_calls:
        minimum=jac[-1]+objective[-1]+decode_calls[-1]['seconds']+QA_seconds
        if deadline-time.monotonic()<=minimum:
            raise _BudgetExhausted('Measured complete native gradient/decode/objective/QA step cannot finish frozen900s budget')


def run_episode(episode,out,ledger,b_binding,torch,state,code):
    out.mkdir(mode=0o755);started=time.monotonic();callbacks=None;raw=None
    report=dict(status='fail',episode=episode,producer_revision=os.environ['WR_CODE_REVISION'],
        A_revision=real.old.SOURCE,B_revision=saved.B_REVISION,ground_truth_used=False,private_truth_read=False,
        baseline_modified=False,production_adopted=False,full_4D_accuracy_verified=False,
        whole_hand_minimum_claimed=False,penetration_evaluated=False,physical_contact_verified=False,
        hand_labeled_test=False,oracle_modes=[],native_decode_calls=[],callback_receipts=[],
        pose_SQP_protocol=asdict(PoseSQPConfig()),placement_protocol=asdict(saved.PLACEMENT),
        interpretation='full_native_training_loss_and_same_witness_RGB_motion_QA_not_truth_or_Kaggle_scores')
    try:
        bank,src=native.original_sources(ledger,episode)
        spec=real.saved.load_npz(ledger,ROOT/'weights/cari4d/refinement/mhr_hand_surface_spec.npz',real.contact.HAND_PIN)
        src['QA_hand_ids']=real.contact.hand_indices(spec,src['native']['human_faces'],len(src['human'][0]))
        b=saved.saved_B(ledger,episode,bank,src,b_binding)
        report.update(B_report_pin=saved.B_REPORT_PINS[episode],baseline_binding=b['report']['baseline_binding'])
        qa_started=time.monotonic()
        a_metrics=native.quality(bank,src,src['human'],src['native']['mhr_keypoints'],bank['rotations'],bank['translations'],b['evidence'],b['activation'])
        b_metrics=native.quality(bank,src,b['human'],b['params']['mhr_keypoints'],bank['rotations'],b['trajectory']['object_translation'],b['evidence'],b['activation'])
        QA_seconds=(time.monotonic()-qa_started)/2
        saved.preserve_reported_quality(a_metrics,b['report']['metrics'][saved.A_NAME])
        saved.preserve_reported_quality(b_metrics,b['report']['metrics'][saved.B_NAME])
        report['metrics']={saved.A_NAME:a_metrics,saved.B_NAME:b_metrics}
        report['A_B_QA_pin']=real.seal_json(out/'A_B_QA.json',dict(metrics=report['metrics'],B_report_pin=saved.B_REPORT_PINS[episode]))
        report['frozen_observation_gates']=[asdict(g) for g in replay.gates(b['cfg'],bank)]
        original,_=replay.frozen_evidence(bank,src,b)
        if state.get('layer') is None:
            state['layer']=replay.layer_factory(torch,ledger,src,out);state['decoder_identity']=src['forward']['decoder_identity']
        elif state['decoder_identity']!=src['forward']['decoder_identity']:
            raise ValueError('One shared native layer must match each original decoder identity')
        def decode(p):
            if time.monotonic()>=state['deadline']:raise _BudgetExhausted('Inclusive900s exhausted before actual native decode')
            result=replay.decode_geometry(torch,state['layer'],p,src,report['native_decode_calls'])
            if time.monotonic()>=state['deadline']:raise _BudgetExhausted('Over-budget actual native geometry discarded')
            return result
        def observe(g,t):return native.quality(bank,src,g['human_vertices'],g['human_keypoints'],bank['rotations'],t,b['evidence'],b['activation'])
        a_parameters={k:src['native'][k] for k in NATIVE_PARAMETER_DIMS}
        report['native_autograd_priming']=prime_native_autograd(torch,state['layer'],a_parameters)
        if state['layer'].decoder_identity()!=src['forward']['decoder_identity']:
            raise ValueError('Autograd priming changed original native decoder identity')
        report['native_autograd_priming_pin']=real.seal_json(out/'native_autograd_priming.json',report['native_autograd_priming'])
        native_a=decode(a_parameters)
        evidence,_=replay.frozen_evidence(bank,src,b,native_a)
        report['baseline_witness_reference']=replay.witness_reference_receipt(out,src,original,evidence,native_a)
        report['native_A_encoding_gauge']=control_gauge_receipt(a_parameters)
        constructor_started=time.monotonic()
        instance,raw,fingerprint,protocol=native_instance(torch,state['layer'],ledger,src,bank,b,code)
        report['native_constructor_seconds']=time.monotonic()-constructor_started;report['native_objective_protocol']=protocol
        real.seal_json(out/'native_objective_protocol.json',protocol)
        def completed(row):
            pin=real.seal_json(out/f'callback_{len(report["callback_receipts"]):03d}.json',row)
            report['callback_receipts'].append(pin)
            print('NATIVE_POSE_SQP '+json.dumps(dict(episode=episode,phase=row['kind'],seconds=round(row['seconds'],3),loss=row['loss'])),flush=True)
        callbacks=NativeSQPCallbacks(torch,state['layer'],instance,decode,evidence,deadline=state['deadline'],
            source_binding=protocol['source_binding'],on_completion=completed)
        def linearize(*args):
            budget_for_next_step(state['deadline'],callbacks.calls,report['native_decode_calls'],QA_seconds)
            value=callbacks.linearize(*args)
            # First measured Jacobian is preserved even if a complete next
            # proposal/decode/objective + final QA no longer fits. No term drop.
            objective_cost=[r['seconds'] for r in callbacks.calls if r['kind']=='full_native_objective']
            if objective_cost and report['native_decode_calls']:
                minimum=objective_cost[-1]+report['native_decode_calls'][-1]['seconds']+2*QA_seconds
                if state['deadline']-time.monotonic()<=minimum:
                    raise _BudgetExhausted('Measured native proposal/decode/objective/final QA cannot finish frozen900s budget')
            return value
        decoder=replay.cached_baseline_decoder(a_parameters,native_a,decode)
        result=optimize_native_pose(a_parameters,bank['translations'],evidence,decode_native=decoder,
            linearize_native=linearize,evaluate_objective=callbacks.objective,evaluate_observations=observe,
            gates=replay.gates(b['cfg'],bank))
        if native.full.fingerprint(raw)!=fingerprint:raise ValueError('Original raw native bundle modified')
        report['SQP']={k:v for k,v in result.items() if k not in ('parameters','object_translation','geometry',
            'witness_gaps_m','original_activations','original_witness_ids')}
        report['candidate_outputs']=replay.seal_selected(out,bank,src,result)
        report.update(fitted_outputs_sealed_before_final_QA=True,full_original_frames=len(bank['frame_index']),
            source_fps=bank['fps'],native_direct136_generated=True,stale_B_pose_used=False,
            native_direct_replay_independently_verified=False,original_raw_bundle_byte_fingerprint_unchanged=True)
        real.seal_json(out/'geometry.json',report)
        final=observe(result['geometry'],result['object_translation'])
        saved.preserve_reported_quality(final,result['observation_metrics'])
        report['metrics']['C_native_pose_SQP']=final
        report['decision']=dict(C_vs_A=native.quality_decision(b['cfg'],a_metrics,final),C_vs_B=native.quality_decision(b['cfg'],b_metrics,final))
        report['status']=result['status']
        report['original_saved_A_final_QA_passed']=report['decision']['C_vs_A']['passed']
    except Exception as exc:
        report.update(error_type=type(exc).__name__,error=str(exc)[:400],no_C_fabricated=True)
        import traceback
        frames=traceback.extract_tb(exc.__traceback__)[-10:]
        report['failure_traceback']=[dict(file=Path(f.filename).name,line=f.lineno,function=f.name) for f in frames]
        if hasattr(exc,'diagnostics'):report['failure_diagnostics']=exc.diagnostics
    finally:
        if callbacks is not None:report['actual_native_callback_calls']=callbacks.calls
        report['elapsed_seconds']=time.monotonic()-started;pin=real.seal_json(out/'report.json',report)
        print('NATIVE_POSE_SQP '+json.dumps(dict(episode=episode,status=report['status'],seconds=round(report['elapsed_seconds'],3))),flush=True)
        gc.collect();torch.cuda.empty_cache()
    return dict(episode=episode,status=report['status'],report=pin)


def run():
    revision=os.environ['WR_CODE_REVISION'];code=Path(os.environ['WR_CODE']);started=time.monotonic()
    if (Path(os.environ['WR_ROOT'])!=ROOT or code!=ROOT/'jobs'/revision/ENTRY/'code' or os.environ.get('WR_IMAGE_ID')!=native.IMAGE):
        raise ValueError('Exact immutable Azure native SQP source/image required')
    out=ROOT/'results'/('native-pose-sqp-real-'+revision);real.old.fresh_runtime_output(out)
    report=dict(schema='world_reward.native_pose_sqp_real.v1',status='fail',producer_revision=revision,
        ground_truth_used=False,private_truth_read=False,baseline_modified=False,production_adopted=False,
        budget_seconds=BUDGET,budget_includes_models_constructor_callbacks_QA=True,GPU_requested=True,
        new_observation_model_inference_calls=0,new_native_optimization_fits=0,
        new_full_native_pose_SQP_experiments=2,original_300step_native_optimizer_loop_rerun=False,
        cohort=dict(random_seed=20261008,population=30,episodes=list(saved.COHORT)),episodes=[])
    ledger=real.old.ArtifactLedger();state=dict(deadline=started+BUDGET)
    try:
        report['source_binding']=source(ROOT,code,revision,ENTRY,HELPERS)
        bcode=ROOT/'jobs'/saved.B_REVISION/native.ENTRY/'code'
        b_binding=source(ROOT,bcode,saved.B_REVISION,native.ENTRY,native.HELPERS);report['B_source_binding']=b_binding
        import torch
        if str(torch.__version__)!='2.5.1+cu124' or not torch.cuda.is_available() or torch.version.cuda!='12.4':
            raise ValueError('Exact original Torch/CUDA runtime required')
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.manual_seed(0);np.random.seed(0);torch.cuda.manual_seed_all(0)
        for episode in saved.COHORT:
            if episode not in saved.FRAMES:report['episodes'].append(saved.unsupported(episode));continue
            if time.monotonic()>=state['deadline']:
                report['episodes'].append(dict(episode=episode,status='not_executed_budget_exhausted',baseline_retained=True,rerolled=False,fabricated_predictions=False));continue
            report['episodes'].append(run_episode(episode,out/f'episode_{episode:06d}',ledger,b_binding,torch,state,code))
        ledger.verify();report.update(status='complete_native_pose_sqp_diagnostic',source_inputs_rehashed=True,
            input_ledger=ledger.records,unexpected_failures=sum(r['status'] in ('fail','not_executed_budget_exhausted') for r in report['episodes']),
            native_layers_loaded=int(state.get('layer') is not None))
    except Exception as exc:report.update(error_type=type(exc).__name__,error=str(exc)[:400])
    finally:
        state.clear();gc.collect();report['elapsed_seconds']=time.monotonic()-started;real.seal_json(out/'report.json',report)
    return 0 if report['status']=='complete_native_pose_sqp_diagnostic' and not report['unexpected_failures'] else 1


if __name__=='__main__':
    def expired(*_):raise TimeoutError('Hard inclusive900second native SQP budget exceeded')
    signal.signal(signal.SIGALRM,expired);signal.alarm(BUDGET)
    try:raise SystemExit(run())
    finally:signal.alarm(0)
