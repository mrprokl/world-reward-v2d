"""Azure full-real native articulation continuation, saved A/B untouched.

All four original random records retained; only complete9/14 are supported.
One native MHR layer, full original chronology, fixed shape/hands/root/objectR.
Frozen automatic witness bounds + same original RGB/motion QA, not held-out
accuracy/PEN/physical contact or a submission. Native outputs seal before final QA.
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

import contact_feasible_real as saved
import native_joint_real as native
from mediapipe_cpu_runtime_verify import source
from world_reward.contact_feasible_placement import freeze_contact_witnesses
from world_reward.contact_placement_branches import freeze_original_branches, project_joint_translations_branches
from world_reward.native_contact_continuation import (BODY_ABI, ContinuationProtocol,
    ObservationGate, continue_native_contact)
from world_reward.shared_identity import NATIVE_PARAMETER_DIMS, validate_native_parameters

real = native.real
ROOT = native.ROOT
ENTRY = 'run_native_contact_continuation_real'
BUDGET = 900
HELPERS = tuple(dict.fromkeys(('infra/native_contact_continuation_real.py',
    'infra/run_native_contact_continuation_real.sh', 'src/world_reward/native_contact_continuation.py',
    *saved.HELPERS)))
METRIC_GATE_KEYS = dict(RGB_mean_px='RGB_reprojection_ratio_maximum',
    RGB_motion_increment_error_mean_px='RGB_motion_increment_ratio_maximum',
    reserved_human_RGB_mean_px='reserved_human_RGB_ratio_maximum',
    same_anatomical_contact_mean_m='contact_gap_ratio_maximum',
    same_anatomical_contact_p95_m='contact_gap_ratio_maximum',
    object_acceleration_p95_m_s2='object_acceleration_p95_ratio_maximum',
    object_angular_acceleration_p95_rad_s2='object_acceleration_p95_ratio_maximum',
    human_centroid_acceleration_p95_m_s2='human_acceleration_p95_ratio_maximum')


def gates(cfg, bank):
    return tuple(ObservationGate(metric, cfg['gates'][key], 1e-12,
        allow_missing=(metric.startswith('RGB') and len(bank['points']) == 0))
        for metric, key in METRIC_GATE_KEYS.items())


def frozen_evidence(bank, src, b):
    active = b['activation']; identifiers = src['QA_witness_ids']
    original = np.full((*active.shape, 3), np.nan, np.float64)
    f, side = np.nonzero(active); original[f, side] = src['human'][f, identifiers[f, side]]
    evidence = freeze_contact_witnesses(original, src['QA_hand_ids'], identifiers,
        bank['rotations'], bank['translations'], bank['vertices'], bank['faces'], active,
        bank['frame_index'], config=saved.PLACEMENT,
        source_reference='independently_pinned_A052_and_B'+saved.B_REVISION,
        witness_selection_reference=saved.WITNESS_DEFINITION, original_selection_is_whole_hand_minimum=False)
    return evidence, freeze_original_branches(evidence, original, bank['translations'])


def decode_geometry(torch, layer, params, src, stats):
    """Same native batch16/direct204 computation as native_joint_real export."""
    count = len(params['mhr_trans']); validate_native_parameters(params, count, require_shared_identity=True)
    target = np.empty((count, 18439, 3), np.float32); joints = np.empty((count, 127, 3), np.float32)
    keypoints = np.empty((count, 70, 3), np.float32); controls = np.empty((count, 204), np.float32)
    started = time.monotonic(); direct_error = 0.; faces = None
    with torch.inference_mode(), torch.jit.optimized_execution(False):
        for start in range(0, count, 16):
            sl = slice(start, min(start+16, count))
            p = {k: torch.tensor(v[sl].copy(), device='cuda', dtype=torch.float32) for k, v in params.items()}
            decoded = layer.mhr_forward(p)
            arrays = [v.cpu().numpy().copy() for v in (decoded.vertices, decoded.joints, decoded.keypoints)]
            if any(not np.isfinite(v).all() or (v[..., 2] <= 0).any() for v in arrays):
                raise ValueError('All full native geometry must remain finite positive-Z')
            target[sl], joints[sl], keypoints[sl] = arrays
            f = decoded.faces.cpu().numpy().copy()
            if (not np.array_equal(f, src['native']['human_faces'])
                    or (faces is not None and not np.array_equal(faces, f))):
                raise ValueError('Full original native human topology changed')
            faces = f
            context = layer.backend._vertices_context(p, detach_fixed=False)
            trans, body, shape, scale = layer.backend._mutable_vertices_inputs(p, context)
            direct, dc = context.head.mhr_forward(global_trans=trans*context.flip,
                global_rot=context.global_rot, body_pose_params=body, hand_pose_params=context.hand,
                scale_params=scale, shape_params=shape, expr_params=context.face, return_model_params=True)
            error = float(np.linalg.norm((direct*context.flip).cpu().numpy().astype(float)-arrays[0].astype(float), axis=-1).max())
            if error > 1e-5 or dc.shape != (sl.stop-sl.start, 204) or not torch.isfinite(dc).all():
                raise ValueError('Actual native direct204 geometry/control parity failed')
            direct_error = max(direct_error, error); controls[sl] = dc.cpu().numpy()
    if any(v.tobytes() != controls[0, 136:].tobytes() for v in controls[:, 136:]):
        raise ValueError('All68 native direct scale controls must remain byte-constant')
    row = dict(call=len(stats)+1, seconds=time.monotonic()-started, full_frames=count,
        direct_geometry_max_error_m=direct_error, direct204_generated=True)
    if not stats:
        delta = float(np.linalg.norm(target.astype(float)-src['human'].astype(float), axis=-1).max())
        row.update(original_saved_A_geometry_max_error_m=delta,
            original_saved_A_geometry_byte_identical=saved.same_bytes(target, src['human']))
        if delta > 1e-5:
            stats.append(row)
            raise ValueError('Replayed native A differs from independently saved baseline geometry')
    stats.append(row)
    print('NATIVE_CONTINUATION '+json.dumps(dict(phase='native_decode_complete', call=row['call'],
        seconds=round(row['seconds'], 3), frames=count)), flush=True)
    return dict(human_vertices=target, human_joints=joints, human_keypoints=keypoints,
        human_faces=faces.astype(np.int64), frame_index=np.arange(count, dtype=np.int64),
        pose=controls[:, :136].copy(), scales=controls[0, 136:].copy())


def layer_factory(torch, ledger, src, out):
    location, assets = native.native_assets(ledger, src)
    ledger.record(location/'lib_mhr/body_pose.py', dict(bytes=BODY_ABI['source_bytes'], sha256=BODY_ABI['source_sha256']))
    sys.path[:0] = [str(location), '/workspace/v2d_sam3d_body/lib']
    os.environ.update(MHR_ASSETS_ROOT=str(ROOT/'weights/cari4d/sam3d_body'), MOMENTUM_ENABLED='0')
    from lib_mhr.mhr_layer import MHRLayer
    if Path(sys.modules[MHRLayer.__module__].__file__).resolve() != location/'lib_mhr/mhr_layer.py':
        raise ValueError('Original verified native decoder import required')
    layer = MHRLayer.from_mhr_assets(mhr_assets_root=Path('/workspace/v2d_sam3d_body/lib'),
        checkpoint_path=assets/'model.ckpt', buffer_path=out/'never_use_unverified_buffer.pt',
        mhr_model_path=assets/'assets/mhr_model.pt', device='cuda')
    if layer.decoder_identity() != src['forward']['decoder_identity']:
        raise ValueError('Original prepared native decoder identity differs')
    return layer


def seal_selected(out, bank, src, selected):
    p = selected['parameters']; g = selected['geometry']; count = len(g['frame_index'])
    poses = np.tile(np.eye(4), (count, 1, 1)); poses[:, :3, :3] = bank['rotations']
    poses[:, :3, 3] = selected['object_translation']
    controls = np.c_[g['pose'], np.broadcast_to(g['scales'], (count, 68))].astype(np.float32)
    trajectory = native.export.trajectory(controls, p, poses, bank['vertices'], bank['faces'], bank['original_K'], src['spec'])
    native.export.object_roundtrip(bank['vertices'], bank['faces'], poses, count)
    arrays = dict(p, mhr_joints=g['human_joints'], mhr_keypoints=g['human_keypoints'],
        human_faces=g['human_faces'], frame_index=g['frame_index'])
    return dict(target=real.seal_file(out/'target.npy', lambda f: np.save(f, g['human_vertices'], allow_pickle=False)),
        trajectory=real.seal_file(out/'trajectory.npz', lambda f: np.savez_compressed(f, **trajectory)),
        native_parameters=real.seal_file(out/'native_parameters.npz', lambda f: np.savez_compressed(f, **arrays)),
        frozen_witnesses=real.seal_file(out/'QA_witnesses.npz', lambda f: np.savez_compressed(f,
            activations=selected['original_activations'], hand_vertex_ids=selected['original_witness_ids'],
            frame_index=g['frame_index'], emitted_same_witness_gaps_m=selected['witness_gaps_m'])))


def run_episode(episode, out, ledger, b_binding, torch, layer_state):
    out.mkdir(mode=0o755); started = time.monotonic()
    report = dict(status='fail', episode=episode, producer_revision=os.environ['WR_CODE_REVISION'],
        A_revision=real.old.SOURCE, B_revision=saved.B_REVISION, ground_truth_used=False, private_truth_read=False,
        baseline_modified=False, production_adopted=False, full_4D_accuracy_verified=False,
        whole_hand_minimum_claimed=False, penetration_evaluated=False, silhouette_re_evaluated=False,
        physical_contact_verified=False, hand_labeled_test=False, oracle_modes=[], original_activations_unchanged=True,
        continuation_protocol=asdict(ContinuationProtocol()), placement_protocol=asdict(saved.PLACEMENT),
        native_decode_calls=[], interpretation='automatic_same_witness_RGB_motion_QA_not_truth_or_Kaggle_scores')
    try:
        bank, src = native.original_sources(ledger, episode)
        spec = real.saved.load_npz(ledger, ROOT/'weights/cari4d/refinement/mhr_hand_surface_spec.npz', real.contact.HAND_PIN)
        src['QA_hand_ids'] = real.contact.hand_indices(spec, src['native']['human_faces'], len(src['human'][0]))
        b = saved.saved_B(ledger, episode, bank, src, b_binding)
        report.update(B_report_pin=saved.B_REPORT_PINS[episode], baseline_binding=b['report']['baseline_binding'])
        a_metrics = native.quality(bank, src, src['human'], src['native']['mhr_keypoints'], bank['rotations'],
            bank['translations'], b['evidence'], b['activation'])
        b_metrics = native.quality(bank, src, b['human'], b['params']['mhr_keypoints'], b['trajectory']['object_rotation'],
            b['trajectory']['object_translation'], b['evidence'], b['activation'])
        saved.preserve_reported_quality(a_metrics, b['report']['metrics'][saved.A_NAME])
        saved.preserve_reported_quality(b_metrics, b['report']['metrics'][saved.B_NAME])
        report['metrics'] = {saved.A_NAME: a_metrics, saved.B_NAME: b_metrics}
        report['A_B_QA_pin'] = real.seal_json(out/'A_B_QA.json', dict(metrics=report['metrics'], B_report_pin=saved.B_REPORT_PINS[episode]))
        evidence, branches = frozen_evidence(bank, src, b)
        report['frozen_observation_gates'] = [asdict(g) for g in gates(b['cfg'], bank)]
        if layer_state.get('layer') is None:
            layer_state['layer'] = layer_factory(torch, ledger, src, out)
            layer_state['decoder_identity'] = src['forward']['decoder_identity']
        elif layer_state['decoder_identity'] != src['forward']['decoder_identity']:
            raise ValueError('Shared native layer differs from original second clip decoder')
        def decode(p):
            if time.monotonic() >= layer_state.get('deadline', float('inf'))-1:
                raise TimeoutError('Frozen full-cohort native continuation budget exhausted before decode')
            return decode_geometry(torch, layer_state['layer'], p, src, report['native_decode_calls'])
        def observe(g, t): return native.quality(bank, src, g['human_vertices'], g['human_keypoints'],
            bank['rotations'], t, b['evidence'], b['activation'])
        def place(p, t, g, e):
            fitted = project_joint_translations_branches(e, g['human_vertices'][:, src['QA_hand_ids']],
                p['mhr_trans'], t, original_branches=branches)
            return fitted['human_translation_camera'], fitted['object_translation_camera']
        result = continue_native_contact({k: src['native'][k] for k in NATIVE_PARAMETER_DIMS},
            {k: b['params'][k] for k in NATIVE_PARAMETER_DIMS}, bank['translations'], b['trajectory']['object_translation'],
            evidence, decode_native=decode, evaluate_observations=observe, gates=gates(b['cfg'], bank), place_translations=place)
        report['continuation'] = {k: v for k, v in result.items() if k not in ('parameters', 'object_translation',
            'geometry', 'witness_gaps_m', 'original_activations', 'original_witness_ids')}
        report['candidate_outputs'] = seal_selected(out, bank, src, result)
        report.update(fitted_outputs_sealed_before_final_QA=True, full_original_frames=len(bank['frame_index']),
            source_fps=bank['fps'], native_direct136_generated=True, stale_B_pose_used=False,
            native_direct_replay_independently_verified=False)
        real.seal_json(out/'geometry.json', report)
        final = observe(result['geometry'], result['object_translation'])
        saved.preserve_reported_quality(final, result['observation_metrics'])
        report['metrics']['C_native_geodesic_continuation'] = final
        report['decision'] = dict(C_vs_A=native.quality_decision(b['cfg'], a_metrics, final),
            C_vs_B=native.quality_decision(b['cfg'], b_metrics, final))
        report['status'] = result['status']
    except Exception as exc:
        report.update(error_type=type(exc).__name__, error=str(exc)[:400], no_C_fabricated=True)
    finally:
        report['elapsed_seconds'] = time.monotonic()-started
        report_pin = real.seal_json(out/'report.json', report)
        print('NATIVE_CONTINUATION '+json.dumps(dict(episode=episode, status=report['status'],
            seconds=round(report['elapsed_seconds'], 3))), flush=True)
        gc.collect(); torch.cuda.empty_cache()
    return dict(episode=episode, status=report['status'], report=report_pin)


def run():
    revision = os.environ['WR_CODE_REVISION']; code = Path(os.environ['WR_CODE']); started = time.monotonic()
    if (Path(os.environ['WR_ROOT']) != ROOT or code != ROOT/'jobs'/revision/ENTRY/'code'
            or os.environ.get('WR_IMAGE_ID') != native.IMAGE):
        raise ValueError('Exact immutable Azure GPU continuation source/image required')
    out = ROOT/'results'/('native-contact-continuation-real-'+revision); real.old.fresh_runtime_output(out)
    report = dict(schema='world_reward.native_contact_continuation_real.v1', status='fail', producer_revision=revision,
        ground_truth_used=False, private_truth_read=False, baseline_modified=False, production_adopted=False,
        new_observation_model_inference_calls=0, new_native_optimization_fits=0, GPU_requested=True, budget_seconds=BUDGET,
        cohort=dict(random_seed=20261008, population=30, episodes=list(saved.COHORT)), episodes=[])
    ledger = real.old.ArtifactLedger(); layer_state = dict(deadline=started+BUDGET)
    try:
        report['source_binding'] = source(ROOT, code, revision, ENTRY, HELPERS)
        oldcode = ROOT/'jobs'/saved.B_REVISION/native.ENTRY/'code'
        b_binding = source(ROOT, oldcode, saved.B_REVISION, native.ENTRY, native.HELPERS)
        report['B_source_binding'] = b_binding
        import torch
        if str(torch.__version__) != '2.5.1+cu124' or not torch.cuda.is_available() or torch.version.cuda != '12.4':
            raise ValueError('Exact original Torch/CUDA native runtime required')
        torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        torch.manual_seed(0); np.random.seed(0); torch.cuda.manual_seed_all(0)
        for episode in saved.COHORT:
            if episode not in saved.FRAMES: report['episodes'].append(saved.unsupported(episode)); continue
            if time.monotonic() >= layer_state['deadline']-1:
                report['episodes'].append(dict(episode=episode, status='not_executed_budget_exhausted',
                    baseline_retained=True, rerolled=False, fabricated_predictions=False))
                continue
            report['episodes'].append(run_episode(episode, out/f'episode_{episode:06d}', ledger, b_binding, torch, layer_state))
        ledger.verify()
        report.update(status='complete_native_continuation_diagnostic', source_inputs_rehashed=True,
            input_ledger=ledger.records, unexpected_failures=sum(r['status'] in ('fail', 'not_executed_budget_exhausted')
                for r in report['episodes']), native_layers_loaded=int(layer_state.get('layer') is not None))
    except Exception as exc:
        report.update(error_type=type(exc).__name__, error=str(exc)[:400])
    finally:
        layer_state.clear(); gc.collect(); report['elapsed_seconds']=time.monotonic()-started
        real.seal_json(out/'report.json', report)
    return 0 if report['status']=='complete_native_continuation_diagnostic' and not report['unexpected_failures'] else 1


if __name__=='__main__':
    def expired(*_): raise TimeoutError('Frozen900second real native continuation budget exceeded')
    signal.signal(signal.SIGALRM, expired); signal.alarm(BUDGET)
    try: raise SystemExit(run())
    finally: signal.alarm(0)
