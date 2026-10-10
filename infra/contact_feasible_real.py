"""Saved full-real A/B/C contact-feasibility hypothesis, CPU-only, no decoder.

C translates the complete B human and rigid object together. The constraint is
ONLY a preserved original automatic QA witness bound, not a whole-hand/contact
truth guarantee. Original native activity, shape, hands, rotations, chronology,
RGB observations and masks are never changed. Direct204 C export is intentionally
unavailable until an independent native decoder replay; this is not a submission.
"""
from dataclasses import asdict
import json
import os
from pathlib import Path
import signal
import time

import numpy as np

import native_joint_real as native
from mediapipe_cpu_runtime_verify import canonical, source, strict
from world_reward.contact_feasible_placement import (ContactPlacementConfig,
    ContactFeasibilityFailure, freeze_contact_witnesses, project_joint_translations)
from world_reward.native_joint_refinement import JointImageEvidence
from world_reward.shared_identity import NATIVE_PARAMETER_DIMS, validate_native_parameters
from world_reward.sequence_pose import _ContactTriangleSurface

real = native.real
ROOT = native.ROOT
ENTRY = 'run_contact_feasible_real'
B_REVISION = 'ebcee73cc76e036b3c2ec9c32e9a72abc951ad78'
BUDGET = 600
COHORT = (9, 1, 14, 7)
FRAMES = {9: 415, 14: 442}
A_NAME = 'A_saved_native_baseline'
B_NAME = 'B_native_joint_RGB_translation'
C_NAME = 'C_joint_contact_feasible_translation'
WITNESS_DEFINITION = 'frozen_baseline_nearest_object_vertex_proposed_hand_vertex_exact_triangle_evaluated_sameIDs'
B_REPORT_PINS = {
    9: dict(bytes=163589, sha256='118276f5f81bf926b08906e2973ef74af213d634631b7cb283611a012db72200'),
    14: dict(bytes=171105, sha256='90d77958a2069922c9de025de414e762d0b2f11dd6d33390ac545bff4ea2e735'),
}
PLACEMENT = ContactPlacementConfig(.05, 1e-7, 1e-10, 400, 8,
    'manufactured_numerical_unit_contract_before_real_C_not_HOI_quality_calibration')
HELPERS = tuple(dict.fromkeys(('infra/contact_feasible_real.py',
    'infra/run_contact_feasible_real.sh', 'src/world_reward/contact_feasible_placement.py',
    'src/world_reward/shared_identity.py', *native.HELPERS)))


def same_bytes(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def validate_saved_arrays(bank, src, trajectory, params, human, witnesses,
                          *, body_count=18439, joint_count=127, keypoint_count=70):
    """Require the frozen A/B topology, native ABI, clock, activity and geometry."""
    count = len(bank['frame_index']); grid = np.arange(count, dtype=np.int64)
    if (human.dtype != np.float32 or human.shape != (count, body_count, 3)
            or not np.isfinite(human).all() or (human[..., 2] <= 0).any()
            or set(trajectory) != native.export.TRAJECTORY_KEYS
            or set(params) != set(NATIVE_PARAMETER_DIMS) | {
                'mhr_joints', 'mhr_keypoints', 'human_faces', 'frame_index'}):
        raise ValueError('Complete original native B geometry/schema required')
    validate_native_parameters({k: params[k] for k in NATIVE_PARAMETER_DIMS},
        count, require_shared_identity=True)
    for key, shape in [('mhr_joints', (count, joint_count, 3)),
                       ('mhr_keypoints', (count, keypoint_count, 3))]:
        if (params[key].dtype != np.float32 or params[key].shape != shape
                or not np.isfinite(params[key]).all() or (params[key][..., 2] <= 0).any()):
            raise ValueError('Finite full native B joints/keypoints required')
    for value in (bank['frame_index'], trajectory['frame_index'], params['frame_index'],
                  witnesses.get('frame_index')):
        if not isinstance(value, np.ndarray) or value.dtype != np.int64 or not np.array_equal(value, grid):
            raise ValueError('Original complete full-T int64 chronology required')
    for key, reference in [('object_vertices', bank['vertices']), ('object_faces', bank['faces']),
                           ('object_rotation', bank['rotations']), ('object_scale', bank['object_scale']),
                           ('camera_K', bank['original_K'])]:
        if not same_bytes(trajectory[key], reference):
            raise ValueError('Frozen A/B camera/rotation/geometry changed: ' + key)
    for key in ('mhr_global_rot6d', 'mhr_hand', 'mhr_shape', 'mhr_scale', 'mhr_face', 'human_faces'):
        if not same_bytes(params[key], src['native'][key]):
            raise ValueError('Frozen A/B native identity/root/hands changed: ' + key)
    if (not same_bytes(params['mhr_body_pose_cont'][:, 254:], src['native']['mhr_body_pose_cont'][:, 254:])
            or trajectory['object_translation'].shape != (count, 3)
            or trajectory['object_translation'].dtype != np.float32
            or not np.isfinite(trajectory['object_translation']).all()):
        raise ValueError('Native internal translations/object pose changed outside declared B fit')
    active = witnesses.get('activations'); ids = witnesses.get('hand_vertex_ids')
    if (set(witnesses) != {'activations', 'hand_vertex_ids', 'frame_index'}
            or not isinstance(active, np.ndarray) or active.dtype != bool or active.shape != (count, 2)
            or not isinstance(ids, np.ndarray) or ids.dtype != np.int64 or ids.shape != (count, 2)
            or (ids[active] < 0).any() or (ids[active] >= body_count).any() or (ids[~active] != -1).any()):
        raise ValueError('Original automatic full-T witness IDs/activation required; never self-disable')
    for side in range(2):
        if not np.isin(ids[active[:, side], side], src['QA_hand_ids'][side]).all():
            raise ValueError('Original witness must belong to exact native anatomical side')
    return active, ids


def validate_dwpose(arrays, bank, spec):
    count = len(bank['frame_index'])
    if (set(arrays) != {'original_xy', 'raw_scores', 'boxes_original_xyxy', 'actor_present', 'frame_index', 'fps'}
            or arrays['original_xy'].shape != (count, 133, 2)
            or arrays['original_xy'].dtype.kind != 'f'
            or arrays['raw_scores'].shape != (count, 133) or arrays['raw_scores'].dtype != np.float32
            or not np.isfinite(arrays['raw_scores']).all() or (arrays['raw_scores'] < 0).any()
            or not np.isfinite(arrays['original_xy'][arrays['raw_scores'] > 0]).all()
            or arrays['boxes_original_xyxy'].shape != (count, 4)
            or arrays['actor_present'].dtype != bool or arrays['actor_present'].shape != (count,)
            or arrays['frame_index'].dtype != np.int64
            or not np.array_equal(arrays['frame_index'], bank['frame_index'])
            or arrays['fps'].shape != () or float(arrays['fps']) != bank['fps'] or bank['fps'] != 30.):
        raise ValueError('Saved automatic original-grid DWPose observations/chronology required')
    xy = arrays['original_xy'] * np.array([640 / spec.width, 480 / spec.height]) - .5
    return JointImageEvidence(bank['points'], bank['xy'], bank['visible'], xy, arrays['raw_scores'],
        bank['K'], bank['frame_index'], allow_empty_object_evidence=(spec.episode_index == 14))


def saved_B(ledger, episode, bank, src, binding):
    directory = ROOT / 'results' / ('native-joint-real-' + B_REVISION) / f'episode_{episode:06d}'
    report = strict(ledger.read(directory / 'report.json', B_REPORT_PINS[episode]))
    expected = dict(status='complete_diagnostic_not_quality_pass', episode=episode,
        producer_revision=B_REVISION, ground_truth_used=False, private_truth_read=False,
        baseline_modified=False, depth_used=False, production_adopted=False,
        source_inputs_rehashed=True, fitted_outputs_sealed_before_QA=True,
        actual_native_fit_completed=True, finite_full_T_native_history=True,
        native_effective_updates=301, frames=FRAMES[episode], source_binding=binding,
        QA_witness_definition=WITNESS_DEFINITION)
    if any(type(report.get(k)) is not type(v) or report[k] != v for k, v in expected.items()):
        raise ValueError('Pinned completed no-oracle native B/source closure required')
    cfg, _ = native.settings(Path(os.environ['WR_CODE']))
    if (report['protocol'] != cfg or report['baseline_binding'] != {
            'directory': str(src['directory']), 'outputs': src['export_pins']}):
        raise ValueError('Original B protocol/A export lineage differs')
    host = strict(ledger.read(directory / 'host-exit.json', maximum=4096))
    if (host.get('producer_revision') != B_REVISION or host.get('process_exit_code') != 0
            or host.get('container_absence_verified') is not True):
        raise ValueError('Completed original B container receipt required')
    pins = report['candidate_outputs']
    if set(pins) != {'target', 'trajectory', 'native_parameters'}:
        raise ValueError('Complete sealed original B output pins required')
    for filename, pin in [('target.npy', pins['target']), ('trajectory.npz', pins['trajectory']),
                          ('native_parameters.npz', pins['native_parameters']),
                          ('automatic_DWPose.npz', report['DWPose_observation_pin']),
                          ('QA_witnesses.npz', report['native_active_witness_pin'])]:
        path = directory / filename
        if path.stat().st_mode & 0o222: raise ValueError('Original B outputs must remain sealed readonly')
        ledger.record(path, pin)
    human = np.load(directory / 'target.npy', mmap_mode='r', allow_pickle=False)
    trajectory = real.saved.load_npz(ledger, directory / 'trajectory.npz', pins['trajectory'])
    params = real.saved.load_npz(ledger, directory / 'native_parameters.npz', pins['native_parameters'])
    observations = real.saved.load_npz(ledger, directory / 'automatic_DWPose.npz', report['DWPose_observation_pin'])
    witnesses = real.saved.load_npz(ledger, directory / 'QA_witnesses.npz', report['native_active_witness_pin'])
    active, ids = validate_saved_arrays(bank, src, trajectory, params, human, witnesses)
    src['QA_witness_ids'] = ids
    return dict(directory=directory, report=report, human=human, trajectory=trajectory,
        params=params, activation=active, evidence=validate_dwpose(observations, bank, src['spec']), cfg=cfg)


def preserve_reported_quality(actual, reference):
    if set(actual) != set(reference): raise ValueError('Original A/B QA operator schema differs')
    for key, value in actual.items():
        old = reference[key]
        if type(value) in (float, int) and type(value) is not bool:
            if not np.isclose(value, old, rtol=1e-10, atol=1e-12):
                raise ValueError('Original A/B remeasured QA changed: ' + key)
        elif value != old: raise ValueError('Original A/B remeasured QA changed: ' + key)


def corrected_geometry(human, params, trajectory, result):
    """Camera-metre translation ABI only; no direct-controls replay claim."""
    dh = result['human_correction_camera']; do = result['object_correction_camera']
    count = len(human)
    if (dh.dtype != np.float64 or do.dtype != np.float64 or dh.shape != (count, 3)
            or do.shape != (count, 3) or not np.isfinite(dh).all() or not np.isfinite(do).all()
            or not np.array_equal(dh, -do)):
        raise ValueError('Complete finite equal-split joint camera correction required')
    target = human.astype(np.float64) + dh[:, None]
    native_params = {k: v.copy() for k, v in params.items()}
    for key in ('mhr_joints', 'mhr_keypoints'):
        native_params[key] = params[key].astype(np.float64) + dh[:, None]
    native_params['mhr_trans'] = params['mhr_trans'].astype(np.float64) + dh
    corrected = {k: v.copy() for k, v in trajectory.items() if k != 'pose'}
    # The unchanged B direct136 controls DO NOT describe C; prevent accidental
    # submission/replay by excluding pose and storing those only as a B source.
    corrected['object_translation'] = trajectory['object_translation'].astype(np.float64) + do
    corrected['human_translation_camera'] = native_params['mhr_trans'].copy()
    for array in (target, native_params['mhr_joints'], native_params['mhr_keypoints'], native_params['mhr_trans']):
        if not np.isfinite(array).all() or (array[..., 2] <= 0).any():
            raise ValueError('C full human geometry/translation must remain finite positive-Z')
    for key in params.keys() - {'mhr_trans', 'mhr_joints', 'mhr_keypoints'}:
        if not same_bytes(params[key], native_params[key]): raise ValueError('Frozen B native block changed')
    for key in trajectory.keys() - {'object_translation', 'pose'}:
        if not same_bytes(trajectory[key], corrected[key]): raise ValueError('Frozen B object/camera/identity changed')
    if not np.array_equal(native_params['mhr_trans'], result['human_translation_camera']):
        raise ValueError('Emitted camera root correction differs from primitive')
    if not np.array_equal(corrected['object_translation'], result['object_translation_camera']):
        raise ValueError('Emitted object correction differs from primitive')
    return target, native_params, corrected


def verify_emitted_witnesses(bank, src, human, trajectory, projection):
    surface = _ContactTriangleSurface(bank['vertices'], bank['faces']); gaps = []
    active = projection['activations']
    for frame, side in zip(*np.nonzero(active)):
        identifier = src['QA_witness_ids'][frame, side]
        local = (human[frame, identifier] - trajectory['object_translation'][frame]) @ trajectory['object_rotation'][frame]
        gap = float(surface.distances(local[None], batch_size=32)[0]); gaps.append(gap)
        if gap > projection['frozen_gap_bound_m'][frame, side]:
            raise ContactFeasibilityFailure(frame, 'actual FP64 full emitted geometry violates same-witness bound',
                gap - projection['frozen_gap_bound_m'][frame, side])
    return dict(active_witnesses=len(gaps), maximum_gap_m=max(gaps) if gaps else None,
        all_original_active_bounds_verified=True, whole_hand_minimum_claimed=False,
        physical_contact_claimed=False, geometry_dtype='float64', full_surface_retained=True)


def project_saved(bank, src, saved):
    active = saved['activation']; ids = src['QA_witness_ids']
    original = np.full((*active.shape, 3), np.nan, np.float64)
    f, s = np.nonzero(active); original[f, s] = src['human'][f, ids[f, s]]
    frozen = freeze_contact_witnesses(original, src['QA_hand_ids'], ids, bank['rotations'],
        bank['translations'], bank['vertices'], bank['faces'], active, bank['frame_index'],
        config=PLACEMENT, source_reference='independently_pinned_A052_and_B' + B_REVISION,
        witness_selection_reference=WITNESS_DEFINITION, original_selection_is_whole_hand_minimum=False)
    projection = project_joint_translations(frozen, np.asarray(saved['human'][:, src['QA_hand_ids']]),
        saved['params']['mhr_trans'], saved['trajectory']['object_translation'])
    target, params, trajectory = corrected_geometry(saved['human'], saved['params'], saved['trajectory'], projection)
    verification = verify_emitted_witnesses(bank, src, target, trajectory, projection)
    return target, params, trajectory, projection, verification


def unsupported(episode):
    return dict(episode=episode, status='unsupported_original_frontend_unchanged',
        baseline_retained=True, rerolled=False, fabricated_predictions=False,
        reason='original random full-T native human pose unavailable after upstream occlusion')


def run_episode(episode, out, ledger, binding):
    started = time.monotonic(); out.mkdir(mode=0o755)
    report = dict(episode=episode, status='fail', producer_revision=os.environ['WR_CODE_REVISION'],
        baseline_revision=real.old.SOURCE, B_revision=B_REVISION, ground_truth_used=False,
        private_truth_read=False, production_adopted=False, baseline_modified=False,
        GPU_requested=False, model_calls=0, new_native_fits=0, full_4D_accuracy_verified=False,
        direct204_C_export_available=False, direct_native_control_parity_claimed=False,
        MHR_translation_ABI='camera_metres_added_after_native_100cm_conversion_and_YZ_flip',
        whole_hand_minimum_claimed=False, physical_contact_verified=False,
        numerical_protocol=asdict(PLACEMENT), constraint_scope='same_preserved_original_QA_witness_only',
        silhouette_re_evaluated=False, penetration_evaluated=False,
        remaining_adoption_requirements=['native_decoder_replay', 'original_RGB_silhouette_QA',
            'full_hand_nonpenetration_QA', 'external_reference_validation'],
        penetration_not_run_reason='full512_hand_winding_cost_not_bounded_by_this_saved_witness_CPU_experiment')
    try:
        bank, src = native.original_sources(ledger, episode)
        if len(bank['frame_index']) != FRAMES[episode] or bank['fps'] != 30.:
            raise ValueError('Exact frozen original full415/442 frame cohort required')
        spec = real.saved.load_npz(ledger, ROOT / 'weights/cari4d/refinement/mhr_hand_surface_spec.npz', real.contact.HAND_PIN)
        src['QA_hand_ids'] = real.contact.hand_indices(spec, src['native']['human_faces'], src['human'].shape[1])
        saved = saved_B(ledger, episode, bank, src, binding)
        report['B_report_pin'] = B_REPORT_PINS[episode]
        report['baseline_binding'] = saved['report']['baseline_binding']
        report['B_source_outputs'] = saved['report']['candidate_outputs']
        activation = saved['activation']; evidence = saved['evidence']
        a = native.quality(bank, src, src['human'], src['native']['mhr_keypoints'],
            bank['rotations'], bank['translations'], evidence, activation)
        b = native.quality(bank, src, saved['human'], saved['params']['mhr_keypoints'],
            saved['trajectory']['object_rotation'], saved['trajectory']['object_translation'], evidence, activation)
        preserve_reported_quality(a, saved['report']['metrics'][A_NAME])
        preserve_reported_quality(b, saved['report']['metrics'][B_NAME])
        report['metrics'] = {A_NAME: a, B_NAME: b}
        report['A_B_QA_pin'] = real.seal_json(out / 'A_B_QA.json', dict(metrics=report['metrics'],
            B_report_pin=B_REPORT_PINS[episode], baseline_binding=report['baseline_binding']))
        print('CONTACT_FEASIBLE ' + json.dumps(dict(episode=episode, phase='A_B_QA_sealed',
            seconds=round(time.monotonic() - started, 3))), flush=True)
        target, params, trajectory, projection, verification = project_saved(bank, src, saved)
        report['candidate_outputs'] = {
            'target': real.seal_file(out / 'target.npy', lambda stream: np.save(stream, target, allow_pickle=False)),
            'trajectory': real.seal_file(out / 'trajectory.npz', lambda stream: np.savez_compressed(stream, **trajectory)),
            'native_parameters': real.seal_file(out / 'native_parameters.npz', lambda stream: np.savez_compressed(stream, **params)),
            'corrections': real.seal_file(out / 'corrections.npz', lambda stream:
                np.savez_compressed(stream, **{k: v for k, v in projection.items() if isinstance(v, np.ndarray)})),
        }
        report['projection'] = {k: v for k, v in projection.items() if not isinstance(v, np.ndarray)}
        report['emitted_geometry_verification'] = verification
        report.update(full_original_frames=len(target), source_fps=bank['fps'],
            original_native_active_hand_frames=int(activation.sum()), fitted_outputs_sealed_before_QA=True,
            C_native_translation_dtype=str(params['mhr_trans'].dtype), C_geometry_dtype=str(target.dtype),
            C_pose_controls_omitted_to_prevent_stale_B_replay=True,
            maximum_human_translation_correction_m=float(np.linalg.norm(projection['human_correction_camera'], axis=1).max()),
            maximum_object_translation_correction_m=float(np.linalg.norm(projection['object_correction_camera'], axis=1).max()),
            inactive_frames_unchanged=bool(np.all(projection['human_correction_camera'][~activation.any(1)] == 0)
                and np.all(projection['object_correction_camera'][~activation.any(1)] == 0)))
        real.seal_json(out / 'geometry.json', report)
        c = native.quality(bank, src, target, params['mhr_keypoints'], trajectory['object_rotation'],
            trajectory['object_translation'], evidence, activation)
        report['metrics'][C_NAME] = c
        report['decision'] = {'C_vs_A': native.quality_decision(saved['cfg'], a, c),
                              'C_vs_B': native.quality_decision(saved['cfg'], b, c)}
        report['status'] = 'complete_saved_real_A_B_C_diagnostic'
    except ContactFeasibilityFailure as exc:
        report.update(status='complete_A_B_C_projection_rejected', C_rejection=dict(
            error_type=type(exc).__name__, frame_index=exc.frame_index, reason=exc.reason,
            maximum_violation_m=exc.maximum_violation_m, physical_infeasibility_claimed=False),
            failed_C_predictions_not_substituted=True)
    except Exception as exc:
        report.update(error_type=type(exc).__name__, error=str(exc)[:400])
    finally:
        report['elapsed_seconds'] = time.monotonic() - started
        report['report_pin_scope'] = 'internal_automatic_observation_QA_not_truth_or_Kaggle_scores'
        report['report_pin'] = real.seal_json(out / 'report.json', report)
        print('CONTACT_FEASIBLE ' + json.dumps(dict(episode=episode, status=report['status'],
            seconds=round(report['elapsed_seconds'], 3))), flush=True)
    return report


def run():
    started = time.monotonic(); revision = os.environ['WR_CODE_REVISION']; code = Path(os.environ['WR_CODE'])
    if (Path(os.environ['WR_ROOT']) != ROOT or code != ROOT / 'jobs' / revision / ENTRY / 'code'
            or os.environ.get('WR_IMAGE_ID') != native.IMAGE or os.environ.get('CUDA_VISIBLE_DEVICES') != '-1'):
        raise ValueError('Exact immutable offline CPU source/image namespace required')
    out = canonical(ROOT / 'results' / ('contact-feasible-real-' + revision))
    real.old.fresh_runtime_output(out)
    report = dict(schema='world_reward.contact_feasible_real.v1', status='fail', producer_revision=revision,
        ground_truth_used=False, private_truth_read=False, production_adopted=False, baseline_modified=False,
        GPU_requested=False, model_calls=0, new_native_fits=0, budget_seconds=BUDGET,
        cohort=dict(random_seed=20261008, population=30, episodes=list(COHORT)), episodes=[])
    ledger = real.old.ArtifactLedger()
    try:
        report['source_binding'] = source(ROOT, code, revision, ENTRY, HELPERS)
        old_code = ROOT / 'jobs' / B_REVISION / native.ENTRY / 'code'
        binding = source(ROOT, old_code, B_REVISION, native.ENTRY, native.HELPERS)
        report['B_source_binding'] = binding
        for episode in COHORT:
            if episode not in FRAMES:
                row = unsupported(episode)
            elif time.monotonic() - started >= BUDGET - 1:
                row = dict(episode=episode, status='not_executed_budget_exhausted',
                    baseline_retained=True, rerolled=False, fabricated_predictions=False)
            else:
                row = run_episode(episode, out / f'episode_{episode:06d}', ledger, binding)
            report['episodes'].append(row)
        ledger.verify()
        report.update(status='complete_saved_real_contact_hypothesis', source_inputs_rehashed=True,
            input_ledger=ledger.records, complete_C_episodes=sum(r['status'] == 'complete_saved_real_A_B_C_diagnostic'
                for r in report['episodes']), unexpected_failures=sum(r['status'] == 'fail' for r in report['episodes']))
    except Exception as exc:
        report.update(error_type=type(exc).__name__, error=str(exc)[:400])
    finally:
        report['elapsed_seconds'] = time.monotonic() - started
        real.seal_json(out / 'report.json', report)
    return 0 if report['status'] == 'complete_saved_real_contact_hypothesis' and not report['unexpected_failures'] else 1


if __name__ == '__main__':
    def expired(*_): raise TimeoutError('Declared600second saved-real C budget exceeded')
    signal.signal(signal.SIGALRM, expired); signal.alarm(BUDGET)
    try: raise SystemExit(run())
    finally: signal.alarm(0)
