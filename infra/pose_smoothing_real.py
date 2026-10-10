"""Saved full-T rigid-pose reference ablation, CPU-only and never a submission.

Both proposals use the SAME frozen authored polynomial. Object-only smoothing
may break contacts. A common rigid scene correction preserves internal geometry
but may damage RGB evidence or the fixed virtual floor. Neither is adopted here.
"""
import json
import os
from pathlib import Path
import signal
import time

import numpy as np
from scipy.spatial.transform import Rotation

import contact_feasible_real as saved
from mediapipe_cpu_runtime_verify import canonical, source
from world_reward.pose_smoothing import local_chart_sg_reference
from world_reward.sequence_pose import _ContactTriangleSurface

native = saved.native
real = native.real
ROOT = native.ROOT
ENTRY = 'run_pose_smoothing_real'
BUDGET = 300
VARIANTS = ('object_only', 'common_SE3_diagnostic')
HELPERS = tuple(dict.fromkeys(('infra/pose_smoothing_real.py',
    'infra/run_pose_smoothing_real.sh', 'src/world_reward/pose_smoothing.py', *saved.HELPERS)))


def authored_DEV():
    """Runtime known moving/noisy geometry sanity BEFORE challenge input reads."""
    count = 121; seconds = np.arange(count) / 30.; centre = np.array([.4, -.2, .3])
    r = Rotation.from_rotvec(seconds[:, None] * np.array([.08, -.12, .2])).as_matrix()
    c = np.c_[seconds, .1 * seconds ** 2, 2 + .02 * seconds ** 3]
    noise = .02 * np.sin(np.arange(count) * 2 * np.pi / 3)
    noisy_r = Rotation.from_rotvec(noise[:, None] * [1., -.4, .7]).as_matrix() @ r
    noisy_c = c + noise[:, None] * [2., -1., .3]
    noisy_t = noisy_c - np.einsum('tij,j->ti', noisy_r, centre)
    output = local_chart_sg_reference(noisy_r, noisy_t, frame_indices=np.arange(count),
        fps=30., canonical_centroid=centre)
    recovered = np.einsum('tij,j->ti', output.rotation, centre) + output.translation
    interior = slice(4, -4)
    before = np.mean((noisy_c[interior] - c[interior]) ** 2)
    after = np.mean((recovered[interior] - c[interior]) ** 2)
    before_r = Rotation.from_matrix(noisy_r[interior] @ r[interior].transpose(0, 2, 1)).magnitude()
    after_r = Rotation.from_matrix(output.rotation[interior] @ r[interior].transpose(0, 2, 1)).magnitude()
    if (not after < .15 * before or not np.mean(after_r ** 2) < .15 * np.mean(before_r ** 2)
            or np.linalg.norm(recovered[-1] - recovered[0]) <= 3.
            or Rotation.from_matrix(output.rotation[-1] @ output.rotation[0].T).magnitude() <= .8
            or not np.array_equal(output.rotation[0], noisy_r[0])
            or not np.array_equal(output.translation[0], noisy_t[0])):
        raise ValueError('Authored moving/noise runtime DEV failed; no challenge reads allowed')
    return dict(passed=True, known_authored_geometry=True, challenge_ground_truth_used=False,
        centre_MSE_before=float(before), centre_MSE_after=float(after),
        angular_MSE_before=float(np.mean(before_r ** 2)), angular_MSE_after=float(np.mean(after_r ** 2)),
        full_frames=count, scope='numerical_sanity_not_HOI_calibration_or_accuracy_validation')


def proposals(bank, src):
    r, t = bank['rotations'], bank['translations']
    smoothed = local_chart_sg_reference(r, t, frame_indices=bank['frame_index'],
        fps=bank['fps'], canonical_centroid=bank['vertices'].astype(np.float64).mean(0))
    # Projection only derives a proper common correction; ORIGINAL near-SO3
    # predictions are not normalized independently or replaced by that projection.
    proper_a = Rotation.from_matrix(r).as_matrix()
    proper_s = Rotation.from_matrix(smoothed.rotation).as_matrix()
    correction = proper_s @ proper_a.transpose(0, 2, 1)
    shift = smoothed.translation - np.einsum('tij,tj->ti', correction, t)
    correction[0] = np.eye(3); shift[0] = 0.
    def move(value):
        moved = np.einsum('tij,tpj->tpi', correction, value) + shift[:, None]
        moved[0] = value[0]
        return moved
    common_r = correction @ r.astype(np.float64)
    common_t = np.einsum('tij,tj->ti', correction, t) + shift
    common_r[0], common_t[0] = r[0], t[0]
    return smoothed.diagnostics, {
        'object_only': dict(rotation=smoothed.rotation, translation=smoothed.translation,
            human=src['human'], joints=src['native']['mhr_joints'], keypoints=src['native']['mhr_keypoints']),
        'common_SE3_diagnostic': dict(rotation=common_r, translation=common_t,
            human=move(src['human']), joints=move(src['native']['mhr_joints']),
            keypoints=move(src['native']['mhr_keypoints']), common_rotation=correction, common_translation=shift),
    }


def contact_gaps(bank, src, human, r, t, active):
    """Exact full ORIGINAL triangles for every frozen SAME anatomical witness."""
    f, s = np.nonzero(active)
    ids = src['QA_witness_ids'][f, s]
    points = np.einsum('ti,tij->tj', human[f, ids].astype(np.float64) - t[f].astype(np.float64), r[f].astype(np.float64))
    gaps = _ContactTriangleSurface(bank['vertices'], bank['faces']).distances(points, batch_size=32)
    return dict(frame_index=f.astype(np.int64), hand=s.astype(np.int64), hand_vertex_ids=ids.copy(), gap_m=gaps)


def motion_and_floor(bank, human, r, t, floor):
    centre = np.einsum('tij,j->ti', r, bank['vertices'].astype(np.float64).mean(0)) + t
    increments = np.linalg.norm(np.diff(centre, axis=0), axis=1)
    angles = Rotation.from_matrix(r[1:] @ r[:-1].transpose(0, 2, 1)).magnitude()
    # Fixed inferred horizontal camera-Y plane only: NOT measured physical floor.
    lowest_object = np.array([(bank['vertices'] @ rotation.T + translation)[:, 1].max()
        for rotation, translation in zip(r, t)])
    human_lowest = np.max(human[..., 1], axis=1)
    return dict(centroid_displacement_m=float(np.linalg.norm(centre[-1] - centre[0])),
        centroid_path_length_m=float(increments.sum()), centroid_speed_p95_m_s=float(np.quantile(increments * bank['fps'], .95)),
        rotation_total_variation_rad=float(angles.sum()),
        rotation_net_change_rad=float(Rotation.from_matrix(r[-1] @ r[0].T).magnitude()),
        fixed_virtual_floor_camera_y_m=floor,
        virtual_human_below_floor_max_m=float(np.maximum(human_lowest - floor, 0).max()),
        virtual_object_below_floor_max_m=float(np.maximum(lowest_object - floor, 0).max()),
        frame_zero_preserved=True, full_frames=len(t), physical_floor_claimed=False,
        motion_retention_verified_against_truth=False)


def seal_proposal(out, bank, proposal):
    for value in (proposal['rotation'], proposal['translation'], proposal['human'], proposal['joints'], proposal['keypoints']):
        if not np.isfinite(value).all(): raise ValueError('Full finite proposed pose/geometry required before sealing')
    for value in (proposal['human'], proposal['joints'], proposal['keypoints']):
        if (value[..., 2] <= 0).any(): raise ValueError('Full original camera-positive proposed human geometry required')
    pose = dict(object_rotation=proposal['rotation'], object_translation=proposal['translation'],
        object_vertices=bank['vertices'], object_faces=bank['faces'], object_scale=bank['object_scale'],
        camera_K=bank['original_K'], frame_index=bank['frame_index'], fps=np.array(bank['fps']))
    if 'common_rotation' in proposal:
        pose.update(common_rotation=proposal['common_rotation'], common_translation=proposal['common_translation'])
    pins = dict(object_pose=real.seal_file(out / 'object_pose.npz', lambda stream: np.savez_compressed(stream, **pose)))
    if 'common_rotation' in proposal:
        pins['target'] = real.seal_file(out / 'target.npy', lambda stream: np.save(stream, proposal['human'], allow_pickle=False))
        pins['human_geometry'] = real.seal_file(out / 'human_geometry.npz', lambda stream: np.savez_compressed(stream,
            mhr_joints=proposal['joints'], mhr_keypoints=proposal['keypoints'], frame_index=bank['frame_index']))
    return pins


def run_episode(episode, out, ledger, binding, deadline):
    started = time.monotonic(); out.mkdir(mode=0o755)
    report = dict(episode=episode, status='fail', producer_revision=os.environ['WR_CODE_REVISION'],
        baseline_revision=real.old.SOURCE, ground_truth_used=False, private_truth_read=False,
        production_adopted=False, baseline_modified=False, model_calls=0, GPU_requested=False,
        new_native_fits=0, full_4D_accuracy_verified=False, native_control_export_available=False,
        native_control_replay_claimed=False, whole_hand_minimum_claimed=False,
        penetration_evaluated=False, silhouette_evaluated=False,
        penetration_not_run_reason='public_PEN_requires_separate_qualified_bounded_runtime_not_this_saved_witness_ablation',
        variants={})
    def check():
        if time.monotonic() >= deadline: raise TimeoutError('Full-cohort saved smoothing budget exhausted')
    try:
        check(); bank, src = native.original_sources(ledger, episode)
        if len(bank['frame_index']) != saved.FRAMES[episode] or bank['fps'] != 30.:
            raise ValueError('Original complete 415/442 frames and 30fps required')
        spec = real.saved.load_npz(ledger, ROOT / 'weights/cari4d/refinement/mhr_hand_surface_spec.npz', real.contact.HAND_PIN)
        src['QA_hand_ids'] = real.contact.hand_indices(spec, src['native']['human_faces'], 18439)
        b = saved.saved_B(ledger, episode, bank, src, binding)
        report.update(baseline_binding=b['report']['baseline_binding'], B_report_pin=saved.B_REPORT_PINS[episode],
            frozen_witness_definition=saved.WITNESS_DEFINITION, full_original_frames=len(bank['frame_index']), source_fps=bank['fps'])
        check()
        qa_a = native.quality(bank, src, src['human'], src['native']['mhr_keypoints'], bank['rotations'], bank['translations'], b['evidence'], b['activation'])
        qa_b = native.quality(bank, src, b['human'], b['params']['mhr_keypoints'], b['trajectory']['object_rotation'], b['trajectory']['object_translation'], b['evidence'], b['activation'])
        saved.preserve_reported_quality(qa_a, b['report']['metrics'][saved.A_NAME])
        saved.preserve_reported_quality(qa_b, b['report']['metrics'][saved.B_NAME])
        original_gaps = contact_gaps(bank, src, src['human'], bank['rotations'], bank['translations'], b['activation'])
        floor = float(np.median(np.max(src['human'][..., 1], axis=1)))
        original_motion = motion_and_floor(bank, src['human'], bank['rotations'], bank['translations'], floor)
        report['A_B_QA_pin'] = real.seal_json(out / 'A_B_QA.json', dict(metrics={saved.A_NAME: qa_a, saved.B_NAME: qa_b},
            original_motion=original_motion))
        check(); method, proposed = proposals(bank, src); report['method'] = method
        # Both proposals are frozen before ANY proposal QA, never selected by score.
        for name, value in proposed.items():
            check(); folder = out / name; folder.mkdir(mode=0o755)
            report['variants'][name] = dict(status='geometry_sealed_QA_pending', candidate_outputs=seal_proposal(folder, bank, value),
                human_source=(report['baseline_binding'] if name == 'object_only' else 'same_rigid_full_saved_A_geometry_transform'),
                source_topology_unchanged=True, shape_scale_unchanged=True, native_controls_emitted=False,
                all_original_frame_indices_preserved=True, per_frame_evaluation_alignment=False)
            real.seal_json(folder / 'geometry.json', report['variants'][name])
        for name, value in proposed.items():
            check(); row = report['variants'][name]
            metrics = native.quality(bank, src, value['human'], value['keypoints'], value['rotation'], value['translation'], b['evidence'], b['activation'])
            gaps = contact_gaps(bank, src, value['human'], value['rotation'], value['translation'], b['activation'])
            row.update(status='complete_saved_pose_diagnostic', metrics=metrics,
                decision_vs_A=native.quality_decision(b['cfg'], qa_a, metrics),
                motion=motion_and_floor(bank, value['human'], value['rotation'], value['translation'], floor),
                same_witness_all_frames_nonworse=bool(np.all(gaps['gap_m'] <= original_gaps['gap_m'] * 1.00000001 + 1e-12)),
                maximum_same_witness_gap_increase_m=float((gaps['gap_m'] - original_gaps['gap_m']).max(initial=0.)),
                production_adopted=False, GT_accuracy_verified=False)
            row['additional_diagnostic_gates'] = dict(
                same_witness_all_frames_nonworse=row['same_witness_all_frames_nonworse'],
                virtual_human_floor_nonworse=row['motion']['virtual_human_below_floor_max_m'] <= original_motion['virtual_human_below_floor_max_m'] + 1e-12,
                virtual_object_floor_nonworse=row['motion']['virtual_object_below_floor_max_m'] <= original_motion['virtual_object_below_floor_max_m'] + 1e-12)
            # Net/path motion is shown, not a calibrated acceptance criterion.
            # Native eight-metric QA alone NEVER labels this an overall success.
            row['motion_retention_validation_pending'] = True
            row['overall_adoption_clearance'] = False
            row['witness_pin'] = real.seal_file(out / name / 'witness_gaps.npz', lambda stream: np.savez_compressed(stream, **gaps))
            row['QA_pin'] = real.seal_json(out / name / 'QA.json', row)
            print('POSE_SMOOTHING ' + json.dumps(dict(episode=episode, variant=name, status=row['status'],
                native_metric_QA_pass=row['decision_vs_A']['passed'],
                additional_diagnostic_gates=row['additional_diagnostic_gates'],
                overall_adoption_clearance=False, seconds=round(time.monotonic() - started, 3))), flush=True)
        report['status'] = 'complete_saved_full_T_pose_smoothing_diagnostic'
    except Exception as exc:
        report.update(error_type=type(exc).__name__, error=str(exc)[:400])
    report['elapsed_seconds'] = time.monotonic() - started
    report['report_pin'] = real.seal_json(out / 'report.json', report)
    return report


def run():
    started = time.monotonic(); revision = os.environ['WR_CODE_REVISION']; code = Path(os.environ['WR_CODE'])
    if (Path(os.environ['WR_ROOT']) != ROOT or code != ROOT / 'jobs' / revision / ENTRY / 'code'
            or os.environ.get('WR_IMAGE_ID') != native.IMAGE or os.environ.get('CUDA_VISIBLE_DEVICES') != '-1'):
        raise ValueError('Exact immutable offline CPU-only source/image namespace required')
    out = canonical(ROOT / 'results' / ('pose-smoothing-real-' + revision)); real.old.fresh_runtime_output(out)
    report = dict(schema='world_reward.pose_smoothing_real.v1', status='fail', producer_revision=revision,
        budget_seconds=BUDGET, cohort=list(saved.COHORT), ground_truth_used=False, private_truth_read=False,
        production_adopted=False, baseline_modified=False, model_calls=0, GPU_requested=False, episodes=[])
    ledger = real.old.ArtifactLedger()
    def timeout(*_): raise TimeoutError('Inclusive saved smoothing 300-second runtime budget exhausted')
    previous = signal.signal(signal.SIGALRM, timeout); signal.setitimer(signal.ITIMER_REAL, BUDGET)
    try:
        report['source_binding'] = source(ROOT, code, revision, ENTRY, HELPERS)
        report['authored_DEV'] = authored_DEV()
        binding = source(ROOT, ROOT / 'jobs' / saved.B_REVISION / native.ENTRY / 'code', saved.B_REVISION, native.ENTRY, native.HELPERS)
        report['B_source_binding'] = binding
        for episode in saved.COHORT:
            row = (run_episode(episode, out / f'episode_{episode:06d}', ledger, binding, started + BUDGET)
                if episode in saved.FRAMES else saved.unsupported(episode))
            report['episodes'].append(row)
        if time.monotonic() >= started + BUDGET:
            raise TimeoutError('Saved outputs retained; inclusive budget expired before source rehash')
        ledger.verify()
        if source(ROOT, code, revision, ENTRY, HELPERS) != report['source_binding']:
            raise ValueError('Own immutable source closure changed')
        report.update(status='complete_saved_pose_smoothing_ablation', input_ledger=ledger.records,
            source_inputs_rehashed=True, complete_episodes=sum(row['status'] == 'complete_saved_full_T_pose_smoothing_diagnostic' for row in report['episodes']),
            unexpected_failures=sum(row['status'] == 'fail' for row in report['episodes']))
    except Exception as exc:
        report.update(error_type=type(exc).__name__, error=str(exc)[:400])
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0); signal.signal(signal.SIGALRM, previous)
        report['elapsed_seconds'] = time.monotonic() - started; real.seal_json(out / 'report.json', report)
    print(json.dumps({k: report.get(k) for k in ('status', 'complete_episodes', 'unexpected_failures', 'elapsed_seconds', 'error')}, allow_nan=False), flush=True)
    if report['status'] == 'fail' or report['unexpected_failures']: raise SystemExit(1)


if __name__ == '__main__':
    if len(os.sys.argv) != 1: raise SystemExit('No arbitrary inputs, models or labels supported')
    run()
