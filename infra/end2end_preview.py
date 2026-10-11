"""Full-timeline RGB / frozen baseline / native candidate diagnostic on Azure.

The candidate may fail QA: that failure is displayed, never hidden. This is not
an official metric evaluation, a new reconstruction, or manual test annotation.
"""
from io import BytesIO
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

from mediapipe_cpu_runtime_verify import canonical, identity, source, strict, require

ROOT = Path('/srv/scenesmith/world-reward')
BASELINE = '052ba1554e9a573d566713a99a61d89a5f27681c'
ENTRY = 'run_end2end_preview'
COHORT = [9, 1, 14, 7]
HELPERS = ('infra/end2end_preview.py', 'infra/run_end2end_preview.sh',
    'infra/full4d_video.py', 'infra/full4d_publish.py',
    'infra/mediapipe_cpu_runtime_verify.py')
FILES = {'target': 'target.npy', 'trajectory': 'trajectory.npz',
         'native_parameters': 'native_parameters.npz'}
KINDS = {'joint': 'native-joint-real-', 'continuation': 'native-contact-continuation-real-', 'sqp': 'native-pose-sqp-real-', 'smoothing': 'pose-smoothing-real-'}
SMOOTHING_STATUSES = {'accepted_rigid_projection_pending_independent_QA', 'dynamic_A_fallback_no_improvement'}
CONTINUATION_ENTRY = 'run_native_contact_continuation_real'
CONTINUATION_STATUSES = {'accepted_native_continuation', 'dynamic_A_fallback_no_improvement'}
SQP_STATUSES = {'accepted_native_pose_sqp', 'dynamic_A_fallback_no_improvement'}
NATIVE_METHODS = {
    'continuation': (CONTINUATION_ENTRY, 'world_reward.native_contact_continuation_real.v1',
        'complete_native_continuation_diagnostic', CONTINUATION_STATUSES, 'native_contact_continuation'),
    'sqp': ('run_native_pose_sqp_real', 'world_reward.native_pose_sqp_real.v1',
        'complete_native_pose_sqp_diagnostic', SQP_STATUSES, 'native_pose_sqp')}
FROZEN_NATIVE_KEYS = ('mhr_global_rot6d', 'mhr_shape', 'mhr_scale', 'mhr_hand', 'mhr_face')
DISPLAY_TRAJECTORY_KEYS = ('object_vertices', 'object_faces', 'object_rotation',
    'object_translation', 'object_scale', 'camera_K', 'frame_index')


def revision(value):
    require(type(value) is str and re.fullmatch('[0-9a-f]{40}', value), 'Exact revision required')
    return value


def candidate_kind(value):
    require(type(value) is str and value in KINDS, 'Explicit known diagnostic candidate required')
    return value


def display_trajectory(trajectory):
    """Project native export controls onto the strict saved-geometry viewer ABI.

    Native pose/shape/scale controls remain available for the separate producer
    checks. The viewer accepts exactly seven geometric fields; return the same
    array objects without repairing, casting, retiming or rescaling anything.
    """
    require(set(DISPLAY_TRAJECTORY_KEYS) <= set(trajectory), 'Complete saved geometry fields required')
    return {key: trajectory[key] for key in DISPLAY_TRAJECTORY_KEYS}


def labels(episode, accepted, kind='joint', status=None):
    candidate_kind(kind)
    title = 'Native joint — QA ' + ('PASS, GT pending' if accepted else 'FAIL / not adopted')
    if kind == 'smoothing':
        require(status in SMOOTHING_STATUSES, 'Actual rigid smoothing outcome required')
        title = ('Baseline fallback / no gain' if status == 'dynamic_A_fallback_no_improvement'
                 else 'Lissage + contacts — diagnostic / GT pending')
    if kind in NATIVE_METHODS:
        require(status in NATIVE_METHODS[kind][3], 'Original native method outcome required')
        title = ('Baseline fallback / no gain' if status == 'dynamic_A_fallback_no_improvement' else
                 ('C sparse pose-SQP' if kind == 'sqp' else 'C contact-continuation') +
                 ' — QA ' + ('PASS, GT pending' if accepted else 'FAIL / not adopted'))
    return ['Original RGB', 'Baseline complete (052)',
            title]


def smoothing_cohort(ledger, candidate_root, candidate_revision):
    """Validate the actual CPU producer, never relabel it as native joint fitting."""
    r = strict(ledger.read(candidate_root/'report.json', maximum=4<<20))
    require(r.get('schema') == 'world_reward.pose_smoothing_real.v1'
        and r.get('status') == 'complete_saved_pose_smoothing_ablation'
        and r.get('producer_revision') == candidate_revision and r.get('cohort') == COHORT
        and r.get('complete_episodes') == 2 and r.get('unexpected_failures') == 0
        and r.get('contact_projection_enabled') is True and r.get('source_inputs_rehashed') is True
        and all(r.get(k) is False for k in ('ground_truth_used','private_truth_read','production_adopted','baseline_modified','GPU_requested'))
        and r.get('model_calls') == 0 and r.get('projection_authored_DEV',{}).get('passed') is True,
        'Complete sealed no-GT CPU smoothing source required')
    rows = r.get('episodes',[])
    require([v.get('episode') for v in rows] == COHORT
        and all(v.get('status') == ('complete_saved_full_T_pose_smoothing_diagnostic' if v['episode'] in (9,14)
            else 'unsupported_original_frontend_unchanged') for v in rows), 'Original four-record denominator required')
    binding = r['source_binding']; helpers = binding.get('helpers',{})
    require({'infra/pose_smoothing_real.py','infra/run_pose_smoothing_real.sh',
        'src/world_reward/pose_smoothing.py','src/world_reward/rigid_pose_contact_projection.py'} <= set(helpers)
        and source(ROOT,ROOT/'jobs'/candidate_revision/'run_pose_smoothing_real'/'code',candidate_revision,
            'run_pose_smoothing_real',tuple(helpers)) == binding, 'Actual immutable smoothing source closure required')
    host = strict(ledger.read(candidate_root/'host-exit.json',maximum=4096))
    require(host == dict(producer_revision=candidate_revision,container_absence_verified=True,
        process_exit_code=0,GPU_requested=False) and type(host['process_exit_code']) is int,
        'Actual successful CPU producer cleanup required')
    require(type(r.get('input_ledger')) is dict and r['input_ledger'], 'Smoothing original input ledger required')
    return r


def smoothing_arrays(ledger, candidate, r, directory, pins, total):
    """Replay frozen object pose and ORIGINAL human; never refit/repair geometry."""
    import numpy as np
    v = r['variants']['contact_constrained_SE3']; c = v['projection']
    require(r.get('full_original_frames') == total and r.get('source_fps') == 30.
        and r.get('baseline_binding',{}).get('outputs') == pins['export_files']
        and v.get('status') == 'complete_saved_pose_diagnostic'
        and c.get('status') in SMOOTHING_STATUSES and c.get('full_original_frames') == total
        and c.get('human_fixed') is True and c.get('exact_full_surface_witness_bounds_preserved') is True
        and c.get('object_geometry_scale_changed') is False and c.get('first_pose_exactly_preserved') is True
        and c.get('ground_truth_used') is False and c.get('production_adopted') is False
        and v.get('production_adopted') is False and v.get('overall_adoption_clearance') is False
        and v.get('shape_scale_unchanged') is True and v.get('native_controls_emitted') is False,
        'Full frozen same-human pose diagnostic, not validated native control export')
    folder=candidate/'contact_constrained_SE3'
    geometry=strict(ledger.read(folder/'geometry.json'))
    require(geometry.get('projection') == c and geometry.get('candidate_outputs') == v['candidate_outputs'],
        'Candidate geometry was sealed before independent QA')
    for name in FILES.values(): ledger.record(directory/name,pins['export_files'][name])
    ah=np.load(directory/'target.npy',mmap_mode='r',allow_pickle=False)
    with np.load(directory/'trajectory.npz',allow_pickle=False) as z:
        a={k:z[k] for k in DISPLAY_TRAJECTORY_KEYS}
    with np.load(directory/'native_parameters.npz',allow_pickle=False) as z:
        faces=z['human_faces'];indices=z['frame_index']
    ledger.record(folder/'object_pose.npz',v['candidate_outputs']['object_pose'])
    with np.load(folder/'object_pose.npz',allow_pickle=False) as z:
        require(set(z.files) == set(DISPLAY_TRAJECTORY_KEYS)|{'fps'} and float(z['fps']) == 30.,
            'Exact complete rigid smoothing artifact required')
        b={k:z[k] for k in DISPLAY_TRAJECTORY_KEYS}
    for k in ('object_vertices','object_faces','object_scale','camera_K','frame_index'):
        require(np.array_equal(a[k],b[k]), 'No geometry, scale, camera or chronology edits')
    require(np.array_equal(a['object_rotation'][0],b['object_rotation'][0])
        and np.array_equal(a['object_translation'][0],b['object_translation'][0]),'Original first pose preserved')
    ledger.record(folder/'frozen_contact_bounds.npz',v['candidate_outputs']['frozen_contact_bounds'])
    with np.load(folder/'frozen_contact_bounds.npz',allow_pickle=False) as z:
        active=z['activations'];gap=z['projected_gaps_m'];base=z['baseline_gaps_m']
        require(active.dtype == np.bool_ and active.shape == gap.shape == base.shape == (total,2)
            and np.array_equal(z['frame_index'],np.arange(total)) and z['hand_vertex_ids'].shape == (total,2)
            and np.isfinite(gap[active]).all() and np.all(gap[active] <= base[active]+1e-7),
            'All original contact bounds, including occluded chronology, required')
    require(type(v['decision_vs_A']['passed']) is bool,'Actual diagnostic QA required')
    return ah,ah,a,b,faces,indices,v['decision_vs_A']['passed'],c['status']


def native_cohort(ledger, candidate_root, candidate_revision, kind):
    """Authenticate the complete native producer and all four retained statuses."""
    entry, schema, status, outcomes, module = NATIVE_METHODS[kind]
    path = candidate_root/'report.json'; identity(path, 4<<20)
    r = strict(ledger.read(path, maximum=4<<20))
    require(r.get('schema') == schema
        and r.get('status') == status
        and r.get('producer_revision') == candidate_revision and r.get('source_inputs_rehashed') is True
        and r.get('unexpected_failures') == 0 and r.get('native_layers_loaded') == 1
        and r.get('ground_truth_used') is False and r.get('private_truth_read') is False
        and r.get('baseline_modified') is False and r.get('production_adopted') is False
        and r.get('cohort') == dict(random_seed=20261008, population=30, episodes=COHORT),
        'Complete sealed no-GT native cohort required')
    rows = r.get('episodes', [])
    require(type(rows) is list and all(type(v) is dict for v in rows)
        and [v.get('episode') for v in rows] == COHORT,
        'All original native cohort statuses required')
    for row in rows:
        require(row.get('status') in (outcomes if row['episode'] in (9,14) else
            {'unsupported_original_frontend_unchanged'}), 'No failed/sampled native cohort records')
        if row['episode'] in (1,7):
            require(row.get('baseline_retained') is True and row.get('rerolled') is False
                and row.get('fabricated_predictions') is False and type(row.get('reason')) is str,
                'Unsupported records retain original baseline without reroll')
    oldcode = ROOT/'jobs'/candidate_revision/entry/'code'
    binding = r['source_binding']; helpers = binding.get('helpers', {})
    require({f'infra/{module}_real.py', f'infra/run_{module}_real.sh',
        f'src/world_reward/{module}.py'} <= set(helpers)
        and source(ROOT, oldcode, candidate_revision, entry, tuple(helpers)) == binding,
        'Original immutable native method source binding required')
    host = candidate_root/'host-exit.json'; identity(host, 4096)
    exit_receipt = strict(ledger.read(host, maximum=4096))
    require(exit_receipt == dict(producer_revision=candidate_revision, container_absence_verified=True,
        process_exit_code=0, GPU_requested=True), 'Actual completed owned native GPU cleanup required')
    require(type(r.get('input_ledger')) is dict and r['input_ledger'], 'Full producer source ledger required')
    return r


def continuation_cohort(ledger, candidate_root, candidate_revision):
    return native_cohort(ledger, candidate_root, candidate_revision, 'continuation')


def complete_candidate(r, kind, total):
    """Method-specific completion gates; historical joint ABI remains unchanged."""
    if kind == 'joint':
        require(r.get('native_effective_updates') == 301 and r.get('fitted_outputs_sealed_before_QA') is True
            and r.get('source_inputs_rehashed') is True, 'Complete sealed native candidate required')
        return r['decision']['passed']
    if kind == 'sqp': return complete_sqp(r, total)
    require(r.get('status') in CONTINUATION_STATUSES
        and r.get('fitted_outputs_sealed_before_final_QA') is True
        and r.get('full_original_frames') == total and r.get('source_fps') == 30.
        and r.get('native_direct136_generated') is True and r.get('stale_B_pose_used') is False
        and r.get('original_activations_unchanged') is True and r.get('whole_hand_minimum_claimed') is False
        and r.get('penetration_evaluated') is False and r.get('hand_labeled_test') is False
        and r.get('oracle_modes') == [], 'Complete full-native same-witness continuation required')
    c = r.get('continuation', {})
    require(c.get('status') == r['status'] and c.get('full_original_frames') == total
        and c.get('clip_global_alpha') is True and c.get('ground_truth_used') is False
        and c.get('private_truth_read') is False and c.get('metric_improvement_claimed') is False
        and ((r['status'] == 'dynamic_A_fallback_no_improvement' and c.get('alpha') == 0.) or
             (r['status'] == 'accepted_native_continuation' and type(c.get('alpha')) in (int,float)
              and 0 < c['alpha'] <= 1)), 'Original truthful continuation/fallback outcome required')
    require(type(r.get('decision', {}).get('C_vs_A', {}).get('passed')) is bool,
        'Actual final C versus original A QA decision required')
    return r['decision']['C_vs_A']['passed']


def complete_sqp(r, total):
    """Validate actual full-native result, never infer gain from QA or fallback."""
    require(r.get('status') in SQP_STATUSES and r.get('fitted_outputs_sealed_before_final_QA') is True
        and r.get('full_original_frames') == total and r.get('source_fps') == 30.
        and r.get('native_direct136_generated') is True and r.get('stale_B_pose_used') is False
        and r.get('whole_hand_minimum_claimed') is False and r.get('penetration_evaluated') is False
        and r.get('physical_contact_verified') is False and r.get('hand_labeled_test') is False
        and r.get('oracle_modes') == [] and r.get('original_raw_bundle_byte_fingerprint_unchanged') is True,
        'Complete sealed no-GT full-native sparse SQP required')
    c=r.get('SQP',{}); protocol=r.get('native_objective_protocol',{})
    before,after=c.get('objective_before'),c.get('objective_after');steps=c.get('accepted_steps')
    require(c.get('schema') == 'world_reward.native_pose_sqp.v1' and c.get('status') == r['status']
        and c.get('full_original_frames') == total and c.get('ground_truth_used') is False
        and c.get('private_truth_read') is False and c.get('heldout_accuracy_verified') is False
        and c.get('whole_hand_minimum_claimed') is False and c.get('physical_contact_verified') is False
        and c.get('production_adopted') is False and c.get('baseline_fallback_dynamic') is True
        and type(steps) is int and 0 <= steps <= 20
        and type(before) in (int,float) and type(after) in (int,float) and 0 <= after <= before
        and ((r['status'] == 'accepted_native_pose_sqp' and steps > 0 and after < before) or
             (r['status'] == 'dynamic_A_fallback_no_improvement' and steps == 0 and after == before))
        and protocol.get('objective_stage') == 181 and protocol.get('scheduled_PEN_and_silhouette_retained') is True
        and protocol.get('raw_priors_rebased_on_baseline_or_candidate') is False,
        'Truthful original sparse native objective/fallback outcome required')
    require(protocol.get('source_binding',{}).get('layer') == dict(bytes=20514,
        sha256='a753ab8e730b6730fca275384fab629859311983292a407390d88c66ffe68c23')
        and protocol.get('source_binding',{}).get('optimizer') == dict(bytes=92824,
        sha256='84e0e818a3bc0935bb30b75fcd82fd7c5e3730ed812864594cd759697ddb406b'),
        'Published original native decoder/objective source pins required')
    require(type(r.get('decision',{}).get('C_vs_A',{}).get('passed')) is bool,
        'Actual final SQP versus original A QA decision required')
    return r['decision']['C_vs_A']['passed']


def probe_contract(probe, total):
    streams = probe.get('streams', [])
    require(len(streams) == 1, 'Exactly one encoded video stream required')
    s = streams[0]
    require(s.get('codec_name') == 'h264' and s.get('width') == 960 and s.get('height') == 280
        and s.get('r_frame_rate') == '30/1' and s.get('avg_frame_rate') == '30/1'
        and s.get('nb_read_frames') == str(total), 'Full original timeline / fixed viewports required')


def render(candidate_revision, kind='joint'):
    import numpy as np
    import cv2
    from PIL import Image, ImageDraw, ImageFont
    import full4d_video as viewer
    from sequence_pose_probe import ArtifactLedger

    started = time.monotonic(); rev = revision(os.environ['WR_CODE_REVISION'])
    code = canonical(os.environ['WR_CODE']); candidate_revision = revision(candidate_revision); candidate_kind(kind)
    require(code == ROOT/'jobs'/rev/ENTRY/'code', 'Immutable render closure required')
    binding = source(ROOT, code, rev, ENTRY, HELPERS)
    output = ROOT/'results'/('end2end-preview-'+rev)
    require(output.is_dir() and set(p.name for p in output.iterdir()) == {'.container.cid'},
            'Fresh owned rendering directory required')
    ledger = ArtifactLedger(); reports = []; states = []
    base = ROOT/'experiments'/('full4d-v1-'+BASELINE)
    candidate_root = ROOT/'results'/(KINDS[kind]+candidate_revision)
    cohort = (smoothing_cohort(ledger, candidate_root, candidate_revision) if kind == 'smoothing' else
        native_cohort(ledger, candidate_root, candidate_revision, kind) if kind in NATIVE_METHODS else None)
    cohort_rows = {r['episode']: r for r in cohort['episodes']} if cohort else {}
    for ep in COHORT:
        candidate = candidate_root/f'episode_{ep:06d}'
        report_path = candidate/'report.json'
        if cohort and ep in (1,7):
            states.append(dict(episode=ep, status='not_reconstructed_in_this_ablation',
                reason=cohort_rows[ep]['reason']))
            continue
        if not report_path.exists():
            states.append(dict(episode=ep, status='not_reconstructed_in_this_ablation',
                reason='upstream full-pose unsupported' if ep in (1,7) else 'candidate missing'))
            continue
        if cohort: identity(report_path, 4<<20)
        r = strict(ledger.read(report_path, cohort_rows[ep]['report_pin' if kind == 'smoothing' else 'report'] if cohort else None, maximum=4<<20))
        require(r['producer_revision'] == candidate_revision and r['episode'] == ep
                and r['ground_truth_used'] is False and r['private_truth_read'] is False
                and r['baseline_modified'] is False, 'Automatic no-GT candidate provenance required')
        if r.get('status') not in ({'complete_saved_full_T_pose_smoothing_diagnostic'} if kind == 'smoothing'
                else NATIVE_METHODS[kind][3] if cohort else {'complete_diagnostic_not_quality_pass'}):
            states.append(dict(episode=ep, status='failed', phase=r.get('phase'),
                reason=r.get('error', 'no complete candidate geometry')))
            continue
        directory = base/'outputs'/f'episode_{ep:06d}'/'cari_shared_export_v1'
        pins = strict(ledger.read(base/'pins'/f'cari_clip_{ep:06d}_shared_export_pins.json'))
        spec = pins['clip_spec']; total = spec['total_frames']
        accepted = None if kind == 'smoothing' else complete_candidate(r, kind, total)
        if cohort:
            require(total == {9:415,14:442}[ep] and r['status'] == cohort_rows[ep]['status'],
                'Exact full original continuation timeline/outcome required')
        require(spec == dict(episode_index=ep, total_frames=total, camera_name='front_stereo_camera_left',
            width=1536, height=1152), 'Original source grid required')
        candidate_status = r['status']
        if kind == 'smoothing':
            ah,bh,a,b,faces,indices,accepted,candidate_status = smoothing_arrays(ledger,candidate,r,directory,pins,total)
        else:
            for name in FILES.values(): ledger.record(directory/name, pins['export_files'][name])
            for key, name in FILES.items():
                if cohort: identity(candidate/name, 2<<30)
                ledger.record(candidate/name, r['candidate_outputs'][key])
            require(r['baseline_binding']['outputs'] == pins['export_files'], 'Same frozen baseline required')
            if cohort:
                require(r['baseline_binding']['directory'] == str(directory), 'Same original baseline route required')
                if kind == 'sqp':
                    geometry=candidate/'geometry.json'; identity(geometry,4<<20)
                    sealed=strict(ledger.read(geometry,maximum=4<<20))
                    require(sealed.get('candidate_outputs') == r['candidate_outputs']
                        and sealed.get('producer_revision') == candidate_revision and sealed.get('episode') == ep
                        and sealed.get('baseline_binding') == r['baseline_binding'] and sealed.get('SQP') == r['SQP']
                        and sealed.get('ground_truth_used') is False and sealed.get('private_truth_read') is False
                        and sealed.get('fitted_outputs_sealed_before_final_QA') is True,
                        'Original SQP geometry receipt sealed before final QA required')
                witness = candidate/'QA_witnesses.npz'; identity(witness, 2<<20)
                ledger.record(witness, r['candidate_outputs']['frozen_witnesses'])
                with np.load(witness, allow_pickle=False) as qa:
                    require(set(qa.files) == {'activations','hand_vertex_ids','frame_index','emitted_same_witness_gaps_m'}
                        and qa['activations'].dtype == np.bool_ and qa['activations'].shape == (total,2)
                        and qa['hand_vertex_ids'].dtype == np.int64 and qa['hand_vertex_ids'].shape == (total,2)
                        and np.array_equal(qa['frame_index'], np.arange(total))
                        and qa['emitted_same_witness_gaps_m'].shape == (total,2)
                        and np.isfinite(qa['emitted_same_witness_gaps_m'][qa['activations']]).all(),
                        'Sealed original all-frame automatic witness geometry required')
            ah = np.load(directory/'target.npy', mmap_mode='r', allow_pickle=False)
            bh = np.load(candidate/'target.npy', mmap_mode='r', allow_pickle=False)
            with np.load(directory/'trajectory.npz', allow_pickle=False) as a:
                keys = ('object_vertices','object_faces','object_rotation','object_translation',
                        'object_scale','camera_K','frame_index') + (('pose','scales','shape') if cohort else ())
                a = {k:a[k] for k in keys}
            with np.load(candidate/'trajectory.npz', allow_pickle=False) as b:
                b = {k:b[k] for k in a}
            with np.load(directory/'native_parameters.npz', allow_pickle=False) as native:
                faces = native['human_faces']; indices = native['frame_index']
                fixed_keys = FROZEN_NATIVE_KEYS
                frozen = {k:native[k] for k in fixed_keys} if cohort else {}
                internal = native['mhr_body_pose_cont'][:,254:].copy() if cohort else None
            with np.load(candidate/'native_parameters.npz', allow_pickle=False) as native:
                require(np.array_equal(native['human_faces'], faces)
                    and np.array_equal(native['frame_index'], indices), 'No topology / timeline substitutions')
                if cohort:
                    require(all(native[k].dtype == frozen[k].dtype and native[k].shape == frozen[k].shape
                        and native[k].tobytes() == frozen[k].tobytes() for k in fixed_keys)
                        and native['mhr_body_pose_cont'][:,254:].tobytes() == internal.tobytes()
                        and native['mhr_trans'].dtype == np.float32 and native['mhr_trans'].shape == (total,3)
                        and native['mhr_joints'].dtype == np.float32 and native['mhr_joints'].shape == (total,127,3)
                        and native['mhr_keypoints'].dtype == np.float32 and native['mhr_keypoints'].shape == (total,70,3),
                        'Frozen native identity, hands, root rotation and complete native C geometry required')
            require(ah.shape == bh.shape == (total,18439,3), 'Every original human vertex/frame required')
            if cohort:
                require(ah.dtype == bh.dtype == np.float32 and b['pose'].dtype == np.float32
                    and b['pose'].shape == (total,136) and np.isfinite(b['pose']).all()
                    and b['scales'].tobytes() == a['scales'].tobytes()
                    and b['shape'].tobytes() == a['shape'].tobytes()
                    and np.array_equal(a['object_rotation'], b['object_rotation']),
                    'Direct full-native C controls and unchanged clip shape/scales/object rotations required')
            for key in ('object_vertices','object_faces','object_scale','camera_K','frame_index'):
                require(np.array_equal(a[key], b[key]), 'Clip-constant geometry, K, scale and original indices required')
        viewer.checked_geometry(ah, faces, display_trajectory(a), indices, total)
        viewer.checked_geometry(bh, faces, display_trajectory(b), indices, total)
        camera = viewer.display_intrinsics(a['camera_K'], 1536, 1152)
        floor = viewer.fixed_floor_height(ah)
        ra = viewer.SavedSceneRenderer(camera, faces, a['object_faces'], floor)
        rb = viewer.SavedSceneRenderer(camera, faces, b['object_faces'], floor)
        video = ROOT/'data/track_1/videos/chunk-000/observation.images.exo_camera'/f'episode_{ep:06d}.mp4'
        input_ledger = cohort['input_ledger'] if cohort else r['input_ledger']
        require(str(video) in input_ledger, 'Source RGB must be bound by native candidate')
        ledger.record(video, input_ledger[str(video)])
        capture = cv2.VideoCapture(str(video))
        require(capture.isOpened() and int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == total
            and abs(capture.get(cv2.CAP_PROP_FPS)-30) < .003, 'Original RGB timeline required')
        template = Image.new('RGB', (960,280), (24,27,33))
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 10)
        draw = ImageDraw.Draw(template)
        for column, title in enumerate(labels(ep, accepted, kind, candidate_status)):
            draw.text((column*320+4,3), title, font=font, fill='white')
        draw.text((324,263), 'Same inferred scale / fixed virtual floor / grid 0.5m', font=font, fill='white')
        destination = output/f'episode_{ep:06d}.mp4'
        executable, probe = shutil.which('ffmpeg'), shutil.which('ffprobe')
        require(executable and probe, 'Existing ffmpeg/ffprobe required')
        ceiling = min(900000, int((2_000_000-150000)*8*30/total*.85))
        command = [executable,'-hide_banner','-loglevel','error','-nostdin','-n','-f','rawvideo',
            '-pix_fmt','rgb24','-s','960x280','-r','30','-i','pipe:0','-an','-c:v','libx264',
            '-preset','fast','-crf','30','-maxrate',str(ceiling),'-bufsize',str(ceiling),
            '-pix_fmt','yuv420p','-movflags','+faststart',str(destination)]
        poster = None
        proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        try:
            for frame in range(total):
                ok, rgb = capture.read()
                require(ok and rgb.shape == (1152,1536,3)
                    and int(capture.get(cv2.CAP_PROP_POS_FRAMES)) == frame+1, 'No lost / repeated RGB frames')
                row = template.copy(); row.paste(Image.fromarray(cv2.cvtColor(
                    cv2.resize(rgb,(320,240),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB)), (0,20))
                row.paste(Image.fromarray(ra.frame(ah[frame], viewer.object_at_frame(a,frame))), (320,20))
                row.paste(Image.fromarray(rb.frame(bh[frame], viewer.object_at_frame(b,frame))), (640,20))
                ImageDraw.Draw(row).text((4,263), f'ep{ep:02d} f{frame}/{total-1} {frame/30:.2f}s', font=font, fill='white')
                proc.stdin.write(np.asarray(row).tobytes())
                if frame == 0:
                    stream = BytesIO(); row.save(stream, format='JPEG', quality=65, optimize=True); poster=stream.getvalue()
            require(not capture.read()[0], 'No extra original RGB frames')
            proc.stdin.close(); require(proc.wait(timeout=60)==0, 'Encoding failed')
        finally:
            capture.release()
            if proc.poll() is None: proc.kill(); proc.wait()
            if proc.stderr: proc.stderr.close()
        require(0 < destination.stat().st_size <= 2_000_000 and poster and len(poster)<=100000,
                'Lightweight full-clip media bounds required')
        destination.chmod(0o444); jpeg=output/f'episode_{ep:06d}.jpg'
        with jpeg.open('xb') as stream: stream.write(poster)
        jpeg.chmod(0o444)
        encoded = json.loads(subprocess.check_output([probe,'-v','error','-count_frames',
            '-select_streams','v:0','-show_streams','-of','json',str(destination)]))
        probe_contract(encoded,total)
        report = dict(episode=ep, frames=total, original_frame_indices=list(range(total)),
            QA_passed=accepted, quality_verified=False, candidate_report=ledger.records[str(report_path)],
            video=identity(destination,2_000_000), poster=identity(jpeg,100000),
            unchanged_baseline=True, one_fixed_camera=True, one_fixed_floor=True, per_frame_alignment=False)
        reports.append(report); states.append(dict(episode=ep,status='complete',QA_passed=accepted))
        if cohort:
            report['candidate_status'] = candidate_status; states[-1]['candidate_status'] = candidate_status
        print(json.dumps(dict(stage='full_video_rendered',episode=ep,frames=total,QA_passed=accepted)),flush=True)
    require(reports, 'No completed candidate available for honest visual comparison')
    ledger.verify(); require(source(ROOT,code,rev,ENTRY,HELPERS)==binding, 'Immutable source changed')
    result = dict(schema='world_reward.end2end_preview.v1',status='complete',producer_revision=rev,
        candidate_revision=candidate_revision,baseline_revision=BASELINE,cohort=COHORT,states=states,
        source_binding=binding,sources=ledger.records,reports=reports,quality_verified=False,
        source_rehashed_after=True,heavy_media_local=False,elapsed_seconds=time.monotonic()-started)
    if cohort: result.update(schema='world_reward.end2end_preview.v4' if kind == 'smoothing' else 'world_reward.end2end_preview.v3' if kind == 'sqp' else 'world_reward.end2end_preview.v2', candidate_kind=kind)
    path=output/'render.json'
    with path.open('xb') as stream: stream.write((json.dumps(result,sort_keys=True,allow_nan=False)+'\n').encode())
    path.chmod(0o444)


def publish(render_revision, kind='joint'):
    from full4d_publish import PrivatePreviews
    rev=revision(os.environ['WR_CODE_REVISION']); render_revision=revision(render_revision)
    candidate_kind(kind); code=canonical(os.environ['WR_CODE']); binding=source(ROOT,code,rev,ENTRY,HELPERS)
    output=ROOT/'results'/('end2end-preview-'+render_revision)
    pin=identity(output/'render.json',4<<20); rendered=strict((output/'render.json').read_bytes())
    require(rendered['status']=='complete' and rendered['producer_revision']==render_revision
        and rendered['quality_verified'] is False and rendered['cohort']==COHORT,
        'Actual complete non-quality render required')
    require((kind == 'joint' and rendered.get('schema') == 'world_reward.end2end_preview.v1'
            and 'candidate_kind' not in rendered) or
        (kind in (*NATIVE_METHODS, 'smoothing') and rendered.get('schema') ==
            ('world_reward.end2end_preview.v4' if kind == 'smoothing' else 'world_reward.end2end_preview.v3' if kind == 'sqp' else 'world_reward.end2end_preview.v2')
            and rendered.get('candidate_kind') == kind), 'Explicit method matches original render receipt')
    client=PrivatePreviews();client.require_private();files=[]
    for row in rendered['reports']:
        for extension,key,mime,cap in [('mp4','video','video/mp4',2_000_000),('jpg','poster','image/jpeg',100000)]:
            p=output/f"episode_{row['episode']:06d}.{extension}"
            require(identity(p,cap)==row[key],'Original bounded rendered bytes required')
            files.append(client.upload(f"full4d-{rev}/episode_{row['episode']:06d}.{extension}",
                p.read_bytes(),mime,rev)|dict(episode_index=row['episode']))
    receipt=dict(schema='world_reward.end2end_preview_publication.v1',status='pass',producer_revision=rev,
        render_revision=render_revision,candidate_revision=rendered['candidate_revision'],
        baseline_revision=BASELINE,render_report_pin=pin,cohort=COHORT,states=rendered['states'],files=files,
        source_binding=binding,endpoint='https://stworldrewardresearch26.blob.core.windows.net/qa-previews',
        private_container_verified=True,public_access_changed=False,account_keys_used=False,
        quality_verified=False,heavy_data_uploaded=False)
    if kind in (*NATIVE_METHODS, 'smoothing'):
        receipt.update(schema='world_reward.end2end_preview_publication.v4' if kind == 'smoothing' else 'world_reward.end2end_preview_publication.v3' if kind == 'sqp' else 'world_reward.end2end_preview_publication.v2', candidate_kind=kind)
    p=ROOT/'results'/('end2end-preview-publication-'+rev+'.json')
    with p.open('xb') as stream: stream.write((json.dumps(receipt,sort_keys=True)+'\n').encode())
    p.chmod(0o444);print(json.dumps(dict(status='published',receipt=str(p),bytes=p.stat().st_size,
        sha256=hashlib.sha256(p.read_bytes()).hexdigest())),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('render','publish'));p.add_argument('revision')
    p.add_argument('--candidate-kind', choices=tuple(KINDS), default='joint')
    args=p.parse_args();(render if args.mode=='render' else publish)(args.revision, args.candidate_kind)
