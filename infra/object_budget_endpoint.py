"""Frozen CPU-only endpoint mesh-budget proposal, not production adoption.

Consumes only predicted Track 1 object geometry and audited metadata. Exact
position welding preserves every oriented triangle; no process=True repair is
used before QEM. The official helper is used only after the candidate fits the
budget, and its export/weld/padding fidelity is independently checked. Grounded
metric scale is baked once, with no change of origin, pose or camera.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import secrets
import subprocess
import sys
import time

import numpy as np

from body_smoke import TRACK1_EPISODE_COUNT, _validate_inputs
from mesh_link_gate import mesh_topology, geometry_gates, _array_hash, _write, VERSION, WHEEL_SHA
from mesh_endpoint_gate import (ENDPOINT_PARAMETERS, STAGE as ENDPOINT_STAGE, endpoint_simplify,
                               source_intersections, true_hollow_containment)
from world_reward.data import sha256
from world_reward.artifact_paths import episode_output

STAGE = 'world_reward_cpu_object_budget_endpoint'
BUDGET_HELPER_SHA = '42ab8ab35f37b806fb1465eadd96abe43eaac04575da47a4855d08eefe6167b0'
MAX_SECONDS = 900.


def _argument_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episode', type=int, choices=range(TRACK1_EPISODE_COUNT), required=True)
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    return p


def regular(root: Path, path: Path):
    """Every prerequisite must be a regular file beneath the fixed runtime root."""
    if not path.is_relative_to(root) or not path.is_file(): raise ValueError('Missing bounded regular artifact')
    for part in (path, *path.parents):
        if part.is_symlink(): raise ValueError('Symlink artifact or ancestor forbidden')
        if part == root: break
    return path


def exact_weld(vertices, faces):
    """Merge only bit-identical positions; preserve each oriented face and order."""
    v, f = np.asarray(vertices), np.asarray(faces)
    if (np.ma.isMaskedArray(vertices) or np.ma.isMaskedArray(faces) or v.ndim != 2 or v.shape[1:] != (3,) or v.dtype.kind != 'f' or not np.isfinite(v).all()
            or f.ndim != 2 or f.shape[1:] != (3,) or f.dtype.kind not in 'iu' or not len(f)
            or np.min(f) < 0 or np.max(f) >= len(v)):
        raise ValueError('Invalid raw GLB triangular arrays')
    unique, inverse = np.unique(v, axis=0, return_inverse=True)
    welded = inverse[f]
    if not np.array_equal(unique[welded], v[f]):
        raise ValueError('Exact welding changed oriented triangle coordinates')
    if np.any(np.diff(np.sort(welded, axis=1), axis=1) == 0):
        raise ValueError('Source contains collapsed triangles; no deletion or healing')
    return unique, welded.astype(np.int64), {'raw_vertices': len(v), 'welded_vertices': len(unique),
        'faces_preserved': len(f), 'face_order_and_coordinates_exact': True, 'faces_removed': 0,
        'position_merging': 'exact equality only', 'source_modified': False}


def _triangle_rows(vertices, faces):
    """Sorted oriented triangle geometry; cyclic corner rotations are equivalent."""
    t = np.asarray(vertices, dtype=np.float64)[np.asarray(faces)]
    choices = np.stack([t, np.roll(t, 1, axis=1), np.roll(t, 2, axis=1)], axis=1).reshape(-1, 3, 9)
    # Canonicalize cyclic, never reversed, orientation using a lexicographic key.
    order = np.lexsort(tuple(choices[:, :, j] for j in range(8, -1, -1)), axis=1)
    rows = choices[np.arange(len(t)), order[:, 0]]
    return rows[np.lexsort(tuple(rows[:, j] for j in range(8, -1, -1)))]


def verify_pack_fidelity(exported, packed_vertices, packed_faces):
    """Prove official process/weld only reordered equivalent exported triangles.

    Padding must be zero faces and repetitions of the first vertex. A meaningful
    face deleted, nonexact positional merge, flip or second simplification fails.
    """
    v, f = np.asarray(packed_vertices), np.asarray(packed_faces)
    if (v.shape != (4096, 3) or f.shape != (4096, 3) or v.dtype.kind != 'f'
            or f.dtype.kind not in 'iu' or not np.isfinite(v).all() or np.min(f) < 0 or np.max(f) >= len(v)):
        raise ValueError('Official fixed 4096 arrays are invalid')
    zero = np.all(f == 0, axis=1)
    active = f[~zero]
    if not np.array_equal(_triangle_rows(*exported), _triangle_rows(v, active)):
        raise ValueError('Official export/weld altered meaningful oriented triangle geometry')
    used = np.unique(active)
    unused = np.setdiff1d(np.arange(len(v)), used)
    if len(unused) and not np.array_equal(v[unused], np.repeat(v[:1], len(unused), axis=0)):
        raise ValueError('Nonpadding unused vertices in official output')
    compact, faces, _ = exact_weld(v[used], np.searchsorted(used, active))
    mesh_topology(compact, faces)
    return (compact, faces), {'active_faces': len(active), 'padding_faces': int(zero.sum()),
        'padding_vertices': len(unused), 'oriented_triangles_exact': True,
        'official_helper_simplification_invoked': False, 'nonexact_merge_or_face_deletion': False}


def _provenance(record, stage, episode, video_hash):
    expected = {'stage': stage, 'status': 'pass', 'episode_index': episode, 'input_track': 'track_1',
                'input_sha256': video_hash, 'ground_truth_used': False, 'hand_labeled_test': False, 'oracle_modes': []}
    if (any(record.get(k) != v for k, v in expected.items()) or type(record.get('episode_index')) is not int
            or record.get('ground_truth_used') is not False or record.get('hand_labeled_test') is not False):
        raise ValueError('Predicted object/alignment provenance mismatch')


def prerequisites(root, episode):
    """Bind source object, grounded transform and selected-video ancestry, no RGB."""
    inputs = _validate_inputs(root, episode_index=episode)
    base = episode_output(root, episode)
    object_path = regular(root, base / 'object_grounded/report.json')
    alignment_path = regular(root, base / 'scale_smoke/report.json')
    obj, alignment = json.loads(object_path.read_text()), json.loads(alignment_path.read_text())
    _provenance(obj, 'sam3d_objects_grounded_fixed_frame', episode, inputs['video_sha256'])
    _provenance(alignment, 'predicted_human_anchored_moge2_pointmaps', episode, inputs['video_sha256'])
    if (obj.get('frame_index') != 0 or type(obj.get('frame_index')) is not int
            or obj.get('scale_source') != 'already_human_anchored_MoGe2_no_second_scalar'
            or obj.get('pointmap_grounding', {}).get('alignment_report_sha256') != sha256(alignment_path)
            or alignment.get('coordinate_frame') != 'OpenCV_x_right_y_down_z_forward'
            or alignment.get('pointmap_scale_application') != 'one_clip_scalar_to_MoGe2_XYZ_already_applied'):
        raise ValueError('Same-camera grounded metric gauge mismatch')
    sources = {'object_report': sha256(object_path), 'alignment_report': sha256(alignment_path),
               'video': inputs['video_sha256'], 'mask_report': inputs['mask_report_sha256'],
               'prompts': inputs['prompts_sha256']}
    for name, field in [('object.glb', 'object_sha256'), ('transform.json', 'transform_sha256'), ('intrinsics.json', 'intrinsics_sha256')]:
        path = regular(root, base / 'object_grounded' / name)
        if sha256(path) != obj.get(field): raise ValueError('Frozen source object artifact hash mismatch')
        sources[name] = sha256(path)
    transform = json.loads((base / 'object_grounded/transform.json').read_text())
    if transform != obj.get('transform'): raise ValueError('Transform/report disagreement')
    s, q, t = [np.asarray(transform[k], dtype=np.float64) for k in ('scale', 'rotation', 'translation')]
    if (s.shape != (3,) or q.shape != (4,) or t.shape != (3,) or not np.isfinite(np.r_[s, q, t]).all()
            or np.min(s) <= 0 or not np.allclose(s, s[0], atol=0, rtol=1e-5)
            or not np.isclose(np.linalg.norm(q), 1., atol=1e-5)):
        raise ValueError('Require finite uniform positive scale and unit quaternion, no averaging')
    # Verify metadata ancestry, never read body/depth predictions or camera labels.
    for folder, stage, field in [('body_smoke', 'sam3d_body_three_frame_smoke', 'body_report_sha256'),
                                 ('depth_smoke', 'monocular_moge2_three_frame', 'depth_report_sha256')]:
        path = regular(root, base / folder / 'report.json')
        if sha256(path) != alignment.get(field): raise ValueError('Alignment ancestry changed')
        _provenance(json.loads(path.read_text()), stage, episode, inputs['video_sha256'])
        sources[folder + '_report'] = sha256(path)
    return inputs, sources, float(s[0])


def experiment_gate(root):
    path = regular(root, root / 'results/mesh-endpoint.json'); gate = json.loads(path.read_text())
    source = Path(__file__).with_name('mesh_endpoint_gate.py')
    old = Path(__file__).with_name('mesh_link_gate.py')
    if (gate.get('stage') != ENDPOINT_STAGE or gate.get('status') != 'pass'
            or gate.get('pymeshlab_version') != VERSION or gate.get('wheel_sha256') != WHEEL_SHA
            or gate.get('script_sha256') != sha256(source) or gate.get('shared_gate_sha256') != sha256(old)
            or gate.get('parameters') != ENDPOINT_PARAMETERS or gate.get('old_cohort_rerun') is not False
            or gate.get('challenge_inputs_used') is not False or gate.get('challenge_ground_truth_used') is not False
            or gate.get('adoption_performed') is not False or gate.get('face_budget') != 4096
            or gate.get('vertex_budget') != 4096 or gate.get('max_gate_seconds') != 120.
            or gate.get('image_id') != os.environ.get('WR_TOPOLOGY_IMAGE_ID')):
        raise ValueError('Exact endpoint experiment must pass before object proposal')
    fixtures = gate.get('fixtures', [])
    names = ['radial_star_hollow', 'torus_thin_wall', 'offcenter_ellipsoid_hollow', 'disconnected_asymmetric']
    if len(fixtures) != 4 or [f.get('fixture') for f in fixtures] != names:
        raise ValueError('Endpoint gate fixture coverage mismatch')
    for fixture in fixtures:
        if (fixture.get('status') != 'pass' or fixture.get('two_runs_arrays_identical') is not True
                or fixture.get('source_arrays_unchanged') is not True or len(fixture.get('runs', [])) != 2):
            raise ValueError('Endpoint experiment incomplete')
        for run in fixture['runs']:
            geometry = run.get('geometry', {})
            errors = geometry.get('per_component_relative_volume_errors', [])
            if (not errors or any(not np.isfinite(x) or x > .05 or x < 0 for x in errors)
                    or not np.isfinite(geometry.get('sampled_bidirectional_chamfer_diagonal_ratio', np.nan))
                    or not np.isfinite(geometry.get('net_volume_relative_error', np.nan))):
                raise ValueError('Endpoint experiment geometry measurements missing/nonfinite')
            if (run.get('self_intersecting_faces') != 0 or run.get('topology', {}).get('vertices', 4097) > 4096
                    or run.get('topology', {}).get('faces', 4097) > 4096
                    or run.get('geometry', {}).get('sampled_bidirectional_chamfer_diagonal_ratio', 1) > .01
                    or run.get('geometry', {}).get('net_volume_relative_error', 1) > .05):
                raise ValueError('Endpoint experiment numerical gates invalid')
            if fixture['fixture'] != 'disconnected_asymmetric' and run.get('containment', {}).get('true_containment_verified') is not True:
                raise ValueError('Endpoint experiment missing physical containment')
    build_path = regular(root, root / 'results/image-topology-cpu.json'); build = json.loads(build_path.read_text())
    if (sha256(build_path) != gate.get('build_manifest_sha256') or build.get('image_id') != gate['image_id']
            or build.get('stage') != 'world_reward_topology_cpu_build' or build.get('status') != 'pass'):
        raise ValueError('Endpoint image build provenance changed')
    return gate, {'endpoint_gate': sha256(path), 'image_build': sha256(build_path)}


def _load_mesh(path):
    import trimesh
    raw = trimesh.load(path, force='mesh', process=False)
    if not isinstance(raw, trimesh.Trimesh): raise ValueError('Source must resolve to one transformed triangle mesh')
    return np.asarray(raw.vertices).copy(), np.asarray(raw.faces).copy()


def produce(root, episode, report, path):
    """One QEM call then official export/weld/pad; fail, never retry or adopt."""
    started = time.perf_counter()
    inputs, sources, scale = prerequisites(root, episode)
    _, gate_hashes = experiment_gate(root)
    if sources != report['source_hashes'] or gate_hashes != report['experiment_hashes']:
        raise ValueError('Inputs changed after parent preflight')
    base = episode_output(root, episode)
    helper = regular(root, root / 'vendor/v2d_submission_kit/v2dlb/mesh_budget.py')
    if sha256(helper) != BUDGET_HELPER_SHA: raise ValueError('Unpinned official budget helper')
    raw = _load_mesh(base / 'object_grounded/object.glb')
    source_v, source_f, welding = exact_weld(*raw)
    source = source_v, source_f
    topology = mesh_topology(*source)
    report.update(exact_source_welding=welding, source_topology=topology)
    _write(path, report)
    intersections = source_intersections(*source)
    report['source_self_intersections'] = intersections
    if intersections: raise ValueError('Source embedding intersects; no healing')
    source_hash = [_array_hash(x) for x in source]
    candidate, intersections = endpoint_simplify(*source)
    post = mesh_topology(*candidate)
    report.update(endpoint_topology=post, endpoint_self_intersections=intersections)
    _write(path, report)
    if intersections or len(candidate[0]) > 4096 or len(candidate[1]) > 4096:
        raise ValueError('Endpoint budget or embedding failed')
    report['endpoint_geometry'] = {}
    try:
        geometry_gates(source, candidate, topology, post, report['endpoint_geometry'])
    finally:
        _write(path, report)
    # QEM endpoints must remain in the exact original canonical vertex set.
    if len(np.unique(np.concatenate([source_v, candidate[0]]), axis=0)) != len(source_v):
        raise ValueError('Endpoint QEM invented a non-source vertex')
    import trimesh
    output = path.parent
    fixed = output / 'object_fixed_canonical.glb'
    trimesh.Trimesh(*candidate, process=False).export(fixed)
    exported_raw = _load_mesh(fixed)
    exported = exact_weld(*exported_raw)[:2]
    export_topology = mesh_topology(*exported)
    report['export_geometry'] = {}
    geometry_gates(candidate, exported, post, export_topology, report['export_geometry'])
    helper = regular(root, root / 'vendor/v2d_submission_kit/v2dlb/mesh_budget.py')
    if sha256(helper) != BUDGET_HELPER_SHA: raise ValueError('Unpinned official budget helper')
    spec = importlib.util.spec_from_file_location('wr_official_budget_helper', helper)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    packed_v, packed_f = module.budget_mesh(str(fixed), faces=4096, vertices=4096)
    compact, fidelity = verify_pack_fidelity(exported, packed_v, packed_f)
    packed_topology = mesh_topology(*compact)
    packed_intersections = source_intersections(*compact)
    if packed_intersections: raise ValueError('Official export/weld introduced self-intersections')
    report.update(official_pack_fidelity=fidelity, packed_topology=packed_topology,
                  packed_self_intersections=packed_intersections)
    report['packed_geometry'] = {}
    geometry_gates(source, compact, topology, packed_topology, report['packed_geometry'])
    signs = sorted(x['volume_sign'] for x in packed_topology['components'])
    report['cavity_containment_verified'] = False
    if signs == [-1, 1]:
        report['packed_containment'] = {}
        true_hollow_containment(*compact, self_intersecting_faces=0, diagnostics=report['packed_containment'])
        report['cavity_containment_verified'] = True
    report['source_cavity_containment_verified'] = False  # full-resolution O(VF) deliberately not claimed
    metric_v = packed_v * scale
    if not np.isfinite(metric_v).all(): raise ValueError('Metric scale multiplication overflow')
    metric_compact = compact[0] * scale, compact[1]
    metric_topology = mesh_topology(*metric_compact)
    metric_source = source_v * scale, source_f
    report['metric_geometry'] = {}
    geometry_gates(metric_source, metric_compact, mesh_topology(*metric_source), metric_topology, report['metric_geometry'])
    report['metric_topology'] = metric_topology
    geometry = output / 'geometry.npz'
    with geometry.open('xb') as h:
        np.savez_compressed(h, vertices=metric_v, faces=packed_f, object_scale=np.array(1.),
                            episode_index=np.array(episode), grounded_scale_baked=np.array(scale))
    if source_hash != [_array_hash(x) for x in source]: raise ValueError('Source arrays changed')
    report.update(status='pass', geometry_sha256=sha256(geometry), canonical_glb_sha256=sha256(fixed),
                  official_helper_sha256=BUDGET_HELPER_SHA, metric_scale_baked_once=scale,
                  submission_object_scale=1., source_arrays_unchanged=True, elapsed_seconds=time.perf_counter()-started)
    _write(path, report)


def main():
    if platform.system() != 'Linux' or {p.name for p in Path('/sys/class/net').iterdir()} != {'lo'}:
        raise RuntimeError('Require Linux CPU network-none executor')
    args = _argument_parser().parse_args()
    root = Path(os.environ.get('WR_ROOT', '/srv/scenesmith/world-reward'))
    output = episode_output(root, args.episode) / 'object_budget_endpoint'
    path = output / 'report.json'
    if args.worker:
        report = json.loads(regular(root, path).read_text()); nonce = os.environ.get('WR_OBJECT_BUDGET_NONCE', '')
        if (report.get('status') != 'running' or report.get('stage') != STAGE or report.get('episode_index') != args.episode
                or report.get('script_sha256') != sha256(Path(__file__))
                or report.get('endpoint_driver_sha256') != sha256(Path(__file__).with_name('mesh_endpoint_gate.py'))
                or report.get('shared_geometry_driver_sha256') != sha256(Path(__file__).with_name('mesh_link_gate.py'))
                or report.get('producer_revision') != os.environ.get('WR_CODE_REVISION')
                or not nonce or report.get('worker_nonce_sha256') != hashlib.sha256(nonce.encode()).hexdigest()):
            raise RuntimeError('Require parent-created running proposal; frozen results immutable')
        try: produce(root, args.episode, report, path)
        except Exception as exc:
            report.update(status='fail', error_type=type(exc).__name__, error=str(exc)); _write(path, report); raise
        return
    if output.exists() or output.is_symlink(): raise RuntimeError('Frozen endpoint object proposal exists')
    inputs, sources, scale = prerequisites(root, args.episode)
    gate, gate_hashes = experiment_gate(root)
    nonce = secrets.token_hex(32)
    report = {'stage': STAGE, 'status': 'running', 'episode_index': args.episode,
              'input_track': 'track_1', 'input_sha256': inputs['video_sha256'], 'source_hashes': sources,
              'experiment_hashes': gate_hashes, 'image_id': gate['image_id'], 'parameters': ENDPOINT_PARAMETERS,
              'ground_truth_used': False, 'hand_labeled_test': False, 'oracle_modes': [],
              'proposal_only': True, 'adoption_performed': False, 'challenge_performance_verified': False,
              'metric_scale_accuracy_verified': False, 'scale_source': 'grounded_generation_scale_baked_once',
              'coordinate_frame': 'fixed_canonical_OpenCV_axes_separate_pose', 'geometry_units': 'metres',
              'frame_translation_or_rotation_changed': False, 'no_holes_filled': True, 'no_winding_repair': True,
              'components_deleted': False, 'worker_nonce_sha256': hashlib.sha256(nonce.encode()).hexdigest(),
              'script_sha256': sha256(Path(__file__)),
              'endpoint_driver_sha256': sha256(Path(__file__).with_name('mesh_endpoint_gate.py')),
              'shared_geometry_driver_sha256': sha256(Path(__file__).with_name('mesh_link_gate.py')), 'producer_revision': os.environ.get('WR_CODE_REVISION'),
              'max_worker_seconds': MAX_SECONDS, 'container_cpus': 4, 'container_memory_bytes': 16*1024**3,
              'pymeshlab_version': VERSION, 'wheel_sha256': WHEEL_SHA, 'grounded_scale': scale,
              'source_containment_preservation_claim': False, 'embedding_predicates_exact_universal_claim': False}
    output.mkdir(parents=True, exist_ok=False)
    with path.open('x') as h: h.write(json.dumps(report, indent=2)+'\n')
    started = time.perf_counter()
    try:
        p = subprocess.run([sys.executable, str(Path(__file__)), '--episode', str(args.episode), '--worker'],
                           check=False, timeout=MAX_SECONDS, env=os.environ | {'WR_OBJECT_BUDGET_NONCE': nonce})
    except subprocess.TimeoutExpired:
        report=json.loads(path.read_text()); report.update(status='fail', error_type='TimeoutError',
            error='Hard900s CPU proposal budget exceeded', elapsed_seconds=time.perf_counter()-started)
        _write(path, report); raise
    report=json.loads(path.read_text())
    if p.returncode or report['status']!='pass': raise RuntimeError(f"CPU object budget proposal failed: {report.get('error', 'worker incomplete')}")
    report.update(worker_elapsed_seconds=report['elapsed_seconds'], elapsed_seconds=time.perf_counter()-started,
                  elapsed_includes_child_startup=True); _write(path, report)
    print(json.dumps({'stage':STAGE,'status':'pass','episode_index':args.episode,'adoption_performed':False}))


if __name__ == '__main__': main()
