"""Prepare a GT-free native CARI export from verified World Reward predictions.

Original RGB/masks, fixed inferred camera, one human-anchored depth gauge and
one fixed object mesh. No tracking, oracle pose, hidden labels or GT files.
All large intermediates and model inputs remain on the remote managed disk.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import errno
import shutil
import stat
from pathlib import Path
from world_reward.artifact_paths import episode_output, pin_path as artifact_pin_path
import platform
import sys
import time

from body_smoke import EPISODE, TRACK1_EPISODE_COUNT, _validate_inputs, _pinned_checkout, UPSTREAM_REVISION
from world_reward.data import sha256
from world_reward.mesh_geometry import normalize_degenerate_faces


# Actual immutable native image source identities, not a newest-version lookup.
SOLID_TRIMESH_SOURCES = {
    'scene/transforms.py': {'bytes': 28938, 'sha256': 'f38beb118974172c42270d035f3bb77eb5d374de8c251aab7bce1a33afbe8ea2'},
    'transformations.py': {'bytes': 74801, 'sha256': '644b112736124b7803c028a248279926d10f748649634006d493a90722eae360'},
    'scene/scene.py': {'bytes': 54380, 'sha256': '6dd01efd09edae58f9d3643d73f9ca943904cb353a52f9d8f6633213b684e15b'},
    'base.py': {'bytes': 110040, 'sha256': '13d002a80f14bfa33cf5e49fab19b60083356c2377d92fc98a701d0d2b3e8706'},
    'exchange/gltf/__init__.py': {'bytes': 79987, 'sha256': '0bebabb3a28a9e75773ddf4135a51198dc107ad61bbbcf4de280191417e2159c'},
    'exchange/load.py': {'bytes': 21516, 'sha256': '6b313c1f0ff9295e1cdf6a5567588f5df7d5c02eea12353fc9d15740d19a9e4e'},
}
SOLID_NATIVE_LOAD_SOURCES = {
    'lib_mhr/contact.py': {'bytes': 9646, 'sha256': 'd4e8a92845d75a7bae962f312dee4d747587c39157a978908293a5645beb6d5c'},
    'learning/training/mhr_opt_refineout.py': {'bytes': 92824, 'sha256': '84e0e818a3bc0935bb30b75fcd82fd7c5e3730ed812864594cd759697ddb406b'},
}


class _QueryFlag(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if getattr(namespace, self.dest, False): parser.error('Repeated --query-requalification')
        setattr(namespace, self.dest, True)


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=int, choices=range(TRACK1_EPISODE_COUNT), default=EPISODE)
    parser.add_argument("--mesh-source", choices=("default", "solid", "surface"), default="default")
    parser.add_argument('--allow-unobserved-poses', action='store_true', default=False,
                        help='Admit explicit full-T latent surface initialization, not final predictions')
    parser.add_argument('--query-requalification', action=_QueryFlag, nargs=0, default=False)
    return parser


def _solid_compact(vertices, faces, np):
    """Remove only official zero padding, never weld or repair a surface."""
    from exact_mesh_geometry import _exact_faces, exact_mesh_topology
    if (vertices.dtype != np.float64 or faces.dtype != np.int64
            or vertices.shape != (4096, 3) or faces.shape != (4096, 3)
            or not np.isfinite(vertices).all() or np.any(faces < 0) or np.any(faces >= 4096)):
        raise ValueError("Solid poses must retain exact F64/I64 official 4096 geometry")
    active = np.flatnonzero(~np.all(faces == 0, axis=1))
    ids, inverse = np.unique(faces[active], return_inverse=True)
    unused = np.setdiff1d(np.arange(len(vertices)), ids)
    if len(unused) and not np.array_equal(vertices[unused], np.repeat(vertices[:1], len(unused), axis=0)):
        raise ValueError("Only original repeated-first-vertex padding is permitted")
    compact = vertices[ids].copy(), inverse.reshape(-1, 3).astype(np.int64)
    _exact_faces(*compact)
    topology = exact_mesh_topology(*compact)
    stored = compact[0].astype(np.float32).astype(np.float64)
    if not np.isfinite(stored).all():
        raise ValueError("Solid metric positions cannot be represented in GLB float32")
    _exact_faces(stored, compact[1])
    _solid_float32_orientation(compact[0], stored, compact[1])
    _solid_topology_equal(topology, exact_mesh_topology(stored, compact[1]))
    if np.max(np.linalg.norm(stored - compact[0], axis=1)) > 1e-5:
        raise ValueError("Solid GLB quantization exceeds inherited 1e-5 metric roundtrip bound")
    return active, compact, topology


def _solid_float32_orientation(original, stored, faces):
    """Exact dyadic local normal dot; no area tolerance or sign repair."""
    from fractions import Fraction
    def normal(triangle):
        a, b, c = [[Fraction(float(x)) for x in row] for row in triangle]
        u, v = [b[i] - a[i] for i in range(3)], [c[i] - a[i] for i in range(3)]
        return [u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0]]
    for face in faces:
        if sum(a*b for a, b in zip(normal(original[face]), normal(stored[face]))) <= 0:
            raise ValueError('Float32 storage collapsed or reversed a meaningful local triangle')


def _solid_topology_equal(before, after):
    signature = lambda t: (t['vertices'], t['active_vertices'], t['faces'],
        sorted((c['euler'], c['volume_sign'], c['vertices'], c['faces']) for c in t['components']))
    if signature(before) != signature(after):
        raise ValueError("Solid serialization changed whole-shell Euler/orientation/counts")


def _solid_preflight(root, episode, inputs, report, pose_path, np, *, query_requalification=False):
    """Independent frozen proposal and full trajectory, before reserving output."""
    from solid_geometry_loader import load, identity, strict_json, SOURCE_HELPERS
    code = Path(__file__).resolve().parent.parent
    base = episode_output(root, episode)
    if type(query_requalification) is not bool: raise ValueError('Explicit query profile required')
    if pose_path != base/'object_pose_full_solid/geometry_and_poses.npz':
        raise ValueError('Canonical solid pose source required')
    options = dict(query_requalification=True) if query_requalification else {}
    pin_path = artifact_pin_path(code, episode, "solid_mesh")
    pin = identity(pin_path)
    pins = strict_json(pin_path.read_text())
    object_path, alignment_path = base / 'object_grounded/report.json', base / 'scale_smoke/report.json'
    transform_path = base / 'object_grounded/transform.json'
    transform = strict_json(transform_path.read_text())
    scale = np.asarray(transform['scale'], np.float64)
    if (scale.shape != (3,) or not np.isfinite(scale).all() or np.any(scale <= 0)
            or not np.allclose(scale, scale[0], atol=0, rtol=1e-5)):
        raise ValueError("Original positive grounding scale is fixed; never average or rebake")
    values = load(root, episode, inputs['video_sha256'], sha256(object_path), sha256(alignment_path),
                  float(scale[0]), pins=pins, **options)
    expected_v, expected_f, expected_active, _, canonical, receipt = values
    if query_requalification:
        if ('query_requalification' not in receipt or report.get('query_requalification') != receipt['query_requalification']
                or report.get('topology_budget', {}).get('query_requalification') != receipt['query_requalification']):
            raise ValueError('Native pose/CPU balanced query provenance differs')
    elif any('query_requalification' in row for row in (receipt, report, report.get('topology_budget', {}))):
        raise ValueError('Balanced query must not be relabeled as the default solid profile')
    pose_canonical = pose_path.parent / 'object_fixed_canonical.glb'
    pose_report_path = pose_path.parent / 'report.json'
    parent = pose_path.parent.lstat()
    if (pose_path.parent.resolve() != pose_path.parent or not stat.S_ISDIR(parent.st_mode)
            or stat.S_IMODE(parent.st_mode) != 0o555
            or {p.name for p in pose_path.parent.iterdir()} != {'report.json', 'geometry_and_poses.npz', 'object_fixed_canonical.glb'}
            or any(stat.S_IMODE(p.lstat().st_mode) != 0o444 for p in (pose_path, pose_canonical, pose_report_path))
            or strict_json(pose_report_path.read_text()) != report):
        raise ValueError('Complete immutable solid pose output required; unsealed/native-only PASS is insufficient')
    ledger = {p: identity(p, readonly=p in (pin_path, pose_path, pose_canonical, canonical)) for p in
              (pin_path, pose_path, pose_canonical, canonical, object_path, alignment_path,
               transform_path, pose_report_path)}
    if (ledger[pin_path] != pin or report.get('mesh_source') != 'solid'
            or report.get('execution_verified') is not True or report.get('original_frame_coverage_verified') is not True
            or report.get('fixed_shape') is not True or report.get('object_report_sha256') != ledger[object_path]['sha256']
            or report.get('geometry_and_poses_sha256') != ledger[pose_path]['sha256']
            or report.get('fixed_canonical_mesh_sha256') != ledger[canonical]['sha256']
            or ledger[pose_canonical] != ledger[canonical]
            or report.get('topology_budget', {}).get('committed_pins_sha256') != pin['sha256']
            or any(report['topology_budget'].get(k) != receipt[k] for k in
                   ('cpu_report_sha256', 'native_report_sha256', 'cpu_producer_revision', 'cpu_script_sha256'))):
        raise ValueError("Solid pose report does not match the independently pinned canonical proposal")
    with np.load(pose_path, allow_pickle=False) as arrays:
        if set(arrays.files) != {'vertices', 'faces', 'frame_index', 'rotation', 'translation', 'object_scale'}:
            raise ValueError("Exact six-array solid full-trajectory payload required")
        v, f, r, t = (arrays[k].copy() for k in ('vertices', 'faces', 'rotation', 'translation'))
        frames, scalar = arrays['frame_index'], arrays['object_scale']
        if (frames.dtype != np.int64 or frames.shape != (inputs['total_frames'],)
                or not np.array_equal(frames, np.arange(inputs['total_frames']))
                or scalar.dtype != np.float64 or scalar.shape != () or scalar.item() != 1.
                or r.dtype != np.float64 or t.dtype != np.float64
                or not np.array_equal(v, expected_v) or not np.array_equal(f, expected_f)):
            raise ValueError("Solid geometry/gauge/original frame IDs differ from frozen CPU inputs")
    active, compact, topology = _solid_compact(v, f, np)
    if not np.array_equal(active, expected_active):
        raise ValueError("Meaningful solid triangles differ from independently verified padding")
    helpers = {*SOURCE_HELPERS, 'infra/cari_prepare.py', 'infra/solid_geometry_loader.py',
               'infra/mesh_precision_diagnostic.py', 'src/world_reward/mesh_geometry.py',
               'infra/cari_wrapper_common.sh', 'infra/run_cari_prepare.sh', 'src/world_reward/artifact_paths.py'}
    ledger.update({code / name: identity(code / name) for name in helpers})
    _solid_recheck(ledger)
    return v, f, active, r, t, compact, topology, ledger


def _solid_recheck(ledger):
    from solid_geometry_loader import identity
    code = Path(__file__).resolve().parent.parent
    if any(identity(path, readonly=path.is_relative_to(code) or path.suffix in ('.npz', '.glb')
                    or path.parent.name == 'object_pose_full_solid') != value for path, value in ledger.items()):
        raise ValueError("Frozen solid geometry, poses, pins or pure source changed")
    pose_parents = {p.parent for p in ledger if p.parent.name == 'object_pose_full_solid'}
    if any(stat.S_IMODE(p.lstat().st_mode) != 0o555 or {x.name for x in p.iterdir()} !=
           {'report.json', 'geometry_and_poses.npz', 'object_fixed_canonical.glb'} for p in pose_parents):
        raise ValueError('Immutable solid pose namespace changed after validation')


def _solid_native_sources(native_root, trimesh, np):
    """Authenticate and execute only two original CPU loader function bodies."""
    import importlib
    from solid_geometry_loader import identity
    if trimesh.__version__ != '5.1.0':
        raise ValueError('Exact qualified native Trimesh version required')
    package = Path(trimesh.__file__).parent
    ledger = {}
    for name, pin in SOLID_TRIMESH_SOURCES.items():
        path = package / name
        module = importlib.import_module('trimesh.' + name.removesuffix('/__init__.py').removesuffix('.py').replace('/', '.'))
        if Path(module.__file__) != path or identity(path, readonly=False) != pin:
            raise ValueError('Native Trimesh transform/scene source differs from qualified image')
        ledger[path] = pin
    if Path(trimesh.transformations.fix_rigid.__code__.co_filename) != package / 'transformations.py':
        raise ValueError('Native rigid projection must execute the original source function')
    namespace = {'np': np, 'trimesh': trimesh, 'Path': Path}
    for name, function in (('lib_mhr/contact.py', 'load_object_mesh'),
                           ('learning/training/mhr_opt_refineout.py', '_load_object_vertices')):
        path = native_root / name
        if identity(path, readonly=False) != SOLID_NATIVE_LOAD_SOURCES[name]:
            raise ValueError('Original native scene/FP32 mesh-loader source differs')
        tree = ast.parse(path.read_text(), filename=str(path))
        definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == function]
        if (len(definitions) != 1 or definitions[0].decorator_list or len(definitions[0].args.args) != 1
                or definitions[0].args.args[0].arg != 'path' or definitions[0].args.posonlyargs
                or definitions[0].args.kwonlyargs or definitions[0].args.defaults
                or definitions[0].args.vararg or definitions[0].args.kwarg):
            raise ValueError('Exact original one-path CPU loader function required')
        futures = [node for node in tree.body if isinstance(node, ast.ImportFrom) and node.module == '__future__']
        exec(compile(ast.Module(body=[*futures, definitions[0]], type_ignores=[]), str(path), 'exec'), namespace)
        if namespace[function].__code__.co_filename != str(path):
            raise ValueError('Original CPU mesh-loader execution binding differs')
        ledger[path] = SOLID_NATIVE_LOAD_SOURCES[name]
    _solid_recheck(ledger)
    return namespace['_load_object_vertices'], ledger


def _solid_scene_matrix(scene, expected, trimesh, np):
    """Single canonical mesh instance; raw edges and native projection distinct."""
    if not isinstance(scene, trimesh.Scene) or len(scene.geometry) != 1 or len(scene.graph.nodes_geometry) != 1:
        raise ValueError('Solid canonical consumer supports exactly one geometry and one instance')
    node = scene.graph.nodes_geometry[0]
    edges = {}
    for parent, child, fields in scene.graph.to_edgelist():
        if child in edges:
            raise ValueError('Ambiguous native scene parent')
        matrix = np.asarray(fields['matrix'], np.float64)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all() or not np.array_equal(matrix[3], [0., 0., 0., 1.]):
            raise ValueError('Finite unmodified affine native scene edges required')
        edges[child] = parent, matrix
    path = []; seen = set(); current = node
    while current != scene.graph.base_frame:
        if current in seen or current not in edges:
            raise ValueError('Incomplete or cyclic native scene transform path')
        seen.add(current); parent, matrix = edges[current]; path.append(matrix); current = parent
    raw = np.eye(4)
    for matrix in reversed(path): raw = raw @ matrix
    if not np.array_equal(raw, expected) or scene.graph.repair_rigid != 1e-5:
        raise ValueError('Raw native scene matrix or qualified rigid-projection policy differs')
    effective, geometry = scene.graph[node]
    effective = np.asarray(effective, np.float64)
    projected = trimesh.transformations.fix_rigid(expected, max_deviance=1e-5)
    if geometry not in scene.geometry or not np.array_equal(effective, projected):
        raise ValueError('Effective native scene matrix differs from source-pinned rigid projection')
    return raw, effective


def _solid_serialized_mesh(path, source, topology, trimesh, np, native_load, transform=None):
    """Exact local/raw stage, then the unchanged native graph/FP32 stage."""
    from mesh_precision_diagnostic import raw_glb, triangle_hash
    from object_budget_endpoint import _load_mesh
    from exact_mesh_geometry import exact_mesh_topology
    local, world, records = raw_glb(path)
    stored = source[0].astype(np.float32).astype(np.float64)
    local_triangles = np.concatenate([v[f] for v, f in local])
    if (triangle_hash(local_triangles) != triangle_hash(stored[source[1]]) or len(local) != 1
            or not np.array_equal(local[0][0], stored) or not np.array_equal(local[0][1], source[1])):
        raise ValueError("GLB POSITION/indices altered meaningful oriented triangles")
    if transform is None:
        if not all(row['node_transform_identity'] for row in records):
            raise ValueError("Metric GLB must not introduce a hidden scene transform")
        transform = np.eye(4)
    raw, effective = _solid_scene_matrix(trimesh.load(path, force='scene', process=False), transform, trimesh, np)
    expected = trimesh.transform_points(stored, raw)
    expected_hash = triangle_hash(expected[source[1]])
    loaded_v, loaded_f = _load_mesh(path)
    represented = trimesh.transform_points(stored, effective)
    if (triangle_hash(world) != expected_hash or not np.array_equal(loaded_f, source[1])
            or not np.array_equal(loaded_v, represented)
            or triangle_hash(loaded_v[loaded_f]) != triangle_hash(represented[source[1]])):
        raise ValueError("Native aligned GLB changed represented rigid-transform geometry")
    _solid_topology_equal(topology, exact_mesh_topology(loaded_v, loaded_f))
    native_v, native_f = native_load(path)
    if (type(native_v) is not np.ndarray or native_v.dtype != np.float32
            or type(native_f) is not np.ndarray or native_f.dtype != np.int64
            or not np.array_equal(native_v, represented.astype(np.float32))
            or not np.array_equal(native_f, source[1]) or not np.isfinite(native_v).all()):
        raise ValueError('Actual native FP32 loader altered the qualified effective mesh')
    native_metric = native_v.astype(np.float64)
    quantization_error = float(np.max(np.linalg.norm(native_metric - represented, axis=1)))
    if quantization_error > 1e-5:
        raise ValueError('Actual native FP32 consumer exceeds inherited 1e-5 metric bound')
    _solid_float32_orientation(represented, native_metric, native_f)
    _solid_topology_equal(topology, exact_mesh_topology(native_metric, native_f))
    proof = {'oriented_triangles_sha256': expected_hash, 'position_accessors_float32': True,
            'faces': len(loaded_f), 'components': len(topology['components']),
            'all_meaningful_triangles_preserved': True, 'local_coordinates_or_topology_repaired': False,
            'canonical_scene_policy': 'single_geometry_single_instance_full_index_lineage',
            'raw_scene_transform': raw.tolist(), 'effective_native_transform': effective.tolist(),
            'native_rigidprojection_used': not np.array_equal(raw, effective),
            'effective_oriented_triangles_sha256': triangle_hash(loaded_v[loaded_f]),
            'native_float32_oriented_triangles_sha256': triangle_hash(native_metric[native_f]),
            'native_float32_topology': exact_mesh_topology(native_metric, native_f),
            'native_float32_quantization_max_error_m': quantization_error,
            'native_float32_loader_replayed': True,
            'independent_embedding_reverified': False}
    return proof, (native_v.copy(), native_f.copy())


def _solid_camera_roundtrip(source, rotations, translations, saved_poses, native_mesh, np):
    """Actual native FP32-loaded positions and the F32 poses written to input."""
    original = source[0][source[1].reshape(-1)]
    native_v, native_f = native_mesh
    if native_v.dtype != np.float32 or native_f.dtype != np.int64 or not np.array_equal(native_f, source[1]):
        raise ValueError('Exact actual native FP32 mesh/full face lineage required')
    aligned = native_v.astype(np.float64)[native_f.reshape(-1)]
    if (type(saved_poses) is not np.ndarray or saved_poses.dtype != np.float32
            or saved_poses.shape != (len(rotations), 4, 4) or not np.isfinite(saved_poses).all()):
        raise ValueError('Actual saved full-timeline FP32 native poses required')
    saved_poses = saved_poses.astype(np.float64)
    error = 0.
    for index in range(len(rotations)):
        before = original @ rotations[index].T + translations[index]
        after = aligned @ saved_poses[index, :3, :3].T + saved_poses[index, :3, 3]
        error = max(error, float(np.max(np.linalg.norm(after - before, axis=1))))
    if not np.isfinite(error) or error > 1e-5:
        raise ValueError('Represented GLB/F32 object poses exceed inherited 1e-5 camera roundtrip bound')
    return error


def _latent_observations(report, flags, total, np):
    """Authenticate missing evidence; never relabel an initializer as measured."""
    if (type(total) is not int or total < 3 or np.ma.isMaskedArray(flags)
            or not isinstance(flags, np.ndarray) or flags.dtype != np.bool_ or flags.shape != (total,)
            or not flags[0] or not flags[-1]):
        raise ValueError('Full-T boolean observations and automatic RGB anchors at both edges required')
    wanted = dict(allow_unobserved_poses=True, latent_pose_initializer=True, latent_poses_measured=False,
        observed_pose_frames=int(flags.sum()), latent_pose_frames=int((~flags).sum()),
        ground_truth_used=False, hand_labeled_test=False, oracle_modes=[])
    if (any(type(report.get(k)) is not type(v) or report[k] != v for k, v in wanted.items())
            or type(report.get('pose_observed')) is not list
            or any(type(v) is not bool for v in report['pose_observed'])
            or report['pose_observed'] != flags.tolist()):
        raise ValueError('Exact original latent observation receipt required')
    rows = report.get('frames')
    if (type(rows) is not list or len(rows) != total
            or any(type(row) is not dict or type(row.get('frame_index')) is not int
                   or row['frame_index'] != i or type(row.get('pose_observed')) is not bool
                   or row['pose_observed'] != bool(flags[i]) for i, row in enumerate(rows))):
        raise ValueError('Every original frame requires explicit consistent observation status')
    for row, observed in zip(rows, flags):
        if (type(row.get('candidates')) is not list
                or (observed and (not row['candidates'] or type(row.get('selected')) is not dict
                    or row.get('observation_status') != 'automatic_mask_and_inferred_depth'))
                or (not observed and (row['candidates'] or row.get('selected') is not None
                    or row.get('observation_status') not in {'empty_automatic_mask',
                        'insufficient_inferred_visible_points', 'no_finite_supported_numerical_hypothesis'}))):
            raise ValueError('Missing evidence must not contain a measured pose or invented candidates')
    temporal = report.get('temporal_selection', {})
    expected = dict(method='observed_frame_Viterbi_then_bilateral_SO3_centroid_latent_initialization',
        pose_observed=flags.tolist(), observed_frame_indices=np.flatnonzero(flags).tolist(),
        latent_frame_indices=np.flatnonzero(~flags).tolist(),
        latent_pose_status='initializer_not_measured_or_final_truth', edge_extrapolation=False,
        zero_velocity_prior=False, mask_interpolation=False, geometry_scale_camera_unchanged=True,
        time_units='original_frame_indices', quality_verified=False)
    if type(temporal) is not dict or any(type(temporal.get(k)) is not type(v) or temporal[k] != v
                                       for k, v in expected.items()):
        raise ValueError('Original bilateral initializer provenance required; no invented edge poses')
    selected = temporal.get('candidate_indices')
    if (type(selected) is not list or len(selected) != total
            or any(type(slot) is not int or (slot < 0 if observed else slot != -1)
                   for slot, observed in zip(selected, flags))):
        raise ValueError('Every latent slot must retain the original unobserved sentinel')
    return np.frombuffer(flags.tobytes(), dtype=np.bool_)


def _latent_surface_geometry_and_poses(path, expected_v, expected_f, report, total, budget, np):
    """Separate seven-field admission; the original six-field reader stays strict."""
    from surface_pose_report_capacity import identity
    from world_reward.surface_pose_geometry import compact_surface
    before = identity(path)
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != {'vertices', 'faces', 'frame_index', 'rotation', 'translation',
                              'object_scale', 'pose_observed'}:
            raise ValueError('Exact latent surface payload with original observation flags required')
        v, f, r, t, ids = (data[n].copy() for n in ('vertices', 'faces', 'rotation', 'translation', 'frame_index'))
        if (v.dtype != np.float64 or f.dtype != np.int64 or v.shape != (4096, 3) or f.shape != (4096, 3)
                or v.tobytes() != expected_v.tobytes() or f.tobytes() != expected_f.tobytes()
                or ids.dtype != np.int64 or ids.shape != (total,) or not np.array_equal(ids, np.arange(total))
                or data['object_scale'].dtype != np.float64 or data['object_scale'].shape != ()
                or data['object_scale'].item() != 1.):
            raise ValueError('Original complete canonical geometry, full timeline and once-baked scale required')
        _latent_observations(report, data['pose_observed'], total, np)
    if (r.dtype != np.float64 or t.dtype != np.float64 or r.shape != (total, 3, 3) or t.shape != (total, 3)
            or not np.isfinite(r).all() or not np.isfinite(t).all()
            or not np.allclose(r @ r.swapaxes(-1, -2), np.eye(3), atol=1e-5, rtol=0)
            or not np.allclose(np.linalg.det(r), 1., atol=1e-5, rtol=0)):
        raise ValueError('Finite full-T proper initializer poses required; no pose repair')
    if (type(budget) is not dict or budget.get('source_domain') != 'surface'
            or type(budget.get('metric_scale_baked_once')) is not float or budget['metric_scale_baked_once'] <= 0):
        raise ValueError('Original independently authenticated surface budget required')
    compact, faces, active, topology = compact_surface(v, f,
        canonical_vertex_count=budget['canonical_vertices_count'], canonical_face_count=budget['canonical_faces_count'])
    if identity(path) != before:
        raise ValueError('Latent surface payload changed during read')
    v, f, r, t = (np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape) for a in (v, f, r, t))
    return v, f, active, r, t, (compact, faces), topology, {Path(path): before}


def _latent_pose_metadata(observed, np):
    """Keep original observation flags in the native initializer, not predictions."""
    if (np.ma.isMaskedArray(observed) or not isinstance(observed, np.ndarray)
            or observed.dtype != np.bool_ or observed.ndim != 1
            or len(observed) < 3 or not observed[0] or not observed[-1]):
        raise ValueError('Explicit full-T initializer observation flags with supported edges required')
    return dict(pose_observed=observed.tolist(), pose_observed_frame_index=np.arange(len(observed)).tolist(),
        latent_pose_initializer=True, latent_poses_measured=False,
        requires_native_refinement=True, final_prediction=False)


def _surface_preflight(root, episode, inputs, report, pose_path, np, *, allow_unobserved_poses=False):
    """Authenticated surface and full poses, before reserving prepared inputs."""
    from surface_geometry_loader import load, strict_json, preflight_geometry_and_poses, SOURCE_HELPERS
    from surface_pose_report_capacity import identity, recheck
    code=Path(__file__).resolve().parent.parent
    base=episode_output(root, episode)
    pinpath=artifact_pin_path(code, episode, "surface_mesh")
    pin=identity(pinpath);pins=strict_json(pinpath.read_bytes())
    objectpath=base/'object_grounded/report.json';alignment=base/'scale_smoke/report.json'
    transformpath=base/'object_grounded/transform.json'
    scale=np.asarray(strict_json(transformpath.read_bytes())['scale'],np.float64)
    if scale.shape!=(3,) or not np.isfinite(scale).all() or np.any(scale<=0) or not np.allclose(scale,scale[0],atol=0,rtol=1e-5):
        raise ValueError('One original positive object scale is required; no averaging')
    v,f,active,_,canonical,receipt=load(root,episode,inputs['video_sha256'],sha256(objectpath),
        sha256(alignment),float(scale[0]),pins=pins)
    if type(allow_unobserved_poses) is not bool:
        raise ValueError('Explicit latent surface admission flag required')
    pose_directory = 'object_pose_full_surface_latent' if allow_unobserved_poses else 'object_pose_full_surface'
    if pose_path!=base/pose_directory/'geometry_and_poses.npz':
        raise ValueError('Canonical surface full-trajectory path required')
    parent=pose_path.parent;copy=parent/'object_fixed_canonical.glb';reportpath=parent/'report.json'
    if parent.resolve()!=parent or parent.stat().st_mode&0o777!=0o555 or {p.name for p in parent.iterdir()}!={'report.json','geometry_and_poses.npz','object_fixed_canonical.glb'}:
        raise ValueError('Exact sealed surface pose namespace required')
    ledger={p:identity(p,readonly=p in (pinpath,pose_path,copy,reportpath,canonical)) for p in
        (pinpath,pose_path,copy,reportpath,canonical,objectpath,alignment,transformpath)}
    wanted=dict(mesh_source='surface',execution_verified=True,original_frame_coverage_verified=True,fixed_shape=True,
        object_report_sha256=ledger[objectpath]['sha256'],geometry_and_poses_sha256=ledger[pose_path]['sha256'],
        fixed_canonical_mesh_sha256=ledger[canonical]['sha256'])
    if any(type(report.get(k)) is not type(value) or report[k]!=value for k,value in wanted.items()) or strict_json(reportpath.read_bytes())!=report or ledger[copy]!=ledger[canonical] or ledger[pinpath]!=pin:
        raise ValueError('Surface full pose provenance or canonical copy differs')
    if report.get('topology_budget',{}).get('committed_pins_sha256')!=pin['sha256']:
        raise ValueError('Surface pose committed pins differ')
    for key in ('producer_report_sha256','cpu_native_report_sha256','source_domain','metric_scale_baked_once',
            'geometry_operations_applied','canonical_vertices_count','canonical_faces_count'):
        if type(report['topology_budget'].get(key)) is not type(receipt[key]) or report['topology_budget'][key]!=receipt[key]:
            raise ValueError('Surface pose and inert CPU proposal differ')
    if allow_unobserved_poses:
        values = _latent_surface_geometry_and_poses(pose_path, v, f, report, inputs['total_frames'], receipt, np)
    else:
        values=preflight_geometry_and_poses(pose_path,expected_v=v,expected_f=f,expected_episode=episode,
            expected_scale=1.,topology_budget=receipt)
    pv,pf,pa,r,t,compact,topology,_=values
    if (r.shape != (inputs['total_frames'],3,3) or t.shape != (inputs['total_frames'],3)
            or not np.array_equal(pa,active)):
        raise ValueError('Surface original full timeline or meaningful face indices changed')
    helpers={*SOURCE_HELPERS,'infra/cari_prepare.py','infra/cari_wrapper_common.sh','infra/run_cari_prepare.sh','src/world_reward/artifact_paths.py','infra/surface_pose_report_capacity.py'}
    ledger.update({code/n:identity(code/n) for n in helpers})
    recheck(ledger)
    return pv,pf,pa,r,t,compact,topology,ledger


def _surface_serialized_mesh(path, source, trimesh, np, native_load, transform=None):
    """Same original scene/FP32 math, with surface rather than solid predicates."""
    from world_reward.surface_pose_geometry import serialized_mesh
    from mesh_precision_diagnostic import raw_glb
    from object_budget_endpoint import _load_mesh
    local,world,records=raw_glb(path)
    if len(local)!=1:raise ValueError('Canonical surface must have one original primitive')
    if local[0][0].dtype!=np.float32 or local[0][1].dtype!=np.int64:
        raise ValueError('Original GLB POSITION F32 and complete I64 indices required')
    if transform is None:
        if not all(r['node_transform_identity'] for r in records):raise ValueError('Metric surface contains a hidden scene transform')
        transform=np.eye(4)
    raw,effective=_solid_scene_matrix(trimesh.load(path,force='scene',process=False),transform,trimesh,np)
    loaded_v,loaded_f=_load_mesh(path);nv,nf=native_load(path)
    native_v,native_f,proof=serialized_mesh(*source,local_vertices=local[0][0].astype(np.float64),local_faces=local[0][1],
        raw_world_triangles=world,loaded_vertices=loaded_v,loaded_faces=loaded_f,native_vertices=nv,native_faces=nf,
        raw_matrix=raw,effective_matrix=effective,projected_matrix=trimesh.transformations.fix_rigid(transform,max_deviance=1e-5))
    return proof,(native_v,native_f)


def _surface_saved_poses(saved, names, aligned_poses, pose_sha, matrix, np, *, pose_observed=None):
    """Verify the actual serialized payload, not merely the pre-save array."""
    metadata={'source':'World_Reward_fixed_scale_depth_ICP_Viterbi_not_FoundationPose',
        'ground_truth_used':False,'hand_labeled_test':False,'oracle_modes':[],
        'source_pose_sha256':pose_sha,'mesh_frame_change':matrix.tolist()}
    if pose_observed is not None:
        metadata.update(_latent_pose_metadata(pose_observed, np))
    if (type(saved) is not dict or set(saved)!={'frames','obj_pose_world','metadata'}
            or saved['frames']!=names or type(saved['obj_pose_world']) is not np.ndarray
            or saved['obj_pose_world'].dtype!=np.float32
            or saved['obj_pose_world'].shape!=aligned_poses.shape
            or saved['obj_pose_world'].tobytes()!=aligned_poses.astype(np.float32).tobytes()
            or saved['metadata']!=metadata):
        raise ValueError('Saved surface full timeline, F32 poses or provenance metadata changed')
    return saved['obj_pose_world']


def main():
    if platform.system() != "Linux" or {p.name for p in Path("/sys/class/net").iterdir()} != {"lo"}:
        raise RuntimeError("Require Azure Linux container with network none")
    args = _argument_parser().parse_args()
    if args.allow_unobserved_poses and args.mesh_source != 'surface':
        raise ValueError('Latent pose admission is surface-only')
    if args.query_requalification and args.mesh_source != 'solid':
        raise ValueError('Query requalification is solid-only')
    root = Path(os.environ["WR_ROOT"])
    inputs = _validate_inputs(root, episode_index=args.episode)
    import cv2
    import h5py
    import joblib
    import numpy as np
    from PIL import Image
    import trimesh
    vendor = root / "vendor/video_to_data"
    _pinned_checkout(vendor, UPSTREAM_REVISION)
    native_root = vendor / "reconstruction/modules/v2d_cari4d/lib/cari4d"
    if sha256(native_root / "prep/mhr_depth_h5.py") != "1429760952205d35c87157c05941defa20dc450f2b014441ca7cd5d39b45b0c5":
        raise RuntimeError("Exact qualified native depth-writer source required")
    sys.path.insert(0, str(native_root))
    from prep.prepare_mhr_wild_export import prepare_mhr_wild_export
    from prep.mhr_depth_h5 import DepthFrameRecord, MHRDepthH5Writer, validate_depth_h5, read_metric_depth
    from prep.mhr_depth_backend import MOGE2_MODEL_ID, MOGE2_MODEL_REVISION, MOGE2_SOURCE_COMMIT
    from prep.mhr_export_utils import MHR_CAMERA_NAMES, frame_names, read_rgb, read_mask, camera_calibration, load_edex
    base = episode_output(root, args.episode)
    output = base / "cari_inputs"
    if output.exists():
        raise RuntimeError("Frozen CARI inputs exist; never overwrite")
    reports = {}
    report_paths = {"body": base / "body_full/report.json", "depth": base / "depth_full/report.json",
                    "object": base / "object_pose_full/report.json", "alignment": base / "scale_smoke/report.json",
                    "adapter": base / "body_full/cari_adapter/report.json"}
    if args.mesh_source in ('solid','surface'):
        report_paths['object'] = base / ('object_pose_full_'+args.mesh_source) / 'report.json'
    if args.allow_unobserved_poses:
        report_paths['object'] = base / 'object_pose_full_surface_latent/report.json'
    for key, path in report_paths.items():
        record = json.loads(path.read_text())
        if type(record.get("episode_index", args.episode)) is not int or record.get("episode_index", args.episode) != args.episode:
            raise RuntimeError(f"{key} report belongs to another episode")
        expected = {"status": "pass", "input_track": "track_1", "ground_truth_used": False,
                    "hand_labeled_test": False, "oracle_modes": []}
        if (any(record.get(field) != value for field, value in expected.items())
                or record["ground_truth_used"] is not False or record["hand_labeled_test"] is not False):
            raise RuntimeError(f"{key} must have explicitly verified no-oracle provenance")
        if key != "adapter" and record.get("input_sha256") != inputs["video_sha256"]:
            raise RuntimeError(f"{key} belongs to another original video")
        reports[key] = record
    expected_stages = {"body": "sam3d_body_full_video_initializer", "depth": "monocular_moge2_full_video",
                       "object": "fixed_scale_full_object_pose_initializer", "alignment": "predicted_human_anchored_moge2_pointmaps",
                       "adapter": "native_cari_body_adapter_full_video"}
    if any(reports[key]["stage"] != value for key, value in expected_stages.items()):
        raise RuntimeError("Require exact full initializer stages, not sparse/test replacements")
    if reports["adapter"]["body_report_sha256"] != sha256(report_paths["body"]):
        raise RuntimeError("CARI Body adapter no longer matches verified original Body source")
    if reports["object"]["full_depth_report_sha256"] != sha256(report_paths["depth"]):
        raise RuntimeError("Object poses no longer match full inferred depth")
    if reports["object"]["alignment_report_sha256"] != sha256(report_paths["alignment"]):
        raise RuntimeError("Object poses no longer match shared human depth gauge")
    adapter_path = base / "body_full/cari_adapter/canonical_initializer.pkl"
    if sha256(adapter_path) != reports["adapter"]["canonical_initializer_sha256"]:
        raise RuntimeError("Canonical human initializer changed")
    pose_path = base / "object_pose_full/geometry_and_poses.npz"
    if args.mesh_source in ('solid','surface'):
        pose_path = base / ('object_pose_full_'+args.mesh_source) / 'geometry_and_poses.npz'
    if args.allow_unobserved_poses:
        pose_path = base / 'object_pose_full_surface_latent/geometry_and_poses.npz'
    if sha256(pose_path) != reports["object"]["geometry_and_poses_sha256"]:
        raise RuntimeError("Frozen object geometry/poses changed")
    count = inputs["total_frames"]
    names = [f"{index:06d}" for index in range(count)]
    body_frames = {record["frame_index"]: record for record in reports["body"]["frames"]}
    depth_frames = {record["frame_index"]: record for record in reports["depth"]["frames"]}
    object_frames = {record["frame_index"]: record for record in reports["object"]["frames"]}
    for key, records in (("body", body_frames), ("depth", depth_frames), ("object", object_frames)):
        if len(records) != len(reports[key]["frames"]) or sorted(records) != list(range(count)):
            raise RuntimeError(f"{key} lacks exact full original-frame coverage")
    pose_observed = None
    if args.mesh_source == 'surface':
        from surface_pose_report_capacity import identity as surface_identity, recheck as surface_recheck
        vertices, faces, active, rotations, translations, surface_compact, surface_topology, surface_ledger = _surface_preflight(
            root,args.episode,inputs,reports['object'],pose_path,np,
            **(dict(allow_unobserved_poses=True) if args.allow_unobserved_poses else {}))
        if args.allow_unobserved_poses:
            with np.load(pose_path, allow_pickle=False) as data:
                pose_observed = _latent_observations(reports['object'], data['pose_observed'], count, np)
        surface_native_load, native_source_ledger = _solid_native_sources(native_root,trimesh,np)
        surface_ledger.update(native_source_ledger)
        native_pins={'prep/prepare_mhr_wild_export.py':{'bytes':13145,'sha256':'b465516cc96a8c5472aec995cff12e32a9d033c7c5157a6a601b96e332e45f4f'},
            'prep/mhr_export_utils.py':{'bytes':25383,'sha256':'a9f499dad2f73eb7b8c526f33c94a785cc9468423760afcf6e5ced46d2f49e3b'}}
        for name,pin in native_pins.items():
            path=native_root/name
            if surface_identity(path,readonly=False)!=pin:
                raise ValueError('Original native surface alignment/export source differs')
            surface_ledger[path]=pin
    if args.mesh_source == 'solid':
        vertices, faces, active, rotations, translations, solid_compact, solid_topology, solid_ledger = _solid_preflight(
            root, args.episode, inputs, reports['object'], pose_path, np,
            **(dict(query_requalification=True) if args.query_requalification else {}))
        from solid_geometry_loader import identity as solid_identity
        native_pins = {'prep/prepare_mhr_wild_export.py': {'bytes': 13145, 'sha256': 'b465516cc96a8c5472aec995cff12e32a9d033c7c5157a6a601b96e332e45f4f'},
            'prep/mhr_export_utils.py': {'bytes': 25383, 'sha256': 'a9f499dad2f73eb7b8c526f33c94a785cc9468423760afcf6e5ced46d2f49e3b'}}
        for name, pin in native_pins.items():
            path = native_root / name
            if solid_identity(path, readonly=False) != pin:
                raise ValueError('Native unprocessed scene alignment/export source differs from pinned upstream')
            solid_ledger[path] = pin
        solid_native_load, native_source_ledger = _solid_native_sources(native_root, trimesh, np)
        solid_ledger.update(native_source_ledger)
    elif args.mesh_source != 'surface':
        with np.load(pose_path, allow_pickle=False) as arrays:
            vertices, faces = arrays["vertices"].copy(), arrays["faces"].copy()
            rotations, translations = arrays["rotation"].copy(), arrays["translation"].copy()
            if not np.array_equal(arrays["frame_index"], np.arange(count)) or float(arrays["object_scale"]) != 1.:
                raise RuntimeError("Require full fixed-metric-gauge mesh with scale already baked once")
    if (rotations.shape != (count, 3, 3) or translations.shape != (count, 3)
            or not np.isfinite(rotations).all() or not np.isfinite(translations).all()
            or not np.allclose(rotations @ rotations.swapaxes(-1, -2), np.eye(3), atol=1e-5, rtol=0)
            or not np.allclose(np.linalg.det(rotations), 1, atol=1e-5, rtol=0)):
        raise RuntimeError("Full object trajectory must contain proper finite rigid poses")
    if args.mesh_source == 'surface':
        surface_ledger.update({p:surface_identity(p,readonly=False) for p in (*report_paths.values(),adapter_path)})
        from surface_geometry_loader import strict_json as surface_json
        if any(surface_json(path.read_bytes())!=reports[key]
                or surface_identity(path,readonly=False)!=surface_ledger[path] for key,path in report_paths.items()):
            raise ValueError('Surface initializer reports changed before output reservation')
        surface_recheck(surface_ledger)
        metric_mesh=trimesh.Trimesh(*surface_compact,process=False)
    elif args.mesh_source == 'solid':
        for path in (*report_paths.values(), adapter_path):
            solid_ledger[path] = solid_identity(path, readonly=False)
        if any(sha256(report_paths[key]) != solid_ledger[report_paths[key]]['sha256'] or
               reports[key] != json.loads(report_paths[key].read_text()) for key in reports):
            raise ValueError('Full initializer reports changed before preparing solid inputs')
        metric_mesh = trimesh.Trimesh(*solid_compact, process=False)
    else:
        active, _ = normalize_degenerate_faces(vertices, faces)
        metric_mesh = trimesh.Trimesh(vertices, faces[active], process=True)
    if args.mesh_source!='surface' and (not metric_mesh.is_watertight or not metric_mesh.is_winding_consistent or metric_mesh.volume <= 0):
        raise RuntimeError("Packed fixed geometry must remain closed and correctly oriented")
    output.mkdir(exist_ok=False)
    started = time.perf_counter()
    sequence = f"episode_{args.episode:06d}"
    video_link = output / (sequence + ".0.color.mp4")
    # Native prep resolves symlinks before validating its .0.color.mp4 ABI.
    # Same managed disk: a hardlink preserves bytes without duplicating video.
    try:
        os.link(inputs["video"], video_link)
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise
        # Docker bind mounts are separate mountpoints even on one disk.
        # Copy only within Azure, never across the laptop/tethered connection.
        shutil.copyfile(inputs["video"], video_link)
    if sha256(video_link) != inputs["video_sha256"]:
        raise RuntimeError("Native video alias differs from original Track 1 bytes")
    metric_path = output / "object_metric.glb"
    metric_mesh.export(metric_path)
    if args.mesh_source == 'surface':
        surface_metric_proof,_=_surface_serialized_mesh(metric_path,surface_compact,trimesh,np,surface_native_load)
        surface_metric_identity=surface_identity(metric_path,readonly=False)
    elif args.mesh_source == 'solid':
        metric_proof, _ = _solid_serialized_mesh(metric_path, solid_compact, solid_topology, trimesh, np, solid_native_load)
        metric_identity = solid_identity(metric_path, readonly=False)
    focal = float(np.hypot(1152, 1536))
    intrinsics = {"fx": focal, "fy": focal, "cx": 768., "cy": 576., "H": 1152, "W": 1536,
                  "depth_backend": "moge2", "model_id": MOGE2_MODEL_ID, "model_revision": MOGE2_MODEL_REVISION,
                  "source_commit": MOGE2_SOURCE_COMMIT, "camera_policy": "original_RGB_size_prior_K_no_GT_calibration"}
    intrinsic_path = output / "intrinsics.pkl"
    joblib.dump(intrinsics, intrinsic_path)
    masks_path = output / "automatic_masks.h5"
    with h5py.File(masks_path, "x") as handle:
        for index, name in enumerate(names):
            for mask_id, kind in ((0, "person_mask.png"), (1, "obj_rend_mask.png")):
                path = base / f"automatic_masks/masks/{mask_id}/{name}.png"
                expected_hash = body_frames[index]["mask_sha256"] if mask_id == 0 else object_frames[index]["object_mask_sha256"]
                if sha256(path) != expected_hash:
                    raise RuntimeError("Automatic mask changed after body/object prediction")
                with Image.open(path) as image:
                    array = np.asarray(image)
                if array.shape != (1152, 1536) or not np.isin(array, [0, 255]).all() or not (array > 0).any():
                    raise RuntimeError("Automatic mask must retain original binary full-resolution observation")
                handle.create_dataset(f"{sequence}/{name}-k0.{kind}", data=array, compression="lzf")
    export_seq = prepare_mhr_wild_export(video_link, masks_path, metric_path, intrinsic_path, output / "export")
    metadata_path = export_seq / "wild_export.json"
    metadata = json.loads(metadata_path.read_text())
    A = np.asarray(metadata["source_object_mesh_to_aligned_transform"], dtype=np.float64)
    if A.shape != (4, 4) or not np.allclose(A[3], [0, 0, 0, 1]) or not np.allclose(A[:3, :3] @ A[:3, :3].T, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(A[:3, :3]), 1, atol=1e-5):
        raise RuntimeError("Native mesh preparation must be a proper rigid frame change, not rescaling")
    poses = np.broadcast_to(np.eye(4), (count, 4, 4)).copy()
    poses[:, :3, :3], poses[:, :3, 3] = rotations, translations
    aligned_poses = poses @ np.linalg.inv(A)
    original_points = vertices[faces[active].reshape(-1)]
    aligned_points = original_points @ A[:3, :3].T + A[:3, 3]
    frame_transform_error = 0.
    for index in range(count):
        before = original_points @ poses[index, :3, :3].T + poses[index, :3, 3]
        after = aligned_points @ aligned_poses[index, :3, :3].T + aligned_poses[index, :3, 3]
        frame_transform_error = max(frame_transform_error, float(np.max(np.linalg.norm(after - before, axis=-1))))
    if frame_transform_error > 1e-5:
        raise RuntimeError("Aligned mesh/pose pair changed camera-space geometry")
    if args.mesh_source == 'solid':
        if not np.array_equal(A.astype(np.float32).astype(np.float64), A) or not np.array_equal(A[3], [0., 0., 0., 1.]):
            raise ValueError('Native aligned transform must preserve its actual float32 affine metadata')
        aligned_path = output / 'export' / sequence / 'object_mesh/output_aligned.glb'
        if metadata.get('object_mesh_file') != str(aligned_path) or export_seq != aligned_path.parent.parent:
            raise ValueError('Native solid export must retain the canonical prepared aligned-GLB route')
        aligned_proof, native_aligned_mesh = _solid_serialized_mesh(
            aligned_path, solid_compact, solid_topology, trimesh, np, solid_native_load, A)
    if args.mesh_source == 'surface':
        if not np.array_equal(A.astype(np.float32).astype(np.float64),A) or not np.array_equal(A[3],[0.,0.,0.,1.]):
            raise ValueError('Native surface alignment must retain actual F32 affine metadata')
        aligned_path=output/'export'/sequence/'object_mesh/output_aligned.glb'
        if metadata.get('object_mesh_file')!=str(aligned_path) or export_seq!=aligned_path.parent.parent:
            raise ValueError('Native surface export changed its canonical aligned-GLB route')
        surface_aligned_proof,surface_native_aligned=_surface_serialized_mesh(
            aligned_path,surface_compact,trimesh,np,surface_native_load,A)
    object_poses_path = output / "own_object_poses.pkl"
    if args.allow_unobserved_poses:
        pose_metadata = dict(source='World_Reward_fixed_scale_depth_ICP_Viterbi_not_FoundationPose',
            ground_truth_used=False, hand_labeled_test=False, oracle_modes=[],
            source_pose_sha256=sha256(pose_path), mesh_frame_change=A.tolist())
        pose_metadata.update(_latent_pose_metadata(pose_observed, np))
        joblib.dump(dict(frames=names, obj_pose_world=aligned_poses.astype(np.float32), metadata=pose_metadata), object_poses_path)
    else:
        joblib.dump({"frames": names, "obj_pose_world": aligned_poses.astype(np.float32),
                     "metadata": {"source": "World_Reward_fixed_scale_depth_ICP_Viterbi_not_FoundationPose",
                                  "ground_truth_used": False, "hand_labeled_test": False, "oracle_modes": [],
                                  "source_pose_sha256": sha256(pose_path), "mesh_frame_change": A.tolist()}}, object_poses_path)
    if args.mesh_source == 'solid':
        saved_object_poses = joblib.load(object_poses_path)
        if (type(saved_object_poses) is not dict or set(saved_object_poses) != {'frames', 'obj_pose_world', 'metadata'}
                or saved_object_poses['frames'] != names
                or not np.array_equal(saved_object_poses['obj_pose_world'], aligned_poses.astype(np.float32))
                or saved_object_poses['metadata'] != {'source': 'World_Reward_fixed_scale_depth_ICP_Viterbi_not_FoundationPose',
                    'ground_truth_used': False, 'hand_labeled_test': False, 'oracle_modes': [],
                    'source_pose_sha256': sha256(pose_path), 'mesh_frame_change': A.tolist()}):
            raise ValueError('Saved native full-timeline solid poses/metadata changed')
        aligned_proof['represented_mesh_pose_frame_roundtrip_max_error_m'] = _solid_camera_roundtrip(
            solid_compact, rotations, translations, saved_object_poses['obj_pose_world'], native_aligned_mesh, np)
    if args.mesh_source == 'surface':
        from world_reward.surface_pose_geometry import camera_roundtrip
        saved=joblib.load(object_poses_path)
        _surface_saved_poses(saved,names,aligned_poses,sha256(pose_path),A,np,
            **(dict(pose_observed=pose_observed) if args.allow_unobserved_poses else {}))
        surface_aligned_proof['represented_mesh_pose_frame_roundtrip_max_error_m']=camera_roundtrip(
            *surface_compact,rotations,translations,saved['obj_pose_world'],*surface_native_aligned)
    aligned_depth_path = output / "aligned_depth.h5"
    scale = reports["alignment"]["depth_alignment"]["shared_scale"]
    identity = {"depth_backend": "moge2", "depth_model_id": MOGE2_MODEL_ID, "depth_model_revision": MOGE2_MODEL_REVISION,
                "depth_source_commit": MOGE2_SOURCE_COMMIT, "alignment_report_sha256": sha256(report_paths["alignment"]),
                "monocular_depth": {"backend": "moge2", "model_id": MOGE2_MODEL_ID, "model_revision": MOGE2_MODEL_REVISION,
                                    "source_commit": MOGE2_SOURCE_COMMIT},
                "ground_truth_used": False, "alignment": "one_predicted_human_anchored_clip_scalar_no_offset"}
    camera_name = MHR_CAMERA_NAMES[0]
    with MHRDepthH5Writer(aligned_depth_path, {camera_name: names}, alignment_method="world_reward_shared_predicted_human_scale",
                          alignment_input_identity=identity, encoding_workers=8) as writer:
        pending = []
        for index, name in enumerate(names):
            path = base / f"depth_full/{name}.npz"
            if sha256(path) != depth_frames[index]["output_sha256"]:
                raise RuntimeError("Full depth artifact changed")
            with np.load(path, allow_pickle=False) as arrays:
                depth, valid = arrays["depth"].copy(), arrays["mask"].copy()
            raw = np.where(valid, depth, 0.)
            aligned = raw * scale
            if not np.isfinite(aligned).all() or (aligned < 0).any() or (raw > 65.535).any() or (aligned > 65.535).any():
                raise RuntimeError("Depth encoding would silently saturate uint16 metres-to-mm representation")
            pending.append(DepthFrameRecord(index, raw, aligned, scale, 0., int(valid.sum())))
            if len(pending) == 8:
                writer.write_frames(camera_name, pending)
                pending = []
            if (index + 1) % 50 == 0:
                print(json.dumps({"stage": "cari_prepare_depth", "frames_submitted": index + 1, "frames_complete": index + 1 - len(pending)}), flush=True)
        if pending:
            writer.write_frames(camera_name, pending)
        writer.mark_complete()
    depth_validation = validate_depth_h5(aligned_depth_path, expected_cameras=[camera_name], expected_alignment_input_identity=identity,
                                         validation_workers=8)
    if frame_names(export_seq) != names:
        raise RuntimeError("Native RGB export changed original frame identities")
    K, extrinsic = camera_calibration(load_edex(export_seq), 0)
    if not np.array_equal(extrinsic, np.eye(4)) or not np.allclose(K, [[focal, 0, 768], [0, focal, 576], [0, 0, 1]], atol=1e-5):
        raise RuntimeError("Native export changed the immutable inferred camera")
    # Original RGB decode hash checked before official JPEG export; quantify,
    # do not pretend JPEG q100 is byte-identical to original model observations.
    cap = cv2.VideoCapture(str(inputs["video"]))
    jpeg_errors = []
    try:
        for index, name in enumerate(names):
            ok, bgr = cap.read()
            if not ok:
                raise RuntimeError("Original RGB verification decode failed")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if hashlib.sha256(rgb.tobytes()).hexdigest() != body_frames[index]["decoded_rgb_sha256"] or body_frames[index]["decoded_rgb_sha256"] != depth_frames[index]["decoded_rgb_sha256"]:
                raise RuntimeError("Prepared RGB must derive from the same Body/depth original video")
            exported = read_rgb(export_seq, 0, name)
            jpeg_errors.append(float(np.mean(np.abs(exported.astype(float) - rgb))))
            for mask_id, kind in ((0, "human"), (1, "object")):
                with Image.open(base / f"automatic_masks/masks/{mask_id}/{name}.png") as image:
                    expected = np.asarray(image) > 0
                if not np.array_equal(read_mask(export_seq, kind, 0, name), expected):
                    raise RuntimeError("Native mask export changed automatic observations")
    finally:
        cap.release()
    for name in names:
        quantized = read_metric_depth(aligned_depth_path, "aligned", camera_name, name)
        if quantized.shape != (1152, 1536) or not np.isfinite(quantized).all():
            raise RuntimeError("Native quantized depth decode contract failed")
    result = {"stage": "world_reward_native_cari_inputs", "status": "pass", "episode_index": args.episode, "frames": count,
              "export_seq": str(export_seq), "depth_h5": str(aligned_depth_path), "mhr_init": str(adapter_path),
              "object_poses": str(object_poses_path), "object_pose_initializer": "own_ICP_Viterbi_not_FoundationPose",
              "depth_validation": depth_validation, "mesh_pose_frame_roundtrip_max_error_m": frame_transform_error,
              "jpeg_original_RGB_mean_absolute_error": float(np.mean(jpeg_errors)), "original_frame_coverage_verified": True,
              "input_track": "track_1", "input_sha256": inputs["video_sha256"],
              "input_dataset_revision": inputs["dataset_revision"], "ground_truth_used": False,
              "hand_labeled_test": False, "oracle_modes": [], "submission_eligible": False,
              "challenge_performance_verified": False, "producer_revision": os.environ.get("WR_CODE_REVISION"),
              "input_report_sha256": {key: sha256(path) for key, path in report_paths.items()},
              "file_sha256": {key: sha256(path) for key, path in {"depth_h5": aligned_depth_path, "mhr_init": adapter_path,
                                                               "object_poses": object_poses_path, "wild_export": metadata_path}.items()},
              "elapsed_seconds": time.perf_counter() - started, "script_sha256": sha256(Path(__file__))}
    result["depth_encoding"] = {"batch_size": 8, "encoding_workers": 8,
        "native_source_sha256": "1429760952205d35c87157c05941defa20dc450f2b014441ca7cd5d39b45b0c5",
        "compression_or_quantization_changed": False, "validation_changed": False}
    if args.mesh_source == 'solid':
        _solid_recheck(solid_ledger)
        _pinned_checkout(vendor, UPSTREAM_REVISION)
        if (solid_identity(metric_path, readonly=False) != metric_identity
                or metadata['source_object_mesh'] != {'path': str(metric_path.resolve()),
                    'size': metric_path.stat().st_size, 'mtime_ns': metric_path.stat().st_mtime_ns}):
            raise ValueError('Native solid metric GLB source changed during preparation')
        final_metric, _ = _solid_serialized_mesh(metric_path, solid_compact, solid_topology, trimesh, np, solid_native_load)
        final_aligned, _ = _solid_serialized_mesh(aligned_path, solid_compact, solid_topology, trimesh, np, solid_native_load, A)
        if (final_metric != metric_proof or any(final_aligned[key] != aligned_proof[key] for key in final_aligned)):
            raise ValueError('Native raw/effective/FP32 solid geometry changed after preparation')
        result['object_source'] = 'solid'
        if args.query_requalification: result['query_requalification'] = reports['object']['query_requalification']
        result['object_pose_source'] = {'report': str(report_paths['object'].relative_to(root)),
            'geometry_and_poses': str(pose_path.relative_to(root)), 'geometry_and_poses_sha256': sha256(pose_path)}
        result['solid_geometry_validation'] = {'metric_glb': metric_proof, 'native_aligned_glb': aligned_proof,
            'source_rehashed_after': True, 'files': {str(p): value for p, value in solid_ledger.items()},
            'local_coordinates_or_topology_repaired': False,
            'native_rigidprojection_used': aligned_proof['native_rigidprojection_used'],
            'metric_scale_applied_again': False}
    if args.mesh_source == 'surface':
        surface_recheck(surface_ledger)
        _pinned_checkout(vendor,UPSTREAM_REVISION)
        if (surface_identity(metric_path,readonly=False)!=surface_metric_identity
                or metadata.get('source_object_mesh')!={'path':str(metric_path.resolve()),
                    'size':metric_path.stat().st_size,'mtime_ns':metric_path.stat().st_mtime_ns}):
            raise ValueError('Native surface metric GLB changed during preparation')
        final_metric,_=_surface_serialized_mesh(metric_path,surface_compact,trimesh,np,surface_native_load)
        final_aligned,_=_surface_serialized_mesh(aligned_path,surface_compact,trimesh,np,surface_native_load,A)
        if final_metric!=surface_metric_proof or any(final_aligned[k]!=surface_aligned_proof[k] for k in final_aligned):
            raise ValueError('Surface geometry changed after preparation')
        result['object_source']='surface'
        result['object_pose_source']=dict(report=str(report_paths['object'].relative_to(root)),
            geometry_and_poses=str(pose_path.relative_to(root)),geometry_and_poses_sha256=sha256(pose_path))
        fields=('committed_pins_sha256','producer_report_sha256','cpu_native_report_sha256','source_domain','metric_scale_baked_once','geometry_operations_applied')
        result['surface_geometry_validation']={k:reports['object']['topology_budget'][k] for k in fields}
        result['surface_geometry_validation'].update(metric_glb=surface_metric_proof,native_aligned_glb=surface_aligned_proof,
            source_rehashed_after=True,files={str(p):pin for p,pin in surface_ledger.items()})
    if args.allow_unobserved_poses:
        result['object_pose_observations'] = _latent_pose_metadata(pose_observed, np)
        result['allow_unobserved_poses'] = True
    (output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    # Transient combined masks are redundant after verified native export.
    masks_path.unlink()
    print(json.dumps({"stage": result["stage"], "status": "pass", "frames": count,
                      "mesh_pose_error_m": frame_transform_error, "elapsed_seconds": result["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
