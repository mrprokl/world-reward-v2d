"""Actual SAME-SAM authored conditional point study, never challenge inference.

Caller authenticates its complete current source, native source/assets and B47
lease. This module reuses the original MHR decoder/optimizer/render contracts;
known geometry, masks, camera and initialization are explicit experimental inputs.
"""
from __future__ import annotations

from dataclasses import asdict
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import sys
import time

ROOT = Path('/srv/scenesmith/world-reward')
IMAGE = 'sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7'
STAGES = {'manufacture': 'articulated_point_manufacture_native_v1',
          'fit': 'articulated_point_fit_native_v1'}
PUBLIC_SCHEMA = 'world_reward.articulated_point_public.v1'
SOURCE_HELPERS = ('infra/articulated_point_native.py', 'infra/joint_point_authored_qualify.py',
    'infra/joint_point_native_qualify.py', 'infra/cari_full_refine.py',
    'src/world_reward/articulated_point_cohort.py', 'src/world_reward/authored_point_study.py',
    'src/world_reward/joint_point_objective.py', 'src/world_reward/joint_point_evidence.py',
    'src/world_reward/fixed_shape_point_pose.py', 'src/world_reward/point_surface_queries.py')
_DYNAMIC = ('stage', 'manufacture', 'tracks')


def policy_sha256(c):
    value = {k: v for k, v in c['articulated_study'].items() if k not in _DYNAMIC}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def _previous(rt, c, kind, stage):
    """Previous independently sealed phase; no model/private imports or reads."""
    binding = c['articulated_study'][kind]
    rt.require(type(binding) is dict and set(binding) == {'path', 'report'}, 'Exact sealed phase binding required')
    path = rt.canonical(Path(binding['path']))
    revision = c['_articulated_proof']['source_binding']['producer_revision']
    rt.require(path == ROOT/'validation/articulated_point_study_v1'/revision/kind/'native', 'Exact stage namespace required')
    # Docker creates mount ancestors independently. Host checks the sealed555
    # directory; offline consumers get ONLY readonly allowlisted leaves, so the
    # synthetic ancestor mode is not evidence and must not expose private data.
    rt.require(path.is_dir(), 'Original phase namespace required')
    if os.geteuid() == 0:
        rt.require(path.stat().st_mode & 0o777 == 0o555, 'Host sealed phase directory required')
    r = rt.pinned(path/'report.json', binding['report'], 1 << 20)
    rt.require(r['stage'] == stage and r['status'] == 'pass' and r['phase'] == 'complete'
        and r['policy_sha256'] == policy_sha256(c) and r['frames'] == 144
        and r['source_binding'] == c['_articulated_proof']['source_binding']
        and r['challenge_inputs_used'] is False and r['adoption'] is False, 'Frozen previous phase differs')
    rt.require(type(r['outputs']) is dict and r['outputs']
        and all(type(n) is str and Path(n).name == n and n not in ('.', '..') for n in r['outputs']), 'Bounded phase leaf inventory required')
    return path, r


def authenticate_manufacture(rt, c):
    path, r = _previous(rt, c, 'manufacture', STAGES['manufacture'])
    from world_reward.articulated_point_cohort import SCENES
    rt.require(r['scene_ids'] == [x[0] for x in SCENES] and r['tracker_executed'] is False
        and r['truth_synthetic_known'] is True and r['conditional_not_end_to_end'] is True,
        'Exact fresh six-scene conditional manufacture required')
    expected = {'mhr_metadata.npz'} | {f'scene_{i:02d}_{suffix}' for i in range(6)
        for suffix in ('public.npz', 'private.npz', 'source.pth', 'object.obj')}
    rt.require(set(r['outputs']) == expected, 'Complete sealed manufacture manifest required')
    views = r['public_views']
    rt.require(type(views) is list and len(views) == 6, 'Six original public views required')
    for i, row in enumerate(views):
        name = f'scene_{i:02d}_public.npz'
        rt.require(row == dict(scene_index=i, scene_id=SCENES[i][0], path=name,
            schema=PUBLIC_SCHEMA, identity=r['outputs'][name]), 'Original public view association differs')
        rt.require(rt.identity(path/name, 512 << 20) == row['identity'], 'Public view changed')
    return path, r


def _save(base, path, arrays):
    import numpy as np
    base.retain(path, lambda stream: np.savez(stream, **arrays))


def _load_npz(rt, path, pin, limit=512 << 20):
    import numpy as np
    rt.require(rt.identity(path, limit) == pin, 'Sealed phase payload differs before read')
    with np.load(path, allow_pickle=False) as archive:
        if len(set(archive.files)) != len(archive.files): raise ValueError('Duplicate NPZ fields')
        value = {name: archive[name].copy() for name in archive.files}
    rt.require(rt.identity(path, limit) == pin, 'Phase payload changed while reading')
    return value


def _cfg(optimizer):
    return optimizer.MHRParityPostOptConfig(
        penetration_collision_proxy_path=str(ROOT/'weights/cari4d/refinement/mhr_collision_proxy_4000v.npz'),
        hand_surface_spec_path=str(ROOT/'weights/cari4d/refinement/mhr_hand_surface_spec.npz'),
        report_every=100, checkpoint_path=None)


def _reset(np, torch):
    random.seed(0); np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)


def _actual_controls(torch, layer, params, *, enforce_hard=True):
    """Original public head return_model_params; exact values presented to JIT."""
    from world_reward.articulated_point_cohort import validate_reconstituted_controls
    inputs = {k: torch.from_numpy(v).cuda() for k, v in params.items()}
    backend = layer.backend
    context = backend._vertices_context(inputs, detach_fixed=False)
    trans, body, shape, scale = backend._mutable_vertices_inputs(inputs, context)
    head = context.head
    with torch.no_grad():
        _, native204 = head.mhr_forward(global_trans=torch.zeros_like(trans),
            global_rot=context.global_rot, body_pose_params=body, hand_pose_params=context.hand,
            scale_params=scale, shape_params=shape, expr_params=context.face,
            return_keypoints=False, return_joint_coords=False, return_model_params=True,
            return_joint_rotations=False)
    original = torch.cat((native204, shape), dim=1).detach().cpu().numpy().copy()
    bounds = head.mhr.get_parameter_limits().detach().cpu().numpy().copy()
    if enforce_hard:
        violations = validate_reconstituted_controls(original, bounds)
    else:
        import numpy as np
        if original.shape != (24, 249) or original.dtype != np.float32 or not np.isfinite(original).all():
            raise ValueError('Full finite actual fitted249 controls required')
        names = list(head.mhr.get_parameter_names())
        bad = (original < bounds[:, 0]) | (original > bounds[:, 1])
        violations = dict(total=int(bad.sum()), by_parameter=[])
        for i in np.flatnonzero(bad.any(axis=0)):
            excess = np.maximum(bounds[i, 0]-original[:, i], original[:, i]-bounds[i, 1])
            violations['by_parameter'].append(dict(parameter_index=int(i), parameter_name=names[int(i)],
                original_frame_indices=np.flatnonzero(bad[:, i]).tolist(), lower=float(bounds[i, 0]),
                upper=float(bounds[i, 1]), maximum_excess=float(excess[bad[:, i]].max())))
    return original, violations


def _mesh(base, optimizer, path, pad):
    raw = ''.join('v '+' '.join(format(float(x), '.17g') for x in v)+'\n' for v in pad.vertices)
    raw += ''.join('f '+' '.join(str(int(x)+1) for x in f)+'\n' for f in pad.faces)
    base.write(path, raw.encode())
    v, f = optimizer._load_object_vertices(path)
    import numpy as np
    if v.dtype != np.float32 or f.dtype != np.int64 or not np.array_equal(v, pad.vertices) or not np.array_equal(f, pad.faces):
        raise ValueError('Actual native pad mesh loader changed original geometry')
    return v, f


def _manufacture(rt, code, out, c, layer, optimizer, spec, r, persist, check):
    import numpy as np
    import torch
    import cari_full_refine as full
    import joint_point_authored_qualify as base
    from world_reward import authored_point_study as old, articulated_point_cohort as cohort
    from world_reward.point_surface_queries import canonical_mask_quantile_queries
    from lib_mhr.body_pose import compact_model_params_to_cont_body_np, compact_model_params_to_cont_hand_np
    from lib_mhr.postopt_crop import build_postopt_crop, stack_postopt_crops
    import Utils
    import nvdiffrast.torch as dr
    head = layer.backend._ensure_head(torch.device('cuda')); model = head.mhr
    cpu = lambda a: a.detach().cpu().numpy().copy()
    names, joints = list(model.get_parameter_names()), list(model.get_joint_names())
    bounds, faces = cpu(model.get_parameter_limits()), cpu(layer.mesh_faces(device='cuda'))
    idx, weights = map(cpu, model.get_lbsw())
    meta = dict(bounds=bounds, faces=faces, scale_mean=cpu(head.scale_mean),
        scale_comps=cpu(head.scale_comps), hand_pose_mean=cpu(head.hand_pose_mean),
        hand_pose_comps=cpu(head.hand_pose_comps), hand_left_indices=cpu(head.hand_joint_idxs_left),
        hand_right_indices=cpu(head.hand_joint_idxs_right), lbs_indices=idx, lbs_weights=weights)
    meta_hash = full.fingerprint(meta); _save(base, out/'mhr_metadata.npz', meta)
    K = np.asarray(c['articulated_study']['cohort']['K'], np.float64)
    rt.require(K.shape == (3, 3) and np.array_equal(K, [[800, 0, 320], [0, 800, 240], [0, 0, 1]]), 'Frozen authored K required')
    Ks = np.broadcast_to(K, (24, 3, 3)).copy(); context = dr.RasterizeCudaContext()
    r.update(scenes=[], public_views=[], native_decode_calls=0, native_decoded_frames=0,
        reconstitution_decode_calls=0, reconstitution_decoded_frames=0,
        primitive_render_calls=0, primitive_render_frames=0, tracker_executed=False)
    def render(v, f, colors):
        check(); r['primitive_render_calls'] += 1; r['primitive_render_frames'] += len(v)
        return Utils.nvdiff_color_depth_render(Ks, context, dict(pos=v[0],
            faces=torch.tensor(f, device='cuda', dtype=torch.int32),
            vertex_color=torch.tensor(colors, device='cuda', dtype=torch.float32)), (480, 640), v.contiguous())
    for i, (name, split, side) in enumerate(cohort.SCENES):
        r['phase'] = name+'_decode'; persist(); check()
        recipe = cohort.named_recipe(names, bounds, i, scale68=meta['scale_mean'], identity45=np.zeros(45, np.float32))
        params, mapping = base.study_parameters(np, recipe, head, compact_model_params_to_cont_body_np, compact_model_params_to_cont_hand_np)
        actual249, violations = _actual_controls(torch, layer, params)
        r['reconstitution_decode_calls'] += 1; r['reconstitution_decoded_frames'] += 24
        rt.require(np.array_equal(actual249[:, 136:204], recipe.parameters[:, 136:204])
            and np.array_equal(actual249[:, 204:], recipe.identity), 'Unchanged SAM mean/identity required')
        with torch.no_grad(): decoded = layer.mhr_forward({k: torch.from_numpy(v).cuda() for k, v in params.items()})
        r['native_decode_calls'] += 1; r['native_decoded_frames'] += 24
        arrays = {key: cpu(getattr(decoded, key)) for key in ('vertices', 'joints', 'keypoints', 'joint_global_rots')}
        if any(a.dtype != np.float32 or len(a) != 24 or not np.isfinite(a).all() for a in arrays.values()):
            raise ValueError('All original24 actual finite decoded frames required')
        flip = np.diag([1., -1., -1.]); rotation = flip@arrays['joint_global_rots'].astype(np.float64)@flip
        articulation = old.articulation_metrics(arrays['joints'], rotation, joints, side)
        hand = np.asarray(spec.vertex_indices[0 if side == 'l' else 1], np.int64)
        face = old.select_hand_triangle(faces, hand, idx, weights, joints, side)
        pad = old.pad_mesh(i); attachment = old.attach_pad(arrays['vertices'], faces, face)
        proximity = old.attachment_proximity(attachment, pad)
        v, f = _mesh(base, optimizer, out/f'scene_{i:02d}_object.obj', pad)
        pose = np.broadcast_to(np.eye(4, dtype=np.float32), (24, 4, 4)).copy()
        pose[:, :3, :3], pose[:, :3, 3] = attachment.rotation, attachment.translation
        initial = pose.copy(); initial[:, :3, 3] += recipe.perturbation
        human = decoded.vertices; world = torch.tensor(v, device='cuda')[None]@torch.tensor(pose[:, :3, :3], device='cuda').transpose(1, 2)+torch.tensor(pose[:, :3, 3], device='cuda')[:, None]
        if not bool((human[..., 2] > 0).all()) or not bool((world[..., 2] > 0).all()): raise ValueError('Every original vertex must have positive Z')
        hc, hz, _ = render(human, faces, np.broadcast_to(np.array([.7, .6, .5], np.float32), (human.shape[1], 3)).copy())
        oc, oz, _ = render(world, f, pad.colors)
        hvalid, ovalid = hz > 0, oz > 0
        if torch.any(hvalid & ovalid & (hz == oz)): raise ValueError('Equal-depth authored visibility ambiguous')
        om, hm = ovalid & (~hvalid | (oz < hz)), hvalid & (~ovalid | (hz < oz))
        rgb = torch.where(om[..., None], oc, torch.where(hm[..., None], hc, torch.zeros_like(hc)))
        depth = torch.where(om, oz, torch.where(hm, hz, torch.zeros_like(hz)))
        rgb, depth, om, hm = map(cpu, (rgb, depth, om, hm))
        rgb8 = np.rint(np.clip(rgb, 0, 1)*255).astype(np.uint8)
        crops = stack_postopt_crops([build_postopt_crop(a, b, K) for a, b in zip(hm, om)], K)
        observations = dict(human_mask=hm, object_mask=om, **crops)
        pr = dict(params, pose_abs=initial, contact_logits=np.broadcast_to(np.array([1., -1.] if side == 'l' else [-1., 1.], np.float32), (24, 2)).copy())
        metadata = dict(c['bundle']['metadata'], object_mesh=str(out/f'scene_{i:02d}_object.obj'))
        for key in ('object_pose_storage_to_training_transform', 'object_mesh_to_training_transform'): metadata[key] = np.eye(4, dtype=np.float32)
        source = dict(schema=c['bundle']['schema'], gt={}, frames=[f'{t:06d}' for t in range(24)],
            pr=pr, pr_initial=copy.deepcopy(pr), observations=observations, metadata=metadata)
        full.validate_source_bundle(source, out/f'scene_{i:02d}_object.obj', 24)
        authored_weights = torch.zeros((24, 2), device='cuda'); authored_weights[:, 0 if side == 'l' else 1] = 1
        effective, _, _ = optimizer._initial_contact_activation(authored_weights,
            decoded.vertices[:, torch.tensor(np.asarray(spec.vertex_indices).copy(), device='cuda'), :],
            torch.tensor(initial[:, :3, :3], device='cuda'), torch.tensor(initial[:, :3, 3], device='cuda'),
            torch.tensor(v, device='cuda'), torch.tensor(f, device='cuda'), _cfg(optimizer).contact_activation_distance_m)
        contact = bool((effective[:, 0 if side == 'l' else 1] > 0).all())
        r['phase'] = name+'_query'; persist(); check()
        from world_reward.point_surface_queries import MaskQueryError
        try:
            selected = canonical_mask_quantile_queries(v, f, initial[0, :3, :3], initial[0, :3, 3], K,
                om[0], np.isfinite(depth[0]) & (depth[0] > 0), image_width=640, image_height=480)
        except MaskQueryError as error:
            r['failed_scene'] = dict(scene_index=i, scene_id=name,
                query_diagnostics=error.diagnostics.scalar_report()); persist(); raise
        q = selected.queries
        texture = old.sentinel_texture_metrics(rgb8[np.array(old.SENTINELS)], om[np.array(old.SENTINELS)])
        observable = old.translation_observability(q.canonical_points@pose[0, :3, :3].T+pose[0, :3, 3], K)
        row = dict(scene_index=i, scene_id=name, split=split, side=side, hand_face_index=int(face),
            articulation=articulation, material_proximity=proximity, sentinel_metrics=texture,
            translation_observability=observable, query_diagnostics=selected.diagnostics.scalar_report(),
            soft_scale_violations=[dict(asdict(x), parameter_name=names[x.parameter_index], frames_affected=24) for x in violations], hand_parameter_mapping=mapping,
            effective_authored_side_all_frames=contact, contact_weights_authored_not_inferred=True,
            reconstructed_controls_sha256=full.fingerprint(actual249), source_bundle_sha256=full.fingerprint(source))
        r['scenes'].append(row); persist()
        if not contact or not all(x['passed'] for x in (articulation, proximity, texture, observable)): raise ValueError('Frozen authored scene gate failed; no rerender')
        public = dict(rgb=rgb8, frame_index=recipe.frame_index, source_frame_ids=recipe.frame_index,
            vertices=v, faces=f, K=K, initial_pose=initial, **vars(q))
        _save(base, out/f'scene_{i:02d}_public.npz', public)
        _save(base, out/f'scene_{i:02d}_private.npz', dict(pose=pose, **arrays,
            frame_index=recipe.frame_index, reconstituted_controls=actual249))
        base.retain(out/f'scene_{i:02d}_source.pth', lambda stream: torch.save(source, stream))
        pin = rt.identity(out/f'scene_{i:02d}_public.npz', 512 << 20)
        r['public_views'].append(dict(scene_index=i, scene_id=name, path=f'scene_{i:02d}_public.npz', schema=PUBLIC_SCHEMA, identity=pin))
        del decoded, human, world, hc, oc, source, observations, public
        gc.collect(); torch.cuda.empty_cache(); check()
    after = dict(bounds=cpu(model.get_parameter_limits()), faces=cpu(layer.mesh_faces(device='cuda')),
        scale_mean=cpu(head.scale_mean), scale_comps=cpu(head.scale_comps), hand_pose_mean=cpu(head.hand_pose_mean),
        hand_pose_comps=cpu(head.hand_pose_comps), hand_left_indices=cpu(head.hand_joint_idxs_left),
        hand_right_indices=cpu(head.hand_joint_idxs_right), lbs_indices=cpu(model.get_lbsw()[0]), lbs_weights=cpu(model.get_lbsw()[1]))
    rt.require(full.fingerprint(after) == meta_hash, 'Actual decoder metadata changed')
    r['native_metadata_rehashed_after'] = True


def _evidence(rt, c, path, mr, tracks_path, tr, i):
    from world_reward.point_surface_queries import SurfaceQueries
    from world_reward.fixed_shape_point_pose import PointTrackEvidence
    from world_reward.joint_point_evidence import bind_joint_point_evidence
    p = _load_npz(rt, path/f'scene_{i:02d}_public.npz', mr['outputs'][f'scene_{i:02d}_public.npz'])
    q = SurfaceQueries(**{key: p[key] for key in SurfaceQueries.__dataclass_fields__})
    for a in vars(q).values(): a.flags.writeable = False
    name = f'scene_{i:02d}_evidence.npz'
    raw = _load_npz(rt, tracks_path/name, tr['outputs'][name])
    tracks = PointTrackEvidence(**raw)
    bound = bind_joint_point_evidence(q, tracks, native_vertices=p['vertices'], native_faces=p['faces'],
        K=p['K'], image_size=(480, 640), frame_index=p['frame_index'], source_frame_ids=p['source_frame_ids'],
        native_frame_names=tuple(f'{t:06d}' for t in range(24)),
        source_references=('authored conditional known geometry and camera', 'Boots native fullT automatic tracks'))
    return p, bound


def _projection(np, p, truth):
    xyz = p['canonical_points'][None]@truth['pose'][:, :3, :3].swapaxes(-1, -2)+truth['pose'][:, None, :3, 3]
    if not np.isfinite(xyz).all() or np.any(xyz[..., 2] <= 0): raise ValueError('All original material projections require positive Z')
    K = p['K']; xy = xyz[..., :2]/xyz[..., 2, None]*[K[0, 0], K[1, 1]]+K[:2, 2]
    return xy*[256/640, 256/480]


def _truth(rt, path, mr, i):
    return _load_npz(rt, path/f'scene_{i:02d}_private.npz', mr['outputs'][f'scene_{i:02d}_private.npz'])


def _source(rt, base, full, torch, path, mr, i):
    name = f'scene_{i:02d}_source.pth'; pin = mr['outputs'][name]
    rt.require(rt.identity(path/name, 512 << 20) == pin, 'Own frozen source differs before load')
    value = torch.load(path/name, map_location='cpu', weights_only=False)
    rt.require(rt.identity(path/name, 512 << 20) == pin and full.fingerprint(value) == mr['scenes'][i]['source_bundle_sha256'], 'Own source byte contract differs')
    full.validate_source_bundle(value, path/f'scene_{i:02d}_object.obj', 24)
    return value


def _metrics(np, torch, layer, source, result, truth, evidence, optimizer, instance):
    with torch.no_grad(): decoded = layer.mhr_forward({k: torch.from_numpy(v).cuda() for k, v in result['pr'].items() if k.startswith('mhr_')})
    vertices, joints = decoded.vertices.cpu().numpy(), decoded.joints.cpu().numpy()
    pose = result['pr']['pose_abs']; truepose = truth['pose']
    def second_error(a, b): return float(np.linalg.norm(np.diff(a, n=2, axis=0)-np.diff(b, n=2, axis=0), axis=-1).mean())
    fitted249, violations = _actual_controls(torch, layer, {k: v for k, v in result['pr'].items() if k.startswith('mhr_')}, enforce_hard=False)
    indices = torch.arange(24, device='cuda')
    with torch.no_grad(): _, diagnostics = instance.loss(indices, 181, include_diagnostics=True)
    pen = {k: float(v.detach().cpu()) for k, v in diagnostics.items() if 'penetr' in k.lower()}
    if not pen: raise ValueError('Actual native penetration proxy diagnostic missing')
    translation = np.linalg.norm(pose[:, :3, 3]-truepose[:, :3, 3], axis=-1)
    return dict(object_translation_mean_m=float(translation.mean()), object_translation_max_m=float(translation.max()),
        human_PVE_mean_m=float(np.linalg.norm(vertices-truth['vertices'], axis=-1).mean()),
        human_MPJPE_mean_m=float(np.linalg.norm(joints-truth['joints'], axis=-1).mean()),
        object_translation_second_difference_error_m_per_frame2=second_error(pose[:, :3, 3], truepose[:, :3, 3]),
        human_joint_second_difference_error_m_per_frame2=second_error(joints, truth['joints']),
        native_penetration_proxy=pen, full_frames=24, aligned_per_frame=False,
        fitted_control_violations=violations, fitted_controls_sha256=hashlib.sha256(fitted249.tobytes()).hexdigest(),
        fitted_limits_report_only_not_projected=True,
        after_initializer_support=evidence.support_counts().tolist(), possible_point_observations=23*len(evidence.query_ids),
        supported_point_observations=int(evidence.native_visible[1:].sum()), missing_point_observations=int((~evidence.native_visible[1:]).sum())), fitted249


def _initial_pose(rt, np, rotation, translation, p):
    """Native fixed-R uses the supplied F32 pose directly; no fitted alignment."""
    r, t = rotation.detach().cpu().numpy(), translation.detach().cpu().numpy()
    rt.require(r.dtype == t.dtype == np.float32 and r.shape == (24, 3, 3) and t.shape == (24, 3)
        and np.array_equal(r, p['initial_pose'][:, :3, :3]) and np.array_equal(t, p['initial_pose'][:, :3, 3]),
        'Frozen supplied query pose differs from actual direct native fixed-R initial state')


def _fit(rt, code, out, c, layer, optimizer, spec, r, persist, check):
    import numpy as np
    import torch
    import cari_full_refine as full
    import joint_point_authored_qualify as base
    from world_reward import joint_point_objective as op
    path, mr = authenticate_manufacture(rt, c)
    for name, pin in mr['outputs'].items(): rt.require(rt.identity(path/name, 512 << 20) == pin, 'Complete manufacture changed before fitting')
    tracks_path, tr = _previous(rt, c, 'tracks', 'articulated_point_tracks_native_v1')
    rt.require(tr['manufacture_report_identity'] == c['articulated_study']['manufacture']['report'], 'Tracks must bind same manufacture')
    for name, pin in tr['outputs'].items(): rt.require(rt.identity(tracks_path/name, 512 << 20) == pin, 'Complete tracks changed before fitting')
    cfg = _cfg(optimizer); r.update(config=asdict(cfg), calibration={}, reserved_results=[], constructor_attempts=0,
        constructor_returns=0, optimizer_run_attempts=0, optimizer_run_returns=0, actual_native_updates=0)
    dev = []; squared = 0.; count = 0
    for i in range(2):
        check(); r['phase'] = f'dev_{i}_evidence'; persist()
        p, evidence = _evidence(rt, c, path, mr, tracks_path, tr, i); truth = _truth(rt, path, mr, i)
        if np.any(evidence.support_counts() == 0): raise ValueError('Original track missing post-initializer support; no drop')
        error = evidence.tracks_256-_projection(np, p, truth); supported = evidence.native_visible.copy(); supported[0] = False
        squared += float(np.square(error[supported]).sum()); count += int(supported.sum())*2
        source = _source(rt, base, full, torch, path, mr, i); dev.append((p, evidence, source))
    sigma = float(np.sqrt(squared/count)) if count else float('nan')
    if not np.isfinite(sigma) or sigma <= 0: raise ValueError('DEV operational residual scale nonpositive/nonfinite; no floor')
    native_sq = point_sq = 0.; gradient_count = 0
    for i, (p, evidence, source) in enumerate(dev):
        check(); _reset(np, torch); r['phase'] = f'dev_{i}_gradient_calibration'; r['constructor_attempts'] += 1; persist()
        before = full.fingerprint(source); instance = optimizer.MHRParityPostOptimizer(source, p['vertices'], p['faces'], cfg, mhr_layer=layer); r['constructor_returns'] += 1
        indices = torch.arange(24, device='cuda'); rotation, translation, _, _ = instance._object_state(indices, include_surface=False)
        _initial_pose(rt, np, rotation, translation, p)
        loss, _ = instance.loss(indices, 181, include_diagnostics=True)
        ng = torch.autograd.grad(loss, instance.object_translation)[0]
        point = op.point_reprojection_loss(rotation, translation, evidence, op.PointObjectiveConfig(sigma, 1., 'DEV operational calibration unit point'), indices)
        pg = torch.autograd.grad(point, instance.object_translation)[0]
        if not torch.isfinite(ng).all() or not torch.isfinite(pg).all(): raise ValueError('Nonfinite actual DEV native/point gradients')
        native_sq += float(ng.double().square().sum().cpu()); point_sq += float(pg.double().square().sum().cpu()); gradient_count += ng.numel()
        rt.require(full.fingerprint(source) == before, 'DEV source mutated during calibration')
        del instance; gc.collect(); torch.cuda.empty_cache()
    weight = float(np.sqrt(native_sq/gradient_count)/np.sqrt(point_sq/gradient_count)) if point_sq > 0 and native_sq > 0 else float('nan')
    if not np.isfinite(weight) or weight <= 0: raise ValueError('Native/point DEV gradient ratio nonpositive/nonfinite; no search')
    config = op.PointObjectiveConfig(sigma, weight, 'two frozen DEV clips operational pooled native_visible residual and step181 gradient ratio')
    _save(base, out/'calibration.npz', dict(residual_scale_256_px=np.array(sigma, np.float64), weight=np.array(weight, np.float64)))
    calibration = dict(config=asdict(config), fit_scene_indices=[0, 1], residual_coordinates=count,
        native_translation_gradient_rms=float(np.sqrt(native_sq/gradient_count)), unit_point_gradient_rms=float(np.sqrt(point_sq/gradient_count)),
        includes_tracking_and_occlusion_errors=True, calibrated_probability=False, heldout_used=False)
    base.write(out/'model.json', (json.dumps(calibration, sort_keys=True, allow_nan=False)+'\n').encode())
    calibration_pins = {n: rt.identity(out/n) for n in ('model.json', 'calibration.npz')}
    r['calibration'] = dict(calibration, files=calibration_pins); persist(); dev.clear(); gc.collect()
    def seal_check():
        check(); rt.require(all(rt.identity(out/n) == pin for n, pin in calibration_pins.items()), 'Calibration must be sealed before ANY reserved read')
    for j, i in enumerate(range(2, 6)):
        seal_check(); r['phase'] = f'reserved_{i}_evidence'; persist()
        p, evidence = _evidence(rt, c, path, mr, tracks_path, tr, i)
        seal_check(); source = _source(rt, base, full, torch, path, mr, i)
        seal_check(); truth = _truth(rt, path, mr, i)
        before = full.fingerprint(source); extension = op.native_point_optimizer_class(optimizer, evidence, config)
        order = ('A_original', 'B_point') if j % 2 == 0 else ('B_point', 'A_original')
        row = dict(scene_index=i, scene_id=mr['scene_ids'][i], order=list(order), results={}, status='fail'); r['reserved_results'].append(row)
        previous_initial = None
        for arm in order:
            seal_check(); _reset(np, torch); r['phase'] = f'{i}_{arm}_constructor'; r['constructor_attempts'] += 1; persist()
            cls = optimizer.MHRParityPostOptimizer if arm == 'A_original' else extension
            instance = cls(source, p['vertices'], p['faces'], cfg, mhr_layer=layer); r['constructor_returns'] += 1
            indices = torch.arange(24, device='cuda'); rotation, translation, _, _ = instance._object_state(indices, include_surface=False)
            _initial_pose(rt, np, rotation, translation, p)
            initial_hash = full.fingerprint(dict(rotation=rotation, translation=translation, body=instance.body_pose, fixed=instance.params_fixed))
            initial_arrays = dict(rotation=rotation.detach().cpu().numpy().copy(), translation=translation.detach().cpu().numpy().copy(), body=instance.body_pose.detach().cpu().numpy().copy())
            if previous_initial is not None:
                row['initial_constructed_max_abs_deltas'] = {name: float(np.max(np.abs(value.astype(np.float64)-previous_initial[name].astype(np.float64)))) for name, value in initial_arrays.items()}
            previous_initial = initial_arrays
            r['phase'] = f'{i}_{arm}_301_updates'; r['optimizer_run_attempts'] += 1
            if arm == 'B_point': r['positive_weight_executed'] = True
            persist()
            result, calls = base.observed_call(instance.run, dict(loss=optimizer.MHRParityPostOptimizer.loss))
            r['optimizer_run_returns'] += 1
            if calls['loss'] != 301: raise ValueError('Exactly301 actual unchanged native losses required')
            check(); full.validate_result(source, result, 24)
            if arm == 'B_point' and result['postopt']['point_objective']['config'] != asdict(config): raise ValueError('Frozen positive objective differs')
            if full.fingerprint(source) != before: raise ValueError('Native fit changed frozen initial inputs')
            step = instance.optimizer.state[instance.object_translation].get('step')
            if step is None or int(step.detach().cpu()) != 301: raise ValueError('Actual full301 native Adam updates required')
            check(); metrics, fitted249 = _metrics(np, torch, layer, source, result, truth, evidence, optimizer, instance)
            _save(base, out/f'scene_{i:02d}_{arm}_controls.npz', dict(reconstituted_controls=fitted249))
            base.retain(out/f'scene_{i:02d}_{arm}_result.pth', lambda stream: torch.save(result, stream))
            saved = torch.load(out/f'scene_{i:02d}_{arm}_result.pth', map_location='cpu', weights_only=False)
            if full.fingerprint(saved) != full.fingerprint(result): raise ValueError('Complete native saved result reload differs')
            row['results'][arm] = dict(metrics=metrics, initial_state_sha256=initial_hash,
                result_sha256=full.fingerprint(result), native_history=result['postopt']['history'], actual_updates=301,
                actual_native_loss_calls=calls['loss'])
            r['actual_native_updates'] += 301; persist(); del instance, result; gc.collect(); torch.cuda.empty_cache()
        row['same_supplied_initial_bundle_sha256'] = before
        row['constructed_initial_bit_equal_descriptive'] = row['results']['A_original']['initial_state_sha256'] == row['results']['B_point']['initial_state_sha256']
        row['status'] = 'pass'; persist()
    seal_check(); authenticate_manufacture(rt, c); _previous(rt, c, 'tracks', 'articulated_point_tracks_native_v1')
    for directory, previous in ((path, mr), (tracks_path, tr)):
        for name, pin in previous['outputs'].items(): rt.require(rt.identity(directory/name, 512 << 20) == pin, 'Sealed previous payload changed during fitting')
    r.update(descriptive_four_pairs_only=True, statistical_population_gain_verified=False, calibrated_before_reserved_reads=True)


def native(rt, code, out, c):
    """One actual GPU phase; parent is the sole report writer/sealing authority."""
    study = c['articulated_study']; stage = study['stage']; start = time.monotonic()
    if stage not in STAGES: raise ValueError('Explicit manufacture or fit stage required')
    budget = {'manufacture': 600, 'fit': 10800}[stage]
    rt.require(type(study['budgets_seconds'][stage]) is int and study['budgets_seconds'][stage] == budget, 'Prospective fixed budget differs')
    deadline = start+budget
    def check(): rt.require(time.monotonic() < deadline, 'Inclusive authored phase deadline')
    out = rt.canonical(Path(out)); code = rt.canonical(Path(code))
    proof = c['_articulated_proof']; revision = proof['source_binding']['producer_revision']
    rt.require(sys.platform == 'linux' and os.geteuid() == 1000 and os.environ.get('WR_IMAGE_ID') == IMAGE
        and {p.name for p in Path('/sys/class/net').iterdir()} == {'lo'} and out.is_dir() and not any(out.iterdir()), 'Fresh actual offline B47 native phase required')
    rt.require(out == ROOT/'validation/articulated_point_study_v1'/revision/stage/'native'
        and study['schema'] == 'world_reward.articulated_point_study_protocol.v1', 'Exact new authored stage namespace/protocol required')
    import joint_point_authored_qualify as base
    base.check_native_sources(rt, ROOT, proof); check()
    import numpy as np
    import torch
    from lib_mhr.mhr_layer import MHRLayer
    from lib_mhr.hand_surface_contact import load_mhr_hand_surface_spec
    from learning.training import mhr_opt_refineout as optimizer
    from world_reward import joint_point_objective as op
    from world_reward.articulated_point_cohort import SCENES
    rt.require(torch.cuda.is_available() and str(torch.__version__) == '2.5.1+cu124'
        and torch.version.cuda == '12.4' and not torch.are_deterministic_algorithms_enabled(), 'Original ordinary CUDA policy required')
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False; _reset(np, torch)
    op._native_binding(optimizer)
    rt.require(Path(sys.modules[MHRLayer.__module__].__file__).resolve() == ROOT/'vendor/video_to_data'/base.NATIVE/'lib_mhr/mhr_layer.py'
        and Path(optimizer.__file__).resolve() == ROOT/'vendor/video_to_data'/base.NATIVE/'learning/training/mhr_opt_refineout.py', 'Actual unchanged pinned native imports required')
    r = dict(stage=STAGES[stage], status='fail', phase='setup', frames=144, scene_ids=[s[0] for s in SCENES],
        policy_sha256=policy_sha256(c), source_binding=proof['source_binding'], budget_seconds=budget,
        truth_synthetic_known=True, known_geometry_camera_masks_initialization=True, conditional_not_end_to_end=True,
        challenge_inputs_used=False, positive_weight_executed=False, adoption=False, scientific_gain_verified=False,
        reconstruction_accuracy_verified=False, ordinary_cuda_no_bitparity_claim=True, outputs={})
    def persist():
        if '_persist' in c: c['_persist'](r)
        check()
        print('WORLD_REWARD_ARTICULATED_PROGRESS '+json.dumps(dict(stage=r['stage'], phase=r['phase'], elapsed_seconds=time.monotonic()-start,
            optimizer_run_returns=r.get('optimizer_run_returns', 0)), sort_keys=True), flush=True)
    old_alarm = signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Inclusive authored phase budget')))
    signal.setitimer(signal.ITIMER_REAL, max(.001, deadline-time.monotonic()))
    failure = None
    try:
        layer = MHRLayer.from_mhr_assets(mhr_assets_root=Path('/workspace/v2d_sam3d_body/lib'),
            checkpoint_path=ROOT/base.BODY/'model.ckpt', buffer_path=out/'never_compact_buffer.pt',
            mhr_model_path=ROOT/base.BODY/'assets/mhr_model.pt', device='cuda')
        spec = load_mhr_hand_surface_spec(ROOT/'weights/cari4d/refinement/mhr_hand_surface_spec.npz', faces=layer.mesh_faces(device='cuda').cpu().numpy())
        rt.require(spec.mhr_model_sha256 == c['model_prerequisites']['native_model_sha256'], 'Actual hand spec/model differs')
        if stage == 'manufacture': _manufacture(rt, code, out, c, layer, optimizer, spec, r, persist, check)
        else: _fit(rt, code, out, c, layer, optimizer, spec, r, persist, check); r['positive_weight_executed'] = True
        check(); base.check_native_sources(rt, ROOT, proof); check(); r['source_inputs_assets_rehashed_after'] = True
        r.update(status='pass', phase='complete')
    except BaseException as error:
        failure = error; r.update(status='fail', error_type=type(error).__name__, failure=str(error)[:400])
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        try:
            r['outputs'] = {p.name: rt.identity(p, 1_000_000_000) for p in out.iterdir() if p.name != 'report.json'}
            if time.monotonic() >= deadline and failure is None: raise TimeoutError('Phase posthash exceeded original budget')
        except BaseException as error:
            failure = failure or error; r.update(status='fail', post_error_type=type(error).__name__)
        r['elapsed_seconds'] = time.monotonic()-start
        if failure is not None:
            r['status'] = 'fail'; r.setdefault('error_type', type(failure).__name__); r.setdefault('failure', str(failure)[:400])
        # Failure reporting is parent's bounded grace, never more computation.
        try:
            if '_persist' in c: c['_persist'](r)
        finally: signal.signal(signal.SIGALRM, old_alarm)
    return r
