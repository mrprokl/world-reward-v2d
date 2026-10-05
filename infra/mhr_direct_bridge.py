"""Fresh named204 -> standalone MHR JIT component control, never SAM inference.

The selected release is authenticated separately from the supplied live model.
An Azure caller must load that exact JIT after authentication and enforce its
source/runtime/deadline before calling ``run_control``. No model is loaded here.
Native zero204 is a new reference-coordinate choice, NOT zeroing the closed
SAM scale-PCA cohort. Dense limits are used literally; soft-limit semantics are
reported, not used to waive a violation. No keypoint70/PCA/contact/quality claim.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

PINS = 'configs/mhr_official_release_selected_qualification_pins.json'
PINS_ID = dict(bytes=1655, sha256='db578ebc8dc680243ba6d3a361a2127ae66ad544e2363d86107bd31c0244bd37')
PROTOCOL = 'configs/mhr_official_release_protocol_v1.json'
PROTOCOL_ID = dict(bytes=1914, sha256='7b5128a0acee793501d685299510933ee4beb2f4605d222beebaae7fa7b84ce1')
OLD_HELPERS = ('infra/acquire_weights.py', 'infra/acquire_weights.sh', PROTOCOL,
               'infra/mediapipe_cpu_runtime_verify.py', 'configs/mhr_official_release_selected_protocol_v1.json')
MODEL_ID = dict(bytes=696110248, sha256='352e271a6c42729c68554ceaea0c955e866970160c31e35506d782dc0f7377bc')
PARAMETERS = ('l_elbow_bend', 'r_elbow_bend', 'l_index2_rz', 'r_index2_rz')
STEPS = (.01, .02)
FRAMES = 1 + len(PARAMETERS) * len(STEPS)
BUDGET_SECONDS = 120
STATE_COMPARISON_ATOL = 1e-7  # representation/FK diagnostic, not an accuracy tolerance


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def authenticate_release(root, code, rt):
    """Hash-only genuine c416 selected proof, complete original source, no imports."""
    root, code = Path(root), Path(code)
    p = rt.pinned(code / PINS, PINS_ID, 1 << 16)
    _require(p['schema'] == 'world_reward.mhr_official_release_selected_qualification_pins.v1'
             and p['producer_revision'] == 'c41620005e8e9167529045d3cae072baa8b79c8c'
             and p['status'] == 'pass' and p['independent_saved_receipts_audit'] is True,
             'Independently selected release PASS required')
    protocol = rt.pinned(code / PROTOCOL, PROTOCOL_ID, 1 << 16)
    _require({k: protocol['model'][k] for k in ('bytes', 'sha256')} == MODEL_ID,
             'Selected native model identity differs')
    out = rt.canonical(root / 'results/mhr-official-release-license-v2')
    _require({x.name for x in out.iterdir()} == set(p['files']) and out.stat().st_mode & 0o777 == 0o555,
             'Exact sealed selected-release inventory required')
    frozen = {code / PINS: PINS_ID, code / PROTOCOL: PROTOCOL_ID}
    for name, pin in p['files'].items():
        _require(Path(name).name == name and rt.identity(out / name, 1 << 20) == pin,
                 'Selected release text/receipt changed')
        frozen[out / name] = pin
    report = rt.pinned(out / 'report.json', p['files']['report.json'], 1 << 20)
    old = root / 'jobs' / p['producer_revision'] / 'acquire_weights/code'
    source = rt.source(root, old, p['producer_revision'], 'acquire_weights', OLD_HELPERS)
    _require(source == report['source_before'] and (old.parent / 'source-sha256').read_bytes()
             == (p['source_archive_sha256'] + '\n').encode(), 'Original selected producer/source differs')
    wanted = dict(stage='mhr_official_release_license_v2', status='pass', phase='complete',
                  producer_revision=p['producer_revision'], existing_model=MODEL_ID, member_model=MODEL_ID,
                  protocol_identity=PROTOCOL_ID, selected_members_decoded=2, selected_expanded_bytes=696121606,
                  selected_payloads_opened=['assets/LICENSE.txt', 'assets/mhr_model.pt'], inactive_payloads_opened=0,
                  asset_license_matches_primary_exactly=True, model_byte_identical=True, model_copy_written=False,
                  model_member_extracted=False, source_rehashed_after=True, existing_model_rehashed_after=True,
                  archive_rehashed_after=True, owned_archive_removed=True, models_loaded=False,
                  packages_installed=False, gpu_used=False, dataset_read=False, sam_provenance_relabelled=False,
                  competition_eligibility_verified=False, training_overlap_verified=False, adoption=False)
    _require(all(type(report.get(k)) is type(v) and report[k] == v for k, v in wanted.items()),
             'Actual selected release scope differs')
    _require(type(report.get('elapsed_seconds')) in (int, float) and 0 < report['elapsed_seconds'] <= 300,
             'Original selected-release deadline differs')
    model = rt.canonical(root / protocol['model']['existing_path'])
    _require(model.lstat().st_uid in protocol['model']['accepted_uids']
             and rt.identity(model, MODEL_ID['bytes'], readonly=False) == MODEL_ID,
             'Genuine standalone model bytes differ')
    frozen[model] = MODEL_ID
    return dict(model_path=model, frozen=frozen, source=source, original_code=old,
                producer_revision=p['producer_revision'], pins_identity=PINS_ID,
                release_report_identity=p['files']['report.json'])


def recheck_release(proof, root, rt):
    for path, pin in proof['frozen'].items():
        _require(rt.identity(path, max(pin['bytes'], 1 << 20), readonly=path != proof['model_path']) == pin,
                 'Original release/model bytes changed during control')
    _require(rt.source(Path(root), proof['original_code'], proof['producer_revision'], 'acquire_weights',
                       OLD_HELPERS) == proof['source'], 'Original full source changed during control')


def _numpy(value):
    import numpy as np
    return np.asarray(value.detach().cpu().numpy() if hasattr(value, 'detach') else value)


def _array_id(value):
    a = _numpy(value)
    return dict(shape=list(a.shape), dtype=a.dtype.str, sha256=hashlib.sha256(a.tobytes(order='C')).hexdigest())


def _owned(value):
    import numpy as np
    a = np.ascontiguousarray(value)
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


def inspect_metadata(model):
    """Validate the REAL MHRDemo schema/metadata, never generic-source padding."""
    import numpy as np
    expected = dict(forward=([('identity_coeffs', 'Tensor'), ('model_parameters', 'Tensor'),
                            ('face_expr_coeffs', 'Tensor'), ('apply_correctives', 'bool')], 'Tuple[Tensor, Tensor]'),
                    get_parameter_names=([], 'List[str]'), get_joint_names=([], 'List[str]'),
                    get_parameter_transform=([], 'Tensor'), get_parameter_limits=([], 'Tensor'),
                    get_num_identity_blendshapes=([], 'int'), get_num_face_expression_blendshapes=([], 'int'))
    methods = {}
    for name, wanted in expected.items():
        method = model._c._get_method(name); schema = method.schema
        _require(schema.arguments and schema.arguments[0].name == 'self' and 'MHRDemo' in str(schema.arguments[0].type),
                 'Real selected MHRDemo method schema required')
        actual = ([(a.name, str(a.type)) for a in schema.arguments[1:]], str(schema.returns[0].type))
        _require(actual == wanted, 'Direct native method ABI differs: ' + name)
        _require(len(method.code.encode()) <= 100000 and len(str(method.graph).encode()) <= 100000,
                 'Bounded actual scripted method source required')
        methods[name] = dict(code_sha256=hashlib.sha256(method.code.encode()).hexdigest(),
                             graph_sha256=hashlib.sha256(str(method.graph).encode()).hexdigest())
    names, joints = list(model.get_parameter_names()), list(model.get_joint_names())
    _require(len(names) == 249 and len(set(names)) == 249 and len(joints) == 127 and len(set(joints)) == 127
             and all(type(n) is str and n for n in names + joints), 'Unique native names/249+127 ABI required')
    _require((model.get_num_identity_blendshapes(), model.get_num_face_expression_blendshapes()) == (45, 72),
             'Actual identity/expression ABI differs')
    character = model.character_torch
    arrays = dict(bounds=_numpy(model.get_parameter_limits()), transform=_numpy(model.get_parameter_transform()),
                  parents=_numpy(character.skeleton.joint_parents), faces=_numpy(character.mesh.faces))
    lbs = model.get_lbsw(); _require(type(lbs) is tuple and len(lbs) == 2, 'Native LBS pair required')
    arrays.update(lbs_indices=_numpy(lbs[0]), lbs_weights=_numpy(lbs[1]))
    b, m, p, f, idx, w = (arrays[k] for k in ('bounds', 'transform', 'parents', 'faces', 'lbs_indices', 'lbs_weights'))
    _require(b.shape == (249, 2) and b.dtype == np.float32 and not np.isnan(b).any() and np.all(b[:, 0] <= b[:, 1]),
             'Literal actual min/max bounds required')
    _require(m.shape == (889, 249) and m.dtype == np.float32 and np.isfinite(m).all(), 'Native linear-map ABI differs')
    _require(p.shape == (127,) and p.dtype.kind in 'iu' and np.all((p >= -1) & (p < 127))
             and np.count_nonzero(p == -1) == 1, 'Native one-root parent tree required')
    chains = []
    for j in range(127):
        chain = set(); current = j
        while current != -1:
            _require(current not in chain, 'Native parent cycle'); chain.add(current); current = int(p[current])
        chains.append(chain)
    _require(idx.shape == w.shape and idx.ndim == 2 and idx.shape[0] == 18439 and idx.dtype.kind in 'iu'
             and w.dtype == np.float32 and np.isfinite(w).all() and np.all((idx >= 0) & (idx < 127))
             and np.all(w >= 0) and np.allclose(w.sum(1), 1., rtol=0, atol=1e-5), 'Actual normalized LBS required')
    _require(f.shape == (36874, 3) and f.dtype.kind in 'iu' and np.all((f >= 0) & (f < len(idx))),
             'Full original LOD1 faces required')
    _require(names == list(character.parameter_transform.parameter_names) and joints == list(character.skeleton.joint_names)
             and np.array_equal(m, _numpy(character.parameter_transform.parameter_transform)),
             'Getter/submodule metadata mismatch')
    controls = []
    for name in PARAMETERS:
        _require(name in names[:204], 'Fixed NEW named control absent: ' + name)
        col = names.index(name); side = name[0]; wrist = joints.index(side + '_wrist')
        rows = np.flatnonzero(m[:, col] != 0); local = sorted(set((rows // 7).tolist()))
        _require(len(rows) and np.isin(rows % 7, [3, 4, 5]).all(), 'Named control must affect rotations only')
        closure = [j for j, chain in enumerate(chains) if any(a in chain for a in local)]
        if 'elbow' in name:
            _require(wrist in closure and all(joints[j].startswith(side + '_') for j in closure),
                     'Named elbow support must include its wrist and no opposite/root joints')
        else:
            stem = name.rsplit('_', 1)[0]
            _require(stem in joints and all(wrist in chains[j] for j in local)
                     and all(joints[j].startswith(side + '_index') for j in closure), 'Named index support differs')
        mass = np.where(np.isin(idx, closure), w, 0.).sum(1)
        _require(np.any(mass > 0), 'Named control lacks actual LBS support')
        controls.append(dict(name=name, column=col, local_joints=local, global_closure=closure,
                             lbs_support_vertices=int(np.count_nonzero(mass > 0))))
    return dict(parameter_names=names, joint_names=joints, arrays={k: _owned(v) for k, v in arrays.items()},
                methods=methods, controls=controls,
                fingerprint=hashlib.sha256(json.dumps(dict(names=names, joints=joints, methods=methods,
                    arrays={k: _array_id(v) for k, v in arrays.items()}), sort_keys=True).encode()).hexdigest())


def named_batch(metadata):
    """NEW nine-pose batch, all literal bounds checked BEFORE any decode."""
    import numpy as np
    q = np.zeros((FRAMES, 204), np.float32); rows = [dict(frame_index=0, name='native_zero204_reference', value=0.)]
    for record in metadata['controls']:
        for step in STEPS:
            row = len(rows); q[row, record['column']] = step
            rows.append(dict(frame_index=row, name=record['name'], value=float(q[row, record['column']])))
    _require(len(rows) == FRAMES, 'Complete fixed control bank required')
    identity = np.zeros((FRAMES, 45), np.float32); expression = np.zeros((FRAMES, 72), np.float32)
    full = np.column_stack((q, identity)); b = metadata['arrays']['bounds']
    _require(np.all(full >= b[:, 0]) and np.all(full <= b[:, 1]), 'Fresh fixed native batch violates literal bounds; STOP')
    return dict(parameters=_owned(q), identity=_owned(identity), expression=_owned(expression), rows=rows)


def run_control(model, torch, *, check):
    """Exactly ONE full nine-pose call. Caller enforces authentic load +120s.

    Results are semantic component evidence, not repeatability/contact/accuracy.
    No masked/guessed keypoints, scale fitting, renders, Bootstrap or optimizer.
    """
    import numpy as np
    check(); meta = inspect_metadata(model); batch = named_batch(meta); check()
    inputs = [torch.as_tensor(batch[k].copy(), device='cuda', dtype=torch.float32)
              for k in ('identity', 'parameters', 'expression')]
    input_ids = [_array_id(value) for value in inputs]
    with torch.no_grad():
        result = model(*inputs, True)
    torch.cuda.synchronize(); check()
    _require([_array_id(value) for value in inputs] == input_ids, 'Native input tensors were modified')
    _require(type(result) is tuple and len(result) == 2, 'Actual native result pair required')
    v, s = map(_numpy, result)
    _require(v.shape == (FRAMES, 18439, 3) and s.shape == (FRAMES, 127, 8)
             and v.dtype == s.dtype == np.float32 and np.isfinite(v).all() and np.isfinite(s).all()
             and np.all(s[..., 7] > 0) and np.allclose(np.linalg.norm(s[..., 3:7], axis=-1), 1., rtol=0, atol=1e-5),
             'Finite original native V/skeleton/quaternion/scale ABI required')
    evidence = []
    for index, control in enumerate(meta['controls']):
        a, b = 1 + 2 * index, 2 + 2 * index
        allowed = control['global_closure']; outside = np.setdiff1d(np.arange(127), allowed)
        q = s[[a, b], :, 3:7].astype(np.float64)
        q *= np.where((q * s[0, :, 3:7]).sum(-1, keepdims=True) < 0, -1., 1.)
        position = np.abs(s[[a, b], :, :3].astype(np.float64) - s[0, :, :3])
        rotation = np.abs(q - s[0, :, 3:7]); scale = np.abs(s[[a, b], :, 7].astype(np.float64) - s[0, :, 7])
        vertex = np.linalg.norm(v[[a, b]].astype(np.float64) - v[0], axis=-1) / 100.
        idx, w = meta['arrays']['lbs_indices'], meta['arrays']['lbs_weights']
        supported = np.where(np.isin(idx, allowed), w, 0.).sum(1) > 0
        maximum = lambda x: float(np.max(x)) if x.size else 0.
        _require(maximum(position[:, outside]) <= STATE_COMPARISON_ATOL
                 and maximum(rotation[:, outside]) <= STATE_COMPARISON_ATOL
                 and maximum(scale) <= STATE_COMPARISON_ATOL, 'Native named support/constant scales failed')
        _require(all(maximum(rotation[k, allowed]) > 0 and maximum(vertex[k, supported]) > 0 for k in (0, 1)),
                 'Both fixed steps must have nonzero skeletal and material effects')
        evidence.append(dict(**control, quaternion_motion_max=maximum(rotation[:, allowed]),
            vertex_motion_max_m=maximum(vertex), lbs_supported_vertex_motion_max_m=maximum(vertex[:, supported]),
            excluded_position_max_native_cm=maximum(position[:, outside]),
            excluded_quaternion_max=maximum(rotation[:, outside]), scale_difference_max=maximum(scale)))
    after = inspect_metadata(model); check()
    _require(after['fingerprint'] == meta['fingerprint'], 'Native metadata changed during direct control')
    return dict(stage='mhr_direct_named_component_v1', status='pass', native_calls=1, decoded_frames=FRAMES,
                frame_index=list(range(FRAMES)), named_rows=batch['rows'],
                method_source=meta['methods'], metadata_fingerprint=meta['fingerprint'], controls=evidence,
                inputs={k: _array_id(batch[k]) for k in ('parameters', 'identity', 'expression')},
                outputs=dict(vertices=_array_id(v), skeleton=_array_id(s)), native_scale_policy='zero204_reference_no_SAM_PCA',
                bounds_used_literally=True, bounds_relaxed=False, parameter_limit_hard_enforcement_claimed=False,
                supplied_live_model_load_authenticated_by_this_function=False, metadata_rehashed_after=True,
                units='native_cm_skeleton_quaternion_xyzw_positive_scale', SAM_checkpoint_used=False,
                keypoint70_bridge_qualified=False, render_calls=0, tracker_calls=0, optimizer_calls=0,
                contact_verified=False, RGB_only_reconstruction_verified=False, quality_verified=False, adoption=False)
