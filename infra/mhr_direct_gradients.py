"""Fresh standalone-MHR directional autograd control, not an optimizer backend.

The caller authenticates the selected JIT/runtime and enforces its inclusive120s
deadline. This module loads no model and imports no scientific package at import
time. Two FP32 forwards and one VJP test only four fixed named articulations;
FP64 scalar reductions are diagnostics, not a replacement decoder. No root,
keypoint70, quaternion-gradient, contact, reconstruction or adoption claim.
"""
from __future__ import annotations

CENTRE = 2 ** -5
FD_STEPS = (2 ** -8, 2 ** -10)
ATOL_M_PER_UNIT = 1e-5
RTOL = 1e-2
PARAMETERS = ('l_elbow_bend', 'r_elbow_bend', 'l_index2_rz', 'r_index2_rz')


def _require(value, message):
    if not value:
        raise ValueError(message)


def gradient_batch(metadata):
    """Freeze centre1/perturbations16; every full249 bound precedes any call."""
    import numpy as np
    import mhr_direct_bridge as bridge
    controls = metadata['controls']; bounds = metadata['arrays']['bounds']
    _require(tuple(c['name'] for c in controls) == PARAMETERS
             and len({c['column'] for c in controls}) == 4
             and all(type(c['column']) is int and 0 <= c['column'] < 136 for c in controls),
             'Four actual fixed named articulation columns required')
    _require(bounds.dtype == np.float32 and bounds.shape == (249, 2)
             and not np.isnan(bounds).any() and np.all(bounds[:, 0] <= bounds[:, 1]),
             'Original literal249 bounds required')
    centre = np.zeros((1, 204), np.float32)
    for c in controls:
        centre[0, c['column']] = CENTRE
    perturbations = np.repeat(centre, 16, axis=0); rows = []
    for direction, c in enumerate(controls):
        for h in FD_STEPS:
            for sign in (-1, 1):
                row = len(rows); perturbations[row, c['column']] += sign * h
                rows.append(dict(row=row, direction=direction, name=c['name'], step=h, sign=sign))
    full = np.column_stack((np.vstack((centre, perturbations)), np.zeros((17, 45), np.float32)))
    _require(np.all(full >= bounds[:, 0]) and np.all(full <= bounds[:, 1]),
             'All fresh centre/FD bounds must pass before either forward; no clipping')
    return dict(centre=bridge._owned(centre), perturbations=bridge._owned(perturbations),
                identity=bridge._owned(np.zeros((17, 45), np.float32)),
                expression=bridge._owned(np.zeros((17, 72), np.float32)), rows=tuple(rows))


def directional_comparison(centre_value, perturbation_values, gradient):
    """Both fixed centred differences must agree; no step selection or retry."""
    import numpy as np
    values = np.asarray(perturbation_values); g = np.asarray(gradient)
    _require(values.shape == (16,) and g.shape == (4,) and np.isfinite(values).all()
             and np.isfinite(g).all() and np.isfinite(centre_value) and np.all(g != 0),
             'Finite scalar controls and four nonzero VJP entries required')
    records = []
    for j, name in enumerate(PARAMETERS):
        for k, h in enumerate(FD_STEPS):
            fd = float((values[4*j+2*k+1] - values[4*j+2*k]) / (2*h))
            exact = float(g[j]); error = abs(fd - exact); magnitude = max(abs(fd), abs(exact))
            limit = ATOL_M_PER_UNIT + RTOL * magnitude
            _require(all(np.isfinite(x) for x in (fd, error, limit)) and error <= limit,
                     'Fixed directional derivative comparison failed; no tolerance/step rescue')
            records.append(dict(name=name, step=h, vjp=exact, centred_difference=fd,
                absolute_error_m_per_unit=error,
                relative_error=error/magnitude if magnitude else None,
                allowed_error_m_per_unit=limit))
    return records


def _scalar_values(vertices, skeleton, torch):
    """Full-V plus all joint positions; fixed count-normalized cotangent axes."""
    import numpy as np
    def weights(count):
        k = np.arange(count, dtype=np.int64)
        return np.column_stack(tuple((1 - 2*((k//divisor) % 2)) * (1 + k % period)
                                     for divisor, period in ((1, 3), (2, 5), (3, 7)))).astype(np.float64) / count
    wv = torch.as_tensor(weights(18439), dtype=torch.float64, device=vertices.device)
    ws = torch.as_tensor(weights(127), dtype=torch.float64, device=skeleton.device)
    # Native outputs remain FP32; only the scalar measurement is reduced in F64.
    return ((vertices.to(dtype=torch.float64)*wv).sum(dim=(-2, -1))
            + (skeleton[:, :, :3].to(dtype=torch.float64)*ws).sum(dim=(-2, -1))) / 100


def run_control(model, torch, *, check):
    """Exactly centre1+perturbations16 forwards, one VJP, no accumulated grads."""
    import numpy as np
    import mhr_direct_bridge as bridge
    check(); metadata = bridge.inspect_metadata(model); batch = gradient_batch(metadata)
    _require(not torch.is_inference_mode_enabled(), 'Autograd control cannot run in inference_mode')
    parameters = tuple(model.named_parameters())
    versions = tuple((name, id(value), value._version) for name, value in parameters)
    _require(all(value.grad is None for _, value in parameters), 'Weight .grad must initially be None')
    def tensor(a):
        return torch.as_tensor(a.copy(), device='cuda', dtype=torch.float32)
    centre = tensor(batch['centre']).requires_grad_(True)
    ci, ce = tensor(batch['identity'][:1]), tensor(batch['expression'][:1])
    perturb = tensor(batch['perturbations'])
    pi, pe = tensor(batch['identity'][1:]), tensor(batch['expression'][1:])
    inputs = (centre, ci, ce, perturb, pi, pe)
    ids = tuple(bridge._array_id(x) for x in inputs); check()
    def validate(pair, count):
        _require(type(pair) is tuple and len(pair) == 2, 'Original native result pair required')
        v, s = pair
        _require(all(torch.is_tensor(x) and x.dtype == torch.float32 and x.device == centre.device
                     and bool(torch.isfinite(x).all()) for x in (v, s))
                 and tuple(v.shape) == (count, 18439, 3) and tuple(s.shape) == (count, 127, 8)
                 and bool((s[:, :, 7] > 0).all()), 'Full finite original FP32 native output required')
        return v, s
    with torch.enable_grad():
        _require(torch.is_grad_enabled() and centre.is_leaf, 'Fresh original leaf autograd required')
        v, s = validate(model(ci, centre, ce, True), 1)
        _require(v.requires_grad and s.requires_grad, 'Both native outputs must retain autograd')
        scalar = _scalar_values(v, s, torch)
        _require(bool(torch.isfinite(scalar).all()), 'Finite centre scalar required')
        (gradient,) = torch.autograd.grad(scalar[0], centre, create_graph=False, retain_graph=False)
    _require(tuple(gradient.shape) == (1, 204) and gradient.dtype == torch.float32
             and bool(torch.isfinite(gradient).all()) and centre.grad is None,
             'Finite full native VJP without leaf-gradient accumulation required')
    g = bridge._numpy(gradient)[0, [c['column'] for c in metadata['controls']]]
    _require(np.all(g != 0), 'Each fixed named VJP entry must be nonzero before FD execution')
    torch.cuda.synchronize(); check()
    with torch.no_grad():
        pv, ps = validate(model(pi, perturb, pe, True), 16)
        values = bridge._numpy(_scalar_values(pv, ps, torch))
    torch.cuda.synchronize(); check()
    va, sa, pva, psa = map(bridge._numpy, (v, s, pv, ps))
    _require(np.array_equal(psa[:, :, 7], np.repeat(sa[:, :, 7], 16, axis=0)),
             'Constant native scales changed during named perturbations')
    comparisons = directional_comparison(float(bridge._numpy(scalar)[0]), values, g)
    motion = []
    idx, weight = (metadata['arrays'][key] for key in ('lbs_indices', 'lbs_weights'))
    for j, c in enumerate(metadata['controls']):
        supported = np.where(np.isin(idx, c['global_closure']), weight, 0).sum(1) > 0
        maxima = np.max(np.abs(pva[4*j:4*j+4, supported].astype(np.float64)
                               - va[0, supported]), axis=(1, 2)) / 100
        _require(np.isfinite(maxima).all() and np.all(maxima > 0),
                 'Every fixed perturbation must move actual LBS-supported material')
        motion.append(dict(name=c['name'], support_vertices=int(supported.sum()),
                           perturbation_motion_max_m=maxima.tolist()))
    _require(tuple(bridge._array_id(x) for x in inputs) == ids, 'Native input arrays changed')
    _require(tuple((name, id(value), value._version) for name, value in model.named_parameters()) == versions
             and all(value.grad is None for _, value in model.named_parameters()),
             'Native weights or their unaccumulated gradients changed')
    after = bridge.inspect_metadata(model); check()
    _require(after['fingerprint'] == metadata['fingerprint'], 'Native metadata changed')
    return dict(stage='mhr_direct_directional_autograd_component_v1', status='pass',
        native_forward_calls=2, decoded_frames=17, vjp_calls=1, centre=CENTRE,
        parameters=list(PARAMETERS), columns=[c['column'] for c in metadata['controls']],
        finite_difference_steps=list(FD_STEPS), directional_atol_m_per_unit=ATOL_M_PER_UNIT,
        directional_rtol=RTOL, comparisons=comparisons, material_support_motion=motion,
        cotangent_policy='full_vertices_and_all_joint_positions_axis_patterns_1_3_2_5_3_7_count_normalized_cm_to_m',
        forward_dtype='float32', scalar_reduction_dtype='float64', apply_correctives=True,
        inputs=[dict(name=name, **row) for name, row in zip(('centre', 'centre_identity', 'centre_expression',
                    'perturbations', 'perturbation_identity', 'perturbation_expression'), ids)],
        metadata_fingerprint=metadata['fingerprint'], native_inputs_unchanged=True,
        constant_identity_expression_scale=True, weights_version_unchanged=True, weight_gradients_none=True,
        metadata_rehashed_after=True, root_gradients_qualified=False, full_jacobian_qualified=False,
        keypoint70_bridge_qualified=False, quaternion_gradients_qualified=False,
        render_calls=0, tracker_calls=0, optimizer_calls=0, SAM_checkpoint_used=False,
        contact_verified=False, quality_verified=False, backend_adoption=False)
