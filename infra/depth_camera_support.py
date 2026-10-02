"""Audited MoGe focal-support instrumentation without human-model dependencies.

The callable is unchanged from the frozen joint RGB producer. It rejects the
native insufficient-support fallback, not an inaccurate camera, and otherwise
returns the original native solver/inference results without modification.
"""
from rgb_cohort_protocol import HEIGHT, WIDTH

GEOMETRY_SHA = "2f8d5de7d671af16d25fe13c555fd73855d8447bc086bb23aa42e859b303fb05"


def checked_depth_infer(torch, module, network, tensor, fov, diagnostics, context):
    """Reject native <2 nearest64x64 solver fallback, calling original unchanged.

    The process-local instrumentation is restored even on failure. Its source
    is SHA-verified before either pass; this is support, not calibration proof.
    """
    original = module.recover_focal_shift
    before = len(diagnostics)
    def checked(points, mask=None, focal=None, downsample_size=(64, 64)):
        if (downsample_size != (64, 64) or not torch.is_tensor(mask) or mask.dtype != torch.bool
                or tuple(mask.shape) != (1, HEIGHT, WIDTH) or tuple(points.shape) != (1, HEIGHT, WIDTH, 3)):
            raise RuntimeError("Native focal solver boolean mask/shape/sampling ABI changed")
        sampled = torch.nn.functional.interpolate(mask.float().unsqueeze(1), (64, 64), mode="nearest").squeeze(1) > 0
        count = int(sampled.sum().item())
        entry = {**context, "native_nearest64_valid_pixels": count, "focal_prior_supplied": focal is not None,
                 "original_solver_returned": False}; diagnostics.append(entry)
        if count < 2: raise ValueError("Native focal solver lacks two sampled valid pixels; reject its default-camera fallback")
        result = original(points, mask, focal=focal, downsample_size=downsample_size)
        entry["original_solver_returned"] = True
        return result
    module.recover_focal_shift = checked
    try:
        with torch.inference_mode(): result = network.infer(tensor[None], fov_x=fov, apply_mask=False)
        if len(diagnostics) != before+1: raise RuntimeError("Each native MoGe inference must invoke its audited focal solver exactly once")
        return result
    finally: module.recover_focal_shift = original
