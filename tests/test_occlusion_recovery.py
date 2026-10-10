"""Tiny manufactured contracts only; not SAM/real-world accuracy validation."""
from dataclasses import replace

import numpy as np
import pytest

from world_reward.occlusion_recovery import (
    AutomaticBoxAnchor, NativeCandidate, RecoveryPolicy, RgbIdentityEvidence,
    fuse_frame, native_candidate, recover_sequence, saved_native_candidate,
)

H, W = 12, 16
SHA = "a" * 64
POLICY = RecoveryPolicy(.5, .5, .95, .5, "external:manufactured-unit-contracts-not-accuracy")


def mask(kind="target"):
    a = np.zeros((H, W), bool)
    if kind == "target": a[3:9, 3:9] = True
    elif kind == "distractor": a[3:9, 10:16] = True
    elif kind == "whole": a[:] = True
    elif kind != "empty": raise ValueError(kind)
    return a


def candidate(branch="forward", kind="target", index=0, logit=None):
    return native_candidate(frame_index=index, object_id=1, rgb_sha256=SHA, branch=branch,
        mask_logits=np.where(mask(kind), 2., -2.).astype(np.float32),
        presence_logit=(-2. if kind == "empty" else 2.) if logit is None else logit)


def evidence(index=0, kind="target", visible=True, matches=True, count=12, occluded=False):
    xy = np.array([(3.+i % 6, 3.+i//6) for i in range(count)], np.float64).reshape(-1, 2)
    return RgbIdentityEvidence(index, 1, SHA, xy, np.full(count, matches, bool),
        np.full(count, visible, bool), mask(kind), "automatic_rgb:owned-seed-feature-contract",
        occluded, "automatic_occlusion:independent-occluder-contract" if occluded else None)


def test_native_presence_sign_and_owned_raw_logit_are_preserved():
    raw = np.ones((H, W), np.float32)
    row = native_candidate(frame_index=0, object_id=1, rgb_sha256=SHA, branch="reverse",
        mask_logits=raw, presence_logit=-.125)
    assert not row.mask.any() and row.presence_logit == -.125 and not row.mask.flags.writeable
    raw[:] = -99; assert not row.mask.any()
    with pytest.raises(ValueError): replace(row, mask=np.ones((H, W), bool))


def test_saved_native_sign_stays_explicit_raw_unavailable_without_pseudo_confidence():
    f = saved_native_candidate(frame_index=0, object_id=1, rgb_sha256=SHA, branch="forward",
        mask=mask("empty"), native_presence=False, tracking_report_sha256="c"*64)
    assert f.presence_logit is None and f.native_presence is False
    row = fuse_frame(f, candidate("reverse"), evidence(), POLICY)
    assert row.source == "reverse_rgb_recovery"
    assert row.diagnostics["native_forward_presence_logit"] is None
    assert row.diagnostics["forward_presence_source"] == "saved_native_presence_sign:"+"c"*64
    with pytest.raises(ValueError): replace(f, native_presence=None)
    with pytest.raises(ValueError): replace(f, mask=mask())
    with pytest.raises(ValueError): replace(f, presence_source="saved_native_presence_sign:unverified")


def test_reverse_recovers_visible_dropout_only_with_rgb_identity_and_geometry():
    row = fuse_frame(candidate(kind="empty"), candidate("reverse"), evidence(), POLICY)
    assert row.state == "observed" and row.source == "reverse_rgb_recovery"
    np.testing.assert_array_equal(row.observed_mask, mask())
    assert row.diagnostics["native_forward_presence_logit"] == -2
    assert row.diagnostics["reverse"]["identity_support_lower_bound"] > .5
    assert not row.observed_mask.flags.writeable


@pytest.mark.parametrize("ev", [None, evidence(matches=False), evidence(visible=False),
                                evidence(count=1), replace(evidence(), geometry_mask=None)])
def test_any_nonempty_reverse_mask_without_independent_support_is_not_recovery(ev):
    row = fuse_frame(candidate(kind="empty"), candidate("reverse"), ev, POLICY)
    assert row.state == "uncertain" and not row.observed_mask.any()
    assert row.diagnostics["full_T_latent_pose_required"]


def test_background_distractor_and_whole_image_shortcuts_are_rejected():
    for bad in ("distractor", "whole"):
        row = fuse_frame(candidate(kind="empty"), candidate("reverse", bad), evidence(), POLICY)
        assert row.state == "uncertain" and not row.observed_mask.any()


def test_dual_absence_is_not_proof_of_occlusion_or_static_pose():
    row = fuse_frame(candidate(kind="empty"), candidate("reverse", "empty"), evidence(visible=False), POLICY)
    assert row.state == "uncertain" and not row.observed_mask.any()
    assert row.diagnostics["observation_interpolated"] is False


def test_independently_supported_true_occlusion_keeps_no_fake_visible_mask():
    row = fuse_frame(candidate(kind="empty"), candidate("reverse", "empty"),
                     evidence(visible=False, occluded=True), POLICY)
    assert row.state == "occluded" and row.source == "independent_automatic_occlusion"
    assert not row.observed_mask.any() and row.diagnostics["full_T_latent_pose_required"]
    with pytest.raises(ValueError): evidence(occluded=True)


def test_forward_keeps_literal_prediction_and_reverse_drift_does_not_override():
    row = fuse_frame(candidate(), candidate("reverse", "distractor"), evidence(), POLICY)
    assert row.state == "observed" and row.source == "forward_rgb_verified"
    np.testing.assert_array_equal(row.observed_mask, mask())


def test_agreement_does_not_union_average_or_modify_native_masks():
    reverse = candidate("reverse"); changed = reverse.mask.copy(); changed[2, 4] = True
    reverse = replace(reverse, mask=changed)
    row = fuse_frame(candidate(), reverse, evidence(), POLICY)
    assert row.source == "bidirectional_rgb_verified"
    np.testing.assert_array_equal(row.observed_mask, mask())


def test_two_qualified_but_disagreeing_branches_abstain_instead_of_score_pick():
    # Both contain the same seed points/independent geometry but disagree in tails.
    left, right = mask(), mask(); left[0:3, 0:12] = True; right[9:12, 0:12] = True
    permissive = RecoveryPolicy(.6, .4, .95, .5, POLICY.calibration_source)
    f = replace(candidate(), mask=left); r = replace(candidate("reverse"), mask=right)
    row = fuse_frame(f, r, evidence(), permissive)
    assert row.source == "qualified_branches_disagree" and row.state == "uncertain"
    assert not row.observed_mask.any()


@pytest.mark.parametrize("fault", ["frame", "id", "hash", "grid", "branch"])
def test_exact_original_reverse_binding_rejects_swaps_and_reindexing(fault):
    r = candidate("reverse")
    kwargs = {"frame":dict(frame_index=1), "id":dict(object_id=2), "hash":dict(rgb_sha256="b"*64),
        "grid":dict(mask=np.zeros((W, H), bool)), "branch":dict(branch="forward")}[fault]
    with pytest.raises(ValueError): fuse_frame(candidate(), replace(r, **kwargs), evidence(), POLICY)


def test_point_grid_never_clamps_hidden_offscreen_features_to_an_edge():
    ev = evidence(); xy = ev.xy.copy(); xy[:] = (-3, -3)
    row = fuse_frame(candidate(kind="empty"), candidate("reverse"), replace(ev, xy=xy), POLICY)
    assert row.state == "uncertain" and row.diagnostics["reverse"]["in_mask_points"] == 0


def test_negative_subpixel_coordinate_is_not_rounded_onto_image_edge():
    edge_mask = np.zeros((H,W), bool); edge_mask[:, 0] = True
    ev = evidence(); xy = ev.xy.copy(); xy[:, 0] = -.1
    r = replace(candidate("reverse"), mask=edge_mask)
    row = fuse_frame(candidate(kind="empty"), r, replace(ev, xy=xy, geometry_mask=edge_mask), POLICY)
    assert row.state == "uncertain" and row.diagnostics["reverse"]["in_mask_points"] == 0


def test_duplicate_correlated_point_votes_cannot_manufacture_identity_support():
    ev = evidence(); xy = ev.xy.copy(); xy[:] = (4, 4)
    row = fuse_frame(candidate(kind="empty"), candidate("reverse"), replace(ev, xy=xy), POLICY)
    assert row.state == "uncertain" and row.diagnostics["reverse"]["identity_points"] == 1


def test_huge_finite_offimage_coordinates_stay_unknown_without_integer_overflow():
    ev = evidence(); xy = ev.xy.copy(); xy[:] = (1e200, 1e200)
    row = fuse_frame(candidate(kind="empty"), candidate("reverse"), replace(ev, xy=xy), POLICY)
    assert row.state == "uncertain" and row.diagnostics["reverse"]["in_mask_points"] == 0


def test_owned_prediction_arrays_cannot_be_mutated_after_binding():
    row = fuse_frame(candidate(), candidate("reverse"), evidence(), POLICY)
    for a in (row.observed_mask, candidate().mask, evidence().xy):
        with pytest.raises(ValueError): a.setflags(write=True)


@pytest.mark.parametrize("field,value", [("identity_confidence", 1.), ("geometry_iou", 0.),
    ("agreement_iou", float("nan")), ("calibration_source", "challenge:episode1")])
def test_policy_requires_explicit_nontrivial_external_calibration(field, value):
    with pytest.raises(ValueError): replace(POLICY, **{field:value})


class Spy:
    def __init__(self, fault=None, anchor=True):
        self.fault = fault; self.anchor = anchor; self.resolutions = []; self.states = []
        self.seeds = []; self.passes = []; self.rows = []; self.released = []
        self.forward_state = {"forward_owned": True}
    def resolve(self, **kwargs):
        self.resolutions.append(kwargs)
        return AutomaticBoxAnchor(kwargs["frame_index"], 1, SHA, (3., 3., 9., 9.), "d"*64) if self.anchor else None
    def init(self):
        state = {} if self.fault != "reuse" else self.forward_state
        self.states.append(state); return state
    def seed(self, state, **kwargs): self.seeds.append(kwargs)
    def propagate(self, state, **kwargs):
        self.passes.append(kwargs); positions = [24, 22, 20]
        if self.fault == "missing": positions.pop()
        if self.fault == "extra": positions += [20]
        if self.fault == "order": positions = [24, 20, 22]
        for i in positions: yield candidate("reverse", "empty" if i == 22 else "target", i)
    def release(self, state): self.released.append(state)
    def forward(self, i): return candidate(kind="target" if i == 20 else "empty", index=i)
    def evidence(self, i): return evidence(i, visible=i != 22, occluded=i == 22)


def run(spy=None, needs=True, **overrides):
    s = Spy() if spy is None else spy
    kwargs = dict(frame_indices=(20, 22, 24), rgb_sha256=(SHA,)*3, image_size=(H,W), object_id=1,
        needs_recovery=needs, forward_at=s.forward, evidence_at=s.evidence,
        resolve_final_anchor=s.resolve,
        prefix_anchors=(AutomaticBoxAnchor(20, 1, SHA, (3.,3.,9.,9.), "c"*64),),
        init_state=s.init, seed_box=s.seed, propagate=s.propagate, emit=s.rows.append,
        policy=POLICY, forward_state=s.forward_state, release_state=s.release)
    kwargs.update(overrides)
    return recover_sequence(**kwargs), s


def test_one_bounded_fresh_pass_all_saved_anchors_exact_full_t_states_and_ids():
    report, s = run()
    assert len(s.resolutions) == len(s.states) == len(s.passes) == len(s.released) == 1
    assert s.passes == [dict(start_frame_idx=2, reverse=True)]
    assert [(v["frame_idx"], v["obj_id"]) for v in s.seeds] == [(0,1), (2,1)]
    assert [(v.frame_index,v.state,v.source) for v in s.rows] == [
        (24,"observed","reverse_rgb_recovery"), (22,"occluded","independent_automatic_occlusion"),
        (20,"observed","bidirectional_rgb_verified")]
    assert report["states"] == dict(observed=2, occluded=1, uncertain=0)
    assert report["original_frame_indices"] == [20,22,24]
    assert report["native_reverse_frames"] == 3 and report["recovered_observation_frames"] == 1
    assert report["additional_reverse_passes"] == report["final_anchor_resolutions"] == 1
    assert not report["mask_interpolation"] and not report["quality_verified"]


def test_unavailable_anchor_is_full_t_latent_fallback_not_a_dropped_clip():
    report, s = run(Spy(anchor=False))
    assert len(s.resolutions) == 1 and not s.states and not s.passes
    assert [r.frame_index for r in s.rows] == [20,22,24]
    assert report["states"] == dict(observed=1, occluded=1, uncertain=1)
    assert report["final_anchor_status"] == "unavailable" and report["full_T_latent_pose_required"]


def test_easy_clip_has_no_additional_vlm_or_video_pass():
    report, s = run(needs=False)
    assert not s.resolutions and not s.states and not s.seeds and not s.passes
    assert report["additional_reverse_passes"] == report["final_anchor_resolutions"] == 0
    assert len(s.rows) == 3


@pytest.mark.parametrize("fault", ["missing", "extra", "order", "reuse"])
def test_native_infrastructure_faults_fail_closed_without_second_pass_or_retry(fault):
    s = Spy(fault)
    with pytest.raises(ValueError): run(s)
    assert len(s.resolutions) == 1 and len(s.states) == 1 and len(s.passes) <= 1
    assert len(s.released) == (0 if fault == "reuse" else 1)


def test_automatic_anchor_call_exception_is_not_masked_as_absence():
    s = Spy()
    def fail(**_): raise TimeoutError("transport")
    with pytest.raises(TimeoutError): run(s, resolve_final_anchor=fail)
    assert not s.states and not s.rows


def test_missing_rgb_or_forged_boolean_logit_evidence_is_not_accepted():
    with pytest.raises(ValueError): candidate(logit=True)
    with pytest.raises(ValueError): run(rgb_sha256=(SHA, SHA))
    with pytest.raises(ValueError): replace(evidence(), rgb_sha256="not-a-hash")
