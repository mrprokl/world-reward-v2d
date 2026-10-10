"""Tiny manufactured recovery/runtime contracts, no models or external data."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import sam31_recovery as runtime
from world_reward.occlusion_recovery import RecoveryPolicy, RgbIdentityEvidence, native_candidate, saved_native_candidate

ROOT = Path(__file__).resolve().parents[1]
SHA = 'a' * 64


def policy():
    return RecoveryPolicy(.5, .5, .95, .5, 'external:manufactured-unit-contracts-not-accuracy')


def mask():
    m = np.zeros((12, 16), np.bool_); m[2:10, 3:14] = True
    return m


def candidate(m, index=0, branch='forward', visible=True):
    if branch == 'forward':
        return saved_native_candidate(frame_index=index, object_id=1, rgb_sha256=SHA,
            branch='forward', mask=m, native_presence=visible, tracking_report_sha256='b' * 64)
    return native_candidate(frame_index=index, object_id=1, rgb_sha256=SHA, branch='reverse',
        mask_logits=np.where(m, 1., -1.), presence_logit=2. if visible else -2.)


def evidence(index=0, geometry=None):
    xy = np.array([[x + .5, y + .5] for y in range(3, 9) for x in range(4, 13)], dtype=np.float64)
    return RgbIdentityEvidence(frame_index=index, object_id=1, rgb_sha256=SHA, xy=xy,
        matched=np.ones(len(xy), np.bool_), visible=np.ones(len(xy), np.bool_),
        geometry_mask=mask() if geometry is None else geometry, source='automatic_rgb:manufactured')


def test_frozen_original_cohort_and_offline_reuse():
    c, r = runtime.settings(ROOT)
    assert c['episodes'] == [9, 1, 14, 7]
    assert c['additional_reverse_passes_per_clip'] == 1
    assert c['saved_forward_native_report']['bytes'] == 23225
    assert r['model_revision'] == 'daa63191845a41281374e725f4c9e51c7a824460'
    assert r['weight_sha256'] == '0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6'
    assert c['recovery_gates']['calibration_source'].endswith('not-accuracy')


@pytest.mark.parametrize('change', [
    lambda c: c.update(episodes=[9, 14]),
    lambda c: c.update(manual_labels=True),
    lambda c: c.update(ground_truth_used=True),
    lambda c: c.update(additional_reverse_passes_per_clip=2),
    lambda c: c.update(fusion_policy='prefer_nonempty'),
    lambda c: c['rgb_witness'].update(anchor_candidates=9),
])
def test_no_scope_leakage_or_silent_model_retry(tmp_path, change):
    (tmp_path / 'configs').mkdir(); c = json.loads((ROOT / runtime.CONFIG).read_text()); change(c)
    (tmp_path / runtime.CONFIG).write_text(json.dumps(c))
    (tmp_path / 'configs/sam31_runtime_v1.json').write_text((ROOT / 'configs/sam31_runtime_v1.json').read_text())
    with pytest.raises(ValueError): runtime.settings(tmp_path)


def test_exclusive_mask_bbox_not_padding_or_clamp():
    assert runtime.mask_box(mask()) == [3, 2, 14, 10]
    with pytest.raises(ValueError): runtime.mask_box(np.zeros((3, 4), np.bool_))
    with pytest.raises(ValueError): runtime.mask_box(np.ones((3, 4), np.uint8))


def test_saved_visible_forward_is_not_erased_by_failed_rgb():
    m = mask(); f = candidate(m)
    chosen, source, fused = runtime.select_preserving_forward(f, candidate(m, branch='reverse'), None, policy())
    assert chosen is f.mask and np.array_equal(chosen, m)
    assert source == 'saved_forward_native' and fused.state == 'uncertain'
    assert fused.diagnostics['native_forward_presence_logit'] is None


def test_actual_empty_forward_only_recovered_with_both_rgb_and_geometry():
    empty = np.zeros((12, 16), np.bool_); f = candidate(empty, visible=False)
    chosen, source, fused = runtime.select_preserving_forward(f, candidate(mask(), branch='reverse'), evidence(), policy())
    assert source == 'reverse_rgb_recovery' and np.array_equal(chosen, mask())
    assert fused.state == 'observed'
    chosen, source, _ = runtime.select_preserving_forward(f, candidate(mask(), branch='reverse'),
        evidence(geometry=np.zeros((12, 16), np.bool_)), policy())
    assert not chosen.any() and source == 'unresolved_native_absence'


def test_nonempty_reverse_without_identity_is_not_fake_recovery():
    empty = np.zeros((12, 16), np.bool_)
    chosen, source, _ = runtime.select_preserving_forward(candidate(empty, visible=False),
        candidate(mask(), branch='reverse'), None, policy())
    assert not chosen.any() and source == 'unresolved_native_absence'


def test_one_latest_visible_anchor_not_search_for_better_result():
    calls = []
    def ev(index): calls.append(index); return None
    assert runtime.reverse_anchor([3, 0, 4, 0, 3], lambda i: mask(), ev, [SHA] * 5, 'b' * 64, policy()) is None
    assert calls == [4]


def test_automatic_anchor_requires_seed_identity_and_literal_maskbox():
    anchor = runtime.reverse_anchor([3, 0, 4], lambda i: mask(), lambda i: evidence(i),
        [SHA] * 3, 'b' * 64, policy())
    assert anchor['frame_index'] == 2 and anchor['object_id'] == 1 and anchor['box'] == [3, 2, 14, 10]
    assert runtime.reverse_anchor([0, 0], lambda i: mask(), lambda i: evidence(i), [SHA] * 2, 'b' * 64, policy()) is None


class Tensor:
    def __init__(self, value): self.value = np.asarray(value)
    def detach(self): return self
    def float(self): return self
    def cpu(self): return self
    def numpy(self): return self.value


def test_native_reverse_preserves_raw_logit_and_original_mask_sign():
    logits = np.ones((1, 1, 12, 16)); logits[0, 0, 0, 0] = -3.
    row, summary = runtime.native_singleton((4, [1], None, Tensor(logits), Tensor([[.75]])), 4, 12, 16, SHA)
    assert row.presence_logit == .75 and row.mask.sum() == 191
    assert summary == dict(min=-3., max=1., positive_pixels=191, raw_presence_logit=.75)
    absent, _ = runtime.native_singleton((4, [1], None, Tensor(logits), Tensor([[-.1]])), 4, 12, 16, SHA)
    assert not absent.mask.any() and absent.presence_logit == -.1


@pytest.mark.parametrize('kind', ['id', 'frame', 'rows', 'score'])
def test_native_reverse_rejects_identity_extra_rows_and_nonfinite(kind):
    out = [4, [1], None, Tensor(np.ones((1, 1, 12, 16))), Tensor([.5])]
    if kind == 'id': out[1] = [0]
    if kind == 'frame': out[0] = 3
    if kind == 'rows': out[3] = Tensor(np.ones((2, 1, 12, 16)))
    if kind == 'score': out[4] = Tensor([float('nan')])
    with pytest.raises(ValueError): runtime.native_singleton(out, 4, 12, 16, SHA)


def test_png_copy_unique_inode_hash_and_readonly_preserves_original(tmp_path):
    from PIL import Image
    src = tmp_path / 'source.png'; dst = tmp_path / 'target.png'
    Image.fromarray(mask().astype('uint8') * 255).save(src); src.chmod(0o444)
    pin = runtime.identity(src)
    runtime._copy(src, dst, pin)
    assert src.stat().st_ino != dst.stat().st_ino and not dst.stat().st_mode & 0o222
    assert runtime.identity(src) == pin == runtime.identity(dst)
    assert np.array_equal(runtime.read_mask(dst, pin, 12, 16), mask())
    with pytest.raises(ValueError): runtime.read_mask(dst, pin, 16, 12)


def test_png_not_arbitrary_threshold_repair(tmp_path):
    from PIL import Image
    path = tmp_path / 'gray.png'; Image.fromarray(np.full((12, 16), 1, np.uint8)).save(path); path.chmod(0o444)
    with pytest.raises(ValueError): runtime.read_mask(path, runtime.identity(path), 12, 16)


def test_reverse_source_bound_only_no_new_downloads_or_manual_model():
    raw = (ROOT / 'infra/sam31_recovery.py').read_text()
    wrapper = (ROOT / 'infra/run_sam31_recovery.sh').read_text()
    assert '--network\', \'none' in raw and '--gpus\', \'all' in raw
    assert 'use_prev_mem_frame=False' in raw and 'reverse=True' in raw
    assert 'max_frame_num_to_track=0' in raw and "output[1] == [1]" in raw
    assert 'HF_HUB_OFFLINE=1' in raw and "gemini_calls=0" in raw
    assert 'world-reward-h100.lock' in raw and 'LOCK_NB' in raw
    assert '/run_sam31_recovery/code' in wrapper and 'set +x' in wrapper
    assert 'ground_truth_used=False' in raw


def test_witness_no_seed_features_abstains_without_fake_points(monkeypatch):
    import sys
    class ORB:
        def detectAndCompute(self, image, mask): return [], None
    cv2 = SimpleNamespace(COLOR_RGB2GRAY=1, ORB_create=lambda **kw: ORB(),
        cvtColor=lambda a, code: a[:, :, 0], erode=lambda a, kernel, **kw: a,
        INTER_NEAREST=0, BORDER_CONSTANT=0)
    monkeypatch.setitem(sys.modules, 'cv2', cv2)
    frames = [np.zeros((12, 16, 3), np.uint8) for _ in range(3)]
    config = runtime.settings(ROOT)[0]['rgb_witness']; calls = []
    witness = runtime.RgbWitness(frames, mask(), [SHA] * 3, config, 'b' * 64, {2}, lambda: calls.append(1))
    assert len(calls) == 3 and witness.match_counts == [0, 0, 0]
    assert witness.evidence(2).geometry_mask is None and not witness.evidence(2).matched.any()


def test_mutual_descriptor_identity_rejects_ambiguous_and_nonreciprocal():
    def match(q, t, d): return SimpleNamespace(queryIdx=q, trainIdx=t, distance=d)
    class Matcher:
        def __init__(self): self.calls = 0
        def knnMatch(self, a, b, k):
            self.calls += 1
            if self.calls == 1:
                return [[match(0, 2, 1), match(0, 1, 8)], [match(1, 3, 5), match(1, 0, 6)]]
            return [[match(2, 0, 1), match(2, 1, 8)], [match(3, 0, 1), match(3, 1, 8)]]
    class Orb:
        def detectAndCompute(self, gray, none):
            return [SimpleNamespace(pt=(float(i), 2.)) for i in range(4)], np.ones((4, 32), np.uint8)
    witness = object.__new__(runtime.RgbWitness)
    witness.descriptors = np.ones((2, 32), np.uint8); witness.orb = Orb(); witness.cfg = {'orb_ratio': .75}
    witness.cv2 = SimpleNamespace(NORM_HAMMING=1, BFMatcher=lambda _: Matcher())
    assert witness._reidentify(np.zeros((2, 3), np.uint8)) == [(0, (2., 2.))]
