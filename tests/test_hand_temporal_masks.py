"""Data-free full-timeline native-shape/callback spies, not model quality."""
import numpy as np
import pytest

from world_reward.hand_observations import HandInstances, LandmarkEvidence
from world_reward.hand_temporal_masks import stream_temporal_hand_masks


H, W = 7, 11


def hands(count=1, unusable=()):
    xy = np.empty((count, 21, 2), np.float64)
    for i in range(count):
        xy[i, :, 0] = np.linspace(1+i, 5+i, 21)
        xy[i, :, 1] = np.linspace(1, 5, 21)
    supported = np.ones((count, 21), bool)
    for i in unusable:
        supported[i, 5] = False
    return HandInstances(LandmarkEvidence(xy, supported, "original_image", "pixel"))


class Image:
    def __init__(self): self.images = []; self.calls = []
    def set_image(self, image): self.images.append(image)
    def predict(self, **kwargs):
        self.calls.append(kwargs); n = len(kwargs['box'])
        masks = np.zeros((n, 1, H, W), bool)  # Empty anchor IMAGE does not remove usable box.
        scores = np.arange(n, dtype=np.float32)+1.5
        return (masks[0], scores, None) if n == 1 else (masks, scores[:, None], None)


class Video:
    def __init__(self, total, fault=None):
        self.total, self.fault = total, fault
        self.states = []; self.seeds = []; self.calls = []; self.released = []; self.logits = []
    def init(self):
        state = {'ordinal': len(self.states), 'ids': []}; self.states.append(state); return state
    def seed(self, state, **kwargs):
        self.seeds.append((state['ordinal'], kwargs)); state['ids'].append(kwargs['obj_id'])
    def propagate(self, state, **kwargs):
        self.calls.append((state['ordinal'], kwargs)); a = kwargs['start_frame_idx']; reverse = kwargs['reverse']
        positions = list(range(a, -1, -1)) if reverse and a > 0 else ([] if reverse else list(range(a, self.total)))
        if self.fault == 'missing': positions = positions[:-1]
        if self.fault == 'extra': positions += positions[-1:]
        if self.fault == 'position': positions = [self.total] + positions[1:]
        for position in positions:
            ids = state['ids'][::-1] if position % 2 else state['ids'].copy()
            if self.fault == 'foreign': ids[-1] = 999
            if self.fault == 'duplicate' and len(ids) > 1: ids[-1] = ids[0]
            if self.fault == 'bool_id': ids[0] = True
            logits = np.full((len(ids), 1, H, W), -1., np.float32)
            for index, proposal_id in enumerate(ids):
                logits[index, 0, position % H, proposal_id % W] = 2.
            if reverse: logits[..., -1, -1] = 3.  # Duplicate anchor must not own B anchor.
            if self.fault == 'nan': logits[0, 0, 0, 0] = np.nan
            if self.fault == 'shape': logits = logits[:, 0]
            if self.fault == 'integer_logits': logits = logits.astype(np.int32)
            self.logits.append(logits)
            yield position, ids, logits
    def release(self, state):
        self.released.append(state['ordinal']); state.clear()


def run(banks=None, fault=None, frame_ids=None, frames=None, init=None, emit=None):
    banks = [hands(0), hands(3, (1,)), hands(0), hands(2)] if banks is None else banks
    ids = np.array([10+2*i for i in range(len(banks))], np.int64) if frame_ids is None else frame_ids
    rgb = np.zeros((H, W, 3), np.uint8)
    source = [(int(i), rgb, h) for i, h in zip(ids, banks)] if frames is None else frames
    image, video, rows = Image(), Video(len(banks), fault), []
    summary = stream_temporal_hand_masks(source, frame_ids=ids, image_size=(H, W), image_predictor=image,
        init_state=video.init if init is None else init, seed_box=video.seed, propagate=video.propagate,
        release_state=video.release, emit=rows.append if emit is None else emit)
    return summary, rows, image, video, rgb, ids


def test_full_t_first_usable_anchor_two_independent_states_all_slots_and_ownership():
    summary, rows, image, video, _, ids = run()
    assert summary.anchor_position == 1 and summary.anchor_frame_id == 12
    assert summary.seeded_proposal_ids == (0, 2) and summary.anchor_box_usable.tolist() == [True, False, True]
    assert (summary.native_forward_frames, summary.native_reverse_frames) == (3, 2)
    assert video.calls == [(0, dict(start_frame_idx=1, reverse=False)), (1, dict(start_frame_idx=1, reverse=True))]
    assert video.released == [0, 1] and video.states == [{}, {}]
    assert [(r.branch, r.position) for r in rows] == [('A', i) for i in range(4)] + [('B', 1), ('B', 2), ('B', 3), ('B', 0)]
    for r in rows: assert r.original_frame_id == ids[r.position]
    b = {r.position:r for r in rows if r.branch == 'B'}
    assert not b[1].masks[:, -1, -1].any() and b[0].masks[[0, 2], -1, -1].all()
    for r in b.values():
        assert r.proposal_ids == (0, 1, 2) and r.supported.tolist() == [True, False, True]
        assert not r.masks[1].any() and r.raw_scores is None
        assert r.masks[0, r.position % H, 0] and r.masks[2, r.position % H, 2]
    assert b[1].evidence == 'anchor_prompt' and b[0].evidence == 'video_inferred'
    assert len(image.calls) == 2 and all(set(c) == {'box', 'multimask_output'} and c['multimask_output'] is False for c in image.calls)
    assert all(r.raw_scores is not None and (r.raw_scores[r.supported] > 1).all() for r in rows if r.branch == 'A')
    for ordinal, seed in video.seeds:
        assert set(seed) == {'frame_idx', 'obj_id', 'box', 'normalize_coords'} and seed['normalize_coords'] is True
        assert seed['frame_idx'] == 1 and seed['box'].dtype == np.float32 and not seed['box'].flags.writeable
        np.testing.assert_array_equal(seed['box'], summary.anchor_native_boxes[seed['obj_id']])


@pytest.mark.parametrize('banks', [[hands(0)]*3, [hands(1, (0,))]*3])
def test_no_anchor_abstains_full_t_without_video_or_image_calls(banks):
    s, rows, image, video, _, _ = run(banks)
    assert s.anchor_position is s.anchor_frame_id is None and s.seeded_proposal_ids == ()
    assert video.states == video.calls == video.seeds == image.calls == []
    b = [r for r in rows if r.branch == 'B']
    assert len(b) == 3 and all(r.masks.shape == (0, H, W) and not r.supported.any() and r.evidence == 'no_anchor_abstention' for r in b)


@pytest.mark.parametrize('anchor', [0, 3])
def test_boundary_anchors_native_reverse_at_zero_is_empty(anchor):
    banks = [hands(0)]*4; banks[anchor] = hands()
    s, rows, _, video, _, _ = run(banks)
    assert s.anchor_position == anchor and s.native_reverse_frames == (anchor+1 if anchor else 0)
    assert sorted(r.position for r in rows if r.branch == 'B') == list(range(4))
    assert len(video.states) == 2 and video.released == [0, 1]


def test_lateborn_native_hands_are_not_added_and_crossing_reordered_ids_are_not_relabelled():
    s, rows, _, video, _, _ = run([hands(), hands(4), hands(0)])
    assert s.seeded_proposal_ids == (0,) and len(video.seeds) == 2
    assert [len(r.proposal_ids) for r in rows if r.branch == 'A'] == [1, 4, 0]
    assert [len(r.proposal_ids) for r in rows if r.branch == 'B'] == [1, 1, 1]


def test_owned_readonly_inputs_outputs_and_native_logits_have_no_alias():
    s, rows, image, video, rgb, ids = run()
    ids[:] = 99; rgb[:] = 255
    assert s.frame_ids.tolist() == [10, 12, 14, 16] and not image.images[0].any()
    for output in video.logits: output[:] = 42
    for r in rows:
        for name in ('raw_boxes', 'native_boxes', 'box_usable', 'masks', 'supported'):
            value = getattr(r, name); assert not value.flags.writeable
            assert all(not np.shares_memory(value, v) for v in video.logits)
    assert not rows[-1].masks[0, 5, 5] and not s.frame_ids.flags.writeable
    with pytest.raises(ValueError): image.images[0][:] = 1


@pytest.mark.parametrize('fault', ['missing', 'extra', 'position', 'foreign', 'duplicate', 'bool_id', 'nan', 'shape', 'integer_logits'])
def test_native_contract_errors_fail_without_missing_frame_repair_and_release_state(fault):
    video = Video(3, fault); image = Image(); rows = []
    with pytest.raises(ValueError):
        stream_temporal_hand_masks([(i, np.zeros((H, W, 3), np.uint8), hands(2)) for i in range(3)],
            frame_ids=np.arange(3, dtype=np.int64), image_size=(H, W), image_predictor=image,
            init_state=video.init, seed_box=video.seed, propagate=video.propagate, emit=rows.append, release_state=video.release)
    assert video.released == [0] and len(video.states) == 1


def test_same_state_reuse_is_rejected_after_first_branch():
    video = Video(2); shared = {'ordinal':0, 'ids':[]}; emitted = []
    with pytest.raises(ValueError, match='distinct fresh'):
        stream_temporal_hand_masks([(i, np.zeros((H, W, 3), np.uint8), hands()) for i in range(2)],
            frame_ids=np.arange(2, dtype=np.int64), image_size=(H, W), image_predictor=Image(),
            init_state=lambda:shared, seed_box=video.seed, propagate=video.propagate, emit=emitted.append)
    assert len(video.calls) == 1


def test_empty_native_masks_are_supported_and_zero_logits_use_native_strict_positive():
    emitted = []
    def propagate(state, **kw):
        if kw['reverse']: return
        for position in range(2):
            logits = np.zeros((1, 1, H, W), np.float32)
            if position: logits[0, 0, 0, 0] = np.nextafter(np.float32(0), np.float32(1))
            yield position, [0], logits
    s = stream_temporal_hand_masks([(i, np.zeros((H, W, 3), np.uint8), hands()) for i in range(2)],
        frame_ids=np.arange(2, dtype=np.int64), image_size=(H, W), image_predictor=Image(),
        init_state=dict, seed_box=lambda *a, **kw:None, propagate=propagate, emit=emitted.append)
    b = [r for r in emitted if r.branch == 'B']
    assert s.seeded_proposal_ids == (0,) and b[0].supported.all() and not b[0].masks.any()
    assert b[1].masks.sum() == 1 and b[1].masks[0, 0, 0]


@pytest.mark.parametrize('fault', ['seed', 'sink', 'propagate'])
def test_callback_exception_remains_failure_and_releases_owned_state(fault):
    video = Video(2); emitted = []
    def fail(*a, **kw): raise RuntimeError('procedural native failure')
    def emit(row):
        if fault == 'sink' and row.branch == 'B': fail()
        emitted.append(row)
    with pytest.raises(RuntimeError, match='procedural native failure'):
        stream_temporal_hand_masks([(i, np.zeros((H, W, 3), np.uint8), hands()) for i in range(2)],
            frame_ids=np.arange(2, dtype=np.int64), image_size=(H, W), image_predictor=Image(),
            init_state=video.init, seed_box=fail if fault == 'seed' else video.seed,
            propagate=fail if fault == 'propagate' else video.propagate, emit=emit, release_state=video.release)
    assert video.released == [0] and len(video.states) == 1


@pytest.mark.parametrize('fault', ['missing', 'extra', 'wrong_id', 'bool_id', 'rgb_dtype', 'rgb_shape'])
def test_input_timeline_errors_precede_any_video_state(fault):
    image, video = Image(), Video(2); rgb = np.zeros((H, W, 3), np.uint8)
    frames = [(10, rgb, hands()), (12, rgb, hands())]
    if fault == 'missing': frames.pop()
    if fault == 'extra': frames.append((14, rgb, hands()))
    if fault == 'wrong_id': frames[1] = (13, rgb, hands())
    if fault == 'bool_id': frames[0] = (True, rgb, hands())
    if fault == 'rgb_dtype': frames[0] = (10, rgb.astype(float), hands())
    if fault == 'rgb_shape': frames[0] = (10, rgb[:-1], hands())
    with pytest.raises(ValueError):
        stream_temporal_hand_masks(frames, frame_ids=np.array([10, 12], np.int64), image_size=(H, W),
            image_predictor=image, init_state=video.init, seed_box=video.seed, propagate=video.propagate, emit=lambda r:None)
    assert video.states == []


@pytest.mark.parametrize('ids', [np.array([], np.int64), np.array([1, 1], np.int64), np.array([2, 1], np.int64), np.array([-1], np.int64), np.array([0.], float)])
def test_declared_original_ids_fail_closed(ids):
    with pytest.raises(ValueError): run(frame_ids=ids)
