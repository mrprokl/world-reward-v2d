"""Procedural 2x2 images and callbacks only; no native model or runtime."""
from dataclasses import FrozenInstanceError, replace
import weakref

import numpy as np
import pytest

from world_reward.hand_scan import (
    PROTOCOL, HandScanResult, ImageHandProtocol, NativeHandResult, scan_hands,
)


def native(count=1):
    return NativeHandResult(np.arange(count * 63, dtype=np.float32).reshape(count, 21, 3) / 100,
        np.zeros((count, 21, 3), np.float32), tuple("Left" if i % 2 else "Right" for i in range(count)),
        np.full(count, .8, np.float32))


def frames(count=3):
    return ((i, np.full((2, 2, 3), i, np.uint8)) for i in range(count))


def scan(**changes):
    return scan_hands(**(dict(frames=frames(), callback=lambda i, rgb: native(), total_frames=3,
        image_size=(2, 2), method="automatic test callback", source_refs={"revision": "a" * 40},
        budget_seconds=10, monotonic=lambda: 0.) | changes))


def test_original_callback_once_all_frames_all_slots_and_empty():
    images = [np.full((2, 2, 3), i, np.uint8) for i in range(3)]
    counts = [0, 4, 6]  # more than capacity is preserved, not silently clipped
    calls = []
    def callback(i, rgb):
        assert rgb is images[i]
        calls.append((i, rgb[0, 0, 0]))
        return native(counts[i])
    result = scan(frames=iter(enumerate(images)), callback=callback)
    assert calls == [(0, 0), (1, 1), (2, 2)]
    assert [r.normalized_xyz.shape[0] for r in result.native_results] == counts
    assert [r.count for r in result.observations.frames] == counts
    assert result.capacity_saturation.tolist() == [False, True, False]
    assert not result.capacity_saturation.flags.writeable
    assert result.observations.frame_index.tolist() == [0, 1, 2]
    assert result.observations.timestamps_seconds is None


def test_native_arrays_independent_readonly_raw_bytes_and_separate_coordinates():
    xyz = np.zeros((2, 21, 3), np.float64)
    xyz[0, 0] = [-.1, .25, np.nan]
    xyz[1, 0] = [.5, np.inf, -0.0]
    world = np.full((2, 21, 3), .3, np.float32)
    scores = np.array([.999, .501], np.float32)
    answer = NativeHandResult(xyz, world, ("Left", "Right"), scores)
    result = scan(callback=lambda i, rgb: answer)
    xyz[:] = 9; world[:] = 9; scores[:] = 0
    for i in range(3):
        raw = result.native_results[i]
        assert raw.normalized_xyz.tobytes() == answer.normalized_xyz.tobytes()
        assert not np.shares_memory(raw.normalized_xyz, answer.normalized_xyz)
        assert not raw.normalized_xyz.flags.writeable
        converted = result.observations.frames[i]
        expected = answer.normalized_xyz[..., :2] * np.array([2, 2], np.float64)
        assert converted.original_xy.values.tobytes() == expected.tobytes()
        assert converted.image_z.values.tobytes() == answer.normalized_xyz[..., 2:3].tobytes()
        assert not converted.original_xy.supported[1, 0] and not converted.image_z.supported[0, 0]
        assert result.observations.outside_image[i][0, 0]
        assert converted.hand_centred_world_xyz.coordinate_frame == "hand_centred_world"
        assert converted.native_handedness_labels == ("Left", "Right")
    assert not np.shares_memory(result.native_results[0].normalized_xyz, result.native_results[1].normalized_xyz)
    with pytest.raises(ValueError): result.native_results[0].normalized_xyz[0, 0, 0] = 4
    with pytest.raises(FrozenInstanceError): result.elapsed_seconds = 1.
    assert not hasattr(result, "accepted") and not hasattr(result, "visibility")


def test_does_not_retain_or_stack_full_rgb_frames():
    refs = []
    def stream():
        for i in range(3):
            if refs:
                assert refs[-1]() is None
            rgb = np.full((2, 2, 3), i, np.uint8)
            refs.append(weakref.ref(rgb))
            yield i, rgb
            del rgb
    result = scan(frames=stream())
    assert all(ref() is None for ref in refs)
    assert len(result.native_results) == 3


@pytest.mark.parametrize("items", [list(frames(2)), list(frames(4)),
    [(1, np.zeros((2, 2, 3), np.uint8))],
    [(0, np.zeros((2, 2, 3), np.uint8)), (0, np.zeros((2, 2, 3), np.uint8))],
    [(np.int64(0), np.zeros((2, 2, 3), np.uint8))],
    [(False, np.zeros((2, 2, 3), np.uint8))],
])
def test_missing_extra_duplicate_or_reordered_frames_fail(items):
    with pytest.raises(ValueError): scan(frames=items)


@pytest.mark.parametrize("rgb", [np.zeros((2, 2, 3), np.float32), np.zeros((2, 2), np.uint8),
    np.zeros((3, 2, 3), np.uint8), np.zeros((2, 2, 4), np.uint8),
    np.ma.array(np.zeros((2, 2, 3), np.uint8), mask=False), [[[0, 0, 0]] * 2] * 2])
def test_bad_rgb_not_converted_or_called(rgb):
    called = []
    with pytest.raises(ValueError): scan(frames=[(0, rgb)], callback=lambda *args: called.append(args))
    assert not called


def test_callback_and_reader_errors_propagate_without_retry_or_empty():
    called = []
    class NativeError(Exception): pass
    def callback(i, rgb):
        called.append(i)
        if i == 1: raise NativeError("native failed")
        return native(0)
    with pytest.raises(NativeError, match="native failed"): scan(callback=callback)
    assert called == [0, 1]
    def stream():
        yield 0, np.zeros((2, 2, 3), np.uint8)
        raise NativeError("reader failed")
    with pytest.raises(NativeError, match="reader failed"): scan(frames=stream())
    with pytest.raises(ValueError): scan(callback=lambda *args: None)


def test_deadline_before_after_reader_and_after_final_callback():
    clock = [0.]
    called = []
    def callback(i, rgb):
        called.append(i); clock[0] += 1.
        return native()
    with pytest.raises(TimeoutError): scan(callback=callback, monotonic=lambda: clock[0], budget_seconds=2.5)
    assert called == [0, 1, 2]  # last call's overrun is still failure
    clock[0] = 0.; called.clear()
    def reader():
        clock[0] = 11.
        yield 0, np.zeros((2, 2, 3), np.uint8)
    with pytest.raises(TimeoutError): scan(frames=reader(), callback=callback, monotonic=lambda: clock[0])
    assert not called


def test_final_construction_included_in_scan_budget(monkeypatch):
    import world_reward.hand_scan as module
    original = module.HandScanResult
    clock = [0.]
    def constructor(*args):
        result = original(*args)
        clock[0] = 11.
        return result
    monkeypatch.setattr(module, "HandScanResult", constructor)
    with pytest.raises(TimeoutError): scan(monotonic=lambda: clock[0])


@pytest.mark.parametrize("change", [{"total_frames": True}, {"total_frames": 0}, {"budget_seconds": True},
    {"budget_seconds": 0}, {"budget_seconds": np.inf}, {"budget_seconds": np.nan},
    {"callback": None}, {"source_refs": {}}, {"image_size": (True, 2)},
    {"timestamps_seconds": np.array([0., 0., 1.])}, {"timestamps_seconds": np.array([0., np.nan, 1.])}])
def test_protocol_metadata_validated_before_any_native_call(change):
    called = []
    args = {"callback": lambda *a: called.append(a)} | change
    with pytest.raises(ValueError): scan(**args)
    assert not called


def test_real_times_frozen_without_implied_video_mode_or_fps():
    times = np.array([1., 1.017, 1.051])
    result = scan(timestamps_seconds=times)
    assert result.protocol.running_mode == "IMAGE"
    assert np.array_equal(result.observations.timestamps_seconds, times)
    times[:] = 0
    assert result.observations.timestamps_seconds[1] == 1.017


@pytest.mark.parametrize("change", [{"running_mode": "VIDEO"}, {"num_hands": 1}, {"num_hands": True},
    {"min_hand_detection_confidence": .3}, {"min_hand_presence_confidence": .6},
    {"min_tracking_confidence": 0}])
def test_fixed_image_protocol_cannot_change_capacity_or_thresholds(change):
    with pytest.raises(ValueError): ImageHandProtocol(**change)
    assert PROTOCOL.num_hands == 4 and PROTOCOL.min_hand_detection_confidence == .5


def test_native_and_result_contracts_cannot_replace_raw_or_labels():
    with pytest.raises(ValueError): NativeHandResult(np.zeros((1, 20, 3)))
    with pytest.raises(ValueError): NativeHandResult(np.zeros((1, 21, 3), int))
    with pytest.raises(ValueError): NativeHandResult(np.zeros((1, 21, 3)), native_handedness_scores=np.array([.5]))
    result = scan()
    with pytest.raises(ValueError): replace(result, native_results=(native(0),) * 3)
    with pytest.raises(ValueError): replace(result, elapsed_seconds=float("nan"))
    wrong = replace(result.observations.frames[0], native_handedness_labels=("Left",))
    with pytest.raises(ValueError): replace(result, observations=replace(result.observations,
        frames=(wrong,) + result.observations.frames[1:]))


@pytest.mark.parametrize("ticks", [[0., -1.], [0., float("nan")], [float("inf")]])
def test_invalid_clock_failclosed(ticks):
    values = iter(ticks)
    with pytest.raises(ValueError): scan(monotonic=lambda: next(values))
