"""Tiny original-index fixtures; no Torch, model, media or remote runtime."""

from types import SimpleNamespace
import sys

import numpy as np
import pytest

from world_reward.timeline import (
    NATIVE_WINDOW, finite_chunks, first_occurrence_ownership,
    native_caller_window_indices, native_window_starts,
)


@pytest.mark.parametrize("count,expected", [
    (96, (0,)), (97, (0, 1)), (191, (0, 95)), (192, (0, 96)),
    (193, (0, 96, 97)), (288, (0, 96, 192)),
    (501, (0, 96, 192, 288, 384, 405)),
    (790, (0, 96, 192, 288, 384, 480, 576, 672, 694)),
])
def test_exact_full_original_native_windows(count, expected):
    assert native_window_starts(count) == expected
    assert native_window_starts(np.int64(count)) == expected


@pytest.mark.parametrize("bad", [0, -1, 95, True, False, np.bool_(True), 501., "501", None])
def test_native_windows_reject_short_or_coerced_frames(bad):
    with pytest.raises(ValueError):
        native_window_starts(bad)


@pytest.mark.parametrize("count,size,counts", [
    (1, 16, [1]), (15, 16, [15]), (16, 16, [16]), (17, 16, [16, 1]),
    (96, 16, [16] * 6), (501, 16, [16] * 31 + [5]),
    (5, 2, [2, 2, 1]), (3, 8, [3]),
])
def test_finite_chunks_cover_all_originals_with_actual_last_count(count, size, counts):
    chunks = finite_chunks(count, size)
    assert isinstance(chunks, tuple)
    assert [selection.stop - selection.start for selection in chunks] == counts
    assert all(selection.step is None for selection in chunks)
    assert [index for selection in chunks for index in range(selection.start, selection.stop)] == list(range(count))
    assert finite_chunks(np.int64(count), np.int64(size)) == chunks


@pytest.mark.parametrize("bad", [0, -1, True, np.bool_(True), 16., "16", None])
@pytest.mark.parametrize("role", ["count", "size"])
def test_finite_chunks_reject_invalid_integers(bad, role):
    with pytest.raises(ValueError):
        finite_chunks(bad, 16) if role == "count" else finite_chunks(96, bad)


def windows(count):
    return [np.arange(start, start + NATIVE_WINDOW, dtype=np.int64) for start in native_window_starts(count)]


@pytest.mark.parametrize("count", [96, 97, 191, 192, 193, 288, 501, 790])
def test_ownership_exact_first_occurrence_no_mutation_or_unowned_frames(count):
    actual = windows(count)
    before = [value.tobytes() for value in actual]
    owner, local = first_occurrence_ownership(count, tuple(actual))
    assert owner.shape == local.shape == (count,)
    assert owner.dtype == local.dtype == np.int64
    assert not owner.flags.writeable and not local.flags.writeable
    for index in range(count):
        first = next(ordinal for ordinal, value in enumerate(actual) if index in value)
        assert owner[index] == first
        assert actual[owner[index]][local[index]] == index
    assert before == [value.tobytes() for value in actual]
    assert not any(np.shares_memory(value, owner) or np.shares_memory(value, local) for value in actual)


def test_501_terminal_overlap_keeps384_window_until479_and405_only480to500():
    owner, local = first_occurrence_ownership(501, windows(501))
    np.testing.assert_array_equal(owner[384:480], np.full(96, 4, np.int64))
    np.testing.assert_array_equal(local[384:480], np.arange(96, dtype=np.int64))
    np.testing.assert_array_equal(owner[480:], np.full(21, 5, np.int64))
    np.testing.assert_array_equal(local[480:], np.arange(75, 96, dtype=np.int64))


@pytest.mark.parametrize("fault", [
    "missing", "repeat", "reorder", "padding", "drop", "duplicateframe", "dtype",
    "float", "list", "masked", "two_dimensional", "generator", "wrongterminal",
])
def test_ownership_rejects_any_noncanonical_actual_window_schedule(fault):
    actual = windows(193)
    if fault == "missing": actual.pop()
    elif fault == "repeat": actual.append(actual[-1].copy())
    elif fault == "reorder": actual.reverse()
    elif fault == "padding": actual[-1] = np.arange(192, 288, dtype=np.int64)
    elif fault == "drop": actual[-1] = actual[-1][:-1]
    elif fault == "duplicateframe": actual[-1][1] = actual[-1][0]
    elif fault == "dtype": actual[-1] = actual[-1].astype(np.int32)
    elif fault == "float": actual[-1] = actual[-1].astype(np.float64)
    elif fault == "list": actual[-1] = actual[-1].tolist()
    elif fault == "masked": actual[-1] = np.ma.array(actual[-1], mask=False)
    elif fault == "two_dimensional": actual[-1] = actual[-1][None]
    elif fault == "generator": actual = iter(actual)
    elif fault == "wrongterminal": actual[-1] = actual[-2].copy()
    with pytest.raises(ValueError):
        first_occurrence_ownership(193, actual)


def run_mhr_wild_inference(callback, *, total=193, ordinal=2, fault=None):
    """Native-shaped live frame, not a claim to be audited upstream code."""
    start = native_window_starts(total)[ordinal]
    clip_len = 96
    names = [f"{index:06d}" for index in range(total)]
    indices = np.arange(start, start + clip_len, dtype=np.int64)
    batch = {"unchanged": np.ones((1, 96, 3), np.float32)}
    if fault == "start": start -= 1
    elif fault == "startbool": start = True
    elif fault == "cliplen": clip_len = 95
    elif fault == "clipbool": clip_len = True
    elif fault == "indices": indices[-1] += 1
    elif fault == "indexdtype": indices = indices.astype(np.int32)
    elif fault == "indexmasked": indices = np.ma.array(indices, mask=False)
    elif fault == "names": names[-1] = "wrong"
    elif fault == "namesreorder": names.reverse()
    elif fault == "namestuple": names = tuple(names)
    elif fault == "missingnames": del names
    elif fault == "missingindices": del indices
    elif fault == "missingbatch": del batch
    returned = callback(locals().get("batch"), locals().get("indices"))
    return returned


def test_real_caller_indices_validated_without_mutating_any_runtime_local():
    def callback(batch, original):
        before = original.tobytes(), batch["unchanged"].tobytes()
        result = native_caller_window_indices(sys._getframe(1),
            native_code=run_mhr_wild_inference.__code__, total_frames=193, window_ordinal=2, batch=batch)
        assert before == (original.tobytes(), batch["unchanged"].tobytes())
        assert not result.flags.writeable and not np.shares_memory(result, original)
        return result
    np.testing.assert_array_equal(run_mhr_wild_inference(callback), np.arange(97, 193, dtype=np.int64))


@pytest.mark.parametrize("fault", [
    "start", "startbool", "cliplen", "clipbool", "indices", "indexdtype", "indexmasked",
    "names", "namesreorder", "namestuple", "missingnames", "missingindices", "missingbatch",
])
def test_actual_caller_locals_must_independently_match_exact_schedule(fault):
    def callback(batch, original):
        return native_caller_window_indices(sys._getframe(1),
            native_code=run_mhr_wild_inference.__code__, total_frames=193, window_ordinal=2, batch=batch)
    with pytest.raises(ValueError):
        run_mhr_wild_inference(callback, fault=fault)


@pytest.mark.parametrize("fault", ["wrongframe", "wrongcode", "batchcopy", "wrongordinal", "overrun", "ordinalbool", "codefake", "framefake", "wrongtotal"])
def test_caller_identity_batch_and_schedule_are_not_guessed_from_ordinal(fault):
    def callback(batch, original):
        caller, code, ordinal = sys._getframe(1), run_mhr_wild_inference.__code__, 2
        if fault == "wrongframe": caller = sys._getframe(0)
        elif fault == "wrongcode": code = callback.__code__
        elif fault == "batchcopy": batch = dict(batch)
        elif fault == "wrongordinal": ordinal = 1
        elif fault == "overrun": ordinal = 3
        elif fault == "ordinalbool": ordinal = True
        elif fault == "codefake": code = SimpleNamespace(co_name="run_mhr_wild_inference")
        elif fault == "framefake": caller = SimpleNamespace(f_code=code)
        return native_caller_window_indices(caller, native_code=code,
            total_frames=194 if fault == "wrongtotal" else 193, window_ordinal=ordinal, batch=batch)
    with pytest.raises(ValueError):
        run_mhr_wild_inference(callback)


def test_identical_inputs_do_not_hide_different_actual_terminal_window_indices():
    def callback(batch, original):
        return native_caller_window_indices(sys._getframe(1),
            native_code=run_mhr_wild_inference.__code__, total_frames=193,
            window_ordinal=2 if original[0] == 97 else 1, batch=batch)
    first = run_mhr_wild_inference(callback, ordinal=1)
    terminal = run_mhr_wild_inference(callback, ordinal=2)
    assert first[0] == 96 and terminal[0] == 97
    assert not np.array_equal(first, terminal)


@pytest.mark.parametrize("total", [96, 97, 192, 193, 501, 790])
def test_every_actual_fullclip_caller_window_retains_original_indices(total):
    for ordinal, start in enumerate(native_window_starts(total)):
        def callback(batch, original):
            return native_caller_window_indices(sys._getframe(1),
                native_code=run_mhr_wild_inference.__code__, total_frames=total,
                window_ordinal=ordinal, batch=batch)
        actual = run_mhr_wild_inference(callback, total=total, ordinal=ordinal)
        np.testing.assert_array_equal(actual, np.arange(start, start + 96, dtype=np.int64))
