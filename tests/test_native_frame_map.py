"""Tiny metadata tests; no data, model, native trajectory math or private reads."""
from dataclasses import FrozenInstanceError
import copy
import hashlib
import json
import pytest
from world_reward.native_frame_map import NativeFrameMap


@pytest.mark.parametrize("first", [0, 1, 27])
def test_native_ids_preserved_separately_from_array_positions(first):
    mapping = NativeFrameMap("external_sequence", tuple(range(first, first + 96)))
    assert mapping.frame_positions == tuple(range(96))
    assert mapping.source_id(0) == first and mapping.position(first) == 0
    assert mapping.source_id(95) == first + 95
    mapping.validate_attachment(tuple(range(96)), tuple(range(first, first + 96)))
    mapping.validate_rows(mapping.to_dict()["frames"])
    raw = mapping.to_json_bytes()
    assert NativeFrameMap.from_dict(json.loads(raw)) == mapping
    assert mapping.receipt() == {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def test_query_position_zero_means_original_source_one_not_renumbering():
    mapping = NativeFrameMap("source_start_one", (1, 2, 3))
    assert mapping.source_id(0) == 1 and mapping.to_dict()["frames"][0] == {"frame_position": 0, "source_frame_id": 1}
    with pytest.raises(ValueError): mapping.position(0)
    with pytest.raises(ValueError): mapping.validate_attachment((0, 1, 2), (0, 1, 2))


@pytest.mark.parametrize("ids", [(), [], [1, 2], (1, 1), (2, 1), (1, 3), (-1, 0), (True, 2), (1., 2), (2**63,), (0, None)])
def test_missing_duplicated_invalid_noncontiguous_source_ids_rejected(ids):
    with pytest.raises(ValueError): NativeFrameMap("seq", ids)


@pytest.mark.parametrize("sequence", ["", " ", " padded", "x\n", "a\x00b", None, 1, True])
def test_explicit_sequence_identity_required(sequence):
    with pytest.raises(ValueError): NativeFrameMap(sequence, (0,))


def test_immutable_one_frame_and_receipt_copies_do_not_alias():
    mapping = NativeFrameMap("single", (10,))
    with pytest.raises(FrozenInstanceError): mapping.source_frame_ids = (11,)
    receipt = mapping.to_dict(); receipt["frames"][0]["source_frame_id"] = 99
    assert mapping.source_id(0) == 10
    assert NativeFrameMap.from_dict(mapping.to_dict()) == mapping


@pytest.mark.parametrize("fault", ["schema", "extra", "missing", "bool_position", "bool_source", "reorder", "count", "source", "position", "sequence"])
def test_serialized_correspondence_strict_before_any_join(fault):
    mapping = NativeFrameMap("seq", (1, 2, 3)); value = copy.deepcopy(mapping.to_dict())
    if fault == "schema": value["schema"] += "wrong"
    elif fault == "extra": value["frame_index"] = [0, 1, 2]
    elif fault == "missing": value["frames"][0].pop("source_frame_id")
    elif fault == "bool_position": value["frames"][0]["frame_position"] = False
    elif fault == "bool_source": value["frames"][0]["source_frame_id"] = True
    elif fault == "reorder": value["frames"].reverse()
    elif fault == "count": value["frames"].pop()
    elif fault == "source": value["frames"][1]["source_frame_id"] = 7
    elif fault == "position": value["frames"][1]["frame_position"] = 2
    else: value["sequence_id"] = None
    if fault == "count":
        # A standalone shorter valid map is legal; attachment to the frozen96 is not.
        shorter = NativeFrameMap.from_dict(value)
        with pytest.raises(ValueError): mapping.validate_attachment(shorter.frame_positions, shorter.source_frame_ids)
    else:
        with pytest.raises(ValueError): NativeFrameMap.from_dict(value)


@pytest.mark.parametrize("positions,ids", [([0, 1], (1, 2)), ((0, 1), [1, 2]), ((False, 1), (1, 2)), ((0, 1), (True, 2)), ((1, 0), (1, 2)), ((0,), (1,)), ((0, 1), (2, 3))])
def test_exact_tuple_attachment_rejects_implicit_casts_or_mismatches(positions, ids):
    with pytest.raises(ValueError): NativeFrameMap("seq", (1, 2)).validate_attachment(positions, ids)


@pytest.mark.parametrize("index", [True, -1, 1., None, 4])
def test_invalid_or_out_of_range_position(index):
    with pytest.raises(ValueError): NativeFrameMap("seq", (1, 2, 3)).source_id(index)


@pytest.mark.parametrize("fault", ["bytes", "hash", "count", "bool_count", "extra", "sequence", "missing_row"])
def test_canonical_receipt_bytes_and_count_before_private_join(fault):
    mapping = NativeFrameMap("seq", (1, 2, 3)); raw = mapping.to_json_bytes(); pin = mapping.receipt()
    mapping.validate_receipt(raw, pin)
    if fault == "bytes": raw += b" "
    elif fault == "hash": pin["sha256"] = "0" * 64
    elif fault == "count": pin["bytes"] -= 1
    elif fault == "bool_count": pin["bytes"] = True
    elif fault == "extra": pin["extra"] = 0
    elif fault == "sequence": raw = NativeFrameMap("another_sequence", (1, 2, 3)).to_json_bytes()
    else: raw = NativeFrameMap("seq", (1, 2)).to_json_bytes()
    with pytest.raises(ValueError): mapping.validate_receipt(raw, pin)


def test_module_stdlib_only_no_data_models_numpy_or_torch():
    import ast
    from pathlib import Path
    import world_reward.native_frame_map as module
    tree = ast.parse(Path(module.__file__).read_text())
    imports = {n.module.split(".")[0] if isinstance(n, ast.ImportFrom) else a.name.split(".")[0]
               for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))
               for a in (n.names if isinstance(n, ast.Import) else [None])}
    assert imports <= {"__future__", "dataclasses", "hashlib", "json"}
