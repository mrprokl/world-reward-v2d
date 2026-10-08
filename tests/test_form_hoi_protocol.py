import hashlib
import json
from pathlib import Path

import pytest

from world_reward.form_hoi_protocol import (
    FRAMES, REQUIRED_GATES, audit_archives, freeze_cohort, require_ready_for_inference,
    sequence_parts, validate_inference_package, verify_metadata_bytes,
)


def row(sid, serial=1):
    return {"sequence_id": sid, "archive_path": f"data/{sid}.tar", "archive_size": 100,
            "archive_sha256": hashlib.sha256(f"{sid}/{serial}".encode()).hexdigest(), "member_count": 5}


CHALLENGE = [{"sequence_id": "2026-03-16_15-04-26_hula_hoop_inside_ground_01", "object": "hula_hoop"}]


def test_metadata_pin_rejects_wrong_bytes_and_size():
    data = b"only metadata"
    verify_metadata_bytes(data, expected_size=len(data), sha256=hashlib.sha256(data).hexdigest())
    for data2, size in [(b"different", len(data)), (data, len(data) + 1)]:
        with pytest.raises(ValueError, match="provenance"):
            verify_metadata_bytes(data2, expected_size=size, sha256=hashlib.sha256(data).hexdigest())


def test_admission_rejects_exact_id_timestamp_alias_and_object_alias():
    rows = [row(CHALLENGE[0]["sequence_id"]),
            row("2026-03-16_15-04-26_different_object_alias_01"),
            row("2026-04-16_15-04-26_hula_hoop_pick_02"),
            row("2026-04-16_15-04-27_blue_trash_can_drag_02")]
    admitted, result = audit_archives(rows, CHALLENGE)
    assert [item["sequence_id"] for item in admitted] == [rows[-1]["sequence_id"]]
    assert result["exact_sequence_overlap"] == 1
    assert result["recording_timestamp_overlap"] == 2
    assert result["challenge_object_slug_prefix_overlap"] == 2
    assert result["metadata_admission_passed"] is False


def test_anti_alias_object_normalization_and_timestamp():
    challenge = [{**CHALLENGE[0], "object": "Hula Hoop"}]
    admitted, result = audit_archives([row("2026-04-16_15-04-26_hula_hoop_pick_02")], challenge)
    assert not admitted and result["challenge_object_slug_prefix_overlap"] == 1


def test_admission_does_not_confuse_action_object_with_target_prefix():
    admitted, result = audit_archives([row("2026-04-16_15-04-26_blue_trash_can_hula_hoop_side_02")], CHALLENGE)
    assert len(admitted) == 1
    assert result["metadata_admission_passed"] is True
    assert result["content_duplicate_check_passed"] is False
    assert result["quality_verified"] is False


def test_reject_duplicate_digest_and_unsafe_path():
    valid = row("2026-04-16_15-04-26_blue_trash_can_drag_02")
    with pytest.raises(ValueError, match="Duplicate"):
        audit_archives([valid, valid], CHALLENGE)
    with pytest.raises(ValueError, match="path"):
        audit_archives([{**valid, "archive_path": "../../forbidden.tar"}], CHALLENGE)
    with pytest.raises(ValueError, match="empty|Empty"):
        audit_archives([valid], [])


def test_unsupported_legacy_ids_are_excluded_not_repaired():
    bad = row("legacy_bad_id")
    admitted, report = audit_archives([bad], CHALLENGE)
    assert not admitted and report["unsupported_sequence_names_excluded"] == 1
    with pytest.raises(ValueError):
        sequence_parts("../../2026-04-16_15-04-26_blue_trash_can_drag_02")


def cohort_rows():
    return [row(f"2026-04-{day:02d}_15-04-{take:02d}_{group}_pick_{take:02d}")
            for day, group in enumerate(["blue_trash", "wooden_crate", "red_bucket", "giant_box",
                                         "soccer_ball", "toy_car", "step_ladder", "dark_book"], 1)
            for take in range(1, 4)]


def test_cohort_order_invariant_unique_family_and_reserved_groups():
    rows = cohort_rows()
    result = freeze_cohort(rows, seed="test")
    assert result == freeze_cohort(reversed(rows), seed="test")
    assert len(result) == 8
    assert len({item["provisional_sequence_family"] for item in result}) == 8
    assert sum(item["split"] == "reserved_unopened" for item in result) == 4
    assert all(item["family_metadata_verified_object_id"] is False for item in result)
    assert all(item["original_frame_indices"] == list(FRAMES) for item in result)
    with pytest.raises(ValueError, match="Insufficient"):
        freeze_cohort(rows[:3], seed="test")


def test_cohort_keeps_single_word_object_different_actions_in_same_family():
    rows = [row(f"2026-04-{day:02d}_15-04-{take:02d}_vacuum_{action}_{take:02d}")
            for day, action in enumerate(["horizontal", "vacuum"], 1)
            for take in range(1, 4)]
    with pytest.raises(ValueError, match="Insufficient"):
        freeze_cohort(rows, seed="test", count=2)


def test_readiness_requires_every_gate_true():
    admission = {"metadata_admission_passed": True}
    gates = dict.fromkeys(REQUIRED_GATES, True)
    require_ready_for_inference(admission, gates)
    for key in REQUIRED_GATES:
        with pytest.raises(ValueError, match="blocked"):
            require_ready_for_inference(admission, {**gates, key: False})
    with pytest.raises(ValueError, match="admission"):
        require_ready_for_inference({"metadata_admission_passed": False}, gates)


def package():
    return {"sequence_id": "2026-04-16_15-04-26_blue_trash_can_drag_02",
            "rgb_video": "/srv/form-hoi-inference/clip.mp4", "rgb_sha256": "a" * 64,
            "object_prompt": "blue trash can", "action": "drag the trash can",
            "camera": "front_stereo_camera_left", "original_frame_indices": list(FRAMES)}


@pytest.mark.parametrize("key", ["masks", "source_intrinsics", "reference_pose", "mesh", "depth", "other_view"])
def test_inference_rejects_annotation_and_multiview_inputs(key):
    validate_inference_package(package())
    with pytest.raises(ValueError, match="forbidden"):
        validate_inference_package({**package(), key: "not allowed"})


def test_inference_rejects_reference_route_or_changed_frame_grid():
    with pytest.raises(ValueError, match="Reference"):
        validate_inference_package({**package(), "rgb_video": "/srv/references/clip.mp4"})
    with pytest.raises(ValueError, match="grid"):
        validate_inference_package({**package(), "original_frame_indices": [0, 15, 30]})
    with pytest.raises(ValueError, match="grid"):
        validate_inference_package({**package(), "original_frame_indices": [False, 14, 29]})


def test_archive_admission_cannot_pass_with_no_eligible_rows():
    with pytest.raises(ValueError, match="Empty"):
        audit_archives([], CHALLENGE)
    _, admission = audit_archives([row("legacy_bad_id")], CHALLENGE)
    assert admission["metadata_admission_passed"] is False


def test_frozen_repository_config_is_metadata_only_not_inference_ready():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/form_hoi_insight_v1.json").read_text())
    audit = json.loads((root / "results/audits/form_hoi_insight_v1_metadata.json").read_text())
    assert len(config["cohort"]) == 8
    assert config["scope"] == "metadata_only_external_insight_admission"
    assert audit["actual_heavy_data_acquired"] is False
    assert audit["metadata_admission"]["metadata_admission_passed"] is True
    assert config["inference_ready"] is False
    with pytest.raises(ValueError, match="blocked"):
        require_ready_for_inference(audit["metadata_admission"], config["qualification_gates"])
