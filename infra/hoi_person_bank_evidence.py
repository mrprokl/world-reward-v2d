"""Saved all-person/HOI Cartesian evidence; never a selector or ownership test.

The root driver authenticates receipt/file pins, sources, runtime and mounts.
This helper verifies the actual person-bank saver ABI and numerical transport.
No model, native inference, GT, threshold fitting or temporal identity is added.
"""
from collections.abc import Mapping
from dataclasses import fields
import hashlib
import json
import os
from pathlib import Path
import stat

import numpy as np

from world_reward.person_pose_observations import PersonPoseObservations
from world_reward.hoi_detr_observations import HOIDetrObservations
from world_reward.interaction_tuple_evidence import build_interaction_tuple_evidence

SCHEMA = "world_reward.hoi_person_bank_evidence.v1"
PERSON_ARRAYS = ("boxes_original_xyxy", "detector_scores", "keypoints_original_xy",
                 "raw_scores", "native_valid", "in_original_image")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def array_identity(value):
    require(type(value) is np.ndarray and not value.dtype.hasobject, "Plain numeric evidence array required")
    return dict(dtype=str(value.dtype), shape=list(value.shape),
                sha256=hashlib.sha256(value.tobytes(order="C")).hexdigest())


def reconstruct_saved_person(row, arrays, rgb):
    """Read exact six-array NPZ ABI against its frozen producer bank row.

    decoded_RGB_sha256 is the ACTUAL original saver field. image_size is derived
    from that same decoded RGB (the historical row does not contain it).
    Missed and off-grid points stay untouched; raw_scores>0 alone is native_valid.
    """
    require(type(row) is dict and isinstance(arrays, Mapping) and set(arrays) == set(PERSON_ARRAYS),
            "Exact saved person-bank fields required")
    require(type(rgb) is np.ndarray and rgb.dtype == np.uint8 and rgb.ndim == 3
            and rgb.shape[2] == 3 and min(rgb.shape[:2]) > 0, "Original uint8 RGB required")
    require(type(row.get("frame_index")) is int and row["frame_index"] >= 0
            and type(row.get("episode")) is int and row["episode"] >= 0
            and type(row.get("persons")) is int and row["persons"] >= 0, "Original frame/episode/count required")
    ids = row.get("person_ids")
    require(type(ids) in (list, tuple), "Original ordered proposal IDs required")
    frame, episode, count = row["frame_index"], row["episode"], row["persons"]
    expected = tuple(f"episode:{episode:06d}/frame:{frame:06d}/person/retained:{i:06d}" for i in range(count))
    require(tuple(ids) == expected and row.get("prediction_file") == f"episode_{episode:06d}_frame_{frame:06d}.npz",
            "Original proposal IDs/order and frame-bound prediction name required")
    require(hashlib.sha256(rgb.tobytes(order="C")).hexdigest() == row.get("decoded_RGB_sha256"),
            "Original decoded RGB bytes differ")
    dtypes = (np.float64, np.float64, np.float64, np.float32, np.bool_, np.bool_)
    for name, dtype in zip(PERSON_ARRAYS, dtypes):
        require(type(arrays[name]) is np.ndarray and arrays[name].dtype == dtype,
                "Actual native saved person array dtype differs")
    before = {k: array_identity(arrays[k]) for k in PERSON_ARRAYS}
    person = PersonPoseObservations(frame, tuple(map(int, rgb.shape[:2])), expected,
        arrays["boxes_original_xyxy"], arrays["detector_scores"], arrays["keypoints_original_xy"], arrays["raw_scores"])
    for name in ("native_valid", "in_original_image"):
        require(arrays[name].shape == (count, 133) and np.array_equal(arrays[name], getattr(person, name)),
                "Stored native validity/original-grid flags differ")
    require(array_identity(person.keypoints_original_xy) == row.get("keypoints")
            and array_identity(person.raw_scores) == row.get("scores"), "Saved native point/score identities differ")
    require(before == {k: array_identity(arrays[k]) for k in PERSON_ARRAYS}, "Saved source arrays changed")
    return person


def save_tuple_evidence(outstem, person, hoi):
    """Save all raw banks and every Nperson*2*P tuple, with NaN unsupported values.

    Fresh .npz/.json files are exclusive, fsynced and readonly. The metadata is
    byte-reference transport only, not independently authenticated provenance.
    Returned metadata contains no chosen person, object, rank or probability.
    """
    require(type(person) is PersonPoseObservations and type(hoi) is HOIDetrObservations,
            "Actual immutable observation contracts required")
    evidence = build_interaction_tuple_evidence(person, hoi)
    payload = {"person__"+name: getattr(person, name) for name in PERSON_ARRAYS}
    payload.update({"hoi__"+f.name: getattr(hoi, f.name) for f in fields(hoi)
                    if type(getattr(hoi, f.name)) is np.ndarray})
    payload.update({"tuple__"+name: value for name, value in evidence.arrays.items()})
    payload.update(tuple__features=evidence.features, tuple__feature_supported=evidence.feature_supported)
    identities = {k: array_identity(v) for k, v in payload.items()}
    n, p = len(person.person_ids), len(hoi.hand_object_pairs)
    require(evidence.features.shape == (n*2*p, len(evidence.feature_names)), "Complete Cartesian tuple count required")
    stem = Path(outstem)
    require(stem.is_absolute() and stem.parent.resolve() == stem.parent and stem.parent.is_dir()
            and not any(p.is_symlink() for p in (stem.parent, *stem.parent.parents)), "Canonical output parent required")
    npz, receipt = Path(str(stem)+".npz"), Path(str(stem)+".json")
    require(not npz.exists() and not receipt.exists() and not npz.is_symlink() and not receipt.is_symlink(),
            "Fresh evidence filenames required")
    metadata = dict(schema=SCHEMA, original_frame_index=person.original_frame_index, image_size=list(person.image_size),
        source_person_ids=list(person.person_ids), source_observation_references=[list(x) for x in evidence.source_observation_references],
        persons=n, native_detections=len(hoi.query_ids), native_hand_object_pairs=p,
        native_object_target_pairs=len(hoi.object_target_pairs), tuple_rows=n*2*p,
        feature_names=list(evidence.feature_names), arrays=identities,
        selection_performed=False, anatomical_ownership_verified=False, physical_ownership_verified=False,
        calibration_verified=False, accuracy_verified=False, adopted=False)
    owned = {}
    def create(path, writer):
        fd = os.open(path, os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW, 0o600)
        s = os.fstat(fd); owned[path] = (s.st_dev, s.st_ino, s.st_uid)
        with os.fdopen(fd, "wb") as stream:
            writer(stream); stream.flush(); os.fsync(stream.fileno()); os.fchmod(stream.fileno(), 0o444)
    try:
        create(npz, lambda stream: np.savez(stream, **payload))
        require(identities == {k: array_identity(v) for k, v in payload.items()}, "Evidence changed while saving")
        raw = npz.read_bytes()
        metadata["prediction"] = dict(file=npz.name, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        create(receipt, lambda stream: stream.write((json.dumps(metadata, sort_keys=True, allow_nan=False)+"\n").encode()))
        for path, key in owned.items():
            s = path.lstat()
            require(stat.S_ISREG(s.st_mode) and s.st_nlink == 1 and (s.st_dev, s.st_ino, s.st_uid) == key
                    and stat.S_IMODE(s.st_mode) == 0o444, "Owned readonly output changed")
    except BaseException:
        for path, key in owned.items():
            s = path.lstat()
            require(stat.S_ISREG(s.st_mode) and s.st_nlink == 1 and (s.st_dev, s.st_ino, s.st_uid) == key,
                    "Never remove foreign evidence output")
            path.unlink()
        raise
    return metadata
