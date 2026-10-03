"""Stdlib-only readonly public reader for the actual twelve-frame TUD-L holdout.

Expected identities come from independently committed pins, not the input
manifest. This reader never opens private acquisition/annotation files, decodes
media or estimates a camera. The frames share development scenes and objects.
"""
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import stat

SCHEMA = "world-reward-tudl-frame-holdout-rgb-v1"
PINS_SCHEMA = "world-reward-tudl-frame-holdout-input-pins-v1"
REVISION = "6527f7d4b25d3e2e8dec84529284d9797b15f7b5"
LICENSE = "CC-BY-SA-4.0"
SELECTION = "sorted_RGB_indices_40_80_120_160_per_scene_before_private_values"
FRAME_IDS = ((1788, 3235, 5138, 6925), (1566, 3137, 4894, 6490), (1512, 3227, 4819, 6647))
DEVELOPMENT_FRAME_IDS = ((0, 4074, 8227), (3, 4013, 7710), (4, 4028, 7969))
WIDTH, HEIGHT = 640, 480
ACQUISITION = {
    "sha256": "4b0f1309490d812ab2eec1bab91110ba953e11b285078a801dedda3b63540378",
    "bytes": 18380,
    "producer_revision": "7305c0fcdb8d079ab2bea216a6f33061d0a2a4cd",
    "script_sha256": "1cc7e7e80e37a5134e2d282d99106b5b28bf5e268476e85805598c8d3b5af6a8",
}


def filenames():
    return tuple(f"scene_{scene:06d}_frame_{frame:06d}.png"
        for scene, frames in enumerate(FRAME_IDS, 1) for frame in frames)


def _identity_record(value):
    if (type(value) is not dict or set(value) != {"sha256", "bytes"}
            or type(value["sha256"]) is not str or re.fullmatch(r"[0-9a-f]{64}", value["sha256"]) is None
            or type(value["bytes"]) is not int or value["bytes"] <= 0):
        raise ValueError("Exact independent SHA256/positive byte identity required")


def validate_pins(pins):
    if (type(pins) is not dict or set(pins) != {"schema", "acquisition_report", "public_files"}
            or type(pins["schema"]) is not str or pins["schema"] != PINS_SCHEMA):
        raise ValueError("Explicit independently committed holdout input pins required")
    receipt = pins["acquisition_report"]
    if (type(receipt) is not dict or set(receipt) != set(ACQUISITION)
            or any(type(receipt.get(key)) is not type(value) or receipt[key] != value for key, value in ACQUISITION.items())):
        raise ValueError("Exact actual acquisition producer/receipt identity required")
    files = pins["public_files"]
    if type(files) is not dict or set(files) != {"manifest.json", *filenames()}:
        raise ValueError("Exactly twelve original public RGB basenames plus manifest must be pinned")
    for value in files.values(): _identity_record(value)
    for frames, development in zip(FRAME_IDS, DEVELOPMENT_FRAME_IDS):
        if len(frames) != 4 or len(set(frames)) != 4 or set(frames) & set(development):
            raise ValueError("All actual held-out frame IDs must remain disjoint from development")


def _canonical(path):
    path = Path(path)
    if (not path.is_absolute() or path.resolve() != path
            or any(p.is_symlink() for p in (path, *path.parents))):
        raise ValueError("Canonical absolute nonsymlink public path required")
    return path


def _state(path):
    s = path.lstat()
    return (s.st_dev, s.st_ino, s.st_mode, s.st_nlink, s.st_size, s.st_mtime_ns, s.st_ctime_ns)


def identity(path):
    """Hash original bytes without decoding; fail closed on concurrent mutation."""
    path = _canonical(path); before = _state(path)
    if (not stat.S_ISREG(before[2]) or before[2] & 0o222 or before[3] != 1 or before[4] <= 0):
        raise ValueError("Nonempty readonly regular public file without aliases required")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""): digest.update(block)
    if _state(path) != before: raise ValueError("Original public file changed during hashing")
    return {"sha256": digest.hexdigest(), "bytes": before[4]}


def strict_json(data):
    def pairs(rows):
        result = {}
        for key, value in rows:
            if key in result: raise ValueError("Duplicate JSON keys forbidden")
            result[key] = value
        return result
    def invalid(_): raise ValueError("Nonfinite JSON constants forbidden")
    result = json.loads(data, object_pairs_hook=pairs, parse_constant=invalid)
    def finite(value):
        if type(value) is float and not math.isfinite(value): raise ValueError("Nonfinite JSON numbers forbidden")
        if type(value) is dict:
            for item in value.values(): finite(item)
        elif type(value) is list:
            for item in value: finite(item)
    finite(result)
    return result


def public_inputs(directory, pins):
    """Return ordered RGB records with Paths, plus pinned manifest byte identity.

All thirteen files are hashed against caller pins BEFORE any JSON is read and
again after interpretation. Acquisition identity is checked as metadata only;
its private receipt and every sibling/private directory remain unopened.
"""
    validate_pins(pins); expected = copy.deepcopy(pins)
    directory = _canonical(directory)
    if (directory.parts[-3:] != ("validation", "tudl_frame_holdout_v1", "inputs") or not directory.is_dir()):
        raise ValueError("Exact public holdout inputs namespace required")
    names = {"manifest.json", *filenames()}
    if {p.name for p in directory.iterdir()} != names:
        raise ValueError("Public directory contains exactly thirteen pinned files and no extras")
    before = {name: identity(directory / name) for name in sorted(names)}
    if before != expected["public_files"]: raise ValueError("Independent public file SHA/bytes differ")
    manifest_data = (directory / "manifest.json").read_bytes()
    if {"sha256": hashlib.sha256(manifest_data).hexdigest(), "bytes": len(manifest_data)} != before["manifest.json"]:
        raise ValueError("Frozen manifest changed before JSON interpretation")
    manifest = strict_json(manifest_data)
    fields = {"schema": SCHEMA, "revision": REVISION, "license": LICENSE, "selection": SELECTION}
    if (type(manifest) is not dict or set(manifest) != {*fields, "images"}
            or any(type(manifest.get(key)) is not type(value) or manifest[key] != value for key, value in fields.items())
            or type(manifest["images"]) is not list or len(manifest["images"]) != 12):
        raise ValueError("Exact actual twelve-frame RGB-only holdout manifest required")
    records = []
    for index, row in enumerate(manifest["images"]):
        scene = index // 4 + 1; frame = FRAME_IDS[scene - 1][index % 4]
        name = filenames()[index]
        if (type(row) is not dict or set(row) != {"scene_id", "frame_id", "file", "sha256", "width", "height"}
                or any(type(row.get(key)) is not int for key in ("scene_id", "frame_id", "width", "height"))
                or (row["scene_id"], row["frame_id"], row["width"], row["height"]) != (scene, frame, WIDTH, HEIGHT)
                or type(row["file"]) is not str or row["file"] != name
                or type(row["sha256"]) is not str or row["sha256"] != before[name]["sha256"]):
            raise ValueError("Original public scene/frame/order/grid/hash fields differ; private fields forbidden")
        records.append({**row, "path": directory / name})
    if ({p.name for p in directory.iterdir()} != names
            or {name: identity(directory / name) for name in sorted(names)} != before
            or pins != expected):
        raise ValueError("Public originals or independent pins changed during reading")
    return records, dict(before["manifest.json"])
