"""Frozen D88 RGB-only cohort contract and private manufacturing recipe.

The public reader returns no camera, geometry, masks or identity labels.
Manufacturing parameters are not estimated values or inference observations.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from world_reward.data import sha256


@dataclass(frozen=True)
class RGBProtocol:
    base: str = "validation/depth_rgb_v1"
    schema: str = "world-reward-depth-rgb-v1"
    width: int = 1024
    height: int = 768
    clips: int = 3
    frames: int = 5
    focals: tuple = (1200., 1500., 1800.)
    shapes: tuple = ((-.21, -.09), (.26, .11), (.43, -.12))
    scales: tuple = (-.03, .025, .06)
    bottle_depth_offsets: tuple = (-.12, -.06, 0., .065, .13)

    def __post_init__(self):
        if (any(type(v) is not int or v < 1 for v in (self.width, self.height, self.clips, self.frames))
                or not re.fullmatch(r"validation/[a-z0-9_]+", self.base)
                or not re.fullmatch(r"world-reward-[a-z0-9-]+", self.schema)
                or not all(isinstance(v, tuple) for v in (self.focals, self.shapes, self.scales, self.bottle_depth_offsets))
                or len(self.focals) != self.clips or len(self.shapes) != self.clips or len(self.scales) != self.clips
                or len(self.bottle_depth_offsets) != self.frames or any(not isinstance(v, tuple) or len(v) != 2 for v in self.shapes)):
            raise ValueError("Complete immutable RGB cohort recipe required")
        values = (*self.focals, *self.scales, *self.bottle_depth_offsets, *(v for pair in self.shapes for v in pair))
        if any(type(v) not in (int, float) or not np.isfinite(v) for v in values) or min(self.focals) <= 0:
            raise ValueError("Finite manufacturing parameters and positive focals required")

    def to_dict(self):
        return asdict(self)

    def digest(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


D88 = RGBProtocol()
BASE, SCHEMA = D88.base, D88.schema
WIDTH, HEIGHT, CLIPS, FRAMES = D88.width, D88.height, D88.clips, D88.frames
FIXED_K = np.array([[1280., 0., WIDTH/2], [0., 1280., HEIGHT/2], [0., 0., 1.]])


def public_manifest(digests):
    if len(digests) != CLIPS*FRAMES or any(not isinstance(v, str) or not re.fullmatch("[0-9a-f]{64}", v) for v in digests):
        raise ValueError("All fifteen original RGB hashes required")
    return {"schema": SCHEMA, "images": [dict(file=f"clip_{c:02d}_frame_{f:03d}.png",
        sha256=digests[c*FRAMES+f], width=WIDTH, height=HEIGHT) for c in range(CLIPS) for f in range(FRAMES)]}


def public_inputs(directory):
    """Strict regular, read-only RGB whitelist; no private observation input."""
    directory = Path(directory)
    if directory.resolve() != directory.absolute() or not directory.is_dir():
        raise ValueError("Canonical public input directory required")
    def identity(path):
        if path.resolve() != path.absolute() or not path.is_file() or path.stat().st_mode & 0o222:
            raise ValueError("Regular immutable public file required")
        return {"sha256": sha256(path), "bytes": path.stat().st_size}
    path = directory/"manifest.json"; receipt = identity(path); manifest = json.loads(path.read_text())
    if (not isinstance(manifest, dict) or set(manifest) != {"schema", "images"} or manifest["schema"] != SCHEMA
            or not isinstance(manifest["images"], list) or len(manifest["images"]) != CLIPS*FRAMES):
        raise ValueError("Exact fifteen-image RGB-only manifest required")
    records = []
    for index, row in enumerate(manifest["images"]):
        clip, frame = divmod(index, FRAMES); filename = f"clip_{clip:02d}_frame_{frame:03d}.png"
        if (not isinstance(row, dict) or set(row) != {"file", "sha256", "width", "height"} or row["file"] != filename
                or type(row["width"]) is not int or type(row["height"]) is not int or (row["width"], row["height"]) != (WIDTH, HEIGHT)
                or not isinstance(row["sha256"], str) or not re.fullmatch("[0-9a-f]{64}", row["sha256"])):
            raise ValueError("Ordered original-grid RGB identities required; no private fields")
        image = directory/filename; actual = identity(image)
        if actual["sha256"] != row["sha256"] or actual["bytes"] <= 0:
            raise ValueError("Public RGB SHA/size differs")
        records.append({**row, "path": image, "clip_index": clip, "frame_index": frame})
    if {p.name for p in directory.iterdir()} != {"manifest.json", *[row["file"] for row in records]}:
        raise ValueError("Only fifteen RGBs and manifest may be exposed")
    return records, receipt


def named_controls(names, limits, protocol=D88):
    """New fixed motions/identities with complete native249 bound validation."""
    bounds = np.asarray(limits)
    if (not isinstance(names, list) or len(names) != 249 or len(set(names)) != 249
            or any(not isinstance(n, str) or not n for n in names) or np.ma.isMaskedArray(limits)
            or bounds.shape != (249, 2) or bounds.dtype.kind != "f" or np.isnan(bounds).any()
            or np.any(bounds[:, 0] > bounds[:, 1])):
        raise ValueError("Actual unique249 native names and bounds required")
    sb = bounds[136:204]; locked = np.all(sb == 0, axis=1)
    patterns = np.broadcast_to(np.asarray(protocol.scales, np.float32)[:, None], (protocol.clips, 68)).copy(); patterns[:, locked] = 0
    if locked.all() or np.any(patterns < sb[:, 0]) or np.any(patterns > sb[:, 1]) or len({r.tobytes() for r in patterns}) != protocol.clips:
        raise ValueError("Three distinct frozen legal native scale patterns required; no clipping")
    controls = np.zeros((protocol.clips*protocol.frames, 204), np.float32); shapes = np.zeros((len(controls), 45), np.float32); changes = []
    for clip in range(protocol.clips):
        side = "l" if clip == 1 else "r"; other = "r" if side == "l" else "l"
        for frame in range(protocol.frames):
            row = clip*protocol.frames+frame; controls[row, 136:] = patterns[clip]; shapes[row, :2] = protocol.shapes[clip]
            recipe = [(f"{side}_uparm_ry", .15+.045*frame), (f"{side}_elbow_bend", .23+.06*frame),
                      (f"{side}_wrist_ry", -.035+.015*frame), (f"{other}_uparm_ry", .020*(frame-2))]
            recipe += [(f"{side}_{finger}1_rz", .105+.035*frame) for finger in ("index", "middle", "ring", "pinky")]
            for name, value in recipe:
                if name not in names[:136]: raise ValueError("Required native motion control absent: "+name)
                column = names.index(name); controls[row, column] = value
                changes.append(dict(clip_index=clip, frame_index=frame, name=name, column=column, value=value))
    full = np.c_[controls, shapes]; neutral = full[::protocol.frames].copy(); neutral[:, :136] = 0
    if np.any(full < bounds[:, 0]) or np.any(full > bounds[:, 1]) or np.any(neutral < bounds[:, 0]) or np.any(neutral > bounds[:, 1]):
        raise ValueError("Every animated and neutral249 value must obey actual bounds")
    return controls, shapes, changes
