"""Immutable H101 public RGB-only contract, with no array/model imports."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import struct
import zlib


@dataclass(frozen=True)
class PublicCohort:
    base: str = "validation/human_photometric_v1"
    schema: str = "world-reward-human-photometric-v1"
    groups: int = 8
    frames: int = 3
    width: int = 1024
    height: int = 768
    prior_focal: float = 1280.

    def __post_init__(self):
        expected = ("validation/human_photometric_v1", "world-reward-human-photometric-v1", 8, 3, 1024, 768, 1280.)
        values = (self.base, self.schema, self.groups, self.frames, self.width, self.height, self.prior_focal)
        if any(type(a) is not type(b) or a != b for a, b in zip(values, expected)):
            raise ValueError("Exact H101 public contract required")

    @property
    def count(self): return self.groups * self.frames

    @property
    def fixed_K(self):
        return ((self.prior_focal, 0., self.width / 2), (0., self.prior_focal, self.height / 2), (0., 0., 1.))

    def frame_name(self, index):
        if type(index) is not int or not 0 <= index < self.count: raise ValueError("Original public frame index required")
        group, frame = divmod(index, self.frames)
        return f"group_{group:02d}_frame_{frame:03d}.png"


COHORT = PublicCohort()
BASE, SCHEMA = COHORT.base, COHORT.schema
GROUPS, FRAMES, WIDTH, HEIGHT = COHORT.groups, COHORT.frames, COHORT.width, COHORT.height
FIXED_K = COHORT.fixed_K


def identity(path):
    path = Path(path)
    if (not path.is_file() or path.resolve() != path.absolute() or path.stat().st_mode & 0o222
            or any(p.is_symlink() for p in (path, *path.parents))):
        raise ValueError("Canonical immutable regular public file required")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""): digest.update(chunk)
    return dict(sha256=digest.hexdigest(), bytes=path.stat().st_size)


def public_manifest(digests, cohort=COHORT):
    if (type(cohort) is not PublicCohort or type(digests) is not list or len(digests) != cohort.count
            or any(type(d) is not str or not re.fullmatch("[0-9a-f]{64}", d) for d in digests)):
        raise ValueError("All24 ordered public RGB SHA values required")
    return dict(schema=cohort.schema, images=[dict(file=cohort.frame_name(i), sha256=d, width=cohort.width, height=cohort.height)
        for i, d in enumerate(digests)])


def validate_png_header(path, cohort=COHORT):
    with Path(path).open("rb") as handle: data = handle.read(33)
    if (len(data) != 33 or data[:16] != b"\x89PNG\r\n\x1a\n\x00\x00\x00\x0dIHDR"
            or struct.unpack(">I", data[29:33])[0] != (zlib.crc32(data[12:29]) & 0xffffffff)
            or struct.unpack(">IIBBBBB", data[16:29]) != (cohort.width, cohort.height, 8, 2, 0, 0, 0)):
        raise ValueError("Original RGB8 PNG grid required")


def public_inputs(directory, cohort=COHORT):
    """Read exactly the complete RGB whitelist, never manufacturing truth."""
    directory = Path(directory)
    if (type(cohort) is not PublicCohort or not directory.is_dir() or directory.resolve() != directory.absolute()
            or any(p.is_symlink() for p in (directory, *directory.parents))):
        raise ValueError("Canonical public directory and explicit cohort required")
    path = directory / "manifest.json"; receipt = identity(path); manifest = json.loads(path.read_text())
    if (type(manifest) is not dict or set(manifest) != {"schema", "images"} or manifest["schema"] != cohort.schema
            or type(manifest["images"]) is not list or len(manifest["images"]) != cohort.count):
        raise ValueError("Exact24 RGB-only manifest required")
    records = []
    for index, row in enumerate(manifest["images"]):
        if (type(row) is not dict or set(row) != {"file", "sha256", "width", "height"} or row["file"] != cohort.frame_name(index)
                or type(row["sha256"]) is not str or not re.fullmatch("[0-9a-f]{64}", row["sha256"])
                or type(row["width"]) is not int or type(row["height"]) is not int
                or (row["width"], row["height"]) != (cohort.width, cohort.height)):
            raise ValueError("Ordered RGB-only records required")
        png = directory / row["file"]; actual = identity(png); validate_png_header(png, cohort)
        if actual["sha256"] != row["sha256"] or actual["bytes"] <= 33: raise ValueError("Original RGB bytes differ")
        group, frame = divmod(index, cohort.frames)
        records.append(row | dict(path=png, group_index=group, frame_index=frame))
    if {p.name for p in directory.iterdir()} != {"manifest.json", *[r["file"] for r in records]}:
        raise ValueError("Only24 original RGB files and manifest may be exposed")
    return records, receipt
