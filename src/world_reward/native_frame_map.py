"""Immutable native IDs versus array positions; metadata only, no I/O or fitting."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json

SCHEMA = "world-reward-native-frame-map-v1"
MAX_ID = 2**63 - 1


def _integer(value, name):
    if type(value) is not int or not 0 <= value <= MAX_ID:
        raise ValueError(name + ": nonnegative native int64-range Python int required")
    return value


@dataclass(frozen=True, slots=True)
class NativeFrameMap:
    sequence_id: str
    source_frame_ids: tuple[int, ...]

    def __post_init__(self):
        if (type(self.sequence_id) is not str or not self.sequence_id
                or self.sequence_id != self.sequence_id.strip() or any(ord(c) < 32 for c in self.sequence_id)):
            raise ValueError("Explicit nonempty sequence ID required")
        ids = self.source_frame_ids
        if type(ids) is not tuple or not ids:
            raise ValueError("Nonempty immutable tuple of original source IDs required")
        for i, value in enumerate(ids):
            _integer(value, "source_frame_id")
            if i and value != ids[i - 1] + 1:
                raise ValueError("Source IDs must be unique ascending contiguous, stride1")

    @property
    def frame_positions(self) -> tuple[int, ...]:
        return tuple(range(len(self.source_frame_ids)))

    def source_id(self, frame_position: int) -> int:
        _integer(frame_position, "frame_position")
        if frame_position >= len(self.source_frame_ids):
            raise ValueError("Frame position outside frozen sequence")
        return self.source_frame_ids[frame_position]

    def position(self, source_frame_id: int) -> int:
        _integer(source_frame_id, "source_frame_id")
        value = source_frame_id - self.source_frame_ids[0]
        if not 0 <= value < len(self.source_frame_ids):
            raise ValueError("Source frame ID outside frozen sequence")
        return value

    def validate_rows(self, rows) -> None:
        if type(rows) is not list or len(rows) != len(self.source_frame_ids):
            raise ValueError("Exact complete ordered frame-map rows required")
        for i, row in enumerate(rows):
            if (type(row) is not dict or set(row) != {"frame_position", "source_frame_id"}
                    or type(row["frame_position"]) is not int or type(row["source_frame_id"]) is not int
                    or row["frame_position"] != i or row["source_frame_id"] != self.source_frame_ids[i]):
                raise ValueError("Array position and original source ID correspondence differs")

    def to_dict(self) -> dict:
        return {"schema": SCHEMA, "sequence_id": self.sequence_id,
                "frames": [{"frame_position": i, "source_frame_id": value}
                           for i, value in enumerate(self.source_frame_ids)]}

    @classmethod
    def from_dict(cls, value: dict) -> NativeFrameMap:
        if (type(value) is not dict or set(value) != {"schema", "sequence_id", "frames"}
                or type(value["schema"]) is not str or value["schema"] != SCHEMA
                or type(value["frames"]) is not list or not value["frames"]):
            raise ValueError("Exact native-frame-map receipt required")
        rows = value["frames"]
        if any(type(row) is not dict or set(row) != {"frame_position", "source_frame_id"} for row in rows):
            raise ValueError("Explicit original-ID rows required")
        result = cls(value["sequence_id"], tuple(row["source_frame_id"] for row in rows))
        result.validate_rows(rows)
        return result

    def to_json_bytes(self) -> bytes:
        return (json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"),
                           ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8")

    def receipt(self) -> dict:
        data = self.to_json_bytes()
        return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}

    def validate_receipt(self, data: bytes, pin: dict) -> None:
        if (type(data) is not bytes or type(pin) is not dict or set(pin) != {"bytes", "sha256"}
                or type(pin["bytes"]) is not int or type(pin["sha256"]) is not str
                or pin != self.receipt() or data != self.to_json_bytes()):
            raise ValueError("Frozen canonical frame-map bytes/hash/count differ")

    def validate_attachment(self, frame_positions, source_frame_ids) -> None:
        """Require both immutable index tuples before joining predictions to private records."""
        if (type(frame_positions) is not tuple or type(source_frame_ids) is not tuple
                or any(type(x) is not int for x in (*frame_positions, *source_frame_ids))
                or frame_positions != self.frame_positions or source_frame_ids != self.source_frame_ids):
            raise ValueError("Complete position/source-ID attachment differs")
