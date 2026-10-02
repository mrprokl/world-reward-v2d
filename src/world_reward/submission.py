"""Leakage-safe Track 1 assembly and exact schema roundtrip checks.

Only sample row IDs define the layout. Sample prediction values and held-out
references are never read. This module packs already-converted native MHR
parameters; it does NOT reinterpret CARI4D's PCA/continuous parameters as MHR.
Schema roundtrip verification is distinct from an MHR model geometry forward
check, reconstruction accuracy, and official scoring.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import tempfile

import numpy as np

from world_reward.contracts import Reconstruction, require_video_only_provenance


# Audited official kit: mhr_submission.array_order(), EPISODE_FRAME and
# PARAMETER_COUNTS. Tests compare these constants with the local official kit.
ARRAY_ORDER = (
    "mhr_pose", "mhr_scales", "mhr_shape", "object_rotation",
    "object_translation", "object_scale", "object_mesh_vertices", "object_mesh_faces",
)
EPISODE_FRAME = 999999
PARAMETER_COUNTS = {"mhr_pose": 136, "mhr_scales": 68, "mhr_shape": 45}
MHR_PARAMETER_FORMAT = "mhr_model_params_136_68_45"
MAX_MESH_ROWS = 4096
_FRAME_ROWS = {0: 46, 3: 3, 4: 1}
_EPISODE_ROWS = {1: 23, 2: 15, 5: 1}
_ROW_ID = re.compile(r"t1_([0-9]{6})_([0-9]{6})_([0-7])_([0-9]{6})")
_COMMIT = re.compile(r"https://github\.com/[\w.-]+/[\w.-]+/commit/[0-9a-fA-F]{40}")
_PARQUET_COLUMNS = ("row_id", "x", "y", "z", "code_commit_url")


def _numeric(value, name: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "fiu":
        raise ValueError(f"{name} must contain real numeric values")
    array = array.astype(np.float64, copy=False)
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name}: expected {shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def _mesh(vertices, faces) -> tuple[np.ndarray, np.ndarray]:
    vertices = _numeric(vertices, "object_vertices")
    faces = _numeric(faces, "object_faces")
    if vertices.ndim != 2 or vertices.shape[1:] != (3,) or not 3 <= len(vertices) <= MAX_MESH_ROWS:
        raise ValueError("object_vertices must have shape [V,3], 3 <= V <= 4096")
    if faces.ndim != 2 or faces.shape[1:] != (3,) or not 1 <= len(faces) <= MAX_MESH_ROWS:
        raise ValueError("object_faces must have shape [F,3], 1 <= F <= 4096")
    if not np.array_equal(faces, np.rint(faces)):
        raise ValueError("object_faces must contain exact integer vertex indices")
    if np.any(faces < 0) or np.any(faces >= len(vertices)):
        raise ValueError("object_faces index outside submitted vertices")
    faces = faces.astype(np.int64)
    try:
        with np.errstate(over="raise", invalid="raise"):
            a, b, c = (vertices[faces[:, index]] for index in range(3))
            area = np.linalg.norm(np.cross(b - a, c - a), axis=1).sum()
    except FloatingPointError as error:
        raise ValueError("object mesh area overflowed; check units") from error
    if not np.isfinite(area) or area <= 0:
        raise ValueError("object mesh must have a nonzero surface area")
    return vertices, faces


def _zero_expression(value) -> None:
    expression = _numeric(value, "human_expression", (72,))
    if np.any(expression != 0):
        raise ValueError("Track 1 uses one zero facial-expression vector, not per-frame expressions")


@dataclass(frozen=True)
class EpisodeLayout:
    frame_indices: np.ndarray
    vertex_rows: int
    face_rows: int


@dataclass(frozen=True)
class Track1Layout:
    """Validated original row order, full ordinal blocks and scored video frames."""

    row_ids: tuple[str, ...]
    keys: np.ndarray  # columns: episode, original video frame, array, point
    episodes: Mapping[int, EpisodeLayout]

    @classmethod
    def from_row_ids(cls, row_ids: Sequence[str]) -> Track1Layout:
        ids = tuple(row_ids)
        if any(not isinstance(row, str) for row in ids):
            raise ValueError("sample row IDs must be strings, not prediction values")
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("sample row IDs must be nonempty and unique")
        parsed = []
        for row in ids:
            match = _ROW_ID.fullmatch(row) if isinstance(row, str) else None
            if match is None:
                raise ValueError(f"not a canonical Track 1 row ID: {row!r}")
            parsed.append(tuple(int(part) for part in match.groups()))
        keys = np.asarray(parsed, dtype=np.int64)
        layouts = {}
        for episode in np.unique(keys[:, 0]):
            block = keys[keys[:, 0] == episode]
            present = set(block[:, 2].tolist())
            if present != set(range(len(ARRAY_ORDER))):
                raise ValueError(f"episode {episode}: all eight arrays are required")
            frames = np.unique(block[block[:, 2] == 0, 1])
            if not len(frames) or frames[-1] == EPISODE_FRAME:
                raise ValueError(f"episode {episode}: pose must use original video frame indices")
            # Match the actual scorer: second differences never cross frame gaps.
            edges = np.r_[0, np.flatnonzero(np.diff(frames) != 1) + 1, len(frames)]
            if np.any(np.diff(edges) < 3):
                raise ValueError(f"episode {episode}: each scored stretch needs at least three frames")
            budgets = {}
            for array in range(len(ARRAY_ORDER)):
                rows = block[block[:, 2] == array]
                scope_frames = frames if array in _FRAME_ROWS else np.array([EPISODE_FRAME])
                if not np.array_equal(np.unique(rows[:, 1]), scope_frames):
                    raise ValueError(f"episode {episode}: {ARRAY_ORDER[array]} has wrong frame scope")
                point_count = _FRAME_ROWS.get(array, _EPISODE_ROWS.get(array, len(rows)))
                if not np.array_equal(np.unique(rows[:, 3]), np.arange(point_count)):
                    raise ValueError(f"episode {episode}: {ARRAY_ORDER[array]} point indices have gaps")
                if len(rows) != len(scope_frames) * point_count:
                    raise ValueError(f"episode {episode}: incomplete {ARRAY_ORDER[array]} block")
                if array in (6, 7):
                    if not 1 <= point_count <= MAX_MESH_ROWS:
                        raise ValueError("mesh row budget must be in [1,4096]")
                    budgets[array] = point_count
            if budgets[6] < 3:
                raise ValueError("mesh vertex row budget must be at least three")
            frames.setflags(write=False)
            layouts[int(episode)] = EpisodeLayout(frames, budgets[6], budgets[7])
        if len({(item.vertex_rows, item.face_rows) for item in layouts.values()}) != 1:
            raise ValueError("official mesh row budgets must be consistent across episodes")
        keys.setflags(write=False)
        return cls(ids, keys, layouts)


@dataclass(frozen=True)
class Track1Episode:
    """One complete predicted video trajectory plus a fixed local object mesh.

    ``reconstruction.pose`` already contains native MHR parameters 0:136, not
    SAM/CARI4D continuous body controls. Shape [45] and scales [68] are shared
    over the full clip; facial expression [72] is explicitly zero. The object
    uses ``scene_point = scale * R @ local_mesh_point + translation`` with
    translations and posed geometry in the same metric frame as the human.
    Arrays are indexed by original frame number, INCLUDING unscored frames.
    No frame flipping, root-translation gain, PCA expansion or fitting is done
    here. Those conversions require the actual upstream model and a separate
    model-forward geometry check against the Apache-licensed reference MHR.
    """

    reconstruction: Reconstruction
    object_vertices: np.ndarray
    object_faces: np.ndarray
    human_expression: np.ndarray
    provenance: Mapping
    total_video_frames: int
    mhr_parameter_format: str = MHR_PARAMETER_FORMAT

    def validate(self) -> None:
        if self.mhr_parameter_format != MHR_PARAMETER_FORMAT:
            raise ValueError("native MHR pose136/scales68/shape45 required; convert upstream output first")
        require_video_only_provenance(dict(self.provenance))
        if self.provenance.get("hand_labeled_test") is not False:
            raise ValueError("explicit hand_labeled_test=False is required; hand-labeled test predictions are forbidden")
        pose = _numeric(self.reconstruction.pose, "pose")
        if pose.ndim != 2 or not len(pose):
            raise ValueError("pose must contain a nonempty full video trajectory")
        if (
            isinstance(self.total_video_frames, (bool, np.bool_))
            or not isinstance(self.total_video_frames, (int, np.integer))
            or self.total_video_frames <= 0
        ):
            raise ValueError("total_video_frames must be a positive integer decoded from the input video")
        for name in ("scales", "shape", "object_rotation", "object_translation", "object_scale"):
            _numeric(getattr(self.reconstruction, name), name)
        if np.asarray(self.reconstruction.object_scale).shape != ():
            raise ValueError("object_scale must be one scalar per clip")
        self.reconstruction.validate(self.total_video_frames)
        _zero_expression(self.human_expression)
        _mesh(self.object_vertices, self.object_faces)


@dataclass(frozen=True)
class ConverterParameters:
    pose: np.ndarray
    scales: np.ndarray
    shape: np.ndarray
    expression: np.ndarray
    report: Mapping


def parameters_from_official_converter(
    result: Mapping, *, max_mean_vertex_error_mm: float
) -> ConverterParameters:
    """Validate real output from kit mesh_to_mhr_params.convert (or its NPZ).

    This performs **no conversion**. Inference must first decode upstream MHR
    meshes [T,18439,3] in the kit's metric frame and use the actual reference
    converter. Unlike its nearest-valid-frame fallback, this boundary refuses
    invalid frames. Choose the mean-vertex residual gate on non-challenge
    validation; passing it verifies the reported fitting contract, not an
    independent forward pass nor accuracy versus held-out ground truth.
    """
    needed = {"pose", "scales", "shape", "valid_input", "per_frame_vertex_error_mm", "report"}
    if not needed.issubset(result):
        raise ValueError(f"official converter result lacks {sorted(needed - set(result))}")
    limit = _numeric(max_mean_vertex_error_mm, "max_mean_vertex_error_mm", ())
    if limit < 0:
        raise ValueError("max_mean_vertex_error_mm must be nonnegative")
    pose = _numeric(result["pose"], "pose")
    if pose.ndim != 2 or pose.shape[1:] != (136,) or not len(pose):
        raise ValueError("official converter pose must have shape [T,136]")
    scales = _numeric(result["scales"], "scales", (68,))
    shape = _numeric(result["shape"], "shape", (45,))
    valid = np.asarray(result["valid_input"])
    if valid.dtype != np.bool_ or valid.shape != (len(pose),) or not valid.all():
        raise ValueError("all original frames must have valid converter input; no nearest-frame fallback")
    errors = _numeric(result["per_frame_vertex_error_mm"], "per_frame_vertex_error_mm", (len(pose),))
    if np.any(errors < 0) or np.any(errors > limit):
        raise ValueError("converter per-frame mean vertex residual exceeds the declared gate")
    report = result["report"]
    if isinstance(report, np.ndarray) and report.shape == ():
        report = report.item()
    if isinstance(report, str):
        try:
            report = json.loads(report)
        except json.JSONDecodeError as error:
            raise ValueError("converter report is not valid JSON") from error
    if not isinstance(report, Mapping):
        raise ValueError("converter report must be an object")
    if (
        report.get("frames") != len(pose)
        or report.get("fitted_frames") != len(pose)
        or report.get("invalid_input_frames") != []
    ):
        raise ValueError("converter report does not certify a fit for every original frame")
    residual = report.get("vertex_error_mm")
    if not isinstance(residual, Mapping):
        raise ValueError("converter report lacks vertex-error summary")
    for name in ("mean", "worst_frame_mean", "max", "p99"):
        value = _numeric(residual.get(name), f"report.vertex_error_mm.{name}", ())
        if value < 0:
            raise ValueError("reported vertex errors must be nonnegative")
    if not np.isclose(residual["mean"], errors.mean(), rtol=1e-4, atol=1e-6):
        raise ValueError("converter mean report disagrees with per-frame residuals")
    if not np.isclose(residual["worst_frame_mean"], errors.max(), rtol=1e-4, atol=1e-6):
        raise ValueError("converter worst-frame report disagrees with per-frame residuals")
    return ConverterParameters(pose.copy(), scales.copy(), shape.copy(), np.zeros(72), dict(report))


def _pack_parameters(values: np.ndarray, count: int) -> np.ndarray:
    # Identical packing to official tools.pack_reconstruction.ms_pack.
    flat = np.zeros(3 * ((count + 2) // 3))
    flat[:count] = np.asarray(values).reshape(count)
    return flat.reshape(-1, 3)


def _padded_mesh(episode: Track1Episode, layout: EpisodeLayout) -> tuple[np.ndarray, np.ndarray]:
    vertices, faces = _mesh(episode.object_vertices, episode.object_faces)
    if len(vertices) > layout.vertex_rows or len(faces) > layout.face_rows:
        raise ValueError("mesh exceeds template budget; use official budget_mesh upstream, not truncation")
    # Official harmless padding: same first vertex, zero-area face (0,0,0).
    return (
        np.concatenate([vertices, np.repeat(vertices[:1], layout.vertex_rows - len(vertices), axis=0)]),
        np.concatenate([faces, np.zeros((layout.face_rows - len(faces), 3), dtype=np.int64)]),
    )


@dataclass(frozen=True)
class SubmissionRows:
    row_ids: tuple[str, ...]
    values: np.ndarray
    code_commit_url: str


def _commit(commit: str) -> None:
    if not isinstance(commit, str) or _COMMIT.fullmatch(commit) is None:
        raise ValueError("code commit must be a pinned https://github.com/owner/repo/commit/<40-hex SHA>")


def assemble_submission(
    layout: Track1Layout, episodes: Mapping[int, Track1Episode], *, code_commit_url: str
) -> SubmissionRows:
    """Slice full trajectories by exact original row keys; preserve row order.

    Supply exactly the episodes declared by the row-ID template. Meshes must
    already fit its budget (at most 4096 rows); only official zero-area padding
    is added. No template values enter this function and no pose/scales/mesh
    are silently repaired, re-aligned, truncated or interpolated.
    """
    _commit(code_commit_url)
    if set(episodes) != set(layout.episodes):
        raise ValueError("predicted episodes must exactly match the official row-ID template")
    values = np.empty((len(layout.row_ids), 3), dtype=np.float64)
    for index, spec in layout.episodes.items():
        episode = episodes[index]
        episode.validate()
        rec = episode.reconstruction
        if spec.frame_indices[-1] >= len(rec.pose):
            raise ValueError(f"episode {index}: full input trajectory does not reach last scored video frame")
        vertices, faces = _padded_mesh(episode, spec)
        shared = {
            1: _pack_parameters(rec.scales, 68),
            2: _pack_parameters(rec.shape, 45),
            5: np.array([[rec.object_scale, 0, 0]], dtype=float),
            6: vertices,
            7: faces,
        }
        locations = np.flatnonzero(layout.keys[:, 0] == index)
        keys = layout.keys[locations]
        for array in range(len(ARRAY_ORDER)):
            selected = keys[:, 2] == array
            points = keys[selected, 3]
            frames = keys[selected, 1]
            if array == 0:
                packed = np.zeros((len(rec.pose), 46, 3))
                packed.reshape(len(rec.pose), -1)[:, :136] = rec.pose
                block = packed[frames, points]
            elif array == 3:
                block = np.asarray(rec.object_rotation)[frames, points]
            elif array == 4:
                block = np.asarray(rec.object_translation)[frames]
            else:
                block = shared[array][points]
            values[locations[selected]] = block
    return SubmissionRows(layout.row_ids, values, code_commit_url)


@dataclass(frozen=True)
class ScoredEpisode:
    """Decoded only at scored ORIGINAL frame indices, not a full-video object."""

    frame_indices: np.ndarray
    reconstruction: Reconstruction
    object_vertices: np.ndarray
    object_faces: np.ndarray


def decode_submission(rows: SubmissionRows, layout: Track1Layout) -> dict[int, ScoredEpisode]:
    """Strict inverse packing, without MHR evaluation or hidden references."""
    _commit(rows.code_commit_url)
    if tuple(rows.row_ids) != layout.row_ids:
        raise ValueError("submission row IDs/order must exactly match the official template")
    values = _numeric(rows.values, "submission values", (len(layout.row_ids), 3))
    decoded = {}
    for episode, spec in layout.episodes.items():
        indices = np.flatnonzero(layout.keys[:, 0] == episode)
        block = layout.keys[indices]
        arrays = {}
        for array in range(len(ARRAY_ORDER)):
            selected = np.flatnonzero(block[:, 2] == array)
            order = np.lexsort((block[selected, 3], block[selected, 1]))
            value = values[indices[selected[order]]]
            arrays[array] = value.reshape(len(spec.frame_indices), -1, 3) if array in _FRAME_ROWS else value
        parameters = {}
        for array, name in ((0, "mhr_pose"), (1, "mhr_scales"), (2, "mhr_shape")):
            packed = arrays[array]
            flat = packed.reshape(*packed.shape[:-2], -1)
            count = PARAMETER_COUNTS[name]
            if np.any(flat[..., count:] != 0):
                raise ValueError(f"{name}: nonzero unused parameter tail")
            parameters[array] = flat[..., :count].copy()
        scale, y, z = arrays[5][0]
        if y != 0 or z != 0:
            raise ValueError("object_scale must carry exactly (scale,0,0)")
        rec = Reconstruction(
            pose=parameters[0], scales=parameters[1], shape=parameters[2],
            object_rotation=arrays[3].copy(), object_translation=arrays[4][:, 0].copy(),
            object_scale=float(scale),
        )
        rec.validate(len(spec.frame_indices))
        vertices, faces = _mesh(arrays[6], arrays[7])
        decoded[episode] = ScoredEpisode(spec.frame_indices.copy(), rec, vertices.copy(), faces.copy())
    return decoded


def verify_roundtrip(
    rows: SubmissionRows, layout: Track1Layout, episodes: Mapping[int, Track1Episode]
) -> dict:
    """Prove exact schema preservation versus predictions, not MHR geometry.

    All full input frames are validated even if not scored. Decoded parameters,
    fixed identity, selected poses and padded geometry must be bit-exact in
    float64 against source arrays. An independent reference-model forward pass
    remains REQUIRED upstream before a native-MHR conversion is trusted.
    """
    decoded = decode_submission(rows, layout)
    if set(episodes) != set(decoded):
        raise ValueError("roundtrip source episodes disagree with the template")
    for index, recovered in decoded.items():
        episode = episodes[index]
        episode.validate()
        expected = episode.reconstruction
        frames = recovered.frame_indices
        if frames[-1] >= len(expected.pose):
            raise ValueError("roundtrip source trajectory does not cover scored original frames")
        for name in ("pose", "object_rotation", "object_translation", "scales", "shape", "object_scale"):
            value = np.asarray(getattr(expected, name))
            if name in ("pose", "object_rotation", "object_translation"):
                value = value[frames]
            if not np.array_equal(value, np.asarray(getattr(recovered.reconstruction, name))):
                raise ValueError(f"episode {index}: {name} changed during schema roundtrip")
        vertices, faces = _padded_mesh(episode, layout.episodes[index])
        if not np.array_equal(vertices, recovered.object_vertices) or not np.array_equal(faces, recovered.object_faces):
            raise ValueError(f"episode {index}: object geometry changed during schema roundtrip")
    return {
        "schema_roundtrip_verified": True,
        "mhr_geometry_forward_verified": False,
        "episodes": len(decoded),
        "rows": len(rows.row_ids),
        "scored_frames": sum(len(spec.frame_indices) for spec in layout.episodes.values()),
    }


def _arrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("Parquet I/O requires optional pyarrow; install the project's data extra") from error
    return pa, pq


def read_template(path: str | Path) -> Track1Layout:
    """Read ONLY row_id from Parquet; no sample values or reference columns."""
    _, pq = _arrow()
    table = pq.read_table(path, columns=["row_id"])
    return Track1Layout.from_row_ids(table["row_id"].to_pylist())


def read_submission(path: str | Path, layout: Track1Layout) -> SubmissionRows:
    _, pq = _arrow()
    schema = pq.read_schema(path)
    if tuple(schema.names) != _PARQUET_COLUMNS:
        raise ValueError("submission must have exactly row_id,x,y,z,code_commit_url columns")
    table = pq.read_table(path, columns=list(_PARQUET_COLUMNS))
    commits = table["code_commit_url"].to_pylist()
    if not commits or len(set(commits)) != 1:
        raise ValueError("one pinned commit URL is required throughout the submission")
    rows = SubmissionRows(
        tuple(table["row_id"].to_pylist()),
        np.column_stack([table[name].to_numpy(zero_copy_only=False) for name in ("x", "y", "z")]),
        commits[0],
    )
    decode_submission(rows, layout)
    return rows


def write_submission(
    path: str | Path, rows: SubmissionRows, layout: Track1Layout, episodes: Mapping[int, Track1Episode]
) -> dict:
    """Write a frozen Parquet after schema checks, verify it again, never overwrite.

    The artifact is valid for all five Track 1 competitions. This function does
    not submit it, consume quota, accept rules, or claim a reconstruction score.
    """
    report = verify_roundtrip(rows, layout, episodes)
    pa, pq = _arrow()
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to replace frozen submission: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temporary = Path(temporary)
    try:
        table = pa.table({
            "row_id": rows.row_ids,
            "x": rows.values[:, 0], "y": rows.values[:, 1], "z": rows.values[:, 2],
            "code_commit_url": [rows.code_commit_url] * len(rows.row_ids),
        })
        pq.write_table(table, temporary, compression="zstd")
        verify_roundtrip(read_submission(temporary, layout), layout, episodes)
        # Same-directory hard link publishes atomically and fails if another
        # process created the destination; unlike replace it cannot overwrite.
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return report
