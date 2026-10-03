"""One-episode unmodified official Track1 packer smoke, not a submission.

The original sample is read through a row_id-only Arrow projection. The official
reader receives a new row_id-only CSV, never the original sample. Full frozen
predictions and aligned geometry are unchanged; all temporary payloads are
removed after source/schema/geometry checks. No models, scoring or upload.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import stat
import sys
import tempfile
import time
from types import SimpleNamespace

ROOT = Path("/srv/scenesmith/world-reward")
IMAGE = "sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7"
STAGE = "world_reward_one_episode_unmodified_official_track1_packer_smoke"
BUDGET = 300
PUBLIC_REPOSITORY = "https://github.com/mrprokl/world-reward-v2d"
SAMPLE_SHA = "1db56c5ec7267cdb7a68d42928f85c092c6b8e9bfbc0dc98a2b020d7a48bc621"
SAMPLE_BYTES = 3495851
SAMPLE_RELATIVE = "vendor/v2d_submission_kit/data/track_1_sample_submission.parquet"
OFFICIAL_SOURCES = {
    "tools/pack_reconstruction.py": {"bytes": 11149, "sha256": "d197a2110c5f52dd08ac65c27aa3d519799a166ab9a4ea06b3f6ceb163f0d6b1"},
    "v2dlb/report_schema.py": {"bytes": 5441, "sha256": "2a14b19dcef1004c8bcce73bb69cb60605e2a54c783df8b52575fc861631222a"},
    "v2dlb/mesh_budget.py": {"bytes": 2031, "sha256": "42ab8ab35f37b806fb1465eadd96abe43eaac04575da47a4855d08eefe6167b0"},
    "v2dlb/mhr_submission.py": {"bytes": 24344, "sha256": "06fbd58d07bae1c583a92598f879771a4805ccea3e8346605975bcf5007b1fa3"},
    "v2dlb/mesh_common.py": {"bytes": 2175, "sha256": "aaeac13286c839a7d7e271888613875c934e46de094b2ac928ff2ff01c5a1f06"},
    "v2dlb/mhr_metrics.py": {"bytes": 25647, "sha256": "73077b65b5c3e0204307feef784317aab8c733d7462b6b8e6f4f397f7b91d2e0"},
}
NPZ_KEYS = ("pose", "scales", "shape", "object_rotation", "object_translation", "object_scale")


def parser():
    class Once(argparse.Action):
        def __call__(self, parser, namespace, value, option_string=None):
            if getattr(namespace, self.dest, None) is not None:
                parser.error("Exactly one explicit episode required")
            setattr(namespace, self.dest, value)
    result = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    result.add_argument("--episode", type=int, choices=range(30), required=True, action=Once)
    return result


def output_relative(episode):
    if type(episode) is not int or not 0 <= episode < 30:
        raise ValueError("Explicit Track1 integer episode0..29 required")
    return f"outputs/episode_{episode:06d}/official_track1_pack_smoke_v1"


def identity(path, *, immutable=True):
    path = Path(path)
    if (not path.is_absolute() or path.resolve() != path
            or any(parent.is_symlink() for parent in (path, *path.parents))
            or not stat.S_ISREG(path.lstat().st_mode)
            or immutable and path.stat().st_mode & 0o222):
        raise ValueError("Canonical regular immutable source/pin required")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return dict(sha256=digest.hexdigest(), bytes=path.stat().st_size)


def strict_json(path):
    def unique(rows):
        result = {}
        for key, value in rows:
            if key in result:
                raise ValueError("Duplicate pin/report JSON keys forbidden")
            result[key] = value
        return result
    def invalid(_):
        raise ValueError("Nonfinite JSON forbidden")
    return json.loads(Path(path).read_text(), object_pairs_hook=unique, parse_constant=invalid)


def code_commit_url(revision):
    if type(revision) is not str or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Exact actual immutable producer revision required")
    return PUBLIC_REPOSITORY + "/commit/" + revision


def source_helpers(code):
    names = ("infra/official_track1_pack_gate.py", "infra/run_official_track1_pack_gate.sh",
             "infra/cari_shared_episode_loader.py", "src/world_reward/submission.py")
    return {name: identity(code / name) for name in names}


def official_sources(root):
    kit = root / "vendor/v2d_submission_kit"
    # This audited kit uses a namespace package, not a fabricated initializer.
    if (kit / "v2dlb/__init__.py").exists() or (kit / "v2dlb/__init__.py").is_symlink():
        raise ValueError("Pinned v2dlb namespace package must not gain an initializer")
    observed = {name: identity(kit / name, immutable=False) for name in OFFICIAL_SOURCES}
    if observed != OFFICIAL_SOURCES:
        raise ValueError("Unmodified official packer/import source hash or size differs")
    return observed


def load_official_packer(root):
    """Import only the verified unmodified original file, with no hooks/patches."""
    official_sources(root)
    kit = root / "vendor/v2d_submission_kit"
    if any(name == "v2dlb" or name.startswith("v2dlb.") for name in sys.modules):
        raise ValueError("Fresh official namespace/import process required")
    sys.path.insert(0, str(kit))
    path = kit / "tools/pack_reconstruction.py"
    name = "_world_reward_original_official_track1_packer"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import inspect
    signature = inspect.signature(module.pack_track1)
    parameters = list(signature.parameters.values())
    if (len(parameters) != 1 or parameters[0].name != "args"
            or parameters[0].kind != inspect.Parameter.POSITIONAL_OR_KEYWORD
            or parameters[0].default is not inspect.Parameter.empty
            or signature.return_annotation not in (None, "None")
            or Path(module.pack_track1.__code__.co_filename).resolve() != path
            or module.pack_track1.__module__ != name):
        raise ValueError("Actual unchanged official pack_track1(args) -> None ABI required")
    verify_official_imports(root)
    return module.pack_track1


def verify_official_imports(root, *, complete=False):
    """Bind lazily imported official source after the real packer call as well."""
    kit = root / "vendor/v2d_submission_kit"
    expected = {name for name in OFFICIAL_SOURCES if name.startswith("v2dlb/")}
    imported = set()
    for name, module in tuple(sys.modules.items()):
        if not name.startswith("v2dlb."):
            continue
        filename = getattr(module, "__file__", None)
        if filename is None:
            raise ValueError("Actual official source module file required")
        actual = Path(filename)
        if actual.resolve() != actual or actual.parent != kit / "v2dlb":
            raise ValueError("Official source module loaded outside exact mounted kit")
        relative = str(actual.relative_to(kit))
        if relative not in expected or identity(actual, immutable=False) != OFFICIAL_SOURCES[relative]:
            raise ValueError("Official source import SHA/path differs from the pinned closure")
        imported.add(relative)
    if complete and imported != expected:
        raise ValueError("Every original lazy official import must be verified after packing")
    if any(name in sys.modules for name in ("torch", "joblib")):
        raise ValueError("Official packer must remain CPU schema-only, no model imports")
    return sorted(imported)


def selected_layout(layout, episode, *, expected_rows=740780, expected_frames=9877, expected_episodes=30):
    """Select original episode IDs in original order, never prediction columns."""
    from world_reward.submission import Track1Layout
    if (len(layout.row_ids) != expected_rows or set(layout.episodes) != set(range(expected_episodes))
            or sum(len(value.frame_indices) for value in layout.episodes.values()) != expected_frames):
        raise ValueError("Exact original sample row-ID dataset envelope required")
    if episode not in layout.episodes:
        raise ValueError("Selected episode missing from original sample row IDs")
    ids = tuple(row for row, key in zip(layout.row_ids, layout.keys) if int(key[0]) == episode)
    selected = Track1Layout.from_row_ids(ids)
    if (set(selected.episodes) != {episode}
            or selected.episodes[episode].vertex_rows != 4096
            or selected.episodes[episode].face_rows != 4096):
        raise ValueError("Exact original episode4096 mesh-row layout required")
    return selected


def row_id_digest(row_ids):
    return hashlib.sha256(("\n".join(row_ids) + "\n").encode("utf-8")).hexdigest()


def array_fingerprint(episode):
    import numpy as np
    result = {}
    for key in (*NPZ_KEYS, "object_vertices", "object_faces", "human_expression"):
        value = np.asarray(getattr(episode.reconstruction if key in NPZ_KEYS else episode, key))
        result[key] = dict(shape=list(value.shape), dtype=value.dtype.str,
                          sha256=hashlib.sha256(value.tobytes(order="C")).hexdigest())
    return result


def oriented_triangles(vertices, faces):
    """Exact cyclic-oriented triangle multiset; ignore only index-degenerate padding."""
    import numpy as np
    vertices, faces = np.asarray(vertices), np.asarray(faces)
    if (vertices.ndim != 2 or vertices.shape[1:] != (3,) or vertices.dtype.kind != "f"
            or not np.isfinite(vertices).all() or faces.ndim != 2 or faces.shape[1:] != (3,)
            or faces.dtype.kind not in "iu" or not len(faces) or np.min(faces) < 0 or np.max(faces) >= len(vertices)):
        raise ValueError("Finite exact triangle arrays required")
    points = vertices.astype(np.float64)[faces]
    cross = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
    if not np.isfinite(cross).all():
        raise ValueError("Triangle arithmetic must remain finite")
    # Exact zero only: no angle/area/score tolerance can shave small surfaces.
    # Repeated indices, repeated coordinates and exactly collinear triangles
    # have no surface; all original nonzero triangles remain in the multiset.
    points = points[np.any(cross != 0, axis=1)]
    if not len(points):
        raise ValueError("At least one original nondegenerate triangle is required")
    choices = np.stack((points, np.roll(points, 1, axis=1), np.roll(points, 2, axis=1)), axis=1).reshape(-1, 3, 9)
    order = np.lexsort(tuple(choices[:, :, j] for j in range(8, -1, -1)), axis=1)
    rows = choices[np.arange(len(points)), order[:, 0]]
    return rows[np.lexsort(tuple(rows[:, j] for j in range(8, -1, -1)))]


def verify_output(rows, layout, episode, commit):
    """Exact controls and oriented surface identity, not an independent model replay."""
    import numpy as np
    from world_reward.submission import decode_submission
    if rows.code_commit_url != commit:
        raise ValueError("Exact published producing code URL required")
    decoded = decode_submission(rows, layout)
    if len(decoded) != 1:
        raise ValueError("Smoke must contain exactly one original episode")
    index, recovered = next(iter(decoded.items()))
    frames = layout.episodes[index].frame_indices
    if episode.total_video_frames <= int(frames[-1]) or not np.array_equal(recovered.frame_indices, frames):
        raise ValueError("Packer must index the complete source by original scored frames")
    for name in NPZ_KEYS:
        expected = np.asarray(getattr(episode.reconstruction, name))
        if name in ("pose", "object_rotation", "object_translation"):
            expected = expected[frames]
        if not np.array_equal(expected, np.asarray(getattr(recovered.reconstruction, name))):
            raise ValueError("Original controls/shared identity changed during official packing: " + name)
    vertices, faces = recovered.object_vertices, recovered.object_faces
    if vertices.shape != (4096, 3) or faces.shape != (4096, 3):
        raise ValueError("Actual official budget must pad exactly4096 vertices and faces")
    zero = (faces == 0).all(axis=1)
    active = faces[~zero]
    if not len(active):
        raise ValueError("Original official mesh must retain nonpadding surfaces")
    source = oriented_triangles(episode.object_vertices, episode.object_faces)
    packed = oriented_triangles(vertices, faces)
    if not np.array_equal(source, packed):
        raise ValueError("Unmodified official budget changed oriented triangle geometry; no second simplification")
    unused = np.setdiff1d(np.arange(len(vertices)), np.unique(active))
    if len(unused) and not np.array_equal(vertices[unused], np.repeat(vertices[:1], len(unused), axis=0)):
        raise ValueError("Official unused vertices must be exact first-vertex padding")
    return dict(episodes=1, rows=len(layout.row_ids), scored_frames=len(frames),
                original_frame_indices=frames.tolist(), full_source_frames=episode.total_video_frames,
                exact_controls_and_shared_identity=True, oriented_triangles_exact=True,
                active_faces=len(packed), source_active_faces=len(source),
                source_exact_degenerate_faces=len(episode.object_faces) - len(source),
                packed_exact_degenerate_faces=len(faces) - len(packed),
                padding_faces=int(zero.sum()), padding_vertices=len(unused),
                official_budget_vertices=4096, official_budget_faces=4096,
                schema_roundtrip_verified=True, numerical_geometry_independently_reverified=False)


def pack_once(root, scratch, loaded, layout, report, persist, *, commit, packer=None):
    """One unchanged official call; only scratch row IDs and six full arrays passed."""
    import numpy as np
    from world_reward.submission import read_submission
    episode = loaded.episode
    index = next(iter(layout.episodes))
    episode.validate()
    before = array_fingerprint(episode)
    episodes = scratch / "episodes"
    episodes.mkdir()
    sample = scratch / "row_ids_only.csv"
    with sample.open("x", newline="") as stream:
        writer = csv.writer(stream); writer.writerow(["row_id"])
        writer.writerows((row,) for row in layout.row_ids)
    payload = episodes / f"episode_{index:06d}.npz"
    values = {key: np.asarray(getattr(episode.reconstruction, key)).copy() for key in NPZ_KEYS}
    with payload.open("xb") as stream:
        np.savez(stream, **values)
    with np.load(payload, allow_pickle=False) as saved:
        if len(saved.files) != len(NPZ_KEYS) or set(saved.files) != set(NPZ_KEYS) or any(
                saved[key].dtype != values[key].dtype or saved[key].shape != values[key].shape
                or saved[key].tobytes() != values[key].tobytes() for key in NPZ_KEYS):
            raise ValueError("Six complete original parameter arrays must survive scratch serialization")
    mesh = root / f"outputs/episode_{index:06d}/cari_shared_export_v1/object_aligned.glb"
    target_mesh = episodes / f"episode_{index:06d}_object.glb"
    original_mesh = identity(mesh)
    shutil.copyfile(mesh, target_mesh)
    if identity(target_mesh, immutable=False) != original_mesh:
        raise ValueError("Scratch mesh must copy byte-exact original aligned geometry")
    # Scratch inputs become readonly before the original packer opens them.
    for path in (sample, payload, target_mesh):
        path.chmod(0o444)
    inputs = {path: identity(path) for path in (sample, payload, target_mesh)}
    with sample.open(newline="") as stream:
        if next(csv.reader(stream)) != ["row_id"]:
            raise ValueError("Only row_id CSV columns may enter the original official reader")
    output = scratch / "one_episode_smoke.parquet"
    actual_official = packer is None
    if actual_official:
        packer = load_official_packer(root)
    report.update(phase="unmodified_official_packer", official_packer_attempts=1,
                  scratch_payload_full_frames=episode.total_video_frames,
                  scratch_mesh_sha256=original_mesh["sha256"], scratch_parameter_keys=list(NPZ_KEYS))
    persist()
    packer(SimpleNamespace(sample=sample, episodes=episodes, commit=commit, out=output))
    report["official_packer_returns"] = 1
    if actual_official:
        report["official_imported_sources"] = verify_official_imports(root, complete=True)
    result = read_submission(output, layout)
    verification = verify_output(result, layout, episode, commit)
    if (array_fingerprint(episode) != before or identity(mesh) != original_mesh
            or any(identity(path) != value for path, value in inputs.items())):
        raise ValueError("Original trajectory/mesh or scratch inputs changed during official packing")
    report.update(official_packer_validated=1, packing_roundtrip=verification,
                  scratch_Parquet=identity(output, immutable=False), source_prediction_arrays_rehashed=True)


def run(root, out, code, episode, report, persist, *, consumer=None, packer=None, template_reader=None):
    from cari_shared_episode_loader import load_shared_track1_episode, validate_export_pins
    import cari_clip_inputs as public
    from world_reward.submission import read_template
    pins_path = code / f"configs/cari_clip_{episode:06d}_shared_export_pins.json"
    pin_id = identity(pins_path)
    pins = strict_json(pins_path)
    spec = public.PublicClipSpec(**pins["clip_spec"])
    if spec.episode_index != episode:
        raise ValueError("Explicit episode differs from original full export pins")
    validate_export_pins(spec, pins)
    own = source_helpers(code)
    official = official_sources(root)
    export_directory = root / f"outputs/episode_{episode:06d}/cari_shared_export_v1"
    export_files = {export_directory / name: value for name, value in pins["export_files"].items()}
    if any(identity(path) != value for path, value in export_files.items()):
        raise ValueError("Complete five frozen source exports must match pins before packing")
    sample_path = root / SAMPLE_RELATIVE
    sample_id = identity(sample_path, immutable=False)
    if sample_id != {"sha256": SAMPLE_SHA, "bytes": SAMPLE_BYTES}:
        raise ValueError("Exact pinned original sample identity required before row-ID projection")
    report.update(source_helpers=own, official_sources=official, export_pins=pin_id,
                  sample_identity=sample_id, clip_spec=asdict(spec), phase="frozen_episode_consumer")
    persist()
    if any(name in sys.modules for name in ("torch", "joblib")):
        raise ValueError("Fresh CPU-only hash/trajectory process required")
    loaded = (consumer or load_shared_track1_episode)(root, code, spec, pins)
    loaded.episode.validate()
    manifest = loaded.manifest
    required = dict(stage="world_reward_shared_native_episode_consumer", episode_index=episode, frames=spec.total_frames,
                    original_frame_coverage_verified=True, integrity_and_schema_verified=True, native_direct_export_consumed=True,
                    old_LM_conversion_used=False, numerical_geometry_independently_reverified=False, quality_verified=False,
                    challenge_performance_verified=False, submission_eligibility_verified=False,
                    submission_eligible=False, final_Parquet_produced=False)
    if (type(manifest) is not dict or any(type(manifest.get(k)) is not type(v) or manifest[k] != v for k, v in required.items())
            or loaded.episode.total_video_frames != spec.total_frames
            or manifest.get("export_report_sha256") != pins["export"]["sha256"]
            or manifest.get("export_files") != pins["export_files"]):
        raise ValueError("Exact real full-native export engineering manifest required")
    # read_template's actual Arrow call requests columns=['row_id'] exclusively.
    layout = selected_layout((template_reader or read_template)(sample_path), episode)
    commit = code_commit_url(report["producer_revision"])
    report.update(consumer_manifest=manifest, phase="row_id_firewall", original_sample_rows=740780,
                  original_sample_scored_frames=9877, selected_row_id_sha256=row_id_digest(layout.row_ids),
                  original_sample_prediction_columns_read=False, official_sample_input="new_row_id_only_CSV",
                  code_commit_url=commit)
    persist()
    scratch = Path(tempfile.mkdtemp(prefix=".official-pack-smoke-", dir=out))
    try:
        pack_once(root, scratch, loaded, layout, report, persist, commit=commit, packer=packer)
    finally:
        shutil.rmtree(scratch)
        report["scratch_removed"] = not scratch.exists()
        persist()
    if (identity(pins_path) != pin_id or strict_json(pins_path) != pins or source_helpers(code) != own
            or official_sources(root) != official or identity(sample_path, immutable=False) != sample_id
            or any(identity(path) != value for path, value in export_files.items())
            or any(name in sys.modules for name in ("torch", "joblib"))
            or {path.name for path in out.iterdir()} != {"report.json"}):
        raise ValueError("Frozen source/pins/sample changed or nonreceipt output remains")
    report.update(status="pass", phase="complete", frames=spec.total_frames,
                  unmodified_official_packer_verified=True, original_row_id_order_verified=True,
                  source_helpers_rehashed=True, official_sources_rehashed=True, export_pins_rehashed=True,
                  complete_export_payloads_rehashed=True,
                  complete_challenge_submission_created=False, final_Parquet_produced=False)


def execute(root, out, code, episode, revision, **callbacks):
    if not out.is_dir() or any(out.iterdir()):
        raise ValueError("Fresh exclusive one-receipt smoke output required")
    report = dict(stage=STAGE, status="fail", phase="public_integrity", episode_index=episode,
                  producer_revision=revision, script_sha256=identity(code / "infra/official_track1_pack_gate.py")["sha256"],
                  image_id=IMAGE, input_track="track_1", network="none", budget_seconds=BUDGET,
                  ground_truth_used=False, hand_labeled_test=False, oracle_modes=[], GPU_used=False,
                  model_calls=0, optimizer_calls=0, render_calls=0, Kaggle_upload_calls=0,
                  official_packer_attempts=0, official_packer_returns=0, official_packer_validated=0,
                  numerical_geometry_independently_reverified=False, quality_verified=False,
                  code_accessibility_verified_offline=False, submission_eligibility_verified=False, submission_eligible=False,
                  complete_challenge_submission_created=False, final_Parquet_produced=False)
    path = out / "report.json"
    started = time.perf_counter()
    with path.open("x") as stream:
        def persist():
            report["elapsed_seconds"] = time.perf_counter() - started
            stream.seek(0)
            json.dump(report, stream, allow_nan=False)
            stream.write("\n"); stream.truncate(); stream.flush(); os.fsync(stream.fileno())
        def expired(*_):
            raise TimeoutError("One-episode official CPU packer smoke exceeded300s")
        alarm = signal.signal(signal.SIGALRM, expired); term = signal.signal(signal.SIGTERM, expired)
        signal.alarm(BUDGET)
        try:
            persist(); run(root, out, code, episode, report, persist, **callbacks)
        except BaseException as error:
            report.update(status="fail", error_type=type(error).__name__, error="One-episode packing failed; inspect phase")
            raise
        finally:
            signal.alarm(0); signal.signal(signal.SIGALRM, alarm); signal.signal(signal.SIGTERM, term)
            persist(); path.chmod(0o444)
    return report


def require_readonly_mounts(paths, *, mountinfo="/proc/self/mountinfo"):
    """Each exact bind must be VFS readonly; vendor original modes are untouched."""
    mounted = {}
    for row in Path(mountinfo).read_text().splitlines():
        fields = row.split()
        if len(fields) < 7 or "-" not in fields:
            raise ValueError("Malformed Linux mountinfo")
        point = re.sub(r"\\([0-7]{3})", lambda value: chr(int(value.group(1), 8)), fields[4])
        mounted[point] = set(fields[5].split(","))
    for path in paths:
        if "ro" not in mounted.get(str(path), set()):
            raise ValueError("Exact official source/sample/code bind must be readonly")


def main(argv=None):
    args = parser().parse_args(argv)
    root = Path(os.environ["WR_ROOT"]); code = Path(os.environ["WR_CODE"]); revision = os.environ["WR_CODE_REVISION"]
    out = root / output_relative(args.episode)
    if (platform.system() != "Linux" or root != ROOT or os.geteuid() != 1000
            or {path.name for path in Path("/sys/class/net").iterdir()} != {"lo"}
            or os.environ["WR_IMAGE_ID"] != IMAGE or not re.fullmatch(r"[0-9a-f]{40}", revision)
            or code != root / "jobs" / revision / "run_official_track1_pack_gate/code"
            or Path(__file__).resolve() != code / "infra/official_track1_pack_gate.py"
            or any(path.resolve() != path.absolute() or any(p.is_symlink() for p in (path, *path.parents)) for path in (root, code, out))):
        raise ValueError("Actual immutable offline CPU official packer launcher/image required")
    kit = root / "vendor/v2d_submission_kit"
    require_readonly_mounts((code, *(kit / name for name in OFFICIAL_SOURCES), root / SAMPLE_RELATIVE))
    execute(root, out, code, args.episode, revision)


if __name__ == "__main__":
    main()
