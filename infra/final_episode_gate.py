"""Remote CPU final-episode integrity/schema gate, not accuracy or eligibility."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import time

from body_smoke import _validate_inputs
from track1_episode_loader import load_track1_episode


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--episode", type=int, choices=range(30), required=True)
    args = parser.parse_args(argv)
    if platform.system() != "Linux" or {path.name for path in Path("/sys/class/net").iterdir()} != {"lo"}:
        raise RuntimeError("Require remote Linux container with network none")
    revision = os.environ.get("WR_CODE_REVISION", "")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise RuntimeError("Require immutable gate source revision")
    root = Path(os.environ["WR_ROOT"])
    output = root / f"outputs/episode_{args.episode:06d}/final_schema"
    if output.exists() or output.is_symlink():
        raise FileExistsError("Final schema outputs are frozen; never overwrite")
    started = time.perf_counter()
    inputs = _validate_inputs(root, episode_index=args.episode)
    loaded = load_track1_episode(root, args.episode, inputs["total_frames"], inputs["video_sha256"])
    loaded.episode.validate()
    report = {"stage": "world_reward_final_episode_integrity_schema", "status": "pass",
              "episode_index": args.episode, "frames": inputs["total_frames"],
              "input_track": "track_1", "input_sha256": inputs["video_sha256"],
              "ground_truth_used": False, "hand_labeled_test": False, "oracle_modes": [],
              "input_dataset_revision": inputs["dataset_revision"],
              "mask_report_sha256": inputs["mask_report_sha256"],
              "manifest": loaded.manifest, "integrity_and_schema_verified": True,
              "numerical_truth_independently_reverified": False, "submission_eligibility_verified": False,
              "license_eligibility_verified": False, "challenge_performance_verified": False,
              "complete_challenge_submission_created": False, "all_stage_producer_revisions_same": False,
              "producer_revision_scope": "this_gate_only_upstream_artifacts_bound_by_report_hashes",
              "gate_code_revision": revision, "network": "none", "execution_device": "CPU",
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "elapsed_seconds": time.perf_counter() - started}
    output.mkdir(exist_ok=False)
    with (output / "report.json").open("x") as handle:
        handle.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in ("stage", "status", "episode_index", "frames",
                                                  "integrity_and_schema_verified", "elapsed_seconds")}), flush=True)


if __name__ == "__main__":
    main()
