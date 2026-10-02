"""Frozen legacy forward loopback-guard/source audit, not network attestation.

Never edits the producer report or treats missing fields as offline by default.
The exact allowlisted producer has a mandatory Linux/loopback-only guard before
inference. Its audited immutable wrapper source declares Docker --network none.
The original transient launch is unavailable; no runtime/launch attestation is
claimed. Hashes bind this source-contract audit to the original report/bundle.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import re
import time

from world_reward.data import sha256


LEGACY_REVISION = "414aac2a988f444eb166494c9d0d6aca5872ad0c"
LEGACY_SCRIPT_SHA256 = "a8569885b0d8b706ebade454e8f9a87fc6e56fce97f46b8511abfe04dcb7305f"
LEGACY_WRAPPER_SHA256 = "b7c7fa28d0117c9707cd24e4137535b8554f8228576d89c591dea6d1d5422754"
LEGACY_REPORT_SHA256 = "c3978dc52748f6c084240f764e6ddcd2442b745e482a10d3c86397a89e412cc7"
LEGACY_BUNDLE_SHA256 = "69d5f9dba0ba521bae227dfa0e6a3705ccb927a3dc35ccd55ecc28248f85f655"
LEGACY_ARCHIVE_SHA256 = "7b961b748322b33091d6ec673ed9c18e79224dfe8c750f532421420722d0c032"
PROOF_STAGE = "audited_legacy_forward_loopback_guard_contract"
PROOF_BASIS = "mandatory_allowlisted_loopback_guard_and_audited_wrapper_source"


def require_digest(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("Require a lowercase SHA-256")
    return value


def validate_proof(proof, forward, report_hash, bundle_hash, episode):
    """Consumer verifies exact historical proof identity, never fills network."""
    required = {"stage": PROOF_STAGE, "status": "pass", "episode_index": 15,
                "legacy_revision": LEGACY_REVISION, "producer_script_sha256": LEGACY_SCRIPT_SHA256,
                "wrapper_sha256": LEGACY_WRAPPER_SHA256, "forward_report_sha256": LEGACY_REPORT_SHA256,
                "bundle_sha256": LEGACY_BUNDLE_SHA256, "source_archive_sha256": LEGACY_ARCHIVE_SHA256,
                "basis": PROOF_BASIS, "runtime_guard_source_verified": True, "immutable_wrapper_source_verified": True,
                "immutable_wrapper_launch_bound": False, "network_security_attestation": False,
                "producer_report_modified": False, "source_archive_rehashed": False}
    if (not isinstance(proof, dict) or not isinstance(forward, dict) or type(episode) is not int or episode != 15
            or "network" in forward or forward.get("script_sha256") != LEGACY_SCRIPT_SHA256
            or report_hash != LEGACY_REPORT_SHA256 or bundle_hash != LEGACY_BUNDLE_SHA256
            or forward.get("bundle_sha256") != LEGACY_BUNDLE_SHA256):
        raise ValueError("Only the missing-field allowlisted legacy episode15 forward can use this proof")
    for key, expected in required.items():
        actual = proof.get(key)
        if actual != expected or (type(expected) in (bool, int) and type(actual) is not type(expected)):
            raise ValueError(f"Legacy network proof mismatch: {key}")
    for key in ("source_archive_sha256", "forward_report_sha256", "bundle_sha256"):
        require_digest(proof.get(key))
    if "launch_exec_start_sha256" in proof:
        raise ValueError("Collected legacy launch cannot be presented as observed")
    if "episode_index" in forward and (type(forward["episode_index"]) is not int or forward["episode_index"] != episode):
        raise ValueError("Legacy forward report belongs to another episode")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.parse_args(argv)
    if platform.system() != "Linux" or {p.name for p in Path("/sys/class/net").iterdir()} != {"lo"}:
        raise RuntimeError("Require remote Linux CPU container with network none")
    root = Path(os.environ["WR_ROOT"])
    gate_revision = os.environ.get("WR_CODE_REVISION", "")
    if re.fullmatch(r"[0-9a-f]{40}", gate_revision) is None: raise ValueError("Require immutable proof source revision")
    base, job = root / "outputs/episode_000015", root / f"jobs/{LEGACY_REVISION}/run_cari_forward"
    output = base / "cari_forward/network_proof.json"
    if output.exists() or output.is_symlink(): raise FileExistsError("Legacy network sidecar is frozen")
    paths = {"forward": base / "cari_forward/report.json", "final": base / "cari_conversion/report.json",
             "bundle": base / "cari_forward/coconet.pth", "script": job / "code/infra/cari_forward.py",
             "wrapper": job / "code/infra/run_cari_forward.sh", "revision": job / "revision", "archive": job / "source-sha256"}
    for path in paths.values():
        if not path.is_file() or any(p.is_symlink() for p in (path, *path.parents) if p.is_relative_to(root)):
            raise ValueError("Proof inputs must be existing regular non-symlink frozen paths")
    if any(paths[name].stat().st_mode & 0o222 for name in ("script", "wrapper")):
        raise ValueError("Audited immutable producer/wrapper source must not be writable")
    started = time.perf_counter()
    forward, final = (json.loads(paths[name].read_text()) for name in ("forward", "final"))
    from cari_converter import require_full_forward_report, require_report
    require_full_forward_report(forward)
    require_report(final, "world_reward_native_cari_official_conversion")
    from track1_episode_loader import legacy_episode15_conversion
    report_hash, bundle_hash = sha256(paths["forward"]), sha256(paths["bundle"])
    selected_final = type(final.get("episode_index")) is int and final["episode_index"] == 15
    if (not (selected_final or legacy_episode15_conversion(final, 15))
            or report_hash != LEGACY_REPORT_SHA256 or bundle_hash != LEGACY_BUNDLE_SHA256
            or final.get("input_report_sha256", {}).get("forward") != report_hash
            or final.get("bundle_sha256") != bundle_hash or forward.get("bundle_sha256") != bundle_hash
            or paths["revision"].read_text().strip() != LEGACY_REVISION
            or paths["archive"].read_text().strip() != LEGACY_ARCHIVE_SHA256
            or sha256(paths["script"]) != LEGACY_SCRIPT_SHA256 or sha256(paths["wrapper"]) != LEGACY_WRAPPER_SHA256):
        raise ValueError("Original report/bundle/source/revision binding failed")
    proof = {"stage": PROOF_STAGE, "status": "pass", "episode_index": 15, "legacy_revision": LEGACY_REVISION,
             "producer_script_sha256": LEGACY_SCRIPT_SHA256, "wrapper_sha256": LEGACY_WRAPPER_SHA256,
             "forward_report_sha256": report_hash, "bundle_sha256": bundle_hash,
             "source_archive_sha256": require_digest(paths["archive"].read_text().strip()),
             "basis": PROOF_BASIS, "runtime_guard_source_verified": True, "immutable_wrapper_source_verified": True,
             "immutable_wrapper_launch_bound": False,
             "network_security_attestation": False, "producer_report_modified": False,
             "source_archive_rehashed": False, "proof_code_revision": gate_revision,
             "legacy_launch_observation_available": False,
             "elapsed_seconds": time.perf_counter() - started}
    validate_proof(proof, forward, report_hash, bundle_hash, 15)
    with output.open("x") as handle: handle.write(json.dumps(proof, indent=2, allow_nan=False) + "\n")
    print(json.dumps(proof), flush=True)


if __name__ == "__main__": main()
