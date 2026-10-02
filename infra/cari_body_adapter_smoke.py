"""Gate the canonical CARI MHR initializer against full original Body geometry."""

import json
import os
from pathlib import Path
import platform
import pickle
import sys
import time

from body_smoke import _body_assets, _source_identity, UPSTREAM_REVISION
from cari_body_adapter import candidate_mhr_parameters
from world_reward.data import sha256


def main():
    if platform.system() != "Linux" or {p.name for p in Path("/sys/class/net").iterdir()} != {"lo"}:
        raise RuntimeError("Require Azure Linux GPU container with network none")
    import numpy as np
    import torch
    root = Path(os.environ["WR_ROOT"])
    directory = root / "outputs/episode_000015/body_full"
    report_path = directory / "report.json"
    report = json.loads(report_path.read_text())
    if (report["status"] != "pass" or report["ground_truth_used"] is not False
            or report["hand_labeled_test"] is not False or report["oracle_modes"] != []
            or report["input_track"] != "track_1" or report["upstream_revision"] != UPSTREAM_REVISION
            or report["mhr_geometry_forward_verified"] is not True):
        raise RuntimeError("Require exact native Body provenance and forward verification")
    path = directory / "predictions.npz"
    if sha256(path) != report["predictions_sha256"] or not torch.cuda.is_available():
        raise RuntimeError("Frozen Body predictions or CUDA missing")
    output = directory / "cari_adapter"
    if output.exists():
        raise RuntimeError("Frozen adapter result exists")
    source = _source_identity(root)
    assets, hashes = _body_assets(root)
    if hashes != report["body_assets"]:
        raise RuntimeError("Body model/asset files differ from original inference")
    upstream = root / "vendor/video_to_data/reconstruction/modules/v2d_cari4d/lib/cari4d"
    if not upstream.is_dir():
        raise RuntimeError("Pinned CARI4D module checkout missing")
    sys.path.insert(0, str(upstream))
    from lib_mhr.mhr_layer import MHRLayer
    # No compact-buffer fallback: use exact original Body checkpoint buffers
    # and its independently validated SAM MHR asset, not the public kit model.
    layer = MHRLayer.from_mhr_assets(mhr_assets_root=Path("/workspace/v2d_sam3d_body/lib"),
                                    checkpoint_path=assets / "model.ckpt",
                                    buffer_path=output / "never_use_an_unverified_compact_buffer.pt",
                                    mhr_model_path=assets / "assets/mhr_model.pt", device="cuda")
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key].copy() for key in data.files}
    height, width = 1152, 1536
    focal = float(np.hypot(height, width))
    K = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]], dtype=np.float32)
    started = time.perf_counter()
    candidate = candidate_mhr_parameters(arrays, K, (height, width), total_frames=report["total_video_frames"],
                                         mhr_layer=layer, decode_batch_size=16)
    output.mkdir(exist_ok=False)
    target = output / "canonical_initializer.pkl"
    with target.open("xb") as handle:
        pickle.dump(candidate, handle, protocol=4)
    result = {"stage": "native_cari_body_adapter_full_video", "status": "pass", "frames": len(candidate["frames"]),
              "metadata": candidate["metadata"], "body_report_sha256": sha256(report_path),
              "canonical_initializer_sha256": sha256(target), "decoder_identity": layer.decoder_identity(),
              "inference_source_identity": source, "script_sha256": sha256(Path(__file__)),
              "input_track": "track_1", "ground_truth_used": False, "hand_labeled_test": False, "oracle_modes": [],
              "submission_eligible": False, "challenge_performance_verified": False,
              "elapsed_seconds": time.perf_counter() - started}
    (output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"stage": result["stage"], "status": "pass", "frames": result["frames"],
                      "native_errors_m": candidate["metadata"]["native_roundtrip_max_error_m"],
                      "projection_error_px": candidate["metadata"]["projection_max_error_px"],
                      "elapsed_seconds": result["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
