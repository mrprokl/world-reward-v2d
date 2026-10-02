"""Convert real three-frame Body predictions with the official shared-identity tool."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import platform
import time

from world_reward.data import sha256


def main() -> None:
    if platform.system() != "Linux":
        raise RuntimeError("Real model/mesh fitting stays on Azure")
    import numpy as np
    import torch
    root = Path(os.environ.get("WR_ROOT", "/srv/scenesmith/world-reward"))
    directory = root / "outputs/episode_000015/body_smoke"
    report = json.loads((directory / "report.json").read_text())
    if report["status"] != "pass" or report["ground_truth_used"] is not False or not report["mhr_geometry_forward_verified"]:
        raise RuntimeError("Require verified real Body geometry and video-only provenance")
    if sha256(directory / "predictions.npz") != report["predictions_sha256"]:
        raise RuntimeError("Body predictions changed after their frozen report")
    model = root / "weights/mhr/mhr_model.pt"
    if sha256(model) != "352e271a6c42729c68554ceaea0c955e866970160c31e35506d782dc0f7377bc":
        raise RuntimeError("Reference MHR must use the Apache audited model")
    tool = root / "vendor/v2d_submission_kit/tools/track1/mesh_to_mhr_params.py"
    if sha256(tool) != "c799ad612fca19620563fcb93bf61e5a4adad0a04251482358746b5f27f8a52e":
        raise RuntimeError("Official converter changed; re-audit before run")
    output = directory / "official_conversion"
    output.mkdir(exist_ok=False)
    with np.load(directory / "predictions.npz", allow_pickle=False) as predictions:
        vertices = predictions["vertices_camera_m"].copy()
        indices = predictions["frame_index"].tolist()
    if indices != [0, 250, 500] or vertices.shape != (3, 18439, 3):
        raise RuntimeError("Predeclared sparse frame gate changed")
    spec = importlib.util.spec_from_file_location("official_converter", tool)
    converter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter)
    started = time.perf_counter()
    converted = converter.convert(vertices, str(model), device="cuda", precision="float32",
                                  model_batch=256, log=lambda *args, **kwargs: None)
    np.savez_compressed(output / "converted_params.npz", **{key: converted[key] for key in ("pose", "scales", "shape")})
    reference = converter.MHR(str(model), "cuda", chunk=256, precision="float32")
    recovered, _ = reference.run(torch.tensor(converted["pose"], device="cuda", dtype=torch.float64),
                                 torch.tensor(np.concatenate([converted["scales"], converted["shape"]])[None],
                                              device="cuda", dtype=torch.float64))
    residual_mm = torch.linalg.vector_norm(recovered / 1000 - torch.tensor(vertices, device="cuda"), dim=-1).mean(1) * 1000
    if not torch.isfinite(residual_mm).all():
        raise RuntimeError("Official conversion has nonfinite independent residual")
    result = {"stage": "real_body_official_converter", "input_frame_indices": indices,
              "mean_vertex_error_mm": float(residual_mm.mean()), "per_frame_mean_mm": residual_mm.cpu().tolist(),
              "elapsed_seconds": time.perf_counter() - started, "converter_report": converted.get("report"),
              "shared_identity": True, "facial_expressions_zero": True,
              "geometry_forward_verified": True, "original_units": "metres",
              "input_sha256": report["predictions_sha256"], "model_sha256": sha256(model),
              "converter_sha256": sha256(tool), "script_sha256": sha256(Path(__file__)),
              "scope": "sparse_engineering_conversion_only_not_full_video_submission",
              "challenge_accuracy_verified": False, "submission_eligible": False}
    (output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("stage", "mean_vertex_error_mm", "per_frame_mean_mm", "elapsed_seconds")}))


if __name__ == "__main__":
    main()
