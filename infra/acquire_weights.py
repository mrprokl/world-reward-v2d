"""Remote-only acquisition of pinned pretrained assets; never writes token values."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import urllib.request
import zipfile


def main() -> None:
    if platform.system() != "Linux":
        raise RuntimeError("Model downloads are restricted to Azure Linux")
    root = Path(os.environ.get("WR_ROOT", "/srv/scenesmith/world-reward"))
    os.environ["HF_TOKEN"] = (root / ".secrets/hf_token").read_text().strip()
    os.environ["HF_HOME"] = str(root / "cache/huggingface")
    from huggingface_hub import snapshot_download

    weights = root / "weights"
    records = []
    specs = [
        ("nvidia/cari4d_commercial", "1f7287ac6fd5f72c30ce2222fb345a3e7d779fc9", "cari4d/cari4d", ["2026-08-25-09-35-57/*"]),
        ("facebook/sam-3d-body-dinov3", "11aaa346c7204874a1cbafe3d39a979080b2c55a", "cari4d/sam3d_body/checkpoints/sam-3d-body-dinov3", ["model.ckpt", "model_config.yaml", "assets/*", "LICENSE", "README.md"]),
        ("facebook/sam-3d-objects", "2e73555018d2741ccd486e56c24fac41155a1dc6", "sam3d/hf-download", ["checkpoints/*", "LICENSE"]),
        ("facebook/sam2.1-hiera-large", "665f8e2ad61cf5f53d65644ff27c8ee525124610", "sam2", ["sam2.1_hiera_large.pt", "sam2.1_hiera_l.yaml", "README.md"]),
        ("IDEA-Research/grounding-dino-base", "12bdfa3120f3e7ec7b434d90674b3396eccf88eb", "grounding_dino", ["*.json", "*.txt", "model.safetensors", "README.md"]),
    ]
    for repo, revision, folder, patterns in specs:
        print(json.dumps({"acquiring": repo, "revision": revision}), flush=True)
        snapshot_download(repo_id=repo, revision=revision, local_dir=weights/folder, allow_patterns=patterns)
        records.append({"repo_id":repo,"revision":revision,"path":str(weights/folder)})
    checkpoint = weights / "cari4d/cari4d/2026-08-25-09-35-57/step200000.pth"
    with checkpoint.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != "78ff5cb874dd012a272382e3f2d8bc11226d5b7d0ecc739a60fbb4a97a5a5ba3":
        raise RuntimeError("Pinned CARI4D checkpoint integrity failed")
    for repo, revision, cache in [
        ("Ruicheng/moge-2-vitl-normal", "b135031bae30b5ac2ae141a0e68717795ce38340", weights/"cari4d/hf_home/hub"),
        ("Ruicheng/moge-vitl", "ad326bfb61facd6c52b5a825bc1e34d7c97d9672", weights/"sam3d/hf_home/hub"),
    ]:
        snapshot_download(repo_id=repo, revision=revision, cache_dir=cache, allow_patterns=["model.pt", "README.md"])
        records.append({"repo_id":repo,"revision":revision,"cache_dir":str(cache)})
    model_dir = weights / "mhr"
    model_dir.mkdir(parents=True, exist_ok=True)
    model = model_dir / "mhr_model.pt"
    if not model.exists():
        archive = model_dir / "assets.zip"
        urllib.request.urlretrieve("https://github.com/facebookresearch/MHR/releases/download/v1.0.1/assets.zip", archive)
        temporary = model.with_suffix(".part")
        with zipfile.ZipFile(archive) as z:
            with z.open("assets/mhr_model.pt") as src, temporary.open("wb") as dst:
                while chunk := src.read(1024*1024):
                    dst.write(chunk)
        with temporary.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != "352e271a6c42729c68554ceaea0c955e866970160c31e35506d782dc0f7377bc":
                raise RuntimeError("Reference MHR integrity failed")
        temporary.replace(model)
        archive.unlink()
    with model.open("rb") as stream:
        mhr_digest = hashlib.file_digest(stream,"sha256").hexdigest()
    if mhr_digest != "352e271a6c42729c68554ceaea0c955e866970160c31e35506d782dc0f7377bc":
        raise RuntimeError("Reference MHR integrity failed")
    result = {"assets":records,"cari4d_sha256":digest,"mhr_license":"Apache-2.0",
              "scope":"principal_hf_and_mhr_assets_only",
              "auxiliary_assets_required":["FoundationPose", "DINOv2", "DINOv3_torch_hub"]}
    (root/"results/weights-acquisition.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"principal_assets":"complete","auxiliary_assets":"separate_stage",
                      "cari4d_integrity":"verified"}))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # Authentication failures must remain actionable without exposing secrets.
        token = os.environ.get("HF_TOKEN")
        message = str(exc).replace(token, "[REDACTED]") if token else str(exc)
        print(json.dumps({"error": type(exc).__name__, "message": message}), file=sys.stderr)
        sys.exit(1)
