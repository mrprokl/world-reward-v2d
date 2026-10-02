#!/usr/bin/env bash
# Import-only CPU gate; alias unchanged CARI image only after actual imports PASS.
set -euo pipefail
(( $# == 0 )) || exit 2
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"
OUT="$ROOT/results/da3-runtime-v1.json"
[[ ! -e "$OUT" && ! -L "$OUT" ]]
export DOCKER_HOST="unix://$ROOT/docker.sock"
EXPECTED=sha256:7ebfff18ba3b76dd919485c19115597d7531dfd3233f69461f1dce3f28a6c6d3
[[ "$(docker image inspect world-reward/cari4d-source:0.1 --format '{{.Id}}')" == "$EXPECTED" ]]
[[ -f "$ROOT/transfer/vm02-v1/import.json" ]]
python3 - "$ROOT/transfer/vm02-v1/import.json" "$EXPECTED" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
assert r['stage']=='world_reward_research_import' and r['status']=='pass'
assert r['image_id']==sys.argv[2] and r['CUDA_execution_verified'] is True
assert r['source_OCI_index_id']=='sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7'
assert r['image_content_changed'] is False and r['image_rebuilt'] is False
PY
SOURCE="$ROOT/vendor/research/da3_metric_v1"
REPORT_DIR="$ROOT/results/da3-runtime-v1"
[[ ! -e "$REPORT_DIR" && ! -L "$REPORT_DIR" ]]
mkdir "$REPORT_DIR"; chown 1000:1000 "$REPORT_DIR"
timeout --signal=TERM --kill-after=5s 90s docker run --rm --network none --memory 4g --cpus 2 \
 --user 1000:1000 --entrypoint python --env HOME=/tmp --env PYTHONDONTWRITEBYTECODE=1 \
 --env PYTHONPATH="$SOURCE/src" --env SOURCE_ROOT="$SOURCE" --env WR_ROOT="$ROOT" \
 --env PRODUCER_REVISION="${WR_CODE_REVISION:?}" --env IMAGE_ID="$EXPECTED" \
 --mount "type=bind,src=$SOURCE,dst=$SOURCE,readonly" \
 --mount "type=bind,src=$ROOT/results/da3-metric-acquisition-v1.json,dst=$ROOT/results/da3-metric-acquisition-v1.json,readonly" \
 --mount "type=bind,src=$REPORT_DIR,dst=/reports" "$EXPECTED" -c '
import hashlib,importlib,importlib.metadata as m,json,os,pathlib,sys
source=pathlib.Path(os.environ["SOURCE_ROOT"]);root=pathlib.Path(os.environ["WR_ROOT"])
receipt=json.loads((root/"results/da3-metric-acquisition-v1.json").read_text())
assert receipt["status"]=="pass" and receipt["source_revision"]=="3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"
manifest=source/"source_manifest.json"
assert hashlib.sha256(manifest.read_bytes()).hexdigest()==receipt["source_manifest_sha256"]
for record in json.loads(manifest.read_text())["files"]:
 p=source/record["file"];assert not p.is_symlink() and hashlib.sha256(p.read_bytes()).hexdigest()==record["sha256"]
names=["depth_anything_3.cfg","depth_anything_3.utils.io.input_processor","depth_anything_3.utils.io.output_processor","depth_anything_3.model.da3","depth_anything_3.model.dinov2.dinov2","depth_anything_3.model.dpt"]
paths={n:importlib.import_module(n).__file__ for n in names}
assert all(pathlib.Path(p).resolve().is_relative_to(source/"src") for p in paths.values())
assert "depth_anything_3.api" not in sys.modules and not any(n=="evo" or n.startswith("evo.") for n in sys.modules)
report={"stage":"da3_minimal_runtime_import_gate","status":"pass","producer_revision":os.environ["PRODUCER_REVISION"],"image_id":os.environ["IMAGE_ID"],"base_image_id":os.environ["IMAGE_ID"],"source_manifest_sha256":receipt["source_manifest_sha256"],"imports":paths,"versions":{p:m.version(p) for p in ["torch","torchvision","omegaconf","einops","addict","imageio","Pillow","tqdm","safetensors"]},"model_instantiated":False,"GPU_forward_performed":False,"dependency_install_performed":False}
with open("/reports/report.json","x") as f:json.dump(report,f);f.write("\n")
print(json.dumps(report),flush=True)
'
docker image tag "$EXPECTED" world-reward/da3-metric:0.1
[[ "$(docker image inspect world-reward/da3-metric:0.1 --format '{{.Id}}')" == "$EXPECTED" ]]
ln "$REPORT_DIR/report.json" "$OUT"
