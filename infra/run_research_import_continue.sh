#!/usr/bin/env bash
# Explicit new import leg: no re-extraction, Docker rebuild or failed-unit restart.
set -euo pipefail
(( $# == 0 )) || exit 2
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; OUT="$ROOT/transfer/vm02-v1"
[[ ! -e "$OUT/import.json" && ! -L "$OUT/import.json" && ! -e "$OUT/image-identity-v2.json" ]]
export DOCKER_HOST="unix://$ROOT/docker.sock"
[[ "$(docker info --format '{{.DockerRootDir}}')" == "$ROOT/docker" ]]
# Repeat the exact extracted public/private asset integrity gate, as their owner.
timeout 120s runuser -u scenesmith -- env WR_ROOT="$ROOT" PYTHONDONTWRITEBYTECODE=1 \
 python3 - "$CODE/infra/research_transfer.py" <<'PY'
import importlib.util,os,pathlib,sys
spec=importlib.util.spec_from_file_location('verified_transfer',sys.argv[1])
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
module.inventory(pathlib.Path(os.environ['WR_ROOT']),True)
print('extracted_task_assets_verified=true',flush=True)
PY
timeout 120s docker image inspect world-reward/cari4d-source:0.1 | \
 timeout 120s python3 "$CODE/infra/research_image_identity.py" --archive "$OUT/image.tar" > "$OUT/image-identity-v2.json"
IMAGE=sha256:7ebfff18ba3b76dd919485c19115597d7531dfd3233f69461f1dce3f28a6c6d3
timeout --signal=TERM --kill-after=5s 90s docker run --rm --gpus all --network none --memory 4g --cpus 2 \
 --user 1000:1000 --entrypoint python --env HOME=/tmp --env PYTHONDONTWRITEBYTECODE=1 "$IMAGE" \
 -c 'import torch,importlib.metadata as m;assert torch.cuda.is_available();x=torch.eye(4,device="cuda");assert float((x@x).sum())==4;print({"CUDA":True,"device":torch.cuda.get_device_name(),"dependencies":{p:m.version(p) for p in ("torch","torchvision","omegaconf","einops","imageio","safetensors")}})' \
 > "$OUT/cuda-smoke-v2.txt"
python3 - "$OUT" "${WR_CODE_REVISION:?}" <<'PY'
import hashlib,json,pathlib,sys
out=pathlib.Path(sys.argv[1]);r=json.loads((out/'image-identity-v2.json').read_text())
export=out/'export.json'
assert hashlib.sha256(export.read_bytes()).hexdigest()=='2bcb3c0e11ecacc6cd6051447380760eda20266a7dfe17420c47e41a2a196d8b'
r.update(stage='world_reward_research_import',status='pass',producer_revision=sys.argv[2],
 export_receipt_sha256=hashlib.sha256(export.read_bytes()).hexdigest(),task_assets_extracted_as_UID=1000,
 private_annotations_mounted_in_GPU_smoke=False,CUDA_execution_verified=True,
 GPU_smoke_sha256=hashlib.sha256((out/'cuda-smoke-v2.txt').read_bytes()).hexdigest(),
 model_inference_verified=False,source_unit_restarted=False)
with (out/'import.json').open('x') as f:json.dump(r,f);f.write('\n')
print(json.dumps(r),flush=True)
PY
