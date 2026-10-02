#!/usr/bin/env bash
# Verify task-only transfer before extracting or importing any executable image.
set -euo pipefail
[[ $# == 2 && "$1" == --export-sha256 && "$2" =~ ^[0-9a-f]{64}$ ]] || exit 2
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ ]]
OUT="$ROOT/transfer/vm02-v1"
[[ -d "$OUT" && ! -L "$OUT" && ! -e "$OUT/import.json" && ! -L "$OUT/import.json" ]]
EXPECTED=sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7
export DOCKER_HOST="unix://$ROOT/docker.sock"
[[ "$(docker info --format '{{.DockerRootDir}}')" == "$ROOT/docker" ]]
[[ "$(docker image ls -q | wc -l)" == 0 ]]
timeout 600s python3 - "$OUT" "$2" "$EXPECTED" <<'PY'
import hashlib,json,pathlib,sys
out=pathlib.Path(sys.argv[1])
def sha(p):
    if p.is_symlink() or not p.is_file() or p.resolve()!=p.absolute(): raise ValueError('Canonical regular transfer files required')
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
p=out/'export.json'
assert p.stat().st_size<10000 and sha(p)==sys.argv[2], 'Unbound export receipt'
r=json.loads(p.read_text())
assert r.get('status')=='pass' and r.get('stage')=='world_reward_task_only_export'
assert r.get('producer_revision')=='671d10cacb146c1ef3c1edd22af155eb994a4122'
assert r.get('image_id')==sys.argv[3] and r.get('image_tag')=='world-reward/cari4d-source:0.1'
assert all(r.get(k) is False for k in ('docker_state_copied','user_disk_copied','secrets_transferred','challenge_inputs_transferred'))
records=r.get('artifacts')
assert isinstance(records,list) and [x.get('file') for x in records]==['assets.tar','image.tar']
for record,maximum in zip(records,(2_200_000_000,32_000_000_000)):
    p=out/record['file']; assert type(record['bytes']) is int and 0<record['bytes']<=maximum
    assert p.stat().st_size==record['bytes'] and sha(p)==record['sha256'], 'Transferred artifact changed'
print(records[0]['sha256'])
PY
ASSET_SHA="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["artifacts"][0]["sha256"])' "$OUT/export.json")"
timeout 610s runuser -u scenesmith -- env WR_ROOT="$ROOT" PYTHONDONTWRITEBYTECODE=1 \
 python3 "$CODE/infra/research_transfer.py" extract --archive "$OUT/assets.tar" --sha256 "$ASSET_SHA"
timeout --signal=TERM --kill-after=10s 1800s docker image load --input "$OUT/image.tar"
[[ "$(docker image inspect world-reward/cari4d-source:0.1 --format '{{.Id}}')" == "$EXPECTED" ]]
timeout --signal=TERM --kill-after=5s 90s docker run --rm --gpus all --network none --memory 4g --cpus 2 \
 --user 1000:1000 --entrypoint python --env HOME=/tmp --env PYTHONDONTWRITEBYTECODE=1 \
 "$EXPECTED" -c 'import torch, importlib.metadata as m; assert torch.cuda.is_available(); x=torch.eye(4,device="cuda"); assert float((x@x).sum())==4; print({"CUDA":True,"device":torch.cuda.get_device_name(),"dependencies":{p:m.version(p) if m.packages_distributions().get(p) else "missing" for p in ("torch","torchvision","kornia","omegaconf","einops","addict","imageio","safetensors")}})' \
 > "$OUT/cuda-smoke.txt"
python3 - "$OUT" "$REV" "$EXPECTED" "$2" <<'PY'
import hashlib,json,pathlib,sys
out=pathlib.Path(sys.argv[1]); r={'stage':'world_reward_research_import','status':'pass',
 'producer_revision':sys.argv[2],'image_id':sys.argv[3],'export_receipt_sha256':sys.argv[4],
 'task_assets_extracted_as_UID':1000,'private_annotations_mounted_in_GPU_smoke':False,
 'CUDA_execution_verified':True,'GPU_smoke_sha256':hashlib.sha256((out/'cuda-smoke.txt').read_bytes()).hexdigest(),
 'model_inference_verified':False,'source_unit_restarted':False}
with (out/'import.json').open('x') as f:json.dump(r,f);f.write('\n')
print(json.dumps(r),flush=True)
PY
