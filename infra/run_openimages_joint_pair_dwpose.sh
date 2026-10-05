#!/usr/bin/env bash
# New external ALL-person CPU crops; projected RGB/GDI only, no GPU requests.
# Source closure: /infra/openimages_joint_pair_dwpose.py /infra/dwpose_smoke.py
# Source closure: /infra/joint_pair_dwpose_acquire.py /infra/dwpose_acquire.py
# Source closure: /infra/dwpose_wheel_audit.py /infra/mediapipe_cpu_runtime_verify.py
set +x
set -euo pipefail
[[ $# == 0 ]] || exit 2
ROOT="${WR_ROOT:?}";CODE="${WR_CODE:?}";REV="${WR_CODE_REVISION:?}"
[[ "$(id -u)" == 0 && "$(hostname)" == world-reward-ncc-h100-02 \
 && "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_openimages_joint_pair_dwpose/code" \
 && "${BASH_SOURCE[0]}" == "$CODE/infra/run_openimages_joint_pair_dwpose.sh" ]] || exit 2
export DOCKER_HOST="unix://$ROOT/docker.sock"
OUT="$ROOT/results/openimages-joint-pair-dwpose-v1"
NAME="world-reward-oi-joint-dwpose-${REV:0:12}"
IMAGE=sha256:7ebfff18ba3b76dd919485c19115597d7531dfd3233f69461f1dce3f28a6c6d3
HOSTENV=(/usr/bin/env -i PATH=/usr/bin:/bin HOME=/nonexistent WR_ROOT="$ROOT" WR_CODE="$CODE" WR_CODE_REVISION="$REV" PYTHONDONTWRITEBYTECODE=1)
control() { timeout --signal=TERM --kill-after=2s 8s "$@"; }
query() { control docker ps -aq --no-trunc --filter "name=^/$NAME$"; }
ACTUAL_IMAGE="$(control docker image inspect "$IMAGE" --format '{{.Id}}')"
[[ "$ACTUAL_IMAGE" == "$IMAGE" && -z "$(query)" && ! -e "$OUT" && ! -L "$OUT" ]] || exit 1
OWNED=0
cleanup() {
 local ids
 [[ "$OWNED" == 1 ]] || return 0
 ids="$(query)" || return 1
 if [[ -n "$ids" ]];then
  control docker stop --time 3 "$NAME" >/dev/null 2>&1 || true
  ids="$(query)" || return 1
  if [[ -n "$ids" ]];then
   control docker kill "$NAME" >/dev/null 2>&1 || true
   control docker rm --force "$NAME" >/dev/null 2>&1 || true
  fi
 fi
 ids="$(query)" || return 1;[[ -z "$ids" ]]
}
finish() {
 local status=$?
 trap - EXIT INT TERM;set +e
 cleanup || status=1
 exit "$status"
}
trap finish EXIT;trap 'exit 130' INT;trap 'exit 143' TERM
PROJECTED="$("${HOSTENV[@]}" /usr/bin/python3 -I -B "$CODE/infra/openimages_joint_pair_dwpose.py" --prepare)"
read -r INPUT_SHA INPUT_BYTES <<<"$PROJECTED"
[[ "$INPUT_SHA" =~ ^[0-9a-f]{64}$ && "$INPUT_BYTES" =~ ^[0-9]+$ ]] || exit 1
MOUNTS=()
for path in "${CODE%/code}" "$ROOT/weights/dwpose_joint_pair_v1" \
 "$ROOT/results/dwpose-joint-pair-acquisition-v1.json" \
 "$ROOT/jobs/37a6d39045db215626b8a0268c1357019835af59/run_joint_pair_dwpose_acquire" \
 "$OUT/inputs.json";do
 [[ -e "$path" && ! -L "$path" ]] || exit 1
 MOUNTS+=(--mount "type=bind,src=$path,dst=$path,readonly")
done
PATHS="$("${HOSTENV[@]}" /usr/bin/python3 -I -B - "$OUT/inputs.json" <<'PY'
import json,re,sys
from pathlib import Path
data=json.load(open(sys.argv[1]));assert set(data)=={'schema','images'}and len(data['images'])==50
for row in data['images']:
 assert set(row)=={'image_id','file','width','height','bytes','sha256','decoded_rgb_sha256','person_ids','gdi_file','gdi_identity'}and re.fullmatch('[0-9a-f]{16}',row['image_id'])
 paths=(Path('/srv/world-reward-data/openimages_joint_pair_acquisition_v1')/row['image_id']/'rgb.jpg',
        Path('/srv/scenesmith/world-reward/results/openimages-joint-pair-gdi-v2/native')/(row['image_id']+'.npz'))
 assert row['file']==str(paths[0])and row['gdi_file']==str(paths[1])
 for p in paths:
  assert p.resolve()==p and not any(a.is_symlink()for a in(p,*p.parents));print(p)
PY
)"
while IFS= read -r path;do
 [[ -f "$path" && ! -L "$path" ]] || exit 1
 MOUNTS+=(--mount "type=bind,src=$path,dst=$path,readonly")
done <<<"$PATHS"
OWNED=1
set +e
timeout --signal=TERM --kill-after=10s 615s docker run --rm --name "$NAME" --network none \
 --user 0:0 --read-only --memory 8g --cpus 4 --cap-drop ALL --security-opt no-new-privileges \
 --tmpfs /tmp:rw,exec,nosuid,size=512m --entrypoint /usr/bin/env \
 "${MOUNTS[@]}" --mount "type=bind,src=$OUT/native,dst=$OUT/native" "$IMAGE" \
 -i PATH=/opt/conda/bin:/usr/bin:/bin HOME=/tmp PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= \
 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled \
 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
 WR_ROOT="$ROOT" WR_CODE="$CODE" WR_CODE_REVISION="$REV" WR_IMAGE_ID="$IMAGE" \
 /opt/conda/bin/python -I -B "$CODE/infra/openimages_joint_pair_dwpose.py" --native \
 --inputs-sha256 "$INPUT_SHA" --inputs-bytes "$INPUT_BYTES"
RESULT=$?
set -e
cleanup
[[ "$(control docker image inspect "$IMAGE" --format '{{.Id}}')" == "$IMAGE" ]] || exit 1
"${HOSTENV[@]}" /usr/bin/python3 -I -B "$CODE/infra/openimages_joint_pair_dwpose.py" --finalize "$RESULT"
