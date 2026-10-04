#!/usr/bin/env bash
# Source closure: /src/world_reward/native_frame_map.py
# Source closure: /infra/ycbv_point_masks.py /infra/ycbv_point_depth.py /infra/tudl_holdout_inputs.py
# Source closure: /infra/bridge_frontend_bindings.py /infra/frontend_selected_assets.py
# Source closure: /infra/frontend_sam2_kernel_gate.py /infra/hand_synthetic_masks.py
# Source closure: /configs/frontend_grounding_source_pins.json /configs/frontend_asset_archive_pins.json
set +x
set -euo pipefail
[[ $# == 0 ]] || exit 2
ROOT="${WR_ROOT:?}";CODE="${WR_CODE:?}";REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_ycbv_point_masks/code" \
 && "${BASH_SOURCE[0]}" == "$CODE/infra/run_ycbv_point_masks.sh" \
 && "$(hostname)" == world-reward-ncc-h100-02 && "$(id -u)" == 0 ]] || exit 2
export DOCKER_HOST="unix://$ROOT/docker.sock"
BASE="$ROOT/validation/ycbv_point_pose_v2";OUT="$BASE/automatic_masks_v1"
IMAGE=sha256:fd26863fd69d8fa1bb0bcc137bc7ddbee18fd5955484dcba672404a73326e252
NAME="world-reward-ycbv-point-masks-${REV:0:12}";LOCK="$ROOT/jobs/.world-reward-h100.lock"
CIDFILE='';LOCK_OPEN=0;LOCK_BEFORE=''
integrity() {
 /usr/bin/env -i PATH=/usr/bin:/bin HOME=/nonexistent WR_ROOT="$ROOT" WR_CODE="$CODE" WR_CODE_REVISION="$REV" PYTHONDONTWRITEBYTECODE=1 \
 /usr/bin/python3 -I -B - "$CODE" <<'PYPROOF'
import runpy,sys
from pathlib import Path
code=Path(sys.argv[1]);sys.path[:0]=[str(code/'infra'),str(code/'src')]
sys.argv=[str(code/'infra/ycbv_point_masks.py'),'--host-proof']
runpy.run_path(sys.argv[0],run_name='__main__')
PYPROOF
}
lock_identity() {
 /usr/bin/env -i PATH=/usr/bin:/bin /usr/bin/python3 -I -B - "$LOCK" "${1:-path}" <<'PYLOCK'
import os,stat,sys
from pathlib import Path
p=Path(sys.argv[1]);s=p.lstat()
if p.resolve()!=p or any(a.is_symlink()for a in(p,*p.parents))or not stat.S_ISREG(s.st_mode)or s.st_nlink!=1:raise ValueError('Existing canonical cooperative lock required')
if sys.argv[2]=='fd'and(os.fstat(9).st_dev,os.fstat(9).st_ino)!=(s.st_dev,s.st_ino):raise ValueError('Lock inode changed')
print(str(s.st_dev)+':'+str(s.st_ino))
PYLOCK
}
gpu_idle() {
 local apps
 apps="$(timeout --signal=TERM --kill-after=2s 5s nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)" || return 1
 [[ -z "${apps//[[:space:]]/}" ]]
}
query() { timeout --signal=TERM --kill-after=2s 5s docker ps -aq --no-trunc --filter "name=^/$NAME$"; }
BEFORE="$(integrity)";[[ "$BEFORE" =~ ^[0-9a-f]{64}$ ]]
finish() {
 local status=$? cid ids owned after
 trap - EXIT INT TERM;set +e
 if [[ -n "$CIDFILE" && -f "$CIDFILE" && ! -L "$CIDFILE" ]];then
  cid="$(cat "$CIDFILE")"
  if [[ "$cid" =~ ^[0-9a-f]{64}$ ]];then
   ids="$(timeout --signal=TERM --kill-after=2s 5s docker ps -aq --no-trunc --filter "id=$cid")";[[ $? == 0 ]] || status=1
   if [[ -n "$ids" ]];then
    owned="$(timeout --signal=TERM --kill-after=2s 5s docker inspect "$cid" --format '{{.Image}}|{{.Name}}|{{index .Config.Labels "world-reward.job"}}|{{index .Config.Labels "world-reward.revision"}}')"
    if [[ "$owned" == "$IMAGE|/$NAME|run_ycbv_point_masks|$REV" ]];then
     timeout --signal=TERM --kill-after=2s 15s docker rm -f "$cid" >/dev/null 2>&1 || status=1
     ids="$(timeout --signal=TERM --kill-after=2s 5s docker ps -aq --no-trunc --filter "id=$cid")";[[ $? == 0 && -z "$ids" ]] || status=1
    else status=1;fi
   fi
   chmod 400 "$CIDFILE"
  else status=1;fi
  gpu_idle || status=1
 fi
 after="$(integrity)";[[ $? == 0 && "$after" == "$BEFORE" ]] || status=1
 if [[ "$LOCK_OPEN" == 1 ]];then
  after="$(lock_identity fd)";[[ $? == 0 && "$after" == "$LOCK_BEFORE" ]] || status=1;exec 9<&-
 fi
 exit "$status"
}
trap finish EXIT;trap 'exit 130' INT;trap 'exit 143' TERM
[[ ! -e "$OUT" && ! -L "$OUT" && -d "$BASE" && -z "$(query)" ]] || exit 1
LOCK_BEFORE="$(lock_identity)";exec 9<"$LOCK";LOCK_OPEN=1
[[ "$(lock_identity fd)" == "$LOCK_BEFORE" ]] || exit 1
flock --nonblock 9 || exit 1;gpu_idle || exit 1
[[ "$BEFORE" == "$(integrity)" ]] || exit 1
umask 077;mkdir -m 700 "$OUT";CIDFILE="$OUT/.container.cid"
DEST=/srv/world-reward-data/frontend-assets-extracted-75fdea08fb3b43f62f1c4b4f5e646674595cdd0b
BUILD="$ROOT/jobs/d1fcb8ad158cf4ad4b1f7fe2bba7ea533ae565cc/run_frontend_grounding_build"
KERNEL="$ROOT/jobs/168a809369a294c1ab65154a585b83014c98d976/run_frontend_sam2_kernel_gate"
MOUNTS=()
# Blind import closure only; full own/archive source is authenticated on host.
for relative in infra/ycbv_point_masks.py infra/run_ycbv_point_masks.sh infra/ycbv_point_depth.py \
 infra/bridge_frontend_bindings.py infra/frontend_selected_assets.py infra/frontend_sam2_kernel_gate.py \
 infra/hand_synthetic_masks.py infra/tudl_holdout_inputs.py src/world_reward/__init__.py src/world_reward/native_frame_map.py src/world_reward/prompt_selection.py \
 configs/frontend_grounding_source_pins.json configs/frontend_asset_archive_pins.json configs/ycbv_point_input_pins_v2.json;do
 path="$CODE/$relative";[[ -f "$path" && ! -L "$path" ]] || exit 1
 MOUNTS+=(--mount "type=bind,src=$path,dst=$path,readonly")
done
for path in "${CODE%/code}/revision" "${CODE%/code}/source-sha256" "$BUILD" "$KERNEL" \
 "$ROOT/results/frontend-grounding-build-v6/report.json" "$ROOT/results/frontend-grounding-build-v6/child-CPU-probe.log" \
 "$ROOT/results/frontend-sam2-kernel-gate-v1/report.json" \
 "$ROOT/results/frontend-asset-extract-75fdea08fb3b43f62f1c4b4f5e646674595cdd0b/report.json" \
 "$DEST/world-reward-frontend-assets-manifest.json" "$DEST/results/weights-acquisition.json";do
 [[ -e "$path" && ! -L "$path" ]] || exit 1
 MOUNTS+=(--mount "type=bind,src=$path,dst=$path,readonly")
done
# Only nine selected model files: no Body/Object checkpoints or complete 19.9GB tree.
for relative in grounding_dino/model.safetensors grounding_dino/config.json grounding_dino/preprocessor_config.json \
 grounding_dino/special_tokens_map.json grounding_dino/tokenizer.json grounding_dino/tokenizer_config.json grounding_dino/vocab.txt \
 sam2/sam2.1_hiera_large.pt sam2/sam2.1_hiera_l.yaml;do
 path="$DEST/weights/$relative";[[ -f "$path" && ! -L "$path" ]] || exit 1
 MOUNTS+=(--mount "type=bind,src=$path,dst=$path,readonly")
done
[[ -d "$BASE/inputs" && ! -L "$BASE/inputs" ]] || exit 1
MOUNTS+=(--mount "type=bind,src=$BASE/inputs,dst=$BASE/inputs,readonly")
timeout --signal=TERM --kill-after=10s 630s docker run --rm --name "$NAME" --cidfile "$CIDFILE" \
 --label world-reward.job=run_ycbv_point_masks --label "world-reward.revision=$REV" \
 --gpus all --network none --user 0:0 --memory 32g --cpus 4 --read-only --cap-drop ALL --cap-add DAC_OVERRIDE --security-opt no-new-privileges \
 --tmpfs /tmp:rw,noexec,nosuid,size=512m --entrypoint /usr/bin/env \
 "${MOUNTS[@]}" --mount "type=bind,src=$OUT,dst=$OUT" "$IMAGE" \
 -i PATH=/opt/conda/bin:/usr/bin:/bin HOME=/tmp XDG_CACHE_HOME=/tmp PYTHONPATH="$CODE/infra:$CODE/src" \
 WR_ROOT="$ROOT" WR_CODE="$CODE" WR_CODE_REVISION="$REV" WR_IMAGE_ID="$IMAGE" WR_YCBV_HOST_PROOF_SHA256="$BEFORE" \
 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled \
 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
 /opt/conda/bin/python -B "$CODE/infra/ycbv_point_masks.py"
