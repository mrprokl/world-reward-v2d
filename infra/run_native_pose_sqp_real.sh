#!/usr/bin/env bash
# Source closure: /infra/native_pose_sqp_real.py
set +x
set -euo pipefail
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; REV="${WR_CODE_REVISION:?}"
[[ $# -eq 0 ]] || exit 2
TRACK_SOURCE=09f516d9085f0d64d725b46b0ad6aa52fe7c0d84
SOURCE=052ba1554e9a573d566713a99a61d89a5f27681c
CONTACT_SOURCE=40183b3a83ba59080c192c3cdf2db9e1d76ef021
B_SOURCE=ebcee73cc76e036b3c2ec9c32e9a72abc951ad78
FRONTEND=b658881079871508c6b3ec14d001dc1299956996
IMAGE=sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_native_pose_sqp_real/code" \
 && "$(hostname)" == scenesmith-ncc-h100-01 && "$(id -u)" == 0 ]] || exit 2
OUT="$ROOT/results/native-pose-sqp-real-$REV"
[[ ! -e "$OUT" ]] || exit 2
# Exclusive original GPU lease; never truncate/mutate its contents.
exec 8<"$ROOT/jobs/.world-reward-h100.lock"
flock -n 8
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]] || exit 2
mkdir -m 755 "$OUT"
export DOCKER_HOST="unix://$ROOT/docker.sock"
NAME="wr-native-pose-sqp-real-$REV"; CIDFILE="$OUT/.container.cid"
cleanup() {
 local status=$? cid='' ids='' found='' verified=false
 trap - EXIT TERM INT
 if [[ -f "$CIDFILE" && ! -L "$CIDFILE" ]]; then
  cid="$(cat "$CIDFILE")"
  if [[ "$cid" =~ ^[0-9a-f]{64}$ ]]; then
   if ids="$(timeout 20s docker ps -aq --no-trunc --filter "id=$cid")" \
    && [[ -z "$ids" || "$ids" == "$cid" ]]; then
    if [[ -n "$ids" ]]; then
     if found="$(timeout 20s docker inspect "$cid" --format '{{.Image}}|{{.Name}}|{{index .Config.Labels "world_reward.native_pose_sqp_real.owner"}}')" \
      && [[ "$found" == "$IMAGE|/$NAME|$REV" ]]; then
      timeout 20s docker rm -f "$cid" >/dev/null || status=1
     else status=1; fi
    fi
    if ids="$(timeout 20s docker ps -aq --no-trunc --filter "id=$cid")" \
     && [[ -z "$ids" ]]; then verified=true; else status=1; fi
   else status=1; fi
  else status=1; fi
  chmod 444 "$CIDFILE"
 else
  # Before a CID is written, do not stop an unproven name-matching container.
  if ids="$(timeout 20s docker ps -aq --no-trunc --filter "name=^/$NAME$")" \
   && [[ -z "$ids" ]]; then verified=true; else status=1; fi
 fi
 printf '{"producer_revision":"%s","container_absence_verified":%s,"process_exit_code":%d,"GPU_requested":true}\n' \
  "$REV" "$verified" "$status" > "$OUT/host-exit.json"
 chmod 444 "$OUT/host-exit.json"
 exit "$status"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
[[ -z "$(docker ps -aq --no-trunc --filter "name=^/$NAME$")" ]] || exit 2
timeout --signal=TERM --kill-after=10s 903s docker run --rm --cidfile "$CIDFILE" \
 --name "$NAME" --label "world_reward.native_pose_sqp_real.owner=$REV" --gpus all \
 --network none --read-only --user 0:0 --cap-drop ALL --security-opt no-new-privileges \
 --memory 64g --cpus 4 --tmpfs /tmp:rw,nosuid,exec,size=1g \
 --mount "type=bind,src=${CODE%/code},dst=${CODE%/code},readonly" \
 --mount "type=bind,src=$ROOT/experiments/full4d-v1-$SOURCE,dst=$ROOT/experiments/full4d-v1-$SOURCE,readonly" \
 --mount "type=bind,src=$ROOT/results/gemini-sam31-$FRONTEND,dst=$ROOT/results/gemini-sam31-$FRONTEND,readonly" \
 --mount "type=bind,src=$ROOT/results/sequence-pose-probe-$TRACK_SOURCE-v2,dst=$ROOT/results/sequence-pose-probe-$TRACK_SOURCE-v2,readonly" \
 --mount "type=bind,src=$ROOT/weights/cari4d/refinement/mhr_hand_surface_spec.npz,dst=$ROOT/weights/cari4d/refinement/mhr_hand_surface_spec.npz,readonly" \
 --mount "type=bind,src=$ROOT/results/sequence-pose-contact-$CONTACT_SOURCE,dst=$ROOT/results/sequence-pose-contact-$CONTACT_SOURCE,readonly" \
 --mount "type=bind,src=$ROOT/jobs/$B_SOURCE/run_native_joint_real,dst=$ROOT/jobs/$B_SOURCE/run_native_joint_real,readonly" \
 --mount "type=bind,src=$ROOT/results/native-joint-real-$B_SOURCE,dst=$ROOT/results/native-joint-real-$B_SOURCE,readonly" \
 --mount "type=bind,src=$ROOT/data/track_1/videos/chunk-000/observation.images.exo_camera/episode_000009.mp4,dst=$ROOT/data/track_1/videos/chunk-000/observation.images.exo_camera/episode_000009.mp4,readonly" \
 --mount "type=bind,src=$ROOT/data/track_1/videos/chunk-000/observation.images.exo_camera/episode_000014.mp4,dst=$ROOT/data/track_1/videos/chunk-000/observation.images.exo_camera/episode_000014.mp4,readonly" \
 --mount "type=bind,src=$ROOT/vendor/video_to_data,dst=$ROOT/vendor/video_to_data,readonly" \
 --mount "type=bind,src=$ROOT/weights/cari4d/sam3d_body,dst=$ROOT/weights/cari4d/sam3d_body,readonly" \
 --mount "type=bind,src=$ROOT/weights/cari4d/refinement,dst=$ROOT/weights/cari4d/refinement,readonly" \
 --mount "type=bind,src=$ROOT/results/cari-refinement-assets.json,dst=$ROOT/results/cari-refinement-assets.json,readonly" \
 --mount "type=bind,src=$ROOT/results/weights-acquisition.json,dst=$ROOT/results/weights-acquisition.json,readonly" \
 --mount "type=bind,src=$OUT,dst=$OUT" \
 --entrypoint /usr/bin/env "$IMAGE" \
 -i PATH=/opt/conda/bin:/usr/bin:/bin HOME=/tmp PYTHONDONTWRITEBYTECODE=1 \
 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MOMENTUM_ENABLED=0 \
 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled \
 WR_ROOT="$ROOT" WR_CODE="$CODE" WR_CODE_REVISION="$REV" WR_IMAGE_ID="$IMAGE" PYTHONPATH="$CODE/src:$CODE/infra:/workspace/v2d_sam3d_body/lib" \
 /opt/conda/bin/python -B "$CODE/infra/native_pose_sqp_real.py"
