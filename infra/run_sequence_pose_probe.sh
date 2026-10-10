#!/usr/bin/env bash
set +x
set -euo pipefail
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$CODE" == "$ROOT/jobs/$REV/run_sequence_pose_probe/code" ]] || exit 2
OUT="$ROOT/results/sequence-pose-probe-$REV"
[[ ! -e "$OUT" ]] || exit 2
mkdir -m 755 "$OUT"
exec 9>"$ROOT/jobs/.world-reward-h100.lock"
flock -n 9
export DOCKER_HOST="unix://$ROOT/docker.sock"
exec timeout --signal=TERM --kill-after=10s 1200s docker run --rm \
  --name "wr-sequence-pose-$REV" --label "world_reward.sequence_pose.owner=$REV" \
  --network none --read-only --user 0:0 --cap-drop ALL --security-opt no-new-privileges \
  --memory 16g --cpus 4 --tmpfs /tmp:rw,nosuid,size=256m \
  --mount "type=bind,src=$CODE,dst=$CODE,readonly" \
  --mount "type=bind,src=$ROOT/experiments,dst=$ROOT/experiments,readonly" \
  --mount "type=bind,src=$ROOT/data/track_1,dst=$ROOT/data/track_1,readonly" \
  --mount "type=bind,src=$OUT,dst=$OUT" \
  --entrypoint /usr/bin/env sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7 \
  -i PATH=/opt/conda/bin:/usr/bin:/bin HOME=/tmp PYTHONDONTWRITEBYTECODE=1 \
  OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=-1 \
  WR_ROOT="$ROOT" WR_CODE="$CODE" WR_CODE_REVISION="$REV" PYTHONPATH="$CODE/src:$CODE/infra" \
  /opt/conda/bin/python -B "$CODE/infra/sequence_pose_probe.py"
