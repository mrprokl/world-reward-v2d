#!/usr/bin/env bash
# Source closure: /infra/person_pose_bank_probe.py /infra/dwpose_smoke.py
# Source closure: /infra/keypoint_rgb_dwpose.py /infra/mediapipe_cpu_runtime_verify.py
# All heavy images/model processing remains offline on Azure, CPU only.
set +x
set -euo pipefail
[[ $# == 0 ]] || exit 2
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; REV="${WR_CODE_REVISION:?}"
[[ "$(uname -s)" == Linux && "$ROOT" == /srv/scenesmith/world-reward \
 && "$REV" =~ ^[0-9a-f]{40}$ && "$CODE" == "$ROOT/jobs/$REV/run_person_pose_bank_probe/code" ]] || exit 2
export DOCKER_HOST="unix://$ROOT/docker.sock"
IMAGE=sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7
OUT="$ROOT/results/person-pose-bank-probe-$REV"; NAME="wr-person-pose-bank-$REV"
ACTUAL_IMAGE="$(timeout --signal=TERM --kill-after=2s 15s docker image inspect "$IMAGE" --format '{{.Id}}')" || exit 2
IDS="$(timeout --signal=TERM --kill-after=2s 15s docker ps -aq --filter "name=^/$NAME$")" || exit 2
[[ "$ACTUAL_IMAGE" == "$IMAGE" && ! -e "$OUT" && -z "$IDS" ]] || exit 2
MOUNTS=(--mount "type=bind,src=$CODE,dst=$CODE,readonly")
JOB="${CODE%/*}"
for marker in revision source-sha256;do
 [[ -f "$JOB/$marker" && ! -L "$JOB/$marker" ]] || exit 2
 MOUNTS+=(--mount "type=bind,src=$JOB/$marker,dst=$JOB/$marker,readonly")
done
PATH_LIST="$(python3 -I -B - "$CODE/configs/person_pose_bank_probe_v1.json" "$ROOT" <<'PYPATHS'
import json,sys
from pathlib import Path
p=json.load(open(sys.argv[1]));root=Path(sys.argv[2]);names=set(p['metadata_files'])
assert names=={'results/input-manifest.json','data/track_1/meta/episodes.jsonl'}
assert [x['episode']for x in p['episodes']]==[9,26]
for row in p['episodes']:
 e=row['episode'];base=f'outputs/episode_{e:06d}/automatic_masks'
 expected={f'data/track_1/videos/chunk-000/observation.images.exo_camera/episode_{e:06d}.mp4',base+'/report.json',base+'/seed-diagnostics.json'}
 assert set(row['files'])==expected
 names.update(row['files'])
for name in sorted(names):
 path=root/name
 if path.resolve()!=path or any(x.is_symlink()for x in(path,*path.parents)):raise ValueError('Canonical original inputs required')
 print(path)
PYPATHS
)" || exit 2
while IFS= read -r path; do
 [[ -f "$path" && ! -L "$path" ]] || exit 2
 MOUNTS+=(--mount "type=bind,src=$path,dst=$path,readonly")
done <<<"$PATH_LIST"
for path in "$ROOT/weights/dwpose_native_v1" "$ROOT/results/dwpose-wheel-audit-v3" \
 "$ROOT/results/dwpose-acquisition-v1.json" "$ROOT/results/dwpose-wheel-audit-v2/report.json" \
 "$ROOT/validation/dwpose_smoke_v1/report.json" "$ROOT/validation/dwpose_smoke_v2/report.json"; do
 [[ -e "$path" && ! -L "$path" ]] || exit 2
 MOUNTS+=(--mount "type=bind,src=$path,dst=$path,readonly")
done
mkdir -m 755 "$OUT"; chown "$(id -u scenesmith):$(id -g scenesmith)" "$OUT"
cleanup() {
 if [[ -f "$OUT/.container.cid" ]];then
  CID="$(cat "$OUT/.container.cid")"
  [[ "$CID" =~ ^[0-9a-f]{64}$ ]] || exit 2
  if timeout --signal=TERM --kill-after=2s 15s docker inspect "$CID" >/dev/null 2>&1;then
   ACTUAL="$(timeout --signal=TERM --kill-after=2s 15s docker inspect "$CID" --format '{{.Image}}|{{.Name}}')" || exit 2
   [[ "$ACTUAL" == "$IMAGE|/$NAME" ]] || exit 2
   timeout --signal=TERM --kill-after=2s 15s docker rm -f "$CID" >/dev/null
  fi
 fi
 IDS="$(timeout --signal=TERM --kill-after=2s 15s docker ps -aq --filter "name=^/$NAME$")" || exit 2
 [[ -z "$IDS" ]] || exit 2
 for path in "$OUT/.container.cid" "$OUT/native.log";do
  [[ ! -e "$path" ]] || chmod 444 "$path"
 done
}
trap cleanup EXIT
timeout --signal=TERM --kill-after=10s 320s docker run --rm --name "$NAME" --cidfile "$OUT/.container.cid" \
 --network none --read-only --cpus 4 --memory 8g --cap-drop ALL --security-opt no-new-privileges \
 --user "$(id -u scenesmith):$(id -g scenesmith)" --tmpfs /tmp:rw,exec,nosuid,size=512m \
 --entrypoint /usr/bin/env "${MOUNTS[@]}" --mount "type=bind,src=$OUT,dst=$OUT" "$IMAGE" -i \
 PATH=/opt/conda/bin:/usr/bin:/bin HOME=/tmp CUDA_VISIBLE_DEVICES= \
 "WR_ROOT=$ROOT" "WR_CODE=$CODE" "WR_CODE_REVISION=$REV" "WR_IMAGE_ID=$IMAGE" \
 "PYTHONPATH=$CODE/src:$CODE/infra" PYTHONDONTWRITEBYTECODE=1 \
 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled \
 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
 /opt/conda/bin/python -B "$CODE/infra/person_pose_bank_probe.py" >"$OUT/native.log" 2>&1
cleanup
