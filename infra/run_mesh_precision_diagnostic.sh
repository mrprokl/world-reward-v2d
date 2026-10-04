#!/usr/bin/env bash
set -euo pipefail
[[ $# == 2 && $1 == --episode && $2 =~ ^(0|[1-9]|[12][0-9])$ ]] || exit 2
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; REV="${WR_CODE_REVISION:?}"; EPISODE="$2"
export DOCKER_HOST="unix://$ROOT/docker.sock"
# Source closure: /infra/mesh_precision_diagnostic.py /infra/object_budget_endpoint.py
# Source closure: /infra/mesh_link_gate.py /infra/mesh_endpoint_gate.py
PIN="$CODE/configs/mesh_precision_diagnostic_v1.json"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$CODE" == "$ROOT/jobs/$REV/run_mesh_precision_diagnostic/code" && "${BASH_SOURCE[0]}" == "$CODE/infra/run_mesh_precision_diagnostic.sh" ]] || exit 2
mapfile -t VALUES < <(python3 - "$PIN" "$EPISODE" "$ROOT" <<'PYHOST'
import json,re,sys,stat,hashlib
from pathlib import Path
root=Path(sys.argv[3]);p=Path(sys.argv[1]);code=p.parents[1];rev=code.parent.parent.name
assert str(root)=='/srv/scenesmith/world-reward' and re.fullmatch('[0-9a-f]{40}',rev)
assert code==root/'jobs'/rev/'run_mesh_precision_diagnostic'/'code'
assert code.parent.joinpath('revision').read_bytes()==(rev+'\n').encode()
assert re.fullmatch(b'[0-9a-f]{64}\n',code.parent.joinpath('source-sha256').read_bytes())
for f in (code,code.parent/'revision',code.parent/'source-sha256',*sorted(code.rglob('*'))):
 assert f.resolve()==f and not any(x.is_symlink() for x in(f,*f.parents))
 st=f.lstat();assert not st.st_mode&0o222 and (stat.S_ISREG(st.st_mode) or stat.S_ISDIR(st.st_mode))
 if stat.S_ISREG(st.st_mode):
  assert st.st_nlink==1
  raw=f.read_bytes();after=f.lstat();assert all(getattr(st,k)==getattr(after,k)for k in('st_dev','st_ino','st_mode','st_size','st_mtime_ns','st_ctime_ns'))
  hashlib.sha256(raw).hexdigest()
assert not p.is_symlink();d=json.loads(p.read_text())
assert d['schema']=='world-reward-mesh-precision-diagnostic-pins-v1' and type(d['episode_index'])is int and d['episode_index']==int(sys.argv[2])
assert re.fullmatch('sha256:[0-9a-f]{64}',d['diagnostic_image_id'])
assert set(d['files'])=={'glb','object_report','original_producer','image_evidence','failed_report'}
print(d['diagnostic_image_id'])
for k in sorted(d['files']):
 r=d['files'][k];q=Path(r['path']);assert not q.is_absolute() and '..'not in q.parts and str(q)==r['path'] and ','not in r['path'] and '\n'not in r['path']
 f=Path(sys.argv[3])/q;assert f.is_file() and not any(x.is_symlink()for x in(f,*f.parents));print(f)
PYHOST
)
[[ ${#VALUES[@]} == 6 && $REV =~ ^[0-9a-f]{40}$ ]]
IMAGE="${VALUES[0]}"
[[ "$(docker image inspect --format '{{.Id}}' "$IMAGE")" == "$IMAGE" ]]
printf -v PADDED '%06d' "$EPISODE"
OUT="$ROOT/diagnostic/episode${PADDED}-mesh-precision-v1"
[[ ! -L "$ROOT" && ! -L "$ROOT/diagnostic" && ! -e "$OUT" && ! -L "$OUT" ]]
mkdir -p "$ROOT/diagnostic"; mkdir -m 700 "$OUT"
MOUNTS=()
for ((i=1;i<${#VALUES[@]};i++)); do MOUNTS+=(--mount "type=bind,src=${VALUES[i]},dst=${VALUES[i]},readonly"); done
NAME="world-reward-mesh-precision-${REV:0:12}-$EPISODE"
# CID is host-only; do not expose or read it in the diagnostic container.
CIDDIR="$(mktemp -d "$ROOT/diagnostic/mesh-precision-cid.XXXXXXXX")"
cleanup() {
 local cid
 if [[ -f "$CIDDIR/container.cid" ]]; then
  cid="$(cat "$CIDDIR/container.cid")"
  if [[ $cid =~ ^[0-9a-f]{64}$ ]] && [[ "$(docker inspect --format '{{.Config.Labels.world_reward_revision}} {{.Image}} {{.Name}}' "$cid" 2>/dev/null || true)" == "$REV $IMAGE /$NAME" ]]; then docker rm -f "$cid" >/dev/null; fi
 fi
 rm -f "$CIDDIR/container.cid"; rmdir "$CIDDIR"
}
trap cleanup EXIT INT TERM
timeout --signal=TERM --kill-after=5s 60s docker run --rm \
 --name "$NAME" --cidfile "$CIDDIR/container.cid" --label "world_reward_revision=$REV" \
 --network none --read-only --cap-drop ALL --security-opt no-new-privileges \
 --memory 8g --cpus 4 --user 0:0 --entrypoint python --tmpfs /tmp:rw,nosuid,nodev,size=128m \
 --env WR_ROOT="$ROOT" --env WR_CODE_REVISION="$REV" --env WR_IMAGE_ID="$IMAGE" \
 --env PYTHONPATH="$CODE/src" --env PYTHONDONTWRITEBYTECODE=1 \
 --env OPENBLAS_NUM_THREADS=1 --env OMP_NUM_THREADS=1 --env MKL_NUM_THREADS=1 \
 --env HOME=/tmp --env XDG_CACHE_HOME=/tmp \
 --mount "type=bind,src=$CODE,dst=$CODE,readonly" \
 --mount "type=bind,src=${CODE%/code}/revision,dst=${CODE%/code}/revision,readonly" \
 --mount "type=bind,src=${CODE%/code}/source-sha256,dst=${CODE%/code}/source-sha256,readonly" \
 "${MOUNTS[@]}" --mount "type=bind,src=$OUT,dst=$OUT" \
 "$IMAGE" "$CODE/infra/mesh_precision_diagnostic.py" "$@"
