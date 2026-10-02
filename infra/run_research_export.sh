#!/usr/bin/env bash
# Task-only immutable export on VM01. Never snapshot Docker state/user disks.
set -euo pipefail
(( $# == 0 )) || exit 2
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ ]]
OUT="$ROOT/transfer/vm02-v1"
[[ -d "$OUT" && ! -L "$OUT" && ! -e "$OUT/assets.tar" && ! -e "$OUT/image.tar" && ! -e "$OUT/export.json" ]]
export DOCKER_HOST="unix://$ROOT/docker.sock"
EXPECTED=sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7
[[ "$(docker image inspect world-reward/cari4d-source:0.1 --format '{{.Id}}')" == "$EXPECTED" ]]
umask 077
timeout --signal=TERM --kill-after=5s 610s nice -n 10 ionice -c 3 python3 "$CODE/infra/research_transfer.py" \
 export --archive "$OUT/assets.tar" --include-private > "$OUT/assets-sha.json"
timeout --signal=TERM --kill-after=10s 1800s nice -n 10 ionice -c 3 docker image save \
 --output "$OUT/image.tar" world-reward/cari4d-source:0.1
[[ "$(docker image inspect world-reward/cari4d-source:0.1 --format '{{.Id}}')" == "$EXPECTED" ]]
python3 - "$OUT" "$REV" "$EXPECTED" <<'PY'
import hashlib,json,pathlib,sys,time
out=pathlib.Path(sys.argv[1]); start=time.monotonic()
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
assets=json.loads((out/'assets-sha.json').read_text())['archive_sha256']
report={'stage':'world_reward_task_only_export','status':'pass','producer_revision':sys.argv[2],
 'image_id':sys.argv[3],'image_tag':'world-reward/cari4d-source:0.1','docker_state_copied':False,
 'user_disk_copied':False,'secrets_transferred':False,'challenge_inputs_transferred':False,
 'artifacts':[{'file':n,'sha256':sha(out/n),'bytes':(out/n).stat().st_size} for n in ('assets.tar','image.tar')]}
assert report['artifacts'][0]['sha256']==assets
with (out/'export.json').open('x') as f:json.dump(report,f);f.write('\n')
print(json.dumps(report),flush=True)
PY
