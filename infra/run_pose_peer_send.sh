#!/usr/bin/env bash
# Exact private SSH transport only; no setup/keys/cloud calls or media locally.
# Source closure: /infra/pose_peer_send.py /infra/pose_peer_inputs.py
set +x
set -euo pipefail
ROOT="${WR_ROOT:?}";CODE="${WR_CODE:?}";REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_pose_peer_send/code" && "${BASH_SOURCE[0]}" == "$CODE/infra/run_pose_peer_send.sh" \
 && "$(hostname -s)" == scenesmith-ncc-h100-01 ]] || exit 2
exec env -i PATH=/usr/bin:/bin HOME=/nonexistent PYTHONDONTWRITEBYTECODE=1 \
 WR_ROOT="$ROOT" WR_CODE="$CODE" WR_CODE_REVISION="$REV" timeout --signal=TERM --kill-after=10s 1800s \
 nice -n 10 python3 -I -B "$CODE/infra/pose_peer_send.py" "$@"
