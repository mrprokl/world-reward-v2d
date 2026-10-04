#!/usr/bin/env bash
# Owned expiring private listener only; no Docker, cloud API or payload extraction.
# Source closure: /infra/pose_peer_server.py /infra/pose_peer_inputs.py
# /infra/run_pose_peer_receive.sh /infra/pose_peer_receive.py /infra/frontend_peer_receive.py
set +x
set -euo pipefail
ROOT="${WR_ROOT:?}";CODE="${WR_CODE:?}";REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_pose_peer_server/code" && "${BASH_SOURCE[0]}" == "$CODE/infra/run_pose_peer_server.sh" \
 && "$(hostname -s)" == world-reward-ncc-h100-02 && "$(uname -s)" == Linux ]] || exit 2
exec env -i PATH=/usr/bin:/bin HOME=/nonexistent WR_ROOT="$ROOT" WR_CODE="$CODE" WR_CODE_REVISION="$REV" \
 PYTHONDONTWRITEBYTECODE=1 timeout --signal=TERM --kill-after=10s 600s python3 -I -B "$CODE/infra/pose_peer_server.py" "$@"
