#!/usr/bin/env bash
# Root installs this already-pinned wrapper as the narrow SSH forced command.
# Source closure: /infra/pose_peer_receive.py /infra/frontend_peer_receive.py /infra/pose_peer_inputs.py
set +x
set -euo pipefail
[[ $# == 2 && "$1" == --episode && "$2" =~ ^(0|[1-9]|[12][0-9])$ ]] || exit 2
ROOT="${WR_ROOT:?}";CODE="${WR_CODE:?}";REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_pose_peer_receive/code" && "${BASH_SOURCE[0]}" == "$CODE/infra/run_pose_peer_receive.sh" \
 && "$(hostname -s)" == world-reward-ncc-h100-02 ]] || exit 2
# Preserve only SSH peer metadata, never agent credentials or caller env.
exec env -i PATH=/usr/bin:/bin HOME=/nonexistent PYTHONDONTWRITEBYTECODE=1 \
 WR_ROOT="$ROOT" WR_CODE="$CODE" WR_CODE_REVISION="$REV" SSH_CONNECTION="${SSH_CONNECTION:-}" \
 SSH_ORIGINAL_COMMAND="${SSH_ORIGINAL_COMMAND:-}" python3 -I -B "$CODE/infra/pose_peer_receive.py" "$@"
