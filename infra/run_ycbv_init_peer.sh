#!/usr/bin/env bash
# Source closure: /src/world_reward/native_frame_map.py
# Source closure: /infra/ycbv_init_peer.py /infra/frontend_peer_receive.py
set +x
set -euo pipefail
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ && "$CODE" == "$ROOT/jobs/$REV/run_ycbv_init_peer/code" && "${BASH_SOURCE[0]}" == "$CODE/infra/run_ycbv_init_peer.sh" && "$(uname -s)" == Linux ]] || exit 2
exec /usr/bin/env -i PATH=/usr/bin:/bin HOME=/nonexistent WR_ROOT="$ROOT" WR_CODE="$CODE" WR_CODE_REVISION="$REV" \
 SSH_CONNECTION="${SSH_CONNECTION:-}" SSH_ORIGINAL_COMMAND="${SSH_ORIGINAL_COMMAND:-}" PYTHONDONTWRITEBYTECODE=1 \
 /usr/bin/python3 -I -B "$CODE/infra/ycbv_init_peer.py" "$@"
