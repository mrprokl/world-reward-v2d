#!/usr/bin/env bash
# Remote durable host orchestrator: all models/data/outputs remain on Azure.
# Source closure: /infra/full4d_coverage.py /infra/terminal_success.py
set +x
set -euo pipefail
WAIT_FOR=''
if [[ $# == 6 && "$5" == --after-terminal && "$6" =~ ^world-reward-[a-z0-9][a-z0-9-]{0,80}(\.service)?$ ]]; then
 WAIT_FOR="${6%.service}.service"; set -- "$1" "$2" "$3" "$4"
fi
[[ $# == 4 && "$1" == --baseline-report-bytes && "$2" =~ ^[1-9][0-9]*$ \
  && "$3" == --baseline-report-sha256 && "$4" =~ ^[0-9a-f]{64}$ ]] || exit 2
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
  && "$CODE" == "$ROOT/jobs/$REV/run_full4d_coverage/code" \
  && "$(hostname)" == scenesmith-ncc-h100-01 && "$(id -u)" == 0 ]] || exit 2
if [[ -n "$WAIT_FOR" ]]; then
 source_identity() {
  env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$CODE/infra" python3 -B - "$ROOT" "$CODE" "$REV" <<'PYSOURCE'
from pathlib import Path
import sys
from mediapipe_cpu_runtime_verify import source
value=source(Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3],'run_full4d_coverage',
 ('infra/full4d_coverage.py','infra/run_full4d_coverage.sh','infra/terminal_success.py'))
print(value['closure_sha256'])
PYSOURCE
 }
 BEFORE="$(source_identity)"
 # Coverage is an independent experiment: prior preview technical failure
 # is NOT a scientific rejection. Wait for the ENTIRE producer to be terminal
 # (success OR explicitly coherent failure), never an inter-clip lease release.
 env PYTHONDONTWRITEBYTECODE=1 python3 -B "$CODE/infra/terminal_success.py" "$WAIT_FOR" --allow-failed-terminal
 [[ "$(source_identity)" == "$BEFORE" && ! -e "$ROOT/experiments/full4d-v1-$REV" \
  && ! -L "$ROOT/experiments/full4d-v1-$REV" ]] || exit 2
fi
exec /usr/bin/env -i PATH=/usr/bin:/bin:/usr/sbin:/sbin HOME=/nonexistent \
  DOCKER_HOST="unix://$ROOT/docker.sock" WR_ROOT="$ROOT" WR_CODE="$CODE" WR_CODE_REVISION="$REV" \
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$CODE/src:$CODE/infra" \
  /usr/bin/timeout --signal=TERM --kill-after=30s 28920s \
  /usr/bin/python3 -B "$CODE/infra/full4d_coverage.py" "$@"
