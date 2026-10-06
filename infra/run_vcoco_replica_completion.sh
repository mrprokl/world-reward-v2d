#!/usr/bin/env bash
# Closure: /infra/vcoco_replica_completion.py /infra/sealed_callback_publication.py
# Complete only exact installed-byte cleanup; no import, download, GPU or model.
set +x
set -euo pipefail
[[ $# == 0 ]] || exit 2
ROOT="${WR_ROOT:?}";CODE="${WR_CODE:?}";REV="${WR_CODE_REVISION:?}"
[[ "$(uname -s)" == Linux && "$(id -u)" == 0 && "$(hostname)" == scenesmith-ncc-h100-01 \
 && "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_vcoco_replica_completion/code" \
 && "${BASH_SOURCE[0]}" == "$CODE/infra/run_vcoco_replica_completion.sh" ]] || exit 2
umask 077
ulimit -v 262144
exec timeout --signal=TERM --kill-after=5s 195s env -i PATH=/usr/bin:/bin HOME=/nonexistent \
 WR_ROOT="$ROOT" WR_CODE="$CODE" WR_CODE_REVISION="$REV" PYTHONDONTWRITEBYTECODE=1 \
 /usr/bin/python3 -I -B "$CODE/infra/vcoco_replica_completion.py"
