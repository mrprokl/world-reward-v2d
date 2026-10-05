#!/usr/bin/env bash
# Closure: /infra/vcoco_role_prepare.py /infra/run_vcoco_role_prepare.sh /configs/vcoco_role_pilot_v1.json
# Closure: /infra/vcoco_role_census.py /infra/metadata_json_stream.py
# Closure: /infra/mediapipe_cpu_runtime_verify.py /infra/openimages_joint_pair_acquire.py /infra/coco_proposal_prepare.py
# Closure: /infra/vcoco_role_stream.py /src/world_reward/vcoco_role_reference.py
set +x
set -euo pipefail
[[ $# == 1 || $# == 5 ]] || exit 2
PHASE="$1";shift
[[ ( "$PHASE" == select && $# == 0 ) || ( "$PHASE" == acquire && $# == 4 ) ]] || exit 2
ROOT="${WR_ROOT:?}";CODE="${WR_CODE:?}";REV="${WR_CODE_REVISION:?}"
[[ "$(uname -s)" == Linux && "$(id -u)" == 0 && "$(hostname)" == world-reward-ncc-h100-02 \
 && "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_vcoco_role_prepare/code" ]] || exit 2
umask 077
ulimit -v 8388608
LIMIT=1215s;[[ "$PHASE" == select ]] || LIMIT=315s
exec env -i PATH=/usr/bin:/bin HOME=/nonexistent CUDA_VISIBLE_DEVICES=-1 \
 WR_CODE="$CODE" WR_CODE_REVISION="$REV" timeout --signal=TERM --kill-after=5s "$LIMIT" \
 nice -n 10 ionice -c 3 /usr/bin/python3 -I -B "$CODE/infra/vcoco_role_prepare.py" "$PHASE" "$@"
