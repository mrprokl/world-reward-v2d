#!/usr/bin/env bash
# Clean public frontend route ONLY, ending before learned CoCoNet or conversion.
# Child wrappers retain their provenance/numerical checks, current image-tag gates
# and broad existing mounts. Resource/mount hardening of those children is separate.
# This scoped lock serializes cooperating frontend launches; legacy/full-refine
# units do not hold it and still require the root scheduler to serialize GPU work.
set -eEuo pipefail
[[ ( $# == 2 || $# == 4 ) && "$1" == --episode && "$2" =~ ^(0|[1-9]|[12][0-9])$ ]] || exit 2
EPISODE="$2"
MASK_POLICY=default_three
if [[ $# == 4 ]];then
 [[ "$3" == --actor-policy && "$4" == fixed_all16 ]] || exit 2
 # Fixed global policy validated on procedural observations. No per-episode
 # thresholds, adaptive until-pass search, fabricated or interpolated support.
 MASK_POLICY=fixed_all16
fi
ROOT="${WR_ROOT:?}";CODE="${WR_CODE:?}";REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
   && ( "$CODE" == "$ROOT/jobs/$REV/run_track1_frontends/code" \
        || "$CODE" == "$ROOT/jobs/$REV/run_track1_frontends_queued/code" ) ]] || exit 2
printf -v PADDED '%06d' "$EPISODE"
BASE="$ROOT/outputs/episode_$PADDED"
CURRENT_STAGE=preflight
phase() {
 printf '{"mode":"public_frontends_only","stage":"%s","phase":"%s","timestamp_utc":"%s"}\n' \
  "$CURRENT_STAGE" "$1" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap 'status=$?; trap - ERR; phase fail; exit "$status"' ERR
phase start
# Pure host stdlib preflight. Never import numerical producer modules on the
# bare Azure host; its Python is not the pinned model/container runtime.
python3 - "$ROOT" "$CODE" "$REV" "$BASE" "${BASH_SOURCE[0]}" <<'PYSAFE'
from pathlib import Path
import re,stat,sys
root,code,revision,base,executing=sys.argv[1:]
root,code,base,executing=map(Path,(root,code,base,executing))
for path in (root,code,base,executing):
 if path.absolute()!=path or path.resolve()!=path or any(parent.is_symlink() for parent in (path,*path.parents)):
  raise ValueError('Canonical launcher/source/output paths required; no symlinks')
if not root.is_dir() or not code.is_dir() or executing!=code/'infra/run_track1_frontends.sh':
 raise ValueError('Actual immutable launcher entrypoint required')
if code==root/'jobs'/revision/'run_track1_frontends_queued/code':
 queued=code/'infra/run_track1_frontends_queued.sh'
 if queued.is_symlink() or not queued.is_file() or queued.stat().st_mode&0o222:
  raise ValueError('Actual readonly scheduling wrapper required for queued namespace')
if (code.parent/'revision').read_text().strip()!=revision or not re.fullmatch('[0-9a-f]{64}',(code.parent/'source-sha256').read_text().strip()):
 raise ValueError('Exact immutable launcher revision/archive identity markers required')
children=('run_automatic_masks.sh','run_episode_initializers.sh','run_body_smoke.sh','run_depth_smoke.sh',
          'run_scale_smoke.sh','run_object_smoke.sh','run_cari_body_adapter_smoke.sh','run_object_pose_smoke.sh',
          'run_cari_prepare.sh','cari_wrapper_common.sh','automatic_masks.py','body_smoke.py','depth_smoke.py',
          'scale_smoke.py','object_smoke.py','cari_body_adapter_smoke.py','object_pose_smoke.py','cari_prepare.py')
for name in children:
 path=code/'infra'/name
 if not path.is_file() or path.is_symlink() or not stat.S_ISREG(path.lstat().st_mode) or path.stat().st_mode&0o222:
  raise ValueError('All actual child source/wrapper files must be immutable regular files')
for path in code.rglob('*'):
 if path.is_symlink() or (not path.is_dir() and not stat.S_ISREG(path.lstat().st_mode)) or path.stat().st_mode&0o222:
  raise ValueError('Complete launcher source snapshot must remain readonly and regular')
if any(path.exists() and not path.is_dir() for path in (root/'outputs',base)):
 raise ValueError('Output ancestors must be directories, never files')
targets=('automatic_masks','body_smoke','depth_smoke','scale_smoke','object_grounded','body_full','depth_full',
         'body_full/cari_adapter','object_pose_full','cari_inputs')
for target in targets:
 path=base/target
 if path.exists() or path.is_symlink():
  raise ValueError('Frozen frontend target already exists; no overwrite/resume: '+str(path))
gate=root/'results/camera-render.json'
if not gate.is_file() or gate.is_symlink() or gate.resolve()!=gate.absolute() or not gate.stat().st_size:
 raise ValueError('Existing reference camera/depth gate must be present')
jobs=root/'jobs'
lock=jobs/'.world-reward-h100.lock'
if not jobs.is_dir() or jobs.resolve()!=jobs.absolute() or lock.is_symlink() or (lock.exists() and not stat.S_ISREG(lock.lstat().st_mode)):
 raise ValueError('Canonical scoped GPU lock path required')
PYSAFE
phase pass

CURRENT_STAGE=gpu_preflight
phase start
exec 9>"$ROOT/jobs/.world-reward-h100.lock"
flock --nonblock 9
gpu_idle() {
 local apps
 apps="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)"
 if [[ -n "${apps//[[:space:]]/}" ]]; then
  echo 'GPU compute applications present; frontend launch aborted' >&2
  return 1
 fi
}
gpu_idle
phase pass

CURRENT_STAGE=automatic_masks
phase start
if [[ "$MASK_POLICY" == fixed_all16 ]];then
 timeout --signal=TERM --kill-after=10s 600s bash "$CODE/infra/run_automatic_masks.sh" --episode "$EPISODE" --seed-frames 16 --actor-seed-observations 16
else
 timeout --signal=TERM --kill-after=10s 600s bash "$CODE/infra/run_automatic_masks.sh" --episode "$EPISODE"
fi
phase pass

CURRENT_STAGE=episode_initializers
phase start
gpu_idle
# This total ceiling includes all seven existing serial initializer children.
timeout --signal=TERM --kill-after=10s 5400s bash "$CODE/infra/run_episode_initializers.sh" --episode "$EPISODE"
phase pass

CURRENT_STAGE=object_pose_full
phase start
gpu_idle
timeout --signal=TERM --kill-after=10s 7200s bash "$CODE/infra/run_object_pose_smoke.sh" --episode "$EPISODE" --full-video
phase pass

# Release the cooperating-GPU lock before CPU-only input assembly. No implicit
# historical unit waits: every child returned within this same bounded route.
flock --unlock 9
exec 9>&-
CURRENT_STAGE=cari_inputs
phase start
timeout --signal=TERM --kill-after=10s 7200s bash "$CODE/infra/run_cari_prepare.sh" --episode "$EPISODE" --no-wait
phase pass
