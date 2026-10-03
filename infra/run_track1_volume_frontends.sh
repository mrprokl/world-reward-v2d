#!/usr/bin/env bash
# Bounded continuation of existing public frontends with a pinned CPU mesh.
# Children retain existing broad mounts and their own provenance/numerical gates.
# This lock covers cooperating frontends only, not legacy/full-refine units;
# the root scheduler must still serialize GPU work outside this wrapper.
# Source closure: /infra/volume_geometry_loader.py /infra/volume_mesh_pin_inventory.py
# and /infra/object_budget_volume.py. No proposal, repair or adoption is run here.
set -eEuo pipefail
[[ $# == 2 && "$1" == --episode && "$2" =~ ^(0|[1-9]|[12][0-9])$ ]] || exit 2
EPISODE="$2"
ROOT="${WR_ROOT:?}";CODE="${WR_CODE:?}";REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
   && "$CODE" == "$ROOT/jobs/$REV/run_track1_volume_frontends/code" ]] || exit 2
printf -v PADDED '%06d' "$EPISODE"
BASE="$ROOT/outputs/episode_$PADDED"
PIN="$CODE/configs/volume_mesh_${PADDED}_pins.json"
CURRENT_STAGE=preflight
phase() {
 printf '{"mode":"pinned_volume_frontends_only","stage":"%s","phase":"%s","timestamp_utc":"%s"}\n' \
  "$CURRENT_STAGE" "$1" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap 'status=$?; trap - ERR; phase fail; exit "$status"' ERR
phase start
# Host imports only the standalone stdlib hash/JSON inventory, never NumPy,
# Torch, the geometry loader, predictions, a model, RGB or a depth array.
env PYTHONDONTWRITEBYTECODE=1 python3 -I -B - "$ROOT" "$CODE" "$REV" "$BASE" "$PIN" "$EPISODE" "${BASH_SOURCE[0]}" <<'PYSAFE'
from pathlib import Path
import re,stat,sys
root,code,revision,base,pin,episode,executing=sys.argv[1:]
root,code,base,pin,executing=map(Path,(root,code,base,pin,executing));episode=int(episode)
for path in (root,code,base,pin,executing):
 if path.absolute()!=path or path.resolve()!=path or any(parent.is_symlink() for parent in (path,*path.parents)):
  raise ValueError('Canonical actual launcher/source/input paths required; no symlinks')
if not root.is_dir() or not code.is_dir() or not base.is_dir() or executing!=code/'infra/run_track1_volume_frontends.sh':
 raise ValueError('Existing selected episode and actual immutable launcher required')
if (code.parent/'revision').read_text().strip()!=revision or not re.fullmatch('[0-9a-f]{64}',(code.parent/'source-sha256').read_text().strip()):
 raise ValueError('Exact immutable launcher revision/archive markers required')
children=('run_object_pose_smoke.sh','object_pose_smoke.py','run_cari_prepare.sh','cari_prepare.py',
          'cari_wrapper_common.sh','body_smoke.py','camera_render.py','volume_geometry_loader.py',
          'volume_mesh_pin_inventory.py','object_budget_volume.py')
for name in children:
 path=code/'infra'/name
 if not path.is_file() or path.is_symlink() or not stat.S_ISREG(path.lstat().st_mode) or path.stat().st_mode&0o222:
  raise ValueError('All required child/loader/inventory sources must be immutable regular files')
for path in (code,*code.rglob('*')):
 if path.is_symlink() or (not path.is_dir() and not stat.S_ISREG(path.lstat().st_mode)) or path.stat().st_mode&0o222:
  raise ValueError('Complete actual source snapshot must be readonly and regular')
if not pin.is_file() or not stat.S_ISREG(pin.lstat().st_mode) or pin.stat().st_mode&0o222:
 raise ValueError('Committed immutable selected volume pins required')
for target in ('object_pose_full','cari_inputs'):
 path=base/target
 if path.exists() or path.is_symlink():
  raise ValueError('Frozen continuation target exists; no removal/overwrite/resume: '+str(path))
sys.path.insert(0,str(code/'infra'))
import volume_mesh_pin_inventory as inventory
pins=inventory.strict_json(pin.read_text())
if type(pins) is not dict or pins.get('episode_index')!=episode or type(pins.get('episode_index')) is not int:
 raise ValueError('Explicit selected episode differs from immutable mesh pins')
paths=inventory.paths(episode)
inventory.verify_pinned_artifacts(root,pins,episode,pins.get('input_sha256'),
 pins.get('files',{}).get(paths['object'],{}).get('sha256'),
 pins.get('files',{}).get(paths['alignment'],{}).get('sha256'),pins.get('metric_scale_baked_once'))
stages={'automatic_masks':'automatic_masks','body_smoke':'sam3d_body_three_frame_smoke',
 'depth_smoke':'monocular_moge2_three_frame','body_full':'sam3d_body_full_video_initializer',
 'depth_full':'monocular_moge2_full_video','body_full/cari_adapter':'native_cari_body_adapter_full_video',
 'object_grounded':'sam3d_objects_grounded_fixed_frame','scale_smoke':'predicted_human_anchored_moge2_pointmaps'}
reports={};identities={name:inventory.identity(base/name/'report.json') for name in stages}
for name,stage in stages.items():
 record=inventory.strict_json((base/name/'report.json').read_text())
 expected=dict(stage=stage,status='pass',episode_index=episode,input_track='track_1',
  ground_truth_used=False,hand_labeled_test=False,oracle_modes=[])
 if name!='body_full/cari_adapter':expected['input_sha256']=pins['input_sha256']
 inventory._require(record,expected);reports[name]=record
 if 'input_sha256' in record and record['input_sha256']!=pins['input_sha256']:
  raise ValueError('Selected original video differs across frontend sources')
count=reports['automatic_masks'].get('frames')
if type(count) is not int or count<96:raise ValueError('Full original native timeline required')
for name in ('body_full','depth_full'):
 frames=reports[name].get('frames')
 if (type(frames) is not list or any(type(row) is not dict or type(row.get('frame_index')) is not int for row in frames)
     or [row['frame_index'] for row in frames]!=list(range(count))):
  raise ValueError('Exact original full initializer frame coverage required')
 if 'total_video_frames' in reports[name] and reports[name]['total_video_frames']!=count:
  raise ValueError('Original full frontend frame counts differ')
adapter=reports['body_full/cari_adapter'];alignment=reports['scale_smoke']
if (type(adapter.get('frames')) is not int or adapter['frames']!=count
    or adapter.get('body_report_sha256')!=identities['body_full']['sha256']
    or alignment.get('body_report_sha256')!=identities['body_smoke']['sha256']
    or alignment.get('depth_report_sha256')!=identities['depth_smoke']['sha256']):
 raise ValueError('Full adapter or original alignment source ancestry differs')
if (identities['automatic_masks']['sha256']!=inventory.strict_json((root/paths['report']).read_text()).get('source_hashes',{}).get('mask_report')
    or {name:inventory.identity(base/name/'report.json') for name in stages}!=identities):
 raise ValueError('Automatic masks or original frontend reports changed')
lock=root/'jobs/.world-reward-h100.lock'
if lock.is_symlink() or (lock.exists() and not stat.S_ISREG(lock.lstat().st_mode)):
 raise ValueError('Canonical scoped GPU lock file required')
PYSAFE
phase pass

CURRENT_STAGE=gpu_preflight
phase start
export DOCKER_HOST="unix://$ROOT/docker.sock"
IMAGE="$(docker image inspect world-reward/cari4d-source:0.1 --format '{{.Id}}')"
if [[ "$IMAGE" != sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7 ]];then
 echo 'Exact original image differs; continuation aborted' >&2
 phase fail
 exit 1
fi
exec 9>"$ROOT/jobs/.world-reward-h100.lock"
flock --nonblock 9
gpu_idle() {
 local apps
 apps="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)"
 [[ -z "${apps//[[:space:]]/}" ]] || { echo 'GPU compute applications present; continuation aborted' >&2; return 1; }
}
gpu_idle
phase pass

CURRENT_STAGE=object_pose_full
phase start
timeout --signal=TERM --kill-after=10s 7200s bash "$CODE/infra/run_object_pose_smoke.sh" --episode "$EPISODE" --full-video --mesh-source volume
phase pass

# CPU-only preparation may overlap separately scheduled GPU work. This is not
# a throughput-isolation claim or a lock protecting noncooperating old units.
flock --unlock 9
exec 9>&-
CURRENT_STAGE=cari_inputs
phase start
timeout --signal=TERM --kill-after=10s 7200s bash "$CODE/infra/run_cari_prepare.sh" --episode "$EPISODE" --no-wait
phase pass
