#!/usr/bin/env bash
# CPU-only hash all original source; never execute historical code or load models.
# Source closure: /infra/cari_historical_source.py /infra/cari_full_forward.py
# /infra/cari_full_refine.py /infra/cari_shared_prepare.py
set -euo pipefail
[[ $# == 0 ]] || exit 2
ROOT="${WR_ROOT:?}";CODE="${WR_CODE:?}";REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_cari_historical_source_gate/code" \
 && "${BASH_SOURCE[0]}" == "$CODE/infra/run_cari_historical_source_gate.sh" ]] || exit 2
export DOCKER_HOST="unix://$ROOT/docker.sock"
IMAGE=sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7
HISTORICAL="$ROOT/jobs/672b10ee5d8b8532686cf39ccd44134adc26178b/run_cari_full_refine_queued/code"
MOUNTS=()
for path in "$CODE" "$HISTORICAL" "${HISTORICAL%/code}/revision" "${HISTORICAL%/code}/source-sha256" \
 "$ROOT/results/episode3-queued-source-cache-audit.json";do
 [[ -e "$path" && ! -L "$path" ]];MOUNTS+=(--mount "type=bind,src=$path,dst=$path,readonly")
done
for stage in prepare forward refined;do
 path="$ROOT/outputs/episode_000003/cari_shared_${stage}_v1/report.json"
 [[ -f "$path" && ! -L "$path" ]];MOUNTS+=(--mount "type=bind,src=$path,dst=$path,readonly")
done
timeout --signal=TERM --kill-after=5s 60s docker run --rm --network none --read-only --memory 2g --cpus 2 \
 --user 0:0 --entrypoint /usr/bin/env "${MOUNTS[@]}" "$IMAGE" \
 -i PATH=/opt/conda/bin:/usr/bin:/bin HOME=/nonexistent PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$CODE/src:$CODE/infra" \
 /opt/conda/bin/python -B -c 'import json,sys;from pathlib import Path;from cari_historical_source import verify_historical_source;import cari_shared_prepare as p,cari_full_forward as f,cari_full_refine as r;root,code=map(Path,sys.argv[1:]);h,proof=verify_historical_source(root,code/"configs/cari_clip_000003_historical_source_pins.json");reports={s:json.loads((root/("outputs/episode_000003/cari_shared_"+s+"_v1/report.json")).read_text())for s in("prepare","forward","refined")};assert all(reports[s]["source_helpers"]==m.source_helpers(h)for s,m in(("prepare",p),("forward",f),("refined",r)));assert "torch"not in sys.modules;print(json.dumps(dict(stage="cari_historical_source_gate",status="pass",proof=proof,all_three_original_helper_inventories_verified=True,historical_code_executed=False,GPU_used=False,models_loaded=False),sort_keys=True))' "$ROOT" "$CODE"
