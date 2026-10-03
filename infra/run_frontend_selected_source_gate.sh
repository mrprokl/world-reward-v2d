#!/usr/bin/env bash
# CPU-only source binding of the actual private selected archive. No models/data.
# Source closure: /infra/frontend_selected_assets.py /infra/body_smoke.py
set -euo pipefail
[[ $# == 0 ]] || exit 2
ROOT="${WR_ROOT:?}";CODE="${WR_CODE:?}";REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_frontend_selected_source_gate/code" \
 && "${BASH_SOURCE[0]}" == "$CODE/infra/run_frontend_selected_source_gate.sh" \
 && "$(hostname)" == world-reward-ncc-h100-02 ]] || exit 2
export DOCKER_HOST="unix://$ROOT/docker.sock"
IMAGE=sha256:7ebfff18ba3b76dd919485c19115597d7531dfd3233f69461f1dce3f28a6c6d3
[[ "$(docker image inspect "$IMAGE" --format '{{.Id}}')" == "$IMAGE" ]]
DEST=/srv/world-reward-data/frontend-assets-extracted-75fdea08fb3b43f62f1c4b4f5e646674595cdd0b
MANIFEST="$DEST/world-reward-frontend-assets-manifest.json"
BODY=vendor/video_to_data/reconstruction/modules/v2d_sam3d_body/lib/sam_3d_body
DINO=weights/cari4d/sam3d_body/torch_home/hub/facebookresearch_dinov3_main
RECEIPT="$ROOT/results/frontend-asset-extract-75fdea08fb3b43f62f1c4b4f5e646674595cdd0b/report.json"
MOUNTS=()
for path in "$CODE" "$CODE/../revision" "$CODE/../source-sha256" "$MANIFEST" "$RECEIPT" "$DEST/$BODY" "$DEST/$DINO";do
 # Source markers use normalized lexical parents, never a runtime alias.
 path="$(readlink -m "$path")";[[ -e "$path" && ! -L "$path" ]]
 MOUNTS+=(--mount "type=bind,src=$path,dst=$path,readonly")
done
MOUNTS+=(--mount "type=bind,src=$DEST/$DINO,dst=$ROOT/$DINO,readonly")
timeout --signal=TERM --kill-after=5s 90s docker run --rm --network none --read-only --memory 2g --cpus 2 \
 --user 0:0 --entrypoint /usr/bin/env "${MOUNTS[@]}" "$IMAGE" \
 -i PATH=/opt/conda/bin:/usr/bin:/bin HOME=/nonexistent PYTHONDONTWRITEBYTECODE=1 \
 PYTHONPATH="$CODE/src:$CODE/infra" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
 /opt/conda/bin/python -B -c 'import json,sys;from pathlib import Path;from frontend_selected_assets import load_contract,bind_body,bind_dinov3;root,manifest=map(Path,sys.argv[1:]);c=load_contract(root,manifest);body=bind_body(c,Path("/workspace/v2d_sam3d_body/lib/sam_3d_body"));dino=bind_dinov3(c,root/"weights/cari4d/sam3d_body/torch_home/hub/facebookresearch_dinov3_main");print(json.dumps(dict(stage="frontend_selected_source_gate",status="pass",body=body,dinov3=dino,GPU_used=False,models_loaded=False,challenge_data_read=False,whole_checkout_verified=False,replica_ready=False),sort_keys=True))' "$ROOT" "$MANIFEST"
