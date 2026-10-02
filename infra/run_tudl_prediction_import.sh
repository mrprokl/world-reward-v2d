#!/usr/bin/env bash
# A frozen inference artifact from VM01, not another inference or private label.
set -euo pipefail
[[ $# == 2 && "$1" == --archive-sha256 && "$2" =~ ^[0-9a-f]{64}$ ]] || exit 2
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"
[[ -f "$ROOT/transfer/vm02-v1/import.json" ]]
export DOCKER_HOST="unix://$ROOT/docker.sock"
IMAGE=sha256:7ebfff18ba3b76dd919485c19115597d7531dfd3233f69461f1dce3f28a6c6d3
timeout 90s runuser -u scenesmith -- env WR_ROOT="$ROOT" PYTHONDONTWRITEBYTECODE=1 python3 - "$2" <<'PY'
import hashlib,json,os,pathlib,tarfile,sys
root=pathlib.Path(os.environ['WR_ROOT']);base=root/'validation/tudl_rgb_v1'
path=root/'transfer/vm02-v1/tudl-predictions.tar'
assert not path.is_symlink() and path.is_file() and path.stat().st_size<=100_000_000
assert hashlib.sha256(path.read_bytes()).hexdigest()==sys.argv[1]
out=base/'predictions_v1';assert not out.exists() and not out.is_symlink()
manifest=json.loads((base/'inputs/manifest.json').read_text());prefix='validation/tudl_rgb_v1/predictions_v1/'
expected={'report.json',*[pathlib.Path(r['file']).stem+'.npz' for r in manifest['images']]}
with tarfile.open(path,'r:') as archive:
 members=archive.getmembers();files=[m for m in members if m.isfile()]
 assert len(members)==11 and len(files)==10
 assert len({m.name for m in members})==11
 assert {m.name for m in files}=={prefix+n for n in expected}
 assert all(m.isfile() or (m.isdir() and m.name==prefix.rstrip('/')) for m in members)
 assert all(0<m.size<=16_000_000 for m in files)
 rp=archive.extractfile(prefix+'report.json').read()
 assert hashlib.sha256(rp).hexdigest()=='a187959136b5aeaca57447ac19e696e119df1607ebc40fac8a48a7d1b680682a'
 receipt=json.loads(rp);assert receipt['status']=='pass' and receipt['outputs_completed']==9 and receipt['MoGe_calls_completed']==27
 assert receipt['private_truth_read'] is False and receipt['ground_truth_used'] is False
 for row in receipt['outputs']:
  assert row['file'] in expected and hashlib.sha256(archive.extractfile(prefix+row['file']).read()).hexdigest()==row['sha256']
 out.mkdir(mode=0o755)
 for member in files:
  target=out/pathlib.PurePosixPath(member.name).name
  with target.open('xb') as f:f.write(archive.extractfile(member).read())
  target.chmod(0o444)
assert hashlib.sha256(path.read_bytes()).hexdigest()==sys.argv[1]
print(json.dumps({'stage':'frozen_TUDL_prediction_import','status':'pass','inference_rerun':False,'private_truth_imported':False}),flush=True)
PY
# Full numeric/schema validation happens before any evaluation-private mount.
timeout 90s docker run --rm --network none --memory 4g --cpus 2 --user 1000:1000 --entrypoint python \
 --env PYTHONDONTWRITEBYTECODE=1 --env WR_ROOT="$ROOT" --env PYTHONPATH="$CODE/src:$CODE/infra" \
 --mount "type=bind,src=$CODE,dst=$CODE,readonly" \
 --mount "type=bind,src=$ROOT/validation/tudl_rgb_v1/inputs,dst=$ROOT/validation/tudl_rgb_v1/inputs,readonly" \
 --mount "type=bind,src=$ROOT/validation/tudl_rgb_v1/predictions_v1,dst=$ROOT/validation/tudl_rgb_v1/predictions_v1,readonly" \
 "$IMAGE" -c 'import os,pathlib;from tudl_evaluate import public_predictions; arrays,records,frozen,hashes=public_predictions(pathlib.Path(os.environ["WR_ROOT"])/"validation/tudl_rgb_v1");assert len(arrays)==9;print(hashes)'
