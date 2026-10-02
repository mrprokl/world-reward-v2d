#!/usr/bin/env bash
# Run ONLY on the Azure Linux host. Downloads never pass through the laptop.
set -euo pipefail
ROOT="${WR_ROOT:-/srv/scenesmith/world-reward}"
CODE="$ROOT/code"
mkdir -p "$ROOT"/{bin,data,weights,outputs,results,vendor}
export UV_INSTALL_DIR="$ROOT/bin"
export UV_PYTHON_INSTALL_DIR="$ROOT/python"
export UV_CACHE_DIR="$ROOT/cache/uv"
export HF_HOME="$ROOT/cache/huggingface"
export PATH="$ROOT/bin:$PATH"
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/0.10.8/install.sh | sh
fi
cd "$CODE"
uv sync --python 3.11 --frozen --extra data --extra dev
uv run pytest -q
uv run wr-data --config configs/sources.json --root "$ROOT/data" \
  --manifest "$ROOT/results/input-manifest.json"

uv run python - "$ROOT" <<'PY'
import hashlib, json, pathlib, subprocess, sys, urllib.request, zipfile
root = pathlib.Path(sys.argv[1])
sources = json.loads(pathlib.Path('configs/sources.json').read_text())
archive = root / 'vendor/kit.zip'
urllib.request.urlretrieve(sources['kit']['url'], archive)
if hashlib.sha256(archive.read_bytes()).hexdigest() != sources['kit']['sha256']:
    raise RuntimeError('Official kit changed: audit new version before use')
with zipfile.ZipFile(archive) as z:
    members = [m for m in z.namelist() if not any(p in m for p in (
        '/data/track_2', '/data/track_3', '/metric_code/track_2/',
        '/metric_code/track_3/', '/flash_chord/', '/tracks/track_2/', '/tracks/track_3/'))]
    z.extractall(root / 'vendor', members)
archive.unlink()
sample = root / 'vendor/v2d_submission_kit/data/track_1_sample_submission.parquet'
if hashlib.sha256(sample.read_bytes()).hexdigest() != sources['kit']['sample_sha256']:
    raise RuntimeError('Sample hash mismatch')
repo = root / 'vendor/video_to_data'
if not repo.exists():
    subprocess.run(['git','clone','--filter=blob:none','--no-checkout',sources['upstream']['url'],str(repo)],check=True)
subprocess.run(['git','-C',str(repo),'sparse-checkout','init','--no-cone'],check=True)
subprocess.run(['git','-C',str(repo),'sparse-checkout','set','--no-cone','/reconstruction/',
                '!/reconstruction/**/assets/','!/reconstruction/**/data/',
                '/reconstruction/modules/v2d_sam3d_body/lib/sam_3d_body/data/',
                '/LICENSE'],check=True)
subprocess.run(['git','-C',str(repo),'checkout',sources['upstream']['revision']],check=True)
provenance = {'upstream_revision':sources['upstream']['revision'],
              'kit_sha256':sources['kit']['sha256'], 'dataset_revision':sources['dataset']['revision'],
              'input_track':'track_1','ground_truth_used':False,'oracle_modes':[],
              'heavy_artifacts_host':'Azure', 'python':sys.version}
(root / 'results/bootstrap.json').write_text(json.dumps(provenance,indent=2)+'\n')
print(json.dumps({'bootstrap':'complete','track':'track_1','data_root':str(root/'data')}))
PY
