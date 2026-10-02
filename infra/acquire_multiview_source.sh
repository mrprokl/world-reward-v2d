#!/usr/bin/env bash
# Azure-only public source acquisition, never assets/models or a local download.
set -euo pipefail
[[ $# -eq 0 ]] || { echo "No acquisition arguments supported" >&2; exit 2; }
ROOT="${WR_ROOT:-/srv/scenesmith/world-reward}"
CODE="${WR_CODE:?Require immutable source}"
REVISION="${WR_CODE_REVISION:?Require immutable source revision}"
[[ "$(uname -s)" == Linux && "$ROOT" == /srv/scenesmith/* && "$REVISION" =~ ^[0-9a-f]{40}$ ]]
export WR_ROOT="$ROOT" WR_CODE_REVISION="$REVISION"
"$ROOT/code/.venv/bin/python" - <<'PY'
import hashlib, json, os, pathlib, shutil, subprocess, tempfile, time
root = pathlib.Path(os.environ['WR_ROOT'])
pin = 'abb04b5e8af5bc33b0265bdf19937e76bbb6bcdd'
base = root / 'vendor/mv-sam3d'
destination = base / pin
manifest = root / 'results/multiview-source.json'
if manifest.exists() or destination.exists():
    raise FileExistsError('Frozen MV source/manifest already exists')
base.mkdir(parents=True, exist_ok=True)
staging = pathlib.Path(tempfile.mkdtemp(prefix='.acquiring-', dir=base))
started = time.perf_counter()
def git(*args):
    return subprocess.check_output(['git', '-C', str(staging), *args], text=True).strip()
try:
    git('init', '-q')
    git('remote', 'add', 'origin', 'https://github.com/devinli123/MV-SAM3D.git')
    git('config', 'core.sparseCheckout', 'true')
    (staging / '.git/info/sparse-checkout').write_text('/sam3d_objects/\n/LICENSE\n/README.md\n')
    git('fetch', '--depth=1', 'origin', pin)
    git('checkout', '--detach', 'FETCH_HEAD')
    if git('rev-parse', 'HEAD') != pin or git('status', '--porcelain'):
        raise RuntimeError('Source pin/clean checkout gate failed')
    records = []
    for name in git('ls-files', '--', 'sam3d_objects', 'LICENSE', 'README.md').splitlines():
        path = staging / name
        if path.is_symlink() or not path.is_file() or path.resolve() != path.absolute():
            raise RuntimeError('Source must be regular, unsymlinked tracked files')
        records.append({'path': name, 'bytes': path.stat().st_size,
                        'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    if not records or not any(r['path'] == 'LICENSE' for r in records):
        raise RuntimeError('Incomplete source inventory')
    staging.rename(destination)  # Same filesystem, complete tree before publication.
    for path in sorted(destination.rglob('*'), key=lambda p: len(p.parts), reverse=True):
        path.chmod(path.stat().st_mode & ~0o222)
    destination.chmod(destination.stat().st_mode & ~0o222)
    report = {'stage': 'pinned_mv_sam3d_source_acquisition', 'status': 'pass', 'revision': pin,
              'repository': 'https://github.com/devinli123/MV-SAM3D.git', 'path': str(destination),
              'code_revision': os.environ['WR_CODE_REVISION'], 'files': records,
              'source_git_clean_verified': True, 'source_read_only': True,
              'weights_downloaded': False, 'challenge_assets_downloaded': False,
              'license_status': 'SAM_custom_competition_eligibility_unresolved',
              'elapsed_seconds': time.perf_counter() - started}
    with manifest.open('x') as handle:
        handle.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in ('stage', 'status', 'revision', 'elapsed_seconds')}))
finally:
    if staging.exists():
        shutil.rmtree(staging)
PY
