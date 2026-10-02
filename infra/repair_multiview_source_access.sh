#!/usr/bin/env bash
# Only public-source access modes change; frozen content and manifests do not.
set -euo pipefail
[[ $# -eq 0 ]] || { echo 'Source access repair has no arguments' >&2; exit 2; }
ROOT="${WR_ROOT:-/srv/scenesmith/world-reward}"
REVISION="${WR_CODE_REVISION:?Require immutable repair source}"
[[ "$(uname -s)" == Linux && "$ROOT" == /srv/scenesmith/world-reward ]]
export WR_ROOT="$ROOT" WR_CODE_REVISION="$REVISION"
"$ROOT/code/.venv/bin/python" - <<'PY'
import hashlib,json,os,pathlib,re
root=pathlib.Path(os.environ['WR_ROOT'])
pin='abb04b5e8af5bc33b0265bdf19937e76bbb6bcdd'
source=root/'vendor/mv-sam3d'/pin
manifest=root/'results/multiview-source.json'
output=root/'results/multiview-source-access.json'
if output.exists() or output.is_symlink(): raise FileExistsError('Access repair manifest is frozen')
revision=os.environ['WR_CODE_REVISION']
if re.fullmatch(r'[0-9a-f]{40}',revision) is None: raise ValueError('Require source revision')
digest=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
data=json.loads(manifest.read_text()); original_hash=digest(manifest)
if (data.get('stage')!='pinned_mv_sam3d_source_acquisition' or data.get('status')!='pass'
        or data.get('revision')!=pin or data.get('path')!=str(source)):
    raise ValueError('Require exact public source manifest')
def check_contents():
    for row in data['files']:
        name=pathlib.Path(row['path']); path=source/name
        if (name.is_absolute() or '..' in name.parts or path.is_symlink() or not path.is_file()
                or path.stat().st_size!=row['bytes'] or digest(path)!=row['sha256']):
            raise ValueError('Frozen source content mismatch; refuse access repair')
check_contents()
paths=[source,*source.rglob('*')]
if any(p.is_symlink() or not (p.is_file() or p.is_dir()) for p in paths):
    raise ValueError('Public tree must contain only regular files/directories')
changed=0
for path in paths:
    before=path.stat().st_mode&0o777
    after=0o555 if path.is_dir() else (before&0o111)|0o444
    if before!=after: path.chmod(after); changed+=1
check_contents()
if digest(manifest)!=original_hash: raise ValueError('Original manifest changed')
report={'stage':'public_mv_source_access_modes_only','status':'pass','revision':pin,
        'producer_revision':revision,'source_manifest_sha256':original_hash,
        'content_hashes_unchanged':True,'original_manifest_unchanged':True,
        'changed_mode_count':changed,'directory_mode':'0555','files_public_read_only':True,
        'weights_or_challenge_inputs_accessed':False,'source_bytes_modified':False}
with output.open('x') as f: f.write(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
PY
