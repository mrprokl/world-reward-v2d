#!/usr/bin/env bash
# One genuine audited pure-Python wheel; no pip resolver or image modification.
set -euo pipefail
(( $# == 0 )) || exit 2
ROOT="${WR_ROOT:?}"; [[ "$ROOT" == /srv/scenesmith/world-reward ]]
OUT="$ROOT/vendor/research/da3_dependencies_v1"
[[ ! -e "$OUT" && ! -L "$OUT" ]]
mkdir "$OUT"; chmod 755 "$OUT"
timeout 60s python3 - "$OUT" "${WR_CODE_REVISION:?}" <<'PY'
import hashlib,io,json,pathlib,sys,urllib.request,zipfile
out=pathlib.Path(sys.argv[1]);name='addict-2.4.0-py3-none-any.whl'
url='https://files.pythonhosted.org/packages/6a/00/b08f23b7d7e1e14ce01419a467b583edbb93c6cdb8654e54a9cc579cd61f/'+name
with urllib.request.urlopen(url,timeout=30) as f:b=f.read(3833)
assert len(b)==3832 and hashlib.sha256(b).hexdigest()=='249bb56bbfd3cdc2a004ea0ff4c2b6ddc84d53bc2194761636eb314d5cfa5dfc'
with zipfile.ZipFile(io.BytesIO(b)) as z:
 names=z.namelist();assert len(names)==7 and len(set(names))==7
 assert set(names)=={'addict/__init__.py','addict/addict.py',*[f'addict-2.4.0.dist-info/{n}' for n in ['LICENSE','METADATA','WHEEL','top_level.txt','RECORD']]}
 license=z.read('addict-2.4.0.dist-info/LICENSE')
 assert hashlib.sha256(license).hexdigest()=='ca488d33c512d0b226142090af90e89ae266a901a293f89fd642dfec931e22c1'
 assert 'Requires-Dist:' not in z.read('addict-2.4.0.dist-info/METADATA').decode()
p=out/name
with p.open('xb') as f:f.write(b)
p.chmod(0o444)
r={'stage':'pinned_DA3_minimal_dependency_acquisition','status':'pass','producer_revision':sys.argv[2],
 'file':name,'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b),'url':url,'license':'MIT',
 'license_sha256':hashlib.sha256(license).hexdigest(),'transitive_dependencies':[],
 'installed_into_image':False,'model_source_modified':False}
with (out/'receipt.json').open('x') as f:json.dump(r,f);f.write('\n')
(out/'receipt.json').chmod(0o444);out.chmod(0o555)
print(json.dumps(r),flush=True)
PY
