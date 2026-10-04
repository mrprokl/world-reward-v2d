#!/usr/bin/env bash
# Source closure: /infra/mediapipe_cpu_dependencies_acquire.py /infra/mediapipe_hands_acquire.py
# Azure CPU exact public wheels only; no install, inference, or dataset.
set -euo pipefail
[[ $# == 0 ]] || exit 2
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_mediapipe_cpu_dependencies_acquire/code" && "$(uname -s)" == Linux ]] || exit 2
ulimit -v 2097152
LEASE="$(timeout --signal=TERM --kill-after=5s 30s python3 -I -B - "$ROOT" "$CODE" "$REV" <<'PYBOOTSTRAP'
import importlib.util, json, pwd, sys
from pathlib import Path
root, code, rev = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
sys.path.insert(0, str(code/'infra'))
spec = importlib.util.spec_from_file_location('wr_mp_dependencies_bootstrap', code/'infra/mediapipe_cpu_dependencies_acquire.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
source = m.source_binding(root, code, rev)
m.verify_prior(root, code/m.PRIOR_PINS); m.load_manifest(code/m.MANIFEST)
account = pwd.getpwnam('scenesmith')
m.require(not (root/m.RESULT).exists(), 'Fresh result before namespace bootstrap required')
print(json.dumps(m.mp.create_namespace_lease([root/m.EVIDENCE], account.pw_uid,
                 account.pw_gid, source['closure_sha256'])))
PYBOOTSTRAP
)"
exec timeout --signal=TERM --kill-after=10s 320s runuser -u scenesmith -- env \
 WR_ROOT="$ROOT" WR_CODE="$CODE" WR_CODE_REVISION="$REV" WR_NAMESPACE_LEASE="$LEASE" PYTHONDONTWRITEBYTECODE=1 \
 python3 -I -B - "$CODE" <<'PYCONTROL'
from pathlib import Path
import runpy, sys
code = Path(sys.argv[1]); driver = code / 'infra/mediapipe_cpu_dependencies_acquire.py'
sys.path.insert(0, str(code / 'infra')); sys.argv = [str(driver)]
runpy.run_path(str(driver), run_name='__main__')
PYCONTROL
