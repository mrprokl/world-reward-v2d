#!/usr/bin/env bash
# Azure CPU only; complete original gzip, RGB headers and opaque private bytes.
# Source closure: /infra/dexycb_hand_acquire.py /infra/dexycb_acquire.py /infra/dexycb_download.py /infra/mediapipe_hands_acquire.py
set -euo pipefail
[[ $# == 0 ]] || exit 2
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_dexycb_hand_acquire/code" && "$(uname -s)" == Linux ]] || exit 2
ulimit -v 2097152
LEASE="$(python3 -I -B - "$ROOT" "$CODE" "$REV" <<'PYBOOTSTRAP'
import json, pwd, sys
from pathlib import Path
root, code, rev = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
sys.path.insert(0, str(code/'infra'))
import dexycb_hand_acquire as hand
source = hand.source_binding(root, code, rev)
account = pwd.getpwnam('scenesmith')
print(json.dumps(hand.lease.create_namespace_lease([root/hand.BASE, hand.INCOMING],
                 account.pw_uid, account.pw_gid, source['closure_sha256'])))
PYBOOTSTRAP
)"
exec timeout --signal=TERM --kill-after=130s 9140s runuser -u scenesmith -- env \
 WR_ROOT="$ROOT" WR_CODE="$CODE" WR_CODE_REVISION="$REV" WR_NAMESPACE_LEASE="$LEASE" \
 PYTHONDONTWRITEBYTECODE=1 python3 -I -B - "$CODE" <<'PYCONTROL'
import runpy, sys
from pathlib import Path
code = Path(sys.argv[1]); sys.path.insert(0, str(code/'infra'))
driver = code/'infra/dexycb_hand_acquire.py'; sys.argv = [str(driver)]
runpy.run_path(str(driver), run_name='__main__')
PYCONTROL
