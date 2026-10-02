#!/usr/bin/env bash
set -euo pipefail
(( $# == 0 )) || exit 2
exec bash "${WR_CODE:?}/infra/research_runtime_bootstrap.sh" --install
