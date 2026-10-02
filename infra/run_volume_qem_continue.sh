#!/usr/bin/env bash
# Only a complete new control gate permits one separate predicted-mesh proposal.
set -euo pipefail
(( $# == 0 )) || exit 2
CODE="${WR_CODE:?Require immutable committed source}"
bash "$CODE/infra/run_volume_mesh_gate.sh"
bash "$CODE/infra/run_object_budget_volume.sh" --episode 0
