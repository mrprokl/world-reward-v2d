#!/usr/bin/env bash
# Sourced by immutable CARI wrappers. No Docker, artifact mutation or retries.

wr_parse_cari_arguments() {
  local mode="$1" episode_seen=0 wait_seen=0 kernel_seen=0
  shift
  WR_EPISODE=15 WR_WAIT_FOR='' WR_KERNEL_ONLY=0
  while (( $# )); do
    case "$1" in
      --episode)
        if (( episode_seen || $# < 2 )) || [[ ! "$2" =~ ^(0|[1-9]|[12][0-9])$ ]]; then
          echo 'Require exactly one --episode integer in 0..29' >&2; return 2
        fi
        WR_EPISODE="$2"; episode_seen=1; shift 2 ;;
      --wait-for)
        if (( wait_seen || $# < 2 )) || [[ ! "$2" =~ ^world-reward-[a-z][a-z0-9-]{0,50}(\.service)?$ ]]; then
          echo 'Require one --wait-for world-reward-<job>[.service] unit' >&2; return 2
        fi
        WR_WAIT_FOR="${2%.service}.service"; wait_seen=1; shift 2 ;;
      --kernel-only)
        if [[ "$mode" != forward ]] || (( kernel_seen )); then
          echo '--kernel-only is a single forward-only checkpoint gate' >&2; return 2
        fi
        WR_KERNEL_ONLY=1; kernel_seen=1; shift ;;
      *) echo "Unsupported wrapper argument: $1" >&2; return 2 ;;
    esac
  done
  if (( WR_KERNEL_ONLY && wait_seen )); then
    echo 'Checkpoint-only mode must not wait for preparation' >&2; return 2
  fi
  if [[ -z "$WR_WAIT_FOR" && "$WR_EPISODE" == 15 && "$WR_KERNEL_ONLY" == 0 ]]; then
    case "$mode" in
      prepare) WR_WAIT_FOR=world-reward-object-pose-full.service ;;
      forward) WR_WAIT_FOR=world-reward-cari-prepare-v2.service ;;
      converter) WR_WAIT_FOR=world-reward-cari-forward.service ;;
    esac
  fi
}

wr_require_dependency_report() {
  local report="$1" unit="${2:-}" state load waited=0
  if [[ -z "$unit" ]]; then
    [[ -f "$report" ]] && return 0
    echo "Selected episode dependency report is absent: $report" >&2; return 1
  fi
  while true; do
    load="$(systemctl show "$unit" --property=LoadState --value)" || return 1
    if [[ "$load" != loaded ]]; then
      # Historical frozen artifacts remain usable after transient units collect.
      [[ "$load" == not-found && -f "$report" ]] && return 0
      echo "Dependency unit is absent/unloaded and report missing: $unit" >&2; return 1
    fi
    state="$(systemctl show "$unit" --property=ActiveState --value)" || return 1
    case "$state" in
      active|activating|reloading|deactivating)
        if (( waited >= 43200 )); then
          echo "Dependency wait exceeded 12 hours: $unit" >&2; return 1
        fi
        sleep 30; waited=$((waited + 30)) ;;
      inactive|failed)
        if [[ "$state" == failed ]]; then
          echo "Dependency producer failed: $unit" >&2; return 1
        fi
        if [[ ! -f "$report" ]]; then
          echo "Dependency producer stopped without selected episode report: $unit / $report" >&2; return 1
        fi
        return 0 ;;
      *) echo "Unknown dependency unit state: $unit / $state" >&2; return 1 ;;
    esac
  done
}

wr_cari_dependency() {
  local mode="$1" stage
  printf -v WR_EPISODE_PADDED '%06d' "$WR_EPISODE"
  case "$mode" in
    prepare) stage=object_pose_full ;;
    forward) if (( WR_KERNEL_ONLY )); then stage=body_full; else stage=cari_inputs; fi ;;
    converter) stage=cari_forward ;;
    adapter) stage=body_full ;;
    *) echo 'Unknown CARI wrapper mode' >&2; return 2 ;;
  esac
  wr_require_dependency_report "$ROOT/outputs/episode_$WR_EPISODE_PADDED/$stage/report.json" "$WR_WAIT_FOR"
}
