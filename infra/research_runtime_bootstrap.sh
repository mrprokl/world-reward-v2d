#!/usr/bin/env bash
# Fresh VM02 only: no installation, host daemon changes, reboot or auto-retry.
set -euo pipefail
MODE="${1:---preflight}"; [[ $# -le 1 && "$MODE" =~ ^--(preflight|install)$ ]] || exit 2
ROOT="${WR_ROOT:?Require task root}"; REV="${WR_CODE_REVISION:?Require committed source}"
[[ "$(uname -s)" == Linux && "$(id -u)" == 0 && "$REV" =~ ^[0-9a-f]{40}$ ]] || exit 2
[[ "$ROOT" == /srv/scenesmith/world-reward ]] || exit 2
[[ "$ROOT" =~ ^/[a-zA-Z0-9_./-]+$ && "$(readlink -m "$ROOT")" == "$ROOT" ]] || exit 2
[[ -d "${ROOT%/*}" && -x "${ROOT%/*}" ]] || exit 2
[[ "$(id -u scenesmith)" == 1000 ]] || exit 2; GROUP="$(id -gn scenesmith)"; GID="$(id -g scenesmith)"
python3 - "$ROOT" <<'PY'
import pathlib,stat,sys
root=pathlib.Path(sys.argv[1])
for p in (root,*root.parents,root/'results'):
    if p.is_symlink() or (p.exists() and not p.is_dir()): raise ValueError('Runtime ancestors must be nonsymlink directories')
for p in (root,root/'results'):
    if p.exists():
        s=p.stat(); mode=stat.S_IMODE(s.st_mode)
        if s.st_uid!=1000 or mode & 0o022 or mode & 0o700 != 0o700:
            raise ValueError('Existing task root/results must be owner1000 writable and not group/world writable')
PY
for name in dockerd docker containerd nvidia-container-runtime python3 systemctl systemd-run timeout; do
 command -v "$name" >/dev/null || { echo "Missing VMI binary: $name" >&2; exit 2; }
done
DOCKERD="$(command -v dockerd)"; DOCKER="$(command -v docker)"; CONTAINERD="$(command -v containerd)"
NVIDIA="$(command -v nvidia-container-runtime)"; NVIDIA_CONFIG=/etc/nvidia-container-runtime/config.toml
for path in "$DOCKERD" "$DOCKER" "$CONTAINERD" "$NVIDIA"; do [[ "$path" == /* && -f "$path" && -x "$path" ]] || exit 2; done
[[ -f "$NVIDIA_CONFIG" && ! -L "$NVIDIA_CONFIG" ]] || exit 2
# Explicit1.x/2.x plugin ABI; unknown versions fail, never install or reuse host state.
CV="$($CONTAINERD --version)"
if [[ "$CV" =~ [[:space:]]v?1\.[0-9]+\. ]]; then CONFIG_VERSION=2; DISABLED='"io.containerd.grpc.v1.cri"'
elif [[ "$CV" =~ [[:space:]]v?2\.[0-9]+\. ]]; then CONFIG_VERSION=3; DISABLED='"io.containerd.cri.v1.runtime","io.containerd.cri.v1.images","io.containerd.grpc.v1.cri"'
else echo 'Unsupported containerd configuration ABI' >&2; exit 2; fi
CU=world-reward-vm02-containerd.service; DU=world-reward-vm02-docker.service
for unit in "$CU" "$DU"; do [[ "$(systemctl show "$unit" -p LoadState --value)" == not-found ]] || exit 2; done
for path in docker docker-exec containerd runtime docker.sock results/research-runtime.json; do
 [[ ! -e "$ROOT/$path" && ! -L "$ROOT/$path" ]] || { echo "Fresh runtime path occupied: $path" >&2; exit 2; }
done
[[ ! -L "$ROOT/results" && ( ! -e "$ROOT/results" || -d "$ROOT/results" ) ]] || exit 2
[[ ! -e /run/world-reward-vm02-docker && ! -L /run/world-reward-vm02-docker ]] || exit 2
printf 'runtime_preflight=pass dockerd=%s containerd=%s nvidia_runtime=%s\n' "$($DOCKERD --version)" "$($CONTAINERD --version)" "$NVIDIA"
[[ "$MODE" == --install ]] || exit 0
umask 077
for path in "$ROOT" "$ROOT/results"; do
 if [[ ! -e "$path" ]]; then mkdir -m 0755 "$path"; chown "1000:$GID" "$path"; fi
done
mkdir "$ROOT/docker" "$ROOT/docker-exec" "$ROOT/containerd" "$ROOT/runtime"
REPORT="$ROOT/results/research-runtime.json"; PHASE=config; START="$SECONDS"
trap 'code=$?; if (( code )); then printf "{\"stage\":\"world_reward_research_runtime\",\"status\":\"fail\",\"phase\":\"%s\",\"exit_code\":%d}\n" "$PHASE" "$code" > "$REPORT"; fi' EXIT
cat > "$ROOT/runtime/containerd.toml" <<EOF
version = $CONFIG_VERSION
root = "$ROOT/containerd"
state = "/run/world-reward-vm02-docker/containerd-state"
disabled_plugins = [$DISABLED]
[grpc]
  address = "/run/world-reward-vm02-docker/containerd.sock"
[debug]
  address = "/run/world-reward-vm02-docker/containerd-debug.sock"
[metrics]
  address = ""
EOF
printf '{"runtimes":{"nvidia":{"path":"%s","runtimeArgs":[]}},"features":{"containerd-snapshotter":false}}\n' "$NVIDIA" > "$ROOT/runtime/daemon.json"
chmod 400 "$ROOT/runtime/containerd.toml" "$ROOT/runtime/daemon.json"
PHASE=containerd_start
systemd-run --unit "$CU" --property=Type=exec --property=Restart=no --property=Delegate=yes \
 --property=RuntimeDirectory=world-reward-vm02-docker --property=RuntimeDirectoryMode=0700 \
 --property=RuntimeDirectoryPreserve=yes --property=TimeoutStartSec=60 \
 "$CONTAINERD" --config "$ROOT/runtime/containerd.toml"
PHASE=docker_start
systemd-run --unit "$DU" --property=Type=exec --property=Restart=no --property=Delegate=yes \
 --property="Requires=$CU" --property="After=$CU" --property="BindsTo=$CU" --property=TimeoutStartSec=60 \
 "$DOCKERD" --config-file "$ROOT/runtime/daemon.json" --data-root "$ROOT/docker" --exec-root "$ROOT/docker-exec" \
 --pidfile "$ROOT/docker-exec/dockerd.pid" --host "unix://$ROOT/docker.sock" --group "$GROUP" \
 --containerd /run/world-reward-vm02-docker/containerd.sock --containerd-namespace world-reward \
 --containerd-plugins-namespace world-reward-plugins --storage-driver overlay2 --bridge none \
 --iptables=false --ip6tables=false --ip-masq=false --ip-forward=false --userland-proxy=false
PHASE=daemon_ready; INFO="$ROOT/runtime/docker-info.json"
for attempt in {1..30}; do
 if timeout 3s "$DOCKER" --host "unix://$ROOT/docker.sock" info --format '{{json .}}' > "$INFO"; then break; fi
 [[ "$(systemctl show "$DU" -p ActiveState --value)" == active ]] || exit 2
 sleep 2
done
[[ -s "$INFO" ]] || exit 2
python3 - "$ROOT" "$REV" "$DOCKERD" "$CONTAINERD" "$NVIDIA" "$NVIDIA_CONFIG" "$((SECONDS-START))" <<'PY'
import hashlib,json,pathlib,subprocess,sys
root=pathlib.Path(sys.argv[1]); info=json.loads((root/'runtime/docker-info.json').read_text())
if info.get('DockerRootDir')!=str(root/'docker') or info.get('Driver')!='overlay2' or 'nvidia' not in info.get('Runtimes',{}):
    raise ValueError('Private daemon root/storage/NVIDIA runtime differs')
if info.get('Containers')!=0 or info.get('Images')!=0:
    raise ValueError('Fresh daemon must have no imported image or container')
files=[pathlib.Path(p) for p in sys.argv[3:7]]+[root/'runtime/containerd.toml',root/'runtime/daemon.json']
report={'stage':'world_reward_research_runtime','status':'pass','phase':'complete','producer_revision':sys.argv[2],
 'DockerRootDir':info['DockerRootDir'],'docker_server_version':info.get('ServerVersion'),
 'dockerd_version':subprocess.check_output([sys.argv[3],'--version'],text=True).strip(),
 'containerd_version':subprocess.check_output([sys.argv[4],'--version'],text=True).strip(),
 'files':[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files],
 'containerd_socket':'/run/world-reward-vm02-docker/containerd.sock','restart_policy':'no',
 'system_daemons_modified':False,'network_configuration_modified':False,'GPU_execution_verified':False,
 'image_import_verified':False,'elapsed_seconds':int(sys.argv[7])}
with (root/'results/research-runtime.json').open('x') as f: json.dump(report,f);f.write('\n')
PY
PHASE=complete; trap - EXIT
printf 'research_runtime=pass gpu_execution_verified=false image_import_verified=false\n'
