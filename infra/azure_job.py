"""Launch one immutable, committed code snapshot via Azure Run Command.

Only a small Git archive is sent from the laptop. Jobs/data/model artifacts stay
remote. Existing units and snapshots are never replaced or restarted implicitly.
"""

from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import io
import inspect
import lzma
from pathlib import Path
import re
import shlex
import subprocess
import tarfile
import json


DEFAULT_RESOURCE_GROUP = "SCENESMITH-H100"
DEFAULT_VM_NAME = "scenesmith-ncc-h100-01"
MAX_CODE_CONTROL_BYTES = 256_000
INLINE_SCRIPT_BYTES = 180_000
STAGED_SCRIPT_BYTES = 120_000
ACK_PREFIX = "WORLD_REWARD_DISPATCH_ACK_V1:"
GITHUB_RAW_ROOT = "https://raw.githubusercontent.com/mrprokl/world-reward-v2d/"


def encoded_runtime_archive(source_archive: bytes) -> tuple[str, str]:
    """Small code-only control message, independently capped from data artifacts.

    The complete statically audited closure must remain present. Do not hide
    imports or weaken provenance to meet the former arbitrary 100KB ceiling.
    """
    archive = lzma.compress(source_archive, preset=6)
    encoded = base64.b64encode(archive).decode()
    if len(encoded) > MAX_CODE_CONTROL_BYTES:
        raise RuntimeError("Run Command payload exceeds the 256KB code-only control budget")
    return encoded, hashlib.sha256(archive).hexdigest()


def azure_resource_group(value: str) -> str:
    """Conservative CLI-safe subset of Azure resource-group names, 1..90."""
    if not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9_-]{0,88}[A-Za-z0-9])?", value):
        raise argparse.ArgumentTypeError("Resource group must be 1..90 alphanumeric/underscore/hyphen characters, alphanumeric endpoints")
    return value


def azure_vm_name(value: str) -> str:
    """Linux VM target, independent of the systemd job --name, 1..64."""
    if not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,62}[A-Za-z0-9])?", value):
        raise argparse.ArgumentTypeError("VM name must be 1..64 alphanumeric/hyphen characters, alphanumeric endpoints")
    return value


def exact_commit_revision(value: str) -> str:
    """Only a complete lowercase Git commit ID, never a ref or revision expression."""
    if re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise argparse.ArgumentTypeError("Revision must be exactly40 lowercase hexadecimal commit characters")
    return value


def runtime_bundle_paths(files: dict[str, bytes], script: str) -> list[str]:
    """Select committed entrypoint/import closure, never data or unused tooling.

    Follow all static imports, including inside functions, for infra and the
    world_reward package. Include committed literal sibling source filenames
    (also in finite lists/dicts used by source-hash checks), not only imports.
    Keep package initializers, config and pyproject; omit
    unrelated experiments. Literal own-package dynamic imports are supported;
    computed own-package names and dynamic infra plugins are unsupported. This
    is not a general Python dependency resolver: external/vendor imports stay
    in the audited image. Each entrypoint retains its own immutable directory.
    """
    if not re.fullmatch(r"infra/[a-z0-9_]+\.sh", script) or script not in files:
        raise ValueError("Require a committed infra shell entrypoint")
    selected = {path for path in files if path.startswith("configs/") or path == "pyproject.toml"}
    pending = [script]
    if "src/world_reward/__init__.py" in files:
        pending.append("src/world_reward/__init__.py")

    def module_paths(module: str, *, required: bool = True) -> set[str]:
        if module != "world_reward" and not module.startswith("world_reward."):
            path = f"infra/{module.split('.')[0]}.py"
            return {path} if path in files else set()
        if not all(part.isidentifier() for part in module.split(".")):
            raise ValueError(f"Invalid own-package module: {module}")
        stem = "src/" + module.replace(".", "/")
        matches = {path for path in (stem + ".py", stem + "/__init__.py") if path in files}
        if not matches and not required:
            return set()  # A from-import name may be a symbol, not a submodule.
        if len(matches) != 1:
            raise ValueError(f"Own-package dependency is missing or ambiguous: {module}")
        parents = module.split(".")[:-1]
        for length in range(1, len(parents) + 1):
            initializer = "src/" + "/".join(parents[:length]) + "/__init__.py"
            if initializer not in files:
                raise ValueError(f"Own-package initializer is not committed: {initializer}")
            matches.add(initializer)
        return matches

    while pending:
        path = pending.pop()
        if path in selected:
            continue
        selected.add(path)
        source = files[path].decode("utf-8")
        dependencies = set()
        if path.endswith(".py"):
            tree = ast.parse(source)
            package = ""
            if path.startswith("src/"):
                module = path[4:-3].replace("/", ".")
                package = module.removesuffix(".__init__") if path.endswith("/__init__.py") else module.rpartition(".")[0]
            dynamic_names = {"__import__"}
            own_names = set()
            package_aliases = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module == "importlib":
                    dynamic_names.update(alias.asname or alias.name for alias in node.names if alias.name == "import_module")
                if isinstance(node, ast.Assign) and any(
                    isinstance(value, ast.Constant) and isinstance(value.value, str)
                    and "world_reward" in value.value for value in ast.walk(node.value)
                ):
                    own_names.update(target.id for target in node.targets if isinstance(target, ast.Name))
                if isinstance(node, ast.Import):
                    package_aliases.update(alias.asname or alias.name for alias in node.names if alias.name == "world_reward")
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str) and re.fullmatch(
                    r"infra/[a-z0-9_]+\.(?:py|sh|cpp)|src/world_reward/[a-zA-Z0-9_/]+\.py", node.value,
                ):
                    # Receipts can name code-relative helpers rather than
                    # sibling filenames. Their complete closure is mandatory.
                    if node.value not in files:
                        raise ValueError(f"Literal code-relative source dependency is not committed: {node.value}")
                    dependencies.add(node.value)
                if isinstance(node, ast.Constant) and isinstance(node.value, str) and re.fullmatch(
                    r"[a-z0-9_]+\.(?:py|sh|cpp)|Dockerfile\.[a-z0-9_]+", node.value,
                ):
                    sibling = str(Path(path).with_name(node.value))
                    # External/vendor filenames are not fabricated as our code.
                    # Known committed siblings must be frozen even when read as
                    # provenance only, e.g. Path(module.__file__).with_name(name).
                    if sibling in files:
                        dependencies.add(sibling)
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "with_name" and node.args
                        and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)
                        and re.fullmatch(r"[a-z0-9_]+\.(?:py|sh|cpp)|Dockerfile\.[a-z0-9_]+", node.args[0].value)):
                    sibling = str(Path(path).with_name(node.args[0].value))
                    if sibling not in files:
                        raise ValueError(f"Literal sibling source dependency is not committed: {sibling}")
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        dependencies.update(module_paths(alias.name))
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    if node.level:
                        parts = package.split(".") if package else []
                        if node.level > len(parts):
                            raise ValueError(f"Unresolved relative runtime import in {path}")
                        module = ".".join(parts[:len(parts) - node.level + 1] + ([module] if module else []))
                    dependencies.update(module_paths(module))
                    if module == "world_reward" or module.startswith("world_reward."):
                        for alias in node.names:
                            if alias.name != "*":
                                dependencies.update(module_paths(module + "." + alias.name, required=False))
                elif isinstance(node, ast.Call) and (
                    (isinstance(node.func, ast.Name) and node.func.id in dynamic_names)
                    or (isinstance(node.func, ast.Attribute) and node.func.attr == "import_module")
                ):
                    argument = node.args[0] if node.args else next((value.value for value in node.keywords if value.arg == "name"), None)
                    dynamic_package = node.args[1] if len(node.args) > 1 else next((value.value for value in node.keywords if value.arg == "package"), None)
                    own_dynamic_package = (isinstance(dynamic_package, ast.Constant) and isinstance(dynamic_package.value, str)
                                           and (dynamic_package.value == "world_reward" or dynamic_package.value.startswith("world_reward.")))
                    if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                        module = argument.value
                        if own_dynamic_package and module.startswith("."):
                            level = len(module) - len(module.lstrip("."))
                            parts = dynamic_package.value.split(".")
                            if level > len(parts):
                                raise ValueError(f"Unresolved relative dynamic runtime import in {path}")
                            module = ".".join(parts[:len(parts) - level + 1] + ([module[level:]] if module[level:] else []))
                        if module == "world_reward" or module.startswith("world_reward."):
                            dependencies.update(module_paths(module))
                    elif argument is not None and (own_dynamic_package or package or any(
                        (isinstance(value, ast.Constant) and isinstance(value.value, str) and "world_reward" in value.value)
                        or (isinstance(value, ast.Name) and value.id in own_names) for value in ast.walk(argument)
                    )):
                        raise ValueError(f"Computed own-package dynamic imports are unsupported: {path}")
                elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "getattr" and node.args:
                    if isinstance(node.args[0], ast.Name) and node.args[0].id in package_aliases:
                        name = node.args[1] if len(node.args) > 1 else None
                        if not isinstance(name, ast.Constant) or not isinstance(name.value, str):
                            raise ValueError(f"Computed own-package attribute imports are unsupported: {path}")
                        dependencies.update(module_paths("world_reward." + name.value, required=False))
                elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in package_aliases:
                    dependencies.update(module_paths("world_reward." + node.attr, required=False))
        # Literal $CODE/infra/foo, including Python subprocess child entrypoints.
        dependencies.update("infra/" + name for name in re.findall(
            r"/infra/([a-z0-9_]+\.(?:py|sh|cpp)|Dockerfile\.[a-z0-9_]+)", source,
        ))
        dependencies.update("src/world_reward/" + name for name in re.findall(
            r"/src/world_reward/([a-zA-Z0-9_/]+\.py)", source,
        ))
        for dependency in sorted(dependencies):
            if dependency not in files:
                raise ValueError(f"Entrypoint dependency is not committed: {dependency}")
            pending.append(dependency)
    if any(not path.startswith(("infra/", "src/", "configs/")) and path != "pyproject.toml" for path in selected):
        raise ValueError("Runtime bundle contains non-code paths")
    return sorted(selected)


def runtime_archive(full_archive: bytes, script: str) -> tuple[bytes, list[str]]:
    """Keep original Git member metadata while dropping irrelevant source files."""
    with tarfile.open(fileobj=io.BytesIO(full_archive), mode="r:") as source:
        members = {member.name: member for member in source.getmembers() if member.isfile()}
        files = {name: source.extractfile(member).read() for name, member in members.items()}
        # Reject symlinks/hardlinks rather than silently invent a runtime alias.
        if any(not member.isfile() and not member.isdir() for member in source.getmembers()):
            raise ValueError("Code snapshots cannot contain symlinks or special files")
        selected = runtime_bundle_paths(files, script)
        output = io.BytesIO()
        with tarfile.open(fileobj=output, mode="w", format=tarfile.PAX_FORMAT) as target:
            for path in selected:
                target.addfile(members[path], io.BytesIO(files[path]))
    return output.getvalue(), selected


def github_archive_descriptor(source_archive: bytes, revision: str, archive_hash: str) -> dict:
    """Preserve every TAR/PAX byte except regular-file payloads fetched remotely."""
    exact_commit_revision(revision)
    skeleton = bytearray(source_archive); rows = []
    with tarfile.open(fileobj=io.BytesIO(source_archive), mode="r:") as archive:
        for member in archive:
            path = member.name
            if (not member.isfile() or not (path.startswith(("infra/", "src/", "configs/")) or path == "pyproject.toml")
                    or "\\" in path or any(x in ("", ".", "..") for x in path.split("/"))):
                raise ValueError("Only canonical regular runtime code may use GitHub transport")
            data = archive.extractfile(member).read(); start = member.offset_data
            rows.append(dict(path=path, offset=start, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
            skeleton[start:start + len(data)] = b"\0" * len(data)
    descriptor = dict(revision=revision, tar_bytes=len(source_archive), tar_sha256=hashlib.sha256(source_archive).hexdigest(),
                      archive_sha256=archive_hash, skeleton=base64.b64encode(lzma.compress(skeleton, preset=6)).decode(), files=rows)
    if len(json.dumps(descriptor, separators=(",", ":")).encode()) > MAX_CODE_CONTROL_BYTES:
        raise RuntimeError("GitHub descriptor exceeds256KB code-only control budget")
    return descriptor


def reconstruct_github_archive(descriptor, *, opener=None):
    """Four bounded, credential/proxy/redirect-free HTTPS fetches; no code executes."""
    import concurrent.futures, time, urllib.parse, urllib.request
    deadline = time.monotonic() + 90
    def require(value):
        if not value: raise RuntimeError("Exact GitHub source retrieval failed closed")
    def remaining():
        value = deadline - time.monotonic(); require(value > 0); return value
    require(re.fullmatch(r"[0-9a-f]{40}", descriptor["revision"]) is not None)
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl): return None
    if opener is None:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    raw = bytearray(lzma.decompress(base64.b64decode(descriptor["skeleton"], validate=True)))
    require(len(raw) == descriptor["tar_bytes"])
    rows = descriptor["files"]
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as archive:
        members = archive.getmembers()
        require(len(rows) == len(members) and len({r["path"] for r in rows}) == len(rows))
        end = 0
        for row, member in zip(rows, members):
            path = row["path"]
            require(member.isfile() and member.name == path and member.offset_data == row["offset"] and member.size == row["bytes"])
            require((path.startswith(("infra/", "src/", "configs/")) or path == "pyproject.toml") and "\\" not in path and all(x not in ("", ".", "..") for x in path.split("/")))
            require(type(row["offset"]) is int and type(row["bytes"]) is int and end <= row["offset"] <= row["offset"] + row["bytes"] <= len(raw))
            end = row["offset"] + row["bytes"]; require(not any(raw[row["offset"]:end]))
    def fetch(row):
        url = GITHUB_RAW_ROOT + descriptor["revision"] + "/" + urllib.parse.quote(row["path"], safe="/")
        request = urllib.request.Request(url, headers={"Accept-Encoding": "identity"})
        with opener.open(request, timeout=remaining()) as response:
            require(response.status == 200 and response.geturl() == url and response.headers.get("Content-Encoding", "identity") == "identity")
            length = response.headers.get("Content-Length")
            require(length is None or length == str(row["bytes"]))
            data = bytearray()
            while len(data) <= row["bytes"]:
                remaining(); part = response.read(min(65536, row["bytes"] + 1 - len(data)))
                if not part: break
                data.extend(part)
            remaining(); require(len(data) == row["bytes"] and hashlib.sha256(data).hexdigest() == row["sha256"])
        return row, data
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)
    try:
        futures = [pool.submit(fetch, row) for row in rows]
        for future in futures:
            row, data = future.result(timeout=remaining()); raw[row["offset"]:row["offset"] + row["bytes"]] = data
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    require(hashlib.sha256(raw).hexdigest() == descriptor["tar_sha256"])
    compressed = lzma.compress(raw, preset=6); remaining()
    require(hashlib.sha256(compressed).hexdigest() == descriptor["archive_sha256"])
    return bytes(raw), compressed



def publication_script(descriptor, identity, script, acquisition, cleanup):
    """Shared exact-byte publication for chunked and GitHub source acquisition."""
    return f"""import base64,ctypes,hashlib,io,json,lzma,os,stat,sys,tarfile
from pathlib import Path,PurePosixPath
stage,job=map(Path,sys.argv[1:]);wanted=json.loads({descriptor!r})
def require(value):
 if not value:raise RuntimeError('Immutable code transport failed closed')
def canonical(p):return p.is_absolute() and p.resolve()==p and not any(x.is_symlink()for x in(p,*p.parents))
require(canonical(stage) and stage.is_dir() and stage.stat().st_mode&0o777==0o700)
identity_file=stage/'identity';s=identity_file.lstat()
require(stat.S_ISREG(s.st_mode) and s.st_nlink==1 and s.st_mode&0o777==0o400 and identity_file.read_bytes()=={(identity+chr(10)).encode()!r})
{acquisition}with (stage/'source.tar.xz').open('xb')as f:os.fchmod(f.fileno(),0o400);f.write(compressed);f.flush();os.fsync(f.fileno())
raw=lzma.decompress(compressed)
require(len(raw)==wanted['tar_bytes'] and hashlib.sha256(raw).hexdigest()==wanted['tar_sha256'])
with tarfile.open(fileobj=io.BytesIO(raw),mode='r:')as archive:
 members=archive.getmembers();seen=set()
 for m in members:
  path=PurePosixPath(m.name)
  require(m.isfile() and m.name not in seen and str(path)==m.name and not path.is_absolute() and '\\\\'not in m.name and all(x not in('','.','..')for x in m.name.split('/')))
  require(m.name.startswith(('infra/','src/','configs/'))or m.name=='pyproject.toml');seen.add(m.name)
 require({script!r}in seen)
 pending=stage/'pending';pending.mkdir(mode=0o755);pending.chmod(0o755);code=pending/'code';code.mkdir(mode=0o755)
 for m in members:
  target=code/m.name;target.parent.mkdir(parents=True,exist_ok=True)
  with target.open('xb')as f:os.fchmod(f.fileno(),m.mode&0o555);f.write(archive.extractfile(m).read());f.flush();os.fsync(f.fileno())
  os.utime(target,(m.mtime,m.mtime))
 for p in sorted(code.rglob('*'),reverse=True):
  if p.is_dir():p.chmod(0o555)
 code.chmod(0o555)
 for name,value in(('revision',wanted['revision']),('source-sha256',wanted['archive_sha256'])):
  with (pending/name).open('x')as f:os.fchmod(f.fileno(),0o444);f.write(value+'\\n');f.flush();os.fsync(f.fileno())
 require(canonical(job.parent.parent))
 if not job.parent.exists():job.parent.mkdir(mode=0o755);job.parent.chmod(0o755)
 require(canonical(job.parent) and job.parent.stat().st_mode&0o005==0o005 and not job.exists() and not job.is_symlink())
 def syncdir(p):
  fd=os.open(p,os.O_RDONLY|os.O_DIRECTORY)
  try:os.fsync(fd)
  finally:os.close(fd)
 for p in sorted(code.rglob('*'),reverse=True):
  if p.is_dir():syncdir(p)
 syncdir(code);syncdir(pending);syncdir(stage)
 libc=ctypes.CDLL(None,use_errno=True);rename=getattr(libc,'renameat2',None);require(rename is not None)
 rename.argtypes=(ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint);rename.restype=ctypes.c_int
 require(rename(-100,os.fsencode(pending),-100,os.fsencode(job),1)==0)
 syncdir(job.parent)
 require({{str(p.relative_to(job/'code'))for p in(job/'code').rglob('*')if p.is_file()}}==seen)
 for m in members:
  target=job/'code'/m.name;require(canonical(target))
  require(target.stat().st_mode&0o777==m.mode&0o555 and hashlib.sha256(target.read_bytes()).digest()==hashlib.sha256(archive.extractfile(m).read()).digest())
 require((job/'revision').read_bytes()==(wanted['revision']+'\\n').encode() and (job/'source-sha256').read_bytes()==(wanted['archive_sha256']+'\\n').encode())
{cleanup}
"""


def transport_commands(encoded, archive_hash, source_archive, revision, script, name, arguments, *, reuse_published=False, github_source=False):
    """Ordered code-only phases; no calls, retries or caller-selected remote paths."""
    if reuse_published and github_source:
        raise ValueError("GitHub source and published reuse are mutually exclusive")
    bundle = script.removeprefix("infra/").removesuffix(".sh")
    unit = "world-reward-" + name
    identity = hashlib.sha256((revision + script + name + archive_hash).encode()).hexdigest()
    source_digest = hashlib.sha256()
    with tarfile.open(fileobj=io.BytesIO(source_archive), mode="r:") as archive:
        for member in sorted(archive.getmembers(), key=lambda row: row.name):
            if member.isfile():
                source_digest.update(member.name.encode() + b"\0" + str(member.mode & 0o555).encode() + b"\0" + hashlib.sha256(archive.extractfile(member).read()).digest())
    prefix = f"""set -eu
set +x
umask 077
ROOT=/srv/scenesmith/world-reward
UNIT={shlex.quote(unit)}
JOB="$ROOT/jobs/{revision}/{bundle}"
STAGE="$ROOT/jobs/.runtime-stage-{revision}-{bundle}-{name}-{identity[:16]}"
/usr/bin/python3 -I -B - "$ROOT" "$JOB" "$STAGE" <<'PY_PATHS'
from pathlib import Path
import sys
for value in sys.argv[1:]:
 p=Path(value)
 if not(p.is_absolute() and p.resolve()==p and not any(x.is_symlink()for x in(p,*p.parents))):raise RuntimeError('Canonical immutable transport paths required')
root,job=map(Path,sys.argv[1:3])
for p in(root,root/'jobs',job.parent,job):
 if p.exists() and not(p.is_dir() and p.stat().st_mode&0o005==0o005):raise RuntimeError('Existing private source ancestor requires independent audit; never repair implicitly')
PY_PATHS
"""
    preflight = f"""test -d "$ROOT/jobs" && test -d "$ROOT/results" || exit 1
test ! -L "$ROOT" && test ! -L "$ROOT/jobs" && test ! -L "$ROOT/results" || exit 1
test -z "$(systemctl list-units --all --plain --no-legend "$UNIT.service")"
test ! -e "$ROOT/results/{name}.log" && test ! -L "$ROOT/results/{name}.log" || exit 1
"""
    launch = f"""test "$(cat "$JOB/revision")" = '{revision}'
test "$(cat "$JOB/source-sha256")" = '{archive_hash}'
/usr/bin/python3 -I -B - "$JOB" <<'PY_SOURCE'
import hashlib,stat,sys
from pathlib import Path
job=Path(sys.argv[1]);code=job/'code';digest=hashlib.sha256()
for p in(job.parent,job,code):
 if not(p.is_dir() and p.stat().st_mode&0o005==0o005):raise RuntimeError('Published source must be traversable without repairs')
for p in(code,*sorted(code.rglob('*'))):
 s=p.lstat()
 if p.resolve()!=p or p.is_symlink()or not(stat.S_ISDIR(s.st_mode)or stat.S_ISREG(s.st_mode)):raise RuntimeError('Unaliased original source required')
 if p.is_dir():
  if s.st_mode&0o777!=0o555:raise RuntimeError('Readonly traversable source directories required')
 else:
  if s.st_nlink!=1 or s.st_mode&0o222 or s.st_mode&0o444!=0o444:raise RuntimeError('Readonly original source file required')
  digest.update(str(p.relative_to(code)).encode()+b'\\0'+str(s.st_mode&0o777).encode()+b'\\0'+hashlib.sha256(p.read_bytes()).digest())
if digest.hexdigest()!={source_digest.hexdigest()!r}:raise RuntimeError('Actual published source bytes/modes differ from the exact archive')
PY_SOURCE
systemd-run --unit "$UNIT" --property=Type=exec \\
  --property=StandardOutput=append:"$ROOT/results/{name}.log" \\
  --property=StandardError=append:"$ROOT/results/{name}.log" \\
  /usr/bin/env WR_ROOT="$ROOT" WR_CODE="$JOB/code" WR_CODE_REVISION='{revision}' \\
  /bin/bash "$JOB/code/{script}" {' '.join(shlex.quote(value) for value in arguments)}
"""
    def phase(label, command):
        ack = ACK_PREFIX + identity + ":" + label
        return label, ack, command + f"printf '%s\\n' '{ack}'\n"
    if github_source:
        wanted = github_archive_descriptor(source_archive, revision, archive_hash)
        descriptor = json.dumps(wanted, separators=(",", ":"))
        acquisition = "require({p.name for p in stage.iterdir()}=={'identity'})\nraw,compressed=reconstruct_github_archive(wanted)\n"
        cleanup = "(stage/'source.tar.xz').unlink();identity_file.unlink();stage.rmdir()"
        publisher = publication_script(descriptor, identity, script, acquisition, cleanup)
        # Embed this transparent audited helper, never import the fetched repository.
        helper = "import re\nGITHUB_RAW_ROOT=" + repr(GITHUB_RAW_ROOT) + "\n" + inspect.getsource(reconstruct_github_archive)
        guarded = "\n".join(" " + line for line in publisher.splitlines())
        cleanup_guard = """except BaseException:
 try:
  require(canonical(stage) and (stage.stat().st_dev,stage.stat().st_ino,stage.stat().st_uid)==stage_identity and identity_file.read_bytes()==identity_bytes)
  allowed={'identity','source.tar.xz','pending/revision','pending/source-sha256',*('pending/code/'+r['path'] for r in wanted['files'])}
  parents={'.',*(str(p.parent) for n in allowed for p in (Path(n),*Path(n).parents))}
  for p in stage.rglob('*'):
   s=p.lstat();relative=str(p.relative_to(stage));require(not p.is_symlink() and p.resolve()==p)
   require((stat.S_ISREG(s.st_mode) and s.st_nlink==1 and relative in allowed)or(stat.S_ISDIR(s.st_mode) and relative in parents))
  for p in stage.rglob('*'):
   if p.is_dir():p.chmod(0o700)
  for p in sorted(stage.rglob('*'),key=lambda p:len(p.parts),reverse=True):
   if p.is_dir():p.chmod(0o700);p.rmdir()
   else:p.unlink()
  stage.rmdir()
 except BaseException:pass
 os._exit(1)
finally:
 signal.setitimer(signal.ITIMER_REAL,0)
"""
        setup = """import os,signal,stat
from pathlib import Path
stage=Path(sys.argv[1]);state=stage.lstat();stage_identity=(state.st_dev,state.st_ino,state.st_uid)
identity_file=stage/'identity';identity_bytes=identity_file.read_bytes()
def timed_out(*unused):raise TimeoutError('GitHub source90s budget exhausted')
signal.signal(signal.SIGALRM,timed_out);signal.setitimer(signal.ITIMER_REAL,90)
try:
"""
        command = prefix + preflight + f"""test ! -e "$JOB" && test ! -L "$JOB" || exit 1
test ! -e "$STAGE" && test ! -L "$STAGE" || exit 1
mkdir -m 700 "$STAGE"
(set -C; printf '%s\\n' '{identity}' > "$STAGE/identity")
chmod 400 "$STAGE/identity"
/usr/bin/python3 -I -B - "$STAGE" "$JOB" <<'PY_PUBLISH'
import sys
""" + helper + setup + guarded + "\n" + cleanup_guard + "PY_PUBLISH\n" + launch
        result = phase("github-published-dispatched", command)
        if len(result[2].encode()) > STAGED_SCRIPT_BYTES:
            raise RuntimeError("GitHub source phase exceeds bounded120KB script budget")
        return [result]
    if reuse_published:
        # Explicit metadata-only dispatch. The local archive independently binds
        # the existing remote bytes; no encoded payload or publication occurs.
        directories = {"."}
        with tarfile.open(fileobj=io.BytesIO(source_archive), mode="r:") as archive:
            for member in archive:
                if member.isfile():
                    directories.update(str(parent) for parent in Path(member.name).parents)
        directories_sha = hashlib.sha256(json.dumps(sorted(directories), separators=(",", ":")).encode()).hexdigest()
        reuse_guard = f"""/usr/bin/python3 -I -B - "$JOB" <<'PY_REUSE'
import hashlib,json,stat,sys
from pathlib import Path
job=Path(sys.argv[1]);code=job/'code'
if not(job.is_dir() and code.is_dir() and {{p.name for p in job.iterdir()}}=={{'code','revision','source-sha256'}}):raise RuntimeError('Published snapshot missing or contains unknown entries; no publication in reuse mode')
for name,expected in(('revision',{(revision+chr(10)).encode()!r}),('source-sha256',{(archive_hash+chr(10)).encode()!r})):
 p=job/name;s=p.lstat()
 if p.resolve()!=p or not stat.S_ISREG(s.st_mode)or s.st_nlink!=1 or s.st_mode&0o777!=0o444 or p.read_bytes()!=expected:raise RuntimeError('Original readonly publication marker mismatch')
dirs=sorted(['.',*(str(p.relative_to(code))for p in code.rglob('*')if p.is_dir())])
if hashlib.sha256(json.dumps(dirs,separators=(',',':')).encode()).hexdigest()!={directories_sha!r}:raise RuntimeError('Published source directory inventory differs from the exact archive')
PY_REUSE
"""
        reused = phase("published-reused-dispatched", prefix + preflight + reuse_guard + launch)
        if len(reused[2].encode()) > STAGED_SCRIPT_BYTES:
            raise RuntimeError("Published-reuse verification exceeds bounded120KB script budget")
        return [reused]
    inline = prefix + preflight + f"""if test ! -d "$JOB"; then
  test ! -e "$JOB" && test ! -L "$JOB" || exit 1
  if test ! -d "${{JOB%/*}}"; then mkdir -m 755 "${{JOB%/*}}"; chmod 755 "${{JOB%/*}}"; fi
  mkdir -m 755 "$JOB"; chmod 755 "$JOB"
  printf '%s' '{encoded}' | base64 -d > "$JOB/source.tar.xz"
  echo '{archive_hash}  '"$JOB/source.tar.xz" | sha256sum -c - >/dev/null
  mkdir "$JOB/code"
  tar -xJf "$JOB/source.tar.xz" -C "$JOB/code"
  /usr/bin/python3 -I -B - "$JOB" <<'PY_NEW_MODES'
from pathlib import Path
import sys,tarfile
job=Path(sys.argv[1]);code=job/'code'
with tarfile.open(job/'source.tar.xz','r:xz')as archive:
 for m in archive:
  if not m.isfile():raise RuntimeError('Only original regular code members allowed')
  (code/m.name).chmod(m.mode&0o555)
for p in(code,*code.rglob('*')):
 if p.is_dir():p.chmod(0o555)
PY_NEW_MODES
  rm "$JOB/source.tar.xz"
  printf '%s\\n' '{revision}' > "$JOB/revision"
  printf '%s\\n' '{archive_hash}' > "$JOB/source-sha256"
  chmod 444 "$JOB/revision" "$JOB/source-sha256"
fi
""" + launch
    inline_phase = phase("inline-dispatched", inline)
    if len(inline_phase[2].encode()) <= INLINE_SCRIPT_BYTES:
        return [inline_phase]
    chunks = [encoded[i:i + 112_000] for i in range(0, len(encoded), 112_000)]
    descriptor = json.dumps(dict(revision=revision, archive_sha256=archive_hash, encoded_bytes=len(encoded),
        tar_bytes=len(source_archive), tar_sha256=hashlib.sha256(source_archive).hexdigest(), chunks=len(chunks)), sort_keys=True)
    guard = f"""test -d "$STAGE" && test ! -L "$STAGE" || exit 1
test "$(cat "$STAGE/identity")" = '{identity}'
test ! -e "$JOB" && test ! -L "$JOB" || exit 1
"""
    commands = [phase("stage-created", prefix + preflight + f"""test ! -e "$JOB" && test ! -L "$JOB" || exit 1
test ! -e "$STAGE" && test ! -L "$STAGE" || exit 1
mkdir -m 700 "$STAGE"
(set -C; printf '%s\\n' '{identity}' > "$STAGE/identity")
chmod 400 "$STAGE/identity"
""")]
    for index, chunk in enumerate(chunks):
        prior = "" if index == 0 else f'test -f "$STAGE/chunk_{index-1:04d}" && test ! -L "$STAGE/chunk_{index-1:04d}" || exit 1\n'
        body = prefix + guard + prior + f"""test ! -e "$STAGE/chunk_{index:04d}" && test ! -L "$STAGE/chunk_{index:04d}" || exit 1
(set -C; printf '%s' '{chunk}' > "$STAGE/chunk_{index:04d}")
chmod 400 "$STAGE/chunk_{index:04d}"
"""
        commands.append(phase(f"chunk-{index:04d}-stored", body))
    acquisition = f"""names=['chunk_%04d'%i for i in range(wanted['chunks'])]
require({{p.name for p in stage.iterdir()}}=={{'identity',*names}})
parts=[]
for name in names:
 p=stage/name;s=p.lstat();require(stat.S_ISREG(s.st_mode) and s.st_nlink==1 and s.st_mode&0o777==0o400 and 0<s.st_size<=112000)
 parts.append(p.read_bytes())
encoded=b''.join(parts);require(len(encoded)==wanted['encoded_bytes'])
compressed=base64.b64decode(encoded,validate=True)
require(hashlib.sha256(compressed).hexdigest()==wanted['archive_sha256'])
"""
    cleanup = "for name in ['identity',*names,'source.tar.xz']:(stage/name).unlink()\nstage.rmdir()"
    publish = prefix + preflight + guard + "/usr/bin/python3 -I -B - \"$STAGE\" \"$JOB\" <<'PY_PUBLISH'\n" + publication_script(descriptor, identity, script, acquisition, cleanup) + "PY_PUBLISH\n" + launch
    commands.append(phase("published-dispatched", publish))
    if any(len(command.encode()) > STAGED_SCRIPT_BYTES for _, _, command in commands):
        raise RuntimeError("Code transport phase exceeds bounded120KB script budget")
    return commands


def invoke_transport(command, ack, resource_group, vm_name):
    """Require explicit remote phase evidence; an empty Azure success is unknown."""
    try:
        result = subprocess.run(["rtk", "proxy", "az", "vm", "run-command", "invoke",
            "--resource-group", resource_group, "--name", vm_name, "--command-id", "RunShellScript", "--scripts", command,
            "--query", "value[0].message", "-o", "tsv"], check=False, capture_output=True, timeout=300)
        stdout = result.stdout
        if isinstance(stdout, bytes): stdout = stdout.decode("utf-8", errors="strict")
        acknowledged = [line for line in stdout.splitlines() if line.startswith(ACK_PREFIX)]
        if result.returncode != 0 or len(stdout.encode()) > 2_000_000 or acknowledged != [ack]:
            raise RuntimeError("Unknown dispatch state")
    except Exception:
        raise RuntimeError("Azure phase unacknowledged; inspect target before any retry (payload omitted)") from None

def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--name", required=True)
    parser.add_argument("--script", required=True, help="Committed infra/*.sh entrypoint")
    parser.add_argument("--revision", type=exact_commit_revision,
                        help="Exact immutable commit; ignores disjoint local changes and archives only this commit")
    parser.add_argument("--resource-group", type=azure_resource_group, default=DEFAULT_RESOURCE_GROUP)
    parser.add_argument("--vm-name", type=azure_vm_name, default=DEFAULT_VM_NAME,
                        help="Azure target VM; --name remains only the immutable job name")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--reuse-published", action="store_true",
                        help="Verify and dispatch the exact existing snapshot only; no upload, repair, retry or publication")
    modes.add_argument("--github-source", action="store_true",
                       help="Fetch only the exact public World Reward commit closure on Azure; no credentials, retry or fallback")
    parser.add_argument("arguments", nargs="*")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[a-z][a-z0-9-]{0,50}", args.name):
        raise ValueError("Job name must be a short lowercase slug")
    if not re.fullmatch(r"infra/[a-z0-9_]+\.sh", args.script):
        raise ValueError("Only explicit committed infra shell entrypoints are allowed")
    # RTK also wraps child shell-facing CLIs on the local host.
    def git(*arguments: str) -> bytes:
        return subprocess.check_output(["rtk", "proxy", "git", *arguments])
    if args.revision is None:
        if git("status", "--porcelain", "--untracked-files=all").strip():
            raise RuntimeError("Commit/clean the worktree before launching a reproducible job")
        revision = git("rev-parse", "HEAD").decode().strip()
    else:
        revision = args.revision
        if git("cat-file", "-t", revision).strip() != b"commit":
            raise ValueError("Explicit revision must identify a Git commit, not a tree/blob/tag")
        if git("rev-parse", "--verify", revision + "^{commit}").decode().strip() != revision:
            raise ValueError("Resolved commit differs from the exact caller revision")
    git("cat-file", "-e", f"{revision}:{args.script}")
    source_archive, paths = runtime_archive(git("archive", "--format=tar", revision,
                                               "infra", "src", "configs", "pyproject.toml"), args.script)
    encoded, archive_hash = encoded_runtime_archive(source_archive)
    commands = transport_commands(encoded, archive_hash, source_archive, revision, args.script, args.name, args.arguments,
                                  reuse_published=args.reuse_published, github_source=args.github_source)
    print(f"immutable_runtime_bundle_files={len(paths)} encoded_bytes={len(encoded)} revision={revision} transport_phases={len(commands)}", flush=True)
    for phase, ack, command in commands:
        invoke_transport(command, ack, args.resource_group, args.vm_name)
        print(f"immutable_transport_phase={phase} acknowledged=true dispatch_only=true", flush=True)



if __name__ == "__main__":
    main()
