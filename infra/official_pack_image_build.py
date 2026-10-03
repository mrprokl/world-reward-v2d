"""Build one CPU packaging image from the unchanged pinned CARI parent.

Only the independently verified Apache PyArrow wheel is added, offline/no-deps.
Wheel acquisition and Docker building occur only on Azure. No videos, models,
GPU, Torch, Joblib, credentials or challenge records are loaded by this builder.
The existing parent and all five existing package versions must stay unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import signal
import stat
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request

ROOT=Path("/srv/scenesmith/world-reward")
BASE_TAG="world-reward/cari4d-source:0.1"
BASE_ID="sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7"
TARGET="world-reward/official-pack-cpu:0.1"
REPORT="results/official-pack-image-build.json"
BUDGET=300
DOWNLOAD_BUDGET=120
MAX_WHEEL_BYTES=80*1024*1024
WHEEL_NAME="pyarrow-19.0.1-cp311-cp311-manylinux_2_28_x86_64.whl"
WHEEL_SHA="49a3aecb62c1be1d822f8bf629226d4a96418228a42f5b40835c1f10d42e4db6"
WHEEL_BYTES=42084055
WHEEL_URL="https://files.pythonhosted.org/packages/b8/82/20f3c290d6e705e2ee9c1fa1d5a0869365ee477e1788073d8b548da8b64c/"+WHEEL_NAME
PYPI_URL="https://pypi.org/pypi/pyarrow/19.0.1/json"
LICENSE="Apache Software License"
EXISTING_VERSIONS={"numpy":"1.26.3","scipy":"1.16.3","pandas":"3.0.6","trimesh":"5.1.0","fast-simplification":"0.2.0"}
HELPERS=("infra/official_pack_image_build.py","infra/run_official_pack_image_build.sh")
DOCKERFILE=(f"FROM {BASE_TAG}\nLABEL world_reward_official_pack_owner=\"{{owner}}\"\nCOPY {WHEEL_NAME} /tmp/{WHEEL_NAME}\n"
    f"RUN python -m pip install --no-deps --no-index --disable-pip-version-check /tmp/{WHEEL_NAME} "
    f"&& rm /tmp/{WHEEL_NAME}\n")
PROBE="""import importlib.metadata as metadata,json,sys
names=('numpy','scipy','pandas','trimesh','fast-simplification')
versions={name:metadata.version(name) for name in names}
assert sys.version_info[:2]==(3,11)
assert 'torch' not in sys.modules and 'joblib' not in sys.modules
try: arrow=metadata.version('pyarrow')
except metadata.PackageNotFoundError: arrow=None
if arrow is not None:
 import numpy,pyarrow
 assert numpy.__version__==versions['numpy'] and pyarrow.__version__==arrow
 assert pyarrow.array([1,2,3]).to_pylist()==[1,2,3]
assert 'torch' not in sys.modules and 'joblib' not in sys.modules
print(json.dumps({'versions':versions,'pyarrow':arrow,'python':sys.version.split()[0],
 'CPU_import_verified':arrow is not None,'Torch_loaded':False,'Joblib_loaded':False},sort_keys=True))
"""


def identity(path,immutable=False):
    path=Path(path)
    if not path.is_absolute() or path.resolve()!=path or any(p.is_symlink() for p in (path,*path.parents)):
        raise ValueError("Canonical regular source required")
    before=path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_size<=0 or immutable and before.st_mode&0o222:
        raise ValueError("Nonempty immutable source required")
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):digest.update(chunk)
    after=path.lstat()
    if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(
            after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns):
        raise ValueError("Original source changed while hashing")
    return dict(sha256=digest.hexdigest(),bytes=after.st_size)


def safe_wheel_url(value):
    parsed=urllib.parse.urlsplit(value)
    if (value!=WHEEL_URL or parsed.scheme!="https" or parsed.netloc!="files.pythonhosted.org"
            or parsed.username is not None or parsed.password is not None or parsed.port is not None
            or parsed.query or parsed.fragment or not parsed.path.endswith("/"+WHEEL_NAME)):
        raise ValueError("Exact independently verified HTTPS primary wheel URL required")
    return value


def validate_metadata(record):
    """Verify current primary small JSON against independent literal wheel pins."""
    info=record.get("info") if type(record) is dict else None
    if (type(info) is not dict or info.get("name")!="pyarrow" or info.get("version")!="19.0.1"
            or info.get("license")!=LICENSE or info.get("requires_python")!=">=3.9"
            or "License :: OSI Approved :: Apache Software License" not in info.get("classifiers",[])):
        raise ValueError("Primary PyArrow19.0.1 Apache/Python package metadata required")
    dependencies=info.get("requires_dist")
    if dependencies is not None and (type(dependencies) is not list or any(type(item) is not str
            or not re.search(r";\s*extra\s*==\s*['\"]test['\"]\s*$",item) for item in dependencies)):
        raise ValueError("No runtime dependency addition permitted")
    rows=record.get("urls")
    if type(rows) is not list:raise ValueError("Primary PyPI wheel listing required")
    matches=[row for row in rows if type(row) is dict and row.get("filename")==WHEEL_NAME]
    if len(matches)!=1:raise ValueError("Exactly one pinned CPython3.11 Linuxx86_64 wheel required")
    row=matches[0];safe_wheel_url(row.get("url"))
    if (row.get("digests",{}).get("sha256")!=WHEEL_SHA or type(row.get("size")) is not int or row["size"]!=WHEEL_BYTES
            or row.get("packagetype")!="bdist_wheel" or row.get("python_version")!="cp311" or row.get("yanked") is not False):
        raise ValueError("Primary wheel SHA/size/ABI differs from independent verified pins")
    return dict(url=WHEEL_URL,filename=WHEEL_NAME,sha256=WHEEL_SHA,bytes=WHEEL_BYTES,
        license=LICENSE,license_assurance="primary_PyPI_package_metadata_not_independent_wheel_notice_audit")


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,req,fp,code,msg,headers,newurl):raise ValueError("Network redirect forbidden for pinned acquisition")


def opener(url,timeout):
    return urllib.request.build_opener(urllib.request.ProxyHandler({}),NoRedirect()).open(url,timeout=timeout)


def download(url,destination,expected_sha,expected_bytes,remaining,*,open_url=opener):
    safe_wheel_url(url)
    if expected_sha!=WHEEL_SHA or type(expected_bytes) is not int or expected_bytes!=WHEEL_BYTES or expected_bytes>MAX_WHEEL_BYTES:
        raise ValueError("Independent exact bounded wheel identity required before download")
    started=time.monotonic();digest=hashlib.sha256();count=0;destination=Path(destination)
    timeout=min(DOWNLOAD_BUDGET,remaining())
    if timeout<=0:raise TimeoutError("No wheel acquisition time remains")
    prior_handler=signal.getsignal(signal.SIGALRM);prior_timer=signal.getitimer(signal.ITIMER_REAL)
    def expired(*_):raise TimeoutError("Hard120s exact wheel acquisition deadline including network read")
    signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,timeout)
    created=False
    try:
        with open_url(url,min(30.,timeout)) as response,destination.open("xb") as stream:
            created=True
            if response.geturl()!=url:raise ValueError("Actual primary wheel endpoint differs")
            size=response.headers.get("Content-Length")
            if size is not None and (not size.isdecimal() or int(size)!=expected_bytes):raise ValueError("Wheel transport size differs")
            while True:
                left=min(remaining(),DOWNLOAD_BUDGET-(time.monotonic()-started))
                if left<=0:raise TimeoutError("Frozen120s exact wheel acquisition deadline")
                block=response.read(1024*1024)
                if not block:break
                count+=len(block)
                if count>expected_bytes or count>MAX_WHEEL_BYTES:raise ValueError("Wheel transport exceeded exact bounded size")
                stream.write(block);digest.update(block)
        if count!=expected_bytes or digest.hexdigest()!=expected_sha:raise ValueError("Downloaded wheel SHA/size differs before Docker")
    except BaseException:
        if created:destination.unlink()
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,prior_handler)
        if prior_timer[0]>0:signal.setitimer(signal.ITIMER_REAL,max(.001,prior_timer[0]-(time.monotonic()-started)),prior_timer[1])
    return dict(url=url,filename=Path(destination).name,sha256=digest.hexdigest(),bytes=count)


def metadata_request(remaining,*,open_url=opener):
    with open_url(PYPI_URL,min(20.,remaining())) as response:
        if response.geturl()!=PYPI_URL:raise ValueError("Primary PyPI metadata redirect forbidden")
        raw=response.read(2*1024*1024+1)
    if len(raw)>2*1024*1024:raise ValueError("Unexpectedly large package metadata JSON")
    record=json.loads(raw);return validate_metadata(record),dict(url=PYPI_URL,sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))


def command(args,remaining,*,runner=subprocess.run,allow_absent=False):
    result=runner(args,check=False,capture_output=True,text=True,timeout=remaining(),
        env=dict(os.environ,DOCKER_HOST="unix://"+str(ROOT/"docker.sock"),DOCKER_BUILDKIT="0"))
    if result.returncode:
        if allow_absent and re.search(r"(?:No such image|No such object)",result.stderr,re.I):return None
        raise RuntimeError("CPU image operation failed: "+args[0]+" "+args[1]+" "+str(result.returncode))
    return result.stdout.strip()


def image_id(tag,remaining,*,runner=subprocess.run,allow_absent=False):
    value=command(["docker","image","inspect",tag,"--format","{{.Id}}"],remaining,runner=runner,allow_absent=allow_absent)
    if value is not None and not re.fullmatch(r"sha256:[0-9a-f]{64}",value):raise ValueError("Exact local immutable image config ID required")
    return value


def probe(tag,remaining,*,runner=subprocess.run):
    value=command(["docker","run","--rm","--network","none","--user","1000:1000","--cpus","2","--memory","2g",
        "--env","HOME=/tmp","--env","CUDA_VISIBLE_DEVICES=",
        "--entrypoint","python",tag,"-c",PROBE],remaining,runner=runner)
    row=json.loads(value)
    if (type(row) is not dict or row.get("versions")!=EXISTING_VERSIONS or row.get("Torch_loaded") is not False
            or row.get("Joblib_loaded") is not False or type(row.get("python")) is not str or not row["python"].startswith("3.11.")):
        raise ValueError("Exact original CPU packaging package versions/runtime required")
    return row


def cleanup_owned(owner,*,runner=subprocess.run):
    """Best-effort narrowly labeled classic-builder containers; never prune."""
    if not re.fullmatch("[0-9a-f]{64}",owner):raise ValueError("Exact source-bound ownership label required")
    env=dict(os.environ,DOCKER_HOST="unix://"+str(ROOT/"docker.sock"),DOCKER_BUILDKIT="0")
    result=runner(["docker","ps","--all","--quiet","--no-trunc","--filter","label=world_reward_official_pack_owner="+owner],
        check=False,capture_output=True,text=True,timeout=5,env=env)
    if result.returncode:return dict(attempted=True,completed=False,hard_daemon_termination_proven=False)
    containers=result.stdout.split()
    if any(not re.fullmatch("[0-9a-f]{64}",value) for value in containers):raise ValueError("Exact own container IDs required")
    complete=True
    for value in containers:
        removed=runner(["docker","rm","--force",value],check=False,capture_output=True,text=True,timeout=5,env=env)
        complete &= removed.returncode==0
    return dict(attempted=True,completed=bool(complete),containers_removed=len(containers),hard_daemon_termination_proven=False)


def build(root,code,report,persist,started,*,runner=subprocess.run,open_url=opener):
    root,code=Path(root),Path(code)
    def remaining():
        left=BUDGET-(time.monotonic()-started)
        if left<=0:raise TimeoutError("Frozen300s overall CPU packaging image build deadline")
        return left
    helpers={name:identity(code/name,True) for name in HELPERS}
    if image_id(BASE_TAG,remaining,runner=runner)!=BASE_ID:raise ValueError("Actual existing parent ID differs from exact pinned CARI parent")
    if image_id(TARGET,remaining,runner=runner,allow_absent=True) is not None:raise FileExistsError("Unique packaging image already exists; no replace/rebuild")
    prior=probe(BASE_ID,remaining,runner=runner)
    if prior.get("pyarrow") is not None or prior.get("CPU_import_verified") is not False:
        raise ValueError("Original parent is not the independently observed PyArrow-absent runtime")
    report.update(source_helpers=helpers,base_image_id=BASE_ID,parent_versions=prior,phase="primary_wheel_metadata");persist()
    wheel,metadata_id=metadata_request(remaining,open_url=open_url)
    report.update(primary_metadata=metadata_id,wheel=wheel,phase="pinned_wheel_acquisition");persist()
    with tempfile.TemporaryDirectory(prefix="world-reward-official-pack-",dir="/tmp") as temporary:
        context=Path(temporary).resolve();path=context/WHEEL_NAME
        revision=report.get("producer_revision")
        if type(revision) is not str or not re.fullmatch("[0-9a-f]{40}",revision):raise ValueError("Actual original source revision required for deterministic ownership")
        owner=hashlib.sha256((revision+TARGET).encode()).hexdigest()
        dockerfile=DOCKERFILE.format(owner=owner)
        acquired=download(WHEEL_URL,path,WHEEL_SHA,WHEEL_BYTES,remaining,open_url=open_url)
        if acquired["sha256"]!=wheel["sha256"] or acquired["bytes"]!=wheel["bytes"]:raise ValueError("Primary and literal wheel identities differ")
        recipe=context/"Dockerfile";recipe.write_text(dockerfile);recipe.chmod(0o444);path.chmod(0o444)
        report.update(dockerfile=dict(content=dockerfile,sha256=identity(recipe)["sha256"],bytes=recipe.stat().st_size),
            wheel_download_verified=True,phase="offline_CPU_image_build");persist()
        if image_id(BASE_TAG,remaining,runner=runner)!=BASE_ID or image_id(TARGET,remaining,runner=runner,allow_absent=True) is not None:
            raise ValueError("Parent/unique target tag changed before offline build")
        args=["docker","build","--pull=false","--rm","--force-rm","--network","none","--tag",TARGET,"--file",str(recipe),str(context)]
        report.update(build_owner_label=owner,classic_builder=True,hard_daemon_termination_proven=False)
        try:
            command(args,remaining,runner=runner)
            built=image_id(TARGET,remaining,runner=runner)
            if built==BASE_ID:raise ValueError("Derived packaging image cannot equal the unchanged parent")
            current=probe(built,remaining,runner=runner)
            if current.get("pyarrow")!="19.0.1" or current.get("CPU_import_verified") is not True or current["versions"]!=prior["versions"]:
                raise ValueError("Actual offline CPU PyArrow import/version or parent packages differ")
            if identity(path)["sha256"]!=WHEEL_SHA or identity(recipe)["sha256"]!=report["dockerfile"]["sha256"]:
                raise ValueError("Frozen wheel/build recipe changed during Docker build")
            report.update(image_id=built,image_tag=TARGET,CPU_probe=current,build_command=args[:-2]+["<immutable-Dockerfile>","<disposable-buildcontext>"],
                phase="source_parent_final_integrity");persist()
        except BaseException:
            try:report["failed_owned_container_cleanup"]=cleanup_owned(owner,runner=runner)
            except BaseException as error:report["failed_owned_container_cleanup"]=dict(attempted=True,completed=False,error_type=type(error).__name__,hard_daemon_termination_proven=False)
            raise
    if (image_id(BASE_TAG,remaining,runner=runner)!=BASE_ID or probe(BASE_ID,remaining,runner=runner)!=prior
            or {name:identity(code/name,True) for name in HELPERS}!=helpers):
        raise ValueError("Original CARI parent/package/source identities changed")
    report.update(status="pass",phase="complete",parent_unchanged_verified=True,temporary_wheel_context_removed=True,
        runtime_dependency_installation=False,only_pyarrow_added=True,source_helpers_rehashed=True)


def main(argv=None):
    argparse.ArgumentParser(description=__doc__,allow_abbrev=False).parse_args(argv)
    root=Path(os.environ["WR_ROOT"]);code=Path(os.environ["WR_CODE"]);revision=os.environ["WR_CODE_REVISION"]
    if (platform.system()!="Linux" or platform.machine()!="x86_64" or root!=ROOT or not re.fullmatch("[0-9a-f]{40}",revision)
            or code!=root/"jobs"/revision/"run_official_pack_image_build"/"code" or Path(__file__).resolve()!=code/"infra/official_pack_image_build.py"):
        raise ValueError("Actual immutable source-bound Azure CPU builder required")
    receipt=root/REPORT
    if (receipt.exists() or receipt.is_symlink() or not receipt.parent.is_dir()
            or any(p.resolve()!=p or any(q.is_symlink() for q in (p,*p.parents)) for p in (root,code,receipt))):
        raise FileExistsError("Frozen exclusive canonical build report/source required")
    report=dict(stage="world_reward_official_pack_CPU_image_build",status="fail",phase="source_integrity",producer_revision=revision,
        script_sha256=identity(Path(__file__),True)["sha256"],budget_seconds=BUDGET,download_budget_seconds=DOWNLOAD_BUDGET,
        GPU_used=False,Torch_loaded=False,Joblib_loaded=False,challenge_inputs_used=False,data_or_models_read=False,
        network_scope="primary_PyPI_metadata_and_exact_files_pythonhosted_wheel_only",build_network="none",base_pull_performed=False,
        secret_material_used=False,license=LICENSE,license_assurance="primary_PyPI_package_metadata")
    started=time.monotonic()
    with receipt.open("x") as stream:
        def persist():
            report["elapsed_seconds"]=time.monotonic()-started;stream.seek(0);json.dump(report,stream,allow_nan=False)
            stream.write("\n");stream.truncate();stream.flush();os.fsync(stream.fileno())
        def expired(*_):raise TimeoutError("Frozen300s CPU build deadline")
        alarm=signal.signal(signal.SIGALRM,expired);term=signal.signal(signal.SIGTERM,expired);signal.alarm(BUDGET)
        try:persist();build(root,code,report,persist,started)
        except BaseException as error:report.update(status="fail",error_type=type(error).__name__,error=str(error));raise
        finally:
            signal.alarm(0);signal.signal(signal.SIGALRM,alarm);signal.signal(signal.SIGTERM,term);persist();receipt.chmod(0o444)


if __name__=="__main__":main()
