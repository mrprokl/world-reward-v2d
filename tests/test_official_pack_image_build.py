"""Mocked CPU image build/network; never download a wheel or run Docker/GPU."""
import ast
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def gate():
    spec=importlib.util.spec_from_file_location("official_pack_image_build_test",ROOT/"infra/official_pack_image_build.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def metadata(gate):
    return dict(info=dict(name="pyarrow",version="19.0.1",license=gate.LICENSE,requires_python=">=3.9",
        classifiers=["License :: OSI Approved :: Apache Software License"],requires_dist=['pytest; extra == "test"']),
        urls=[dict(filename=gate.WHEEL_NAME,url=gate.WHEEL_URL,digests=dict(sha256=gate.WHEEL_SHA),size=gate.WHEEL_BYTES,
            packagetype="bdist_wheel",python_version="cp311",yanked=False)])


class Response(io.BytesIO):
    def __init__(self,data,url,content_length=None):
        super().__init__(data);self.url=url;self.headers={}
        if content_length is not None:self.headers["Content-Length"]=str(content_length)
    def geturl(self):return self.url


def tiny_wheel(gate,monkeypatch):
    data=b"own tiny test bytes not a wheel"
    monkeypatch.setattr(gate,"WHEEL_SHA",hashlib.sha256(data).hexdigest())
    monkeypatch.setattr(gate,"WHEEL_BYTES",len(data))
    return data


def test_exact_primary_pinned_wheel_license_and_no_dependency_recipe(gate):
    row=gate.validate_metadata(metadata(gate))
    assert row["sha256"]=="49a3aecb62c1be1d822f8bf629226d4a96418228a42f5b40835c1f10d42e4db6" and row["bytes"]==42084055
    assert row["license"]=="Apache Software License"
    assert gate.DOCKERFILE.startswith("FROM world-reward/cari4d-source:0.1\n")
    assert "--no-deps --no-index" in gate.DOCKERFILE and "curl" not in gate.DOCKERFILE and "numpy" not in gate.DOCKERFILE


@pytest.mark.parametrize("fault",["license","classifier","version","python","dependency","duplicate","sha","bytes","boolbytes",
    "URL","yanked","ABI","sdist"])
def test_primary_metadata_faults_cannot_change_independent_wheel(gate,fault):
    row=metadata(gate);info=row["info"];wheel=row["urls"][0]
    if fault=="license":info["license"]="unknown"
    elif fault=="classifier":info["classifiers"]=[]
    elif fault=="version":info["version"]="20.0.0"
    elif fault=="python":info["requires_python"]=">=3.12"
    elif fault=="dependency":info["requires_dist"]=["numpy>=2"]
    elif fault=="duplicate":row["urls"].append(dict(wheel))
    elif fault=="sha":wheel["digests"]["sha256"]="0"*64
    elif fault=="bytes":wheel["size"]+=1
    elif fault=="boolbytes":wheel["size"]=True
    elif fault=="URL":wheel["url"]="https://evil.example/"+gate.WHEEL_NAME
    elif fault=="yanked":wheel["yanked"]=True
    elif fault=="ABI":wheel["python_version"]="cp312"
    else:wheel["packagetype"]="sdist"
    with pytest.raises(ValueError):gate.validate_metadata(row)


@pytest.mark.parametrize("url",["http://files.pythonhosted.org/x", "https://files.pythonhosted.org:443/x",
    "https://token@files.pythonhosted.org/x","https://evil.example/x","file:///tmp/wheel", "https://files.pythonhosted.org/x?token=secret"])
def test_only_exact_primary_https_url_no_credentials_queries_or_redirects(gate,url):
    with pytest.raises(ValueError):gate.safe_wheel_url(url)
    with pytest.raises(ValueError):gate.NoRedirect().redirect_request(None,None,302,"",{},gate.WHEEL_URL)


def test_tiny_download_hash_size_before_build_no_overwrite(gate,tmp_path,monkeypatch):
    data=tiny_wheel(gate,monkeypatch);path=tmp_path/gate.WHEEL_NAME
    row=gate.download(gate.WHEEL_URL,path,gate.WHEEL_SHA,len(data),lambda:60.,
        open_url=lambda url,timeout:Response(data,url,len(data)))
    assert row["sha256"]==hashlib.sha256(data).hexdigest() and path.read_bytes()==data
    with pytest.raises(FileExistsError):gate.download(gate.WHEEL_URL,path,gate.WHEEL_SHA,len(data),lambda:60.,
        open_url=lambda url,timeout:Response(data,url,len(data)))


@pytest.mark.parametrize("fault",["short","extra","wronghash","header","redirect","expired","maxsize","untrustedhash"])
def test_wheel_transport_fails_closed(gate,tmp_path,monkeypatch,fault):
    data=tiny_wheel(gate,monkeypatch);content=data;header=len(data);url=gate.WHEEL_URL;sha=gate.WHEEL_SHA;size=len(data)
    if fault=="short":content=data[:-1]
    elif fault=="extra":content=data+b"x"
    elif fault=="wronghash":content=b"x"*len(data)
    elif fault=="header":header+=1
    elif fault=="redirect":url="https://evil.example/x"
    elif fault=="maxsize":monkeypatch.setattr(gate,"MAX_WHEEL_BYTES",len(data)-1)
    elif fault=="untrustedhash":sha="0"*64
    with pytest.raises((ValueError,TimeoutError)):gate.download(gate.WHEEL_URL,tmp_path/gate.WHEEL_NAME,sha,size,
        lambda:0. if fault=="expired" else 60.,open_url=lambda _,timeout:Response(content,url,header))


def fake_builder(gate,tmp_path,monkeypatch,fault=None):
    data=tiny_wheel(gate,monkeypatch);code=(tmp_path/"code").resolve();code.mkdir();root=tmp_path.resolve();contexts=[];commands=[]
    for name in gate.HELPERS:
        path=code/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text("immutable builder");path.chmod(0o444)
    new_id="sha256:"+"c"*64;built=False
    def probe_row(new=False):
        versions=dict(gate.EXISTING_VERSIONS)
        if new and fault=="numpy_changed":versions["numpy"]="2.0.0"
        return dict(versions=versions,pyarrow=("18.0.0" if fault=="wrong_arrow" else "19.0.1") if new else None,
            python="3.11.10",CPU_import_verified=new,Torch_loaded=fault=="torch" and new,Joblib_loaded=False)
    def runner(args,**kwargs):
        nonlocal built
        commands.append(args)
        assert kwargs["timeout"]<=300 and kwargs["env"]["DOCKER_HOST"].endswith("docker.sock")
        assert "--gpus" not in args and "pull" not in args
        if args[:3]==["docker","image","inspect"]:
            tag=args[3]
            if tag==gate.BASE_TAG:return SimpleNamespace(returncode=0,stdout=(new_id if fault=="parent" else gate.BASE_ID),stderr="")
            if tag==gate.TARGET and not built and fault!="target_exists":return SimpleNamespace(returncode=1,stdout="",stderr="Error: No such image")
            return SimpleNamespace(returncode=0,stdout=new_id,stderr="")
        if args[1]=="build":
            context=Path(args[-1]);contexts.append(context);assert context.is_dir()
            recipe=(context/"Dockerfile").read_text()
            owner=recipe.split('owner="',1)[1].split('"',1)[0]
            assert recipe==gate.DOCKERFILE.format(owner=owner) and (context/gate.WHEEL_NAME).read_bytes()==data
            assert "--pull=false" in args and args[args.index("--network")+1]=="none"
            if fault=="build":return SimpleNamespace(returncode=9,stdout="",stderr="failure")
            built=True;return SimpleNamespace(returncode=0,stdout="built",stderr="")
        if args[1]=="run":
            assert "--mount" not in args and args[args.index("--network")+1]=="none"
            return SimpleNamespace(returncode=0,stdout=json.dumps(probe_row(new_id in args)),stderr="")
        if args[1]=="ps":return SimpleNamespace(returncode=0,stdout="",stderr="")
        raise AssertionError(args)
    def open_url(url,timeout):
        assert 0<timeout<=30
        if url==gate.PYPI_URL:return Response(json.dumps(metadata(gate)).encode(),url)
        assert url==gate.WHEEL_URL
        return Response(data if fault!="download" else b"wrong",url,len(data))
    return root,code,runner,open_url,contexts,commands,new_id


def test_mocked_complete_offline_cpu_build_parent_versions_and_scratch_removed(gate,tmp_path,monkeypatch):
    root,code,runner,open_url,contexts,commands,new_id=fake_builder(gate,tmp_path,monkeypatch)
    report={"producer_revision":"a"*40};gate.build(root,code,report,lambda:None,time.monotonic(),runner=runner,open_url=open_url)
    assert report["status"]=="pass" and report["image_id"]==new_id and report["base_image_id"]==gate.BASE_ID
    assert report["CPU_probe"]["versions"]==gate.EXISTING_VERSIONS and report["CPU_probe"]["pyarrow"]=="19.0.1"
    assert all(not p.exists() for p in contexts) and report["temporary_wheel_context_removed"] is True
    assert report["only_pyarrow_added"] is True and report["parent_unchanged_verified"] is True


@pytest.mark.parametrize("fault",["parent","target_exists","build","numpy_changed","wrong_arrow","torch","download"])
def test_build_fault_never_pass_or_replace_parent_and_cleans_scratch(gate,tmp_path,monkeypatch,fault):
    root,code,runner,open_url,contexts,commands,_=fake_builder(gate,tmp_path,monkeypatch,fault)
    report={"producer_revision":"a"*40}
    with pytest.raises((ValueError,RuntimeError,FileExistsError)):gate.build(root,code,report,lambda:None,time.monotonic(),runner=runner,open_url=open_url)
    assert report.get("status")!="pass" and all(not p.exists() for p in contexts)
    assert not any(args[1:3]==["image","tag"] for args in commands)


def test_stdlib_host_source_cpu_wrapper_no_data_model_secret_closure(gate):
    source=(ROOT/"infra/official_pack_image_build.py").read_text();tree=ast.parse(source)
    imported={node.module for node in ast.walk(tree) if isinstance(node,ast.ImportFrom)}
    imported|={alias.name for node in ast.walk(tree) if isinstance(node,ast.Import) for alias in node.names}
    assert imported<={"__future__","argparse","hashlib","json","os","pathlib","platform","re","signal","stat","subprocess","sys","tempfile","time","urllib.parse","urllib.request"}
    shell=ROOT/"infra/run_official_pack_image_build.sh";text=shell.read_text()
    subprocess.run(["rtk","proxy","bash","-n",str(shell)],check=True)
    result=subprocess.run(["rtk","proxy","bash",str(shell),"--GPU"],capture_output=True,env={"PATH":os.environ["PATH"]})
    assert result.returncode==2 and "303s python3 -I -B" in text
    assert "--gpus" not in text and "--mount" not in text and "run_official_pack_image_build/code" in text
    assert gate.BUDGET==300 and gate.DOWNLOAD_BUDGET==120 and gate.MAX_WHEEL_BYTES==80*1024*1024


def test_hard_download_deadline_interrupts_blocked_read_and_removes_partial(gate,tmp_path,monkeypatch):
    import signal
    data=tiny_wheel(gate,monkeypatch);monkeypatch.setattr(gate,"DOWNLOAD_BUDGET",.02)
    prior_handler=signal.getsignal(signal.SIGALRM)
    class Blocked(Response):
        def read(self,*args):time.sleep(.2);return data
    path=tmp_path/gate.WHEEL_NAME
    with pytest.raises(TimeoutError):gate.download(gate.WHEEL_URL,path,gate.WHEEL_SHA,len(data),lambda:1.,
        open_url=lambda url,timeout:Blocked(data,url,len(data)))
    assert not path.exists() and signal.getsignal(signal.SIGALRM)==prior_handler


def test_cleanup_only_exact_owned_classic_builder_containers_never_global(gate):
    calls=[];owner="a"*64;container="b"*64
    def runner(args,**kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=0,stdout=container if args[1]=="ps" else "removed",stderr="")
    row=gate.cleanup_owned(owner,runner=runner)
    assert row["containers_removed"]==1 and row["hard_daemon_termination_proven"] is False
    assert calls[0][-1]=="label=world_reward_official_pack_owner="+owner
    assert calls[1]==["docker","rm","--force",container]
    assert not any("prune" in args or "kill" in args or gate.BASE_TAG in args for args in calls)


def test_static_runtime_builder_closure_only_two_new_source_files(gate,monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/"infra"));import azure_job
    files={str(p.relative_to(ROOT)):p.read_bytes() for folder in ("infra","src","configs") for p in (ROOT/folder).rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.suffix in (".py",".sh",".json",".toml")}
    files["pyproject.toml"]=(ROOT/"pyproject.toml").read_bytes()
    paths=azure_job.runtime_bundle_paths(files,"infra/run_official_pack_image_build.sh")
    assert {p for p in paths if p.startswith("infra/")}==set(gate.HELPERS)
