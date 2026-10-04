"""Tiny ZIP/HTTP fixtures only: no real data, Azure, media, model or GPU."""
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import stat
import struct
import sys
from types import SimpleNamespace
import zipfile
import zlib

import pytest

REPO=Path(__file__).resolve().parents[1]


@pytest.fixture
def gate(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO/"infra"))
    spec=importlib.util.spec_from_file_location("test_ycbv_acquisition",REPO/"infra/ycbv_point_acquire.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def sha(data): return hashlib.sha256(data).hexdigest()


def png(bits=8,color=2):
    ihdr=struct.pack(">IIBBBBB",640,480,bits,color,0,0,0)
    return b"\x89PNG\r\n\x1a\n"+struct.pack(">I",13)+b"IHDR"+ihdr+struct.pack(">I",zlib.crc32(b"IHDR"+ihdr))


def zipped(files):
    out=io.BytesIO()
    with zipfile.ZipFile(out,"w",compression=zipfile.ZIP_DEFLATED) as z:
        for name,data in files.items(): z.writestr(name,data)
    return out.getvalue()


def member(name="test/000048/rgb/000001.png",size=20,kind=stat.S_IFREG):
    return SimpleNamespace(filename=name,file_size=size,external_attr=kind<<16,flag_bits=0,compress_type=zipfile.ZIP_DEFLATED,is_dir=lambda:name.endswith("/"))


@pytest.mark.parametrize("fault",["traversal","absolute","backslash","nul","duplicate","symlink","fifo","dir_kind","file_kind","encrypted","compression","ancestor","member","aggregate","count","nonempty_dir"])
def test_all_zip_members_safe_before_selection(gate,fault):
    item=member();items=[item];budget=[0,0]
    if fault=="traversal": item.filename="test/../escape"
    elif fault=="absolute": item.filename="/escape"
    elif fault=="backslash": item.filename="test\\escape"
    elif fault=="nul": item.filename="test/\x00escape"
    elif fault=="duplicate": items.append(item)
    elif fault=="symlink": item.external_attr=stat.S_IFLNK<<16
    elif fault=="fifo": item.external_attr=stat.S_IFIFO<<16
    elif fault=="dir_kind": item.external_attr=stat.S_IFDIR<<16
    elif fault=="file_kind": items=[member("test/",0,stat.S_IFREG)]
    elif fault=="encrypted": item.flag_bits=1
    elif fault=="compression": item.compress_type=zipfile.ZIP_BZIP2
    elif fault=="ancestor": items=[member("test"),item]
    elif fault=="member": item.file_size=2_000_000_001
    elif fault=="aggregate": budget[0]=60_000_000_000
    elif fault=="count": budget[1]=1_000_000
    else: items=[member("test/",1,stat.S_IFDIR)]
    with pytest.raises(ValueError):gate.zip_inventory(SimpleNamespace(infolist=lambda:items),budget)


def filenames():
    files={}
    for scene in range(48,60):
        for kind in ("camera","gt","gt_info"): files[f"test/{scene:06d}/scene_{kind}.json"]=object()
        for frame in range(1,101): files[f"test/{scene:06d}/rgb/{frame:06d}.png"]=object()
    return files


def test_first_three96_filenames_only_never_labels(gate):
    members=filenames();gate.inspect_layout("ycbv_test_all.zip",members)
    selected=gate.select_rgb_names(dict(reversed(list(members.items()))))
    assert [(s,f) for s,f,_ in selected]==[(s,f) for s in (48,49,50) for f in range(1,97)]
    assert len(selected)==288
    for fault in ("missing","shift","root","sparse","model"):
        values=members.copy()
        if fault=="missing": values.pop("test/000048/rgb/000040.png")
        elif fault=="shift":values.pop("test/000048/rgb/000001.png")
        elif fault=="root":values={"ycbv/"+k:v for k,v in values.items()}
        elif fault=="sparse":values={k:v for k,v in values.items() if "/rgb/"not in k or int(Path(k).stem)%2==0}
        else: values["models/obj_000001.ply"]=b"NEVER"
        with pytest.raises(ValueError):
            gate.inspect_layout("ycbv_test_all.zip",values);gate.select_rgb_names(values)


def test_base_and_protocol_exact_no_numeric_or_source_changes(gate):
    expected=copy.deepcopy(gate.EXPECTED_PROTOCOL);gate.exact(expected,gate.EXPECTED_PROTOCOL)
    assert expected["archives"]["ycbv_test_all.zip"]["bytes"]==14969383039
    assert expected["limits"]["seconds"]==3600 and sum(x["bytes"] for x in expected["archives"].values())<30*1024**3
    for fault in ("bool","extra","sparse","scene","license"):
        value=copy.deepcopy(expected)
        if fault=="bool":value["selection"]["first_frame_position"]=False
        elif fault=="extra":value["oracle"]=False
        elif fault=="sparse":value["archives"]["ycbv_test_all.zip"]["url"]+="?alternate=bop19"
        elif fault=="scene":value["selection"]["scene_ids"]=[49,50,51]
        else:value["license"]="unknown"
        with pytest.raises(ValueError):gate.exact(value,expected)
    gate.inspect_layout("ycbv_base.zip",{k:None for k in ("ycbv/camera_cmu.json","ycbv/camera_uw.json","ycbv/dataset_info.md","ycbv/test_targets_bop19.json")})
    with pytest.raises(ValueError):gate.inspect_layout("ycbv_models.zip",{})
    for data in (b'{"k":1,"k":2}',b'{"k":NaN}'):
        with pytest.raises(ValueError):gate.strict_json(data)


@pytest.mark.parametrize("url",["http://huggingface.co/x","https://u:p@huggingface.co/x","https://huggingface.co/x?token=secret","https://huggingface.co.evil/x","https://raw.githubusercontent.com:444/x","https://example.com/x"])
def test_public_https_no_credentials_or_untrusted_redirect(gate,url):
    with pytest.raises(ValueError):gate.validate_https(url)
    with pytest.raises(ValueError):gate.SafeRedirect().redirect_request(None,None,302,"",{},url)


def test_download_stream_fullsha_and_cleanup_owned_partial(gate,tmp_path,monkeypatch):
    class Response(io.BytesIO):
        def geturl(self): return "https://cas-bridge.xethub.hf.co/x?temporary=signed"
    payload=b"immutable archive"
    opener=SimpleNamespace(open=lambda *a,**k:Response(payload))
    monkeypatch.setattr(gate.urllib.request,"build_opener",lambda *a:opener)
    for fault in (None,"size","sha"):
        path=tmp_path/(str(fault)+".zip");owned=[]
        spec={"url":"https://huggingface.co/immutable.zip","bytes":len(payload),"sha256":sha(payload)}
        if fault=="size":spec["bytes"]-=1
        elif fault=="sha":spec["sha256"]="f"*64
        old=os.umask(0o077)
        try:
            if fault:
                with pytest.raises(ValueError):gate.transfer(spec,path,owned)
            else:
                gate.transfer(spec,path,owned);assert path.read_bytes()==payload and path.stat().st_mode&0o777==0o400
            gate.cleanup(owned);assert not path.exists()
        finally:os.umask(old)


def test_cleanup_refuses_changed_unknown_symlink_hardlink(gate,tmp_path):
    owned=[];path=tmp_path/"owned";gate.save(path,b"original",owned)
    path.chmod(0o600);path.write_bytes(b"changed")
    with pytest.raises(ValueError):gate.cleanup(owned)
    assert path.read_bytes()==b"changed"
    unknown=tmp_path/"unknown";unknown.write_bytes(b"untouched")
    with pytest.raises(ValueError):gate.prune_empty(tmp_path)
    assert unknown.read_bytes()==b"untouched"
    other=tmp_path/"bound";tracked=[];gate.save(other,b"data",tracked);os.link(other,tmp_path/"hardlink")
    with pytest.raises(ValueError):gate.cleanup(tracked)


def test_crc_all_reads_unselected_corrupt_member(gate):
    data=bytearray(zipped({"unselected.bin":b"bad member"}));data[data.index(b"unselected.bin")+len("unselected.bin")+1]^=1
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        with pytest.raises((zipfile.BadZipFile,zlib.error)):gate.crc_all(archive)


def full_fixture():
    files={}
    for scene in range(48,60):
        for kind in ("camera","gt","gt_info"):
            values={str(f):({"private_K":[1,2,3],"source_frame_id":f} if kind=="camera" else [{"obj_id":1},{"obj_id":2}]) for f in range(1,101)}
            files[f"test/{scene:06d}/scene_{kind}.json"]=json.dumps(values).encode()
        for frame in range(1,101):
            prefix=f"test/{scene:06d}/";files[prefix+f"rgb/{frame:06d}.png"]=png()
            if scene<51 and frame<=96:
                files[prefix+f"depth/{frame:06d}.png"]=png(16,0)
                for folder in ("mask","mask_visib"):
                    for instance in range(2):files[prefix+f"{folder}/{frame:06d}_{instance:06d}.png"]=png(8,0)
    return files


def test_retains288_rgb_and_all_instances_private(gate,tmp_path):
    private=tmp_path/"eval_private";inputs=tmp_path/"inputs";private.mkdir(mode=0o700);inputs.mkdir();owned=[]
    files=full_fixture()
    with zipfile.ZipFile(io.BytesIO(zipped(files))) as archive:
        members=gate.zip_inventory(archive,[0,0]);selected=gate.select_rgb_names(members)
        gate.crc_all(archive);manifest=gate.retain_subset(archive,members,private,inputs,selected,owned)
    assert set(manifest)=={"schema","revision","license","selection","attribution","frame_maps","images"}
    assert len(manifest["images"])==288 and len(list(inputs.iterdir()))==288
    for row in manifest["images"]:
        assert set(row)=={"scene_id","frame_position","source_frame_id","file","sha256","width","height"}
        assert (inputs/row["file"]).read_bytes()==png() and row["sha256"]==sha(png())
        assert (inputs/row["file"]).stat().st_mode&0o777==0o444
    for scene in (48,49,50):
        folder=private/f"source/test/{scene:06d}"
        assert len(list((folder/"depth").iterdir()))==96
        assert len(list((folder/"mask").iterdir()))==192 and len(list((folder/"mask_visib").iterdir()))==192
        for kind in ("camera","gt","gt_info"):
            path=folder/f"scene_{kind}.json";value=json.loads(path.read_text())
            assert set(value)=={str(f) for f in range(1,97)} and path.stat().st_mode&0o777==0o400
            if kind=="camera": assert value["1"]["source_frame_id"]==1 and value["96"]["source_frame_id"]==96
    with pytest.raises(FileExistsError):gate.save(inputs/manifest["images"][0]["file"],png(),owned)


@pytest.mark.parametrize("fault",["selection","mask","instance","depth","rgb_type"])
def test_bad_coverage_fails_no_alternative_or_instance_choice(gate,tmp_path,fault):
    files=full_fixture()
    if fault=="mask":files.pop("test/000048/mask/000001_000001.png")
    elif fault=="instance":files["test/000048/scene_gt_info.json"]=b'{"0":[]}'
    elif fault=="depth":files.pop("test/000048/depth/000001.png")
    elif fault=="rgb_type":files["test/000048/rgb/000001.png"]=png(8,6)
    with zipfile.ZipFile(io.BytesIO(zipped(files))) as archive:
        members=gate.zip_inventory(archive,[0,0]);selected=gate.select_rgb_names(members)
        if fault=="selection":selected=selected[1:]
        with pytest.raises((ValueError,KeyError)):
            gate.retain_subset(archive,members,tmp_path/"private",tmp_path/"inputs",selected,[])


def test_full_acquire_transaction_tiny_sources_and_failure_pruning(gate,tmp_path,monkeypatch):
    protocol=copy.deepcopy(gate.EXPECTED_PROTOCOL);files=full_fixture()
    embedded=b"Permission is hereby granted, free of charge: MIT"
    base=zipped({"ycbv/camera_cmu.json":b"{}","ycbv/camera_uw.json":b"{}","ycbv/dataset_info.md":embedded,"ycbv/test_targets_bop19.json":b"[]"})
    payloads={"ycbv_base.zip":base,"ycbv_test_all.zip":zipped(files)}
    sources={"hf_readme":b"license: mit","publisher_readme":b"YCB-Video dataset is released under the MIT License","publisher_license":b"Permission is hereby granted, free of charge","bop_format":b"format","bop_params":b"params"}
    for name,value in payloads.items():protocol["archives"][name].update(bytes=len(value),sha256=sha(value))
    for name,value in sources.items():protocol["evidence"][name].update(bytes=len(value),sha256=sha(value))
    protocol["embedded_dataset_info"].update(bytes=len(embedded),sha256=sha(embedded));monkeypatch.setattr(gate,"EXPECTED_PROTOCOL",protocol)
    events=[]
    def fake_transfer(spec,path,owned):
        name=path.name;payload=payloads[name] if name in payloads else sources[name.removesuffix(".txt")]
        events.append(name);gate.save(path,payload,owned)
    monkeypatch.setattr(gate,"transfer",fake_transfer)
    old=os.umask(0o077)
    try:
        for failure in (False,True):
            out=tmp_path/str(failure);out.mkdir();(out/"eval_private").mkdir(mode=0o700);(out/"inputs").mkdir()
            report={};states=[]
            if failure:monkeypatch.setattr(gate,"retain_subset",lambda *a:(_ for _ in ()).throw(ValueError("fixture failure")))
            if failure:
                with pytest.raises(ValueError):gate.acquire(out,report,lambda:states.append(report.copy()))
                assert list(out.iterdir())==[]
            else:
                owned=gate.acquire(out,report,lambda:states.append(report.copy()))
                assert len(json.loads((out/"inputs/manifest.json").read_text())["images"])==288
                assert report["all_instances_retained"] and report["disposable_archives_removed"]
                assert not(out/"eval_private/.downloads").exists() and len(owned)>1700
            assert any(s.get("selection_before_private_annotation_values")for s in states)
    finally:os.umask(old)
    assert events.index("ycbv_base.zip")<events.index("ycbv_test_all.zip")


def test_sources_dispatch_protocol_bound_and_rehashed(gate,tmp_path,monkeypatch):
    root=tmp_path/"world-reward";root.mkdir();revision="a"*40;code=root/"jobs"/revision/gate.JOB/"code";code.mkdir(parents=True)
    for name in gate.HELPERS:
        path=code/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(json.dumps(gate.EXPECTED_PROTOCOL).encode() if name==gate.PROTOCOL else b"source");path.chmod(0o444)
    (code.parent/"revision").write_text(revision+"\n");(code.parent/"source-sha256").write_text("b"*64+"\n")
    monkeypatch.setattr(gate,"ROOT",root);monkeypatch.setattr(gate.shared,"__file__",str(code/gate.HELPERS[2]))
    before=gate.bound_source(root,code,revision,code/gate.HELPERS[0]);assert len(before["files"])==5
    path=code/gate.HELPERS[0];path.chmod(0o600);path.write_bytes(b"changed")
    with pytest.raises(ValueError):gate.bound_source(root,code,revision,path)
    path.chmod(0o444);assert gate.bound_source(root,code,revision,path)!=before
    (code.parent/"revision").write_text("c"*40+"\n")
    with pytest.raises(ValueError):gate.bound_source(root,code,revision,path)


def test_wrapper_zeroargs_offline_gpu_firewall_and_real_closure(gate):
    shell=(REPO/"infra/run_ycbv_point_acquire.sh").read_text()
    assert "[[ $# == 0 ]]"in shell and "--network host"in shell and "--gpus"not in shell
    assert "--user 1000:1000"in shell and "CUDA_VISIBLE_DEVICES="in shell and "--read-only"in shell
    assert "3810s docker run"in shell and "run_ycbv_point_acquire/code"in shell and "chown 1000:1000"in shell
    assert "--mount \"type=bind,src=$OUT,dst=$OUT\""in shell and "world-reward.job=run_ycbv_point_acquire"in shell
    spec=importlib.util.spec_from_file_location("test_ycbv_closure",REPO/"infra/azure_job.py");module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    files={str(p.relative_to(REPO)):p.read_bytes() for folder in ("infra","configs","src/world_reward") for p in (REPO/folder).rglob("*") if p.is_file() and p.suffix in (".py",".sh",".json",".cpp")}
    files["pyproject.toml"]=(REPO/"pyproject.toml").read_bytes();paths=module.runtime_bundle_paths(files,"infra/run_ycbv_point_acquire.sh")
    assert set(gate.HELPERS)|{"infra/run_ycbv_point_acquire.sh"}<=set(paths)
    assert not any("robotap"in x or "sam3d"in x for x in paths if x.startswith("infra/"))
    assert {x for x in paths if x.startswith('infra/')}=={x for x in gate.HELPERS if x.startswith('infra/')}


def test_only_acquisition_time_contract_changes_from_original(gate):
    import subprocess,ast
    original=subprocess.run(['git','show','40ee2cb4710ec15f0b2bda32bbf0e1f727804b18:infra/ycbv_point_acquire.py'],cwd=REPO,capture_output=True,check=True,text=True).stdout
    namespace={'__name__':'original_ycbv_prereg'};exec(compile(original,'original','exec'),namespace)
    expected=copy.deepcopy(namespace['EXPECTED_PROTOCOL']);expected['limits']['seconds']=3600;expected['limits']['cleanup_seconds']=180
    assert {k:gate.EXPECTED_PROTOCOL[k] for k in ('archives','evidence','embedded_dataset_info','limits')}=={k:expected[k] for k in ('archives','evidence','embedded_dataset_info','limits')}
    old=ast.parse(original);new=ast.parse(Path(gate.__file__).read_text())
    for name in ('transfer','zip_inventory','inspect_layout','crc_all','png_header','cleanup'):
        a=next(n for n in old.body if isinstance(n,ast.FunctionDef)and n.name==name)
        b=next(n for n in new.body if isinstance(n,ast.FunctionDef)and n.name==name)
        assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)


def test_absolute_cleanup_grace_does_not_renew_and_never_masks_timeout(gate,monkeypatch):
    calls=[];handlers={};report={}
    monkeypatch.setattr(gate.signal,'signal',lambda s,h:handlers.update({s:h}))
    monkeypatch.setattr(gate.signal,'alarm',lambda seconds:calls.append(seconds))
    gate.acquisition_alarm(report)
    with pytest.raises(TimeoutError,match='3600'):handlers[gate.signal.SIGALRM](None,None)
    gate.cleanup_alarm(report);gate.cleanup_alarm(report)
    assert calls==[3600,180]
    with pytest.raises(TimeoutError,match='180'):handlers[gate.signal.SIGALRM](None,None)
    assert report['cleanup_grace_exhausted']is True


def test_timeout_main_seals_failure_and_cleanup_grace_without_retry(gate,tmp_path,monkeypatch):
    root=tmp_path/'root';out=root/gate.BASE;out.mkdir(parents=True);(out/'.container.cid').write_text('c'*64)
    monkeypatch.setenv('WR_ROOT',str(root));monkeypatch.setenv('WR_CODE',str(tmp_path/'code'));monkeypatch.setenv('WR_CODE_REVISION','a'*40)
    monkeypatch.setenv('WR_IMAGE_ID',gate.IMAGE);monkeypatch.setenv('WR_AZURE_VM02_VERIFIED','1');monkeypatch.setenv('WR_YCBV_V2_AUTHORIZED','1');monkeypatch.setattr(gate,'require_execution',lambda:None)
    monkeypatch.setattr(gate.platform,'system',lambda:'Linux');monkeypatch.setattr(gate.os,'getuid',lambda:1000)
    before={'files':{gate.HELPERS[0]:{'sha256':'d'*64,'bytes':1}}};monkeypatch.setattr(gate,'bound_source',lambda *a:before)
    alarms=[];handlers={}
    def handler(s,h):old=handlers.get(s);handlers[s]=h;return old
    monkeypatch.setattr(gate.signal,'signal',handler);monkeypatch.setattr(gate.signal,'alarm',lambda s:alarms.append(s))
    def failure(output,report,persist):
        report.update(phase='download',active_archive='ycbv_test_all.zip');persist()
        handlers[gate.signal.SIGALRM](None,None)
    monkeypatch.setattr(gate,'acquire',failure)
    with pytest.raises(SystemExit):gate.main([])
    result=json.loads((out/'report.json').read_text())
    assert result['status']=='fail'and result['error_type']=='TimeoutError'and result['budget_seconds']==3600
    assert result['engineering_replay']is True and result['cleanup_grace_seconds']==180
    assert (out/'report.json').stat().st_mode&0o777==0o400
    assert alarms.count(3600)==alarms.count(180)==1 and alarms[-1]==0


def test_expired_cleanup_not_restarted_or_reported_success(gate,tmp_path,monkeypatch):
    root=tmp_path/'root';out=root/gate.BASE;out.mkdir(parents=True);(out/'.container.cid').write_text('c'*64)
    monkeypatch.setenv('WR_ROOT',str(root));monkeypatch.setenv('WR_CODE',str(tmp_path/'code'));monkeypatch.setenv('WR_CODE_REVISION','a'*40)
    monkeypatch.setenv('WR_IMAGE_ID',gate.IMAGE);monkeypatch.setenv('WR_AZURE_VM02_VERIFIED','1');monkeypatch.setenv('WR_YCBV_V2_AUTHORIZED','1');monkeypatch.setattr(gate,'require_execution',lambda:None)
    monkeypatch.setattr(gate.platform,'system',lambda:'Linux');monkeypatch.setattr(gate.os,'getuid',lambda:1000)
    monkeypatch.setattr(gate,'bound_source',lambda *a:{'files':{gate.HELPERS[0]:{'sha256':'d'*64,'bytes':1}}})
    alarms=[];handlers={}
    def handler(s,h):old=handlers.get(s);handlers[s]=h;return old
    monkeypatch.setattr(gate.signal,'signal',handler);monkeypatch.setattr(gate.signal,'alarm',lambda s:alarms.append(s))
    def expired(output,report,persist):
        gate.cleanup_alarm(report);handlers[gate.signal.SIGALRM](None,None)
    monkeypatch.setattr(gate,'acquire',expired)
    with pytest.raises(SystemExit):gate.main([])
    result=json.loads((out/'report.json').read_text())
    assert result['cleanup_grace_exhausted']is True and result['status']=='fail'
    assert result['error_type']=='TimeoutError'and (out/'report.json').stat().st_mode&0o777==0o400
    assert alarms.count(180)==1


def test_v2_requires_exact_one_replay_authorization_before_any_io(gate,monkeypatch):
    def forbidden(*args,**kwargs):pytest.fail("Unapproved acquisition reached I/O")
    monkeypatch.setattr(Path,"open",forbidden)
    monkeypatch.setattr(gate.shutil,"disk_usage",forbidden)
    monkeypatch.setattr(gate.urllib.request,"build_opener",forbidden)
    assert gate.EXPECTED_PROTOCOL["execution"]==dict(authorized=True,status="authorized_one_corrected_engineering_replay",engineering_replay=True,reason="explicit_user_authorization_2026-10-04")
    gate.require_execution()
    monkeypatch.setitem(gate.EXPECTED_PROTOCOL,"execution",dict(authorized=False,status="prepared_non_executable",engineering_replay=True))
    for call in (lambda:gate.main([]),lambda:gate.preflight(Path("/not/read"),Path("/not/read"),"a"*40)):
        with pytest.raises(ValueError,match="Exactly one explicitly authorized"):call()
    assert gate.BASE=="validation/ycbv_point_pose_v2"
    assert gate.PROTOCOL=="configs/ycbv_point_protocol_v2.json"


def test_original_v1_protocol_and_failure_pins_not_modified(gate):
    import subprocess
    names=["configs/ycbv_point_protocol.json","configs/ycbv_point_failed_acquisition_pins.json","configs/ycbv_point_inventory_failed_pins.json"]
    for name in names:
        old=subprocess.run(["git","show","HEAD:"+name],cwd=REPO,capture_output=True,check=True).stdout
        assert (REPO/name).read_bytes()==old
    assert json.loads((REPO/gate.PROTOCOL).read_bytes())==gate.EXPECTED_PROTOCOL
