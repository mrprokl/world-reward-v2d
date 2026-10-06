"""Tiny manufactured JPEG headers/catalogues; no real pixels, roles or HTTP."""
import ast
import base64
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import stat
import struct
import time
import zipfile

import pytest
import vcoco_fit_cal_acquire as p
import test_vcoco_fit_cal_freeze as helpers


def rows():
    result=[]
    for i in range(48):
        partition="train2014"if i%2==0 else"val2014";iid=1000+i;name=f"COCO_{partition}_{iid:012d}.jpg"
        result.append(dict(slot=i,split="FIT"if i<32 else"CAL",official_split="train"if i<32 else"val",
            image_id=iid,photo_id=str(100000+i),rank_sha256=hashlib.sha256((p.context.selection.NAMESPACE+f"{iid:012d}").encode()).hexdigest(),
            width=3,height=2,partition=partition,file_name=name,original_url=p.prep.BUCKET+partition+"/"+name,
            license_id=4,license_url=p.c.GRANT,creator_identity="UNKNOWN"))
    return result


def jpeg(i,width=3,height=2):
    # Header qualification fixture only, NOT a decodable image.
    return b"\xff\xd8\xff\xc0"+struct.pack(">HBHHB",17,8,height,width,3)+bytes([1,17,0,2,17,0,3,17,0,i])


def catalogue(tmp_path,mutate=lambda values:None):
    selected=rows();values={}
    for partition in("train2014","val2014"):
        images=[dict(id=r["image_id"],file_name=r["file_name"],width=r["width"],height=r["height"],license=4,
            coco_url="http://images.cocodataset.org/"+partition+"/"+r["file_name"],
            flickr_url=f"http://farm1.staticflickr.com/1/{r['photo_id']}_abc.jpg")for r in selected if r["partition"]==partition]
        values[partition]=dict(images=images,licenses=[dict(id=4,url=p.c.GRANT)])
    mutate(values);archive=tmp_path/"catalogue.zip";members=[];expanded=[]
    with zipfile.ZipFile(archive,"w",compression=zipfile.ZIP_DEFLATED)as stream:
        for partition,value in values.items():
            # Excluded semantic poison is lexically traversed, never json decoded.
            raw=("{\"annotations\":[{\"bbox\":1e999,\"image_id\":999}],"+json.dumps(value)[1:]).encode()
            name="annotations/instances_"+partition+".json";stream.writestr(name,raw)
            expanded.append(dict(member=name,expanded_bytes=len(raw),expanded_sha256=hashlib.sha256(raw).hexdigest()))
    archive.chmod(0o400)
    with zipfile.ZipFile(archive)as stream:
        members=[dict(member=i.filename,expanded_bytes=i.file_size,crc32=f"{i.CRC:08x}")for i in stream.infolist()]
    report=tmp_path/"metadata_report.json";pin=helpers.save(report,dict(archive_members=expanded))
    cfg=dict(inputs=dict(coco_archive=dict(path=str(archive)),metadata_report=dict(path=str(report),pin=pin)),
        archive=dict(member_catalogue=members,instances_members={n:"annotations/instances_"+n+".json"for n in values}),
        max_expanded_bytes=1<<20,max_row_bytes=1<<14)
    cohort=dict(records=[{k:r[k]for k in("slot","split","official_split","image_id","photo_id","rank_sha256")}for r in selected])
    return cfg,cohort,selected


def test_source_recipe_no_old_execution_or_global_mutation():
    tree=ast.parse(Path(p.__file__).read_text())
    forbidden={"run","census","reproduce","census_population","freeze_fit_cal_cohort","project_vcoco_actions","parse_vcoco_role_reference"}
    for node in ast.walk(tree):
        if isinstance(node,ast.Call)and isinstance(node.func,ast.Attribute)and isinstance(node.func.value,ast.Name):
            if node.func.value.id in("c","prep","context"):assert node.func.attr not in forbidden
    assert p.RECIPE["all_48_acquired_required"]is True and p.RECIPE["selected_slots"]==48
    assert p.RECIPE["budget_seconds"]==1200 and p.RECIPE["http_budget_seconds"]==300
    assert p.RECIPE["max_total_image_bytes"]==48*p.RECIPE["max_image_bytes"]
    assert p.context.freeze.ENTRY=="run_vcoco_fit_cal_freeze"and p.prep.ENTRY=="run_vcoco_role_prepare"


def test_real_stream_catalogue_only_metadata_with_eof_crc_and_sha(tmp_path):
    cfg,cohort,expected=catalogue(tmp_path);actual,expanded=p.catalog(cfg,cohort,lambda:None)
    assert actual==expected and len(expanded)==2
    assert p.rt.identity(Path(cfg["inputs"]["coco_archive"]["path"]),readonly=True)["bytes"]<10000


@pytest.mark.parametrize("fault",["grid","grant","photo","name","url","missing","duplicates","expanded"])
def test_catalogue_mismatch_no_http(tmp_path,fault):
    def mutate(values):
        image=values["train2014"]["images"][0]
        if fault=="grid":image["width"]=True
        elif fault=="grant":image["license"]=1
        elif fault=="photo":image["flickr_url"]="http://farm1.staticflickr.com/1/999_abc.jpg"
        elif fault=="name":image["file_name"]="alias.jpg"
        elif fault=="url":image["coco_url"]="https://example.invalid/alias.jpg"
        elif fault=="missing":values["train2014"]["images"].pop()
        elif fault=="duplicates":values["val2014"]["images"].append(dict(image))
    cfg,cohort,_=catalogue(tmp_path,mutate)
    if fault=="expanded":
        path=Path(cfg["inputs"]["metadata_report"]["path"]);raw=p.rt.strict(path.read_bytes());path.chmod(0o600);raw["archive_members"][0]["expanded_sha256"]="f"*64
        cfg["inputs"]["metadata_report"]["pin"]=helpers.save(path,raw)
    with pytest.raises(ValueError):p.catalog(cfg,cohort,lambda:None)


def test_all48_exact_one_request_original_headers_opaque_manifest(tmp_path):
    out=tmp_path/"inputs";out.mkdir(mode=0o700);ledger={};calls=[];selected=rows();by_url={r["original_url"]:r["slot"]for r in selected}
    def request(url,maximum,deadline,timeout):calls.append(url);assert maximum==16<<20 and timeout==15;return jpeg(by_url[url])
    result=p.acquire(selected,dict(md5=set()),out,time.monotonic()+30,ledger,request=request)
    assert result["availability_gate_passed"]is True and result['acquisition_records_complete']is True and result["counts"]==dict(slots=48,acquired=48,missing=0,acquired_FIT=32,acquired_CAL=16)
    assert len(calls)==len(set(calls))==48 and len(ledger)==49
    manifest=p.rt.strict((out/"manifest.json").read_bytes())
    assert len(manifest["images"])==48 and all(set(r)==p.prep.PUBLIC_KEYS for r in manifest["images"])
    assert all(r["image_id"]not in{str(v["image_id"])for v in selected}for r in manifest["images"])
    assert all(stat.S_IMODE(q.lstat().st_mode)==0o400 for q in out.iterdir())
    assert all(r["jpeg_header"]["decoded_content_verified"]is False for r in result["records"])


@pytest.mark.parametrize("fault",["missing","historical_md5","header","duplicates","write","expired"])
def test_fixed48_failure_no_subset_or_replacements(tmp_path,monkeypatch,fault):
    out=tmp_path/"inputs";out.mkdir(mode=0o700);ledger={};selected=rows();calls=[];by_url={r["original_url"]:r["slot"]for r in selected}
    def request(url,*_):
        i=by_url[url];calls.append(i)
        if fault=="missing"and i==33:raise OSError("private server detail")
        return jpeg(0 if fault=="duplicates"else i,width=4 if fault=="header"and i==3 else 3)
    exclude={base64.b64encode(hashlib.md5(jpeg(3)).digest()).decode()}if fault=="historical_md5"else set()
    deadline=time.monotonic()-1 if fault=="expired"else time.monotonic()+30
    if fault=="write":
        original=p.prep.raw_write
        def failure(path,raw,ledger):
            if path.name=="image_000003.jpg":raise OSError("write unavailable")
            return original(path,raw,ledger)
        monkeypatch.setattr(p.prep,"raw_write",failure)
    result=p.acquire(selected,dict(md5=exclude),out,deadline,ledger,request=request)
    assert result["availability_gate_passed"]is False and len(result["records"])==48
    assert result["counts"]["slots"]==48 and result["decision"].startswith("CLOSED_")
    assert len(calls)<=48 and len(calls)==len(set(calls))
    if fault=="duplicates":assert result["counts"]["acquired"]==48 and result["duplicate_new_bytes"]is True and len(ledger)==49
    elif fault=="expired":assert not calls and result["counts"]["acquired"]==0
    else:assert result["counts"]["acquired"]<48
    assert "private server detail"not in p.c.encode(result).decode()
    if fault=="write":assert not(out/"image_000003.jpg").exists()and"image_000003.jpg"not in ledger


def test_original_raw_writer_removes_only_own_partial_on_directory_sync_failure(tmp_path,monkeypatch):
    path=tmp_path/'image_000000.jpg';ledger={}
    monkeypatch.setattr(p.prep.acq,'sync_directory',lambda _:(_ for _ in()).throw(OSError('dirsync')))
    with pytest.raises(OSError):p.prep.raw_write(path,jpeg(0),ledger)
    assert not path.exists()and not ledger


@pytest.mark.parametrize("fault",["slots","boolean_slot","split","url","name","grant","grid"])
def test_invalid_pre_request_bank_never_requests(tmp_path,fault):
    selected=rows();r=selected[0]
    if fault=='boolean_slot':selected[1]['slot']=True
    else:r[{"slots":"slot","split":"split","url":"original_url","name":"file_name","grant":"license_id","grid":"height"}[fault]]={"slots":1,"split":"CAL","url":"https://example.invalid/a","name":"alias","grant":0,"grid":0}[fault]
    calls=[]
    with pytest.raises(ValueError):p.acquire(selected,dict(md5=set()),tmp_path,time.monotonic()+30,{},request=lambda *a:calls.append(a))
    assert not calls


def runtime(tmp_path,monkeypatch):
    data=tmp_path/"data";data.mkdir(mode=0o700);meta=data/"metadata";meta.mkdir(mode=0o700)
    helpers.save(meta/"report.json",b"original freeze");helpers.save(meta/"cohort.json",b"original identities")
    meta.chmod(0o500)
    helpers.root_stat(monkeypatch,data);monkeypatch.setattr(p,"DATA",data)
    code=tmp_path/"code";code.mkdir();monkeypatch.setattr(p,"__file__",str(code/"infra/vcoco_fit_cal_acquire.py"))
    monkeypatch.setenv("WR_CODE",str(code));monkeypatch.setenv("WR_CODE_REVISION","d"*40)
    monkeypatch.setattr(p.os,"geteuid",lambda:0);monkeypatch.setattr(p.sys,"platform","linux")
    class Host:nodename="world-reward-ncc-h100-02"
    monkeypatch.setattr(p.os,"uname",lambda:Host())
    class Usage:ru_maxrss=1024
    monkeypatch.setattr(p.resource,"getrusage",lambda *_:Usage())
    cfg,cohort,selected=catalogue(tmp_path);proof=dict(current_source={"actual":True},freeze_source={"old":True},live_states={"old":1})
    calls=[]
    def auth(*_):calls.append(1);return cfg,dict(md5=set()),cohort,deepcopy(proof)
    monkeypatch.setattr(p,"authenticate",auth)
    acquire=p.acquire;by_url={r["original_url"]:r["slot"]for r in selected}
    def acquire_fake(*args):return acquire(*args,request=lambda url,*_:jpeg(by_url[url]))
    monkeypatch.setattr(p,"acquire",acquire_fake)
    return data,calls,proof


def test_complete_runtime_auth_twice_and_private_public_seals(tmp_path,monkeypatch):
    data,calls,proof=runtime(tmp_path,monkeypatch);original={q.name:(q.read_bytes(),p.c.state(q))for q in(data/"metadata").iterdir()}
    r=p.run(time.monotonic(),time.monotonic()+30)
    assert r["status"]=="pass"and r["stage"]=="complete"and r["counts"]["acquired"]==48 and len(calls)==2
    assert r["input_proof"]==proof and r["source_and_inputs_rehashed_after"]is r["outputs_sealed"]is True
    assert {q.name:(q.read_bytes(),p.c.state(q))for q in(data/"metadata").iterdir()}==original
    assert stat.S_IMODE((data/"inputs").lstat().st_mode)==stat.S_IMODE((data/"acquisition").lstat().st_mode)==0o500
    assert all(stat.S_IMODE(q.lstat().st_mode)==0o400 for d in("inputs","acquisition")for q in(data/d).iterdir())
    for k in("annotation_values_consulted","role_values_consulted","pilot_reference_values_read","RGB_decoded","GPU_used","models_loaded","FIT_performed","CAL_evaluated","selection_performed","adopted"):assert r[k]is False


@pytest.mark.parametrize("fault",["catalogue","posthash","foreign","late_deadline","existing"])
def test_runtime_failures_no_pass_receipt(tmp_path,monkeypatch,fault):
    data,calls,_=runtime(tmp_path,monkeypatch)
    if fault=="catalogue":monkeypatch.setattr(p,"catalog",lambda *a:(_ for _ in()).throw(ValueError("invalid metadata")))
    elif fault=="posthash":
        auth=p.authenticate
        def change(*a):
            v=auth(*a)
            if len(calls)>1:v[3]["live_states"]["old"]=2
            return v
        monkeypatch.setattr(p,"authenticate",change)
    elif fault=="foreign":
        acquire=p.acquire
        def foreign(*a):
            v=acquire(*a);helpers.save(data/"inputs/foreign",b"unknown");return v
        monkeypatch.setattr(p,"acquire",foreign)
    elif fault=="late_deadline":
        original=p.c.check
        def expiry(d):
            if len(calls)>1:raise ValueError("expired")
            original(d)
        monkeypatch.setattr(p.c,"check",expiry)
    else:(data/"inputs").mkdir(mode=0o700)
    if fault in("foreign","existing"):
        with pytest.raises(ValueError):p.run(time.monotonic(),time.monotonic()+30)
        assert not(data/"acquisition/report.json").exists()
    else:
        r=p.run(time.monotonic(),time.monotonic()+30);assert r["status"]=="fail"
        assert p.rt.strict((data/"acquisition/report.json").read_bytes())["status"]=="fail"
        if fault=="catalogue":assert len(r["records"])==48 and r["counts"]["missing"]==48 and not r["network_used"]


def test_alarm_before_preflight_and_restored(monkeypatch):
    events=[];monkeypatch.setattr(p.signal,"signal",lambda s,h:events.append(("signal",s))or "old")
    monkeypatch.setattr(p.signal,"setitimer",lambda *a:events.append(("timer",a[1])))
    monkeypatch.setattr(p.resource,"setrlimit",lambda *a:None)
    monkeypatch.setattr(p.os,"sched_setaffinity",lambda *a:None,raising=False)
    monkeypatch.setattr(p,"run",lambda *a:events.append(("run",))or dict(status="pass"))
    assert p.main()==dict(status="pass")and events.index(("timer",1200))<events.index(("run",))and("timer",0)in events


@pytest.mark.parametrize('fault',['deadline','foreign_leaf','parent','report_inode','public_inode','sync'])
def test_publication_ownership_and_same_fd_late_failure(tmp_path,monkeypatch,fault):
    data=tmp_path/'data';data.mkdir(mode=0o700);public=data/'inputs';private=data/'acquisition'
    public.mkdir(mode=0o700);private.mkdir(mode=0o700);helpers.root_stat(monkeypatch,data);monkeypatch.setattr(p,'DATA',data)
    parent=p.prep.namespace_identity(data);op,oq=public.lstat(),private.lstat();ledger={}
    p.prep.write(public/'manifest.json',dict(schema='world_reward.rgb_proposal_inputs.v1',images=[]),ledger)
    report=dict(status='pass',decision='pending',outputs_sealed=False)
    if fault=='deadline':monkeypatch.setattr(p.c,'check',lambda _:(_ for _ in()).throw(ValueError('expired')))
    elif fault=='foreign_leaf':helpers.save(public/'foreign',b'unowned')
    elif fault=='parent':data.chmod(0o755)
    else:
        sync=p.prep.acq.sync_directory;done=[]
        def malicious(path):
            if not done and path==public:
                done.append(1)
                if fault=='report_inode':
                    private.chmod(0o700);(private/'report.json').rename(private/'original.json');helpers.save(private/'report.json',b'foreign');private.chmod(0o500)
                elif fault=='public_inode':
                    public.rename(data/'original_inputs');public.mkdir(mode=0o700);helpers.save(public/'manifest.json',b'foreign')
                else:raise OSError('bounded sync failure')
            return sync(path)
        monkeypatch.setattr(p.prep.acq,'sync_directory',malicious)
    if fault in('foreign_leaf','parent','report_inode'):
        with pytest.raises(ValueError):p.publish(public,private,report,time.monotonic()+30,time.monotonic(),op,oq,ledger,{},parent)
        if fault=='report_inode':assert(private/'report.json').read_bytes()==b'foreign'
    else:
        p.publish(public,private,report,time.monotonic()+30,time.monotonic(),op,oq,ledger,{},parent)
        assert report['status']=='fail'and p.rt.strict((private/'report.json').read_bytes())['status']=='fail'
        if fault=='public_inode':assert(public/'manifest.json').read_bytes()==b'foreign'
