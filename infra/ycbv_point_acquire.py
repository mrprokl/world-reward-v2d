"""Azure-only full YCBV acquisition; filename-frozen RGBs public, labels private."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import signal
import stat
import struct
import time
import urllib.parse
import urllib.request
import zipfile
import zlib

import tudl_acquire as shared
from world_reward.native_frame_map import NativeFrameMap

ACQUISITION_SECONDS = 3600
CLEANUP_SECONDS = 180

ROOT = Path("/srv/scenesmith/world-reward")
JOB = "run_ycbv_point_acquire"
BASE = "validation/ycbv_point_pose_v2"
STAGE = "external_ycbv_contiguous_rgb_only_acquisition"
IMAGE = "sha256:7ebfff18ba3b76dd919485c19115597d7531dfd3233f69461f1dce3f28a6c6d3"
REVISION = "5c2c4aa229800355648cd268040aa814f8dc94f0"
URL = f"https://huggingface.co/datasets/bop-benchmark/ycbv/resolve/{REVISION}/"
PUBLISHER = "https://raw.githubusercontent.com/yuxng/YCB_Video_toolbox/98630204fc73c1dbc2ff66ddea22cf517f269f5f/"
PROTOCOL = "configs/ycbv_point_protocol_v2.json"
HELPERS = ("infra/ycbv_point_acquire.py", "infra/run_ycbv_point_acquire.sh", "infra/tudl_acquire.py", "src/world_reward/native_frame_map.py", PROTOCOL)
SELECTION = "first_three_sorted_scene_directories_first_96_contiguous_RGB_names_before_private_annotations"
EXPECTED_PROTOCOL = {
 "schema":"world-reward-ycbv-point-protocol-v2", "dataset_revision":REVISION, "license":"MIT",
 "archives":{
  "ycbv_base.zip":{"url":URL+"ycbv_base.zip","bytes":15805,"sha256":"98440f8bd403100b21cf11a6729fabe8b3d5ce714472edc57a18b7f1fcd4bb18"},
  "ycbv_test_all.zip":{"url":URL+"ycbv_test_all.zip","bytes":14969383039,"sha256":"fea2ab5f18aba1857acd320827cec10d9dbf258e4940ea4b52f5dd51cb2356a7"}},
 "evidence":{
  "hf_readme":{"url":f"https://huggingface.co/datasets/bop-benchmark/ycbv/raw/{REVISION}/README.md","bytes":24,"sha256":"d8d7a46d41a1a37fe4f0a5f637bf55c649310185329127d8a2204632e480be17"},
  "publisher_readme":{"url":PUBLISHER+"README.md","bytes":2787,"sha256":"12d222b033b2c74d7b1d7fca93175d5a8eae5e40f83b050631d949e9c6ee8ee0"},
  "publisher_license":{"url":PUBLISHER+"LICENSE","bytes":1065,"sha256":"8a236ca04573bec835d33fb30b98b5bd3c551718a6ef7b33854d9202a8e2d167"},
  "bop_format":{"url":"https://raw.githubusercontent.com/thodan/bop_toolkit/af97c1938083dfd512eb6f32a85921ea38198ee4/docs/bop_datasets_format.md","bytes":9821,"sha256":"6994afd65c29c2a198ea6e08e138f586ace316aaacda21ae3c498b37d1535222"},
  "bop_params":{"url":"https://raw.githubusercontent.com/thodan/bop_toolkit/b72b3015c87a96fa6398c2ef4c196e85f798d3e6/bop_toolkit_lib/dataset_params.py","bytes":32303,"sha256":"a935fc4f6fd42f367a0bbf816036f783fb4c5d87200a8615ce1e08e6d51af05e"}},
 "embedded_dataset_info":{"file":"ycbv/dataset_info.md","bytes":4035,"sha256":"4766684f25f165c1c312745e3583fe248161c56c35a862bfa04134141151882e"},
 "selection":{"split_prefix":"test","scene_ids":[48,49,50],"frames_per_scene":96,"first_source_frame_id":1,"first_frame_position":0,"width":640,"height":480,"rule":SELECTION},
 "limits":{"seconds":3600,"cleanup_seconds":180,"download_bytes":32212254720,"expanded_bytes":60000000000,"member_bytes":2000000000,"members":1000000,"min_free_bytes":45000000000},
 "execution":{"authorized":False,"status":"prepared_non_executable","engineering_replay":True},
 "output":{"base":BASE,"public_schema":"world-reward-ycbv-point-rgb-v2"}}
FIELDS = lambda s:(s.st_dev,s.st_ino,s.st_mode,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
CHUNK = 1024*1024


def canonical(path):
    path = Path(path)
    if not path.is_absolute() or path.resolve()!=path or any(p.is_symlink() for p in (path,*path.parents)):
        raise ValueError("Canonical nonsymlink path required")
    return path


def identity(path, readonly=True):
    path=canonical(path); before=path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_size<=0 or (readonly and before.st_mode&0o222):
        raise ValueError("Nonempty readonly regular source required")
    value={"bytes":before.st_size,"sha256":shared.digest(path)}
    if FIELDS(before)!=FIELDS(path.lstat()): raise ValueError("File changed during hashing")
    return value


def strict_json(data):
    def pairs(rows):
        result={}
        for k,v in rows:
            if k in result: raise ValueError("Duplicate JSON keys")
            result[k]=v
        return result
    return json.loads(data,object_pairs_hook=pairs,parse_constant=lambda _:(_ for _ in ()).throw(ValueError("Nonfinite JSON")))


def exact(actual, expected):
    if type(actual) is not type(expected): raise ValueError("Protocol type differs")
    if isinstance(expected,dict):
        if set(actual)!=set(expected): raise ValueError("Protocol keys differ")
        for k in expected: exact(actual[k],expected[k])
    elif isinstance(expected,list):
        if len(actual)!=len(expected): raise ValueError("Protocol length differs")
        for a,b in zip(actual,expected): exact(a,b)
    elif actual!=expected: raise ValueError("Protocol value differs")


def bound_source(root,code,revision,executing):
    root,code,executing=map(canonical,(root,code,executing))
    if root!=ROOT or not re.fullmatch(r"[0-9a-f]{40}",revision) or code!=root/"jobs"/revision/JOB/"code" or executing!=code/HELPERS[0] or Path(shared.__file__).resolve()!=code/HELPERS[2]:
        raise ValueError("Exact immutable dispatch source required")
    files={name:identity(code/name) for name in HELPERS}
    markers={name:identity(code.parent/name,False) for name in ("revision","source-sha256")}
    if (code.parent/"revision").read_bytes()!=(revision+"\n").encode() or re.fullmatch(b"[0-9a-f]{64}\n",(code.parent/"source-sha256").read_bytes()) is None:
        raise ValueError("Original dispatch markers differ")
    protocol=strict_json((code/PROTOCOL).read_bytes()); exact(protocol,EXPECTED_PROTOCOL)
    return {"files":files,"markers":markers}


def require_execution():
    if EXPECTED_PROTOCOL["execution"]["authorized"] is not True:
        raise ValueError("Prepared v2 interface: no third acquisition authorized")


def preflight(root,code,revision):
    require_execution()
    sources=bound_source(root,code,revision,code/HELPERS[0]); canonical(root/BASE)
    if not (root/BASE).parent.is_dir() or (root/BASE).exists(): raise ValueError("Fresh absent cohort required")
    if shutil.disk_usage(root).free<EXPECTED_PROTOCOL["limits"]["min_free_bytes"]: raise ValueError("45GB free space required")
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(urllib.request.Request("http://169.254.169.254/metadata/instance/compute?api-version=2021-02-01",headers={"Metadata":"true"}),timeout=5) as response:
        data=response.read(65537)
    if len(data)>65536: raise ValueError("Azure metadata byte bound exceeded")
    value=strict_json(data)
    if value.get("name")!="world-reward-ncc-h100-02" or str(value.get("resourceGroupName","")).lower()!="world-reward-research":
        raise ValueError("Owned Azure VM02 required")
    return sources


def validate_https(url, redirect=False):
    p=urllib.parse.urlsplit(url); host=p.hostname or ""
    allowed=host in ("huggingface.co","raw.githubusercontent.com") or (redirect and (host.endswith(".hf.co") or host.endswith(".huggingface.co")))
    if p.scheme!="https" or p.username or p.password or p.port not in (None,443) or p.fragment or not allowed or (p.query and not redirect) or any(k.lower() in ("token","access_token","authorization","api_key") for k,_ in urllib.parse.parse_qsl(p.query)):
        raise ValueError("Public allowlisted HTTPS required")


class SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,req,fp,code,msg,headers,newurl):
        validate_https(newurl,True)
        return super().redirect_request(req,fp,code,msg,headers,newurl)


def transfer(spec,path,owned):
    validate_https(spec["url"]); h=hashlib.sha256(); count=0
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),SafeRedirect())
    with path.open("xb") as stream:
        owned.append((path,path.stat().st_dev,path.stat().st_ino,None))
        with opener.open(urllib.request.Request(spec["url"],headers={"Accept-Encoding":"identity"}),timeout=30) as response:
            validate_https(response.geturl(),True)
            while block:=response.read(min(CHUNK,spec["bytes"]-count+1)):
                count+=len(block)
                if count>spec["bytes"]: raise ValueError("Source byte bound exceeded")
                h.update(block);stream.write(block)
    if count!=spec["bytes"] or h.hexdigest()!=spec["sha256"]: raise ValueError("Full source SHA/bytes differ")
    path.chmod(0o400)
    owned[-1]=(path,path.stat().st_dev,path.stat().st_ino,spec["sha256"])


def save(path,data,owned,mode=0o400):
    path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    with path.open("xb") as stream:
        s=path.stat();owned.append((path,s.st_dev,s.st_ino,None));stream.write(data)
    path.chmod(mode);owned[-1]=(path,s.st_dev,s.st_ino,hashlib.sha256(data).hexdigest())


def cleanup(owned):
    for path,dev,ino,sha in reversed(owned):
        s=path.lstat()
        if (path.resolve()!=path or any(p.is_symlink() for p in (path,*path.parents)) or not stat.S_ISREG(s.st_mode) or s.st_nlink!=1 or s.st_uid!=os.getuid() or s.st_mode&0o777 not in (0o400,0o444,0o600) or (s.st_dev,s.st_ino)!=(dev,ino) or (sha and shared.digest(path)!=sha)):
            raise ValueError("Refuse cleanup of changed or unknown owned file")
        path.unlink()


def zip_inventory(archive,budget):
    members={}; names=set(); limits=EXPECTED_PROTOCOL["limits"]
    for info in archive.infolist():
        name=info.filename; p=PurePosixPath(name); kind=stat.S_IFMT(info.external_attr>>16)
        if (not name or "\\" in name or "\x00" in name or p.is_absolute() or str(p)!=name.rstrip("/") or any(x in ("",".","..") for x in name.rstrip("/").split("/")) or name.rstrip("/") in names or kind not in ((0,stat.S_IFDIR) if info.is_dir() else (0,stat.S_IFREG)) or info.flag_bits&1 or info.compress_type not in (zipfile.ZIP_STORED,zipfile.ZIP_DEFLATED)):
            raise ValueError("Unsafe or duplicate ZIP member")
        names.add(name.rstrip("/"));budget[0]+=info.file_size;budget[1]+=1
        if (info.is_dir() and info.file_size) or info.file_size<0 or info.file_size>limits["member_bytes"] or budget[0]>limits["expanded_bytes"] or budget[1]>limits["members"]:
            raise ValueError("ZIP expansion bound exceeded")
        if not info.is_dir(): members[name]=info
    if not members or any(str(p) in members for n in members for p in PurePosixPath(n).parents if str(p)!="."):
        raise ValueError("Empty ZIP or file ancestor collision")
    return members


def inspect_layout(name,members):
    if name=="ycbv_base.zip":
        if set(members)!={"ycbv/camera_cmu.json","ycbv/camera_uw.json","ycbv/dataset_info.md","ycbv/test_targets_bop19.json"}: raise ValueError("Base layout differs")
    elif name=="ycbv_test_all.zip":
        pattern=r"test/0000(?:4[89]|5[0-9])/(?:scene_(?:camera|gt|gt_info)\.json|(?:rgb|depth)/[0-9]{6}\.png|(?:mask|mask_visib)/[0-9]{6}_[0-9]{6}\.png)"
        if any(not re.fullmatch(pattern,n) for n in members): raise ValueError("Full test root/layout differs")
        scenes=sorted({int(PurePosixPath(n).parts[1]) for n in members})
        if scenes!=list(range(48,60)): raise ValueError("Full native test scene set differs")
        if any(f"test/{s:06d}/scene_{k}.json" not in members for s in scenes for k in ("camera","gt","gt_info")): raise ValueError("Full original scene metadata missing")
    else: raise ValueError("Forbidden archive")


def select_rgb_names(members):
    scenes=sorted({int(PurePosixPath(n).parts[1]) for n in members if re.fullmatch(r"test/[0-9]{6}/rgb/[0-9]{6}\.png",n)})
    if scenes[:3]!=[48,49,50]: raise ValueError("First three sorted RGB scenes differ")
    selected=[]
    for scene in scenes[:3]:
        names=sorted(n for n in members if re.fullmatch(fr"test/{scene:06d}/rgb/[0-9]{{6}}\.png",n))[:96]
        mapping=NativeFrameMap(f"ycbv_scene_{scene:06d}", tuple(int(PurePosixPath(n).stem) for n in names))
        if mapping.source_frame_ids!=tuple(range(1,97)): raise ValueError("First96 native source IDs 1..96 missing")
        selected.extend((scene,source,n) for source,n in zip(mapping.source_frame_ids,names))
    return selected


def crc_all(archive):
    for info in archive.infolist():
        if not info.is_dir():
            with archive.open(info) as stream:
                while stream.read(CHUNK): pass


def png_header(data,bits,color):
    shared.png_dimensions(data) if color==2 else None
    if len(data)<33 or data[:16]!=b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR": raise ValueError("Original PNG header required")
    w,h,b,c,compression,filtering,interlace=struct.unpack(">IIBBBBB",data[16:29])
    if (w,h,b,c)!=(640,480,bits,color) or compression or filtering or interlace not in (0,1) or zlib.crc32(data[12:29])!=struct.unpack(">I",data[29:33])[0]: raise ValueError("Original PNG grid/type differs")


def retain_subset(archive,members,private,inputs,selected,owned):
    if selected!=select_rgb_names(members): raise ValueError("Selection must precede all private annotations")
    images=[]
    for scene in (48,49,50):
        prefix=f"test/{scene:06d}/";values={}
        for kind in ("camera","gt","gt_info"):
            full=strict_json(archive.read(prefix+f"scene_{kind}.json"))
            if type(full) is not dict or any(str(f) not in full for f in range(1,97)): raise ValueError("Private coverage missing")
            values[kind]={str(f):full[str(f)] for f in range(1,97)}
        for frame in range(1,97):
            gt,info=values["gt"][str(frame)],values["gt_info"][str(frame)]
            if type(gt) is not list or not gt or type(info) is not list or len(gt)!=len(info): raise ValueError("All object instances required")
            names=[prefix+f"depth/{frame:06d}.png"]
            for folder in ("mask","mask_visib"):
                expected={prefix+f"{folder}/{frame:06d}_{i:06d}.png" for i in range(len(gt))}
                actual={n for n in members if n.startswith(prefix+f"{folder}/{frame:06d}_")}
                if actual!=expected: raise ValueError("All original instance masks required")
                names.extend(sorted(expected))
            for name in names:
                data=archive.read(name);png_header(data,16 if "/depth/" in name else 8,0);save(private/"source"/name,data,owned)
        for kind,value in values.items(): save(private/"source"/prefix/f"scene_{kind}.json",(json.dumps(value,sort_keys=True,allow_nan=False)+"\n").encode(),owned)
    for scene,frame,name in selected:
        data=archive.read(name);png_header(data,8,2);filename=f"scene_{scene:06d}_frame_{frame:06d}.png";save(inputs/filename,data,owned,0o444)
        images.append({"scene_id":scene,"frame_position":NativeFrameMap(f"ycbv_scene_{scene:06d}",tuple(range(1,97))).position(frame),"source_frame_id":frame,"file":filename,"sha256":hashlib.sha256(data).hexdigest(),"width":640,"height":480})
    return {"schema":EXPECTED_PROTOCOL["output"]["public_schema"],"revision":REVISION,"license":"MIT","selection":SELECTION,
            "attribution":"YCB-Video: Yu Xiang et al.; BOP conversion: Hodan et al.",
            "frame_maps":[NativeFrameMap(f"ycbv_scene_{scene:06d}",tuple(range(1,97))).to_dict() for scene in (48,49,50)],"images":images}


def prune_empty(folder):
    if not folder.exists(): return
    canonical(folder)
    for path in sorted(folder.rglob("*"),key=lambda p:len(p.parts),reverse=True):
        canonical(path);s=path.lstat()
        if not stat.S_ISDIR(s.st_mode) or s.st_uid!=os.getuid() or any(path.iterdir()): raise ValueError("Unknown failure artifact retained, not deleted")
        path.rmdir()
    folder.rmdir()


def acquisition_alarm(report):
    def expired(*_): raise TimeoutError("Whole 3600s acquisition budget exhausted")
    signal.signal(signal.SIGALRM,expired);signal.alarm(ACQUISITION_SECONDS)


def cleanup_alarm(report):
    """One absolute cleanup deadline; repeated cleanup calls never renew it."""
    if report.get("cleanup_grace_started") is True: return
    report["cleanup_grace_started"] = True
    def expired(*_):
        report["cleanup_grace_exhausted"] = True
        raise TimeoutError("Bounded 180s acquisition cleanup grace exhausted")
    signal.signal(signal.SIGALRM,expired);signal.alarm(CLEANUP_SECONDS)


def acquire(out,report,persist):
    private,inputs=out/"eval_private",out/"inputs";downloads=private/".downloads";downloads.mkdir(mode=0o700)
    owned=[]; disposable=[];archives={};budget=[0,0];complete=False
    try:
        report.update(phase="primary_evidence");persist()
        for name,spec in EXPECTED_PROTOCOL["evidence"].items():
            path=private/"source/licenses"/(name+".txt");path.parent.mkdir(parents=True,exist_ok=True,mode=0o700);transfer(spec,path,owned)
        text=(private/"source/licenses/publisher_readme.txt").read_bytes()
        license_text=(private/"source/licenses/publisher_license.txt").read_bytes()
        if b"license: mit" not in (private/"source/licenses/hf_readme.txt").read_bytes().lower() or b"YCB-Video dataset is released under the MIT License" not in text or b"Permission is hereby granted, free of charge" not in license_text: raise ValueError("Publisher MIT evidence differs")
        if sum(x["bytes"] for x in EXPECTED_PROTOCOL["archives"].values())>EXPECTED_PROTOCOL["limits"]["download_bytes"]: raise ValueError("Download bound exceeded")
        for name,spec in EXPECTED_PROTOCOL["archives"].items():
            report.update(phase="download",active_archive=name);persist();transfer(spec,downloads/name,disposable)
        report.update(phase="inventory");persist()
        for name in EXPECTED_PROTOCOL["archives"]:
            archive=zipfile.ZipFile(downloads/name);archives[name]=archive;members=zip_inventory(archive,budget);inspect_layout(name,members)
            archives[name]=(archive,members)
        selected=select_rgb_names(archives["ycbv_test_all.zip"][1]);report.update(selection_before_private_annotation_values=True,selected_frames=288,phase="crc");persist()
        for archive,members in archives.values(): crc_all(archive)
        spec=EXPECTED_PROTOCOL["embedded_dataset_info"];data=archives["ycbv_base.zip"][0].read(spec["file"])
        if len(data)!=spec["bytes"] or hashlib.sha256(data).hexdigest()!=spec["sha256"] or b"Permission is hereby granted, free of charge" not in data or re.search(rb"(?i)non.?commercial|by-nc",data): raise ValueError("Embedded dataset MIT evidence differs")
        save(private/"source/licenses/dataset_info.md",data,owned);report.update(phase="retain");persist()
        manifest=retain_subset(*archives["ycbv_test_all.zip"],private,inputs,selected,owned)
        retention={"files":[{"file":str(p.relative_to(private)),**identity(p)} for p,_,_,_ in owned if p.is_relative_to(private)],"all_instances_retained":True}
        save(private/"retention-receipt.json",(json.dumps(retention,sort_keys=True)+"\n").encode(),owned)
        save(inputs/"manifest.json",(json.dumps(manifest,indent=2)+"\n").encode(),owned,0o444)
        report.update(frame_maps=manifest["frame_maps"],public_manifest=identity(inputs/"manifest.json"),retention_receipt=identity(private/"retention-receipt.json"),archives=EXPECTED_PROTOCOL["archives"],expanded_bytes_inspected=budget[0],archive_members=budget[1],all_instances_retained=True,phase="retained")
        for path,_,_,sha in owned:
            if identity(path)["sha256"]!=sha: raise ValueError("Retained source changed")
        complete=True
        return owned
    finally:
        cleanup_alarm(report)
        for value in archives.values(): (value[0] if isinstance(value,tuple) else value).close()
        try:
            cleanup(disposable);downloads.rmdir();report["disposable_archives_removed"]=True
        except Exception:
            if report.get("cleanup_grace_exhausted") is not True:
                cleanup(owned);prune_empty(inputs)
            raise
        if not complete:
            cleanup(owned);prune_empty(private);prune_empty(inputs)
        report["cleanup_completed"] = True


def main(argv=None):
    require_execution()
    parser=argparse.ArgumentParser(allow_abbrev=False);parser.add_argument("--preflight",action="store_true");args=parser.parse_args(argv)
    root=Path(os.environ["WR_ROOT"]);code=Path(os.environ["WR_CODE"]);revision=os.environ["WR_CODE_REVISION"]
    if platform.system()!="Linux": raise RuntimeError("Heavy acquisition is Azure-only")
    if args.preflight:
        print(hashlib.sha256(json.dumps(preflight(root,code,revision),sort_keys=True).encode()).hexdigest());return
    if os.getuid()!=1000 or os.environ.get("WR_IMAGE_ID")!=IMAGE or os.environ.get("WR_AZURE_VM02_VERIFIED")!="1" or os.environ.get("WR_YCBV_V2_AUTHORIZED")!="1": raise ValueError("Exact CPU image/UID1000/VM02 required")
    out=canonical(root/BASE)
    if not out.is_dir() or set(p.name for p in out.iterdir())!={".container.cid"}: raise ValueError("Exclusively reserved new output required")
    before=bound_source(root,code,revision,Path(__file__));oldmask=os.umask(0o077)
    (out/"eval_private").mkdir(mode=0o700);(out/"inputs").mkdir(mode=0o755)
    report={"stage":STAGE,"status":"fail","phase":"start","producer_revision":revision,"script_sha256":before["files"][HELPERS[0]]["sha256"],"source_helpers":before,"dataset_revision":REVISION,"license":"MIT","image_id":IMAGE,"budget_seconds":ACQUISITION_SECONDS,"cleanup_grace_seconds":CLEANUP_SECONDS,"engineering_replay":True,"native_frame_ids_preserved":True,"device":"cpu","gpu_used":False,"inference_performed":False,"challenge_inputs_used":False,"challenge_overlap_verified":False,"accuracy_verified":False,"models_downloaded":False,"train_downloaded":False,"sparse_test_downloaded":False,"private_annotations_exported_as_inference_inputs":False}
    started=time.perf_counter();path=out/"report.json";retained=[]
    with path.open("x") as stream:
        def persist():
            stream.seek(0);json.dump(report,stream,indent=2,allow_nan=False);stream.write("\n");stream.truncate();stream.flush();os.fsync(stream.fileno())
        def timeout(signum,frame): raise TimeoutError("Whole acquisition budget exhausted")
        previous={signum:signal.signal(signum,timeout) for signum in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT)};acquisition_alarm(report)
        try:
            retained=acquire(out,report,persist)
            if bound_source(root,code,revision,Path(__file__))!=before: raise ValueError("Source/protocol changed")
            if report.get("cleanup_grace_exhausted") is True: raise TimeoutError("Cleanup exceeded declared grace")
            report.update(status="pass",phase="complete",source_rehashed_after=True)
        except Exception as exc:
            report.update(status="fail",error_type=type(exc).__name__,error="Pinned acquisition contract failed; URLs/tokens and private values omitted")
            cleanup_alarm(report)
            if retained and report.get("cleanup_grace_exhausted") is not True:
                try: cleanup(retained);prune_empty(out/"eval_private");prune_empty(out/"inputs");retained=[]
                except Exception as cleanup_error: report["cleanup_error_type"]=type(cleanup_error).__name__
        finally:
            cleanup_alarm(report)
            report["elapsed_seconds"]=time.perf_counter()-started
            try: report["source_rehashed_after"]=bound_source(root,code,revision,Path(__file__))==before
            except Exception: report["source_rehashed_after"]=False;report["status"]="fail"
            if not report["source_rehashed_after"]:
                report["status"]="fail"
                if retained and report.get("cleanup_grace_exhausted") is not True:
                    try: cleanup(retained);prune_empty(out/"eval_private");prune_empty(out/"inputs")
                    except Exception as cleanup_error: report["cleanup_error_type"]=type(cleanup_error).__name__
            try:
                # Stop the cleanup timer only for the small, mandatory receipt seal.
                signal.alarm(0)
                if report.get("cleanup_grace_exhausted") is True: report["status"]="fail"
                report["elapsed_seconds"]=time.perf_counter()-started
                persist();path.chmod(0o400);os.umask(oldmask)
            finally:
                signal.alarm(0)
                for signum,handler in previous.items(): signal.signal(signum,handler)
    if report["status"]!="pass": raise SystemExit(1)


if __name__=="__main__": main()
