"""Frozen public OWLv2 assets only; no decode, install, inference or dataset.

Unchanged MASA streamer and atomic NOREPLACE publication are reused. Signed
CDN URLs stay in memory, never receipts/errors. Apache model-card declaration
is retained evidence, not independent legal/training-overlap clearance.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import stat
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

sys.path.insert(0,str(Path(__file__).resolve().parent))
import mediapipe_cpu_runtime_verify as rt
import mediapipe_hands_acquire as mp
import masa_acquire as acq

ROOT=Path('/srv/scenesmith/world-reward')
BASE='weights/owlv2_objectness_v2'
REPORT='results/owlv2-objectness-assets-v2.json'
ENTRY='run_owlv2_acquire'
CONFIG='configs/owlv2_assets_v2.json'
REVISION='57beb61adb5abda3de4a9796bc35ae60bc4b9802'
REPO='google/owlv2-base-patch16-ensemble'
BUDGET=600
HELPERS=('infra/owlv2_acquire.py','infra/run_owlv2_acquire.sh',CONFIG,
         'infra/mediapipe_cpu_runtime_verify.py','infra/mediapipe_hands_acquire.py','infra/masa_acquire.py')
ASSETS=(('README.md',4838,'7c7426bc5ec939a42d1f96fb093031b6263400cceac4129ebb941a0c8c11b9b9','primary_model_card'),
        ('config.json',414,'ba9df8c25a4b8461887dd0a93d9252c9cd84697fe8d49a9d8794ce409af9acb2','primary_configuration'),
        ('preprocessor_config.json',425,'cf3e396635b797ee1a464e1b2836e98748f8edac19e89aaa2c93b55ac15b0064','primary_configuration'),
        ('model.safetensors',619918824,'e1e130b9e404cf91a75ad45644c1da9d7fa5284085eecc864266a6923efb99e7','opaque_checkpoint'))
HOSTS=frozenset(('huggingface.co','cdn-lfs.huggingface.co','cdn-lfs.hf.co',
                 'cdn-lfs-us-1.hf.co','cdn-lfs-eu-1.hf.co','cas-bridge.xethub.hf.co',
                 'us.aws.cdn.hf.co'))


def encode(value):return (json.dumps(value,sort_keys=True,allow_nan=False)+'\n').encode()


def pin(raw):return dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


def endpoint(url):
    try:
        p=urllib.parse.urlsplit(url)
        rt.require(p.scheme=='https' and p.hostname in HOSTS and p.port in (None,443)
            and not p.username and not p.password and not p.fragment and '\\' not in url
            and not any(ord(c)<33 or ord(c)==127 for c in url), 'Closed public HTTPS endpoint required')
        rt.require(not any(k.lower() in ('token','access_token','authorization','auth','api_key','cookie')
            for k,_ in urllib.parse.parse_qsl(p.query)), 'User credentials in URL forbidden')
    except (ValueError,TypeError):raise ValueError('Closed credential-free HTTPS endpoint required') from None
    return url


def credentials(request):
    return any(k.lower() in ('authorization','cookie','proxy-authorization')
               for k in (*request.headers,*request.unredirected_hdrs))


class PublicRedirect(urllib.request.HTTPRedirectHandler):
    def __init__(self):super().__init__();self.count=0
    def redirect_request(self,request,fp,code,message,headers,url):
        endpoint(request.full_url);endpoint(url);self.count+=1
        rt.require(self.count<=3 and code in (301,302,303,307,308) and not credentials(request), 'Bounded credential-free publisher redirect required')
        return super().redirect_request(request,fp,code,message,headers,url)


class PublicOpener:
    """Per-asset redirect counter, proxy-off and standard verified TLS."""
    def __init__(self,rows):self.urls=frozenset(r['url'] for r in rows)
    def open(self,request,timeout):
        endpoint(request.full_url)
        rt.require(request.full_url in self.urls and not urllib.parse.urlsplit(request.full_url).query
                   and not credentials(request), 'Frozen unauthenticated initial URL required')
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),PublicRedirect())
        response=opener.open(request,timeout=timeout)
        try:
            endpoint(response.geturl())
            rt.require(response.status==200,'Exact publisher response status required')
            return response
        except BaseException:
            response.close();raise


def configuration(code,source):
    cfg=rt.pinned(code/CONFIG,source['helpers'][CONFIG],16<<10)
    fixed=dict(schema='world_reward.owlv2_assets.v1',publisher_repository=REPO,publisher_revision=REVISION,
        output=BASE,report=REPORT,budget_seconds=600,retry_count=0,parallel_downloads=1,maximum_total_bytes=620000000)
    rt.require(all(type(cfg.get(k)) is type(v) and cfg[k]==v for k,v in fixed.items()), 'Frozen OWLv2 acquisition scope differs')
    expected=[]
    for name,size,sha,kind in ASSETS:
        url=f"https://huggingface.co/{REPO}/{'resolve' if kind=='opaque_checkpoint' else 'raw'}/{REVISION}/{name}"
        expected.append(dict(file=name,bytes=size,sha256=sha,kind=kind,url=url))
    rt.require(cfg['assets']==expected and sum(r['bytes'] for r in expected)<=cfg['maximum_total_bytes'], 'Exact four original assets required')
    rt.require(set(cfg['reused_helper_pins'])==set(HELPERS[3:]), 'All original imported helpers required')
    for module,name in ((rt,HELPERS[3]),(mp,HELPERS[4]),(acq,HELPERS[5])):
        rt.require(Path(module.__file__).resolve()==code/name and source['helpers'][name]==cfg['reused_helper_pins'][name], 'Original helper bytes/origin differ')
    return cfg


def cleanup(partials):
    for path,wanted in partials:
        if not path.exists():continue
        rt.canonical(path);s=path.lstat()
        rt.require(stat.S_ISREG(s.st_mode) and s.st_nlink==1 and (s.st_dev,s.st_ino,s.st_uid)==wanted, 'Cannot remove foreign partial')
        path.unlink()
    rt.require(all(not path.exists() for path,_ in partials), 'Owned partial survives')


def acquire_assets(base,cfg,report,deadline,*,opener=None,fetch=acq.fetch,publisher=mp):
    partials=[];records=report['assets'];opener=PublicOpener(cfg['assets']) if opener is None else opener
    try:
        for row in cfg['assets']:
            report['phase']=row['kind']
            record=fetch(rt,publisher,base,row,cfg['assets'],opener,deadline,partials)
            (base/row['file']).chmod(0o400);records.append(record)
            # Verify the tiny notice/configurations before requesting opaque
            # weights, not after spending bandwidth on an ineligible candidate.
            if row['file']=='README.md':
                card=(base/'README.md').read_text(encoding='utf-8')
                rt.require(card.startswith('---\n') and 'license: apache-2.0' in card.split('---',2)[1], 'Exact primary model license declaration required')
                report['primary_model_license_recorded']=True
            elif row['file']=='config.json':
                rt.require(rt.strict((base/'config.json').read_bytes())['architectures']==['Owlv2ForObjectDetection'], 'Original model configuration required')
            elif row['file']=='preprocessor_config.json':
                rt.require(rt.strict((base/'preprocessor_config.json').read_bytes())['image_processor_type']=='Owlv2ImageProcessor', 'Original processor configuration required')
    finally:
        cleanup(partials);report['owned_partials_removed']=True


def posthash(base,cfg,report):
    wanted={r['file']:{k:r[k] for k in ('bytes','sha256')} for r in cfg['assets']}
    rt.require({p.name for p in base.iterdir()}==set(wanted) and len(report['assets'])==4
        and {r['file']:{k:r[k] for k in ('bytes','sha256')} for r in report['assets']}==wanted, 'Exclusive complete original asset inventory required')
    for name,expected in wanted.items():rt.require(rt.identity(base/name,620000000)==expected, 'Original final asset changed')
    report['artifacts_rehashed_after']=True;report['final_asset_pins']=wanted


def run():
    started=time.monotonic();deadline=started+BUDGET
    rt.require(sys.platform=='linux' and os.geteuid()==0 and os.uname().nodename=='world-reward-ncc-h100-02', 'Exact Azure root CPU acquisition host required')
    code=Path(os.environ['WR_CODE']);revision=os.environ['WR_CODE_REVISION']
    rt.require(Path(__file__).resolve()==code/HELPERS[0], 'Immutable original driver required')
    before=rt.source(ROOT,code,revision,ENTRY,HELPERS);cfg=configuration(code,before)
    base=rt.canonical(ROOT/BASE);path=rt.canonical(ROOT/REPORT)
    rt.require(not base.exists() and not path.exists() and base.parent.is_dir() and path.parent.is_dir()
        and shutil.disk_usage(base.parent).free>=4<<30, 'Fresh canonical namespace and bounded free storage required')
    old_umask=os.umask(0o077);base.mkdir(mode=0o700)
    report=dict(schema='world_reward.owlv2_assets_receipt.v1',status='fail',producer_revision=revision,source_binding=before,
        configuration_identity=before['helpers'][CONFIG],publisher_revision=REVISION,assets=[],budget_seconds=600,
        phase='start',primary_model_license_recorded=False,license_clearance_verified=False,standalone_model_license_file_verified=False,
        declared_model_license='Apache-2.0',training_overlap_verified=False,challenge_overlap_verified=False,
        training_source_status='model_card_COCO_OpenImages_web_mentions_v2_update_pending_exact_checkpoint_overlap_unknown',
        checkpoint_decoded=False,models_loaded=False,gpu_used=False,packages_installed=False,inference_performed=False,
        dataset_read=False,challenge_inputs_used=False,credentials_used=False,quality_verified=False,adopted=False,
        source_rehashed_after=False,artifacts_rehashed_after=False,owned_partials_removed=False,outputs_sealed=False,
        retry_count=0,local_heavy_transfer=False,network='public_credential_free_verified_TLS_closed_publisher_endpoints')
    def interrupted(*_):raise TimeoutError('Acquisition interrupted')
    handlers={s:signal.signal(s,interrupted) for s in (signal.SIGTERM,signal.SIGINT,signal.SIGALRM)}
    signal.alarm(max(1,math.ceil(deadline-time.monotonic())))
    try:
        acquire_assets(base,cfg,report,deadline)
        rt.require(rt.source(ROOT,code,revision,ENTRY,HELPERS)==before and configuration(code,before)==cfg, 'Full source/config changed')
        report['source_rehashed_after']=True;posthash(base,cfg,report)
        rt.require(time.monotonic()<deadline, 'Inclusive600s acquisition deadline');report['status']='pass'
    except BaseException as exc:
        report['error_type']=type(exc).__name__ if type(exc) in (ValueError,OSError,TimeoutError,KeyError,urllib.error.HTTPError,urllib.error.URLError) else 'other'
    finally:
        signal.alarm(0)
        with path.open('xb') as stream:
            os.fchmod(stream.fileno(),0o400)
            for leaf in base.iterdir():
                rt.canonical(leaf);rt.require(leaf.is_file() and leaf.stat().st_nlink==1, 'Cannot seal foreign output');leaf.chmod(0o400)
            base.chmod(0o500);report['outputs_sealed']=True
            for folder in (base,path.parent):
                fd=os.open(folder,os.O_RDONLY|os.O_DIRECTORY)
                try:os.fsync(fd)
                finally:os.close(fd)
            report['elapsed_seconds']=time.monotonic()-started
            if report['elapsed_seconds']>=BUDGET:report.update(status='fail',error_type='TimeoutError')
            raw=encode(report);stream.write(raw);stream.flush();os.fsync(stream.fileno())
            if time.monotonic()>=deadline and report['status']=='pass':
                report.update(status='fail',error_type='TimeoutError',elapsed_seconds=time.monotonic()-started)
                raw=encode(report);stream.seek(0);stream.write(raw);stream.truncate();stream.flush();os.fsync(stream.fileno())
        os.umask(old_umask)
        for s,h in handlers.items():signal.signal(s,h)
        print(json.dumps(dict(status=report['status'],report_identity=pin(raw),assets=len(report['assets']),elapsed_seconds=report['elapsed_seconds']),sort_keys=True))
    return report


if __name__=='__main__':
    rt.require(len(sys.argv)==1,'No runtime/query/retry overrides allowed')
    result=run();sys.exit(0 if result['status']=='pass' else 1)
