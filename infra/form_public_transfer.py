"""Direct private Azure transfer of FOUR qualified external RGB/text packages.

No reference, checkpoint, original archive or credential is transported. Files
retain their independently sealed identities and absolute input routes. Media
never traverses the local host. This is acquisition, never inference or scoring.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import urllib.request

import form_hoi_external_dev as dev
from full4d_publish import PrivatePreviews, ENDPOINT
from mediapipe_cpu_runtime_verify import canonical, identity, source, strict, require

ROOT=Path('/srv/scenesmith/world-reward')
ENTRY='run_form_public_transfer'
HELPERS=('infra/form_public_transfer.py','infra/run_form_public_transfer.sh',
         'infra/form_hoi_external_dev.py','infra/full4d_publish.py')
MAX_VIDEO=128<<20


def revision(value):
    require(type(value) is str and re.fullmatch('[0-9a-f]{40}',value),'Exact frozen producer required')
    return value


def cohorts(code):
    c=strict((code/'configs/form_hoi_insight_v1.json').read_bytes())
    seq=[r['sequence_id'] for r in c['cohort'] if r['split']=='development']
    require(len(seq)==4 and all(re.fullmatch('[A-Za-z0-9_-]{1,128}',s)for s in seq),'Four frozen DEV required')
    return seq


def allowed_blob(name):
    return bool(type(name) is str and re.fullmatch(
        r'form-public-[0-9a-f]{40}/(?:manifest.json|[A-Za-z0-9_-]{1,128}/(?:rgb.mp4|input.json))',name))


class PrivatePublicPackages(PrivatePreviews):
    def request(self,method,name=None,data=None,headers=None,*,container_acl=False):
        if container_acl:return super().request(method,container_acl=True)
        require(method in {'PUT','HEAD','GET'} and allowed_blob(name),'Only exact public RGB/text routes')
        try:
            req=urllib.request.Request(ENDPOINT+'/'+name,data=data,method=method,
                headers={'x-ms-version':'2023-11-03',**self.authorization(),**(headers or {})})
            return self.opener.open(req,timeout=120)
        except Exception:raise RuntimeError('Private Azure public-package transport failed') from None


def contract(manifest,producer,seqs):
    require(manifest.get('schema')=='world_reward.form_public_transfer.v1' and
        manifest.get('dev_revision')==producer and manifest.get('sequences')==seqs and
        manifest.get('private_references_included') is False and manifest.get('models_included') is False and
        manifest.get('private_container_verified') is True and len(manifest.get('files',[]))==8,
        'Exact four-DEV public-only transfer receipt required')
    expected={(s,n)for s in seqs for n in ('input.json','rgb.mp4')};seen=set()
    for row in manifest['files']:
        key=(row.get('sequence_id'),row.get('file'))
        require(key in expected and key not in seen and row.get('name')==f'form-public-{producer}/{key[0]}/{key[1]}'
            and type(row.get('bytes')) is int and 0<row['bytes']<=(MAX_VIDEO if key[1]=='rgb.mp4' else 16384)
            and re.fullmatch('[0-9a-f]{64}',str(row.get('sha256')))
            and re.fullmatch(r'"[A-Za-z0-9-]{1,100}"',str(row.get('etag'))),'Exact bounded SHA/ETag public file required')
        seen.add(key)
    require(seen==expected,'Missing frozen public package')


def publish(code,producer,rev,binding):
    base=dev.DATA/producer;report_pin=identity(base/'report.json',1<<20)
    report=strict((base/'report.json').read_bytes());seqs=cohorts(code)
    require(report['status']=='pass' and report['inference_ready'] is True and report['producer_revision']==producer
        and report['reserved_acquired']==0 and [r['sequence_id']for r in report['sequences']]==seqs,
        'Complete qualified four-DEV inputs required before transport')
    client=PrivatePublicPackages();client.require_private();files=[]
    cfg=strict((code/'configs/form_hoi_external_dev_v1.json').read_bytes())
    for row in report['sequences']:
        s=row['sequence_id'];d=base/s
        require(identity(d/'receipt.json',1<<20)==row['receipt'],'Independent sequence receipt required')
        receipt=strict((d/'receipt.json').read_bytes())
        inp=d/'inputs/input.json';p=strict(inp.read_bytes())
        dev.validate_public_package(p,cfg,require_ready=True)
        require(identity(inp,16384)==receipt['input'],'Original public input pin differs')
        require({f.name for f in inp.parent.iterdir()}=={'rgb.mp4','input.json'},'Only RGB and text package')
        for n,mime,cap,pin in [('input.json','application/json',16384,receipt['input']),
                              ('rgb.mp4','video/mp4',MAX_VIDEO,p['video_pin'])]:
            f=inp.parent/n;require(identity(f,cap)==pin,'Original public payload changed')
            b=client.upload(f'form-public-{producer}/{s}/{n}',f.read_bytes(),mime,producer)
            client.head(b['name'],pin,b['etag']);files.append(dict(b,sequence_id=s,file=n))
    require(identity(base/'report.json',1<<20)==report_pin,'DEV source report changed')
    manifest=dict(schema='world_reward.form_public_transfer.v1',dev_revision=producer,sequences=seqs,
        files=files,dev_report=report_pin,private_references_included=False,models_included=False,
        private_container_verified=True,source_binding=binding,producer_revision=rev)
    contract(manifest,producer,seqs);raw=(json.dumps(manifest,sort_keys=True)+'\n').encode()
    row=client.upload(f'form-public-{producer}/manifest.json',raw,'application/json',producer)
    client.head(row['name'],row,row['etag'])
    dest=ROOT/'results'/f'form-public-transfer-{rev}.json'
    with dest.open('xb')as f:f.write(raw);os.fchmod(f.fileno(),0o444)
    print(json.dumps(dict(status='published',manifest={k:row[k]for k in ('bytes','sha256')},dev_revision=producer)),flush=True)


def fetch(code,producer,manifest_pin,rev,binding):
    client=PrivatePublicPackages();client.require_private()
    def read(name,pin,etag=None):
        with client.request('GET',name,headers={'If-Match':etag}if etag else {})as r:
            require(r.status==200 and int(r.headers.get('Content-Length','-1'))==pin['bytes']
                and r.headers.get('x-ms-meta-sha256')==pin['sha256'],'Pinned private package response required')
            if etag:require(r.headers.get('ETag')==etag,'Original public blob replaced')
            raw=r.read(pin['bytes']+1)
        require(len(raw)==pin['bytes'] and hashlib.sha256(raw).hexdigest()==pin['sha256'],'Transport SHA differs')
        return raw
    require(0<manifest_pin['bytes']<=64<<10 and re.fullmatch('[0-9a-f]{64}',manifest_pin['sha256']),
            'Independent bounded transfer manifest pin required')
    raw=read(f'form-public-{producer}/manifest.json',manifest_pin);m=strict(raw);seqs=cohorts(code)
    contract(m,producer,seqs);target=dev.DATA/producer
    require(not target.exists(),'Never overwrite transferred inputs');target.mkdir(mode=0o755,parents=True)
    for row in m['files']:
        path=target/row['sequence_id']/'inputs'/row['file'];path.parent.mkdir(mode=0o755,parents=True,exist_ok=True)
        data=read(row['name'],row,row['etag'])
        with path.open('xb')as f:f.write(data);f.flush();os.fsync(f.fileno());os.fchmod(f.fileno(),0o444)
        require(identity(path,MAX_VIDEO)=={k:row[k]for k in ('bytes','sha256')},'Transferred payload differs')
    cfg=strict((code/'configs/form_hoi_external_dev_v1.json').read_bytes())
    for s in seqs:
        p=strict((target/s/'inputs/input.json').read_bytes());dev.validate_public_package(p,cfg,require_ready=True)
        require(p['video']==str(target/s/'inputs/rgb.mp4') and identity(Path(p['video']),MAX_VIDEO)==p['video_pin'],
            'Transferred public routes/bytes differ')
    receipt=dict(status='pass',dev_revision=producer,manifest_identity=manifest_pin,sequences=seqs,
        source_binding=binding,private_references_transferred=False,heavy_data_local=False)
    dev.seal(target/'public-transfer.json',receipt)
    print(json.dumps(dict(status='fetched',dev_revision=producer,public_sequences=4,private_references_transferred=False)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('publish','fetch'));p.add_argument('dev_revision',type=revision)
    p.add_argument('--manifest-bytes',type=int);p.add_argument('--manifest-sha256');a=p.parse_args()
    code=canonical(os.environ['WR_CODE']);rev=revision(os.environ['WR_CODE_REVISION'])
    binding=source(ROOT,code,rev,ENTRY,HELPERS)
    require(os.uname().nodename==('world-reward-ncc-h100-02'if a.mode=='publish'else'scenesmith-ncc-h100-01'),
            'Direct Azure source/destination only')
    if a.mode=='publish':publish(code,a.dev_revision,rev,binding)
    else:fetch(code,a.dev_revision,dict(bytes=a.manifest_bytes,sha256=a.manifest_sha256),rev,binding)
