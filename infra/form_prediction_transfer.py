"""Sealed four-DEV A/B predictions, directly Azure01 -> private blob -> CPU02.

No reference bytes/values, RGB, checkpoint, credentials, NPZ decode or model
execution. Publish starts only after all eight literal final stages complete.
Payload SHA identities never change; only manifest paths relocate on CPU02.
"""
from __future__ import annotations

import argparse
from email.utils import formatdate
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import sys
import time
import urllib.request

import form_hoi_external_dev as dev
from full4d_publish import PrivatePreviews, ENDPOINT
from mediapipe_cpu_runtime_verify import canonical, identity, pinned, require, source, strict

ROOT=Path('/srv/scenesmith/world-reward')
ENTRY='run_form_prediction_transfer'
HELPERS=('infra/form_prediction_transfer.py','infra/run_form_prediction_transfer.sh',
    'infra/full4d_publish.py','infra/mediapipe_cpu_runtime_verify.py','infra/form_hoi_external_dev.py',
    'infra/form_hoi_external_acquire.py','configs/form_hoi_external_dev_v1.json',
    'configs/form_hoi_insight_v1.json','src/world_reward/form_hoi_protocol.py')
MAX_GEOMETRY=128 << 20
MAX_REPORT=1 << 20
MAX_METADATA=128 << 10
FILES=('eval_geometry.npz','report.json')
STAGES={'A':'fit_A','B':'fit_B'}
BUDGET=900


def revision(value):
    require(type(value) is str and re.fullmatch('[0-9a-f]{40}',value),'Exact immutable producer revision required')
    return value


def cohorts(code):
    c=pinned(code/dev.acquisition.PROTOCOL,dev.acquisition.PROTOCOL_PIN,16 << 10)
    seqs=[r['sequence_id'] for r in c['cohort'] if r['split']=='development']
    require(len(seqs)==4 and len(set(seqs))==4 and all(re.fullmatch('[A-Za-z0-9_-]{1,128}',s) for s in seqs),
        'Exactly four frozen DEV sequences required; no reserved or reroll')
    return seqs,c['dataset_revision']


def prefix(predrev,devrev):return 'form-prediction-'+revision(predrev)+'-'+revision(devrev)


def allowed_blob(name):
    return bool(type(name) is str and re.fullmatch(
        r'form-prediction-[0-9a-f]{40}-[0-9a-f]{40}/(?:package.json|manifest.json|[A-Za-z0-9_-]{1,128}/fit_[AB]/(?:eval_geometry.npz|report.json))',name))


class PrivatePredictionPackages(PrivatePreviews):
    def request(self,method,name=None,data=None,headers=None,*,container_acl=False):
        if container_acl:return super().request(method,container_acl=True)
        require(method in {'PUT','HEAD','GET'} and allowed_blob(name),'Only fixed bounded prediction routes allowed')
        if method=='PUT':
            maximum=MAX_GEOMETRY if name.endswith('/eval_geometry.npz') else MAX_REPORT if name.endswith('/report.json') else MAX_METADATA
            require(type(data) is bytes and 0<len(data)<=maximum,'Bounded literal prediction payload only')
        try:
            req=urllib.request.Request(ENDPOINT+'/'+name,data=data,method=method,
                headers={'x-ms-version':'2023-11-03','x-ms-date':formatdate(usegmt=True),
                    **self.authorization(),**(headers or {})})
            return self.opener.open(req,timeout=120)
        except Exception:raise RuntimeError('Private Azure prediction-package transport failed') from None


def seal_bytes(path,raw):
    path=canonical(path);tmp=path.with_name('.'+path.name+'.part')
    require(not path.exists() and not tmp.exists(),'Exclusive transfer output; never overwrite a baseline/package')
    try:
        with tmp.open('xb') as f:
            f.write(raw);f.flush();os.fsync(f.fileno());os.fchmod(f.fileno(),0o444)
        os.link(tmp,path);tmp.unlink()
    finally:tmp.unlink(missing_ok=True)
    return identity(path,MAX_GEOMETRY)


def seal_json(path,value):
    raw=(json.dumps(value,sort_keys=True,allow_nan=False)+'\n').encode()
    require(len(raw)<=MAX_METADATA,'Bounded nonsensitive transport metadata required')
    return seal_bytes(path,raw)


def pin(value,maximum):
    require(type(value) is dict and set(value)=={'bytes','sha256'} and type(value['bytes']) is int and
        0<value['bytes']<=maximum and re.fullmatch('[0-9a-f]{64}',str(value['sha256'])), 'Bounded independent SHA/bytes required')
    return value


def stage_contract(report,predrev,sid,variant,input_pin,geometry_pin,native_binding=None):
    require(report.get('schema')=='world_reward.form_external_prediction_stage.v1' and
        report.get('status')=='complete' and report.get('stage')==STAGES[variant] and
        report.get('producer_revision')==predrev and report.get('sequence_id')==sid and
        report.get('dataset')=='nvidia/form-hoi' and report.get('input_pin')==input_pin and
        report.get('original_frame_indices')==list(range(96)) and report.get('requested_steps')==300 and
        report.get('full_predictions_sealed_before_evaluation') is True and
        report.get('source_rehashed_after') is True and report.get('oracle_modes')==[] and
        all(report.get(k) is False for k in ('ground_truth_used','private_truth_read','hand_labeled_test',
            'reference_inputs_mounted','training_overlap_verified','production_adopted')) and
        report.get('artifacts',{}).get('eval_geometry.npz')==geometry_pin,
        'Every literal sealed full96 native final stage must be complete and public-only')
    if native_binding is not None:
        b=report.get('source_binding',{})
        require(all(b.get(k)==native_binding[k] for k in ('producer_revision','markers','closure_sha256')),
            'Every final stage must bind original unchanged authenticated native producer source')


def prediction_manifest_contract(m,predrev,devrev,seqs,dataset_revision,base):
    require(m.get('schema')=='world_reward.form_hoi_external_prediction_set.v1' and
        m.get('producer_revision')==predrev and m.get('dataset_revision')==dataset_revision and
        m.get('split')=='development' and m.get('ground_truth_used') is False and
        m.get('private_truth_read') is False and m.get('original_frame_indices')==list(range(96)) and
        len(m.get('sequences',[]))==4 and [r.get('sequence_id') for r in m['sequences']]==seqs,
        'All four paired prediction manifest required, no omission or GT use')
    for row in m['sequences']:
        sid=row['sequence_id'];require(set(row)=={'sequence_id','input','variants'} and
            set(row['variants'])=={'A','B'}, 'Exactly both variants per DEV')
        inp=row['input'];require(set(inp)=={'path','pin'} and inp['path']==str(dev.DATA/devrev/sid/'inputs/input.json'),
            'Public input original absolute route must remain unchanged')
        pin(inp['pin'],16384)
        for name,stage in STAGES.items():
            variant=row['variants'][name];require(set(variant)=={'geometry','report'},'Only literal final geometry and report')
            for role,filename,maximum in [('geometry','eval_geometry.npz',MAX_GEOMETRY),('report','report.json',MAX_REPORT)]:
                r=variant[role];require(set(r)=={'path','pin'} and r['path']==str(base/sid/stage/filename), 'Exact relocated/original payload route required')
                pin(r['pin'],maximum)


def assemble(code,predrev,devrev):
    """Validate/hash ALL8 before any upload; never decode private references/NPZ."""
    seqs,dataset_revision=cohorts(code);base=ROOT/'results'/('form-hoi-external-predict-'+predrev)
    original_code=ROOT/'jobs'/predrev/'run_form_hoi_external_predict/code'
    native=source(ROOT,original_code,predrev,'run_form_hoi_external_predict',
        ('infra/form_hoi_external_predict.py','infra/run_form_hoi_external_predict.sh','configs/form_hoi_external_predict_v1.json'))
    require(original_code.resolve()==original_code,'Original actual native job source required')
    public_receipt=dev.DATA/devrev/'public-transfer.json';public_pin=identity(public_receipt,MAX_METADATA)
    prior=strict(public_receipt.read_bytes())
    require(prior.get('status')=='pass' and prior.get('dev_revision')==devrev and prior.get('sequences')==seqs and
        prior.get('private_references_transferred') is False and prior.get('heavy_data_local') is False and
        prior.get('manifest_identity') is not None,
        'Independent actual Azure public-only four-DEV input transfer receipt required')
    cfg=strict((code/'configs/form_hoi_external_dev_v1.json').read_bytes());rows=[];files=[];tracked=[(public_receipt,public_pin)]
    for sid in seqs:
        inp=dev.DATA/devrev/sid/'inputs/input.json';ip=identity(inp,16384);package=strict(inp.read_bytes())
        dev.validate_public_package(package,cfg,require_ready=True)
        require(package['sequence_id']==sid and package['dataset_revision']==dataset_revision and
            package['video']==str(inp.parent/'rgb.mp4'),'Frozen qualified public RGB/text lineage required')
        row=dict(sequence_id=sid,input=dict(path=str(inp),pin=ip),variants={});tracked.append((inp,ip))
        for variant,stage in STAGES.items():
            directory=canonical(base/sid/stage);report_path=directory/'report.json';gp=identity(directory/'eval_geometry.npz',MAX_GEOMETRY)
            rp=identity(report_path,MAX_REPORT);report=strict(report_path.read_bytes())
            stage_contract(report,predrev,sid,variant,ip,gp,native)
            row['variants'][variant]=dict(geometry=dict(path=str(directory/'eval_geometry.npz'),pin=gp),report=dict(path=str(report_path),pin=rp))
            for filename,expected,mime in [('eval_geometry.npz',gp,'application/octet-stream'),('report.json',rp,'application/json')]:
                path=directory/filename;tracked.append((path,expected))
                files.append(dict(sequence_id=sid,variant=variant,file=filename,path=path,pin=expected,mime=mime))
        rows.append(row)
    manifest=dict(schema='world_reward.form_hoi_external_prediction_set.v1',producer_revision=predrev,
        dataset_revision=dataset_revision,split='development',ground_truth_used=False,private_truth_read=False,
        original_frame_indices=list(range(96)),sequences=rows)
    prediction_manifest_contract(manifest,predrev,devrev,seqs,dataset_revision,base)
    return manifest,files,tracked,dict(native_source_binding=native,public_transfer_receipt=public_pin)


def rehash(tracked):
    for path,expected in tracked:require(identity(path,MAX_GEOMETRY)==expected,'Authoritative sealed prediction/input source changed')


def package_contract(p,predrev,devrev,seqs,dataset_revision):
    require(p.get('schema')=='world_reward.form_prediction_transfer.v1' and p.get('prediction_revision')==predrev and
        p.get('dev_revision')==devrev and p.get('dataset_revision')==dataset_revision and p.get('sequences')==seqs and
        p.get('all_four_paired_predictions_sealed') is True and p.get('ground_truth_read') is False and
        p.get('reference_bytes_included') is False and p.get('RGB_included') is False and p.get('models_included') is False and
        p.get('private_container_verified') is True and len(p.get('files',[]))==16,
        'Literal all4xA/B geometry/report-only private transfer package required')
    expected={(s,v,f) for s in seqs for v in ('A','B') for f in FILES};seen=set()
    for row in p['files']:
        key=(row.get('sequence_id'),row.get('variant'),row.get('file'));maximum=MAX_GEOMETRY if key[2]=='eval_geometry.npz' else MAX_REPORT
        require(key in expected and key not in seen and row.get('name')==prefix(predrev,devrev)+'/'+key[0]+'/'+STAGES[key[1]]+'/'+key[2],
            'Exact complete bounded geometry/report routes only')
        pin({k:row.get(k) for k in ('bytes','sha256')},maximum)
        require(re.fullmatch(r'"[A-Za-z0-9-]{1,100}"',str(row.get('etag'))),'Literal immutable private blob ETag required')
        seen.add(key)
    require(seen==expected,'Missing paired prediction payload')
    m=p.get('prediction_manifest');require(type(m) is dict and m.get('name')==prefix(predrev,devrev)+'/manifest.json' and
        re.fullmatch(r'"[A-Za-z0-9-]{1,100}"',str(m.get('etag'))), 'Exclusive source prediction manifest required')
    pin({k:m.get(k) for k in ('bytes','sha256')},MAX_METADATA)


def publish(code,predrev,devrev,revision_,binding,*,client=None):
    manifest,files,tracked,provenance=assemble(code,predrev,devrev)
    seqs,dataset_revision=cohorts(code);rehash(tracked)
    client=client or PrivatePredictionPackages();client.require_private();uploaded=[]
    for row in files:
        raw=row['path'].read_bytes();expected=row['pin']
        require(len(raw)==expected['bytes'] and hashlib.sha256(raw).hexdigest()==expected['sha256'],'Source payload differs before Azure upload')
        name=prefix(predrev,devrev)+'/'+row['sequence_id']+'/'+STAGES[row['variant']]+'/'+row['file']
        b=client.upload(name,raw,row['mime'],predrev);client.head(name,expected,b['etag'])
        require({k:b[k] for k in ('bytes','sha256')}==expected,'Published payload identity differs')
        uploaded.append(dict(b,sequence_id=row['sequence_id'],variant=row['variant'],file=row['file']))
    rehash(tracked)
    native=source(ROOT,ROOT/'jobs'/predrev/'run_form_hoi_external_predict/code',predrev,'run_form_hoi_external_predict',
        ('infra/form_hoi_external_predict.py','infra/run_form_hoi_external_predict.sh','configs/form_hoi_external_predict_v1.json'))
    require(native==provenance['native_source_binding'],'Original native producer source changed during publication')
    require(source(ROOT,code,revision_,ENTRY,HELPERS)==binding,'Actual transport source changed')
    raw=(json.dumps(manifest,sort_keys=True,allow_nan=False)+'\n').encode()
    require(len(raw)<=MAX_METADATA,'Bounded four-DEV manifest required')
    m=client.upload(prefix(predrev,devrev)+'/manifest.json',raw,'application/json',predrev);client.head(m['name'],m,m['etag'])
    p=dict(schema='world_reward.form_prediction_transfer.v1',prediction_revision=predrev,dev_revision=devrev,
        dataset_revision=dataset_revision,sequences=seqs,files=uploaded,prediction_manifest=m,provenance=provenance,
        all_four_paired_predictions_sealed=True,ground_truth_read=False,reference_bytes_included=False,
        RGB_included=False,models_included=False,private_container_verified=True,source_binding=binding,producer_revision=revision_)
    package_contract(p,predrev,devrev,seqs,dataset_revision);raw=(json.dumps(p,sort_keys=True,allow_nan=False)+'\n').encode()
    require(len(raw)<=MAX_METADATA,'Bounded private package commit marker required')
    b=client.upload(prefix(predrev,devrev)+'/package.json',raw,'application/json',predrev);client.head(b['name'],b,b['etag'])
    dest=ROOT/'results'/('form-prediction-transfer-publish-'+revision_+'.json');seal_json(dest,dict(status='pass',package={k:b[k] for k in ('bytes','sha256')},package_blob=b['name'],prediction_revision=predrev,dev_revision=devrev,source_binding=binding))
    print(json.dumps(dict(status='published',package={k:b[k] for k in ('bytes','sha256')},prediction_revision=predrev,DEV_sequences=4,variants=8)),flush=True)
    return p,b


def read_blob(client,name,expected,*,etag=None,maximum=MAX_GEOMETRY):
    pin({k:expected[k] for k in ('bytes','sha256')},maximum)
    with client.request('GET',name,headers={'If-Match':etag} if etag else {}) as r:
        require(r.status==200 and int(r.headers.get('Content-Length','-1'))==expected['bytes'] and
            r.headers.get('x-ms-meta-sha256')==expected['sha256'] and
            (etag is None or r.headers.get('ETag')==etag),'Private blob length/SHA/ETag differs')
        raw=r.read(expected['bytes']+1)
    require(len(raw)==expected['bytes'] and hashlib.sha256(raw).hexdigest()==expected['sha256'],'Private transfer full payload SHA differs')
    return raw


def relocate_manifest(m,predrev,devrev,seqs,dataset_revision,target):
    # Validate original route first; arbitrary incoming paths never become local files.
    original=ROOT/'results'/('form-hoi-external-predict-'+predrev)
    prediction_manifest_contract(m,predrev,devrev,seqs,dataset_revision,original)
    result=strict(json.dumps(m,sort_keys=True))
    for row in result['sequences']:
        for variant,stage in STAGES.items():
            row['variants'][variant]['geometry']['path']=str(target/row['sequence_id']/stage/'eval_geometry.npz')
            row['variants'][variant]['report']['path']=str(target/row['sequence_id']/stage/'report.json')
    prediction_manifest_contract(result,predrev,devrev,seqs,dataset_revision,target)
    return result


def fetch(code,predrev,devrev,package_pin,revision_,binding,*,client=None):
    client=client or PrivatePredictionPackages();client.require_private();seqs,dataset_revision=cohorts(code)
    raw=read_blob(client,prefix(predrev,devrev)+'/package.json',package_pin,maximum=MAX_METADATA)
    package=strict(raw);package_contract(package,predrev,devrev,seqs,dataset_revision)
    m=package['prediction_manifest'];original_raw=read_blob(client,m['name'],m,etag=m['etag'],maximum=MAX_METADATA)
    original=strict(original_raw);target=ROOT/'results'/('form-hoi-prediction-package-'+predrev+'-'+devrev)
    manifest=relocate_manifest(original,predrev,devrev,seqs,dataset_revision,target)
    # Validate EVERY payload declaration against the original manifest before writes/GET.
    for r in package['files']:
        role='geometry' if r['file']=='eval_geometry.npz' else 'report'
        row=next(x for x in original['sequences'] if x['sequence_id']==r['sequence_id'])
        require(row['variants'][r['variant']][role]['pin']=={k:r[k] for k in ('bytes','sha256')}, 'Independent package/manifest payload identities differ')
    require(not target.exists(),'Do not overwrite completed/incomplete transfer; diagnose then new source')
    target.mkdir(mode=0o755);tracked=[]
    for r in package['files']:
        path=target/r['sequence_id']/STAGES[r['variant']]/r['file'];path.parent.mkdir(mode=0o755,parents=True,exist_ok=True)
        payload=read_blob(client,r['name'],r,etag=r['etag'],maximum=MAX_GEOMETRY if r['file']=='eval_geometry.npz' else MAX_REPORT)
        p=seal_bytes(path,payload);require(p=={k:r[k] for k in ('bytes','sha256')},'Sealed fetched payload differs');tracked.append((path,p))
    # Decode only bounded plain final report JSON; never geometry NPZ or external references.
    for row in manifest['sequences']:
        for variant,data in row['variants'].items():
            report=strict(Path(data['report']['path']).read_bytes())
            stage_contract(report,predrev,row['sequence_id'],variant,row['input']['pin'],data['geometry']['pin'],package['provenance']['native_source_binding'])
    rehash(tracked);require(source(ROOT,code,revision_,ENTRY,HELPERS)==binding,'Actual fetch source changed')
    source_pin=seal_bytes(target/'source-manifest.json',original_raw);seal_bytes(target/'package.json',raw)
    manifest_pin=seal_json(target/'manifest.json',manifest)
    receipt=dict(schema='world_reward.form_prediction_fetch.v1',status='pass',prediction_revision=predrev,dev_revision=devrev,
        package_identity=package_pin,source_manifest_identity=source_pin,local_manifest=dict(path=str(target/'manifest.json'),pin=manifest_pin),
        source_binding=binding,all_four_paired_predictions_sealed=True,original_payloads_rehashed_after=True,
        input_routes_unchanged=True,payload_pins_unchanged=True,native_prediction_producer_unchanged=True,
        private_references_transferred=False,RGB_transferred=False,models_transferred=False,heavy_data_local=False)
    seal_json(target/'transfer.json',receipt)
    print(json.dumps(dict(status='fetched',prediction_revision=predrev,local_manifest=receipt['local_manifest'],variants=8,private_references_transferred=False)),flush=True)
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False);p.add_argument('mode',choices=('publish','fetch'))
    p.add_argument('prediction_revision',type=revision);p.add_argument('dev_revision',type=revision)
    p.add_argument('--package-bytes',type=int);p.add_argument('--package-sha256');a=p.parse_args()
    code=canonical(os.environ['WR_CODE']);rev=revision(os.environ['WR_CODE_REVISION'])
    binding=source(ROOT,code,rev,ENTRY,HELPERS)
    require(sys.platform=='linux' and os.geteuid()==0 and os.uname().nodename==
        ('scenesmith-ncc-h100-01' if a.mode=='publish' else 'world-reward-ncc-h100-02'),'Direct Azure01 prediction source -> CPU02 destination only')
    def timeout(*_):raise TimeoutError('Inclusive direct private prediction-transfer budget')
    signal.signal(signal.SIGALRM,timeout);signal.signal(signal.SIGTERM,timeout);signal.alarm(BUDGET)
    if a.mode=='publish':
        require(a.package_bytes is None and a.package_sha256 is None,'Publish cannot accept unknown package pins')
        publish(code,a.prediction_revision,a.dev_revision,rev,binding)
    else:fetch(code,a.prediction_revision,a.dev_revision,dict(bytes=a.package_bytes,sha256=a.package_sha256),rev,binding)
    signal.alarm(0)


if __name__=='__main__':
    try:main()
    except Exception as error:
        # Do not expose Azure auth/errors/request bodies or report internals.
        print(json.dumps(dict(status='fail',error_type=type(error).__name__)),flush=True);raise SystemExit(1)
