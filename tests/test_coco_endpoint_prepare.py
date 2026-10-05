"""Procedural tiny COCO/JPEG fixtures only; no real media or network."""
from copy import deepcopy
import base64
import hashlib
import json
import io
from pathlib import Path
import stat
import time
import zipfile

import pytest

import coco_endpoint_prepare as p


def cfg():
    return json.loads((Path(__file__).resolve().parents[1]/p.CONFIG).read_bytes())


def excluded():
    return dict(ids={f'{i:016x}' for i in range(240)}, photos=set(), md5=set(), coco_ids=set())


def native(n=72):
    value=dict(licenses=[dict(id=4,url=p.old.LICENSE_URL)],categories=[dict(id=1,name='person'),dict(id=7,name='object')],images=[],annotations=[])
    for iid in range(1000,1000+n):
        value['images'].append(dict(id=iid,license=4,width=16,height=12,file_name=f'{iid:012d}.jpg',
            coco_url=f'http://images.cocodataset.org/val2017/{iid:012d}.jpg',flickr_url=f'http://farm1.staticflickr.com/1/{iid}_abcdef_z.jpg'))
        for j,box in enumerate(([0,0,2,2],[8,0,2,2],[0,8,2,2],[8,8,2,2])):
            value['annotations'].append(dict(id=iid*4+j,image_id=iid,category_id=1 if j<2 else 7,bbox=list(box),area=4,iscrowd=0))
    return value


def jpeg(url):
    frame=bytes([8])+bytes([0,12,0,16,3,1,0x11,0,2,0x11,0,3,0x11,0])
    iid=int(url.rsplit('/',1)[1][:-4])
    return b'\xff\xd8\xff\xe1\x00\x0a'+iid.to_bytes(8,'big')+b'\xff\xc0\x00\x11'+frame+b'\xff\xd9'


def selected():
    return p.select(native(),excluded())[0]


def old_values():
    values=[];offset=0
    for count in (16,128,64,32):
        rows=[dict(ImageID=f'{i:016x}',OriginalLandingURL=f'https://www.flickr.com/photos/a/{100000+i}/',
            OriginalMD5=base64.b64encode(i.to_bytes(16,'big')).decode()) for i in range(offset,offset+count)]
        if count in (16,128):values.append(dict(selected_ids=[r['ImageID'] for r in rows],public_metadata=rows))
        else:values.append(dict(schema='world_reward.openimages_joint_pair_cohort.v1' if count==64 else 'world_reward.proposal_external_cohort.v1',
            no_replacements=True,records=[dict(slot=j,split='DEV' if j<count//2 else 'TEST' if count==64 else 'RESERVED',publisher_metadata=r) for j,r in enumerate(rows)]))
        offset+=count
    return values


def test_source_reuse_exact_and_config_bounds():
    c=cfg();root=Path(__file__).resolve().parents[1]
    assert set(c['reused_helper_pins'])==set(p.old.HELPERS)
    for name,want in c['reused_helper_pins'].items():assert p.pin((root/name).read_bytes())==want
    assert p.old.NAMESPACE != p.NAMESPACE and p.old.DATA != p.DATA and len(p.HELPERS)==8
    assert c['slots']==64 and c['dev']==c['reserved']==32 and c['min_acquired_dev']==24
    assert c['workers']==6 and c['overall']==300 and c['request_timeout']==15 and c['retry_count']==0
    assert c['max_total_image_bytes']==c['slots']*c['max_image_bytes']==1<<30
    assert c['primary_terms']['creator_identity']=='UNKNOWN'
    assert p.acq.fetch.__module__=='openimages_joint_pair_acquire' and p.old.countable.__module__=='coco_proposal_prepare'


def test_deterministic64_separated_native_and_fresh_namespace_nonmutation():
    v=native();before=deepcopy(v);ex=excluded();ex['photos'].add('1000');ex['coco_ids'].add(1001)
    rows,counts=p.select(v,ex);v['images'].reverse();v['annotations'].reverse();reverse,_=p.select(v,ex)
    assert [r['image_id'] for r in rows]==[r['image_id'] for r in reverse]
    assert len(rows)==64 and [r['slot'] for r in rows]==list(range(64))
    assert [r['split'] for r in rows]==['DEV']*32+['RESERVED']*32 and counts['eligible']==70
    assert not {1000,1001}&{r['image_id'] for r in rows}
    assert before['annotations'][0] in v['annotations'] and all(len(r['annotations'])==4 for r in rows)
    assert rows[0]['image_id']!=p.old.select(before,dict(ids=ex['ids'],photos=set(),md5=set()))[0][0]['image_id']


def test_excluded_photo_annotations_skipped_before_bbox_values():
    v=native();ex=excluded();ex['coco_ids']={1000};ex['photos']={'1001'}
    for a in v['annotations']:
        if a['image_id'] in (1000,1001):
            del a['bbox'];del a['area'];del a['iscrowd'];del a['category_id'];del a['id']
    rows,_=p.select(v,ex)
    assert len(rows)==64 and not {1000,1001}&{r['image_id'] for r in rows}


@pytest.mark.parametrize('failure',['crowd','zero','overlap','license','samephoto'])
def test_capacity_fail_no_resample(failure):
    v=native(64)
    if failure=='crowd':v['annotations'][0]['iscrowd']=1
    elif failure=='zero':v['annotations'][0]['bbox'][2]=0
    elif failure=='overlap':v['annotations'][1]['bbox']=v['annotations'][0]['bbox'][:]
    elif failure=='license':v['images'][0]['license']=1
    else:v['images'][0]['flickr_url']=v['images'][1]['flickr_url']
    rows,counts=p.select(v,excluded());assert len(rows)==counts['selected']==63


@pytest.mark.parametrize('failure',['grant','duplicateimage','duplicateannotation','nan','foreigncategory','filename','grid'])
def test_native_schema_rejects_no_repair(failure):
    v=native()
    if failure=='grant':v['licenses'][0]['url']='https://creativecommons.org/licenses/by-nc/2.0/'
    elif failure=='duplicateimage':v['images'].append(v['images'][0])
    elif failure=='duplicateannotation':v['annotations'].append(v['annotations'][0])
    elif failure=='nan':v['annotations'][0]['bbox'][0]=float('nan')
    elif failure=='foreigncategory':v['annotations'][0]['category_id']=999
    elif failure=='filename':v['images'][0]['file_name']='../bad.jpg'
    else:v['images'][0]['height']=0
    with pytest.raises(ValueError):p.select(v,excluded())


def historical_fixture(tmp_path):
    c=cfg();c['historical']=[]
    for i,value in enumerate(old_values()):
        path=tmp_path/f'old{i}.json';want=p.write(path,value);c['historical'].append(dict(path=str(path),pin=want,readonly=True))
    root=tmp_path/'closed';root.mkdir();records=[]
    for slot in range(32):
        iid=200000+slot;ref=f'reference_{slot:06d}.json';want=p.write(root/ref,{'NEVER_DECODE':'closed reserved truth'})
        records.append(dict(slot=slot,split='DEV' if slot<16 else 'RESERVED',image_id=iid,image=dict(id=iid,flickr_url=f'http://farm1.staticflickr.com/1/{iid}_abcdef_z.jpg'),
            photo_id=str(iid),reference_file=ref,reference_identity=want))
    cohort=dict(schema='world_reward.coco_proposal_cohort.v1',producer_revision=p.CACHE_REV,freeze_before_rgb=True,
        no_replacements=True,hash_namespace=p.old.NAMESPACE,records=records,source_binding={'old':True},configuration_identity={'old':True})
    c['closed_coco_cohort']=dict(path=str(root/'cohort.json'),pin=p.write(root/'cohort.json',cohort))
    return c,cohort


def test_all272_historical_slots_reference_hash_only(tmp_path,monkeypatch):
    c,cohort=historical_fixture(tmp_path);original=Path.read_bytes
    def guarded(path):
        assert not path.name.startswith('reference_'),'closed truth never decoded'
        return original(path)
    monkeypatch.setattr(Path,'read_bytes',guarded)
    ex,files,value=p.historical(c)
    assert len(ex['ids'])==240 and len(ex['coco_ids'])==32 and len(ex['photos'])==272
    assert len(files)==37 and value==cohort


@pytest.mark.parametrize('bad',['cohortpin','referencepin','slot','split','photobinding','traversal','duplicatephoto'])
def test_historical_fail_closed(tmp_path,bad):
    c,cohort=historical_fixture(tmp_path);path=Path(c['closed_coco_cohort']['path'])
    if bad=='cohortpin':c['closed_coco_cohort']['pin']['sha256']='0'*64
    elif bad=='referencepin':cohort['records'][31]['reference_identity']['sha256']='0'*64
    elif bad=='slot':cohort['records'][31]['slot']=30
    elif bad=='split':cohort['records'][31]['split']='DEV'
    elif bad=='photobinding':cohort['records'][31]['photo_id']='1'
    elif bad=='traversal':cohort['records'][31]['reference_file']='../bad.json'
    else:cohort['records'][31]['photo_id']=cohort['records'][0]['photo_id']
    if bad!='cohortpin':path.chmod(0o600);path.unlink();c['closed_coco_cohort']['pin']=p.write(path,cohort)
    with pytest.raises(ValueError):p.historical(c)


def fake_proof(monkeypatch,c,value=None):
    ex=excluded();closed=dict(source_binding={'old':True},configuration_identity={'oldcfg':True})
    receipt=dict(configuration_identity={'oldcfg':True});files={'cache':{'bytes':1,'sha256':'a'*64}}
    monkeypatch.setattr(p,'cached_archive',lambda _: (receipt,{'old':True},files))
    monkeypatch.setattr(p,'historical',lambda _: (ex,{},closed))
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr(p.old.MEMBER,p.encode(native() if value is None else value))
        z.writestr('annotations/instances_train2017.json',b'UNPARSEABLE TRAIN CONTENT')
    raw=stream.getvalue();c['annotation_cache']['archive_pin']=p.pin(raw);c['archive_md5']=hashlib.md5(raw).hexdigest()
    return raw


def test_census_64_private_freeze_and_no_http(tmp_path,monkeypatch):
    c=cfg();raw=fake_proof(monkeypatch,c);cache=tmp_path/'archive.zip';cache.write_bytes(raw);c['annotation_cache']['archive_path']=str(cache)
    metadata=tmp_path/'metadata';metadata.mkdir();source={'producer_revision':'c'*40,'helpers':{p.CONFIG:{'bytes':1,'sha256':'a'*64}}}
    result=p.census(c,source,'c'*40,metadata,time.monotonic()+30)
    value=p.rt.pinned(metadata/'cohort.json',result['cohort_identity']);assert len(value['records'])==64
    assert result['rgb_read'] is False and result['network_used'] is False
    assert all('annotations' not in r for r in value['records']) and len(list(metadata.glob('reference_*.json')))==64


def test_census_capacity_failure_writes_no_cohort_or_references(tmp_path,monkeypatch):
    c=cfg();raw=fake_proof(monkeypatch,c,native(63));cache=tmp_path/'archive.zip';cache.write_bytes(raw);c['annotation_cache']['archive_path']=str(cache)
    out=tmp_path/'metadata';out.mkdir()
    result=p.census(c,{},'a'*40,out,time.monotonic()+30)
    assert result['counts']['selected']==63 and result['decision']=='INCONCLUSIVE_CLOSED_INSUFFICIENT_METADATA'
    assert not tuple(out.iterdir())


def test_original64_jpeg_requests_opaque_and_allslots(tmp_path):
    rows=selected();calls=[]
    def request(url,*_):calls.append(url);return jpeg(url)
    result=p.acquire(rows,excluded(),cfg(),tmp_path,time.monotonic()+30,request=request);p.verify_public(tmp_path,result,cfg())
    public=p.rt.pinned(tmp_path/'manifest.json',result['public_inputs_identity'])
    assert len(calls)==len(set(calls))==len(public['images'])==64
    assert all(u.startswith(p.old.BUCKET+'val2017/') for u in calls)
    assert all(set(r)==p.old.PUBLIC_KEYS and len(r['image_id'])==32 for r in public['images'])
    assert not any(k in json.dumps(public) for k in ('split','license','photo','bbox','DEV','RESERVED','annotations'))
    assert result['decision']=='READY_PENDING_SEPARATE_NATIVE_ENDPOINTS'
    assert all((tmp_path/r['file']).read_bytes()==jpeg(p.old.BUCKET+f"val2017/{rows[i]['image_id']:012d}.jpg") for i,r in enumerate(public['images']))


@pytest.mark.parametrize('failure',['http','grid','invalidjpeg','large','oldmd5','duplicatebytes'])
def test_missing64_no_retry_substitution_secrets_or_fake_rgb(tmp_path,failure):
    rows=selected();ex=excluded();calls=[];one=jpeg(p.old.BUCKET+'val2017/000000001000.jpg')
    if failure=='oldmd5':ex['md5'].add(base64.b64encode(hashlib.md5(one).digest()).decode())
    def request(url,*_):
        calls.append(url)
        if failure=='http':raise OSError('SECRET_IGNORE')
        if failure=='grid':return jpeg(url).replace(bytes([0,16,3]),bytes([0,17,3]))
        if failure=='invalidjpeg':return b'bad'
        if failure=='large':return one  # tiny lowered bound below tests actual oversize gate
        return one
    c=cfg()
    if failure=='large':c['max_image_bytes']=4
    result=p.acquire(rows,ex,c,tmp_path,time.monotonic()+30,request=request);p.verify_public(tmp_path,result,c)
    assert len(calls)==64 and result['counts']['missing']==64 and result['counts']['acquired']==0
    assert 'SECRET' not in json.dumps(result) and result['decision']=='INCONCLUSIVE_CLOSED_NO_MODEL'
    assert {x.name for x in tmp_path.iterdir()}=={'manifest.json'}


@pytest.mark.parametrize('good,ready',[(24,True),(23,False)])
def test_dev_capacity_not_reserved_tuning_and_original_gaps(tmp_path,good,ready):
    rows=selected();allowed={r['image_id'] for r in rows if r['slot']<good or r['split']=='RESERVED'}
    def request(url,*_):
        if int(url.rsplit('/',1)[1][:-4]) not in allowed:raise OSError('missing')
        return jpeg(url)
    result=p.acquire(rows,excluded(),cfg(),tmp_path,time.monotonic()+30,request=request);p.verify_public(tmp_path,result,cfg())
    assert result['counts']['acquired_DEV']==good and result['counts']['acquired_RESERVED']==32
    assert (result['decision']=='READY_PENDING_SEPARATE_NATIVE_ENDPOINTS') is ready
    assert result['public_mappings'][good]['public_file']=='image_000032.jpg'


def test_no_output_overwrite_or_foreign_file_reinterpretation(tmp_path):
    p.write(tmp_path/'manifest.json',{'old':True})
    with pytest.raises(FileExistsError):p.acquire(selected(),excluded(),cfg(),tmp_path,time.monotonic()+30,request=jpeg)


def test_wrapper_exact_immutable_entry_budget_environment_and_no_gpu():
    text=(Path(__file__).resolve().parents[1]/'infra/run_coco_endpoint_prepare.sh').read_text()
    assert 'run_coco_endpoint_prepare/code' in text and '310s /usr/bin/python3 -I -B' in text
    assert '--signal=TERM --kill-after=5s' in text and 'env -i' in text and 'set +x' in text
    assert 'docker' not in text and '--gpus' not in text and 'world-reward-ncc-h100-02' in text


def source_for_config():
    root=Path(__file__).resolve().parents[1]
    return dict(helpers={n:p.pin((root/n).read_bytes()) for n in p.HELPERS})


def test_configuration_unchanged_helpers_and_actual_origins(monkeypatch):
    root=Path(__file__).resolve().parents[1];source=source_for_config();c=cfg()
    def pinned(path,want,*_):
        assert path==root/p.CONFIG and want==source['helpers'][p.CONFIG]
        return c
    monkeypatch.setattr(p.rt,'pinned',pinned)
    assert p.configuration(root,source)==c
    c['slots']=32
    with pytest.raises(ValueError,match='recipe'):p.configuration(root,source)


def test_cached_archive_actual_source_receipt_and_byte_binding(monkeypatch):
    c=cfg();source=dict(helpers={n:c['reused_helper_pins'][n] for n in p.old.HELPERS});original=json.loads((Path(__file__).resolve().parents[1]/p.old.CONFIG).read_bytes())
    cache=c['annotation_cache'];receipt=dict(schema='world_reward.coco_proposal_prepare.v1',phase='census',status='pass',
        producer_revision=p.CACHE_REV,source_binding=source,configuration_identity=source['helpers'][p.old.CONFIG],
        archive_url=c['archive_url'],archive_publisher_md5=c['archive_md5'],archive_identity=cache['archive_pin'],archive_path=cache['archive_path'],
        cohort_identity=c['closed_coco_cohort']['pin'],freeze_before_rgb=True,rgb_read=False,source_and_inputs_rehashed_after=True,outputs_sealed=True)
    calls=[]
    def bound(root,code,rev,entry,helpers):
        calls.append((code,entry));assert rev==p.CACHE_REV and entry==p.old.ENTRY and helpers==p.old.HELPERS
        return source
    monkeypatch.setattr(p.rt,'source',bound)
    monkeypatch.setattr(p.rt,'pinned',lambda path,*_:original if path.name==Path(p.old.CONFIG).name else receipt)
    monkeypatch.setattr(p.rt,'canonical',lambda path:Path(path))
    monkeypatch.setattr(p.rt,'identity',lambda *_args,**_kwargs:cache['archive_pin'])
    value,oldsource,files=p.cached_archive(c)
    assert value==receipt and oldsource==source and len(files)==2 and len(calls)==1
    receipt['source_and_inputs_rehashed_after']=False
    with pytest.raises(ValueError,match='PASS'):p.cached_archive(c)


@pytest.mark.parametrize('failure',['receipt','source','bytes','paths'])
def test_cache_prevents_circular_or_contradictory_identity(monkeypatch,failure):
    c=cfg();s=dict(helpers={n:c['reused_helper_pins'][n] for n in p.old.HELPERS});orig=json.loads((Path(__file__).resolve().parents[1]/p.old.CONFIG).read_bytes())
    if failure=='source':s['helpers'][p.old.HELPERS[0]]=dict(bytes=1,sha256='0'*64)
    if failure=='paths':c['annotation_cache']['archive_path']='/foreign.zip'
    monkeypatch.setattr(p.rt,'source',lambda *_:s)
    monkeypatch.setattr(p.rt,'pinned',lambda path,*_:orig if path.name==Path(p.old.CONFIG).name else dict(status='fail'))
    monkeypatch.setattr(p.rt,'canonical',lambda path:Path(path))
    monkeypatch.setattr(p.rt,'identity',lambda *_args,**_kwargs:dict(bytes=1,sha256='0'*64))
    with pytest.raises((ValueError,KeyError)):p.cached_archive(c)


def frozen_fixture(tmp_path,monkeypatch):
    c=cfg();raw=fake_proof(monkeypatch,c);cache=tmp_path/'archive.zip';cache.write_bytes(raw);c['annotation_cache']['archive_path']=str(cache)
    metadata=tmp_path/'metadata';metadata.mkdir();source={'producer_revision':'c'*40,'helpers':{p.CONFIG:{'bytes':1,'sha256':'a'*64}}}
    report=p.census(c,source,'c'*40,metadata,time.monotonic()+30)
    report.update(schema=c['schema'],phase='census',status='pass',producer_revision='c'*40,source_binding=source,
        configuration_identity=source['helpers'][p.CONFIG],source_and_inputs_rehashed_after=True,outputs_sealed=True)
    rp=p.write(metadata/'report.json',report)
    return c,source,metadata,rp,report['cohort_identity']


def test_independent64_freeze_required_before_acquire_and_no_reference_decode(tmp_path,monkeypatch):
    c,s,m,rp,cp=frozen_fixture(tmp_path,monkeypatch);read=Path.read_bytes
    def safe(path):
        assert not path.name.startswith('reference_')
        return read(path)
    monkeypatch.setattr(Path,'read_bytes',safe)
    rows,ex,report=p.frozen_cohort(c,s,m,rp,cp)
    assert len(rows)==64 and report['freeze_before_rgb'] is True and report['network_used'] is False


@pytest.mark.parametrize('failure',['reportpin','cohortpin','ref','source','split','rawrefs','status','unsealed','producer','filename'])
def test_frozen_consumer_rejects_changed_preconditions(tmp_path,monkeypatch,failure):
    c,s,m,rp,cp=frozen_fixture(tmp_path,monkeypatch)
    if failure=='reportpin':rp=dict(rp,sha256='0'*64)
    elif failure=='cohortpin':cp=dict(cp,sha256='0'*64)
    elif failure=='ref':(m/'reference_000063.json').chmod(0o600)
    elif failure=='source':s=deepcopy(s);s['helpers'][p.CONFIG]['sha256']='0'*64
    else:
        path=m/('cohort.json' if failure in ('split','rawrefs','producer','filename') else 'report.json');value=p.rt.strict(path.read_bytes())
        if failure=='split':value['records'][63]['split']='DEV'
        elif failure=='rawrefs':value['records'][63]['annotations']=[]
        elif failure=='producer':value['producer_revision']='d'*40
        elif failure=='filename':value['records'][63]['image']['file_name']='alias.jpg'
        elif failure=='status':value['status']='fail'
        else:value['outputs_sealed']=False
        path.chmod(0o600);path.unlink();new=p.write(path,value)
        if path.name=='cohort.json':cp=new
        else:rp=new
    with pytest.raises(ValueError):p.frozen_cohort(c,s,m,rp,cp)


def run_fixture(tmp_path,monkeypatch):
    data=tmp_path/'data';c=cfg();c['output']=str(data);source={'helpers':{p.CONFIG:{'bytes':1,'sha256':'a'*64}}}
    monkeypatch.setattr(p,'DATA',data);monkeypatch.setattr(p.sys,'platform','linux')
    monkeypatch.setattr(p.os,'geteuid',lambda:0);monkeypatch.setattr(p.os,'uname',lambda:type('Host',(),{'nodename':'world-reward-ncc-h100-02'})())
    monkeypatch.setenv('WR_CODE',str(Path(__file__).resolve().parents[1]));monkeypatch.setenv('WR_CODE_REVISION','a'*40)
    monkeypatch.setattr(p.rt,'source',lambda *_:source);monkeypatch.setattr(p,'configuration',lambda *_:c)
    return data,c,source


def test_real_run_closed_capacity_receipt_never_pass_or_cohort(tmp_path,monkeypatch,capsys):
    data,c,source=run_fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(p,'census',lambda *_:dict(decision='INCONCLUSIVE_CLOSED_INSUFFICIENT_METADATA',counts={'selected':63},rgb_read=False,network_used=False))
    result=p.run('census');assert result['status']=='fail' and result['source_and_inputs_rehashed_after'] is True
    assert result['outputs_sealed'] is True and {x.name for x in (data/'metadata').iterdir()}=={'report.json'}
    assert stat.S_IMODE(data.stat().st_mode)==stat.S_IMODE((data/'metadata').stat().st_mode)==0o500
    assert stat.S_IMODE((data/'metadata'/'report.json').stat().st_mode)==0o400
    assert json.loads(capsys.readouterr().out)['status']=='fail'


def test_real_run_failure_seals_without_secret_or_false_success(tmp_path,monkeypatch,capsys):
    data,c,source=run_fixture(tmp_path,monkeypatch)
    def failing(*_):raise OSError('PRIVATE_SECRET_DO_NOT_PRINT')
    monkeypatch.setattr(p,'census',failing)
    result=p.run('census');assert result['status']=='fail' and result['error_type']=='OSError'
    assert result['source_and_inputs_rehashed_after'] is False and result['outputs_sealed'] is True
    assert 'PRIVATE_SECRET' not in (data/'metadata'/'report.json').read_text()+capsys.readouterr().out


def test_real_run_acquire_before_after_freeze_and_failure_seals(tmp_path,monkeypatch,capsys):
    data,c,s=run_fixture(tmp_path,monkeypatch);data.mkdir();(data/'metadata').mkdir();events=[];rows=selected()
    def freeze(*_):events.append('freeze');return rows,excluded(),{'old':True}
    def acquire(*_):events.append('request');raise OSError('SECRET')
    monkeypatch.setattr(p,'frozen_cohort',freeze);monkeypatch.setattr(p,'acquire',acquire)
    result=p.run('acquire',({'bytes':1,'sha256':'a'*64},{'bytes':1,'sha256':'b'*64}))
    assert events==['freeze','request'] and result['status']=='fail' and len(result['selected_slots'])==64
    assert all(stat.S_IMODE(x.stat().st_mode)==0o500 for x in (data,data/'inputs',data/'eval_private'))
    assert stat.S_IMODE((data/'eval_private'/'manifest.json').stat().st_mode)==0o400
    assert result['reference_values_read_for_acquisition'] is False and 'SECRET' not in capsys.readouterr().out


def test_real_run_success_requires_post_source_and_seals(tmp_path,monkeypatch,capsys):
    data,c,s=run_fixture(tmp_path,monkeypatch);events=[]
    def census(*_):events.append('freeze');return dict(decision='FROZEN64_PENDING_SEPARATE_ACQUISITION',cohort_identity={'bytes':1,'sha256':'a'*64},rgb_read=False)
    def source(*_):events.append('source');return s
    monkeypatch.setattr(p,'census',census);monkeypatch.setattr(p.rt,'source',source)
    report=p.run('census')
    assert events==['source','freeze','source'] and report['status']=='pass'
    assert report['source_and_inputs_rehashed_after'] is True and report['outputs_sealed'] is True
    assert json.loads(capsys.readouterr().out)['status']=='pass'


def test_after_source_mismatch_cannot_be_pass_receipt(tmp_path,monkeypatch,capsys):
    data,c,s=run_fixture(tmp_path,monkeypatch);calls=[]
    def source(*_):calls.append(1);return s if len(calls)==1 else {'changed':True}
    monkeypatch.setattr(p.rt,'source',source)
    monkeypatch.setattr(p,'census',lambda *_:dict(decision='FROZEN64_PENDING_SEPARATE_ACQUISITION'))
    report=p.run('census')
    assert report['status']=='fail' and report['source_and_inputs_rehashed_after'] is False and report['outputs_sealed'] is True
    assert json.loads(capsys.readouterr().out)['status']=='fail'


def test_run_complete_but_dev_capacity_closed_is_failure_receipt(tmp_path,monkeypatch,capsys):
    data,c,s=run_fixture(tmp_path,monkeypatch);data.mkdir();(data/'metadata').mkdir()
    monkeypatch.setattr(p,'frozen_cohort',lambda *_:(selected(),excluded(),{}))
    monkeypatch.setattr(p,'acquire',lambda *_:dict(decision='INCONCLUSIVE_CLOSED_NO_MODEL',counts={'acquired_DEV':23},records=[]))
    monkeypatch.setattr(p,'verify_public',lambda *_:None)
    report=p.run('acquire',({'bytes':1,'sha256':'a'*64},{'bytes':1,'sha256':'b'*64}))
    assert report['status']=='fail' and report['decision']=='INCONCLUSIVE_CLOSED_NO_MODEL'
    assert report['source_and_inputs_rehashed_after'] is True and report['outputs_sealed'] is True
    assert json.loads(capsys.readouterr().out)['status']=='fail'
