"""Manufactured metadata/JPEG and mocked HTTPS only, no real data or quality."""
from copy import deepcopy
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import sys
import time
import zipfile

import pytest

import coco_proposal_prepare as p


def config():
    return json.loads((Path(__file__).resolve().parents[1]/p.CONFIG).read_bytes())


def jpeg(width=16,height=12):
    frame=bytes([8])+height.to_bytes(2,'big')+width.to_bytes(2,'big')+bytes([3])+bytes([1,0x11,0,2,0x11,0,3,0x11,0])
    return b'\xff\xd8\xff\xe0\x00\x04AB\xff\xc0\x00\x11'+frame+b'\xff\xd9'


def image_jpeg(url,width=16,height=12):
    """Distinct original-byte identities, identical manufactured grid."""
    raw=jpeg(width,height);value=int(url.rsplit('/',1)[1][:-4]);payload=value.to_bytes(8,'big')
    return raw[:2]+b'\xff\xe1\x00\x0a'+payload+raw[2:]


def native(count=40):
    value=dict(licenses=[dict(id=4,url=p.LICENSE_URL,name='Attribution License')],
        categories=[dict(id=1,name='person'),dict(id=7,name='object')],images=[],annotations=[])
    for i in range(1000,1000+count):
        value['images'].append(dict(id=i,license=4,width=16,height=12,file_name=f'{i:012d}.jpg',
            coco_url=f'http://images.cocodataset.org/val2017/{i:012d}.jpg',flickr_url=f'http://farm1.staticflickr.com/1/{i}_abcdef_z.jpg',date_captured='original'))
        for j,box in enumerate(([0,0,2,2],[8,0,2,2],[0,8,2,2],[8,8,2,2])):
            value['annotations'].append(dict(id=i*4+j,image_id=i,category_id=1 if j<2 else 7,
                bbox=list(box),area=4,iscrowd=0,segmentation=[[box[0],box[1],box[0]+2,box[1],box[0]+2,box[1]+2]]))
    return value


def excluded():
    return dict(ids={f'{i:016x}' for i in range(240)},photos=set(),md5=set())


def archive(value=None,*,extras=(),duplicate=False):
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr('annotations/',''); z.writestr('annotations/instances_train2017.json',b'INVALID SECRET TRAIN JSON')
        z.writestr(p.MEMBER,p.encode(native() if value is None else value))
        if duplicate:
            with pytest.warns(UserWarning,match='Duplicate'):z.writestr(p.MEMBER,b'{}')
        for name,content in extras:z.writestr(name,content)
    return stream.getvalue()


def old_values():
    values=[];offset=0
    for count in (16,128,64,32):
        rows=[dict(ImageID=f'{i:016x}',OriginalLandingURL=f'https://www.flickr.com/photos/a/{100000+i}/',
                   OriginalMD5=base64.b64encode(i.to_bytes(16,'big')).decode()) for i in range(offset,offset+count)]
        if count<64 or count==128:
            if count==32:
                values.append(dict(schema='world_reward.proposal_external_cohort.v1',no_replacements=True,
                    records=[dict(slot=j,split='DEV' if j<16 else 'RESERVED',publisher_metadata=r) for j,r in enumerate(rows)]))
            else:values.append(dict(selected_ids=[r['ImageID'] for r in rows],public_metadata=rows))
        else:
            values.append(dict(schema='world_reward.openimages_joint_pair_cohort.v1',no_replacements=True,
                records=[dict(slot=j,split='DEV' if j<32 else 'TEST',publisher_metadata=r) for j,r in enumerate(rows)]))
        offset+=count
    return values


def test_original_safe_helpers_and_frozen_config_pins():
    cfg=config();root=Path(__file__).resolve().parents[1]
    for name,wanted in cfg['reused_helper_pins'].items():assert p.pin((root/name).read_bytes())==wanted
    assert p.acq.fetch.__module__=='openimages_joint_pair_acquire'
    assert p.acq.jpeg_header.__module__=='openimages_joint_pair_acquire'
    assert cfg['archive_bytes']==252907541 and cfg['annotation_cache'] is None
    assert cfg['primary_terms']['image_license_id']==4 and cfg['primary_terms']['creator_identity']=='UNKNOWN'
    assert cfg['workers']==6 and cfg['request_timeout']==15 and cfg['overall']==300


def test_native_val_only_not_train_inflated_or_parsed():
    raw=archive();v,pin=p.val_metadata(raw)
    assert v==native() and pin==p.pin(p.encode(native()))
    assert 'SECRET TRAIN' not in p.encode(v).decode()


@pytest.mark.parametrize('names',['duplicate','path','symlink','missing','badzip'])
def test_archive_boundaries(names):
    if names=='duplicate':raw=archive(duplicate=True)
    elif names=='path':raw=archive(extras=[('../private.json',b'{}')])
    elif names=='missing':
        s=io.BytesIO()
        with zipfile.ZipFile(s,'w') as z:z.writestr('annotations/instances_train2017.json',b'{}')
        raw=s.getvalue()
    elif names=='symlink':
        s=io.BytesIO()
        with zipfile.ZipFile(s,'w') as z:
            z.writestr(p.MEMBER,p.encode(native())); info=zipfile.ZipInfo('annotations/link.json');info.external_attr=(stat.S_IFLNK|0o777)<<16;z.writestr(info,b'outside')
        raw=s.getvalue()
    else:raw=b'not ZIP'
    with pytest.raises((ValueError,zipfile.BadZipFile)):p.val_metadata(raw)


@pytest.mark.parametrize('url',[
    'http://farm9.staticflickr.com/123/456_abcdef.jpg','https://live.staticflickr.com/123/456_0123_z.jpg'])
def test_native_flickr_photo_routes(url):assert p.photo_identity(url)=='456'


@pytest.mark.parametrize('url',['https://farm1.staticflickr.com@evil/1/3_a.jpg','https://evil/1/3_a.jpg',
    'https://farm1.staticflickr.com/1/3_a.jpg?secret=x','https://farm1.staticflickr.com/1/3_a.jpg#x',
    'https://farm1.staticflickr.com/1/nonid_a.jpg'])
def test_static_url_rejects_foreign_identity(url):
    with pytest.raises(ValueError):p.photo_identity(url)


def test_all240_historical_slots_including_missing():
    values=old_values();ex=p.exclusions(values)
    assert len(ex['ids'])==len(ex['photos'])==len(ex['md5'])==240
    values[3]['records'][0]['publisher_metadata']['ImageID']=values[0]['selected_ids'][0]
    with pytest.raises(ValueError):p.exclusions(values)


def test_selection_once_deterministic_disjoint_photo_and_split():
    v=native();before=deepcopy(v);ex=excluded();ex['photos'].add('1000')
    a,counts=p.select(v,ex);v['images'].reverse();v['annotations'].reverse();b,_=p.select(v,ex)
    assert [r['image_id'] for r in a]==[r['image_id'] for r in b]
    assert [r['slot'] for r in a]==list(range(32)) and [r['split'] for r in a]==['DEV']*16+['RESERVED']*16
    assert len({r['photo_id'] for r in a})==32 and '1000' not in {r['photo_id'] for r in a}
    assert counts['eligible']==39 and counts['excluded_ids']==240
    assert a[0]['annotations'] and before['annotations'][0] in v['annotations']


@pytest.mark.parametrize('failure',['grant','person','image_duplicate','annotation_duplicate','nan','foreign_category','filename','grid'])
def test_native_metadata_mismatch_no_repair(failure):
    v=native()
    if failure=='grant':v['licenses'][0]['url']='http://creativecommons.org/licenses/by-nc/2.0/'
    elif failure=='person':v['categories'][0]['name']='human_alias'
    elif failure=='image_duplicate':v['images'].append(v['images'][0])
    elif failure=='annotation_duplicate':v['annotations'].append(v['annotations'][0])
    elif failure=='nan':v['annotations'][0]['bbox'][0]=float('nan')
    elif failure=='foreign_category':v['annotations'][0]['category_id']=99
    elif failure=='filename':v['images'][0]['file_name']='../x.jpg'
    else:v['images'][0]['width']=0
    with pytest.raises(ValueError):p.select(v,excluded())


@pytest.mark.parametrize('failure',['crowd','zero','overlap','license','samephoto'])
def test_insufficient_metadata_closes_before_rgb(failure):
    v=native(32)
    if failure=='crowd':v['annotations'][0]['iscrowd']=1
    elif failure=='zero':v['annotations'][0]['bbox'][2]=0
    elif failure=='overlap':v['annotations'][1]['bbox']=v['annotations'][0]['bbox'][:]
    elif failure=='license':v['images'][0]['license']=1
    else:v['images'][0]['flickr_url']=v['images'][1]['flickr_url']
    with pytest.raises(ValueError,match='Insufficient'):p.select(v,excluded())


def test_census_private_references_freeze_without_any_rgb_request(tmp_path,monkeypatch):
    cfg=config();raw=archive();cfg.update(archive_bytes=len(raw),archive_md5=hashlib.md5(raw).hexdigest())
    source=dict(helpers={p.CONFIG:dict(bytes=12,sha256='a'*64)});ex=excluded();frozen={'old':{'bytes':1,'sha256':'b'*64}};calls=[]
    monkeypatch.setattr(p,'historical',lambda _: (ex,frozen))
    def request(url,*_):calls.append(url);return raw
    result=p.census(cfg,source,'c'*40,tmp_path,time.monotonic()+30,request=request)
    assert calls==[cfg['archive_url']] and result['rgb_read'] is False and result['freeze_before_rgb'] is True
    cohort=p.rt.pinned(tmp_path/'cohort.json',result['cohort_identity'],16<<20)
    assert len(cohort['records'])==32 and cohort['author_identity']=='UNKNOWN'
    assert all('annotations' not in r and set(r)=={'slot','split','image_id','image','photo_id','reference_file','reference_identity'} for r in cohort['records'])
    for r in cohort['records']:
        refs=p.rt.pinned(tmp_path/r['reference_file'],r['reference_identity'],4<<20)
        assert set(refs)=={'annotations'} and len(refs['annotations'])==4
    assert result['archive_identity']==p.pin(raw) and result['member_identity']==p.pin(p.encode(native()))


def test_annotation_bytes_md5_failure_never_writes_cohort(tmp_path,monkeypatch):
    cfg=config();monkeypatch.setattr(p,'historical',lambda _: (excluded(),{}))
    with pytest.raises(ValueError,match='publisher'):p.census(cfg,{'helpers':{}},'a'*40,tmp_path,time.monotonic()+30,request=lambda *_:b'bad')
    assert not tuple(tmp_path.iterdir())


def freeze_fixture(tmp_path,monkeypatch):
    cfg=config();raw=archive();cfg.update(archive_bytes=len(raw),archive_md5=hashlib.md5(raw).hexdigest())
    source=dict(helpers={p.CONFIG:dict(bytes=12,sha256='a'*64)});ex=excluded();frozen={'old':{'bytes':1,'sha256':'b'*64}}
    monkeypatch.setattr(p,'historical',lambda _: (ex,frozen))
    receipt=p.census(cfg,source,'c'*40,tmp_path,time.monotonic()+30,request=lambda *_:raw)
    receipt.update(schema=cfg['schema'],phase='census',status='pass',source_binding=source,
        configuration_identity=source['helpers'][p.CONFIG],source_and_inputs_rehashed_after=True,outputs_sealed=True)
    rp=p.write(tmp_path/'report.json',receipt)
    return cfg,source,rp,receipt['cohort_identity']


def test_frozen_consumer_hashes_all_references_without_decoding(tmp_path,monkeypatch):
    cfg,source,rp,cp=freeze_fixture(tmp_path,monkeypatch)
    read=Path.read_bytes
    def safe_read(path):
        assert not path.name.startswith('reference_'), 'References cannot be semantically read in acquisition'
        return read(path)
    monkeypatch.setattr(Path,'read_bytes',safe_read)
    rows,ex,report=p.frozen_cohort(cfg,source,tmp_path,rp,cp)
    assert len(rows)==32 and len(ex['ids'])==240 and report['freeze_before_rgb'] is True


@pytest.mark.parametrize('failure',['reportpin','cohortpin','archive','reference','source','reportfail','unsealed','rawrefs'])
def test_frozen_consumer_rejects_changed_proof_before_rgb(tmp_path,monkeypatch,failure):
    cfg,source,rp,cp=freeze_fixture(tmp_path,monkeypatch)
    if failure=='reportpin':rp=dict(rp,sha256='0'*64)
    elif failure=='cohortpin':cp=dict(cp,sha256='0'*64)
    elif failure in ('archive','reference'):
        leaf=tmp_path/(p.ARCHIVE if failure=='archive' else 'reference_000000.json');leaf.chmod(0o600);leaf.write_bytes(b'changed');leaf.chmod(0o400)
    elif failure=='source':source={'helpers':{p.CONFIG:dict(bytes=12,sha256='0'*64)}}
    else:
        leaf=tmp_path/('cohort.json' if failure=='rawrefs' else 'report.json');value=p.rt.strict(leaf.read_bytes())
        if failure=='rawrefs':value['records'][0]['annotations']=[]
        elif failure=='reportfail':value['status']='fail'
        else:value['outputs_sealed']=False
        leaf.chmod(0o600);leaf.write_bytes(p.encode(value));leaf.chmod(0o400)
        if failure=='rawrefs':cp=p.pin(leaf.read_bytes());r=tmp_path/'report.json';report=p.rt.strict(r.read_bytes());report['cohort_identity']=cp;r.chmod(0o600);r.write_bytes(p.encode(report));r.chmod(0o400);rp=p.pin(r.read_bytes())
        else:rp=p.pin(leaf.read_bytes())
    with pytest.raises(ValueError):p.frozen_cohort(cfg,source,tmp_path,rp,cp)


def test_authenticated_cache_no_http_and_no_fake_whole_sha(tmp_path,monkeypatch):
    cfg=config();raw=archive();cache=tmp_path/'cache';cache.mkdir();metadata=tmp_path/'metadata';metadata.mkdir()
    p.rt.write(cache/p.ARCHIVE,raw);receipt=dict(status='pass',archive_url=cfg['archive_url'],archive_identity=p.pin(raw),archive_publisher_md5=hashlib.md5(raw).hexdigest())
    rp=p.write(cache/'receipt.json',receipt);cfg.update(archive_bytes=len(raw),archive_md5=hashlib.md5(raw).hexdigest(),
        annotation_cache=dict(archive_path=str(cache/p.ARCHIVE),report=dict(path=str(cache/'receipt.json'),pin=rp)))
    monkeypatch.setattr(p,'historical',lambda _: (excluded(),{}));source={'helpers':{p.CONFIG:dict(bytes=12,sha256='a'*64)}}
    def forbidden(*_):pytest.fail('Authenticated metadata cache must not download')
    result=p.census(cfg,source,'c'*40,metadata,time.monotonic()+30,request=forbidden)
    assert result['archive_identity']==p.pin(raw) and result['archive_path']==str(cache/p.ARCHIVE)
    assert not (metadata/p.ARCHIVE).exists()


def selected():return p.select(native(32),excluded())[0]


def test_original_rgb_fixed_slots_public_private_boundary(tmp_path):
    cfg=config();rows=selected();before=deepcopy(rows);calls=[]
    def request(url,maximum,deadline,timeout):calls.append((url,maximum,timeout));return image_jpeg(url)
    result=p.acquire(rows,excluded(),cfg,tmp_path,time.monotonic()+30,request=request)
    p.verify_public(tmp_path,result,cfg)
    assert rows==before and result['counts']==dict(slots=32,acquired=32,missing=0,acquired_DEV=16,acquired_RESERVED=16)
    assert len(calls)==32 and all(url.startswith(p.BUCKET+'val2017/') and maximum==16<<20 and timeout==15 for url,maximum,timeout in calls)
    manifest=p.rt.pinned(tmp_path/'manifest.json',result['public_inputs_identity'])
    assert all(set(r)==p.PUBLIC_KEYS and (tmp_path/r['file']).read_bytes()==image_jpeg(p.BUCKET+f"val2017/{rows[i]['image_id']:012d}.jpg") for i,r in enumerate(manifest['images']))
    assert [r['file'] for r in manifest['images']]==[f'image_{i:06d}.jpg' for i in range(32)]
    assert not any(s in json.dumps(manifest) for s in ('annotations','flickr','license','DEV','category','bbox','photo'))
    assert result['decision']=='READY_PENDING_SEPARATE_NATIVE_BANK'


@pytest.mark.parametrize('failure',['http','large','grid','invalidjpeg','historical_md5'])
def test_no_retry_replacement_or_fake_rgb_and_no_secret_error(tmp_path,failure):
    cfg=config();rows=selected();ex=excluded();calls=[]
    if failure=='historical_md5':ex['md5'].add(base64.b64encode(hashlib.md5(jpeg()).digest()).decode())
    def request(url,*_):
        calls.append(url)
        if failure=='http':raise OSError('SECRET_MUST_NOT_LEAK')
        if failure=='large':return b'x'*(cfg['max_image_bytes']+1)
        if failure=='grid':return jpeg(17,12)
        if failure=='invalidjpeg':return b'not RGB'
        if failure=='historical_md5':return jpeg()
        return image_jpeg(url)
    result=p.acquire(rows,ex,cfg,tmp_path,time.monotonic()+60,request=request)
    p.verify_public(tmp_path,result,cfg)
    assert len(calls)==len(result['records'])==32 and result['counts']['missing']==32
    assert result['decision']=='INCONCLUSIVE_CLOSED_NO_MODEL' and result['public_mappings']==[]
    assert p.rt.pinned(tmp_path/'manifest.json',result['public_inputs_identity'])['images']==[]
    assert 'SECRET' not in json.dumps(result) and {r['slot'] for r in result['records']}==set(range(32))
    assert {p.name for p in tmp_path.iterdir()}=={'manifest.json'}


def test_missing_slot_keeps_original_gapped_filename(tmp_path):
    rows=selected();missing_id=rows[0]['image_id'];calls=[]
    def request(url,*_):
        calls.append(url)
        if url.endswith(f'{missing_id:012d}.jpg'):raise OSError('not available')
        return image_jpeg(url)
    result=p.acquire(rows,excluded(),config(),tmp_path,time.monotonic()+30,request=request)
    p.verify_public(tmp_path,result,config())
    assert len(calls)==32 and result['records'][0]['status']=='unavailable'
    assert result['public_mappings'][0]['public_file']=='image_000001.jpg'


@pytest.mark.parametrize('dev,decision',[(12,'READY_PENDING_SEPARATE_NATIVE_BANK'),(11,'INCONCLUSIVE_CLOSED_NO_MODEL')])
def test_predeclared_dev_capacity_not_global_or_reserved_count(tmp_path,dev,decision):
    rows=selected();good={r['image_id'] for r in rows if r['slot']<dev or r['split']=='RESERVED'}
    def request(url,*_):
        iid=int(url.rsplit('/',1)[1][:-4])
        if iid not in good:raise OSError('missing')
        return image_jpeg(url)
    result=p.acquire(rows,excluded(),config(),tmp_path,time.monotonic()+30,request=request)
    assert result['counts']['acquired_DEV']==dev and result['counts']['acquired_RESERVED']==16 and result['decision']==decision


def test_jpeg_postwrite_mismatch_removes_only_owned_image(tmp_path,monkeypatch):
    original=p.rt.identity
    def fail(path,*args,**kwargs):
        if Path(path).suffix=='.jpg':raise ValueError('SECRET_BAD_POSTHASH')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(p.rt,'identity',fail)
    result=p.acquire(selected(),excluded(),config(),tmp_path,time.monotonic()+30,request=lambda *_:jpeg())
    assert result['counts']['missing']==32 and {path.name for path in tmp_path.iterdir()}=={'manifest.json'}
    assert 'SECRET' not in json.dumps(result)


def test_duplicate_newcohort_original_bytes_censors_all_conflicting_slots(tmp_path):
    result=p.acquire(selected(),excluded(),config(),tmp_path,time.monotonic()+30,request=lambda *_:jpeg())
    p.verify_public(tmp_path,result,config())
    assert result['counts']['missing']==32 and result['counts']['acquired']==0
    assert all(r['status']=='unavailable' and r['reason']=='duplicate_cohort_bytes' for r in result['records'])
    assert {path.name for path in tmp_path.iterdir()}=={'manifest.json'}


def test_only_duplicate_group_removed_not_firstslot_favored(tmp_path):
    rows=selected();duplicate={rows[0]['image_id'],rows[1]['image_id']}
    def request(url,*_):return jpeg() if int(url.rsplit('/',1)[1][:-4]) in duplicate else image_jpeg(url)
    result=p.acquire(rows,excluded(),config(),tmp_path,time.monotonic()+30,request=request)
    p.verify_public(tmp_path,result,config())
    assert result['counts']['acquired']==30 and result['records'][0]['reason']==result['records'][1]['reason']=='duplicate_cohort_bytes'
    assert result['public_mappings'][0]['slot']==2


def test_private_public_mutation_rejected(tmp_path):
    result=p.acquire(selected(),excluded(),config(),tmp_path,time.monotonic()+30,request=lambda url,*_:image_jpeg(url))
    file=tmp_path/'manifest.json';v=p.rt.pinned(file,result['public_inputs_identity']);v['images'][0]['annotations']=[]
    file.chmod(0o600);file.write_bytes(p.encode(v));file.chmod(0o400);result['public_inputs_identity']=p.pin(file.read_bytes())
    with pytest.raises(ValueError):p.verify_public(tmp_path,result,config())


def test_deadline_prevents_first_request(tmp_path):
    calls=[]
    with pytest.raises(ValueError):p.acquire(selected(),excluded(),config(),tmp_path,time.monotonic()-1,request=lambda *a:calls.append(a))
    assert calls==[] and not tuple(tmp_path.iterdir())


def test_script_small_ast_import_has_no_implicit_http_and_clean_wrapper():
    root=Path(__file__).resolve().parents[1];src=(root/p.HELPERS[0]).read_text();compile(src,p.HELPERS[0],'exec')
    wrapper=(root/p.HELPERS[1]).read_text()
    assert '-I -B' in wrapper and 'env -i' in wrapper and 'set +x' in wrapper and '310s' in wrapper
    assert 'creator_rights(' not in src and 'requests.' not in src and 'PIL' not in src and 'import torch' not in src
    assert 'str(exc)' not in src and 'traceback' not in src
    assert "sys.exit(0 if report['status']=='pass' else 1)" in src


def test_lifecycle_phase2_requires_freeze_and_seals_no_raw_errors(tmp_path,monkeypatch,capsys):
    data=tmp_path/'data';metadata=data/'metadata';data.mkdir(mode=0o700);metadata.mkdir(mode=0o500)
    cfg=config();source={'helpers':{p.CONFIG:dict(bytes=1,sha256='a'*64)}};rows=selected();ex=excluded();prior={'prior':'proof'};calls=[]
    monkeypatch.setattr(p,'DATA',data);monkeypatch.setattr(p.os,'geteuid',lambda:0);monkeypatch.setattr(p.sys,'platform','linux')
    monkeypatch.setattr(p.os,'uname',lambda:type('U',(),{'nodename':'world-reward-ncc-h100-02'})())
    code=Path(p.__file__).resolve().parents[1];monkeypatch.setenv('WR_CODE',str(code));monkeypatch.setenv('WR_CODE_REVISION','a'*40)
    monkeypatch.setattr(p.rt,'source',lambda *_:source);monkeypatch.setattr(p,'configuration',lambda *_:cfg)
    monkeypatch.setattr(p,'frozen_cohort',lambda *_:(calls.append('freeze') or (rows,ex,prior)))
    def fail(*_):calls.append('http');raise OSError('SECRET_NEVER_OUTPUT')
    monkeypatch.setattr(p,'acquire',fail)
    rp=dict(bytes=10,sha256='b'*64);cp=dict(bytes=20,sha256='c'*64);report=p.run('acquire',(rp,cp))
    assert calls==['freeze','http'] and report['status']=='fail' and report['error_type']=='OSError'
    assert len(report['selected_slots'])==32 and report['outputs_sealed'] is True
    assert report['reference_values_read_for_acquisition'] is False and report['rgb_requests_before_freeze']==0
    text=(data/'eval_private'/'manifest.json').read_text()+capsys.readouterr().out
    assert 'SECRET' not in text
    assert all(path.stat().st_mode&0o777==0o500 for path in (data,data/'inputs',data/'eval_private'))
    assert (data/'eval_private'/'manifest.json').stat().st_mode&0o777==0o400
