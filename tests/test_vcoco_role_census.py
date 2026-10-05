"""Data-free census fixtures; no original annotation values, media or Torch."""
from copy import deepcopy
from collections import Counter
import io
import json
import os
from pathlib import Path
import stat
import zipfile

import pytest
import vcoco_role_census as c


def raw(value): return json.dumps(value,allow_nan=False).encode()
def categories(): return [dict(id=1,name='person')]+[dict(id=i,name=f'category_{i}') for i in range(2,81)]
def image(iid,partition='val2014',photo=None):
    filename=f'COCO_{partition}_{iid:012d}.jpg'
    return dict(id=iid,width=80,height=60,file_name=filename,license=4,
        coco_url=f'http://images.cocodataset.org/{partition}/{filename}',
        flickr_url=f'http://farm3.staticflickr.com/8/{photo or iid}_abcdef.jpg')
def annotation(aid,iid,category,*,crowd=0,box=None,area=4):
    return dict(id=aid,image_id=iid,category_id=category,iscrowd=crowd,bbox=box or [1,1,2,2],area=area)
def catalog_source(images,annotations=()):
    return dict(images=images,licenses=[dict(id=4,url=c.GRANT)],categories=categories(),annotations=list(annotations))


def test_original2014_catalog_skips_annotation_and_historical_semantics(monkeypatch):
    value=catalog_source([image(101),image(102)])
    source=raw(value).replace(b'"annotations": []',b'"annotations": [{"marker":"OLD_POISON","bbox":[1e999]}]')
    decoded=[]; original=c.js.strict_decode
    def recording(value): decoded.append(value); return original(value)
    monkeypatch.setattr(c.js,'strict_decode',recording)
    bank,taxonomy,counts,ids=c.catalog(io.BytesIO(source),'val2014',{'102'})
    assert ids=={101,102} and set(bank)=={101} and taxonomy[1]=='person'
    assert counts['historical_photo_images']==1 and bank[101]['creator_identity']=='UNKNOWN'
    assert not any(b'OLD_POISON' in r or b'1e999' in r for r in decoded)


def test_unknown_new_rights_identity_rejected_not_rewritten():
    values=[image(101),image(102),image(103)];values[1]['flickr_url']='unknown'
    values[2].update(license=2,flickr_url='not-even-parsed')
    bank,_,counts,_=c.catalog(io.BytesIO(raw(catalog_source(values))),'val2014',set())
    assert set(bank)=={101} and counts['publisher_photo_identity_unknown_images']==counts['publisher_rights_unknown_images']==1


@pytest.mark.parametrize('fault',['filename','coco_url','dimensions','duplicate_image','license_url','person_id','duplicate_category'])
def test_native_catalog_contradictions_fail_without_fallback(fault):
    value=catalog_source([image(101)])
    if fault=='filename':value['images'][0]['file_name']='000000000101.jpg'
    elif fault=='coco_url':value['images'][0]['coco_url']='http://images.cocodataset.org/val2017/000000000101.jpg'
    elif fault=='dimensions':value['images'][0]['width']=True
    elif fault=='duplicate_image':value['images'].append(deepcopy(value['images'][0]))
    elif fault=='license_url':value['licenses'][0]['url']='https://creativecommons.org/licenses/by/2.0/'
    elif fault=='person_id':value['categories'][0]['id']=99
    elif fault=='duplicate_category':value['categories'][-1]['id']=1
    with pytest.raises(ValueError):c.catalog(io.BytesIO(raw(value)),'val2014',set())


@pytest.mark.parametrize('source',[b'1\n1\n',b'0\n',b'1.0\n',b'1 \n',b'\n',b'1\n2\n\xff'])
def test_original_split_ids_ambiguous_or_noninteger_fail(source):
    with pytest.raises((ValueError,UnicodeError)):c.split_ids(source)


def test_original_split_order_preserved_as_ids_not_sampled():
    assert c.split_ids(b'17\n3\n21\n')=={17,3,21}


def test_all432_historical_slots_include_missing_without_opening_reference_paths():
    import base64
    md5=base64.b64encode(b'1'*16).decode();groups=[];counter=1
    def oi(number):
        return dict(ImageID=f'{number:016x}',OriginalLandingURL=f'https://www.flickr.com/photos/author/{number}/',
            AuthorProfileURL='https://www.flickr.com/people/author/',OriginalMD5=md5)
    for index,count in enumerate((16,128,64,32,32,64)):
        if index<4:
            rows=[oi(i) for i in range(counter,counter+count)]
            value=dict(public_metadata=rows,selected_ids=[r['ImageID'] for r in rows]) if index<2 else dict(records=[
                dict(slot=j,publisher_metadata=r,reference_file='/forbidden/not-opened',status='missing') for j,r in enumerate(rows)])
        else:value=dict(records=[dict(slot=j,image=image(i),reference_file='/forbidden/not-opened') for j,i in enumerate(range(counter,counter+count))])
        groups.append(value);counter+=count
    cohort=dict(records=[dict(slot=j,split=(['FIT']*32+['CAL']*16+['RESERVED']*48)[j],publisher_metadata=oi(counter+j)) for j in range(96)])
    acquired=dict(records=[dict(slot=j,status='missing') for j in range(96)])
    ledgers=[dict(records=[dict(slot=j,status='acquired',original_md5=md5) for j in range(n)]) for n in (32,64)]
    out=c.exclusions(groups,cohort,acquired,ledgers)
    assert len(out['photos'])==432 and len(out['authors'])==len(out['md5'])==1
    acquired['records'].pop()
    with pytest.raises(ValueError):c.exclusions(groups,cohort,acquired,ledgers)


def fixture(tmp_path,*,duplicate_photo=False,old_poison=False):
    values={p:catalog_source([]) for p in ('train2014','val2014')}
    for iid,partition in ((101,'train2014'),(202,'val2014')):
        values[partition]['images']=[image(iid,partition,photo=101 if duplicate_photo else None)]
        values[partition]['annotations']=[annotation(iid*10+j,iid,1 if j<3 else 17) for j in range(1,5)]
    if old_poison:
        values['train2014']['images'].append(image(303,'train2014'))
        values['train2014']['annotations'].append(dict(image_id=303,marker='OLD_POISON',bbox=['unconsulted']))
    archive=tmp_path/'metadata.zip'; catalogue=[]; expanded=[]
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for partition,value in values.items():
            member=f'annotations/instances_{partition}.json';content=raw(value);z.writestr(member,content)
            expanded.append(dict(member=member,expanded_bytes=len(content),expanded_sha256=c.pin(content)['sha256']))
    with zipfile.ZipFile(archive) as z:
        for i in z.infolist():catalogue.append(dict(member=i.filename,compressed_bytes=i.compress_size,expanded_bytes=i.file_size,
            crc32=f'{i.CRC:08x}',external_attr=i.external_attr,compression=i.compress_type,flags=i.flag_bits))
    tail_bytes=min(64,archive.stat().st_size); tail=c.pin(archive.read_bytes()[-tail_bytes:])
    cfg=dict(inputs={},archive=dict(member_catalogue=catalogue,catalogue_tail_identity=tail,
        instances_members={p:f'annotations/instances_{p}.json' for p in values}),splits={},roles={},
        max_expanded_bytes=512 << 20,max_row_bytes=1 << 20,max_role_bytes=16 << 20,max_field_bytes=2 << 20,
        minimum_distinct_photos=dict(val=1,test=1))
    def save(name,content):
        p=tmp_path/name;p.write_bytes(content);p.chmod(0o400)
        cfg['inputs'][name]=dict(path=str(p),pin=c.pin(content));return p
    cfg['inputs']['coco_archive']=dict(path=str(archive),pin=c.pin(archive.read_bytes()));archive.chmod(0o400)
    save('metadata_report',raw(dict(archive_members=expanded)))
    for name,ids in dict(train=[404],val=[101],test=[202],trainval=[404,101],all=[404,101,202]).items():
        if name in ('train','trainval','all'):ids=[i for i in ids if i != 404]  # Empty train is invalid; use val's empty substitute below.
        if name=='train':ids=[303 if old_poison else 505]
        if name=='trainval':ids=[303 if old_poison else 505,101]
        if name=='all':ids=[303 if old_poison else 505,101,202]
        key=f'split_{name}';save(key,(''.join(f'{i}\n' for i in ids)).encode());cfg['splits'][name]=key
    if not old_poison:
        # Structural train ID need only exist in the catalog, never semantic eligibility.
        with zipfile.ZipFile(archive) as z:contents={i.filename:z.read(i) for i in z.infolist()}
        value=json.loads(contents['annotations/instances_train2014.json']);value['images'].append(image(505,'train2014'))
        contents['annotations/instances_train2014.json']=raw(value)
        archive.chmod(0o600);archive.unlink();catalogue.clear();expanded.clear()
        with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
            for member,content in contents.items():z.writestr(member,content);expanded.append(dict(member=member,expanded_bytes=len(content),expanded_sha256=c.pin(content)['sha256']))
        with zipfile.ZipFile(archive) as z:
            for i in z.infolist():catalogue.append(dict(member=i.filename,compressed_bytes=i.compress_size,expanded_bytes=i.file_size,crc32=f'{i.CRC:08x}',external_attr=i.external_attr,compression=i.compress_type,flags=i.flag_bits))
        cfg['archive']['catalogue_tail_identity']=c.pin(archive.read_bytes()[-tail_bytes:]);cfg['inputs']['coco_archive']['pin']=c.pin(archive.read_bytes());archive.chmod(0o400)
        p=Path(cfg['inputs']['metadata_report']['path']);p.chmod(0o600);p.write_bytes(raw(dict(archive_members=expanded)));p.chmod(0o400);cfg['inputs']['metadata_report']['pin']=c.pin(p.read_bytes())
    for name,iid in (('val',101),('test',202)):
        a=dict(action_name='new_action',role_name=['agent','obj','instr'],ann_id=[iid*10+1,iid*10+2],image_id=[iid,iid],
            label=[1,1],role_object_id=[iid*10+1,iid*10+2,iid*10+3,iid*10+4,0,0])
        key=f'role_{name}';save(key,raw([a]));cfg['roles'][name]=key
    return cfg


def test_complete_fresh_metadata_counts_no_cohort_or_reference_exports(tmp_path):
    cfg=fixture(tmp_path);out=c.census(cfg,dict(photos=set()))
    assert out['capacity_gate_passed'] and out['eligibility_inventory_rows']==2
    for name in ('val','test'):
        assert out['splits'][name]['distinct_eligible_photos']==1
        assert out['splits'][name]['localized_positive_pairs']==2
        assert out['splits'][name]['eligible_images_two_positive_agents']==1
    assert 'images' not in out and 'references' not in out and 'eligibility_inventory_identity' in out


def test_historical_annotation_values_never_decoded(tmp_path,monkeypatch):
    cfg=fixture(tmp_path,old_poison=True);decoded=[];original=c.js.strict_decode
    def recording(value):decoded.append(value);return original(value)
    monkeypatch.setattr(c.js,'strict_decode',recording)
    out=c.census(cfg,dict(photos={'303'}))
    assert out['capacity_gate_passed'] and not any(b'OLD_POISON' in value for value in decoded)


def test_duplicate_photos_rejected_before_annotations_not_extra_capacity(tmp_path):
    cfg=fixture(tmp_path,duplicate_photo=True);out=c.census(cfg,dict(photos=set()))
    assert not out['capacity_gate_passed'] and out['eligibility_inventory_rows']==0
    assert out['catalog_counts']['duplicate_fresh_photo_images_rejected']==2


def test_insufficient_objects_or_positive_roles_close_capacity_not_repaired(tmp_path):
    cfg=fixture(tmp_path);p=Path(cfg['inputs']['role_val']['path']);p.chmod(0o600)
    a=json.loads(p.read_bytes());a[0]['label']=[0,0];p.write_bytes(raw(a));p.chmod(0o400);cfg['inputs']['role_val']['pin']=c.pin(p.read_bytes())
    out=c.census(cfg,dict(photos=set()))
    assert not out['capacity_gate_passed'] and out['splits']['val']['localized_positive_pairs']==0


def test_role_source_changes_before_second_open_detected_before_semantics(tmp_path,monkeypatch):
    cfg=fixture(tmp_path);original=c.rt.identity;calls=Counter()
    def changing(path,*args,**kwargs):
        calls[str(path)]+=1
        if str(path)==cfg['inputs']['role_val']['path'] and calls[str(path)]==2:return dict(bytes=1,sha256='0'*64)
        return original(path,*args,**kwargs)
    monkeypatch.setattr(c.rt,'identity',changing)
    with pytest.raises(ValueError,match='each open'):c.census(cfg,dict(photos=set()))


def test_member_expansion_original_hash_not_two_pass_self_assertion(tmp_path):
    cfg=fixture(tmp_path);p=Path(cfg['inputs']['metadata_report']['path']);p.chmod(0o600)
    r=json.loads(p.read_bytes());r['archive_members'][0]['expanded_sha256']='0'*64;p.write_bytes(raw(r));p.chmod(0o400);cfg['inputs']['metadata_report']['pin']=c.pin(p.read_bytes())
    with pytest.raises(ValueError,match='complete member SHA'):c.census(cfg,dict(photos=set()))


def test_complete_member_crc_failure_not_ignored(tmp_path):
    import struct
    cfg=fixture(tmp_path);p=Path(cfg['inputs']['coco_archive']['path']);p.chmod(0o600)
    value=bytearray(p.read_bytes());i=value.index(b'PK\x01\x02');crc=struct.unpack_from('<I',value,i+16)[0]
    struct.pack_into('<I',value,i+16,crc^1);p.write_bytes(value);p.chmod(0o400)
    with zipfile.ZipFile(p) as z:info=z.infolist()[0]
    spec=deepcopy(cfg['archive']['member_catalogue'][0]);spec['crc32']=f'{info.CRC:08x}'
    with pytest.raises(zipfile.BadZipFile):
        with c.member_stream(p,info.filename,spec,lambda:None) as source:
            while source.read(64):pass


def test_configuration_prospective_frozen_source_and_scope():
    cfg=json.loads((Path(c.__file__).resolve().parents[1]/c.CONFIG).read_bytes())
    assert c.configuration(cfg) is cfg
    for field,value in [('budget_seconds',600),('network_allowed',True),('minimum_noncrowd_people',1),('max_host_memory_bytes',16<<30)]:
        changed=deepcopy(cfg);changed[field]=value
        with pytest.raises(ValueError):c.configuration(changed)


def test_report_deadline_demotes_same_owned_inode_and_seals(tmp_path,monkeypatch):
    out=tmp_path/'owned';out.mkdir(mode=0o700);owned=out.lstat();report=dict(status='pass',decision='CAPACITY_METADATA_ONLY_NO_RGB')
    monkeypatch.setattr(c.os,'geteuid',lambda:0)
    # Ownership checks are exact native root; local test substitutes only captured UID.
    original=c.Path.lstat
    def metadata(path):
        value=original(path);return os.stat_result((value.st_mode,value.st_ino,value.st_dev,value.st_nlink,0,value.st_gid,value.st_size,value.st_atime,value.st_mtime,value.st_ctime))
    monkeypatch.setattr(c.Path,'lstat',metadata);owned=out.lstat()
    c.publish_report(out,report,-1,0,owned)
    saved=json.loads((out/'report.json').read_bytes())
    assert saved['status']=='fail' and saved['decision']=='CLOSED_CENSUS_PUBLICATION'
    assert stat.S_IMODE(out.stat().st_mode)==0o500 and stat.S_IMODE((out/'report.json').stat().st_mode)==0o400


def test_alarm_installed_before_preflight_and_restored(monkeypatch):
    calls=[]
    monkeypatch.setattr(c.signal,'signal',lambda s,h:calls.append(('handler',s)) or 'old')
    monkeypatch.setattr(c.signal,'setitimer',lambda *v:calls.append(('timer',v)))
    def preflight(started,deadline):
        assert calls[-1]==('timer',(c.signal.ITIMER_REAL,1200));raise ValueError('preflight')
    monkeypatch.setattr(c,'run',preflight)
    with pytest.raises(ValueError,match='preflight'):c.main()
    assert ('timer',(c.signal.ITIMER_REAL,0)) in calls


def test_foreign_output_directory_mode_never_repaired_or_written(tmp_path):
    out=tmp_path/'foreign';out.mkdir(mode=0o755)
    with pytest.raises(ValueError):c.publish_report(out,dict(status='pass'),999999999,0,out.lstat())
    assert not tuple(out.iterdir()) and stat.S_IMODE(out.stat().st_mode)==0o755
