"""Authored metadata/JPEG headers and mocked requests only, never real photos."""
from collections import Counter
from copy import deepcopy
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import time
import zipfile

import pytest
import vcoco_role_prepare as p


def pin(value):return p.c.pin(p.c.encode(value))
def image(iid,partition='val2014'):
    name=f'COCO_{partition}_{iid:012d}.jpg'
    return dict(id=iid,width=16,height=12,file_name=name,partition=partition,license=4,license_url=p.c.GRANT,
        coco_url=f'http://images.cocodataset.org/{partition}/{name}',
        flickr_url=f'http://farm3.staticflickr.com/8/{iid}_abcdef.jpg',creator_identity='UNKNOWN',photo_id=str(iid))


def rows():
    result=[]
    for slot in range(16):
        iid=1000+slot;r=image(iid);r['publisher_image_url']=r.pop('coco_url')
        result.append(dict(slot=slot,split='DEV'if slot<8 else'RESERVED',official_split='val'if slot<8 else'test',
            image_id=iid,image=r,reference_file=f'reference_{slot:06d}.json',reference_identity=pin(dict(raw_slot=slot))))
    return result


def jpeg(iid=1,width=16,height=12):
    frame=bytes([8])+height.to_bytes(2,'big')+width.to_bytes(2,'big')+bytes([3])+bytes([1,0x11,0,2,0x11,0,3,0x11,0])
    return b'\xff\xd8\xff\xe1\x00\x0a'+iid.to_bytes(8,'big')+b'\xff\xc0\x00\x11'+frame+b'\xff\xd9'


def request(url,*_):return jpeg(int(url.rsplit('_',1)[1][:-4]))
def private(tmp_path,name='out'):
    path=tmp_path/name;path.mkdir(mode=0o700);path.chmod(0o700);return path


def proof():return dict(source_binding=dict(producer_revision='a'*40),census_binding=dict(authenticated=True),
    census_report_identity=pin(dict(native=True)))


def frozen_fixture(tmp_path):
    metadata=private(tmp_path);expected={};records=rows();q=proof()
    for r in records:r['reference_identity']=p.write(metadata/r['reference_file'],dict(raw_slot=r['slot']),expected)
    cohort=dict(schema='world_reward.vcoco_role_pilot_cohort.v1',producer_revision='a'*40,source_binding=q['source_binding'],
        census_report_identity=q['census_report_identity'],namespace=p.NAMESPACE,retry_count=0,replacement_count=0,records=records)
    cp=p.write(metadata/'cohort.json',cohort,expected)
    report=dict(status='pass',phase='select',source_binding=q['source_binding'],cohort_identity=cp,
        source_and_inputs_rehashed_after=True,outputs_sealed=True,artifact_identities=dict(expected))
    rp=p.write(metadata/'report.json',report,expected);metadata.chmod(0o500)
    return metadata,rp,cp,q,expected


def test_frozen_config_exact_helpers_and_public_scope():
    root=Path(__file__).resolve().parents[1];cfg=json.loads((root/p.CONFIG).read_bytes())
    assert p.c.encode(cfg['select'])==p.c.encode(p.SELECT) and p.c.encode(cfg['acquire'])==p.c.encode(p.ACQUIRE)
    assert set(cfg['helper_pins'])==set(p.HELPERS[3:])
    for n,wanted in cfg['helper_pins'].items():assert p.c.pin((root/n).read_bytes())==wanted
    assert cfg['select']['network_allowed']is False and cfg['acquire']['FIT_allowed']is False
    assert p.acq.fetch.__module__=='openimages_joint_pair_acquire' and p.HELPERS[3]=='infra/vcoco_role_census.py'
    assert 'visual_genome'not in Path(p.__file__).read_text() and 'lifecycle'not in Path(p.__file__).read_text()


def test_project_one_image_preserves_action_and_role_major_slots():
    a=dict(action_name='all_general_roles',role_name=['agent','obj','instr'],ann_id=[11,22,33],image_id=[1,2,1],
        label=[1,0,1],role_object_id=[11,22,33,101,202,303,0,0,404],
        projection_source=dict(action_slot=7,row_slots=[6,8,11],row_count=20))
    b=deepcopy(a);out=p.projected_image([a],1)[0]
    assert a==b and out['ann_id']==[11,33] and out['role_object_id']==[11,33,101,303,0,404]
    assert out['projection_source']==dict(action_slot=7,row_slots=[6,11],row_count=20)
    assert p.projected_image([a],4)[0]['role_object_id']==[]


def test_hash_only_frozen_reference_boundary(tmp_path,monkeypatch):
    metadata,rp,cp,q,_=frozen_fixture(tmp_path);opened=[];original=p.rt.strict
    def strict(raw):
        assert b'raw_slot'not in raw;opened.append(raw);return original(raw)
    monkeypatch.setattr(p.rt,'strict',strict)
    result,state=p.frozen(metadata,rp,cp,q,lambda:None)
    assert [r['slot']for r in result]==list(range(16)) and len(state)==19 and len(opened)==2


@pytest.mark.parametrize('fault',['extra','writable','changed_reference','missing_reference','wrong_grid','filename','grant','photo_alias','split'])
def test_frozen16_no_extra_or_repaired_metadata(tmp_path,fault):
    metadata,rp,cp,q,expected=frozen_fixture(tmp_path);metadata.chmod(0o700)
    if fault=='extra':(metadata/'foreign').write_bytes(b'not owned')
    elif fault=='writable':(metadata/'reference_000000.json').chmod(0o600)
    elif fault=='changed_reference':
        r=metadata/'reference_000000.json';r.chmod(0o600);r.write_bytes(b'changed');r.chmod(0o400)
    elif fault=='missing_reference':(metadata/'reference_000000.json').unlink()
    else:
        cohort=p.rt.strict((metadata/'cohort.json').read_bytes());r=cohort['records'][0]
        if fault=='wrong_grid':r['image']['width']=True
        elif fault=='filename':r['image']['file_name']='../foreign.jpg'
        elif fault=='grant':r['image']['license_url']='https://unverified'
        elif fault=='photo_alias':r['image']['photo_id']=cohort['records'][1]['image']['photo_id']
        else:r['official_split']='test'
        file=metadata/'cohort.json';file.unlink();cp=p.write(file,cohort,{})
        report=p.rt.strict((metadata/'report.json').read_bytes());report['cohort_identity']=cp;report['artifact_identities']['cohort.json']=cp
        (metadata/'report.json').unlink();rp=p.write(metadata/'report.json',report,{})
    metadata.chmod(0o500)
    with pytest.raises((ValueError,FileNotFoundError)):p.frozen(metadata,rp,cp,q,lambda:None)


def test_acquire16_original_slots_sixkeys_no_labels_or_resize(tmp_path):
    public=private(tmp_path);ledger={};calls=[]
    def fetch(url,maximum,deadline,timeout):calls.append((url,maximum,timeout));return request(url)
    before=rows();original=deepcopy(before);out=p.acquire(before,dict(md5=set()),public,time.monotonic()+20,ledger,request=fetch)
    assert before==original and out['availability_gate_passed'] and out['counts']==dict(slots=16,acquired=16,missing=0,acquired_DEV=8,acquired_RESERVED=8)
    assert len(calls)==16 and all(u.startswith(p.BUCKET+'val2014/COCO_val2014_')and m==16<<20 and t==15 for u,m,t in calls)
    manifest=p.rt.strict((public/'manifest.json').read_bytes());assert len(manifest['images'])==16
    assert all(set(r)==p.PUBLIC_KEYS for r in manifest['images']) and len(ledger)==17
    for row in out['public_mappings']:
        r=next(v for v in manifest['images']if v['image_id']==row['public_image_id'])
        assert r['file']==f"image_{row['slot']:06d}.jpg" and (public/r['file']).read_bytes()==jpeg(1000+row['slot'])


def test_missing_fixed_slots_no_retry_holes_and_no_raw_error(tmp_path):
    public=private(tmp_path);ledger={};calls=Counter()
    def fetch(url,*_):
        iid=int(url.rsplit('_',1)[1][:-4]);calls[iid]+=1
        if iid in (1000,1009):raise ValueError('SECRET_TOKEN=DO_NOT_PUBLISH')
        return jpeg(iid)
    out=p.acquire(rows(),dict(md5=set()),public,time.monotonic()+20,ledger,request=fetch)
    assert out['availability_gate_passed'] and out['counts']['missing']==2 and all(c==1 for c in calls.values())
    assert [r['slot']for r in out['records']]==list(range(16))
    assert not(public/'image_000000.jpg').exists()and not(public/'image_000009.jpg').exists()
    assert 'SECRET'not in p.c.encode(out).decode() and out['records'][9]['error_type']=='ValueError'


@pytest.mark.parametrize('fault',['empty','wrong_grid','oversize','historical_md5','duplicate','malformed','url','license'])
def test_acquisition_failure_keeps16_denominator_and_no_replacement(tmp_path,fault):
    public=private(tmp_path);ledger={};selected=rows();excluded=dict(md5=set());calls=[]
    if fault=='historical_md5':excluded['md5'].add(base64.b64encode(hashlib.md5(jpeg(1)).digest()).decode())
    if fault=='url':
        for r in selected:r['image']['publisher_image_url']='https://foreign'
    if fault=='license':
        for r in selected:r['image']['license']=2
    def fetch(url,*_):
        calls.append(url)
        if fault=='empty':return b''
        if fault=='wrong_grid':return jpeg(1,width=17)
        if fault=='oversize':return b'0'*(p.ACQUIRE['max_image_bytes']+1)
        if fault=='malformed':return b'notJPEG'
        return jpeg(1)
    out=p.acquire(selected,excluded,public,time.monotonic()+20,ledger,request=fetch)
    assert not out['availability_gate_passed'] and out['counts']==dict(slots=16,acquired=0,missing=16,acquired_DEV=0,acquired_RESERVED=0)
    assert len(out['records'])==16 and ledger=={'manifest.json':out['public_inputs_identity']}
    assert len(calls)==(0 if fault in('url','license')else 16)
    if fault=='duplicate':assert all(r['reason']=='duplicate_cohort_bytes'for r in out['records'])


def test_duplicate_group_rejects_all_copies_not_first_slot(tmp_path):
    public=private(tmp_path);ledger={}
    def fetch(url,*_):
        iid=int(url.rsplit('_',1)[1][:-4]);return jpeg(5 if iid in (1000,1015)else iid)
    out=p.acquire(rows(),dict(md5=set()),public,time.monotonic()+20,ledger,request=fetch)
    assert out['counts']['acquired']==14 and out['availability_gate_passed']
    assert out['records'][0]['reason']==out['records'][15]['reason']=='duplicate_cohort_bytes'
    assert 'image_000000.jpg'not in ledger and 'image_000015.jpg'not in ledger


def test_owned_partial_write_fsync_failure_cleans_only_owned_inode(tmp_path,monkeypatch):
    out=private(tmp_path);foreign=out/'foreign';foreign.write_bytes(b'retain');ledger={}
    monkeypatch.setattr(p.os,'fsync',lambda _:(_ for _ in()).throw(OSError('SECRET')))
    with pytest.raises(OSError):p.raw_write(out/'image_000000.jpg',jpeg(),ledger)
    assert ledger=={} and list(out.iterdir())==[foreign] and foreign.read_bytes()==b'retain'


def test_existing_foreign_filename_not_deleted_or_mutated(tmp_path):
    out=private(tmp_path);file=out/'image_000000.jpg';file.write_bytes(b'foreign');before=p.c.state(file)
    with pytest.raises(FileExistsError):p.raw_write(file,jpeg(),{})
    assert p.c.state(file)==before


@pytest.mark.parametrize('late',[False,True])
def test_same_fd_publisher_pass_or_late_fail_with_sealed_artifacts(tmp_path,late):
    out=private(tmp_path);owned=out.lstat();ledger={};p.write(out/'cohort.json',dict(authored=True),ledger)
    before=p.c.state(out/'cohort.json');report=dict(status='pass',outputs_sealed=False)
    now=time.monotonic();p.publish(out,report,now-1 if late else now+20,now,owned,ledger)
    saved=p.rt.strict((out/'report.json').read_bytes())
    assert saved==report and p.c.state(out/'cohort.json')==before and out.stat().st_mode&0o777==0o500
    assert (out/'report.json').stat().st_mode&0o777==0o400
    assert saved['status']==('fail'if late else'pass')
    if late:assert saved['publication_failed']


@pytest.mark.parametrize('fault',['foreign','directory_mode','file_mode','changed','hardlink'])
def test_publisher_refuses_unowned_or_changed_inventory(tmp_path,fault):
    out=private(tmp_path);owned=out.lstat();ledger={};p.write(out/'cohort.json',dict(authored=True),ledger)
    if fault=='foreign':(out/'foreign').write_bytes(b'foreign')
    elif fault=='directory_mode':out.chmod(0o755)
    elif fault=='file_mode':(out/'cohort.json').chmod(0o600)
    elif fault=='hardlink':os.link(out/'cohort.json',tmp_path/'alias')
    else:
        file=out/'cohort.json';file.chmod(0o600);file.write_bytes(b'changed');file.chmod(0o400)
    with pytest.raises(ValueError):p.publish(out,dict(status='pass'),time.monotonic()+20,time.monotonic(),owned,ledger)
    assert not(out/'report.json').exists()


def population(tmp_path):
    cfg=dict(inputs={},archive=dict(instances_members={},member_catalogue=[]),roles={},splits={},max_expanded_bytes=512<<20,
        max_row_bytes=1<<20,max_role_bytes=16<<20,max_field_bytes=2<<20)
    expected=[];values={};expansion=[]
    for official,partition,start,count in (('val','train2014',1000,199),('test','val2014',2000,362)):
        ids=list(range(start,start+count));imgs=[image(iid,partition)for iid in ids]
        ann=[dict(id=iid*10+j,image_id=iid,category_id=1 if j<3 else 17,iscrowd=0,area=4,bbox=[j*3,0,2,2])for iid in ids for j in range(1,5)]
        values[partition]=dict(images=imgs,licenses=[dict(id=4,url=p.c.GRANT)],
            categories=[dict(id=1,name='person')]+[dict(id=j,name=f'category_{j}')for j in range(2,81)],annotations=ann)
        action=dict(action_name='general_action',role_name=['agent','obj'],ann_id=[iid*10+1 for iid in ids],image_id=ids,
            label=[1]*count,role_object_id=[iid*10+1 for iid in ids]+[iid*10+3 for iid in ids])
        key='role_'+official;file=tmp_path/key;file.write_bytes(p.c.encode([action]));file.chmod(0o400)
        cfg['inputs'][key]=dict(path=str(file),pin=p.c.pin(file.read_bytes()));cfg['roles'][official]=key
        expected.extend(dict(split=official,image_id=iid,photo_id=str(iid))for iid in ids)
    # Same source role dictionary order as original canonical config, TEST before VAL.
    cfg['roles']={k:cfg['roles'][k]for k in sorted(cfg['roles'])};expected.sort(key=lambda r:(r['split'],r['image_id']))
    values['train2014']['images'].append(image(999,'train2014'))
    values['train2014']['annotations'].append(dict(image_id=999,marker='OLD_POISON',bbox=['NEVER_CONSULT']))
    archive=tmp_path/'archive.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED)as z:
        for partition,value in values.items():
            name=f'annotations/instances_{partition}.json';content=p.c.encode(value);z.writestr(name,content)
            cfg['archive']['instances_members'][partition]=name
            expansion.append(dict(member=name,expanded_bytes=len(content),expanded_sha256=p.c.pin(content)['sha256']))
    with zipfile.ZipFile(archive)as z:
        cfg['archive']['member_catalogue']=[dict(member=i.filename,compressed_bytes=i.compress_size,expanded_bytes=i.file_size,
            crc32=f'{i.CRC:08x}',external_attr=i.external_attr,compression=i.compress_type,flags=i.flag_bits)for i in z.infolist()]
    cfg['archive']['catalogue_tail_identity']=p.c.pin(archive.read_bytes()[-64:]);archive.chmod(0o400)
    cfg['inputs']['coco_archive']=dict(path=str(archive),pin=p.c.pin(archive.read_bytes()))
    native=tmp_path/'metadata_report';native.write_bytes(p.c.encode(dict(archive_members=expansion)));native.chmod(0o400)
    cfg['inputs']['metadata_report']=dict(path=str(native),pin=p.c.pin(native.read_bytes()))
    for name,ids in dict(train=[999],val=list(range(1000,1199)),test=list(range(2000,2362)),
        trainval=[999]+list(range(1000,1199)),all=[999]+list(range(1000,1199))+list(range(2000,2362))).items():
        file=tmp_path/('split_'+name);file.write_text(''.join(f'{i}\n'for i in ids));file.chmod(0o400)
        cfg['inputs']['split_'+name]=dict(path=str(file),pin=p.c.pin(file.read_bytes()));cfg['splits'][name]='split_'+name
    return cfg,dict(eligibility_inventory_identity=pin(expected))


def test_same561_population_before_rank_and_fresh16_reference_freeze(tmp_path,monkeypatch):
    cfg,expected=population(tmp_path);decoded=[];original=p.js.strict_decode
    def decode(raw):decoded.append(raw);return original(raw)
    monkeypatch.setattr(p.js,'strict_decode',decode)
    chosen,banks,images,instances,actions=p.reproduce(cfg,dict(photos={'999'}),expected,lambda:None)
    assert len(chosen)==16 and [r['study_split']for r in chosen]==['DEV']*8+['RESERVED']*8
    for split in ('val','test'):
        ids=list(range(1000,1199)if split=='val'else range(2000,2362))
        ranked=sorted(ids,key=lambda iid:(hashlib.sha256((p.NAMESPACE+f'{iid:012d}').encode()).hexdigest(),iid))[:8]
        assert [r['image_id']for r in chosen if r['split']==split]==ranked
    assert not any(b'OLD_POISON'in raw or b'NEVER_CONSULT'in raw for raw in decoded)
    metadata=private(tmp_path,'selected');ledger={};cohort,cp=p.select_artifacts(metadata,chosen,banks,images,instances,actions,proof(),lambda:None,ledger)
    assert len(ledger)==17 and 'localized_positive_pairs'not in p.c.encode(cohort).decode()
    assert all('flickr_url'in r['image']and r['image']['creator_identity']=='UNKNOWN'for r in cohort['records'])
    assert cp==ledger['cohort.json'] and all(p.rt.strict((metadata/r['reference_file']).read_bytes())['localized_positive_pairs']for r in cohort['records'])


@pytest.mark.parametrize('fault',['inventory','catalogue','tail','role_changed'])
def test_reproduced_inventory_or_original_source_mismatch_cannot_freeze(tmp_path,fault):
    cfg,expected=population(tmp_path)
    if fault=='inventory':expected['eligibility_inventory_identity']['sha256']='0'*64
    elif fault=='catalogue':cfg['archive']['member_catalogue'][0]['flags']=1
    elif fault=='tail':cfg['archive']['catalogue_tail_identity']['sha256']='0'*64
    else:cfg['inputs']['role_test']['pin']['sha256']='0'*64
    with pytest.raises(ValueError):p.reproduce(cfg,dict(photos={'999'}),expected,lambda:None)


def fake_host(tmp_path,monkeypatch,*,fail=False,postfail=False):
    code=private(tmp_path,'code');(code/'infra').mkdir();driver=code/p.HELPERS[0];driver.write_bytes(b'authored fixture');driver.chmod(0o400)
    monkeypatch.setattr(p,'__file__',str(driver));monkeypatch.setenv('WR_CODE',str(code));monkeypatch.setenv('WR_CODE_REVISION','a'*40)
    monkeypatch.setattr(p.os,'geteuid',lambda:0);monkeypatch.setattr(p.sys,'platform','linux')
    def namespace(path):
        # Exact production policy is separately tested; local fixtures belong to the user.
        s=p.rt.canonical(path).lstat();assert s.st_mode&0o777==0o700
        return [s.st_dev,s.st_ino,s.st_mode,s.st_uid,s.st_gid]
    monkeypatch.setattr(p,'namespace_identity',namespace)
    monkeypatch.setattr(p.os,'uname',lambda:type('Host',(),dict(nodename='world-reward-ncc-h100-02'))())
    monkeypatch.setattr(p,'DATA',tmp_path/'fresh');calls=[];q=proof();result=({}, {}, {},dict(md5=set()),q)
    def authenticate(*_):
        calls.append('auth')
        if postfail and len(calls)>1:raise ValueError('PRIVATE_SECRET')
        return result
    monkeypatch.setattr(p,'authenticate',authenticate)
    def reproduce(*_):
        if fail:raise ValueError('PRIVATE_SECRET')
        return tuple({}for _ in range(5))
    monkeypatch.setattr(p,'reproduce',reproduce)
    def select(metadata,*args):
        ledger=args[-1]
        for i in range(16):p.write(metadata/f'reference_{i:06d}.json',dict(authored=i),ledger)
        cp=p.write(metadata/'cohort.json',dict(authored=True),ledger);return {},cp
    monkeypatch.setattr(p,'select_artifacts',select)
    # No native records/annotations or real source are touched by these control fixtures.
    result[2]['eligibility_inventory_identity']=pin(dict(authored=True))
    return calls


@pytest.mark.parametrize('fail,postfail',[(False,False),(True,False),(False,True)])
def test_fake_host_postauth_even_failure_and_no_raw_error(tmp_path,monkeypatch,capsys,fail,postfail):
    calls=fake_host(tmp_path,monkeypatch,fail=fail,postfail=postfail)
    report=p.run('select',[]);saved=p.rt.strict((p.DATA/'metadata'/'report.json').read_bytes())
    assert saved==report and calls==['auth','auth'] and 'PRIVATE_SECRET'not in p.c.encode(saved).decode()
    assert saved['status']==('pass'if not fail and not postfail else'fail')
    assert saved['source_and_inputs_rehashed_after']is(not postfail)
    assert (p.DATA/'metadata').stat().st_mode&0o777==0o500 and 'PRIVATE_SECRET'not in capsys.readouterr().out


def test_invalid_phase_fails_before_any_owned_output():
    with pytest.raises(ValueError):p.run('retry',[])


@pytest.mark.parametrize('fault',[None,'worker','source','public_seal'])
def test_fake_acquire_run_seals_before_postauth_and_never_opens_reference_values(tmp_path,monkeypatch,capsys,fault):
    calls=fake_host(tmp_path,monkeypatch);p.DATA.mkdir(mode=0o700)
    metadata,rp,cp,q,_=frozen_fixture(p.DATA);metadata.rename(p.DATA/'metadata');q['census_binding']=dict(authenticated=True)
    # Fresh isolated fixture source and metadata, no real repo closure or data.
    inv=p.inventory
    def audit(path,*args):
        if path.name=='inputs'and args[2]==0o700 and fault=='public_seal':raise OSError('PRIVATE_SECRET')
        return inv(path,*args)
    monkeypatch.setattr(p,'inventory',audit)
    def authenticate(*_):
        calls.append('auth')
        if len(calls)>1:
            if fault!='public_seal':assert (p.DATA/'inputs').stat().st_mode&0o777==0o500
            if fault=='source':raise ValueError('PRIVATE_SECRET')
        return {},{},{},dict(md5=set()),q
    monkeypatch.setattr(p,'authenticate',authenticate)
    original_acquire=p.acquire
    def acquire(*args):
        if fault=='worker':raise TimeoutError('PRIVATE_SECRET')
        return original_acquire(*args,request=request)
    monkeypatch.setattr(p,'acquire',acquire)
    original_strict=p.rt.strict
    def strict(raw):
        assert b'raw_slot'not in raw
        return original_strict(raw)
    monkeypatch.setattr(p.rt,'strict',strict)
    result=p.run('acquire',[str(rp['bytes']),rp['sha256'],str(cp['bytes']),cp['sha256']])
    assert len(calls)==2 and result['selected_slots']==16 and len(result['records'])==16
    assert result['status']==('pass'if fault is None else'fail')
    assert 'PRIVATE_SECRET'not in p.c.encode(result).decode() and 'PRIVATE_SECRET'not in capsys.readouterr().out
    assert (p.DATA/'eval_private').stat().st_mode&0o777==0o500
    if fault is None:assert result['availability_gate_passed'] and result['source_and_inputs_rehashed_after']


@pytest.mark.parametrize('fault',[None,'mode','uid','gid','notdir','symlink'])
def test_root_private_namespace_identity_rejects_wrong_owner_or_mode(tmp_path,monkeypatch,fault):
    from types import SimpleNamespace
    path=private(tmp_path);before=path.lstat()
    fake=SimpleNamespace(st_dev=before.st_dev,st_ino=before.st_ino,st_mode=before.st_mode,st_uid=0,st_gid=0)
    if fault=='mode':fake.st_mode=(fake.st_mode&~0o777)|0o755
    elif fault=='uid':fake.st_uid=501
    elif fault=='gid':fake.st_gid=20
    elif fault=='notdir':fake.st_mode=0o100700
    elif fault=='symlink':
        link=tmp_path/'link';link.symlink_to(path,target_is_directory=True)
        with pytest.raises(ValueError):p.namespace_identity(link)
        return
    original=Path.lstat
    monkeypatch.setattr(Path,'lstat',lambda self:fake if self==path else original(self))
    if fault is None:assert p.namespace_identity(path)==[before.st_dev,before.st_ino,before.st_mode,0,0]
    else:
        with pytest.raises(ValueError):p.namespace_identity(path)
