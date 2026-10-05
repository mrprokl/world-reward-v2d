"""Procedural metadata and JPEG headers only; no dataset/network/model actions."""
import base64
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import urllib.request

import pytest
import ownership_pair_prepare as p


def metadata(i, *, author=None):
    return dict(ImageID=f'{i:016x}',OriginalLandingURL=f'https://www.flickr.com/photos/a{i}/{10000+i}/',
        OriginalURL=f'https://live.staticflickr.com/42/{10000+i}_abc.jpg',OriginalSize='123',
        OriginalMD5=base64.b64encode(i.to_bytes(16,'big')).decode(),Rotation='0.0',Author=f'Author{i}',
        AuthorProfileURL=f'https://www.flickr.com/photos/{author or "a"+str(i)}/',Title=f'Title{i}')


def records(n=110):
    return [dict(image_id=f'{i:016x}',publisher_metadata=metadata(i),eligible=True) for i in range(1000,1000+n)]


def empty():
    return {k:set() for k in ('ids','authors','photos','md5','urls','coco_ids')}


def config():
    return json.loads((Path(__file__).resolve().parents[1]/p.CONFIG).read_text())


def jpeg(width=4,height=3):
    # Qualified three-channel SOF header, not falsely described as decoded RGB.
    segment=bytes([8])+height.to_bytes(2,'big')+width.to_bytes(2,'big')+bytes([3,1,0x11,0,2,0x11,0,3,0x11,0])
    return b'\xff\xd8\xff\xc0'+(17).to_bytes(2,'big')+segment+b'\xff\xd9'


def acquire_rows():
    rows=[]; raw=jpeg()
    for slot in range(96):
        m=metadata(1000+slot); data=raw+slot.to_bytes(2,'big')
        m.update(OriginalSize=str(len(data)),OriginalMD5=base64.b64encode(hashlib.md5(data).digest()).decode())
        rows.append(dict(slot=slot,split=p.SPLITS[slot],publisher_metadata=m))
    return rows


def rights(row,license='https://creativecommons.org/licenses/by/2.0/'):
    return ('<script type="application/ld+json">'+json.dumps(dict(**{'@type':'ImageObject'},
        acquireLicensePage=row['OriginalLandingURL'],license=license,author={'name':row['Author']}))+'</script>').encode()


def scripted(rows, *, fail_rights=(), bad_md5=(), bad_size=(), bad_header=()):
    calls=[]; lookup={m['OriginalLandingURL']:(r,'rights') for r in rows for m in [r['publisher_metadata']]}
    lookup.update({m['OriginalURL']:(r,'rgb') for r in rows for m in [r['publisher_metadata']]})
    def request(url,maximum,deadline,timeout):
        r,kind=lookup[url]; slot=r['slot']; calls.append((slot,kind))
        assert timeout==15 and maximum<=16<<20
        if kind=='rights':
            return rights(r['publisher_metadata'],license='bad' if slot in fail_rights else p.acq.LICENSE_URL)
        data=jpeg()+slot.to_bytes(2,'big')
        if slot in bad_md5: data=data[:-1]+bytes([data[-1]^1])
        if slot in bad_size: data+=b'x'
        if slot in bad_header: data=b'x'*len(data)
        return data
    return request,calls


def test_fixed_recipe_closure_and_no_network_on_import(monkeypatch):
    def deny(*a,**k): raise AssertionError('No network')
    monkeypatch.setattr(socket,'socket',deny); monkeypatch.setattr(urllib.request,'urlopen',deny)
    cfg=config(); root=Path(__file__).resolve().parents[1]
    assert set(cfg['reused_helper_pins'])==set(p.REUSED)
    assert all(p.pin((root/n).read_bytes())==expected for n,expected in cfg['reused_helper_pins'].items())
    assert cfg['slots']==96 and cfg['minimum_acquired']=={'FIT':24,'CAL':12,'RESERVED':36}
    assert cfg['max_total_image_bytes']==96*(16<<20) and p.SPLITS==['FIT']*32+['CAL']*16+['RESERVED']*48
    assert config()['original_census']['counts']['metadata_inventory_identity']==dict(bytes=22230,sha256='62e9e0879b7b06b70136bd90e9880f761ae0b7923ad33e8a15beabec8bc1092f')


def test_sha_order_exact96_three_splits_metadata_only_no_mutation():
    rows=records(); before=deepcopy(rows)
    selected,counts=p.select_cohort(rows,empty())
    ordered=sorted(rows,key=lambda r:(hashlib.sha256((p.NAMESPACE+r['image_id']).encode()).hexdigest(),r['image_id']))[:96]
    assert [r['publisher_metadata']['ImageID'] for r in selected]==[r['image_id'] for r in ordered]
    assert len(selected)==96 and [r['split'] for r in selected]==p.SPLITS and rows==before
    assert all(set(r)=={'slot','split','publisher_metadata'} and set(r['publisher_metadata'])==set(p.acq.METADATA_KEYS) for r in selected)
    assert counts=={'new_sha_order_independent_slots':96,'eligible_after_extra_md5':110}
    assert p.select_cohort(list(reversed(rows)),empty())==(selected,counts)


@pytest.mark.parametrize('field,key',[('ImageID','ids'),('AuthorProfileURL','authors'),('OriginalLandingURL','photos'),('OriginalMD5','md5'),('OriginalURL','urls')])
def test_every_exclusion_applies_before_selection(field,key):
    rows=records(96); m=rows[0]['publisher_metadata']; identity=p.old._metadata_identity(m)
    excluded=empty(); excluded[key].add(dict(ids=m['ImageID'],authors=identity[0],photos=identity[1],md5=identity[2],urls=identity[3])[key])
    selected,c=p.select_cohort(rows,excluded)
    assert len(selected)==95 and c['new_sha_order_independent_slots']==95
    assert m['ImageID'] not in {r['publisher_metadata']['ImageID'] for r in selected}


@pytest.mark.parametrize('field',['AuthorProfileURL','OriginalLandingURL','OriginalMD5','OriginalURL'])
def test_duplicate_identity_reduces_capacity_not_replaced_or_normalized(field):
    rows=records(96);rows[1]['publisher_metadata'][field]=rows[0]['publisher_metadata'][field]
    assert len(p.select_cohort(rows,empty())[0])==95


def test_extra_coco_md5_can_close_old132_gate():
    rows=records(132); excluded=empty();excluded['md5'].update(r['publisher_metadata']['OriginalMD5'] for r in rows[:37])
    assert len(p.select_cohort(rows,excluded)[0])==95


def test_sha_greedy_gate_not_inferred_from_lexical_capacity():
    # Edges: A-X, A-Y, B-Y. Lexical greedy gets2; alternate SHA order may get1.
    rows=records(98)
    order=sorted(rows[:3],key=lambda r:hashlib.sha256((p.NAMESPACE+r['image_id']).encode()).hexdigest())
    first,middle,last=order
    first['publisher_metadata']['AuthorProfileURL']=middle['publisher_metadata']['AuthorProfileURL']
    first['publisher_metadata']['OriginalMD5']=last['publisher_metadata']['OriginalMD5']
    assert len(p.select_cohort(rows,empty())[0])==96  # fixed bound, not assumed132


@pytest.mark.parametrize('change',[lambda r:r.update(eligible=1),lambda r:r['publisher_metadata'].update(Rotation='90.0'),
                                 lambda r:r['publisher_metadata'].update(OriginalLandingURL='https://flickr.com/x/123/')])
def test_invalid_records_rejected_not_rewritten(change):
    rows=records();change(rows[0]);before=deepcopy(rows)
    with pytest.raises(ValueError):p.select_cohort(rows,empty())
    assert rows==before


def test_duplicate_ids_rejected_and_ineligible_slots_do_not_leak():
    rows=records(96); rows[0]['eligible']=False
    assert len(p.select_cohort(rows,empty())[0])==95
    rows=records(96);rows.append(deepcopy(rows[0]))
    with pytest.raises(ValueError):p.select_cohort(rows,empty())


def coco_fixture(n):
    records=[];expected=[]
    for i in range(n):
        records.append(dict(slot=i,split='DEV' if i<n//2 else 'RESERVED',image_id=i,status='acquired',
            image_pin=dict(bytes=42,sha256='0'*64),jpeg_header=dict(width=4,height=3),
            original_md5=base64.b64encode(i.to_bytes(16,'big')).decode()))
        expected.append(dict(slot=i,split=records[-1]['split'],image_id=i,image=dict(width=4,height=3)))
    return dict(records=records),dict(records=expected)


@pytest.mark.parametrize('n',[32,64])
def test_all_original_coco_metadata_md5_without_reference_or_rgb_reads(n):
    ledger,cohort=coco_fixture(n);before=deepcopy((ledger,cohort))
    assert len(p.coco_md5(ledger,cohort,n))==n and (ledger,cohort)==before


@pytest.mark.parametrize('change',[lambda r:r.pop(),lambda r:r[0].update(status='unavailable'),lambda r:r[0].update(original_md5='bad'),
    lambda r:r[0].update(slot=1),lambda r:r[0].update(image_id=999),lambda r:r[0]['jpeg_header'].update(width=999),
    lambda r:r[1].update(original_md5=r[0]['original_md5']),lambda r:r[0].update(secret='notaccepted')])
def test_coco_acquisition_missing_duplicate_alias_unbound_metadata_fails(change):
    ledger,cohort=coco_fixture(32);change(ledger['records'])
    with pytest.raises((ValueError,KeyError)):p.coco_md5(ledger,cohort,32)


def test_one_original_per_slot_all96_flat_public_private_rights(tmp_path):
    public=tmp_path/'inputs';private=tmp_path/'private';public.mkdir();private.mkdir();rows=acquire_rows();before=deepcopy(rows)
    request,calls=scripted(rows); result=p.acquire(rows,public,private,config(),p.time.monotonic()+30,request=request)
    assert rows==before and len(calls)==192 and len(set(calls))==192
    assert result['counts']==dict(slots=96,acquired=96,missing=0,acquired_FIT=32,acquired_CAL=16,acquired_RESERVED=48)
    assert result['acquisition_capacity_gate_passed'] and result['decision']=='READY_PENDING_SEPARATE_NATIVE_BANK'
    p.verify_outputs(public,private,result,config())
    value=json.loads((public/'manifest.json').read_text());assert set(value)=={'schema','images'}
    assert all(set(r)==p.PUBLIC_KEYS and r['image_id']!=rows[i]['publisher_metadata']['ImageID'] for i,r in enumerate(value['images']))
    assert len(tuple(public.iterdir()))==97 and len(tuple(private.iterdir()))==96
    assert all((public/r['file']).stat().st_nlink==1 for r in value['images'])


def test_rights_failure_requests_no_rgb_retains_allslots_no_retry(tmp_path):
    public=tmp_path/'inputs';private=tmp_path/'private';public.mkdir();private.mkdir();rows=acquire_rows()
    missing=tuple(range(10));request,calls=scripted(rows,fail_rights=missing)
    result=p.acquire(rows,public,private,config(),p.time.monotonic()+30,request=request)
    assert len(result['records'])==96 and all((s,'rgb') not in calls for s in missing)
    assert all(result['records'][s]['failed_phase']=='creator_rights' for s in missing)
    assert result['counts']['acquired_FIT']==22 and not result['acquisition_capacity_gate_passed']
    assert result['decision']=='CLOSED_ACQUISITION_CAPACITY_INCONCLUSIVE'
    assert len(calls)==182 and not any((public/f'image_{s:06d}.jpg').exists() for s in missing)
    p.verify_outputs(public,private,result,config())


@pytest.mark.parametrize('fail_kind',['bad_md5','bad_size','bad_header'])
def test_original_bytes_not_repaired_and_failedjpeg_not_published(tmp_path,fail_kind):
    public=tmp_path/'inputs';private=tmp_path/'private';public.mkdir();private.mkdir();rows=acquire_rows()
    request,calls=scripted(rows,**{fail_kind:(0,)})
    result=p.acquire(rows,public,private,config(),p.time.monotonic()+30,request=request)
    assert result['records'][0]['status']=='unavailable' and not (public/'image_000000.jpg').exists()
    assert calls.count((0,'rgb'))==1 and result['records'][0]['creator_grant_verified']
    assert result['records'][0]['failed_phase']=='original_rgb'
    p.verify_outputs(public,private,result,config())


@pytest.mark.parametrize('split,index,minimum',[('FIT',0,24),('CAL',32,12),('RESERVED',48,36)])
def test_three_independent_availability_gates(tmp_path,split,index,minimum):
    public=tmp_path/'inputs';private=tmp_path/'private';public.mkdir();private.mkdir();rows=acquire_rows()
    n={'FIT':32,'CAL':16,'RESERVED':48}[split]-minimum+1
    request,_=scripted(rows,fail_rights=range(index,index+n))
    result=p.acquire(rows,public,private,config(),p.time.monotonic()+30,request=request)
    assert result['counts']['acquired_'+split]==minimum-1 and not result['acquisition_capacity_gate_passed']


def test_existing_media_or_symlink_not_overwritten(tmp_path):
    public=tmp_path/'inputs';private=tmp_path/'private';public.mkdir();private.mkdir();rows=acquire_rows()
    (public/'image_000000.jpg').write_bytes(b'userwork');request,_=scripted(rows)
    result=p.acquire(rows,public,private,config(),p.time.monotonic()+30,request=request)
    assert result['records'][0]['status']=='unavailable' and (public/'image_000000.jpg').read_bytes()==b'userwork'
    with pytest.raises(ValueError):p.verify_outputs(public,private,result,config())


def test_expired_slots_retained_without_requests_or_substitution(tmp_path):
    public=tmp_path/'inputs';private=tmp_path/'private';public.mkdir();private.mkdir();rows=acquire_rows();calls=[]
    result=p.acquire(rows,public,private,config(),p.time.monotonic()-1,request=lambda *a:calls.append(a))
    assert not calls and result['counts']['missing']==96 and len(result['records'])==96


def test_wrapper_restricted_control_and_closure():
    root=Path(__file__).resolve().parents[1];text=(root/'infra/run_ownership_pair_prepare.sh').read_text()
    assert 'env -i' in text and 'python3 -I -B' in text and '--kill-after=5s' in text
    assert 'LIMIT=190' in text and 'LIMIT=310' in text and 'run_ownership_pair_prepare/code' in text
    assert 'docker' not in text and 'set +x' in text
    assert all('/'+name in text for name in p.HELPERS)
    subprocess.run(['bash','-n',str(root/'infra/run_ownership_pair_prepare.sh')],check=True)
    source=Path(p.__file__).read_text();assert 'old._select_version(' not in source
    assert "['reference_file']" not in source and "['reference_identity']" not in source


def test_publication_failure_removes_only_owned_partial(tmp_path,monkeypatch):
    path=tmp_path/'owned.jpg'; original=p.os.fsync
    def failing(fd):
        if p.stat.S_ISREG(p.os.fstat(fd).st_mode): raise OSError('Manufactured write failure')
        return original(fd)
    monkeypatch.setattr(p.os,'fsync',failing)
    with pytest.raises(OSError):p.publish_owned(path,b'abcdef')
    assert not path.exists()
    path.write_bytes(b'user work')
    with pytest.raises(FileExistsError):p.publish_owned(path,b'new')
    assert path.read_bytes()==b'user work'


def test_write_failure_is_one_unavailable_slot_not_wholebank(tmp_path,monkeypatch):
    public=tmp_path/'inputs';private=tmp_path/'private';public.mkdir();private.mkdir();rows=acquire_rows()
    original=p.publish_owned
    def failing(path,raw):
        if path.name=='image_000000.jpg': raise OSError('No details logged')
        return original(path,raw)
    monkeypatch.setattr(p,'publish_owned',failing); request,calls=scripted(rows)
    result=p.acquire(rows,public,private,config(),p.time.monotonic()+30,request=request)
    assert result['records'][0]['status']=='unavailable' and len(result['records'])==96
    assert result['records'][0]['error_type']=='OSError' and 'No details' not in json.dumps(result)
    p.verify_outputs(public,private,result,config())


def fake_run(monkeypatch,tmp_path,*,phase='select',enough=True,changed=False):
    data=tmp_path/'data';root=tmp_path/'root';root.mkdir();code=root/'code';code.mkdir()
    cfg=config();cfg['output']=str(data);source={'producer_revision':'a'*40,'helpers':{p.CONFIG:{'bytes':1,'sha256':'0'*64}}}
    proof={'bound':True};monkeypatch.setattr(p,'DATA',data);monkeypatch.setattr(p,'ROOT',root)
    monkeypatch.setattr(p.sys,'platform','linux');monkeypatch.setattr(p.os,'geteuid',lambda:0)
    monkeypatch.setattr(p.os,'uname',lambda:type('U',(),{'nodename':'world-reward-ncc-h100-02'})())
    monkeypatch.setenv('WR_CODE_REVISION','a'*40);monkeypatch.setenv('WR_CODE',str(code))
    monkeypatch.setattr(p,'__file__',str(code/p.HELPERS[0]))
    monkeypatch.setattr(p.rt,'source',lambda *a:source);monkeypatch.setattr(p,'source_proof',lambda *a:source)
    monkeypatch.setattr(p,'configuration',lambda *a:(cfg,{}));checks=[]
    def auth(*a):
        checks.append(1);return empty(),set(),{'bound':True if not changed or len(checks)<2 else False}
    monkeypatch.setattr(p,'authenticate',auth)
    def freeze(cfg,source,revision,original,metadata,deadline):
        cp=p.acq.publish(metadata/'cohort.json',{'records':[]}) if enough else None
        return dict(decision='FROZEN96_PENDING_INDIVIDUAL_RIGHTS' if enough else 'CLOSED_CAPACITY_INCONCLUSIVE',
                    capacity_gate_passed=enough,cohort_identity=cp,freeze_before_rgb=enough,input_proof=proof)
    monkeypatch.setattr(p,'freeze_selection',freeze)
    report=p.run('select');return report,data,checks


def test_run_seals_success_and_failure_without_false_capacity_pass(tmp_path,monkeypatch):
    report,data,checks=fake_run(monkeypatch,tmp_path,enough=False)
    assert report['status']=='fail' and report['decision']=='CLOSED_CAPACITY_INCONCLUSIVE'
    assert report['outputs_sealed'] and report['source_and_inputs_rehashed_after']
    assert not (data/'metadata/cohort.json').exists() and not (data/'inputs').exists()
    assert (data/'metadata/report.json').stat().st_mode&0o777==0o400
    assert (data/'metadata').stat().st_mode&0o777==0o500 and data.stat().st_mode&0o777==0o500


def test_run_changed_source_input_postcheck_cannot_publish_pass(tmp_path,monkeypatch):
    report,data,checks=fake_run(monkeypatch,tmp_path,enough=True,changed=True)
    assert report['status']=='fail' and report['source_and_inputs_rehashed_after'] is False
    assert (data/'metadata/cohort.json').exists()  # frozen before failure, not rewritten/replaced
    assert json.loads((data/'metadata/report.json').read_text())['status']=='fail'


def test_fixed_original_recipe_configuration_guards(tmp_path,monkeypatch):
    cfg=config();code=tmp_path/'code';code.mkdir()
    oldcfg={k:deepcopy(cfg[k]) for k in ('files','historical','human_classes','body_part_classes','identity_policy')}
    source={'helpers':{p.CONFIG:p.pin(p.encode(cfg)),**cfg['reused_helper_pins']}}
    source['helpers'][p.OLD_HELPERS[2]]=cfg['original_census']['configuration_identity']
    original=p.rt.pinned
    monkeypatch.setattr(p.rt,'pinned',lambda path,*a:deepcopy(cfg if path.name==Path(p.CONFIG).name else oldcfg))
    for module,name in ((p.old,p.old.V1_HELPERS[0]),(p.rt,p.old.V1_HELPERS[3]),(p.acq,p.old.V1_HELPERS[5]),
                        (p.old.census,p.old.V1_HELPERS[4]),(p.old.coco,p.old.V1_HELPERS[6]),(p.endpoint,p.endpoint.HELPERS[0])):
        monkeypatch.setattr(module,'__file__',str(code/name))
    assert p.configuration(code,source)==(cfg,oldcfg)
    cfg['slots']=95
    with pytest.raises(ValueError,match='Frozen96'):p.configuration(code,source)


def test_directory_fsync_failure_removes_owned_publication(tmp_path,monkeypatch):
    original=p.os.fsync
    def failing(fd):
        if p.stat.S_ISDIR(p.os.fstat(fd).st_mode): raise OSError('Directory fsync failure')
        return original(fd)
    monkeypatch.setattr(p.os,'fsync',failing)
    path=tmp_path/'owned.jpg'
    with pytest.raises(OSError):p.publish_owned(path,b'abcdef')
    assert not path.exists()
