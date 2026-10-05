"""Tiny manufactured publisher/transport controls, no data or accuracy claim."""
from copy import deepcopy
import base64
import hashlib
import json
import os
from pathlib import Path
import time

import pytest

import proposal_external_acquire as p


def jpeg(width=16,height=12):
    frame=bytes([8])+height.to_bytes(2,'big')+width.to_bytes(2,'big')+bytes([3])+bytes([1,0x11,0,2,0x11,0,3,0x11,0])
    return b'\xff\xd8\xff\xe0\x00\x04AB\xff\xc0\x00\x11'+frame+b'\xff\xd9'


def row(slot,raw=None):
    raw=jpeg() if raw is None else raw
    return dict(slot=slot,split='DEV' if slot<16 else 'RESERVED',publisher_metadata=dict(
        ImageID=f'{1000+slot:016x}',OriginalLandingURL=f'https://www.flickr.com/photos/a{slot}/{1000+slot}/',
        OriginalURL=f'https://live.staticflickr.com/10/{slot}_unique.jpg',OriginalSize=str(len(raw)),
        OriginalMD5=base64.b64encode(hashlib.md5(raw).digest()).decode(),Rotation='0.0',Author=f'A{slot}',
        AuthorProfileURL=f'https://www.flickr.com/people/a{slot}/',Title=f'Title{slot}'))


def rights(r,**change):
    m=r['publisher_metadata']; node={'@type':'ImageObject','acquireLicensePage':m['OriginalLandingURL'],
                                    'license':p.acq.LICENSE_URL,'author':{'name':m['Author']}}
    node.update(change)
    return ('<script type="application/ld+json">'+json.dumps(node)+'</script>').encode()


def config():
    return json.loads((Path(__file__).resolve().parents[1]/p.CONFIG).read_bytes())


def slot_bank(tmp_path, *, slots=(0,2), bad=None):
    private=tmp_path/'eval_private';public=tmp_path/'inputs';private.mkdir(mode=0o700);public.mkdir(mode=0o700)
    selected=[row(i) for i in slots];calls=[];cfg=config()
    def request(url,maximum,deadline,timeout):
        calls.append(url)
        r=next(r for r in selected if url in (r['publisher_metadata']['OriginalLandingURL'],r['publisher_metadata']['OriginalURL']))
        if bad is not None and r['slot']==bad and url==r['publisher_metadata']['OriginalLandingURL']:
            raise OSError('SECRET_NO_RAW_ERROR')
        return rights(r) if url==r['publisher_metadata']['OriginalLandingURL'] else jpeg()
    records=[p.acq.acquire_slot(r,private,cfg,time.monotonic()+30,request=request) for r in selected]
    return private,public,records,cfg,calls


def test_original_helpers_not_copied_modified_or_new_http_stack():
    assert p.acq.acquire_cohort.__module__ == 'openimages_joint_pair_acquire'
    src=Path(p.__file__).read_text()
    assert 'acq.acquire_cohort(selected,private,cfg,deadline-15,cohort_path' in src
    assert 'urllib' not in src and 'requests.' not in src and 'PIL' not in src
    cfg=config();root=Path(__file__).resolve().parents[1]
    for name,pin in cfg['reused_helper_pins'].items():assert p.pin((root/name).read_bytes())==pin


def test_original_inodes_and_SHA_move_to_original_slot_filenames(tmp_path):
    private,public,records,cfg,_=slot_bank(tmp_path)
    originals={r['slot']:(private/r['image_id']/'rgb.jpg').stat().st_ino for r in records}
    pp,mapping=p.make_public(records,private,public,cfg,time.monotonic()+30)
    assert [m['public_file'] for m in mapping]==['image_000000.jpg','image_000002.jpg']
    for r,m in zip(records,mapping):
        assert (public/m['public_file']).stat().st_ino==originals[r['slot']]
        assert p.rt.identity(public/m['public_file'])==r['image_pin']
        assert not (private/r['image_id']/'rgb.jpg').exists()
    p.verify_output(records,mapping,private,public,pp,cfg)


def test_missing_original_slot_is_retained_privately_not_fake_public_rgb(tmp_path):
    private,public,records,cfg,calls=slot_bank(tmp_path,bad=0)
    pp,mapping=p.make_public(records,private,public,cfg,time.monotonic()+30)
    assert len(records)==2 and records[0]['status']=='unavailable'
    assert [m['slot'] for m in mapping]==[2] and len(calls)==3
    assert 'SECRET_NO_RAW_ERROR' not in (private/records[0]['image_id']/'record.json').read_text()
    p.verify_output(records,mapping,private,public,pp,cfg)


def test_public_exact_sixkeys_opaque_ids_no_rights_split_metadata(tmp_path):
    private,public,records,cfg,_=slot_bank(tmp_path)
    pp,mapping=p.make_public(records,private,public,cfg,time.monotonic()+30)
    manifest=p.rt.pinned(public/'manifest.json',pp)
    assert set(manifest)=={'schema','images'} and manifest['schema']=='world_reward.rgb_proposal_inputs.v1'
    assert all(set(r)==p.PUBLIC_KEYS for r in manifest['images'])
    text=json.dumps(manifest)
    assert not any(x in text for x in ('Author','Title','flickr','DEV','RESERVED','split','rights','Box','holds'))
    assert all(m['public_image_id']!=r['image_id'] for m,r in zip(mapping,records))


def test_private_rootonly_modes_and_public_input_readonly(tmp_path):
    private,public,records,cfg,_=slot_bank(tmp_path)
    pp,mapping=p.make_public(records,private,public,cfg,time.monotonic()+30)
    assert public.stat().st_mode&0o777==0o500
    assert all(x.stat().st_mode&0o777==0o400 for x in public.iterdir())
    assert all(x.stat().st_mode&0o777==0o700 for x in private.iterdir())
    assert all(x.stat().st_mode&0o777==0o400 for x in private.rglob('*') if x.is_file())
    p.verify_output(records,mapping,private,public,pp,cfg)


@pytest.mark.parametrize('failure',['hash','extra','write_mode','no_rights','record_mutation'])
def test_changed_original_leaf_or_record_fails_not_repaired(tmp_path,failure):
    private,public,records,cfg,_=slot_bank(tmp_path)
    folder=private/records[0]['image_id']
    if failure=='hash':
        f=folder/'rgb.jpg';f.chmod(0o600);f.write_bytes(b'notoriginal');f.chmod(0o444)
    elif failure=='extra':
        folder.chmod(0o700);(folder/'foreign').write_bytes(b'noise')
    elif failure=='record_mutation':records[0]['slot']=31
    elif failure=='no_rights':records[0]['creator_grant_verified']=False
    else:
        # Move succeeds, but later privacy check must reject a writable leaf.
        pp,mapping=p.make_public(records,private,public,cfg,time.monotonic()+30)
        (public/mapping[0]['public_file']).chmod(0o600)
        with pytest.raises(ValueError):p.verify_output(records,mapping,private,public,pp,cfg)
        return
    with pytest.raises(ValueError):p.make_public(records,private,public,cfg,time.monotonic()+30)


def test_all_missing_bank_is_explicit_empty_public_no_dummy_jpeg(tmp_path):
    private,public,records,cfg,_=slot_bank(tmp_path,slots=(0,),bad=0)
    pp,mapping=p.make_public(records,private,public,cfg,time.monotonic()+30)
    assert mapping==[] and p.rt.pinned(public/'manifest.json',pp)['images']==[]
    p.verify_output(records,mapping,private,public,pp,cfg)


def test_public_manifest_private_field_or_missing_row_rejected(tmp_path):
    private,public,records,cfg,_=slot_bank(tmp_path)
    pp,mapping=p.make_public(records,private,public,cfg,time.monotonic()+30)
    value=p.rt.pinned(public/'manifest.json',pp);value['images'][0]['split']='DEV'
    file=public/'manifest.json';file.chmod(0o600);file.write_bytes(p.encode(value));file.chmod(0o400)
    with pytest.raises(ValueError,match='six-key'):p.verify_output(records,mapping,private,public,p.pin(file.read_bytes()),cfg)


def test_original_creator_conditions_and_MD5_are_unchanged(tmp_path):
    r=row(0);cfg=config();private=tmp_path/'private';private.mkdir()
    def request(url,*_):
        return rights(r,license='https://creativecommons.org/licenses/by-nc/4.0/') if 'www.flickr' in url else jpeg()
    result=p.acq.acquire_slot(r,private,cfg,time.monotonic()+30,request=request)
    assert result['status']=='unavailable' and result['failed_phase']=='creator_rights' and 'image_pin' not in result


def test_config_actual_census_and_32slots_sixworkers300s():
    cfg=config()
    assert cfg['slots']==32 and cfg['dev']==cfg['reserved']==16 and cfg['workers']==6 and cfg['overall']==300
    assert cfg['retry_count']==0 and cfg['no_replacements'] is True and cfg['max_image_bytes']==16<<20
    assert cfg['census']['producer_revision']=='d1de716d9d6e94165d14224d17dee36fe2ef2fe5'
    assert cfg['census']['cohort']['pin']==dict(bytes=18076,sha256='4db28db7b8c9beb0ce58e574e8f012f05c86ef7447ff056a269d4b04cd74ed33')
    assert cfg['census']['report']['pin']==dict(bytes=3847,sha256='5c9362852da341159ceefdfe37e4b866a06840ff4d69fbf281424d38b28e3849')


def fake_run(monkeypatch,tmp_path, *, mismatch=False,late=False):
    root=Path(__file__).resolve().parents[1];cfg=config();cfg['output']=str(tmp_path/'output')
    selected=[row(i) for i in range(32)];before=deepcopy(selected)
    source={'helpers':{p.CONFIG:p.pin((root/p.CONFIG).read_bytes())}}
    calls=[];clock=[0.];cohort=tmp_path/'cohort.json';cohort.write_bytes(b'cohort');cohort.chmod(0o444)
    monkeypatch.setattr(p,'DATA',tmp_path/'output');monkeypatch.setenv('WR_CODE',str(root));monkeypatch.setenv('WR_CODE_REVISION','a'*40)
    monkeypatch.setattr(p.sys,'platform','linux');monkeypatch.setattr(p.os,'geteuid',lambda:0)
    monkeypatch.setattr(p.os,'uname',lambda:type('U',(),{'nodename':'world-reward-ncc-h100-02'})())
    monkeypatch.setattr(p.time,'monotonic',lambda:clock[0])
    def auth(*_):
        calls.append('auth')
        return cfg,selected,dict(changed=True) if mismatch and len(calls)>1 else source,{}, {},cohort
    monkeypatch.setattr(p,'authenticate',auth)
    def acquire(rows,private,actualcfg,deadline,cohortpath,cohortpin):
        assert rows==before and actualcfg['workers']==6 and deadline==285. and cohortpath==cohort
        calls.append('acquire')
        def request(url,*_):
            r=next(r for r in rows if url in r['publisher_metadata'].values())
            if r['slot']%2==0:raise OSError('NEVER_PRINT_SECRET')
            return rights(r) if 'www.flickr' in url else jpeg()
        return [p.acq.acquire_slot(r,private,actualcfg,deadline,request=request) for r in rows]
    monkeypatch.setattr(p.acq,'acquire_cohort',acquire)
    monkeypatch.setattr(p,'verify_output',lambda *_:None) # manufactured local owner differs from production root
    if late:
        original=p.os.fsync;syncs=[0]
        def fsync(fd):
            result=original(fd);syncs[0]+=1
            if syncs[0]>150:clock[0]=301.
            return result
        monkeypatch.setattr(p.os,'fsync',fsync)
    return tmp_path/'output',calls,clock


def test_full_driver_all32ledger_acquiredonly_public_pins_no_secret_stdout(monkeypatch,tmp_path,capsys):
    out,calls,_=fake_run(monkeypatch,tmp_path)
    report=p.run();output=capsys.readouterr().out
    assert report['status']=='pass' and report['source_and_inputs_rehashed_after']
    assert report['counts']==dict(slots=32,acquired=16,missing=16,acquired_DEV=8,acquired_RESERVED=8,original_rgb_bytes=16*len(jpeg()))
    assert calls==['auth','acquire','auth'] and len(report['records'])==32
    public=json.loads((out/'inputs/manifest.json').read_bytes())
    assert len(public['images'])==16 and public['images'][0]['file']=='image_000001.jpg'
    assert (out/'eval_private/manifest.json').stat().st_mode&0o777==0o400
    assert all(x.stat().st_mode&0o777==0o500 for x in (out,out/'eval_private',out/'inputs',*(x for x in (out/'eval_private').iterdir() if x.is_dir())))
    assert 'NEVER_PRINT_SECRET' not in output and 'flickr' not in output and 'Title' not in output


def test_changed_authentication_is_closed_fail_not_published_pass(monkeypatch,tmp_path,capsys):
    out,_,_=fake_run(monkeypatch,tmp_path,mismatch=True)
    with pytest.raises(ValueError,match='lineage changed'):p.run()
    output=capsys.readouterr().out
    report=json.loads((out/'eval_private/manifest.json').read_bytes())
    assert report['status']=='fail' and report['source_and_inputs_rehashed_after'] is False and 'lineage changed' not in output


def test_late_finalization_serializes_fail(monkeypatch,tmp_path,capsys):
    out,_,clock=fake_run(monkeypatch,tmp_path)
    original=p.os.fsync;private_manifest=out/'eval_private/manifest.json'
    def fsync(fd):
        result=original(fd)
        if private_manifest.exists():clock[0]=301.
        return result
    monkeypatch.setattr(p.os,'fsync',fsync)
    report=p.run();capsys.readouterr()
    assert report['status']==json.loads(private_manifest.read_bytes())['status']=='fail'
    assert report['error_type']=='TimeoutError'


def test_failed_transport_retains_all32slots_and_private_permissions(monkeypatch,tmp_path,capsys):
    out,_,_=fake_run(monkeypatch,tmp_path)
    def failed(rows,private,*_):
        folder=private/rows[0]['publisher_metadata']['ImageID'];folder.mkdir(mode=0o755)
        p.rt.write(folder/'partial-record.json',b'private metadata',0o444)
        raise OSError('NEVER_PRINT_SECRET')
    monkeypatch.setattr(p.acq,'acquire_cohort',failed)
    with pytest.raises(OSError):p.run()
    text=capsys.readouterr().out;report=json.loads((out/'eval_private/manifest.json').read_bytes())
    assert report['status']=='fail' and len(report['selected_slots'])==32 and report['records']==[]
    assert 'NEVER_PRINT_SECRET' not in text and 'NEVER_PRINT_SECRET' not in (out/'eval_private/manifest.json').read_text()
    assert all(x.stat().st_mode&0o777==0o400 for x in (out/'eval_private').rglob('*') if x.is_file())


def test_authenticate_consumes_real_census_producer_schema_and_all7pins(monkeypatch,tmp_path):
    root=Path(__file__).resolve().parents[1];cfg=config();revision='b'*40
    cfg['census']['report']['path']='census/report.json';cfg['census']['cohort']['path']='census/cohort.json'
    source={'helpers':{p.CONFIG:p.pin((root/p.CONFIG).read_bytes())}}
    old_source={'helpers':cfg['census_helper_pins'],'producer_revision':cfg['census']['producer_revision']}
    oldcfg=json.loads((root/'configs/proposal_external_census_v1.json').read_bytes())
    old_code=tmp_path/'jobs'/cfg['census']['producer_revision']/p.prior.ENTRY/'code'
    rows=[row(i) for i in range(32)]
    cohort=dict(schema='world_reward.proposal_external_cohort.v1',producer_revision=cfg['census']['producer_revision'],
        source_binding=old_source,configuration_identity=cfg['census_helper_pins'][p.prior.CONFIG],
        frozen_input_identities=oldcfg['files'],no_replacements=True,rights_requests_before_freeze=0,
        reference_geometry_exposed=False,hash_namespace=p.prior.HASH_NAMESPACE,records=rows)
    report=dict(schema='world_reward.proposal_external_census.v1',status='pass',producer_revision=cfg['census']['producer_revision'],
        source_binding=old_source,configuration_identity=cfg['census_helper_pins'][p.prior.CONFIG],source_and_inputs_rehashed_after=True,
        cohort_identity=cfg['census']['cohort']['pin'],decision='FROZEN32_PENDING_INDIVIDUAL_RIGHTS_AND_SEPARATE_INFERENCE',
        network_used=False,rgb_read=False,predictions_read=False,models_loaded=False,gpu_used=False,old_studies_reopened=False,
        rights_requests_before_freeze=0,frozen_inputs=oldcfg['files'],counts=dict(excluded_ids=208,unique_author_photo_md5_slots=164))
    values={tmp_path/'census/report.json':report,tmp_path/'census/cohort.json':cohort,old_code/p.prior.CONFIG:oldcfg}
    wanted={tmp_path/'census/report.json':cfg['census']['report']['pin'],tmp_path/'census/cohort.json':cfg['census']['cohort']['pin'],
            **{Path(v['path']):v['pin'] for v in oldcfg['files'].values()}}
    checks=[]
    monkeypatch.setattr(p,'ROOT',tmp_path);monkeypatch.setattr(p,'configuration',lambda *_:cfg)
    monkeypatch.setattr(p.rt,'source',lambda _root,code,*_:source if code==root else old_source)
    def pinned(path,pin,maximum):return deepcopy(values[path])
    def identity(path,maximum,readonly):checks.append((path,readonly));return wanted[path]
    monkeypatch.setattr(p.rt,'pinned',pinned);monkeypatch.setattr(p.rt,'identity',identity)
    result=p.authenticate(root,revision)
    assert result[1]==rows and len(checks)==9 and len(result[4])==9
    assert dict(checks)[Path(oldcfg['files']['old64']['path'])] is True
    for field in ('status','source_and_inputs_rehashed_after','rgb_read'):
        before=report[field];report[field]='fail' if field=='status' else not before
        with pytest.raises(ValueError,match='census PASS'):p.authenticate(root,revision)
        report[field]=before
    cohort['records'][31]['split']='DEV'
    with pytest.raises(ValueError,match='ordered32'):p.authenticate(root,revision)
