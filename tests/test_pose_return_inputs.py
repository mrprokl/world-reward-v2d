"""Procedural tiny original-output pins/archives, no decoded media or real data."""
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tarfile

import pytest

REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'infra'))
import pose_return_inputs as payload
import pose_return_inventory as inventory
from test_pose_peer_inputs import cohort
from test_pose_peer_run import dispatch,seal,write as original_write


def write(path,raw,mode=0o444):
    if path.exists():path.chmod(0o644)
    return original_write(path,raw,mode)


@pytest.fixture
def result_fixture(cohort,tmp_path,monkeypatch):
    root=cohort['root'];index=8;revision='c'*40
    producer_helpers=('infra/pose_peer_run.py','infra/run_pose_peer_run.sh','infra/pose_peer_inputs.py')
    old=dispatch(root,revision,'run_pose_peer_run',producer_helpers)
    trackers=cohort['source']['tracker_helpers']
    for name in trackers:write(old/name,(cohort['code']/name).read_bytes())
    incoming=dict(schema='world_reward.pose_peer.pins.v1',episode_index=8,clip_spec=cohort['manifest']['clip_spec'],mesh_source='default',
        tracker_helpers=trackers,source_pins=cohort['manifest']['source_pins_identity'],manifest=dict(bytes=3,sha256='a'*64),archive=dict(bytes=9,sha256='a'*64),
        inventory_report=dict(bytes=9,sha256='a'*64,producer_revision='b'*40,script_sha256='c'*64))
    inputid=write(old/'configs/pose_peer_000008_pins.json',payload.raw_json(incoming));seal(old)
    disk=tmp_path/'disk';disk.mkdir();monkeypatch.setattr(payload.original,'DEST',disk)
    received=dict(schema='world_reward.pose_peer.received_pins.v1',episode_index=8,producer_revision='d'*40,
        receipt=write(disk/('pose-peer-000008-'+('d'*40))/'pose-receipt.json',payload.raw_json(dict(stage='pose_peer_receive',status='pass',episode_index=8,
            producer_revision='d'*40,archive=incoming['archive'],manifest=incoming['manifest'],original_frame_coverage_verified=True,extraction_performed=True))))
    folder=root/'outputs/episode_000008/object_pose_full'
    outputs={name:write(folder/name,b'original tiny bytes '+name.encode())for name in payload.FILES if name!='report.json'}
    native=dict(stage='fixed_scale_full_object_pose_initializer',status='pass',episode_index=8,input_track='track_1',ground_truth_used=False,
        hand_labeled_test=False,oracle_modes=[],fixed_shape=True,original_frame_coverage_verified=True,
        script_sha256=trackers['infra/object_pose_smoke.py']['sha256'],geometry_and_poses_sha256=outputs['geometry_and_poses.npz']['sha256'],
        fixed_canonical_mesh_sha256=outputs['object_fixed_canonical.glb']['sha256'],input_sha256=incoming['clip_spec']['video_sha256'],
        frames=[dict(frame_index=i)for i in range(3)])
    outputs['report.json']=write(folder/'report.json',payload.raw_json(native))
    helpers={name:payload.identity(old/name,True)for name in producer_helpers}
    parent=dict(stage='pose_peer_run',status='pass',episode_index=8,producer_revision=revision,script_sha256=helpers['infra/pose_peer_run.py']['sha256'],
        frames=3,full_original_indices=True,prediction_algorithm_changed=False,model_weights_read=False,ground_truth_used=False,quality_verified=False,
        source_rehashed_after=True,source_helpers=helpers,tracker_helpers=trackers,pins=inputid,received_receipt=received['receipt'],output_files=outputs)
    parentpath=f'results/pose-peer-run-000008-{revision}/report.json';parentid=write(root/parentpath,payload.raw_json(parent))
    source=dict(schema='world_reward.pose_return.source_pins.v1',episode_index=8,frames=3,full_original_indices=True,ground_truth_used=False,
        input_pins=inputid,outputs=outputs,prediction_algorithm_changed=False,producer_entrypoint='run_pose_peer_run',producer_helpers=helpers,
        producer_report=dict(path=parentpath,**parentid),producer_revision=revision,producer_script_sha256=helpers['infra/pose_peer_run.py']['sha256'],
        quality_verified=False,received_full_input_files_independently_rehashed=True,received_receipt=received,source_independently_rehashed=True,
        source_vm='world-reward-ncc-h100-02',tracker_helpers=trackers)
    currentrev='e'*40;code=dispatch(root,currentrev,'run_pose_return_inventory',inventory.HELPERS)
    sourceid=write(code/'configs/pose_return_000008_source_pins.json',payload.raw_json(source));seal(code)
    # Real production keeps UID1000; procedural tests emulate metadata only.
    original_stat=Path.stat
    def stat_uid(path,*args,**kwargs):
        s=original_stat(path,*args,**kwargs)
        if 'object_pose_full'in path.parts:
            fields=list(s);fields[4]=fields[5]=1000;return os.stat_result(fields)
        return s
    monkeypatch.setattr(Path,'stat',stat_uid);monkeypatch.setattr(payload,'ROOT',root)
    return dict(root=root,code=code,revision=currentrev,old=old,source=source,sourceid=sourceid,folder=folder,native=native,parent=parent)


def archive_fixture(value,tmp_path):
    source=value['source'];old=payload.provenance(value['root'],source);path=tmp_path/'archive.tar'
    raw,archive=payload.archive(value['root'],source,value['sourceid'],old,path)
    pins=dict(schema='world_reward.pose_return.pins.v1',episode_index=8,source_pins=value['sourceid'],manifest=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()),archive=archive,
        inventory_report=dict(path='results/pose-return-inventory-000008-'+value['revision']+'/report.json',bytes=1,sha256='a'*64,producer_revision=value['revision'],script_sha256='b'*64))
    return path,pins,payload.inspect_archive(path,pins,source)


def test_actual_frozen_source_pin_schema_and_no_first_seen_identity():
    actual=payload.strict((REPO/'configs/pose_return_000008_source_pins.json').read_bytes())
    assert payload.source_pins(actual,8)==actual and actual['frames']==634 and set(actual['outputs'])==set(payload.FILES)


def test_original_complete_provenance_and_deterministic_archive(result_fixture,tmp_path):
    value=result_fixture;source=value['source'];before=payload.provenance(value['root'],source)
    path,pins,manifest=archive_fixture(value,tmp_path)
    second=tmp_path/'second.tar';raw,archive=payload.archive(value['root'],source,value['sourceid'],before,second)
    assert archive==pins['archive']and raw==payload.raw_json(manifest)
    out=tmp_path/'fresh';payload.extract(path,manifest,out)
    assert set(p.name for p in out.iterdir())==set(payload.FILES)|{payload.PARENT}
    assert all(payload.identity(out/n,True)==row for n,row in manifest['files'].items())
    with pytest.raises(ValueError):payload.extract(path,manifest,out)
    with pytest.raises(FileExistsError):payload.archive(value['root'],source,value['sourceid'],before,path)


@pytest.mark.parametrize('fault',['GT','frame','static_claim','extra_file','parent_sha','npz_sha','producer_helper','input_pin','receiver','native_oracle'])
def test_original_failure_or_changed_bytes_cannot_be_returned(result_fixture,fault):
    value=result_fixture;source=copy.deepcopy(value['source'])
    if fault=='GT':source['ground_truth_used']=True
    elif fault=='frame':
        native=copy.deepcopy(value['native']);native['frames'][2]['frame_index']=1
        source['outputs']['report.json']=write(value['folder']/'report.json',payload.raw_json(native))
    elif fault=='native_oracle':
        native=copy.deepcopy(value['native']);native['oracle_modes']=['GT'];source['outputs']['report.json']=write(value['folder']/'report.json',payload.raw_json(native))
    elif fault=='static_claim':source['prediction_algorithm_changed']=True
    elif fault=='extra_file':write(value['folder']/'unknown',b'foreign')
    elif fault=='parent_sha':source['producer_report']['sha256']='f'*64
    elif fault=='npz_sha':write(value['folder']/'geometry_and_poses.npz',b'changed')
    elif fault=='producer_helper':source['producer_helpers']['infra/pose_peer_run.py']['sha256']='f'*64
    elif fault=='input_pin':source['input_pins']['sha256']='f'*64
    else:source['received_receipt']['receipt']['sha256']='f'*64
    with pytest.raises((ValueError,FileNotFoundError)):payload.provenance(value['root'],payload.source_pins(source,8))


@pytest.mark.parametrize('fault',['traversal','link','duplicate','short','extra','manifest','missing'])
def test_untrusted_archive_never_merges_or_accepts_foreign_files(result_fixture,tmp_path,fault):
    _,_,manifest=archive_fixture(result_fixture,tmp_path);path=tmp_path/'bad.tar'
    with tarfile.open(path,'w')as tar:
        raw=b'{}'if fault=='manifest'else payload.raw_json(manifest);member=tarfile.TarInfo(payload.MANIFEST);member.size=len(raw);tar.addfile(member,io.BytesIO(raw))
        for index,(name,row)in enumerate(sorted(manifest['files'].items())):
            if fault=='missing'and index==0:continue
            original=result_fixture['root']/result_fixture['source']['producer_report']['path']if name==payload.PARENT else result_fixture['folder']/name
            data=original.read_bytes();m=tarfile.TarInfo('../escape'if fault=='traversal'and index==0 else'foreign'if fault=='extra'and index==0 else name)
            if fault=='link'and index==0:m.type=tarfile.SYMTYPE;m.linkname='/foreign';tar.addfile(m);continue
            m.size=len(data)-1 if fault=='short'and index==0 else len(data);tar.addfile(m,io.BytesIO(data))
            if fault=='duplicate'and index==0:tar.addfile(m,io.BytesIO(data))
    with pytest.raises(ValueError):payload.extract(path,manifest,tmp_path/'owned')
    assert not(tmp_path/'escape').exists()


def test_manifest_hash_verified_before_json_interpretation(result_fixture,tmp_path,monkeypatch):
    path,pins,_=archive_fixture(result_fixture,tmp_path);pins['manifest']['sha256']='f'*64
    monkeypatch.setattr(payload,'strict',lambda *a:pytest.fail('WrongSHA reachedJSON'))
    with pytest.raises(ValueError):payload.inspect_archive(path,pins,result_fixture['source'])


def test_inventory_seals_only_remote_summary_and_rechecks_original(result_fixture):
    value=result_fixture;report=inventory.inventory(value['root'],value['code'],value['revision'],8)
    assert report['status']=='pass'and report['frames']==3 and report['source_rehashed_after']and report['quality_verified']is False
    out=value['root']/'results'/f'pose-return-inventory-000008-{value["revision"]}'
    assert {p.name for p in out.iterdir()}=={'archive.tar','manifest.json','report.json'}
    assert all(p.stat().st_mode&0o777==0o400 for p in out.iterdir())
