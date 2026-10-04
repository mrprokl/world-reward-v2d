"""Procedural tiny archives/public receipts; no media/model/cloud tests."""
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tarfile

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'infra'))
import pose_peer_inputs as inputs


def digest(raw):return dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def put(root,name,raw):
    path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw);return digest(raw)


@pytest.fixture
def cohort(tmp_path,monkeypatch):
    root=tmp_path/'root';root.mkdir();index=8;count=3;base='outputs/episode_000008';entry='run_track1_initializers_only';rev='a'*40
    producer=root/'jobs'/rev/entry/'code';code=root/'jobs'/('b'*40)/'run_pose_peer_inventory/code'
    helpers={}
    for name in (f'infra/{entry}.sh',*(f'infra/{n}.py'for n in('automatic_masks','body_smoke','depth_smoke','object_smoke','scale_smoke'))):
        helpers[name]=put(producer,name,('producer '+name).encode())
    trackers={name:put(code,name,('tracker '+name).encode())for name in inputs.TRACKER_HELPERS}
    for directory,revision in((producer,rev),(code,'b'*40)):
        put(directory.parent,'revision',(revision+'\n').encode());put(directory.parent,'source-sha256',('c'*64+'\n').encode())
        for path in(directory,*directory.rglob('*')):path.chmod(0o555 if path.is_dir()else 0o444)
    video=put(root,inputs.video_path(index),b'tiny fake encoded video never decoded')
    episodes=put(root,'data/track_1/meta/episodes.jsonl',b'{"episode_index":8,"length":3}\n')
    inputmanifest=dict(track='track_1',repo_id='nvidia/video_to_data_challenge',revision=inputs.DATASET,files=[
        dict(path=inputs.video_path(index).removeprefix('data/'),**video),dict(path='track_1/meta/episodes.jsonl',**episodes)])
    put(root,'results/input-manifest.json',inputs.digest_json(inputmanifest))
    prompts=put(root,base+'/automatic_masks/prompts.json',inputs.digest_json({'prompts':[{'object_id':0},{'object_id':1}]}))
    for kind in(0,1):
        for i in range(count):put(root,f'{base}/automatic_masks/masks/{kind}/{i:06d}.png',('mask'+str(i)).encode())
    depth_frames=[];body_frames=[]
    for i in range(count):
        row=put(root,f'{base}/depth_full/{i:06d}.npz',('fake no decode npz'+str(i)).encode())
        depth_frames.append(dict(frame_index=i,decoded_rgb_sha256='d'*64,output_sha256=row['sha256']));body_frames.append(dict(frame_index=i,decoded_rgb_sha256='d'*64,
            mask_sha256=inputs.identity(root/f'{base}/automatic_masks/masks/0/{i:06d}.png')['sha256']))
    common=dict(status='pass',episode_index=index,input_track='track_1',input_sha256=video['sha256'],ground_truth_used=False,hand_labeled_test=False,oracle_modes=[])
    masks={**common,'stage':'automatic_masks','frames':count,'script_sha256':helpers['infra/automatic_masks.py']['sha256']}
    put(root,base+'/automatic_masks/report.json',inputs.digest_json(masks));masksha=inputs.identity(root/base/'automatic_masks/report.json')['sha256']
    body={**common,'stage':'sam3d_body_full_video_initializer','total_video_frames':count,'frames':body_frames,'mask_report_sha256':masksha,'prompts_sha256':prompts['sha256'],'script_sha256':helpers['infra/body_smoke.py']['sha256']}
    depth={**common,'stage':'monocular_moge2_full_video','total_video_frames':count,'frames':depth_frames,'script_sha256':helpers['infra/depth_smoke.py']['sha256']}
    scale={**common,'stage':'predicted_human_anchored_moge2_pointmaps','depth_alignment':{'shared_scale':1.2},'script_sha256':helpers['infra/scale_smoke.py']['sha256']}
    for name,value in(('body_full',body),('depth_full',depth),('scale_smoke',scale)):put(root,base+'/'+name+'/report.json',inputs.digest_json(value))
    transform=dict(scale=[1.,1.,1.],rotation=[1.,0.,0.,0.],translation=[0.,0.,1.])
    objectfiles={name:put(root,base+'/object_grounded/'+name,raw)for name,raw in(
        ('object.glb',b'tiny fake no decode mesh'),('transform.json',inputs.digest_json(transform)),
        ('intrinsics.json',inputs.digest_json(dict(width=4,height=3,fx=4.,fy=4.,cx=2.,cy=1.5))))}
    obj={**common,'stage':'sam3d_objects_grounded_fixed_frame','scale_source':'already_human_anchored_MoGe2_no_second_scalar',
        'pointmap_grounding':{'alignment_report_sha256':inputs.identity(root/base/'scale_smoke/report.json')['sha256']},'transform':transform,
        'object_sha256':objectfiles['object.glb']['sha256'],'transform_sha256':objectfiles['transform.json']['sha256'],
        'intrinsics_sha256':objectfiles['intrinsics.json']['sha256'],'script_sha256':helpers['infra/object_smoke.py']['sha256']}
    put(root,base+'/object_grounded/report.json',inputs.digest_json(obj))
    kit=put(root,inputs.KIT,b'procedural genuine kit stand-in');monkeypatch.setattr(inputs,'KIT_PIN',kit)
    source=dict(schema='world_reward.pose_peer.source_pins.v1',episode_index=index,producer_revision=rev,producer_entrypoint=entry,
        producer_helpers=helpers,tracker_helpers=trackers,reports={name:inputs.identity(root/name)for name in inputs.reports(index)},mesh_source='default',volume_pins=None)
    spec=inputs.provenance(root,index,source)
    manifest=dict(schema='world_reward.pose_peer.inputs.v1',episode_index=index,clip_spec=spec,mesh_source='default',
        producer=dict(revision=rev,entrypoint=entry,script_sha256=helpers[f'infra/{entry}.sh']['sha256']),
        producer_source_binding=inputs.code_binding(producer,rev,helpers),tracker_helpers=trackers,source_pins=source,
        source_pins_identity=digest(inputs.digest_json(source)),volume_pins=None,
        files={name:inputs.identity(root/name)for name in inputs.paths(index,count,'default')})
    return dict(root=root,code=code,producer=producer,index=index,count=count,source=source,manifest=manifest)


def test_minimal_whitelist_no_body_or_models(cohort):
    manifest=cohort['manifest'];assert inputs.validate_manifest(manifest,8)==manifest
    assert len(manifest['files'])==22 and 'outputs/episode_000008/body_full/report.json'in manifest['files']
    assert not any(word in name for name in manifest['files']for word in('predictions.npz','checkpoint','weights/','.git','multiview','track_2','cari_inputs'))
    assert set(inputs.paths(8,3,'volume'))-set(manifest['files'])=={
        'outputs/episode_000008/object_budget_volume/'+name for name in('report.json','geometry.npz','object_fixed_canonical.glb')}|{'validation/volume_qem_v1/report.json','results/image-volume-qem.json'}


def queued_cohort(cohort):
    """A genuine queued snapshot, not a fabricated initializer child job."""
    manifest=copy.deepcopy(cohort['manifest']);source=manifest['source_pins'];entry='run_track1_frontends_queued'
    parent=cohort['producer'].parent;queued=parent.with_name(entry);parent.rename(queued)
    producer=queued/'code';directory=producer/'infra';directory.chmod(0o755)
    for name in ('infra/run_track1_frontends_queued.sh','infra/run_track1_frontends.sh'):
        source['producer_helpers'][name]=put(producer,name,('original queued child '+name).encode())
        (producer/name).chmod(0o444)
    directory.chmod(0o555)
    source['producer_entrypoint']=entry
    manifest['producer']['entrypoint']=entry
    manifest['producer']['script_sha256']=source['producer_helpers']['infra/'+entry+'.sh']['sha256']
    manifest['source_pins_identity']=digest(inputs.digest_json(source))
    manifest['producer_source_binding']=inputs.code_binding(producer,source['producer_revision'],source['producer_helpers'])
    return producer,manifest


def test_queued_initializers_retain_actual_namespace_and_exact_full_inputs(cohort):
    original=cohort['manifest'];producer,manifest=queued_cohort(cohort)
    assert 'run_track1_frontends_queued' in inputs.PRODUCERS
    assert producer.parent.name=='run_track1_frontends_queued'
    assert not producer.parent.with_name('run_track1_initializers_only').exists()
    assert inputs.validate_manifest(manifest,8) is manifest
    assert inputs.provenance(cohort['root'],8,manifest['source_pins'])==original['clip_spec']
    assert manifest['files']==original['files'] and manifest['tracker_helpers']==original['tracker_helpers']
    assert manifest['source_pins']['reports']==original['source_pins']['reports']


@pytest.mark.parametrize('name',[
    'infra/run_track1_frontends_queued.sh','infra/run_track1_frontends.sh',
    'infra/run_track1_initializers_only.sh'])
@pytest.mark.parametrize('fault',['missing','empty','wrong_sha'])
def test_queued_wrapper_and_both_children_require_independent_nonempty_pins(cohort,name,fault):
    _,manifest=queued_cohort(cohort);source=manifest['source_pins'];producer=manifest['producer']
    if fault=='missing':del source['producer_helpers'][name]
    elif fault=='empty':source['producer_helpers'][name]=digest(b'')
    else:source['producer_helpers'][name]['sha256']='f'*64
    if fault=='wrong_sha' and name!='infra/run_track1_frontends_queued.sh':
        # Pin shape alone cannot authenticate content: the original source
        # binding, used by inventory before report parsing, must reject it.
        with pytest.raises(ValueError):
            inputs.code_binding(cohort['root']/'jobs'/producer['revision']/producer['entrypoint']/'code',
                producer['revision'],source['producer_helpers'])
    else:
        with pytest.raises(ValueError):
            inputs.validate_source_pins(source,8,producer['revision'],producer['entrypoint'],producer['script_sha256'])


@pytest.mark.parametrize('fault',['mutated','writable','symlink','hardlink'])
def test_queued_actual_child_bytes_and_readonly_closure_are_authenticated(cohort,tmp_path,fault):
    producer,manifest=queued_cohort(cohort);child=producer/'infra/run_track1_initializers_only.sh'
    if fault=='mutated':child.chmod(0o644);child.write_bytes(b'changed child');child.chmod(0o444)
    elif fault=='writable':child.chmod(0o644)
    elif fault=='hardlink':os.link(child,tmp_path/'foreign-child-alias')
    else:
        child.parent.chmod(0o755);child.unlink();child.symlink_to('run_track1_frontends.sh');child.parent.chmod(0o555)
    with pytest.raises(ValueError):
        inputs.code_binding(producer,manifest['producer']['revision'],manifest['source_pins']['producer_helpers'])


def test_direct_initializer_producer_does_not_require_queued_children(cohort):
    manifest=cohort['manifest'];assert inputs.validate_manifest(manifest,8) is manifest
    assert 'infra/run_track1_frontends_queued.sh' not in manifest['source_pins']['producer_helpers']


@pytest.mark.parametrize('name',['../escape','/absolute','a/../b','a//b','a\\b','./a',''])
def test_paths_fail_closed(name):
    with pytest.raises(ValueError):inputs.safe_name(name)


@pytest.mark.parametrize('fault',['missing_scale','missing_depth','GT','script','manual','duplicateframe','reportpin'])
def test_not_ready_or_foreign_provenance_fails_before_archive(cohort,fault):
    root=cohort['root'];source=copy.deepcopy(cohort['source']);base='outputs/episode_000008'
    if fault=='missing_scale':(root/base/'scale_smoke/report.json').unlink()
    elif fault=='missing_depth':(root/base/'depth_full/000002.npz').unlink()
    elif fault=='manual':put(root,base+'/automatic_masks/prompts.json',inputs.digest_json({'prompts':[{'object_id':0,'points':[[1,2]]},{'object_id':1}]}))
    elif fault=='reportpin':source['reports'][base+'/body_full/report.json']['sha256']='f'*64
    else:
        name=base+'/body_full/report.json';value=inputs.strict_json((root/name).read_bytes())
        if fault=='GT':value['ground_truth_used']=True
        elif fault=='script':value['script_sha256']='f'*64
        else:value['frames'][1]['frame_index']=0
        source['reports'][name]=put(root,name,inputs.digest_json(value))
    with pytest.raises((ValueError,FileNotFoundError)):inputs.provenance(root,8,source)


def test_archive_deterministic_exact_extraction_and_no_merge(cohort,tmp_path):
    a,b=tmp_path/'a.tar',tmp_path/'b.tar';first=inputs.archive(cohort['root'],cohort['manifest'],a);assert inputs.archive(cohort['root'],cohort['manifest'],b)==first
    assert a.stat().st_mode&0o777==0o400
    dest=tmp_path/'new';report=inputs.extract(a,cohort['manifest'],dest);assert report['files']==22
    assert all(inputs.identity(dest/name,True)==row for name,row in cohort['manifest']['files'].items())
    with pytest.raises(ValueError):inputs.extract(a,cohort['manifest'],dest)
    with pytest.raises(ValueError):inputs.archive(cohort['root'],cohort['manifest'],a)


@pytest.mark.parametrize('fault',['symlink','hardlink','tamper'])
def test_archive_rejects_alias_or_changed_input(cohort,tmp_path,fault):
    path=cohort['root']/'outputs/episode_000008/depth_full/000000.npz'
    if fault=='symlink':path.unlink();path.symlink_to('missing')
    elif fault=='hardlink':os.link(path,tmp_path/'alias')
    else:path.write_bytes(b'changed')
    with pytest.raises((ValueError,FileNotFoundError)):inputs.archive(cohort['root'],cohort['manifest'],tmp_path/'archive.tar')


@pytest.mark.parametrize('fault',['traversal','link','duplicate','short','manifest','foreign'])
def test_untrusted_tar_structure_fail(cohort,tmp_path,fault):
    path=tmp_path/'bad.tar';manifest=cohort['manifest'];names=sorted(manifest['files'])
    with tarfile.open(path,'w')as tar:
        raw=b'{}\n'if fault=='manifest'else inputs.digest_json(manifest);m=tarfile.TarInfo(inputs.MANIFEST_NAME);m.size=len(raw);tar.addfile(m,io.BytesIO(raw))
        name='../escape'if fault=='traversal'else 'weights/model.pt'if fault=='foreign'else names[0]
        data=(cohort['root']/names[0]).read_bytes();m=tarfile.TarInfo(name)
        if fault=='link':m.type=tarfile.SYMTYPE;m.linkname='foreign';tar.addfile(m)
        else:m.size=len(data);tar.addfile(m,io.BytesIO(data))
        if fault=='duplicate':tar.addfile(m,io.BytesIO(data))
    with pytest.raises((ValueError,tarfile.TarError)):inputs.extract(path,manifest,tmp_path/'staging')


def test_runtime_contract_no_history_mutation():
    raw=(ROOT/'infra/frontend_peer_receive.py').read_bytes()
    assert digest(raw)['sha256'] and 'ARCHIVE_BYTES = 19_911_464_960'in raw.decode()
    for name in('run_pose_peer_inventory','run_pose_peer_receive','run_pose_peer_send','run_pose_peer_run'):
        source=(ROOT/'infra'/f'{name}.sh').read_text();assert 'set +x'in source
        assert 'world-reward-ncc-h100-02'in source if name in('run_pose_peer_receive','run_pose_peer_run')else 'scenesmith-ncc-h100-01'in source
