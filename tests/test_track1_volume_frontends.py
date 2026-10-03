"""Tiny actual stdlib inventory plus fake child commands; no arrays/GPU/cloud."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess

import pytest

ROOT=Path(__file__).resolve().parents[1]
WRAPPER=ROOT/'infra/run_track1_volume_frontends.sh'
IMAGE='sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7'
CHILDREN=('run_object_pose_smoke.sh','object_pose_smoke.py','run_cari_prepare.sh','cari_prepare.py',
          'cari_wrapper_common.sh','body_smoke.py','camera_render.py','volume_geometry_loader.py',
          'volume_mesh_pin_inventory.py','object_budget_volume.py')


@pytest.fixture
def inventory():
    spec=importlib.util.spec_from_file_location('volume_continuation_actual_inventory',ROOT/'infra/volume_mesh_pin_inventory.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


@pytest.fixture
def runtime(tmp_path,inventory):
    root=tmp_path/'runtime';revision='a'*40
    code=root/'jobs'/revision/'run_track1_volume_frontends/code';(code/'infra').mkdir(parents=True)
    (code/'configs').mkdir();(code.parent/'revision').write_text(revision+'\n')
    (code.parent/'source-sha256').write_text('b'*64+'\n')
    wrapper=code/'infra/run_track1_volume_frontends.sh'
    wrapper.write_text(WRAPPER.read_text().replace('/srv/scenesmith/world-reward',str(root)))
    for name in CHILDREN:(code/'infra'/name).write_text('tiny immutable source\n')
    (code/'infra/volume_mesh_pin_inventory.py').write_bytes((ROOT/'infra/volume_mesh_pin_inventory.py').read_bytes())
    child='''#!/usr/bin/env bash
set -euo pipefail
name="$(basename "$0")";episode="$2";printf -v padded '%06d' "$episode"
if [[ "$name" == run_object_pose_smoke.sh ]];then
 stage=object_pose_full;[[ "$*" == "--episode $episode --full-video --mesh-source volume" ]] || exit 9
else
 stage=cari_inputs;[[ "$*" == "--episode $episode --no-wait" ]] || exit 9
fi
printf 'child|%s|%s\\n' "$stage" "$*" >> "$FAKE_LOG"
if [[ "$stage" == "${FAKE_FAIL_STAGE:-}" ]];then exit 7;fi
mkdir "$WR_ROOT/outputs/episode_$padded/$stage"
printf '{"status":"pass"}' > "$WR_ROOT/outputs/episode_$padded/$stage/report.json"
'''
    for name in ('run_object_pose_smoke.sh','run_cari_prepare.sh'):(code/'infra'/name).write_text(child)
    bin=tmp_path/'bin';bin.mkdir();log=tmp_path/'calls.log'
    commands={
        'flock':'''printf 'lock|%s\\n' "$*" >> "$FAKE_LOG"
if [ "$1" = --nonblock ] && [ "${FAKE_LOCK_BUSY:-0}" = 1 ];then exit 1;fi
''',
        'nvidia-smi':'''printf 'smi|%s\\n' "$*" >> "$FAKE_LOG"
if [ "${FAKE_SMI_FAIL:-0}" = 1 ];then exit 8;fi
[ "${FAKE_GPU_BUSY:-0}" != 1 ] || printf '1234\\n'
''',
        'docker':'''printf 'docker|%s\\n' "$*" >> "$FAKE_LOG"
if [ "${FAKE_DOCKER_FAIL:-0}" = 1 ];then exit 8;fi
printf '%s\\n' "$FAKE_IMAGE"
''',
        'timeout':'''printf 'timeout|%s|%s|%s|%s\\n' "$1" "$2" "$3" "$5" >> "$FAKE_LOG"
shift 3;exec "$@"
''',
    }
    for name,body in commands.items():path=bin/name;path.write_text('#!/bin/sh\n'+body);path.chmod(0o755)
    env=dict(os.environ,WR_ROOT=str(root),WR_CODE=str(code),WR_CODE_REVISION=revision,
        FAKE_IMAGE=IMAGE,FAKE_LOG=str(log),PATH=str(bin)+':'+os.environ['PATH'])
    def write(path,value):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value)if isinstance(value,dict)else value);return inventory.identity(path)
    def setup(episode):
        base=root/f'outputs/episode_{episode:06d}';paths=inventory.paths(episode);video='1'*64;scale=2.5;count=97
        flags=dict(status='pass',episode_index=episode,input_track='track_1',input_sha256=video,
            ground_truth_used=False,hand_labeled_test=False,oracle_modes=[])
        reports={}
        reports['automatic_masks']=dict(flags,stage='automatic_masks',frames=count)
        masks=write(base/'automatic_masks/report.json',reports['automatic_masks'])
        for name,stage in (('body_smoke','sam3d_body_three_frame_smoke'),('depth_smoke','monocular_moge2_three_frame'),
                ('body_full','sam3d_body_full_video_initializer'),('depth_full','monocular_moge2_full_video')):
            reports[name]=dict(flags,stage=stage,frames=[{'frame_index':i}for i in range(count)],total_video_frames=count)
            write(base/name/'report.json',reports[name])
        reports['body_full/cari_adapter']=dict(flags,stage='native_cari_body_adapter_full_video',frames=count,
            body_report_sha256=inventory.identity(base/'body_full/report.json')['sha256'])
        write(base/'body_full/cari_adapter/report.json',reports['body_full/cari_adapter'])
        alignment=dict(flags,stage='predicted_human_anchored_moge2_pointmaps',coordinate_frame='OpenCV_x_right_y_down_z_forward',
            pointmap_scale_application='one_clip_scalar_to_MoGe2_XYZ_already_applied',
            body_report_sha256=inventory.identity(base/'body_smoke/report.json')['sha256'],
            depth_report_sha256=inventory.identity(base/'depth_smoke/report.json')['sha256'])
        alignment_id=write(root/paths['alignment'],alignment)
        obj=dict(flags,stage='sam3d_objects_grounded_fixed_frame',frame_index=0,scale_source='already_human_anchored_MoGe2_no_second_scalar',
            pointmap_grounding={'alignment_report_sha256':alignment_id['sha256']},transform={'scale':[scale]*3},
            object_sha256='2'*64,transform_sha256='3'*64,intrinsics_sha256='4'*64)
        object_id=write(root/paths['object'],obj)
        geometry=write(root/paths['geometry'],'tiny bytes not decoded NPZ');glb=write(root/paths['glb'],'tiny bytes not decoded GLB')
        build=write(root/paths['build'],dict(stage='world_reward_volume_qem_build',status='pass',image_id='sha256:'+'c'*64))
        evidence=dict(build_report_sha256=build['sha256'],binary_sha256='c'*64,build_info={'source_sha256':'d'*64})
        control=write(root/paths['control'],dict(stage='own_volume_constrained_intersection_qem_geometry',status='pass',script_sha256=inventory.CONTROL_SHA,
            challenge_inputs_used=False,adoption_performed=False,target_faces=4096,target_vertices=4096,build=evidence))
        metrics=dict(birthface_matched_shells=[{'relative_volume_error':.001}],sampled_bidirectional_chamfer_diagonal_ratio=.001,net_volume_relative_error=.001)
        report=dict(flags,stage=inventory.STAGE,producer_revision='e'*40,script_sha256='f'*64,image_id='sha256:'+'c'*64,
            adoption_performed=False,challenge_performance_verified=False,budget_seconds=900,target_faces=4096,target_vertices=4096,
            source_shell_volume_relative_limit=.05,components_deleted=False,holes_filled=False,normals_repaired=False,
            frame_poses_changed=False,native_cost_and_placement_unchanged=True,source_embedding_exact_universal_proof=False,
            source_intersecting_faces=0,independent_candidate_intersecting_faces=0,packed_intersecting_faces=0,metric_scale_baked_once=scale,
            source_arrays_unchanged=True,official_helper_sha256=inventory.OFFICIAL_HELPER_SHA,geometry_sha256=geometry['sha256'],canonical_glb_sha256=glb['sha256'],
            candidate_geometry=copy.deepcopy(metrics),export_geometry=copy.deepcopy(metrics),
            official_pack_fidelity=dict(oriented_triangles_exact=True,official_helper_simplification_invoked=False,nonexact_merge_or_face_deletion=False),
            source_hashes=dict(object_report=object_id['sha256'],alignment_report=alignment_id['sha256'],video=video,
                **{'object.glb':obj['object_sha256'],'transform.json':obj['transform_sha256'],'intrinsics.json':obj['intrinsics_sha256']},
                body_smoke_report=alignment['body_report_sha256'],depth_smoke_report=alignment['depth_report_sha256'],mask_report=masks['sha256'],prompts='6'*64),
            control_evidence=dict(evidence,volume_gate_sha256=control['sha256']))
        write(root/paths['report'],report)
        pins=inventory.inventory(root,episode,'e'*40,'f'*64,video,object_id['sha256'],alignment_id['sha256'],scale)
        pin=code/f'configs/volume_mesh_{episode:06d}_pins.json';write(pin,pins)
        for path in (code,*code.rglob('*')):path.chmod(0o555 if path.is_dir() else 0o444)
        return base,pin,pins,paths
    def run(*args):return subprocess.run(['rtk','proxy','bash',str(wrapper),*args],env=env,capture_output=True,text=True,timeout=8)
    return root,code,env,log,wrapper,setup,run


@pytest.mark.parametrize('episode',range(30))
def test_two_exact_children_all_episodes_existing_sources_and_seven_hashes(runtime,episode):
    root,code,env,log,wrapper,setup,run=runtime;base,pin,pins,paths=setup(episode)
    before={name:(root/name).read_bytes()for name in pins['files']}
    result=run('--episode',str(episode));assert result.returncode==0,result.stderr
    rows=log.read_text().splitlines()
    assert [r for r in rows if r.startswith('child|')]==[
        f'child|object_pose_full|--episode {episode} --full-video --mesh-source volume',
        f'child|cari_inputs|--episode {episode} --no-wait']
    assert [r.split('|')[3]for r in rows if r.startswith('timeout|')]==['7200s','7200s']
    assert sum(r.startswith('smi|')for r in rows)==1
    assert rows.index('lock|--nonblock 9')<next(i for i,r in enumerate(rows)if r.startswith('child|'))
    assert rows.index('lock|--unlock 9')<next(i for i,r in enumerate(rows)if r.startswith('child|cari_inputs'))
    assert before=={name:(root/name).read_bytes()for name in pins['files']}
    events=[json.loads(line)for line in result.stdout.splitlines()]
    assert [(e['stage'],e['phase'])for e in events]==[(s,p)for s in ('preflight','gpu_preflight','object_pose_full','cari_inputs')for p in ('start','pass')]
    assert all(set(e)=={'mode','stage','phase','timestamp_utc'}and e['mode']=='pinned_volume_frontends_only'for e in events)


@pytest.mark.parametrize('args',[[],['--episode'],['--episode','30'],['--episode','-1'],['--episode','00'],
    ['--episode','true'],['--episode','2','--episode','2'],['--episode=2'],['--episode','2','--resume'],
    ['--episode','2','--mesh-source','volume'],['--episode','2','--root','/tmp'],['--ep','2']])
def test_invalid_controls_no_mutation_before_any_command(runtime,args):
    root,code,env,log,wrapper,setup,run=runtime;result=run(*args)
    assert result.returncode==2 and not log.exists() and not (root/'outputs').exists()


@pytest.mark.parametrize('target',['object_pose_full','cari_inputs'])
@pytest.mark.parametrize('kind',['directory','file','broken_symlink'])
def test_absence_even_empty_failed_directory_never_removed(runtime,target,kind):
    root,code,env,log,wrapper,setup,run=runtime;base,*_=setup(2);path=base/target
    if kind=='directory':path.mkdir()
    elif kind=='file':path.write_text('retained')
    else:path.symlink_to('missing')
    result=run('--episode','2');assert result.returncode!=0 and not log.exists()
    assert path.is_symlink()or path.exists()
    assert 'no removal/overwrite/resume' in result.stderr


@pytest.mark.parametrize('mode',['GPU_BUSY','SMI_FAIL','LOCK_BUSY','DOCKER_FAIL','IMAGE'])
def test_busy_gpu_or_image_fail_before_children(runtime,mode):
    root,code,env,log,wrapper,setup,run=runtime;base,*_=setup(2)
    env['FAKE_'+mode]='sha256:'+'0'*64 if mode=='IMAGE' else '1'
    result=run('--episode','2');assert result.returncode!=0
    assert not any(r.startswith('child|')for r in log.read_text().splitlines())
    assert json.loads(result.stdout.splitlines()[-1])['phase']=='fail'
    assert not (base/'object_pose_full').exists()


@pytest.mark.parametrize('stage',['object_pose_full','cari_inputs'])
def test_abort_preserves_child_code_no_resume_or_rerun(runtime,stage):
    root,code,env,log,wrapper,setup,run=runtime;setup(2);env['FAKE_FAIL_STAGE']=stage
    result=run('--episode','2');assert result.returncode==7
    rows=log.read_text().splitlines();children=[r.split('|')[1]for r in rows if r.startswith('child|')]
    assert children==(['object_pose_full']if stage=='object_pose_full'else ['object_pose_full','cari_inputs'])
    assert ('lock|--unlock 9' in rows)==(stage=='cari_inputs')
    assert json.loads(result.stdout.splitlines()[-1])['phase']=='fail'


@pytest.mark.parametrize('role',['report','geometry','glb','control','build','object','alignment'])
def test_all7_artifact_hashes_before_gpu_no_arrays_decoded(runtime,role):
    root,code,env,log,wrapper,setup,run=runtime;base,pin,pins,paths=setup(2)
    with (root/paths[role]).open('a')as h:h.write('changed')
    result=run('--episode','2');assert result.returncode!=0 and not log.exists()
    assert not (base/'object_pose_full').exists()


@pytest.mark.parametrize('fault',['pinsmissing','pinsepisode','pinsschema','pinssymlink','pinswritable',
    'childmissing','childwritable','childsymlink','revision','archive','WR_ROOT','WR_CODE','WR_CODE_REVISION','entrypoint'])
def test_committed_actual_immutable_source_and_config(runtime,fault,tmp_path):
    root,code,env,log,wrapper,setup,run=runtime;base,pin,pins,paths=setup(2)
    if fault=='pinsmissing':pin.parent.chmod(0o755);pin.unlink()
    elif fault in ('pinsepisode','pinsschema'):
        pin.chmod(0o644);pins['episode_index'if fault=='pinsepisode'else 'schema']=0 if fault=='pinsepisode'else 'wrong'
        pin.write_text(json.dumps(pins));pin.chmod(0o444)
    elif fault=='pinssymlink':pin.parent.chmod(0o755);pin.unlink();pin.symlink_to('missing')
    elif fault=='pinswritable':pin.chmod(0o644)
    elif fault.startswith('child'):
        path=code/'infra/volume_geometry_loader.py';path.parent.chmod(0o755)
        if fault=='childmissing':path.unlink()
        elif fault=='childwritable':path.chmod(0o644)
        else:path.unlink();path.symlink_to('body_smoke.py')
    elif fault=='revision':(code.parent/'revision').write_text('main')
    elif fault=='archive':(code.parent/'source-sha256').write_text('bad')
    elif fault=='WR_ROOT':env['WR_ROOT']=str(tmp_path/'wrong')
    elif fault=='WR_CODE':env['WR_CODE']=str(tmp_path/'wrong')
    elif fault=='WR_CODE_REVISION':env['WR_CODE_REVISION']='A'*40
    else:
        outside=tmp_path/'outside.sh';outside.write_text(wrapper.read_text())
        result=subprocess.run(['rtk','proxy','bash',str(outside),'--episode','2'],env=env,capture_output=True,text=True)
        assert result.returncode!=0 and not log.exists();return
    result=run('--episode','2');assert result.returncode!=0 and not log.exists()


@pytest.mark.parametrize('name',['automatic_masks','body_smoke','depth_smoke','body_full','depth_full',
    'body_full/cari_adapter','object_grounded','scale_smoke'])
def test_each_existing_frontend_report_required(runtime,name):
    root,code,env,log,wrapper,setup,run=runtime;base,*_=setup(2)
    (base/name/'report.json').unlink();result=run('--episode','2')
    assert result.returncode!=0 and not log.exists()


def test_actual_bash_and_sourceonly_archive_closure_no_later_producers():
    subprocess.run(['rtk','proxy','bash','-n',str(WRAPPER)],check=True)
    text=WRAPPER.read_text()
    assert '7200s' in text and '--kill-after=10s' in text and 'legacy/full-refine' in text and 'broad mounts' in text
    assert 'systemctl' not in text and 'rmdir ' not in text and 'rm ' not in text and '&\n' not in text
    for forbidden in ('run_automatic_masks.sh','run_depth_smoke.sh','run_cari_forward.sh','run_cari_converter.sh','run_cari_full_refine.sh'):
        assert forbidden not in text
    spec=importlib.util.spec_from_file_location('volume_frontend_archive',ROOT/'infra/azure_job.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    files={str(p.relative_to(ROOT)):p.read_bytes()for base in ('infra','src','configs')for p in (ROOT/base).rglob('*')if p.is_file()and '__pycache__'not in p.parts}
    files['pyproject.toml']=(ROOT/'pyproject.toml').read_bytes()
    selected=set(module.runtime_bundle_paths(files,'infra/run_track1_volume_frontends.sh'))
    assert selected>={'infra/'+name for name in CHILDREN}
    assert 'infra/cari_forward.py'not in selected and 'infra/cari_converter.py'not in selected
    assert 'infra/volume_mesh_pin_inventory.py'in selected and 'infra/volume_geometry_loader.py'in selected
