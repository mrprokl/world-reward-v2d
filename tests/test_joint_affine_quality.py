"""Tiny analytic paired-depth plumbing only; no GPU/synthetic-media downloads."""
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest


@pytest.fixture
def modules(monkeypatch):
    root=Path(__file__).resolve().parents[1];monkeypatch.syspath_prepend(str(root/'infra'))
    import joint_affine_fit as fit
    import joint_affine_evaluate as evaluate
    monkeypatch.setattr(fit,'HEIGHT',32);monkeypatch.setattr(fit,'WIDTH',32)
    monkeypatch.setattr(fit,'K',np.array([[40.,0.,16.],[0.,40.,16.],[0.,0.,1.]]))
    return fit,evaluate


def fixture(fit,offset=True,constant=False):
    yy,xx=np.indices((32,32));values=[]
    for i in range(6):
        z=np.full((32,32),1.5+i*.3,np.float32) if constant else (1.5+.01*xx+.01*yy+i*.3).astype(np.float32)
        h=(1.7*z+(.2+i*.05 if offset else 0.)).astype(np.float32);mask=np.ones(z.shape,bool)
        values.append(dict(raw_depth=z,rendered_depth=h,silhouette=mask.copy(),human_mask=mask.copy(),
            object_mask=np.zeros(z.shape,bool),validity=mask.copy(),K=fit.K.copy()))
    return values


def test_actual_paired_train_baseline_and_selected_rays(modules):
    fit,_=modules;values=fixture(fit);before,after,r=fit.fit_clip(values)
    assert r['selected']['selection']=='affine' and len(before)==len(after)==6
    np.testing.assert_allclose(after[2][...,2],values[2]['rendered_depth'],rtol=1e-6)
    assert np.max(np.abs(after[2][...,2]-before[2][...,2]))>1e-3
    yy,xx=np.indices((32,32))
    np.testing.assert_allclose(after[2][...,0]/after[2][...,2]*40+16,xx+.5,atol=2e-6)


def test_underconstrained_keeps_all_frames_baseline_no_offset(modules):
    fit,_=modules;values=fixture(fit,constant=True);before,after,r=fit.fit_clip(values)
    assert r['selected']['selection']=='alpha_only' and r['selected_offsets']==[0.]*6
    assert all(np.array_equal(a,b) for a,b in zip(before,after))


@pytest.mark.parametrize('fault',['short','support','nan','negative','wrongK','badvalid'])
def test_invalid_inputs_fatal_not_abstained_or_repaired(modules,fault):
    fit,_=modules;v=fixture(fit)
    if fault=='short':v=v[:5]
    elif fault=='support':v[1]['human_mask'][:]=False
    elif fault=='nan':v[0]['raw_depth'][0,0]=np.nan
    elif fault=='negative':v[0]['raw_depth'][0,0]=-1
    elif fault=='wrongK':v[0]['K'][0,0]+=1
    elif fault=='badvalid':v[0]['validity']=v[0]['validity'].astype(np.uint8)
    with pytest.raises(ValueError):fit.fit_clip(v)


def test_private_gate_allclips_nohidden_regression_or_coverage(modules):
    _,evaluate=modules
    def scenes(vals):return [dict(baseline_visible_chamfer_half_cm=10.,selected_visible_chamfer_half_cm=v) for v in vals]
    assert evaluate.decision(scenes([9.,8.,9.]),True)['synthetic_affine_grounding_hypothesis_supported']
    assert not evaluate.decision(scenes([9.,8.,11.]),True)['synthetic_affine_grounding_hypothesis_supported']
    assert not evaluate.decision(scenes([9.,8.,9.]),False)['synthetic_affine_grounding_hypothesis_supported']


def test_wrappers_only_privatequalitymount_noGPUfit():
    root=Path(__file__).resolve().parents[1]
    for name in ['run_joint_affine_fit.sh','run_joint_affine_evaluate.sh','run_joint_affine_prepare.sh','run_joint_affine_validation.sh']:
        subprocess.run(['bash','-n',str(root/'infra'/name)],check=True)
    fit=(root/'infra/run_joint_affine_fit.sh').read_text();quality=(root/'infra/run_joint_affine_evaluate.sh').read_text()
    assert 'eval_private' not in fit and '--gpus' not in fit and '--gpus' not in quality
    assert 'src=$BASE/eval_private,dst=$BASE/eval_private,readonly' in quality
    assert 'evaluation_alignment_performed=False' not in quality
    source=(root/'infra/joint_affine_evaluate.py').read_text()
    assert source.index('aligned_public(base,values)')<source.index("private=base/'eval_private'")


@pytest.fixture
def frozen_cohort(modules, monkeypatch, tmp_path):
    """Eighteen tiny local analytic arrays; not a GPU accuracy experiment."""
    fit,evaluate=modules;infer=fit.observations
    for key,value in [('HEIGHT',32),('WIDTH',32),('VERTICES',4),('FACES',4),('CAMERA_K',fit.K)]:
        monkeypatch.setattr(infer,key,value)
    from PIL import Image
    base=tmp_path/'validation/joint_affine_rgb_v1'
    for name in ['inputs','automatic_masks','predictions_v1','alignment_v1','eval_private']:(base/name).mkdir(parents=True)
    images=[];masks=[];values=[]
    faces=np.array([[0,2,1],[0,1,3],[1,2,3],[2,0,3]],np.int64)
    vertices=np.array([[0,0,3],[.1,0,3],[0,.1,3],[0,0,3.1]],np.float32)
    for index in range(18):
        clip,frame=divmod(index,6);stem=f'clip_{clip:02d}_frame_{frame:03d}'
        v=fixture(fit)[frame];v['human_mask'][-4:]=False;v['object_mask'][-4:]=True
        v['silhouette']=v['human_mask'].copy();v['rendered_depth'][~v['silhouette']]=np.nan
        v.update(raw_points=fit.unproject(v['raw_depth'],v['validity'],1.,0.),human_vertices_camera_m=vertices.copy(),
                 human_faces=faces.copy(),clip_index=np.array(clip,np.int64),frame_index=np.array(frame,np.int64))
        values.append(v);p=base/'inputs'/(stem+'.png');Image.fromarray(np.zeros((32,32,3),np.uint8)).save(p)
        images.append(dict(file=p.name,sha256=fit.sha256(p),width=32,height=32))
        row=dict(file=p.name,rgb_sha256=fit.sha256(p),clip_index=clip,frame_index=frame)
        for kind in ('human','object'):
            p=base/'automatic_masks'/f'{stem}_{kind}.png';Image.fromarray(v[kind+'_mask'].astype(np.uint8)*255).save(p)
            row[kind+'_mask_file']=p.name;row[kind+'_mask_sha256']=fit.sha256(p)
        masks.append(row)
    (base/'inputs/manifest.json').write_text(json.dumps(dict(schema=infer.SCHEMA,images=images)))
    maskreport=dict(stage='public_joint_affine_rgb_automatic_masks',status='pass',frames=18,private_truth_read=False,
        challenge_inputs_used=False,ground_truth_used=False,hand_labeled_test=False,oracle_modes=[],human_query='person.',object_query='bottle.',
        input_manifest_sha256=fit.sha256(base/'inputs/manifest.json'),records=masks)
    (base/'automatic_masks/report.json').write_text(json.dumps(maskreport))
    records,hashes=infer.public_inputs(tmp_path);body=infer.human;depth=infer.depth_model
    report=dict(stage=infer.STAGE,status='pass',phase='complete',body_calls_completed=18,MoGe_calls_completed=18,
        actual_body_inference=True,actual_MoGe_inference=True,actual_predicted_human_render=True,inputs_assets_sources_rechecked=True,
        public_inputs_sha=hashes['public_inputs_sha'],mask_report_sha=hashes['mask_report_sha'],
        public_records=[{k:v for k,v in r.items() if not k.endswith('_path')} for r in records],
        private_truth_read=False,ground_truth_used=False,challenge_inputs_used=False,hand_labeled_test=False,oracle_modes=[],outputs=[],
        network='none',affine_fit_performed=False,scale_fit_performed=False,camera_source='fixed_RGB_size_prior',body_inference_type='body',
        geometry_filled=False,apply_mask=False,force_projection=True,frames_completed=18,adoption_performed=False,
        producer_revision='a'*40,image_id='sha256:'+'b'*64,script_sha256=fit.sha256(Path(infer.__file__)),helper_source_sha256=infer.helper_identities(),
        camera_K=infer.CAMERA_K.tolist(),focal_geometry_source_sha256=infer.GEOMETRY_SHA,MoGe_model_revision=depth.MODEL_REVISION,
        MoGe_model_asset=dict(sha256=depth.MODEL_SHA,bytes=depth.MODEL_BYTES),
        MoGe_source=dict(revision=depth.SOURCE_REVISION,v2_source_sha256=depth.SOURCE_V2_SHA),
        body_model=dict(body_revision=body.body.BODY_REVISION,upstream_revision=body.body.UPSTREAM_REVISION,dinov3_revision=body.body.DINOV3_REVISION,
                        body_assets={'model.ckpt':dict(sha256=body.BODY_SHA,bytes=body.BODY_BYTES)}),
        native_focal_solver_calls=[dict(file=r['file'],clip_index=r['clip_index'],frame_index=r['frame_index'],native_nearest64_valid_pixels=100,
                                       focal_prior_supplied=True,original_solver_returned=True) for r in records])
    for v,r in zip(values,records):
        p=base/'predictions_v1'/r['file'].replace('.png','.npz');np.savez_compressed(p,**v)
        report['outputs'].append(dict(file=p.name,sha256=fit.sha256(p),clip_index=r['clip_index'],frame_index=r['frame_index'],rgb_sha256=r['rgb_sha256']))
    (base/'predictions_v1/report.json').write_text(json.dumps(report))
    alignment=dict(stage=fit.STAGE,status='pass',phase='complete',private_truth_read=False,ground_truth_used=False,challenge_inputs_used=False,
        hand_labeled_test=False,oracle_modes=[],outputs_completed=18,frames=18,adoption_performed=False,human_geometry_modified=False,
        canonical_object_mesh_modified=False,camera_modified=False,source_validity_modified=False,network='none',producer_revision='c'*40,
        image_id='sha256:'+'b'*64,script_sha256=fit.sha256(Path(fit.__file__)),outputs=[],clips=[],public_hashes=hashes,
        observation_report_sha256=fit.sha256(base/'predictions_v1/report.json'))
    cases=[]
    for clip in range(3):
        before,after,diagnostic=fit.fit_clip(values[clip*6:(clip+1)*6]);alignment['clips'].append(dict(clip_index=clip,**diagnostic))
        for frame in range(6):
            v=values[clip*6+frame];stem=f'clip_{clip:02d}_frame_{frame:03d}';p=base/'alignment_v1'/(stem+'.npz')
            np.savez_compressed(p,baseline_points=before[frame],selected_points=after[frame],validity=v['validity'],K=v['K'],
                                clip_index=v['clip_index'],frame_index=v['frame_index'])
            alignment['outputs'].append(dict(file=p.name,sha256=fit.sha256(p),clip_index=clip,frame_index=frame))
            p=base/'eval_private'/(stem+'.npz');ids=np.zeros((32,32),np.int64);ids[-4:]=4
            np.savez_compressed(p,human_vertices_camera_m=vertices,human_faces=faces,object_vertices_camera_m=vertices+.2,object_faces=faces,
                camera_K=fit.K,scene_depth_m=after[frame][...,2],visible_face_indices=ids,clip_index=v['clip_index'],frame_index=v['frame_index'])
            cases.append(dict(file=stem+'.png',clip_index=clip,frame_index=frame,truth_sha256=fit.sha256(p),rgb_sha256=records[clip*6+frame]['rgb_sha256']))
    (base/'alignment_v1/report.json').write_text(json.dumps(alignment))
    (base/'eval_private/render-report.json').write_text(json.dumps(dict(stage='own_joint_affine_human_object_rgb_render',status='pass',phase='complete',frames=18,
        public_manifest_sha256=hashes['public_inputs_sha'],challenge_inputs_used=False,inference_performed=False,cases=cases,network='none',all_truth_private=True,
        generated_GT_masks_supplied_to_inference=False,code_revision='d'*40,image_id='sha256:'+'b'*64)))
    return tmp_path,base,report,alignment


def test_full_paired_analytic_quality_and_public_replay(modules,frozen_cohort):
    fit,evaluate=modules;root,base,_,_=frozen_cohort;report={}
    evaluate.run(root,report)
    assert report['status']=='pass' and len(report['frames'])==18 and len(report['clips'])==3
    assert report['decision']['synthetic_affine_grounding_hypothesis_supported']
    assert all(r['selected']['visible_chamfer_half_cm']<1e-5 for r in report['frames'])


@pytest.mark.parametrize('fault',['mask','row_bool','source','helper','network','fit','camera','model','native_count','native_order','native_support',
                                  'native_return','aligned_Z','aligned_beta','aligned_validity','aligned_report_link'])
def test_bad_public_data_rejected_before_first_private_read(modules,frozen_cohort,monkeypatch,fault):
    fit,evaluate=modules;root,base,report,alignment=frozen_cohort
    if fault in ('mask','aligned_Z','aligned_beta','aligned_validity'):
        which='predictions_v1' if fault=='mask' else 'alignment_v1';p=base/which/'clip_00_frame_000.npz'
        with np.load(p,allow_pickle=False) as f:v={k:f[k].copy() for k in f.files}
        if fault=='mask':v['object_mask'][0,0]=True
        elif fault=='aligned_validity':v['validity']=v['validity'].astype(np.uint8)
        elif fault=='aligned_beta':
            v['selected_points']=fit.unproject(np.full((32,32),4.,np.float32),v['validity'],1.,.2)
        else:v['selected_points']=fit.unproject(np.full((32,32),4.,np.float32),v['validity'],1.,0.)
        np.savez_compressed(p,**v)
        (report if which=='predictions_v1' else alignment)['outputs'][0]['sha256']=fit.sha256(p)
    elif fault=='row_bool':report['outputs'][0]['clip_index']=False
    elif fault=='source':report['script_sha256']='0'*64
    elif fault=='helper':report['helper_source_sha256']['camera_render']='0'*64
    elif fault=='network':report['network']='host'
    elif fault=='fit':report['scale_fit_performed']=True
    elif fault=='camera':report['camera_K'][0][0]+=1
    elif fault=='model':report['MoGe_model_asset']['sha256']='0'*64
    elif fault=='native_count':report['native_focal_solver_calls'].pop()
    elif fault=='native_order':report['native_focal_solver_calls'][0]['frame_index']=5
    elif fault=='native_support':report['native_focal_solver_calls'][0]['native_nearest64_valid_pixels']=1
    elif fault=='native_return':report['native_focal_solver_calls'][0]['original_solver_returned']=False
    (base/'predictions_v1/report.json').write_text(json.dumps(report))
    alignment['observation_report_sha256']=fit.sha256(base/'predictions_v1/report.json')
    (base/'alignment_v1/report.json').write_text(json.dumps(alignment))
    if fault=='aligned_report_link':
        p=base/'alignment_v1/report.json';q=base/'alias.json';p.rename(q);p.symlink_to(q)
    original=Path.read_text
    def guard(path,*args,**kwargs):
        if 'eval_private' in path.parts:raise AssertionError('Private read occurred before public rejection')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'read_text',guard)
    with pytest.raises(ValueError):evaluate.run(root,{})
