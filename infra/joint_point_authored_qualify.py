"""Authored actual-MHR A/B, fresh A/A, or single zero-delegation control.

Reuse native constructors/losses and the existing paired sequencing operator.
No challenge records, inherited predictions, GT, renderer retries or calibration.
"""
from __future__ import annotations
import argparse
import copy
from dataclasses import asdict
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import re
import signal
import stat
import subprocess
import sys
import tarfile
import time

ROOT=Path('/srv/scenesmith/world-reward');ENTRY='run_joint_point_authored_qualify'
PROTOCOL='configs/joint_point_authored_runtime_protocol_v1.json'
PROTOCOL_PIN={'bytes':8529,'sha256':'559c41d113f85083dd1cfb60346cb7f56cad656e050542f7d5094ddee03f668a'}
REPEAT_PROTOCOL='configs/joint_point_native_repeat_protocol_v1.json'
REPEAT_PIN={'bytes':3591,'sha256':'db82e8b8d51058920e02301f20887d84d52e1b25d6c5488596ebd6e51c23706e'}
DELEGATE_PROTOCOL='configs/joint_point_zero_delegate_protocol_v1.json'
DELEGATE_PIN={'bytes':3759,'sha256':'b9d9590b9bafdd8568ee0730a69b57a3def041041522fbe15be191a56ab1d6ed'}
ARCHIVE_PINS='configs/frontend_asset_archive_pins.json'
IMAGE='sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7'
BODY='weights/cari4d/sam3d_body/checkpoints/sam-3d-body-dinov3'
NATIVE='reconstruction/modules/v2d_cari4d/lib/cari4d'
BODY_SOURCE='reconstruction/modules/v2d_sam3d_body/lib/sam_3d_body'
HELPERS=('infra/joint_point_authored_qualify.py','infra/run_joint_point_authored_qualify.sh',
 'infra/joint_point_native_qualify.py','infra/cari_full_refine.py','infra/cari_full_forward.py',
 'infra/body_smoke.py','infra/mediapipe_cpu_runtime_verify.py',
 'src/world_reward/joint_point_objective.py','src/world_reward/joint_point_evidence.py',
 'src/world_reward/point_surface_queries.py','src/world_reward/fixed_shape_point_pose.py',PROTOCOL,ARCHIVE_PINS)


def runtime(code):
    sys.path[:0]=[str(code/'infra'),str(code/'src')]
    import mediapipe_cpu_runtime_verify as rt
    return rt


def profile(mode=None):
    if mode is None:return PROTOCOL_PIN,'joint-point-authored-qualify-',('A_original','B_point_weight_zero')
    if mode=='native_repeat':return REPEAT_PIN,'joint-point-native-repeat-',('A_native_first','A_native_second')
    if mode=='zero_delegate':return DELEGATE_PIN,'joint-point-zero-delegate-',('Z_point_zero_delegate',)
    raise ValueError('Explicit known native control required')


def helpers(mode=None):
    profile(mode);return HELPERS if mode is None else (*HELPERS,DELEGATE_PROTOCOL if mode=='zero_delegate' else REPEAT_PROTOCOL)


def output_path(root,revision,mode=None):
    if not re.fullmatch('[0-9a-f]{40}',revision):raise ValueError('Exact producer revision required')
    return root/'results'/(profile(mode)[1]+revision)


def protocol(rt,code,mode=None):
    profile(mode);c=rt.pinned(code/PROTOCOL,PROTOCOL_PIN,16<<10)
    if mode is None:return c
    selected=rt.pinned(code/(DELEGATE_PROTOCOL if mode=='zero_delegate' else REPEAT_PROTOCOL),profile(mode)[0],16<<10)
    rt.require(selected['control']==mode and selected['base_protocol']==dict(path=PROTOCOL,**PROTOCOL_PIN)
        and selected['upstream_revision']==c['upstream_revision'] and selected['runtime_image_id']==c['runtime_image_id']
        and selected['optimizer_source']==dict(path='learning/training/mhr_opt_refineout.py',**c['primary_sources']['learning/training/mhr_opt_refineout.py']), 'Exact inherited native contracts required')
    c['control']=mode;c['human']['x_coefficients']=selected['human_translation']
    c['object']['vertices_float32_m']=selected['object']['vertices_float32_m'];c['optimizer']['arms']=selected['optimizer']['arms']
    if mode=='zero_delegate':c['point_attachment']['reference']=selected['optimizer']['calibration_reference']
    return c


def control(args,timeout=15):
    result=subprocess.run(args,capture_output=True,timeout=timeout)
    if result.returncode or len(result.stdout)>32<<10:raise ValueError('Bounded runtime control failed')
    return result.stdout


def write(path,raw,mode=0o444):
    with path.open('xb') as stream:
        os.fchmod(stream.fileno(),mode);stream.write(raw);stream.flush();os.fsync(stream.fileno())


class LossDelegationObserver:
    """Transient return-object identity proof; never retains Python frames."""
    def __init__(self,native_loss,subclass_loss,point_loss):
        self.native,self.subclass,self.point=(f.__code__ for f in (native_loss,subclass_loss,point_loss))
        if len({self.native,self.subclass,self.point})!=3:raise ValueError('Distinct exact loss code objects required')
        self.steps=[];self.point_calls=0;self.active=None;self.returned=None;self.native_frame=None

    def clear(self):self.active=None;self.returned=None;self.native_frame=None

    @staticmethod
    def binding(frame):
        fields=frame.f_locals;value=(id(fields['self']),id(fields['indices']),fields['step'],fields['include_diagnostics'])
        if type(value[2]) is not int or type(value[3]) is not bool:raise ValueError('Typed native loss arguments required')
        return value

    def __call__(self,frame,event,value):
        code=frame.f_code
        if event=='call' and code is self.point:self.point_calls+=1;raise ValueError('Zero delegation called point arithmetic')
        if code not in (self.native,self.subclass) or event not in ('call','return'):return
        if event=='call':
            binding=self.binding(frame)
            if code is self.subclass:
                expected=[0,181]+list(range(301))
                if self.active is not None or len(self.steps)>=303 or binding[2]!=expected[len(self.steps)]:raise ValueError('Exact ordered delegation steps required')
                self.active=(id(frame),binding)
            elif self.active is None or frame.f_back.f_code is not self.subclass or id(frame.f_back)!=self.active[0] or binding!=self.active[1]:
                raise ValueError('Direct same-instance native delegation required')
            elif self.returned is not None or self.native_frame is not None:raise ValueError('Only one native loss per delegation')
            else:self.native_frame=id(frame)
        elif code is self.native:
            if self.active is None or id(frame)!=self.native_frame or id(frame.f_back)!=self.active[0] or frame.f_back.f_code is not self.subclass or self.binding(frame)!=self.active[1] or self.binding(frame.f_back)!=self.active[1] or type(value) is not tuple or len(value)!=2:raise ValueError('Actual native loss return required')
            self.returned=value;self.native_frame=None
        else:
            try:
                if self.active is None or id(frame)!=self.active[0] or self.binding(frame)!=self.active[1] or self.native_frame is not None or self.returned is None or type(value) is not tuple or len(value)!=2 or value[0] is not self.returned[0] or value[1] is not self.returned[1]:raise ValueError('Native total/metrics objects changed')
                self.steps.append(self.active[1][2])
            finally:self.clear()

    def report(self):
        return dict(pairs=len(self.steps),ordered_steps=self.steps.copy(),point_reprojection_calls=self.point_calls,
                    total_object_identity_verified=len(self.steps)==303,metrics_object_identity_verified=len(self.steps)==303,frames_retained=False)


def observed_call(callback,functions,delegate=None):
    """Observe delegated real Python entrypoints; no monkeypatch or substitute."""
    if sys.getprofile() is not None:raise ValueError('Unmodified native profiler state required')
    codes={f.__code__:name for name,f in functions.items()};counts={name:0 for name in functions}
    def profile(frame,event,arg):
        if event=='call' and frame.f_code in codes:counts[codes[frame.f_code]]+=1
        if delegate is not None:delegate(frame,event,arg)
    sys.setprofile(profile)
    try:return callback(),counts
    finally:
        sys.setprofile(None)
        if delegate is not None:delegate.clear()


def cleanup(rt,name,revision,path):
    """Separate stdout/stderr; daemon failure never means exact CID absence."""
    pin=rt.identity(path,65,readonly=False);raw=path.read_bytes()
    rt.require(re.fullmatch(b'[0-9a-f]{64}\n?',raw),'Owned exact CID required');cid=raw.decode().strip()
    def inspect():
        result=subprocess.run(['docker','container','inspect',cid],capture_output=True,timeout=10)
        rt.require(rt.identity(path,65,readonly=False)==pin,'CID changed');return result
    def absent(result):
        variants=[(s+cid).encode() for s in ('Error: No such container: ','Error response from daemon: No such container: ','Error: No such object: ','error: no such object: ')]
        return result.returncode==1 and result.stdout.strip() in (b'',b'[]') and result.stderr.strip() in variants
    result=inspect()
    if result.returncode:rt.require(absent(result),'Daemon failure is not owned absence')
    else:
        rows=rt.strict(result.stdout);rt.require(len(rows)==1 and rows[0]['Id']==cid and rows[0]['Name']=='/'+name
            and rows[0]['Image']==IMAGE and rows[0]['Config']['Labels'].get('world_reward.authored_pair.owner')==revision,'Foreign container refused')
        rt.require(subprocess.run(['docker','rm','-f',cid],capture_output=True,timeout=10).returncode==0,'Owned cleanup failed')
        rt.require(absent(inspect()),'Exact owned absence not verified')
    rt.require(rt.identity(path,65,readonly=False)==pin,'CID replaced');path.chmod(0o400)


def retain(path,writer):
    with path.open('xb') as stream:
        writer(stream);stream.flush();os.fsync(stream.fileno());os.fchmod(stream.fileno(),0o444)


def precomparison_record(report,record,persist):
    report.setdefault('retained_precomparison',{})[report['phase']]=record;persist()


def parameters(np,c,body_converter,hand_converter,dims):
    from cari_converter import PARAMETER_DIMS
    if dims!=PARAMETER_DIMS:raise ValueError('Exact primary MHR parameter ABI required')
    values={k:np.zeros((3,n),np.float32) for k,n in dims.items()};index=np.arange(3)
    values['mhr_global_rot6d'][:]=c['human']['root_rot6d_interleaved'];values['mhr_trans'][:,0]=(.01*index+.005*index**2).astype(np.float32);values['mhr_trans'][:,2]=4
    if c.get('control') in ('native_repeat','zero_delegate'):
        coefficients=c['human']['x_coefficients'];values['mhr_trans'][:,0]=(coefficients['linear_x_m']*index+coefficients['quadratic_x_m']*index**2).astype(np.float32)
    values['mhr_body_pose_cont'][:]=body_converter(np.zeros((3,133),np.float32))
    hand=hand_converter(np.zeros((3,27),np.float32));values['mhr_hand'][:]=np.concatenate((hand,hand),axis=1)
    return values


def inventory(rt,directory):
    rt.canonical(directory);rows={}
    for p in sorted(directory.rglob('*')):
        rt.canonical(p)
        if p.is_dir():continue
        if p.suffix=='.py':rows[str(p.relative_to(directory))]=rt.identity(p,2<<20,readonly=False,empty=True)
    rt.require(rows,'Complete actual Python source required');return rows


def archive_model_pins(rt,root,code):
    """Read independently pinned first manifest only, not the 20GB payload."""
    pins=rt.strict((code/ARCHIVE_PINS).read_bytes());path=root/pins['archive_pathrelative']
    rt.require(pins['schema']=='world_reward.frontend_asset_archive.pins.v1' and pins['independent_external_archive_verification_completed'] is True
        and path==root/'results'/('frontend-asset-archive-'+pins['producer_revision'])/'archive.tar','Original independent archive proof required')
    before=rt.canonical(path).lstat();rt.require(stat.S_ISREG(before.st_mode) and before.st_nlink==1
        and before.st_size==pins['archive_identity']['bytes'] and not before.st_mode&0o222,'Original sealed archive required')
    with tarfile.open(path,'r|') as tar:
        member=next(iter(tar));rt.require(member.name=='world-reward-frontend-assets-manifest.json' and member.isfile()
            and member.size==pins['manifest_identity']['bytes']<=2<<20,'First bounded original manifest required')
        raw=tar.extractfile(member).read(member.size+1)
    rt.require({'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}==pins['manifest_identity'],'Independent manifest differs')
    after=path.lstat();rt.require(all(getattr(before,k)==getattr(after,k) for k in ('st_dev','st_ino','st_size','st_mode','st_mtime_ns','st_ctime_ns')),'Archive changed')
    manifest=rt.strict(raw);rt.require(manifest['original_root']==str(root) and manifest['images_exported'] is False,'Original asset-only manifest required')
    names=[BODY+'/'+n for n in ('model.ckpt','model_config.yaml','assets/mhr_model.pt','LICENSE')]+['results/weights-acquisition.json']
    selected={}
    for name in names:
        row=manifest['entries'][name];rt.require(row['type']=='file','Original regular model/receipt required')
        pin={k:row[k] for k in ('bytes','sha256')};rt.require(rt.identity(root/name,7_000_000_000,readonly=False)==pin,'Independent original asset differs')
        selected[str(root/name)]=pin
    return selected,pins['manifest_identity']


def prerequisites(rt,root,code,revision,mode=None):
    c=protocol(rt,code,mode);source=rt.source(root,code,revision,ENTRY,helpers(mode))
    vendor=root/'vendor/video_to_data'
    result=subprocess.run(['git','-c','safe.directory='+str(vendor),'-C',str(vendor),'rev-parse','HEAD'],capture_output=True,timeout=10)
    rt.require(result.returncode==0 and result.stdout.decode().strip()==c['upstream_revision'],'Actual pinned upstream checkout required')
    result=subprocess.run(['git','-c','safe.directory='+str(vendor),'-C',str(vendor),'status','--porcelain','--untracked-files=all'],capture_output=True,timeout=10)
    rt.require(result.returncode==0 and not result.stdout.strip(),'Unmodified upstream source required')
    frozen,manifest=archive_model_pins(rt,root,code)
    for name,pin in c['primary_sources'].items():
        p=vendor/NATIVE/name;rt.require(rt.identity(p,2<<20,readonly=False)==pin,'Pinned primary native source differs');frozen[str(p)]=pin
    p=vendor/BODY_SOURCE/'models/heads/mhr_head.py';pin={k:c['body_head_source'][k] for k in ('bytes','sha256')}
    rt.require(rt.identity(p,2<<20,readonly=False)==pin,'Pinned actual Body head differs');frozen[str(p)]=pin
    for name,pin in c['refinement_assets'].items():
        p=root/'weights/cari4d/refinement'/name;rt.require(rt.identity(p,readonly=False)==pin,'Original contact/collision asset differs');frozen[str(p)]=pin
    rt.require(frozen[str(root/BODY/'assets/mhr_model.pt')]['sha256']==c['model_prerequisites']['native_model_sha256'],'Actual native MHR model differs')
    receipt=rt.strict((root/'results/weights-acquisition.json').read_bytes())
    rows=[r for r in receipt['assets'] if r['repo_id']=='facebook/sam-3d-body-dinov3']
    rt.require(len(rows)==1 and rows[0]['revision']=='11aaa346c7204874a1cbafe3d39a979080b2c55a' and Path(rows[0]['path'])==root/BODY,'Actual original Body acquisition required')
    return dict(source_binding=source,manifest_identity=manifest,frozen=frozen,
        native_source=inventory(rt,vendor/NATIVE),body_source=inventory(rt,vendor/BODY_SOURCE))


def check_native_sources(rt,root,proof):
    vendor=root/'vendor/video_to_data'
    rt.require(inventory(rt,vendor/NATIVE)==proof['native_source'] and inventory(rt,vendor/BODY_SOURCE)==proof['body_source']
        and inventory(rt,Path('/workspace/v2d_sam3d_body/lib/sam_3d_body'))==proof['body_source'],'Actual complete installed native/Body source differs')
    rt.require(all(rt.identity(Path(p),7_000_000_000,readonly=False)==pin for p,pin in proof['frozen'].items()),'Assets changed')


def manufacture(np,torch,layer,optimizer,spec,c,out):
    from lib_mhr.body_pose import compact_model_params_to_cont_body_np,compact_model_params_to_cont_hand_np
    from lib_mhr.postopt_crop import build_postopt_crop,stack_postopt_crops
    from lib_mhr.hand_surface_contact import MHR_HAND_ORDER
    from lib_mhr.schema import MHR_PARAM_DIMS
    import Utils
    import nvdiffrast.torch as dr
    import cari_full_refine as full
    t=3;params=parameters(np,c,compact_model_params_to_cont_body_np,compact_model_params_to_cont_hand_np,MHR_PARAM_DIMS);move=params['mhr_trans'][:,0]
    with torch.no_grad():decoded=layer.mhr_forward({k:torch.from_numpy(v).cuda() for k,v in params.items()})
    human=decoded.vertices;faces=layer.mesh_faces(device='cuda')
    if MHR_HAND_ORDER!=('left_hand','right_hand') or any(x.dtype!=torch.float32 or not torch.isfinite(x).all() for x in (human,decoded.joints,decoded.keypoints)) or not bool((layer.neutral_height({k:torch.from_numpy(v).cuda() for k,v in params.items()})>0).all()):raise ValueError('Actual neutral geometry invalid')
    centre=human[0,torch.as_tensor(spec.vertex_indices[0].copy(),device='cuda',dtype=torch.long)].mean(0).detach().cpu().numpy()+np.array([0,0,-.045])
    vertices=np.asarray(c['object']['vertices_float32_m'],np.float32);triangles=np.asarray(c['object']['faces_int64_outward'],np.int64)
    mesh=out/'authored_object.obj'
    with mesh.open('x') as f:
        for v in vertices:f.write('v '+' '.join(format(float(x),'.17g') for x in v)+'\n')
        for frow in triangles:f.write('f '+' '.join(str(int(x)+1) for x in frow)+'\n')
    mesh.chmod(0o444);native_v,native_f=optimizer._load_object_vertices(mesh)
    if native_v.dtype!=np.float32 or native_f.dtype!=np.int64 or not np.array_equal(native_v,vertices) or not np.array_equal(native_f,triangles):raise ValueError('Authored native mesh load changed arrays')
    pose=np.broadcast_to(np.eye(4,dtype=np.float32),(t,4,4)).copy();pose[:,:3,3]=centre;pose[:,0,3]+=move
    K=np.asarray(c['cohort']['K'],np.float64);Ks=np.broadcast_to(K,(t,3,3)).copy();context=dr.RasterizeCudaContext()
    def render(v,f,color):
        return Utils.nvdiff_color_depth_render(Ks,context,dict(pos=v[0],faces=f.to(torch.int32).contiguous(),vertex_color=torch.tensor(color,device='cuda',dtype=torch.float32).expand(v.shape[1],3).contiguous()),(480,640),v.contiguous())
    hc,hz,_=render(human,faces,[.7,.6,.5])
    ov=torch.tensor(native_v,device='cuda')[None]+torch.tensor(pose[:,:3,3],device='cuda')[:,None]
    oc,oz,_=render(ov,torch.tensor(native_f,device='cuda'),[.2,.8,.3])
    hvalid,ovalid=hz>0,oz>0
    if torch.any(hvalid&ovalid&(hz==oz)):raise ValueError('Equal-depth authored visibility ambiguous')
    obj=ovalid&(~hvalid|(oz<hz));person=hvalid&(~ovalid|(hz<oz))
    rgb=torch.where(obj[...,None],oc,torch.where(person[...,None],hc,torch.zeros_like(hc)))
    depth=torch.where(obj,oz,torch.where(person,hz,torch.zeros_like(hz)))
    person=person.cpu().numpy();obj=obj.cpu().numpy();rgb=rgb.cpu().numpy();depth=depth.cpu().numpy()
    crops=stack_postopt_crops([build_postopt_crop(a,b,K) for a,b in zip(person,obj)],K)
    observations=dict(human_mask=person,object_mask=obj,**crops)
    if rgb.dtype!=np.float32 or depth.dtype!=np.float32 or not np.isfinite(rgb).all() or not np.isfinite(depth).all() or np.any(depth<0):raise ValueError('Finite original rendered RGB/Z required')
    pr={**params,'pose_abs':pose,'contact_logits':np.asarray(c['bundle']['contact_logits_authored'],np.float32)}
    metadata=dict(c['bundle']['metadata']);metadata['object_mesh']=str(mesh)
    for key in ('object_pose_storage_to_training_transform','object_mesh_to_training_transform'):metadata[key]=np.eye(4,dtype=np.float32)
    source=dict(schema=c['bundle']['schema'],gt={},frames=c['cohort']['frame_names'],pr=pr,pr_initial=copy.deepcopy(pr),observations=observations,metadata=metadata)
    full.validate_source_bundle(source,mesh,t)
    retain(out/'authored_raw.npz',lambda stream:np.savez(stream,rgb=rgb,depth=depth,depth_valid=depth>0,K=K,vertices=native_v,faces=native_f,pose=pose,
        decoded_vertices=human.detach().cpu().numpy(),decoded_joints=decoded.joints.detach().cpu().numpy(),decoded_keypoints=decoded.keypoints.detach().cpu().numpy(),human_faces=faces.cpu().numpy(),**params,**{k:v for k,v in observations.items() if isinstance(v,np.ndarray)})
    )
    retain(out/'authored_source.pth',lambda stream:torch.save(source,stream));return source,native_v,native_f,depth


def paired_native(np,torch,layer,optimizer,source,vertices,faces,depth,c,out,report,persist,check):
    import cari_full_refine as full
    import joint_point_native_qualify as pair
    from world_reward.point_surface_queries import canonical_mask_quantile_queries,MaskQueryError
    from world_reward import joint_point_objective as op
    import Utils
    from kaolin.ops.mesh import check_sign
    from kaolin.metrics.trianglemesh import point_to_mesh_distance
    repeat=c.get('control')=='native_repeat';single=c.get('control')=='zero_delegate';arms=profile(c.get('control'))[2]
    config=None if repeat else op.PointObjectiveConfig(1.,0.,c['point_attachment']['reference']);cfg=optimizer.MHRParityPostOptConfig(
        penetration_collision_proxy_path=str(ROOT/'weights/cari4d/refinement/mhr_collision_proxy_4000v.npz'),
        hand_surface_spec_path=str(ROOT/'weights/cari4d/refinement/mhr_hand_surface_spec.npz'),report_every=100,checkpoint_path=None)
    fingerprint=full.fingerprint(source);extension=None;delegate=None;retained={p.name:runtime(Path(os.environ['WR_CODE'])).identity(p) for p in out.iterdir() if p.name!='report.json'}
    functions=dict(contact=optimizer.MHRParityPostOptimizer._contact_loss,render=Utils.nvdiff_color_depth_render,
        penetration=optimizer.object_inside_human_penetration_loss,kaolin_sign=check_sign,kaolin_distance=point_to_mesh_distance)
    report.update(config=asdict(cfg),point_config=None if repeat else asdict(config),raw_source_sha256=fingerprint,raw_artifacts=retained)
    if repeat:report.update(control='native_repeat',constructor_kind='original_both',arm_names=list(arms),point_optimizer_factory_calls=0,point_evidence_bound=False)
    if single:
        from world_reward.fixed_shape_point_pose import PointTrackEvidence
        from world_reward.joint_point_evidence import bind_joint_point_evidence
        pose=source['pr']['pose_abs']
        if pose.dtype!=np.float32 or pose.shape!=(3,4,4) or not pose.flags.c_contiguous:raise ValueError('Exact native F32 initial pose required before constructor')
        r0,t0=pose[0,:3,:3].copy(),pose[0,:3,3].copy()
        try:selected=canonical_mask_quantile_queries(vertices,faces,r0,t0,np.asarray(c['cohort']['K'],np.float64),source['observations']['object_mask'][0],np.isfinite(depth[0])&(depth[0]>0),image_width=640,image_height=480);q=selected.queries
        except MaskQueryError as error:report['query_diagnostics']=error.diagnostics.scalar_report();raise
        report['query_diagnostics']=selected.diagnostics.scalar_report();n=len(q.face_indices);timeline=np.arange(3,dtype=np.int64);xy=q.query_points[:,[2,1]]*np.array([256/640,256/480])
        tracks=PointTrackEvidence(timeline,timeline,np.arange(n,dtype=np.int64),q.query_points,np.broadcast_to(xy,(3,n,2)).copy(),np.zeros((3,n),bool))
        evidence=bind_joint_point_evidence(q,tracks,native_vertices=vertices,native_faces=faces,K=np.asarray(c['cohort']['K'],np.float64),image_size=(480,640),frame_index=timeline,source_frame_ids=timeline,native_frame_names=tuple(source['frames']),source_references=(c['point_attachment']['reference'],'NUMERICALCONTROL_NOTTRACKER'))
        extension=op.native_point_optimizer_class(optimizer,evidence,config);delegate=LossDelegationObserver(optimizer.MHRParityPostOptimizer.loss,extension.loss,op.point_reprojection_loss)
        retain(out/'point_evidence.npz',lambda stream:np.savez(stream,**{k:getattr(evidence,k) for k in evidence.__dataclass_fields__ if isinstance(getattr(evidence,k),np.ndarray)}))
        retained['point_evidence.npz']=runtime(Path(os.environ['WR_CODE'])).identity(out/'point_evidence.npz')
        report.update(control='zero_delegate',constructor_kind='one_real_weight_zero_point_subclass',arm_names=list(arms),point_optimizer_factory_calls=1,point_evidence_bound=True,point_evidence_sha256=evidence.evidence_sha256,bit_parity_qualified=False,stochastic_lifecycle_qualified=False)
    def reset():random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
    def construct(arm):
        nonlocal extension
        check()
        if full.fingerprint(source)!=fingerprint:raise ValueError('Authored source changed')
        instance=optimizer.MHRParityPostOptimizer(source,vertices,faces,cfg,mhr_layer=layer) if not single and (repeat or arm==arms[0]) else extension(source,vertices,faces,cfg,mhr_layer=layer)
        if not bool((instance.contact_mask[:,0]>0).all()):raise ValueError('Effective authored left contact absent')
        if single:
            r,t,_,_=instance._object_state(torch.arange(3,device='cuda'),include_surface=False)
            actual_r,actual_t=r[0].detach().cpu().numpy(),t[0].detach().cpu().numpy()
            if actual_r.dtype!=r0.dtype or actual_t.dtype!=t0.dtype or actual_r.tobytes()!=r0.tobytes() or actual_t.tobytes()!=t0.tobytes():raise ValueError('Preconstructor query pose differs from actual native initial state')
            report['query_pose_verified_after_constructor']=True
        elif arm==arms[0]:
            indices=torch.arange(3,device='cuda');r,t,_,_=instance._object_state(indices,include_surface=False)
            try:selected=canonical_mask_quantile_queries(vertices,faces,r[0].detach().cpu().numpy(),t[0].detach().cpu().numpy(),np.asarray(c['cohort']['K'],np.float64),source['observations']['object_mask'][0],np.isfinite(depth[0])&(depth[0]>0),image_width=640,image_height=480);q=selected.queries
            except MaskQueryError as error:report['query_diagnostics']=error.diagnostics.scalar_report();raise
            report['query_diagnostics']=selected.diagnostics.scalar_report()
            if repeat:
                retain(out/'quantile_diagnostic.npz',lambda stream:np.savez(stream,**vars(q),**{k:v for k,v in vars(selected.diagnostics).items() if isinstance(v,np.ndarray)}))
                return instance
            from world_reward.fixed_shape_point_pose import PointTrackEvidence
            from world_reward.joint_point_evidence import bind_joint_point_evidence
            n=len(q.face_indices);timeline=np.arange(3,dtype=np.int64);xy=q.query_points[:,[2,1]]*np.array([256/640,256/480])
            tracks=PointTrackEvidence(timeline,timeline,np.arange(n,dtype=np.int64),q.query_points,np.broadcast_to(xy,(3,n,2)).copy(),np.zeros((3,n),bool))
            evidence=bind_joint_point_evidence(q,tracks,native_vertices=vertices,native_faces=faces,K=np.asarray(c['cohort']['K'],np.float64),image_size=(480,640),frame_index=timeline,source_frame_ids=timeline,native_frame_names=tuple(source['frames']),source_references=(c['point_attachment']['reference'],'NUMERICALCONTROL_NOTTRACKER'))
            extension=op.native_point_optimizer_class(optimizer,evidence,config);report['point_evidence_sha256']=evidence.evidence_sha256
            retain(out/'point_evidence.npz',lambda stream:np.savez(stream,**{k:getattr(evidence,k) for k in evidence.__dataclass_fields__ if isinstance(getattr(evidence,k),np.ndarray)}))
        return instance
    def initial(instance):
        state={k:v for k,v in vars(instance).items() if torch.is_tensor(v) or isinstance(v,np.ndarray)};state['params_fixed']=instance.params_fixed
        retain(out/f"{report['phase']}_initial.pth",lambda stream:torch.save(dict(state=state,optimizer=instance.optimizer.state_dict(),scheduler=instance.scheduler.state_dict()),stream))
        record=dict(state_sha256=full.fingerprint(state),optimizer_sha256=full.fingerprint(instance.optimizer.state_dict()),scheduler_sha256=full.fingerprint(instance.scheduler.state_dict()))
        if repeat or single:precomparison_record(report,record,persist)
        return record
    def probe(instance,step):
        check();indices=torch.arange(3,device='cuda');instance.optimizer.zero_grad(set_to_none=True)
        (total,metrics),counts=observed_call(lambda:instance.loss(indices,step,include_diagnostics=True),functions,delegate);total.backward();torch.cuda.synchronize()
        gradients=[p.grad for g in instance.optimizer.param_groups for p in g['params']]
        if not torch.isfinite(total) or any(g is not None and not torch.isfinite(g).all() for g in gradients):raise ValueError('Nonfinite real native probe')
        if not counts['contact'] or not counts['render'] or step==181 and any(not n for n in counts.values()):raise ValueError('Real contact/render/Kaolin path not executed')
        retain(out/f"{report['phase']}.pth",lambda stream:torch.save(dict(total=total,metrics=metrics,gradients=gradients),stream))
        record=dict(step=step,loss_sha256=full.fingerprint(total),metrics_sha256=full.fingerprint(metrics),gradients=full.fingerprint(gradients),native_calls=counts)
        if repeat or single:precomparison_record(report,record,persist)
        instance.optimizer.zero_grad(set_to_none=True);return record
    def validate(result,arm):
        retain(out/(arm+'_result.pth'),lambda stream:torch.save(result,stream))
        if full.fingerprint(torch.load(out/(arm+'_result.pth'),map_location='cpu',weights_only=False))!=full.fingerprint(result):raise ValueError('Full native saved result reload differs')
        if not repeat and not single and arm=='B_point_weight_zero':
            extra=result['postopt'].pop('point_objective')
            if extra['config']!=asdict(config) or any(extra['after_initializer_support']):raise ValueError('Zero weight evidence changed')
        if single and result['postopt']['point_objective']!=dict(config=asdict(config),evidence_sha256=evidence.evidence_sha256,
            mesh_sha256=evidence.mesh_sha256,after_initializer_support=evidence.support_counts().tolist(),native_source_sha256=op.NATIVE_SOURCE_SHA256,
            object_rotation_fixed=True,track_equal=True,support_is_confidence=False,calibration_reference_authenticated=False,calibrated_probabilities=False,quality_verified=False,adoption=False):raise ValueError('Full zero-delegate point metadata differs')
        metadata=full.validate_result(source,result,3);check()
        if full.fingerprint(source)!=fingerprint:raise ValueError('Raw source changed')
        return dict(result_sha256=full.fingerprint(result),metadata_sha256=full.fingerprint(metadata))
    if single:
        arm=arms[0];reset();report['phase']=arm+'_constructor';report['constructor_attempts']+=1;persist();instance=construct(arm);report['constructor_returns']+=1
        observed=dict(initial_state=initial(instance),probes=[])
        try:
            for step in (0,181):
                report['phase']=arm+f'_loss_gradient_{step}';report['probe_attempts']+=1;persist();observed['probes'].append(probe(instance,step));report['probe_returns']+=1
            report['phase']=arm+'_301_updates';report['run_attempts']+=1;persist();result,counts=observed_call(lambda:instance.run(),functions,delegate);report['run_returns']+=1
            checked=validate(result,arm);report[arm]=dict(initial=observed,result=checked,native_calls=counts)
            if delegate.steps!=[0,181]+list(range(301)) or delegate.point_calls:raise ValueError('Complete303 ordered zero-loss delegations required')
            report.update(semantic_delegation_verified=True,actual_native_updates_total=301,actual_native_updates_per_arm=301)
        finally:report['loss_delegation']=delegate.report();delegate.clear();persist()
        del instance,result;gc.collect();torch.cuda.empty_cache()
    else:pair.paired_execution(construct,probe,lambda instance:instance.run(),reset,lambda:(gc.collect(),torch.cuda.empty_cache()),initial,validate,report,persist,arm_names=arms)
    rt=runtime(Path(os.environ['WR_CODE']));rt.require(all(rt.identity(out/n)==pin for n,pin in retained.items()),'Frozen raw control changed during pair')
    rt.require(full.fingerprint(torch.load(out/'authored_source.pth',map_location='cpu',weights_only=False))==fingerprint,'Saved raw control reload differs')


def native(rt,root,code,revision,out,proof,persist,report,mode=None):
    c=protocol(rt,code,mode);start=proof['host_start_monotonic'];deadline=start+1380
    rt.require(proof.get('control')==mode,'Host/native control selection differs')
    check=lambda:rt.require(time.monotonic()<deadline,'Inclusive authored deadline')
    rt.require(sys.platform=='linux' and os.geteuid()==1000 and {p.name for p in Path('/sys/class/net').iterdir()}=={'lo'}
        and os.environ.get('WR_IMAGE_ID')==IMAGE and not any(out.iterdir()),'Actual fresh offline native GPU process required')
    check_native_sources(rt,root,proof);rt.require(rt.source(root,code,revision,ENTRY,helpers(mode))==proof['source_binding'],'Own native source differs before decode');check()
    if 'torch' in sys.modules:raise ValueError('Fresh native Torch process required')
    import numpy as np
    import torch
    import cari_full_refine as full
    from lib_mhr.mhr_layer import MHRLayer
    from lib_mhr.hand_surface_contact import load_mhr_hand_surface_spec
    from learning.training import mhr_opt_refineout as optimizer
    from world_reward import joint_point_objective as op
    rt.require(Path(sys.modules[MHRLayer.__module__].__file__).resolve()==root/'vendor/video_to_data'/NATIVE/'lib_mhr/mhr_layer.py'
        and Path(optimizer.__file__).resolve()==root/'vendor/video_to_data'/NATIVE/'learning/training/mhr_opt_refineout.py','Actual pinned native imports required')
    rt.require(torch.cuda.is_available() and str(torch.__version__)=='2.5.1+cu124' and torch.version.cuda=='12.4'
        and not torch.are_deterministic_algorithms_enabled(),'Original native CUDA policy required')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0);op._native_binding(optimizer)
    layer=MHRLayer.from_mhr_assets(mhr_assets_root=Path('/workspace/v2d_sam3d_body/lib'),checkpoint_path=root/BODY/'model.ckpt',buffer_path=out/'never_compact_buffer.pt',mhr_model_path=root/BODY/'assets/mhr_model.pt',device='cuda')
    spec=load_mhr_hand_surface_spec(root/'weights/cari4d/refinement/mhr_hand_surface_spec.npz',faces=layer.mesh_faces(device='cuda').cpu().numpy())
    rt.require(spec.mhr_model_sha256==c['model_prerequisites']['native_model_sha256'],'Canonical hand spec model differs')
    report['phase']='manufacture';source,v,f,z=manufacture(np,torch,layer,optimizer,spec,c,out)
    report['decoder_identity']=layer.decoder_identity();report['manufacture_seconds']=time.monotonic()-start
    rt.require(report['manufacture_seconds']<=180,'Inclusive authored manufacture180 exceeded');persist()
    pair_start=time.monotonic();deadline=min(deadline,pair_start+1200);signal.setitimer(signal.ITIMER_REAL,max(.001,deadline-time.monotonic()));pair_check=check
    paired_native(np,torch,layer,optimizer,source,v,f,z,c,out,report,persist,pair_check)
    pair_check();check_native_sources(rt,root,proof);rt.require(rt.source(root,code,revision,ENTRY,helpers(mode))==proof['source_binding'],'Own source changed')
    report.update(status='pass',phase='complete',source_inputs_assets_rehashed_after=True,elapsed_seconds=time.monotonic()-start)


def host(rt,root,code,revision,mode=None):
    start=time.monotonic();rt.require(sys.platform=='linux' and os.geteuid()==0 and root==ROOT
        and Path(__file__).resolve()==code/HELPERS[0] and os.environ['DOCKER_HOST']=='unix://'+str(root/'docker.sock'),'Actual private immutable VM01 host required')
    pin,_,arms=profile(mode);proof=prerequisites(rt,root,code,revision,mode);proof['host_start_monotonic']=start
    if mode is not None:proof['control']=mode
    out=output_path(root,revision,mode);rt.canonical(out);rt.require(not out.exists(),'Fresh control only')
    label='zero-delegate' if mode=='zero_delegate' else 'native-repeat' if mode else 'authored'
    name='world-reward-joint-point-'+label+'-'+revision;rt.require(not control(['docker','ps','-aq','--filter','name=^/'+name+'$']).strip(),'Owned name occupied')
    projection=rt.strict(control(['docker','image','inspect',IMAGE,'--format','{"Id":{{json .Id}},"Architecture":{{json .Architecture}},"Os":{{json .Os}},"RootFS":{{json .RootFS}}}']))
    rt.require(projection['Id']==IMAGE and projection['Architecture']=='amd64' and projection['Os']=='linux' and projection['RootFS']['Type']=='layers','Actual native image required')
    out.mkdir(mode=0o755);out.chmod(0o755);pred=out/'native';pred.mkdir(mode=0o700);os.chown(pred,1000,1000)
    write(out/'proof.json',(json.dumps(proof,sort_keys=True)+'\n').encode(),0o444);proof_pin=rt.identity(out/'proof.json')
    lock=rt.canonical(root/'jobs/.world-reward-h100.lock');s=lock.lstat();rt.require(stat.S_ISREG(s.st_mode) and s.st_nlink==1,'Existing cooperative lock required')
    fd=os.open(lock,os.O_RDONLY|os.O_NOFOLLOW);os.dup2(fd,9)
    if fd!=9:os.close(fd)
    report=dict(stage='joint_point_zero_delegate_host_seal_v1' if mode=='zero_delegate' else 'joint_point_native_repeat_host_seal_v1' if mode else 'joint_point_authored_host_seal_v1',status='fail',producer_revision=revision,source_binding=proof['source_binding'],protocol_identity=pin,
        image=projection,runtime_qualified=False,quality_verified=False,adoption=False,challenge_inputs_used=False,gt_used=False)
    if mode is not None:report['control']=mode
    cid=out/'.container.cid';failure=None
    def expired(*_):raise TimeoutError('Inclusive authored runtime deadline')
    old_alarm=signal.signal(signal.SIGALRM,expired);old_term=signal.signal(signal.SIGTERM,expired);signal.setitimer(signal.ITIMER_REAL,max(.001,start+1380-time.monotonic()))
    try:
        subprocess.run(['flock','--nonblock','9'],check=True,pass_fds=(9,),timeout=5,capture_output=True)
        rt.require((os.fstat(9).st_dev,os.fstat(9).st_ino)==(lock.lstat().st_dev,lock.lstat().st_ino)==(s.st_dev,s.st_ino),'Lock changed')
        rt.require(not control(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits']).strip(),'GPU compute active')
        mounts=[code.parent,root/'vendor/video_to_data'/NATIVE,root/'vendor/video_to_data'/BODY_SOURCE,*map(Path,proof['frozen'])]
        rt.require({p.name for p in code.parent.iterdir()}=={'code','revision','source-sha256'},'Exact own snapshot parent required')
        args=['docker','run','--name',name,'--cidfile',str(cid),'--label','world_reward.authored_pair.owner='+revision,'--gpus','all','--network','none','--read-only','--user','1000:1000','--cap-drop','ALL','--security-opt','no-new-privileges','--cpus','4','--memory','64g','--tmpfs','/tmp:rw,nosuid,size=2g']
        for p in sorted(set(mounts)):args+=['--mount',f'type=bind,src={p},dst={p},readonly']
        args+=['--mount',f'type=bind,src={out}/proof.json,dst=/opt/authored-proof.json,readonly','--mount',f'type=bind,src={pred},dst={pred}',
            '--entrypoint','/usr/bin/env',IMAGE,'-i','PATH=/opt/conda/bin:/usr/bin:/bin','HOME=/tmp','HF_HUB_OFFLINE=1','TRANSFORMERS_OFFLINE=1','MOMENTUM_ENABLED=0','OMP_NUM_THREADS=4','PYTHONDONTWRITEBYTECODE=1',
            'PYTHONPATH='+str(code/'infra')+':'+str(code/'src')+':'+str(root/'vendor/video_to_data'/NATIVE)+':/workspace/v2d_sam3d_body/lib',
            'WR_ROOT='+str(root),'WR_CODE='+str(code),'WR_CODE_REVISION='+revision,'WR_IMAGE_ID='+IMAGE,'python','-B',str(code/HELPERS[0]),'--native',str(proof_pin['bytes']),proof_pin['sha256']]
        if mode is not None:args+=['--control',mode]
        with (out/'native.log').open('xb') as log:
            os.fchmod(log.fileno(),0o444);result=subprocess.run(args,stdout=log,stderr=log,timeout=max(.001,1380-(time.monotonic()-start)))
        rt.require(result.returncode==0,'Actual native pair failed')
        native_report=rt.strict((pred/'report.json').read_bytes());validate_native(rt,native_report,proof,mode)
        expected={'authored_object.obj','authored_raw.npz','authored_source.pth','quantile_diagnostic.npz' if mode=='native_repeat' else 'point_evidence.npz','report.json'}|{arm+s for arm in arms for s in ('_constructor_initial.pth','_loss_gradient_0.pth','_loss_gradient_181.pth','_result.pth')}
        rt.require({p.name for p in pred.iterdir()}==expected and pred.stat().st_mode&0o777==0o555,'Exact complete retained native control required')
        report['native_report_identity']=rt.identity(pred/'report.json');report['outputs']={p.name:rt.identity(p,1_000_000_000) for p in pred.iterdir()}
    except BaseException as error:failure=error;report['error_type']=type(error).__name__
    finally:
        try:
            signal.setitimer(signal.ITIMER_REAL,0);cleanup(rt,name,revision,cid);report['owned_cleanup_verified']=True
        except BaseException as error:failure=failure or error;report['cleanup_error_type']=type(error).__name__
        os.close(9);signal.setitimer(signal.ITIMER_REAL,max(.001,start+1380-time.monotonic()))
        try:
            after=prerequisites(rt,root,code,revision,mode);rt.require(after=={k:v for k,v in proof.items() if k not in ('host_start_monotonic','control')},'Source/assets postcheck differs')
            rt.require(time.monotonic()-start<=1380,'Inclusive host sealing deadline');report['source_rehashed_after']=True
        except BaseException as error:failure=failure or error;report['post_error_type']=type(error).__name__
        report.update(status='fail' if failure else 'pass',elapsed_seconds=time.monotonic()-start)
        write(out/'report.json',(json.dumps(report,sort_keys=True,allow_nan=False)+'\n').encode(),0o444);out.chmod(0o555)
        try:rt.require(time.monotonic()-start<=1380,'Inclusive host final seal exceeded')
        finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old_alarm);signal.signal(signal.SIGTERM,old_term)
    if failure:raise ValueError('Authored runtime qualification failed closed')


def validate_native(rt,r,proof,mode=None):
    pin,_,arms=profile(mode)
    required=dict(status='pass',phase='complete',frames=3,constructor_attempts=2,constructor_returns=2,probe_attempts=4,probe_returns=4,run_attempts=2,run_returns=2,
        actual_native_updates_total=602,actual_native_updates_per_arm=301,exact_full_result_history_parity=True,exact_initial_state_loss_gradient_parity=True,
        manufactured_not_inferred=True,tracker_executed=False,positive_weight_executed=False,quality_verified=False,adoption=False,ground_truth_used=False,challenge_inputs_used=False,source_inputs_assets_rehashed_after=True)
    if mode=='zero_delegate':
        for k in ('exact_full_result_history_parity','exact_initial_state_loss_gradient_parity'):required.pop(k);rt.require(k not in r,'Delegation is not independent bit parity')
        required.update(constructor_attempts=1,constructor_returns=1,probe_attempts=2,probe_returns=2,run_attempts=1,run_returns=1,actual_native_updates_total=301,
            bit_parity_qualified=False,stochastic_lifecycle_qualified=False,semantic_delegation_verified=True,point_optimizer_factory_calls=1,point_evidence_bound=True,query_pose_verified_after_constructor=True)
        rt.require(r.get('control')==mode and r.get('constructor_kind')=='one_real_weight_zero_point_subclass' and r.get('arm_names')==list(arms)
            and r.get('point_config')==dict(residual_scale_256_px=1.,weight=0.,calibration_reference='runtime-only-zero-delegation-not-calibration'),'One native zero-delegate subclass required')
        d=r['loss_delegation'];rt.require(type(d['pairs']) is int and d['pairs']==303 and type(d['point_reprojection_calls']) is int and d['point_reprojection_calls']==0
            and type(d['ordered_steps']) is list and all(type(s) is int for s in d['ordered_steps']) and d['ordered_steps']==[0,181]+list(range(301))
            and d['total_object_identity_verified'] is True and d['metrics_object_identity_verified'] is True and d['frames_retained'] is False,'Complete semantic delegation required')
    rt.require(all(type(r.get(k)) is type(v) and r[k]==v for k,v in required.items()) and r['source_binding']==proof['source_binding'] and r['protocol_identity']==pin and proof.get('control')==mode,'Complete actual authored pair proof required')
    if mode=='native_repeat':
        rt.require(r.get('control')==mode and r.get('constructor_kind')=='original_both' and r.get('arm_names')==list(arms)
            and type(r.get('point_optimizer_factory_calls')) is int and r['point_optimizer_factory_calls']==0 and r.get('point_evidence_bound') is False and r.get('point_config') is None,'Original-only native repeat proof required')
    for arm in arms:
        rows=r[arm]['initial']['probes'];rt.require(type(rows) is list and [p['step'] for p in rows]==[0,181],'Both exact probe steps required')
        for row in rows:
            rt.require(type(row['step']) is int and set(row['native_calls'])=={'contact','render','penetration','kaolin_sign','kaolin_distance'} and all(type(n) is int and n>=0 for n in row['native_calls'].values())
                and row['native_calls']['contact']>0 and row['native_calls']['render']>0
                and (row['step']!=181 or all(n>0 for n in row['native_calls'].values())),'Real contact/render/Kaolin proof required')


def arguments(argv=None):
    parser=argparse.ArgumentParser(allow_abbrev=False);parser.add_argument('--native',nargs=2);parser.add_argument('--control',action='append',choices=['native_repeat','zero_delegate']);args=parser.parse_args(argv)
    if args.control and len(args.control)!=1:parser.error('Control selected exactly once')
    args.control=args.control[0] if args.control else None;return args


def main(argv=None):
    args=arguments(argv);pin,_,_=profile(args.control)
    root=Path(os.environ['WR_ROOT']);code=Path(os.environ['WR_CODE']);revision=os.environ['WR_CODE_REVISION'];rt=runtime(code)
    if not args.native:return host(rt,root,code,revision,args.control)
    proof=rt.pinned(Path('/opt/authored-proof.json'),dict(bytes=int(args.native[0]),sha256=args.native[1]),4<<20)
    rt.require(proof.get('control')==args.control,'Host/native control selection differs')
    out=output_path(root,revision,args.control)/'native'
    report=dict(stage='joint_point_zero_delegate_v1' if args.control=='zero_delegate' else 'joint_point_native_repeat_v1' if args.control else 'joint_point_authored_native_pair_v1',status='fail',phase='preflight',source_binding=proof['source_binding'],protocol_identity=pin,
        frames=3,constructor_attempts=0,constructor_returns=0,probe_attempts=0,probe_returns=0,run_attempts=0,run_returns=0,
        manufactured_not_inferred=True,tracker_executed=False,positive_weight_executed=False,quality_verified=False,adoption=False,ground_truth_used=False,challenge_inputs_used=False)
    if args.control is not None:report['control']=args.control
    def persist():
        temporary=out/'report.tmp';temporary.write_text(json.dumps(report,sort_keys=True,allow_nan=False));temporary.replace(out/'report.json')
    def expired(*_):raise TimeoutError('Inclusive authored runtime1380')
    signal.signal(signal.SIGALRM,expired);signal.signal(signal.SIGTERM,expired)
    signal.setitimer(signal.ITIMER_REAL,max(.001,proof['host_start_monotonic']+1380-time.monotonic()))
    try:native(rt,root,code,revision,out,proof,persist,report,args.control)
    except BaseException as error:report.update(status='fail',error_type=type(error).__name__,error_context=str(error)[-500:]);raise
    finally:
        report['elapsed_seconds']=time.monotonic()-proof['host_start_monotonic'];persist()
        for p in out.iterdir():p.chmod(0o444)
        out.chmod(0o555);rt.require(time.monotonic()-proof['host_start_monotonic']<=1380,'Inclusive native seal exceeded');signal.setitimer(signal.ITIMER_REAL,0)


if __name__=='__main__':main()
