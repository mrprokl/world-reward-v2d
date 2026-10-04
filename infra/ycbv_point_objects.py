"""Three public YCBV frame-zero native Objects initializers, never a submission.

The caller freezes runtime/assets/installed-source receipts before this run.
Missing facts fail before CUDA or model reads. Native image_to_mesh is unchanged;
no human scale, sensor depth, private calibration, labels, mesh or GT mounts.
"""
from __future__ import annotations
import hashlib
from importlib import util
import json
import os
from pathlib import Path, PurePosixPath
import posixpath
import re
import signal
import stat
import subprocess
import sys
import time

ROOT=Path('/srv/scenesmith/world-reward');BASE='validation/ycbv_point_pose_v1';OUTPUT='objects_init_v1'
ENTRY='run_ycbv_point_objects';PIN_FILE='configs/ycbv_point_objects_pins.json'
IMAGE='sha256:eb389b26358c49778a14303b5875c66d887824011388ce9f8666ed7cc1841ce5'
STAGE='external_ycbv_three_anchor_native_Objects_initializer';BUDGET=900;SCENES=(48,49,50)
OBJECT_REV='2e73555018d2741ccd486e56c24fac41155a1dc6';DINO_REV='7764ea0f912e53c92e82eb78a2a1631e92725fc8'
MOGE_REV='ad326bfb61facd6c52b5a825bc1e34d7c97d9672'
WEIGHTS='weights/sam3d';OBJECT=WEIGHTS+'/hf-download';DINO=WEIGHTS+'/torch_home/hub/facebookresearch_dinov2_main'
HF=WEIGHTS+'/hf_home/hub';MOGE=HF+'/models--Ruicheng--moge-vitl';SNAPSHOT=MOGE+'/snapshots/'+MOGE_REV
CKPTS=('ss_generator','slat_generator','ss_decoder','slat_decoder_gs','slat_decoder_gs_4','slat_decoder_mesh')
YAMLS=('pipeline',*CKPTS);REG4=('dinov2_vitl14_reg4_pretrain.pth','dinov2_vitb14_reg4_pretrain.pth')
HELPERS=('infra/ycbv_point_objects.py','infra/run_ycbv_point_objects.sh','src/world_reward/__init__.py',
    'src/world_reward/pointmap.py','src/world_reward/mesh_geometry.py')
MODULES=('sam3d_objects','v2d.common','v2d.sam3d.lib','moge')
# Independently published model identities, not hashes observed during inference.
CHECKPOINT_PINS=(
 (6690136964,'225f40479e4cff4f39d6fa14c55be3abad1475bf55b61af3bec1e19ed2f6c146'),
 (4906537684,'91529bde8e7daa12d09618a66c319e3a5a6398db6b23b958cedcb1c3f28faabb'),
 (147609242,'6dac1cd7b7fda5a38e0614fadae441f1794f80e39ea2981f1ac8aff0a7e99340'),
 (171476155,'f8077c36a06eaf890dd93cda1937411f793dea1eb80b3dd9329f2038ba84a111'),
 (170269801,'731a0eceaa47945b52aa27f650d695b2aea9cc70945751e5609e5cb5b49f0186'),
 (363726862,'85907b37b67d8ce5b099a96629bdcfbd873eb407dee6b3aa9a75deb15038db33'))
YAML_PINS=(
 (3548,'53c3d226b21df85c0bb3d16e6e4fa63abde0d6167525765eb929d02bfa9d358c'),
 (5076,'3c265448bca7c057f94e3ef56adea3a895a10bcd9f15f992a41dd03fa35412cd'),
 (1986,'53029fadff6fe34a0344381a16d64d65d0567d603484952712e8969319559c4e'),
 (244,'baacff269b664f84f7aa1896ebd66128065f6568cdb88d419d5c3d2ccb4193ae'),
 (576,'53f054e02a0c185f0a6885d30bb6ca0ec92efe0a98148688e7f05f7c26afe670'),
 (575,'3d1dfd4c56cdac56f30e0cf5eb310d4df973fe25c60ac0a462c86ca4e3bf8b48'),
 (300,'8f46952764aa985c50109a56f0b4f07625cdb06457aaded74957d26f3520c69e'))
SOURCE_LICENSE_PIN=dict(bytes=8204,sha256='b3a5a0e2d973ab80e6610ccf1cffc40756050d0ace3cd4fec879b3ec290b2e9b')  # Primary source LICENSE only; NOT assumed equal to HF model LICENSE.


def require(value,message):
    if not value:raise ValueError(message)


def safe(name):
    require(type(name)is str and name and str(PurePosixPath(name))==name and not name.startswith('/')and '\\'not in name and '\0'not in name and
        all(p not in('','.','..')for p in name.split('/')),'Exact canonical public relative path required');return name


def canonical(path):
    path=Path(path);require(path.is_absolute()and path.resolve()==path and not any(p.is_symlink()for p in(path,*path.parents)),'Canonical nonsymlink path required');return path


def pin(value,empty=False):
    require(type(value)is dict and set(value)=={'bytes','sha256'}and type(value['bytes'])is int and
        (0 if empty else 1)<=value['bytes']<=7_000_000_000 and re.fullmatch('[0-9a-f]{64}',value.get('sha256','')),'Independent exact SHA/byte pin required')


def identity(path,readonly=False,empty=False):
    path=canonical(path);s=path.lstat();require(stat.S_ISREG(s.st_mode)and s.st_nlink==1 and
        (0 if empty else 1)<=s.st_size<=7_000_000_000 and(not readonly or not s.st_mode&0o222),'Original bounded regular bytes required')
    h=hashlib.sha256()
    with path.open('rb')as stream:
        for chunk in iter(lambda:stream.read(4*1024*1024),b''):h.update(chunk)
    a=path.lstat();require(all(getattr(s,k)==getattr(a,k)for k in('st_dev','st_ino','st_mode','st_size','st_mtime_ns','st_ctime_ns','st_nlink')),'Original bytes changed while hashing')
    return dict(bytes=s.st_size,sha256=h.hexdigest())


def strict(raw):
    def pairs(rows):
        value={}
        for k,v in rows:require(k not in value,'Duplicate JSON key forbidden');value[k]=v
        return value
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda _:(_ for _ in()).throw(ValueError('Nonfinite JSON forbidden')))


def bound_json(path,expected):
    require(expected['bytes']<=2_000_000,'Bounded public receipt JSON required')
    require(identity(path)==expected,'Independent receipt/manifest pin differs before JSON');raw=path.read_bytes()
    require(dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())==expected,'Receipt changed before parsing');return strict(raw)


def source(code,revision,container=False):
    canonical(code);require(code==ROOT/'jobs'/revision/ENTRY/'code'and re.fullmatch('[0-9a-f]{40}',revision),'Actual immutable source namespace required')
    digest=hashlib.sha256();rows={}
    for name in('revision','source-sha256'):
        path=code.parent/name;row=identity(path);raw=path.read_bytes()
        require(raw==(revision+'\n').encode()if name=='revision'else bool(re.fullmatch(b'[0-9a-f]{64}\n',raw)),'Actual dispatch marker required');digest.update(raw)
    if container:
        paths=[code/name for name in(*HELPERS,PIN_FILE)]
        require({str(p.relative_to(code))for p in code.rglob('*')if p.is_file()}==set((*HELPERS,PIN_FILE)),'Only exact blind helper/pin closure may enter GPU')
    else:paths=sorted(code.rglob('*'))
    for path in paths:
        canonical(path);s=path.lstat();require(not s.st_mode&0o222 and(stat.S_ISREG(s.st_mode)or stat.S_ISDIR(s.st_mode)),'Readonly original full source closure required')
        if path.is_file():row=identity(path,True,True);digest.update(str(path.relative_to(code)).encode()+b'\0'+bytes.fromhex(row['sha256']));rows[str(path.relative_to(code))]=row
    require(set((*HELPERS,PIN_FILE))<=set(rows),'Complete caller helper/pin closure required')
    return dict(closure_sha256=digest.hexdigest(),helpers={n:rows[n]for n in(*HELPERS,PIN_FILE)})


def validate_pins(value):
    require(type(value)is dict and set(value)=={'schema','inputs','runtime'}and value['schema']=='world_reward.ycbv_point_objects.pins.v1','Independent committed Objects input/runtime pins required')
    incoming=value['inputs'];require(set(incoming)=={'manifest','acquisition_report','mask_report','depth_report','bundles'},'Exact frozen public producer bundle pins required');pin(incoming['manifest'])
    for name in('acquisition_report','mask_report','depth_report'):
        row=incoming[name];require(set(row)=={'bytes','sha256','producer_revision','script_sha256'},'Actual historical producer receipt identity required')
        pin({k:row[k]for k in('bytes','sha256')});require(re.fullmatch('[0-9a-f]{40}',row['producer_revision'])and re.fullmatch('[0-9a-f]{64}',row['script_sha256']),'Exact actual producer source identity required')
    require(type(incoming['bundles'])is list and len(incoming['bundles'])==3,'Exactly three original frame-zero bundles required')
    for scene,row in zip(SCENES,incoming['bundles']):
        require(set(row)=={'scene_id','rgb','mask','depth'}and type(row['scene_id'])is int and row['scene_id']==scene,'Fixed external scene order required')
        expected=dict(rgb=f'{BASE}/inputs/scene_{scene:06d}_frame_000000.png',mask=f'{BASE}/automatic_masks_v1/scene_{scene:06d}/masks/1/000000.png',depth=f'{BASE}/depth_init_v1/scene_{scene:06d}_frame_000000.npz')
        for kind,name in expected.items():require(set(row[kind])=={'path','bytes','sha256'}and row[kind]['path']==name,'Only original public RGB/automaticmask/nativeMoGe2 NPZ allowed');pin({k:row[kind][k]for k in('bytes','sha256')})
    runtime=value['runtime'];require(set(runtime)=={'image_receipt','model_files','source_files','moge_links','installed_sources','acquisition_receipts'},'Exact frozen existing runtime evidence required')
    image=runtime['image_receipt'];require(set(image)=={'path','bytes','sha256'}and type(image['path'])is str and re.fullmatch(r'results/ycbv-objects-runtime-[0-9a-f]{40}/image\.json',image['path']),'Exact independently hashed safe Objects image projection required');pin({k:image[k]for k in('bytes','sha256')})
    required={OBJECT+'/checkpoints/'+n+'.ckpt'for n in CKPTS}|{OBJECT+'/checkpoints/'+n+'.yaml'for n in YAMLS}|{OBJECT+'/LICENSE',*(WEIGHTS+'/torch_home/hub/checkpoints/'+n for n in REG4)}
    for name,row in runtime['model_files'].items():safe(name);pin(row)
    require(required<=set(runtime['model_files'])and all(n in required or re.fullmatch(re.escape(MOGE)+r'/blobs/[0-9a-f]{40,64}',n)or re.fullmatch(re.escape(HF)+r'/blobs/[0-9a-f]{2}/[0-9a-f]{40,64}',n)for n in runtime['model_files']),'Exact Objects/DINO/MoGe1 model allowlist required')
    for names,suffix,known in ((CKPTS,'.ckpt',CHECKPOINT_PINS),(YAMLS,'.yaml',YAML_PINS)):
        for name,(size,sha) in zip(names,known):
            require(runtime['model_files'][OBJECT+'/checkpoints/'+name+suffix]==dict(bytes=size,sha256=sha),'Original primary Objects model/YAML identity required')
    require(runtime['model_files'][OBJECT+'/LICENSE']['bytes']<=100_000,'Independently pinned actual model notice required; source/model license equality not assumed')
    for name,row in runtime['source_files'].items():
        safe(name);pin(row,True);require(name.startswith(DINO+'/')and(name.endswith('.py')or name in(DINO+'/LICENSE',DINO+'/MODEL_CARD.md')),'Only audited DINO public source/cards required')
    require({DINO+'/hubconf.py',DINO+'/LICENSE',DINO+'/MODEL_CARD.md'}<=set(runtime['source_files']),'DINO public loader/license/modelcard required')
    require(set(runtime['acquisition_receipts'])=={'results/weights-acquisition.json','results/auxiliary-assets.json'},'Only original two model acquisition receipts required')
    for row in runtime['acquisition_receipts'].values():pin(row)
    require(set(runtime['installed_sources'])==set(MODULES),'All four installed native package fingerprints required before inference')
    for rows in runtime['installed_sources'].values():
        require(type(rows)is dict and rows,'Nonempty complete installed Python source pins required')
        for name,row in rows.items():safe(name);require(name.endswith('.py'),'Installed evidence onlyPython');pin(row,True)
    require(sum(row['bytes']for rows in runtime['installed_sources'].values()for row in rows.values())<=32_000_000,'Bounded installed native source-only evidence required')
    require(type(runtime['moge_links'])is dict and {SNAPSHOT+'/model.pt',SNAPSHOT+'/README.md'}<=set(runtime['moge_links']),'Actual original MoGe1 snapshot/blob link graph required')
    for name in runtime['moge_links']:
        safe(name);require(name in (SNAPSHOT+'/model.pt',SNAPSHOT+'/README.md')or re.fullmatch(re.escape(MOGE)+r'/blobs/[0-9a-f]{40,64}',name),'Only original model/card snapshot or intermediate repo blobs allowed')
    return value


def runtime_proof(root,pins,host=True):
    runtime=pins['runtime'];image=None
    if host:
        image=bound_json(root/runtime['image_receipt']['path'],{k:runtime['image_receipt'][k]for k in('bytes','sha256')})
        require(type(image)is dict and set(image)=={'Id','Architecture','Os','RootFS'}and image['Id']==IMAGE and image['Architecture']=='amd64'and image['Os']=='linux'and type(image['RootFS'])is dict and set(image['RootFS'])=={'Type','Layers'}and image['RootFS']['Type']=='layers'and type(image['RootFS']['Layers'])is list and image['RootFS']['Layers']and all(type(n)is str and re.fullmatch('sha256:[0-9a-f]{64}',n)for n in image['RootFS']['Layers']),'Exact safe immutable Objects image projection required; no raw configuration')
    for group in('model_files','source_files','acquisition_receipts'):
        for name,row in runtime[group].items():require(identity(root/name,empty=group=='source_files')==row,'Independently pinned runtime asset/source differs')
    links=runtime['moge_links'];seen=set()
    for name,target in links.items():
        safe(name);require(type(target)is str and target and not target.startswith('/')and '\\'not in target,'Relative original MoGe1 symlink required')
        path=root/name;canonical(path.parent);require(path.is_symlink()and os.readlink(path)==target,'Original MoGe1 link target differs')
        resolved=posixpath.normpath(posixpath.join(posixpath.dirname(name),target));safe(resolved)
        require((re.fullmatch(re.escape(MOGE)+r'/blobs/[0-9a-f]{40,64}',resolved)or re.fullmatch(re.escape(HF)+r'/blobs/[0-9a-f]{2}/[0-9a-f]{40,64}',resolved))and(resolved in links or resolved in runtime['model_files']),'MoGe1 link may resolve only exact frozen graph');seen.add(resolved)
    for first in(SNAPSHOT+'/model.pt',SNAPSHOT+'/README.md'):
        name=first;visited=set()
        while name in links:
            require(name not in visited and len(visited)<4,'MoGe1 graph cycle/excesslinks forbidden');visited.add(name);name=posixpath.normpath(posixpath.join(posixpath.dirname(name),links[name]))
        require(name in runtime['model_files'],'Pinned MoGe1 terminal bytes required')
        if first.endswith('model.pt'):require(runtime['model_files'][name]==dict(bytes=1256823446,sha256='da96b09a0485a3c45a5aa455e67743c8b4efc4dd8437c1f2aa93c2b4303d957f'),'Original primary MoGe1 checkpoint identity required')
    require(set(links)<=seen|{SNAPSHOT+'/model.pt',SNAPSHOT+'/README.md'}and all(not n.startswith((MOGE+'/blobs/',HF+'/blobs/'))or n in seen for n in runtime['model_files']),'No unrelated MoGe1 link or terminal blob allowed')
    require({p.name for p in(root/MOGE/'snapshots').iterdir()}=={MOGE_REV},'Nativefirstsnapshot selection must see only original revision')
    acquisition=bound_json(root/'results/weights-acquisition.json',runtime['acquisition_receipts']['results/weights-acquisition.json'])
    for repo,rev,field,path in(('facebook/sam-3d-objects',OBJECT_REV,'path',OBJECT),('Ruicheng/moge-vitl',MOGE_REV,'cache_dir',HF)):
        rows=[r for r in acquisition.get('assets',[])if r.get('repo_id')==repo];require(len(rows)==1 and rows[0].get('revision')==rev and rows[0].get(field)==str(root/path),'Original model acquisition revision/path required')
    aux=bound_json(root/'results/auxiliary-assets.json',runtime['acquisition_receipts']['results/auxiliary-assets.json'])
    require(aux.get('source_revisions',{}).get('dinov2')==DINO_REV,'Pinned DINO source acquisition required')
    for name in REG4:
        rows=[r for r in aux.get('checkpoints',[])if r.get('filename')==name];require(len(rows)==1 and rows[0].get('hash_source')=='first_observed_https_download'and
            {k:rows[0].get(k)for k in('bytes','sha256')}==runtime['model_files'][WEIGHTS+'/torch_home/hub/checkpoints/'+name],'Actual first-observed DINO register receipt required, no independent release claim')
    proof=dict(image_id=IMAGE,assets=runtime['model_files'],DINO_source=runtime['source_files'],installed_source_commit_verified=False)
    if host:proof['rootfs']=image['RootFS']
    return proof


def inputs_proof(root,pins,host=True):
    incoming=pins['inputs'];manifest=bound_json(root/BASE/'inputs/manifest.json',incoming['manifest'])
    require(manifest.get('schema')=='world-reward-ycbv-point-rgb-v1'and manifest.get('revision')=='5c2c4aa229800355648cd268040aa814f8dc94f0'and
        manifest.get('license')=='MIT'and len(manifest.get('images',[]))==288 and set(manifest)=={'schema','revision','license','selection','attribution','images'},'Original independently acquired contiguous MITRGB cohort required')
    require(manifest['selection']=='first_three_sorted_scene_directories_first_96_contiguous_RGB_names_before_private_annotations' and
        manifest['attribution']=='YCB-Video: Yu Xiang et al.; BOP conversion: Hodan et al.','Frozen RGB-only cohort selection required')
    for index,row in enumerate(manifest['images']):
        scene,frame=SCENES[index//96],index%96
        require(set(row)=={'scene_id','frame_id','file','sha256','width','height'} and
            all(type(row[k])is int for k in ('scene_id','frame_id','width','height'))and
            (row['scene_id'],row['frame_id'],row['width'],row['height'])==(scene,frame,640,480) and
            row['file']==f'scene_{scene:06d}_frame_{frame:06d}.png'and re.fullmatch('[0-9a-f]{64}',row['sha256']),'Every original full-clip index/grid must stay unchanged')
    stages=dict(acquisition_report=('external_ycbv_contiguous_rgb_only_acquisition','report.json'),mask_report=('public_ycbv_point_native_object_masks','automatic_masks_v1/report.json'),depth_report=('public_ycbv_three_frame_zero_native_MoGe2_preflight','depth_init_v1/report.json'))
    receipts={}
    for role,(stage,path)in stages.items():
        if role=='acquisition_report'and not host:continue
        row=incoming[role];receipt=bound_json(root/BASE/path,{k:row[k]for k in('bytes','sha256')});require(receipt.get('stage')==stage and receipt.get('status')=='pass'and
            receipt.get('producer_revision')==row['producer_revision']and receipt.get('script_sha256')==row['script_sha256']and receipt.get('phase')=='complete','Original passed public producer required')
        post=dict(acquisition_report='source_rehashed_after',mask_report='all_inputs_sources_assets_outputs_rehashed',depth_report='sources_after_reverified')[role]
        require(receipt.get(post)is True,'Actual original producer post-verification field required')
        require(receipt.get('challenge_inputs_used')is False and(receipt.get('private_truth_read')is False if role=='depth_report'else receipt.get('ground_truth_used')is False if role=='mask_report'else receipt.get('private_annotations_exported_as_inference_inputs')is False),'Original noGT/challenge provenance required');receipts[role]=receipt
    if host:
        acquire=receipts['acquisition_report'];require(acquire.get('public_manifest')==incoming['manifest'] and acquire.get('selected_frames')==288 and acquire.get('all_instances_retained')is True and
            acquire.get('selection_before_private_annotation_values')is True and acquire.get('license')=='MIT','Original pinned acquisition cohort/no-label export required')
    require(receipts['mask_report'].get('frames_completed')==288 and receipts['depth_report'].get('outputs_completed')==3,'All three full masks and native framezero depth required')
    for role,budget in (('mask_report',600),('depth_report',300)):
        row=receipts[role];elapsed=row.get('GPU_budget_elapsed_seconds',row.get('elapsed_seconds'))
        require(row.get('budget_seconds')==budget and type(elapsed)in(int,float)and 0<elapsed<=budget,'Original frozen frontend budget must have passed')
    require([(r.get('scene_id'),r.get('frame_id'))for r in receipts['mask_report']['masks']]==[(s,f)for s in SCENES for f in range(96)] and
        [(r.get('scene_id'),r.get('frame_id'))for r in receipts['depth_report']['outputs']]==[(s,0)for s in SCENES],'Full original automatic-mask/native-depth order required')
    for bundle in incoming['bundles']:
        scene=bundle['scene_id'];rgb=bundle['rgb'];rows=[r for r in manifest['images']if r.get('scene_id')==scene and r.get('frame_id')==0]
        require(len(rows)==1 and rows[0].get('sha256')==rgb['sha256']and rows[0].get('file')==Path(rgb['path']).name,'Same original RGB manifest required')
        for kind in('rgb','mask','depth'):require(identity(root/bundle[kind]['path'])=={k:bundle[kind][k]for k in('bytes','sha256')},'Original public bundle bytes required')
        mask=[r for r in receipts['mask_report']['masks']if r['scene_id']==scene and r['frame_id']==0];depth=[r for r in receipts['depth_report']['outputs']if r['scene_id']==scene and r['frame_id']==0]
        require(len(mask)==len(depth)==1 and {k:mask[0][k]for k in('bytes','sha256')}=={k:bundle['mask'][k]for k in('bytes','sha256')}and
            {k:depth[0][k]for k in('bytes','sha256')}=={k:bundle['depth'][k]for k in('bytes','sha256')}and
            mask[0]['rgb_sha256']==depth[0]['rgb_sha256']==rgb['sha256'] and mask[0]['file']==f'scene_{scene:06d}/masks/1/000000.png' and
            depth[0]['file']==Path(bundle['depth']['path']).name,'Original mask/depth sameRGB lineage required')
    return receipts


def installed_proof(pins):
    rows={}
    for name in MODULES:
        spec=util.find_spec(name);require(spec is not None and spec.submodule_search_locations,'Exact installed native package required')
        require(len(spec.submodule_search_locations)==1,'Single exact installed package root required');folder=Path(next(iter(spec.submodule_search_locations)));canonical(folder)
        actual={str(p.relative_to(folder)):identity(p,empty=True)for p in sorted(folder.rglob('*.py'))}
        require(actual==pins['runtime']['installed_sources'][name],'Independent actual installed-source fingerprint differs');rows[name]=actual
    return rows


def bundle_arrays(root,bundle):
    import numpy as np
    from PIL import Image
    from world_reward.pointmap import validate_camera_pointmap
    for kind in ('rgb','mask','depth'):require(identity(root/bundle[kind]['path'])=={k:bundle[kind][k]for k in ('bytes','sha256')},'Bundle hash required before any decode')
    with Image.open(root/bundle['rgb']['path'])as image:require(image.format=='PNG'and image.mode=='RGB'and image.size==(640,480),'OriginalRGBPNG640480 required');rgb=np.asarray(image).copy()
    with Image.open(root/bundle['mask']['path'])as image:require(image.format=='PNG'and image.mode=='L'and image.size==(640,480),'OriginalautomaticmaskPNG640480 required');mask=np.asarray(image).copy()
    require(mask.dtype==np.uint8 and np.isin(mask,[0,255]).all()and np.count_nonzero(mask)>0,'Nonempty automaticbinaryobjectmask required')
    with np.load(root/bundle['depth']['path'],allow_pickle=False)as file:arrays={k:file[k]for k in file.files}
    require(set(arrays)=={'depth','points','mask','intrinsics','frame_index'},'Exact untouched nativeMoGe2 arrays required')
    for key,shape,dtype in(('depth',(480,640),np.float32),('points',(480,640,3),np.float32),('mask',(480,640),np.bool_),('intrinsics',(3,3),np.float32),('frame_index',(),np.int64)):
        require(arrays[key].shape==shape and arrays[key].dtype==dtype,'Nativefullgrid dtype/shape required')
    require(arrays['frame_index'].item()==0 and np.count_nonzero((mask>0)&arrays['mask']&np.isfinite(arrays['points']).all(-1)&(arrays['points'][...,2]>0))>=8,'Originalframezero/visibleinferredobjectsupport required')
    focal=float(np.hypot(640,480));K=np.array([[focal,0.,320.],[0.,focal,240.],[0.,0.,1.]])
    checked=validate_camera_pointmap(arrays['depth'],arrays['points'],arrays['mask'],arrays['intrinsics'],K)
    K=np.asarray(checked['pixel_K'],dtype=np.float64)  # Exact inferred normalized-FP32 K; no rounding/snap.
    for kind in ('rgb','mask','depth'):require(identity(root/bundle[kind]['path'])=={k:bundle[kind][k]for k in ('bytes','sha256')},'Original bundle changed after decode')
    return rgb,mask,arrays,K


def native_pose(transform,K,output_K):
    import numpy as np
    from scipy.spatial.transform import Rotation
    require(type(transform)is dict and set(transform)=={'rotation','translation','scale'},'Exact native OpenCVtransform required')
    q,t,s=(np.asarray(transform[k],dtype=np.float64)for k in('rotation','translation','scale'))
    require(q.shape==(4,)and t.shape==(3,)and s.shape==(3,)and np.isfinite(np.r_[q,t,s]).all()and np.isclose(np.linalg.norm(q),1.,atol=1e-5,rtol=0)and
        (s>0).all()and np.allclose(s,s[0],atol=0,rtol=1e-5),'NativeunitWXYZ/finitepose/positiveuniformscale required; noaveraging')
    expected=dict(fx=float(K[0,0]),fy=float(K[1,1]),cx=float(K[0,2]),cy=float(K[1,2]),width=640,height=480)
    require(output_K==expected,'NativeprovidedcameraKmustbeunchanged')
    R=Rotation.from_quat([q[1],q[2],q[3],q[0]]).as_matrix();return R,t,float(s[0])


def export_geometry(mesh,transform,K,output_K,path):
    import numpy as np
    import trimesh
    require(isinstance(mesh,trimesh.Trimesh),'Native canonical triangular mesh required')
    vertices=np.asarray(mesh.vertices,dtype=np.float64);faces=np.asarray(mesh.faces)
    require(vertices.ndim==2 and vertices.shape[1]==3 and np.isfinite(vertices).all()and len(vertices)>=4 and faces.ndim==2 and faces.shape[1]==3 and
        faces.dtype.kind in'iu'and len(faces)>=4 and(faces>=0).all()and(faces<len(vertices)).all(),'All original finite nativevertices/triangles required')
    from world_reward.mesh_geometry import normalize_degenerate_faces
    active,diagnostics=normalize_degenerate_faces(vertices,faces)
    require(len(active)==len(faces)and len(np.unique(np.sort(faces,axis=1),axis=0))==len(faces),'Zero-area or duplicate native triangles cannot become a valid surface; no repair')
    components=trimesh.graph.connected_components(mesh.face_adjacency,nodes=np.arange(len(faces)))
    qualification=dict(component_count=len(components),watertight=bool(mesh.is_watertight),winding_consistent=bool(mesh.is_winding_consistent),
        signed_volume_generated_units=float(mesh.volume),zero_area_faces=diagnostics['excluded_faces'],self_intersections_verified=False,
        self_intersections=None,self_intersection_operator='unsupported_in_this_runtime_no_new_dependency')
    require(np.isfinite(mesh.area)and mesh.area>0 and np.isfinite(mesh.volume),'Native finite nonzero-area geometry required')
    R,t,scale=native_pose(transform,K,output_K)
    arrays=dict(vertices=vertices*scale,faces=faces.astype(np.int64),R0=R,t0=t,K=K.astype(np.float64),scale=np.array(1.,dtype=np.float64))
    require(np.isfinite(arrays['vertices']).all(),'Scale application must remain finite')
    with Path(path).open('xb')as stream:os.fchmod(stream.fileno(),0o444);np.savez_compressed(stream,**arrays)
    return dict(**identity(path,True),vertices=len(vertices),faces=len(faces),native_generated_scale=scale,scale_baked_once=True,
        pose_applied_to_mesh=False,source_triangles_retained=True,topology_qualification=qualification,geometry_budget_verified=False,
        canonical_array_sha256={k:hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()for k,v in arrays.items()})


def observe(root,pins,out,report,persist,native):
    import numpy as np
    import trimesh
    for bundle in pins['inputs']['bundles']:
        scene=bundle['scene_id'];folder=out/f'scene_{scene:06d}';folder.mkdir(mode=0o700)
        rgb,_,arrays,K=bundle_arrays(root,bundle);pointmap=folder/'pointmap.npy';camera=folder/'pointmap_intrinsics.json'
        with pointmap.open('xb')as stream:os.fchmod(stream.fileno(),0o444);np.save(stream,arrays['points'],allow_pickle=False)
        depth_receipt=bound_json(root/BASE/'depth_init_v1/report.json',{k:pins['inputs']['depth_report'][k]for k in ('bytes','sha256')})
        matching=[r for r in depth_receipt['outputs']if r['scene_id']==scene and r['frame_id']==0]
        require(len(matching)==1 and matching[0]['decoded_RGB_sha256']==hashlib.sha256(rgb.tobytes()).hexdigest(),'Original native depth decodedRGB lineage differs')
        camera_dict=dict(fx=float(K[0,0]),fy=float(K[1,1]),cx=float(K[0,2]),cy=float(K[1,2]),width=640,height=480)
        with camera.open('x')as stream:os.fchmod(stream.fileno(),0o444);json.dump(camera_dict,stream)
        pointpin,cam_pin=identity(pointmap),identity(camera);report['native_calls_attempted']+=1;persist()
        native(str(root/bundle['rgb']['path']),str(root/bundle['mask']['path']),str(folder/'object.glb'),str(folder/'transform.json'),str(folder/'intrinsics.json'),str(root/WEIGHTS),
            seed=0,stage1_only=False,with_mesh_postprocess=False,with_texture_baking=False,with_layout_postprocess=False,use_vertex_color=True,
            pointmap_path=str(pointmap),pointmap_intrinsics_path=str(camera))
        report['native_calls_returned']+=1
        require(identity(pointmap)==pointpin and identity(camera)==cam_pin,'NativeinputXYZ/Kchanged')
        rawpins={n:identity(folder/n)for n in ('object.glb','transform.json','intrinsics.json')}
        mesh=trimesh.load(folder/'object.glb',force='mesh',process=False)
        geometry=export_geometry(mesh,strict((folder/'transform.json').read_bytes()),K,strict((folder/'intrinsics.json').read_bytes()),folder/'canonical.npz')
        require({n:identity(folder/n)for n in rawpins}==rawpins,'Native outputs changed during canonical export')
        for name in('object.glb','transform.json','intrinsics.json'):identity(folder/name);(folder/name).chmod(0o444)
        pointmap.unlink();camera.unlink();report['native_calls_completed']+=1
        report['outputs'].append(dict(scene_id=scene,frame_id=0,geometry=geometry,files={name:identity(folder/name)for name in('object.glb','transform.json','intrinsics.json','canonical.npz')},
            input_bundle=bundle,camera_unchanged=True,generated_metric_scale_accuracy_verified=False));persist()


def run(root,code,revision,out,report,persist):
    pins=validate_pins(strict((code/PIN_FILE).read_bytes()));before=source(code,revision,True)
    require(not any((root/name).exists()for name in(BASE+'/eval_private','data',BASE+'/report.json','vendor','results/image-sam3d-runtime.json')),'NoGT/acquisition/challengepaths mayenterGPU')
    started=float(os.environ['WR_OBJECTS_STARTED']);remaining=BUDGET-(time.monotonic()-started);require(remaining>0,'Whole GPU budget exhausted before native model load')
    signal.alarm(max(1,int(remaining)));runtime=runtime_proof(root,pins,False);receipts=inputs_proof(root,pins,False);installed=installed_proof(pins)
    report['inputs']=pins['inputs'];report['source_helpers']=before['helpers'];report['phase']='native_source_verified';persist()
    import torch
    require(torch.cuda.is_available(),'OriginalGPUrequired; nofallback')
    original=torch.hub.load;calls=[]
    def local(repo_or_dir,*args,**kwargs):
        model=kwargs.get('model',args[0]if args else None);require(repo_or_dir in('facebookresearch/dinov2','facebookresearch/dinov2:main')and model in('dinov2_vitl14_reg','dinov2_vitb14_reg'),'Only originalDINOregisterloads allowed')
        calls.append(model);kwargs['source']='local';return original(str(root/DINO),*args,**kwargs)
    try:
        torch.hub.load=local
        from v2d.sam3d.lib.image_to_mesh import image_to_mesh
        report.update(runtime_proof=runtime,installed_sources=installed,phase='native_model_load');persist()
        observe(root,pins,out,report,persist,image_to_mesh);torch.cuda.synchronize()
        require(report['native_calls_attempted']==report['native_calls_returned']==report['native_calls_completed']==3,'All three actualnativecallsrequired')
    finally:
        torch.hub.load=original
        require(runtime_proof(root,pins,False)==runtime and installed_proof(pins)==installed and source(code,revision,True)==before and inputs_proof(root,pins,False)==receipts,'Originalnativeassets/sourcechangedafterinference')
        # Acquisition receipt is host-only; publicbundle receipts mounted below.
        report['GPU_budget_elapsed_seconds']=time.monotonic()-started;signal.alarm(0)
        require(report['GPU_budget_elapsed_seconds']<=BUDGET,'Whole3anchorGPUbudgetincludesproof/hash/post');report['source_rehashed_after']=True;persist()
    require(set(calls)=={'dinov2_vitl14_reg','dinov2_vitb14_reg'},'Both genuine original DINO register networks must be loaded')
    report.update(status='pass',phase='complete',DINO_hub_calls=calls,outputs_completed=3)


def host_proof(root,code,revision):
    before=source(code,revision);pins=validate_pins(strict((code/PIN_FILE).read_bytes()));receipts=inputs_proof(root,pins);runtime=runtime_proof(root,pins)
    return pins,dict(source=before,receipts={name:{k:pins['inputs'][name][k]for k in('bytes','sha256')}for name in receipts},runtime=runtime)



def runtime_mounts(root,pins):
    """Preserve only the authenticated model/card link graph, never HF caches."""
    runtime=pins['runtime'];repo_blobs=MOGE+'/blobs';dirs=[SNAPSHOT,repo_blobs]
    for relative in dirs:
        path=canonical(root/relative);require(path.is_dir(),'Existing original narrow MoGe1 directory required')
        expected={Path(n).name for n in (*runtime['moge_links'],*runtime['model_files'])if str(Path(n).parent)==relative}
        require({p.name for p in path.iterdir()}==expected,'Only exact MoGe1 model/card graph may enter native constructor')
    names=set(runtime['model_files'])|set(runtime['source_files'])|set(runtime['acquisition_receipts'])
    names={n for n in names if not n.startswith(repo_blobs+'/')}
    return [root/n for n in sorted(names)]+[root/n for n in dirs]


def command(args,deadline,allowed=(0,)):
    env={'PATH':'/usr/bin:/bin:/usr/sbin:/sbin','HOME':'/nonexistent','LANG':'C.UTF-8','DOCKER_HOST':'unix://'+str(ROOT/'docker.sock')}
    result=subprocess.run(args,env=env,capture_output=True,timeout=min(15,max(.01,deadline-time.monotonic())))
    require(result.returncode in allowed and len(result.stdout)<=2_000_000,'Bounded metadata/control command failed; subprocess details omitted')
    return result.stdout


def output_inventory(out,report):
    require([(r['scene_id'],r['frame_id'])for r in report['outputs']]==[(s,0)for s in SCENES],'All three original scene outputs required')
    for row in report['outputs']:
        folder=canonical(out/f"scene_{row['scene_id']:06d}")
        require(set(row['files'])=={'object.glb','transform.json','intrinsics.json','canonical.npz'}and {p.name for p in folder.iterdir()}==set(row['files']),'No extra/missing native outputs or retained staging files')
        for name,expected in row['files'].items():require(identity(folder/name,True)==expected,'Original native output changed before final receipt')
    require({p.name for p in out.iterdir()}=={'.native-report.json','.container.cid','.probe.cid','probe.log','gpu.log',*[f'scene_{s:06d}'for s in SCENES]},'Complete owned output inventory required')


def actual_image(deadline):
    row=strict(command(['docker','image','inspect',IMAGE,'--format','{"Id":{{json .Id}},"Architecture":{{json .Architecture}},"Os":{{json .Os}},"RootFS":{{json .RootFS}}}'],deadline))
    require(row['Id']==IMAGE and row['Architecture']=='amd64'and row['Os']=='linux'and set(row['RootFS'])=={'Type','Layers'}and
        row['RootFS']['Type']=='layers'and row['RootFS']['Layers']and all(re.fullmatch('sha256:[0-9a-f]{64}',n)for n in row['RootFS']['Layers']),'Actual full immutable Objects image identity required')
    return row


def lock_identity(root,fd):
    path=canonical(root/'jobs/.world-reward-h100.lock');before=path.lstat();opened=os.fstat(fd)
    require(stat.S_ISREG(before.st_mode)and before.st_nlink==1 and(before.st_dev,before.st_ino)==(opened.st_dev,opened.st_ino),'Inherited FD9 must hold exact original regular GPU lock')
    import fcntl
    fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    return [getattr(before,k)for k in('st_dev','st_ino','st_mode','st_nlink','st_size','st_mtime_ns','st_ctime_ns')]


def docker_arguments(root,code,revision,pins,out,proof,started,mode,name):
    paths=[code/n for n in(*HELPERS,PIN_FILE)]+[code.parent/n for n in ('revision','source-sha256')]
    if mode=='gpu':
        paths+=runtime_mounts(root,pins)
        paths+=[root/BASE/'inputs/manifest.json',root/BASE/'automatic_masks_v1/report.json',root/BASE/'depth_init_v1/report.json']
        paths+=[root/b[k]['path']for b in pins['inputs']['bundles']for k in ('rgb','mask','depth')]
    require(len(paths)==len(set(paths)),'Duplicate/nongeneric mount allowlist required')
    mounts=[]
    for path in paths:mounts+=['--mount',f'type=bind,src={path},dst={path},readonly']
    args=['docker','run','--rm','--interactive','--name',name,'--cidfile',str(out/('.container.cid'if mode=='gpu'else '.probe.cid')),
        '--label','world-reward.job='+ENTRY,'--label','world-reward.revision='+revision,'--network','none','--read-only','--user','0:0',
        '--cap-drop','ALL','--security-opt','no-new-privileges','--memory','32g','--cpus','4','--tmpfs','/tmp:rw,noexec,nosuid,size=512m',*mounts]
    if mode=='gpu':args+=['--gpus','all','--mount',f'type=bind,src={out},dst={out}']
    args+=['--entrypoint','/usr/bin/env',IMAGE,'-i','PATH=/opt/conda/bin:/usr/local/bin:/usr/bin:/bin','HOME=/tmp','XDG_CACHE_HOME=/tmp',
        'WR_ROOT='+str(root),'WR_CODE='+str(code),'WR_CODE_REVISION='+revision,'WR_IMAGE_ID='+IMAGE,'WR_HOST_PROOF='+proof,
        'WR_OBJECTS_STARTED='+str(started),'WR_OBJECTS_MODE='+mode,'HF_HOME='+str(root/WEIGHTS/'hf_home'),'TORCH_HOME='+str(root/WEIGHTS/'torch_home'),
        'HF_HUB_OFFLINE=1','TRANSFORMERS_OFFLINE=1','PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4','MKL_NUM_THREADS=4',
        '/opt/conda/bin/python','-I','-B','-',str(code)]
    return args


BOOTSTRAP=b"from pathlib import Path\nimport runpy,sys\ncode=Path(sys.argv[1]);sys.path[:0]=[str(code/'infra'),str(code/'src')];sys.argv=[str(code/'infra/ycbv_point_objects.py')];runpy.run_path(sys.argv[0],run_name='__main__')\n"


def cleanup(out,revision,name,deadline):
    for file,expected_name in (('.probe.cid',name+'-probe'),('.container.cid',name)):
        path=out/file
        if not path.exists():continue
        canonical(path);cid=path.read_text().strip();require(re.fullmatch('[0-9a-f]{64}',cid),'Owned CID only; no arbitrary cleanup')
        if command(['docker','ps','-aq','--no-trunc','--filter','id='+cid],deadline).strip():
            raw=command(['docker','inspect',cid,'--format','{{.Image}}|{{.Name}}|{{index .Config.Labels "world-reward.job"}}|{{index .Config.Labels "world-reward.revision"}}'],deadline).decode().strip()
            require(raw==f'{IMAGE}|/{expected_name}|{ENTRY}|{revision}','Only exact owned container may be removed')
            command(['docker','rm','-f',cid],deadline)
            require(not command(['docker','ps','-aq','--no-trunc','--filter','id='+cid],deadline).strip(),'Owned cleanup incomplete')
        path.chmod(0o400)


def launch(root,code,revision):
    """Host-only binding and whole-budget gate; final receipt after all checks."""
    require(os.geteuid()==0 and os.uname().nodename=='scenesmith-ncc-h100-01','Original owned VM01 root host required')
    source_before=source(code,revision);out=canonical(root/BASE/OUTPUT)
    require(out.parent.is_dir()and not out.exists()and not out.is_symlink(),'Fresh external output namespace required; no overwrite/resume')
    # Unknown runtime/input pins fail here, before creating an output or GPU job.
    validate_pins(strict((code/PIN_FILE).read_bytes()))
    out.mkdir(mode=0o700);started=time.monotonic();deadline=started+BUDGET;name='world-reward-ycbv-point-objects-'+revision[:12]
    report=dict(stage=STAGE,status='fail',phase='host_provenance',producer_revision=revision,script_sha256=source_before['helpers'][HELPERS[0]]['sha256'],
        budget_seconds=BUDGET,outputs=[],native_calls_attempted=0,native_calls_returned=0,native_calls_completed=0,ground_truth_used=False,private_annotations_read=False,
        sensor_depth_used=False,source_camera_calibration_used=False,human_scale_used=False,human_scalar_used=False,hand_observations_used=False,challenge_inputs_used=False,
        oracle_modes=[],source_helpers=source_before['helpers'],installed_source_commit_verified=False,submission_eligible=False,
        baseline_native_not_packed_track1=True,geometry_budget_verified=False,license_eligibility_verified=False,training_overlap_verified=False)
    host=None;image=None;held=None;pins=None;failure=None
    def expired(*_):raise TimeoutError('Whole900sObjectsbudget exhausted')
    handlers={s:signal.signal(s,expired)for s in(signal.SIGALRM,signal.SIGTERM,signal.SIGINT)};signal.alarm(BUDGET)
    try:
        held=lock_identity(root,9)
        require(not command(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],deadline).strip(),'GPU busy; no duplicate job')
        pins,host=host_proof(root,code,revision);proof=hashlib.sha256(json.dumps(host,sort_keys=True).encode()).hexdigest()
        image=actual_image(deadline);require(image['RootFS']==host['runtime']['rootfs'],'Actual image differs from original pinned receipt')
        runtime_mounts(root,pins)
        for candidate in (name,name+'-probe'):require(not command(['docker','ps','-aq','--filter','name=^/'+candidate+'$'],deadline).strip(),'Owned container namespace occupied')
        env={'PATH':'/usr/bin:/bin','HOME':'/nonexistent','DOCKER_HOST':'unix://'+str(root/'docker.sock')}
        for mode in ('probe','gpu'):
            report['phase']='installed_source_probe'if mode=='probe'else'native_inference'
            args=docker_arguments(root,code,revision,pins,out,proof,started,mode,name+'-probe'if mode=='probe'else name)
            with (out/(mode+'.log')).open('xb')as log:
                os.fchmod(log.fileno(),0o400)
                result=subprocess.run(args,input=BOOTSTRAP,stdout=log,stderr=log,env=env,timeout=max(.01,deadline-time.monotonic()),pass_fds=(9,))
            require(result.returncode==0,'Owned source/native container failed; original log preserved')
        path=out/'.native-report.json';native_id=identity(path);native=bound_json(path,native_id)
        require(native.get('stage')==STAGE and native.get('status')=='pass'and native.get('phase')=='complete'and
            native.get('source_helpers')==source_before['helpers']and all(native.get(k)==3 for k in ('native_calls_attempted','native_calls_returned','native_calls_completed','outputs_completed')),'Actual complete source-bound native receipt required')
        output_inventory(out,native)
        report.update(native,host_proof_sha256=proof,acquisition_receipt_checked_host_only=True,raw_image_receipt_mounted=False,inputs=pins['inputs'])
    except BaseException as error:failure=error;report.update(status='fail',error='Original public Objects initializer failed at recorded phase',error_type=type(error).__name__)
    finally:
        try:
            cleanup(out,revision,name,time.monotonic()+45)
            require(source(code,revision)==source_before and(host is None or host_proof(root,code,revision)[1]==host)and
                (image is None or actual_image(time.monotonic()+15)==image)and(held is None or lock_identity(root,9)==held),'Original inputs/assets/source/image/lock changed after inference')
            if report.get('status')=='pass':
                output_inventory(out,report)  # Final12rawoutputs rehashed after assets/source/image/lock checks.
                report['outputs_rehashed_after_host_post']=True
            report['source_rehashed_after']=True
        except BaseException as error:failure=error;report.update(status='fail',source_rehashed_after=False,error='Original post-verification or owned cleanup failed',error_type=type(error).__name__)
        report['GPU_budget_elapsed_seconds']=time.monotonic()-started
        report['GPU_budget_includes_model_hash_source_probe_load_decode_infer_export_and_host_post']=True
        if report['GPU_budget_elapsed_seconds']>BUDGET:failure=TimeoutError('Whole frozen budget exceeded');report.update(status='fail',error='Whole frozen budget exceeded')
        signal.alarm(0)
        for s,h in handlers.items():signal.signal(s,h)
        with (out/'report.json').open('x')as stream:os.fchmod(stream.fileno(),0o400);json.dump(report,stream,allow_nan=False);stream.write('\n')
    print(json.dumps({k:report.get(k)for k in ('stage','status','outputs_completed','GPU_budget_elapsed_seconds')}))
    if failure is not None:raise SystemExit(1)


def main(argv=None):
    require(not argv,'No publicselectorargumentsallowed');root=Path(os.environ['WR_ROOT']);code=Path(os.environ['WR_CODE']);revision=os.environ['WR_CODE_REVISION']
    require(root==ROOT and Path(__file__)==code/HELPERS[0]and sys.platform=='linux','ActualboundLinuxentryrequired')
    if os.environ.get('WR_OBJECTS_MODE')=='host':return launch(root,code,revision)
    if os.environ.get('WR_OBJECTS_MODE')=='probe':
        require(os.geteuid()==0 and os.environ.get('WR_IMAGE_ID')==IMAGE and {p.name for p in Path('/sys/class/net').iterdir()}=={'lo'},'Exact offline root source-only probe required')
        source(code,revision,True);installed_proof(validate_pins(strict((code/PIN_FILE).read_bytes())));print('native_installed_sources_verified');return
    require(os.geteuid()==0 and os.environ.get('WR_IMAGE_ID')==IMAGE and re.fullmatch('[0-9a-f]{64}',os.environ.get('WR_HOST_PROOF',''))and {p.name for p in Path('/sys/class/net').iterdir()}=={'lo'},'ActualofflineObjectscontainerrequired')
    out=canonical(root/BASE/OUTPUT);require(out.is_dir()and {p.name for p in out.iterdir()}=={'.container.cid','.probe.cid','probe.log','gpu.log'},'Freshreservedoutputrequired')
    report=dict(stage=STAGE,status='fail',producer_revision=revision,script_sha256=identity(Path(__file__))['sha256'],image_id=IMAGE,budget_seconds=BUDGET,
        native_calls_attempted=0,native_calls_returned=0,native_calls_completed=0,outputs=[],ground_truth_used=False,private_annotations_read=False,
        challenge_inputs_used=False,human_scale_used=False,human_scalar_used=False,hand_observations_used=False,sensor_depth_used=False,source_camera_calibration_used=False,hand_labeled_test=False,oracle_modes=[],
        geometry_budget_verified=False,baseline_native_not_packed_track1=True,submission_eligible=False,installed_source_commit_verified=False,license_eligibility_verified=False,training_overlap_verified=False)
    with (out/'.native-report.json').open('x')as stream:
        os.fchmod(stream.fileno(),0o400)
        def persist():stream.seek(0);json.dump(report,stream,allow_nan=False);stream.write('\n');stream.truncate();stream.flush();os.fsync(stream.fileno())
        def stopped(*_):raise TimeoutError('Frozen900sObjectsbudgetexceeded')
        handlers={s:signal.signal(s,stopped)for s in(signal.SIGALRM,signal.SIGTERM,signal.SIGINT)}
        try:persist();run(root,code,revision,out,report,persist)
        except Exception:report.update(status='fail',error='NativepublicObjectsinitializerfailedatrecordedphase');raise
        finally:
            signal.alarm(0)
            for sig,h in handlers.items():signal.signal(sig,h)
            persist()
    print(json.dumps({k:report.get(k)for k in('stage','status','outputs_completed','GPU_budget_elapsed_seconds')}))


if __name__=='__main__':main(sys.argv[1:])
