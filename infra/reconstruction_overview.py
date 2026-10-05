"""One small Azure-only contact sheet of independently frozen exports.

Original middle RGB and unchanged predicted meshes only. No model, labels,
quality evaluation, crop, fitting, or historical-producer execution occurs.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import time

ROOT=Path('/srv/scenesmith/world-reward')
ENTRY='run_reconstruction_overview'
IMAGE='sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7'
EPISODES=(0,1,2,3,5,6,8,9,12,13,14,15,21,26)
WIDTH,HEIGHT,ROW,HEADER=160,120,154,64
BUDGET,OUTER,MAX_JPEG_BYTES=180,210,240_000
EXPORT_FILES={'report.json','trajectory.npz','native_parameters.npz','target.npy','object_aligned.glb'}
FACE_SHA='f6748e290ef37fbb6877c4cc5bd7287105db9e98252b0ba170ae9ac3c45eacd6'
HELPERS=('infra/reconstruction_overview.py','infra/run_reconstruction_overview.sh',
         'infra/reconstruction_preview.py','infra/camera_render.py','infra/cari_clip_inputs.py',
         'infra/mediapipe_cpu_runtime_verify.py',
         *(f'configs/cari_clip_{e:06d}_{kind}_pins.json'for e in EPISODES for kind in('shared_export','input')))


def runtime(code):
    p=Path(code)/'infra/mediapipe_cpu_runtime_verify.py'
    if p.resolve()!=p or p.is_symlink()or p.stat().st_mode&0o222:raise ValueError('Readonly source helper required')
    spec=importlib.util.spec_from_file_location('wr_overview_runtime',p)
    rt=importlib.util.module_from_spec(spec);spec.loader.exec_module(rt);return rt


def midpoint(total):
    if type(total)is not int or total<3:raise ValueError('Full original timeline requires at least three frames')
    return(total-1)//2


def pin_name(episode,kind):return f'configs/cari_clip_{episode:06d}_{kind}_pins.json'


def host_proof(root,code,revision):
    rt=runtime(code);root,code=rt.canonical(root),rt.canonical(code)
    rt.require(root==ROOT and Path(__file__).resolve()==code/HELPERS[0], 'Actual source-bound overview required')
    rt.require({p.name for p in code.parent.iterdir()}=={'code','revision','source-sha256'},'Exact current source parent required')
    source=rt.source(root,code,revision,ENTRY,HELPERS)
    names={p.name for p in(code/'configs').glob('cari_clip_*_shared_export_pins.json')}
    rt.require(names=={Path(pin_name(e,'shared_export')).name for e in EPISODES},'All fourteen committed exports required; no clip selection')
    manifest_path=root/'results/input-manifest.json';mpin=rt.identity(manifest_path,1<<20,readonly=False)
    manifest=rt.strict(manifest_path.read_bytes())
    rt.require((manifest.get('track'),manifest.get('repo_id'),manifest.get('revision'))==('track_1','nvidia/video_to_data_challenge','5f68335f3acc802033d1e80728c1633197521de8'), 'Official Track1-only input manifest required')
    frozen={str(manifest_path):mpin};clips=[];mounts=[code.parent,manifest_path]
    for e in EPISODES:
        pin_path=code/pin_name(e,'shared_export');ipath=code/pin_name(e,'input')
        pp=rt.identity(pin_path,100000);ip=rt.identity(ipath,100000)
        pins=rt.strict(pin_path.read_bytes());inputs=rt.strict(ipath.read_bytes());spec=pins.get('clip_spec')
        rt.require(type(pins)is dict and set(pins)=={'schema','clip_spec','export','export_files'}and pins['schema']=='world-reward-cari-shared-export-pins-v1'
            and type(spec)is dict and set(spec)=={'episode_index','total_frames','camera_name','height','width'}
            and type(spec['episode_index'])is int and spec['episode_index']==e and spec==inputs.get('clip_spec')
            and type(spec['height'])is int and type(spec['width'])is int and spec['height']>0 and spec['width']*3==spec['height']*4
            and type(spec['total_frames'])is int and spec['total_frames']>=3,'Exact original full clip/export specification required')
        producer=pins['export'];rt.require(set(producer)=={'bytes','sha256','producer_revision','script_sha256'}and re.fullmatch('[0-9a-f]{40}',producer['producer_revision'])and re.fullmatch('[0-9a-f]{64}',producer['script_sha256']), 'Actual independently pinned export producer required')
        base=root/f'outputs/episode_{e:06d}/cari_shared_export_v1';rt.canonical(base)
        rt.require(base.is_dir()and{p.name for p in base.iterdir()}==EXPORT_FILES and set(pins['export_files'])==EXPORT_FILES, 'Exact five original export files required')
        for name,pin in pins['export_files'].items():
            rt.require(type(pin)is dict and set(pin)=={'bytes','sha256'}and rt.identity(base/name,1_000_000_000,readonly=False)==pin,'Frozen export bytes differ')
            frozen[str(base/name)]=pin
        rt.require(pins['export_files']['report.json']=={k:producer[k]for k in('bytes','sha256')},'Export report pin differs')
        report=rt.strict((base/'report.json').read_bytes())
        expected=dict(stage='world_reward_native_cari_shared_full_video_direct_export',status='pass',phase='complete',episode_index=e,clip_spec=spec,
            producer_revision=producer['producer_revision'],script_sha256=producer['script_sha256'],frames=spec['total_frames'],
            input_track='track_1',ground_truth_used=False,ground_truth_read=False,private_truth_read=False,hand_labeled_test=False,oracle_modes=[],
            unchanged_refined_predictions_verified=True,full_original_native_export_verified=True,original_frame_coverage_verified=True,
            source_inputs_assets_rehashed=True,source_helpers_rehashed=True)
        rt.require(all(type(report.get(k))is type(v)and report[k]==v for k,v in expected.items()),'Complete unchanged original export receipt required')
        rt.require(report.get('input_pins')==ip and report.get('source_files')==inputs.get('source_files')
            and report.get('output_files')=={n:r for n,r in pins['export_files'].items()if n!='report.json'}
            and report.get('original_frame_indices')==list(range(spec['total_frames'])),'Frozen export/public-input manifest binding differs')
        relative=f'track_1/videos/chunk-000/observation.images.exo_camera/episode_{e:06d}.mp4'
        rows=[r for r in manifest['files']if r.get('path')==relative]
        rt.require(len(rows)==1,'Exactly one official original RGB video required')
        video=root/'data'/relative;vp={k:rows[0][k]for k in('bytes','sha256')}
        rt.require(rt.identity(video,10_000_000_000,readonly=False)==vp,'Original official video SHA differs')
        frozen.update({str(video):vp,str(pin_path):pp,str(ipath):ip})
        clips.append(dict(episode_index=e,clip_spec=spec,frame_index=midpoint(spec['total_frames']),video=str(video),export=str(base),export_report_sha256=producer['sha256'],video_sha256=vp['sha256']))
        mounts.extend((video,base))
    return dict(source_binding=source,files=frozen,clips=clips,mounts=[str(p)for p in mounts],image_id=IMAGE)


def recheck(rt,proof):
    rt.require(all(rt.identity(Path(p),10_000_000_000,readonly=False)==pin for p,pin in proof['files'].items()),'Original visual QA source changed')


def arrays(np,base,spec):
    total=spec['total_frames'];target=np.load(base/'target.npy',mmap_mode='r',allow_pickle=False)
    with np.load(base/'trajectory.npz',allow_pickle=False)as z:
        required={'pose','scales','shape','expression','object_rotation','object_translation','object_scale','object_vertices','object_faces','camera_K','frame_index'}
        if len(z.files)!=len(required)or set(z.files)!=required:raise ValueError('Exact frozen trajectory keys required')
        values={k:z[k]for k in('object_vertices','object_faces','object_rotation','object_translation','object_scale','camera_K','frame_index')}
    with np.load(base/'native_parameters.npz',allow_pickle=False)as z:
        faces=z['human_faces'];indices=z['frame_index']
    if(target.dtype!=np.float32 or target.shape!=(total,18439,3)or target.flags.writeable or faces.dtype!=np.int64 or faces.shape!=(36874,3)
       or hashlib.sha256(faces.astype('<i4').tobytes()).hexdigest()!=FACE_SHA or indices.dtype!=np.int64 or not np.array_equal(indices,np.arange(total))):
        raise ValueError('Full original human geometry/topology/indices required')
    shapes={'object_vertices':(len(values['object_vertices']),3),'object_faces':(len(values['object_faces']),3),'object_rotation':(total,3,3),
            'object_translation':(total,3),'object_scale':(),'camera_K':(3,3),'frame_index':(total,)}
    for k,v in values.items():
        dtype=np.int64 if k in('object_faces','frame_index')else np.float64 if k=='camera_K'else np.float32
        if type(v)is not np.ndarray or v.dtype!=dtype or v.shape!=shapes[k]or not np.isfinite(v).all():raise ValueError('Exact original native trajectory arrays required')
    if not 3<=len(values['object_vertices'])<=4096 or not 1<=len(values['object_faces'])<=4096 or float(values['object_scale'])!=1 or not np.array_equal(values['frame_index'],np.arange(total)):
        raise ValueError('Original constant geometry and full timeline required')
    from cari_clip_inputs import PublicClipSpec,inferred_camera,_rigid
    if not np.array_equal(values['camera_K'],inferred_camera(PublicClipSpec(**spec))):raise ValueError('Original inferred camera changed')
    poses=np.tile(np.eye(4,dtype=np.float32),(total,1,1));poses[:,:3,:3]=values['object_rotation'];poses[:,:3,3]=values['object_translation'];_rigid(poses)
    return target,faces,values


def tile_position(position):
    if type(position)is not int or not 0<=position<len(EPISODES):raise ValueError('Original sorted clip position required')
    return(position%2)*WIDTH*2,HEADER+(position//2)*ROW


def encode_sheet(sheet):
    encoded=io.BytesIO();sheet.save(encoded,format='JPEG',quality=70,optimize=True)
    raw=encoded.getvalue()
    if not 0<len(raw)<=MAX_JPEG_BYTES:raise ValueError('Tiny overview byte cap exceeded; no quality rescue')
    return raw


def native(root,code,revision,out):
    started=time.monotonic();rt=runtime(code);proof=host_proof(root,code,revision)
    rt.require(sys.platform=='linux'and{p.name for p in Path('/sys/class/net').iterdir()}=={'lo'}and os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
        and os.environ.get('WR_IMAGE_ID')==IMAGE and rt.canonical(out)==root/'results'/('reconstruction-overview-'+revision)
        and out.is_dir()and{p.name for p in out.iterdir()}=={'container.cid'},'Fresh offline CPU overview with owned Docker CID required')
    cidfile=out/'container.cid';cidpin=rt.identity(cidfile,65,readonly=False)
    rt.require(re.fullmatch(b'[0-9a-f]{64}\n?',cidfile.read_bytes()),'Exact owned Docker CID required before native rendering')
    sys.path[:0]=[str(code/'infra'),str(code/'src')]
    import numpy as np
    import cv2
    from PIL import Image,ImageDraw,ImageFont
    import reconstruction_preview as preview
    rt.require(Path(preview.__file__).resolve()==code/'infra/reconstruction_preview.py','Original CPU preview source required')
    sheet=Image.new('RGB',(WIDTH*4,HEADER+ROW*7),(24,26,31));draw=ImageDraw.Draw(sheet)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
    draw.text((8,5),'World Reward | 14 frozen reconstructions | original middle frames',font=font,fill=(235,235,235))
    draw.text((8,26),'Cyan human / orange object | NO ground truth / NO accuracy score',font=font,fill=(235,235,235))
    diagnostics=[]
    for position,clip in enumerate(proof['clips']):
        rt.require(time.monotonic()-started<BUDGET,'Inclusive overview deadline')
        spec=clip['clip_spec'];index=clip['frame_index'];base=Path(clip['export']);target,faces,v=arrays(np,base,spec)
        capture=cv2.VideoCapture(clip['video'])
        try:
            rt.require(int(capture.get(cv2.CAP_PROP_FRAME_COUNT))==spec['total_frames']and capture.set(cv2.CAP_PROP_POS_FRAMES,index),'Original frame count/seek differs')
            ok,bgr=capture.read();rt.require(ok and bgr.dtype==np.uint8 and bgr.shape==(spec['height'],spec['width'],3)
                and int(capture.get(cv2.CAP_PROP_POS_FRAMES))==index+1,'Exact original middle frame decode required')
        finally:capture.release()
        frame_sha=hashlib.sha256(bgr.tobytes()).hexdigest()
        rgb=cv2.cvtColor(cv2.resize(bgr,(WIDTH,HEIGHT),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB)
        K=v['camera_K'].copy();K[0]*=WIDTH/spec['width'];K[1]*=HEIGHT/spec['height']
        human=target[index];obj=v['object_vertices']@v['object_rotation'][index].T+v['object_translation'][index]
        human_depth=preview.raster_depth(human,faces,K,WIDTH,HEIGHT);object_depth=preview.raster_depth(obj,v['object_faces'],K,WIDTH,HEIGHT)
        rendered=preview.overlay(rgb,human_depth,object_depth);x,y=tile_position(position)
        draw.text((x+5,y),f'EP {clip["episode_index"]:02d} | frame {index}/{spec["total_frames"]-1}',font=font,fill=(235,235,235))
        draw.text((x+5,y+16),'Original RGB',font=font,fill=(235,235,235));draw.text((x+WIDTH+5,y+16),'Mesh depth overlay',font=font,fill=(235,235,235))
        sheet.paste(Image.fromarray(rgb),(x,y+34));sheet.paste(Image.fromarray(rendered),(x+WIDTH,y+34))
        diagnostics.append(dict(episode_index=clip['episode_index'],original_frames=spec['total_frames'],frame_index=index,
            decoded_original_bgr_sha256=frame_sha,export_report_sha256=clip['export_report_sha256'],video_sha256=clip['video_sha256'],
            human_frame_sha256=hashlib.sha256(human.tobytes()).hexdigest(),human_faces_sha256=hashlib.sha256(faces.tobytes()).hexdigest(),
            camera_K_sha256=hashlib.sha256(v['camera_K'].tobytes()).hexdigest(),object_vertices_sha256=hashlib.sha256(v['object_vertices'].tobytes()).hexdigest(),object_faces_sha256=hashlib.sha256(v['object_faces'].tobytes()).hexdigest()))
        del target,faces,v,human,obj,human_depth,object_depth
    raw=encode_sheet(sheet);recheck(rt,proof);rt.require(rt.identity(cidfile,65,readonly=False)==cidpin and host_proof(root,code,revision)==proof and time.monotonic()-started<=BUDGET,'Inclusive unchanged overview source/budget proof required')
    rt.write(out/'overview.jpg',raw,0o444)
    result=dict(schema='world_reward.reconstruction_overview.v1',stage='reconstruction_overview_native',status='pass',phase='complete',producer_revision=revision,
        input_track='track_1',episodes=list(EPISODES),clips=diagnostics,source_binding=proof['source_binding'],sources=proof['files'],
        image_id=IMAGE,image=rt.identity(out/'overview.jpg'),image_hw=[sheet.height,sheet.width],viewport_hw=[HEIGHT,WIDTH],
        frame_selection='floor((T-1)/2)',geometry_operations_applied=False,original_geometry_unchanged=True,
        render='original_cpu_preview_perspective_correct_camera_z',ground_truth_used=False,manual_annotation=False,fitting=False,
        metric_evaluation=False,quality_verified=False,model_execution=False,gpu_used=False,crop_applied=False,source_rehashed_after=True,
        budget_seconds=BUDGET,elapsed_seconds=time.monotonic()-started)
    recheck(rt,proof);rt.require(host_proof(root,code,revision)==proof and time.monotonic()-started<=BUDGET,'All source and output checks inside native budget')
    rt.require(rt.identity(out/'overview.jpg',MAX_JPEG_BYTES)==result['image'],'Written overview image identity changed')
    result['elapsed_seconds']=time.monotonic()-started
    rt.write(out/'native.json',(json.dumps(result,sort_keys=True,allow_nan=False)+'\n').encode(),0o444)
    return result


def host(root,code,revision):
    started=time.monotonic();rt=runtime(code);proof=host_proof(root,code,revision)
    out=rt.canonical(root/'results'/('reconstruction-overview-'+revision));rt.require(not out.exists()and out.parent.is_dir(),'Fresh overview only')
    out.mkdir(mode=0o700);out.chmod(0o700);owner=(out.stat().st_dev,out.stat().st_ino,out.stat().st_uid)
    cidfile=out/'container.cid';name='world-reward-overview-'+revision[:12];error=None;removed=False
    def call(argv,maximum=8192):
        remaining=BUDGET-(time.monotonic()-started);rt.require(remaining>0,'Inclusive host deadline')
        r=subprocess.run(argv,capture_output=True,timeout=min(remaining,30),check=False)
        rt.require(len(r.stdout)<=maximum and len(r.stderr)<=maximum,'Bounded Docker control output');return r
    def cleanup():
        if not cidfile.exists():return False
        rt.identity(cidfile,65,readonly=False)
        raw=cidfile.read_bytes();rt.require(re.fullmatch(b'[0-9a-f]{64}\n?',raw),'Exact owned CID required');cid=raw.decode().strip()
        r=subprocess.run(['docker','inspect',cid,'--format','{{.Id}}|{{.Image}}|{{.Name}}|{{index .Config.Labels "world_reward.overview.owner"}}'],capture_output=True,timeout=5)
        absent=lambda q:q.returncode==1 and q.stdout.strip()in(b'',b'[]')and q.stderr.strip()in((f'Error: No such object: {cid}').encode(),(f'error: no such object: {cid}').encode())
        if not absent(r):
            rt.require(r.returncode==0 and r.stdout.decode().strip()==cid+'|'+IMAGE+'|/'+name+'|'+revision,'Foreign container cannot be removed')
            subprocess.run(['docker','kill','--signal','TERM',cid],capture_output=True,timeout=5)
            r=subprocess.run(['docker','inspect',cid,'--format','{{.Id}}'],capture_output=True,timeout=5)
            if not absent(r):
                rt.require(r.returncode==0 and r.stdout.decode().strip()==cid,'Owned container changed')
                subprocess.run(['docker','rm','-f',cid],capture_output=True,timeout=5)
                r=subprocess.run(['docker','inspect',cid,'--format','{{.Id}}'],capture_output=True,timeout=5)
            rt.require(absent(r),'Owned overview container survives')
        cidfile.chmod(0o444);return True
    report=dict(stage='reconstruction_overview_host',status='fail',producer_revision=revision,source_binding=proof['source_binding'],source_rehashed_after=False,owned_container_removed=False)
    try:
        r=call(['docker','image','inspect',IMAGE,'--format','{{.Id}}']);rt.require(r.returncode==0 and r.stdout.decode().strip()==IMAGE,'Qualified B47 image required')
        mounts=[]
        for p in proof['mounts']:mounts+=['--mount','type=bind,src='+p+',dst='+p+',readonly']
        command=['docker','run','--rm','--cidfile',str(cidfile),'--name',name,'--label','world_reward.overview.owner='+revision,
            '--network','none','--read-only','--memory','3g','--cpus','2','--cap-drop','ALL','--security-opt','no-new-privileges',
            '--tmpfs','/tmp:rw,noexec,nosuid,size=64m','--entrypoint','python','--env','WR_IMAGE_ID='+IMAGE,'--env','CUDA_VISIBLE_DEVICES=-1',
            '--env','OMP_NUM_THREADS=2','--env','OPENBLAS_NUM_THREADS=2','--env','MKL_NUM_THREADS=2','--env','HOME=/tmp',
            '--env','PYTHONDONTWRITEBYTECODE=1',*mounts,'--mount','type=bind,src='+str(out)+',dst='+str(out),IMAGE,
            '-I','-B',str(code/HELPERS[0]),'--native',str(root),str(code),revision,str(out)]
        remaining=BUDGET-(time.monotonic()-started);rt.require(remaining>0,'Host native budget exhausted')
        r=subprocess.run(command,capture_output=True,timeout=remaining,check=False)
        rt.require(len(r.stdout)<=8192 and len(r.stderr)<=8192 and r.returncode==0,'Native overview failed')
        removed=cleanup();rt.identity(out/'native.json',100000);result=rt.strict((out/'native.json').read_bytes())
        rt.require(result.get('status')=='pass'and result.get('source_binding')==proof['source_binding']and result.get('sources')==proof['files']
            and result.get('episodes')==list(EPISODES)and result.get('image_id')==IMAGE and result.get('source_rehashed_after')is True
            and result.get('image')==rt.identity(out/'overview.jpg',MAX_JPEG_BYTES),'Actual complete native overview required')
        flags={'phase':'complete','input_track':'track_1','original_geometry_unchanged':True,'geometry_operations_applied':False,
            'ground_truth_used':False,'manual_annotation':False,'fitting':False,'metric_evaluation':False,'quality_verified':False,
            'model_execution':False,'gpu_used':False,'crop_applied':False,'image_hw':[HEADER+ROW*7,WIDTH*4],
            'viewport_hw':[HEIGHT,WIDTH],'producer_revision':revision,'budget_seconds':BUDGET}
        rt.require(all(type(result.get(k))is type(v)and result[k]==v for k,v in flags.items())
            and type(result.get('clips'))is list and [x.get('episode_index')for x in result['clips']]==list(EPISODES)
            and [x.get('frame_index')for x in result['clips']]==[x['frame_index']for x in proof['clips']]
            and 0<result.get('elapsed_seconds',0)<=BUDGET,'Exact native overview scope/census required')
        report.update(status='pass',native_report=result,image=result['image'])
    except Exception as exc:error=exc;report['error_type']=type(exc).__name__
    finally:
        try:removed=cleanup()or removed
        except Exception as exc:error=error or exc;report['cleanup_error_type']=type(exc).__name__
        try:report['source_rehashed_after']=host_proof(root,code,revision)==proof
        except Exception as exc:error=error or exc;report['post_error_type']=type(exc).__name__
        rt.require((out.stat().st_dev,out.stat().st_ino,out.stat().st_uid)==owner,'Owned output namespace replaced')
        report.update(owned_container_removed=removed,elapsed_seconds=time.monotonic()-started)
        if error or not removed or not report['source_rehashed_after']or report['elapsed_seconds']>BUDGET:report['status']='fail'
        rt.write(out/'report.json',(json.dumps(report,sort_keys=True,allow_nan=False)+'\n').encode(),0o444)
        if report['status']=='pass':out.chmod(0o555)
    if report['status']!='pass':raise ValueError('Overview failed; immutable receipt retained')from None
    return report


def main():
    p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--native',nargs=4);p.add_argument('--host',nargs=3);a=p.parse_args()
    if bool(a.native)==bool(a.host):raise ValueError('Exactly one source-bound mode required')
    if a.native:
        r,c,rev,o=a.native;native(Path(r),Path(c),rev,Path(o))
    else:
        r,c,rev=a.host
        if sys.platform!='linux'or os.geteuid()!=0 or os.environ.get('DOCKER_HOST')!='unix://'+r+'/docker.sock':raise ValueError('Actual Azure CPU host required')
        def cancelled(*_):raise TimeoutError('Overview host cancellation')
        signal.signal(signal.SIGTERM,cancelled);signal.signal(signal.SIGINT,cancelled)
        host(Path(r),Path(c),rev)


if __name__=='__main__':main()
