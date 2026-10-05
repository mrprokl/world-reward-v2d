"""Private HO-Cap semantic-mask diagnostic AFTER both frozen Boots predictions.

No physical/material identity, contact, calibrated confidence or 3D claim.
Only the three declared reference arrays are decoded; models are never imported.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import time
import zipfile

ROOT=Path('/srv/scenesmith/world-reward'); ENTRY='run_hocap_point_retention_eval'
PROTOCOL='configs/hocap_point_retention_protocol_v1.json'
PROTOCOL_PIN={'bytes':11108,'sha256':'9a7d9650639590a50c4c907b84bee4b6bb09f89209f450bac20ebad4489678a3'}
IMAGE='sha256:ef12f589dd270e56be3a2d2e2f33ccd356e5b160a5c6ca03b8a9449ccc10d1e4'
PREDICTIONS='validation/hocap_boots_track_v1'; BANK='validation/hocap_amg_bank_v1'
HELPERS=('infra/hocap_point_retention_eval.py','infra/run_hocap_point_retention_eval.sh',
         'infra/mediapipe_cpu_runtime_verify.py',PROTOCOL,'configs/robotap_boots_inference_pins.json')
CATALOG=tuple(f'G{group:02d}_{k}' for group in (1,2,4,5,6,7,9,10,11,15,16,18,19,20,21,22)
              for k in range(1,5))+('RIGHT_HAND','LEFT_HAND')
LABEL_KEYS={'seg_mask','obj_class_inds','obj_class_names'}
MAX_REPORT=2<<20; MAX_PRED=1<<30; BUDGET=300; GRACE=60


def require(ok,message):
    if not ok: raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def runtime(code):
    import importlib.util
    path=Path(code)/'infra/mediapipe_cpu_runtime_verify.py'
    require(path.resolve()==path and not path.is_symlink() and not path.stat().st_mode&0o222,'Readonly real runtime helper required')
    spec=importlib.util.spec_from_file_location('hocap_retention_rt',path)
    rt=importlib.util.module_from_spec(spec);spec.loader.exec_module(rt);return rt


def facts(value,expected):
    require(all(type(value.get(k))is type(v) and value[k]==v for k,v in expected.items()),'Complete frozen producer facts differ')


def check(deadline):
    if time.monotonic()>=deadline: raise TimeoutError('Inclusive semantic evaluation deadline exhausted')


def source(rt,code,revision):
    require(Path(__file__).resolve()==code/HELPERS[0] and {p.name for p in code.parent.iterdir()}=={'code','revision','source-sha256'},'Actual bounded immutable source parent required')
    return rt.source(ROOT,code,revision,ENTRY,HELPERS)


def recheck(rt,files):
    for path,pin in files.items():require(rt.identity(Path(path),max(MAX_PRED,pin['bytes']))==pin,'Frozen source/prediction/reference changed')


def authenticate(rt,code,native_pin,host_pin):
    """All public provenance and BOTH complete arrays BEFORE even stat/open of labels."""
    p=rt.pinned(code/PROTOCOL,PROTOCOL_PIN,16384);folder=rt.canonical(ROOT/PREDICTIONS)
    require(stat.S_IMODE(folder.stat().st_mode)==0o555 and {v.name for v in folder.iterdir()}=={
        'report.json','host.json','proof.json','.container.cid',*(c['clip']+'_tracks.npz' for c in p['cohort']['clips'])},'Exact sealed two-clip output inventory required')
    native=rt.pinned(folder/'report.json',native_pin,MAX_REPORT);host=rt.pinned(folder/'host.json',host_pin,MAX_REPORT)
    facts(native,dict(schema='world_reward.hocap_boots_tracks.v1',stage='hocap_native_all_seed_full_t_tracks',status='pass',phase='complete',
        image_id=IMAGE,model_loads=1,rgb_decodes=1493,native_calls_attempted=2,native_calls_returned=2,native_calls_completed=2,
        all_original_jpegs_decoded_before_model=True,all_source_public_bank_assets_rehashed_after=True,private_labels_read=False,
        opaque_metadata_read=False,calibration_read=False,ground_truth_used=False,challenge_inputs_used=False,oracle_modes=[],quality_verified=False,adoption=False))
    revision=native.get('producer_revision');require(type(revision)is str and re.fullmatch('[0-9a-f]{40}',revision),'Frozen prediction producer required')
    facts(host,dict(stage='hocap_boots_host',status='pass',producer_revision=revision,image_id=IMAGE,native_exit_status=0,
        native_report_identity=native_pin,owned_cleanup_verified=True,source_public_bank_assets_rehashed_after=True,private_labels_read=False,quality_verified=False,adoption=False))
    require(type(native.get('elapsed_seconds'))in(int,float) and 0<native['elapsed_seconds']<=1800,'Original prediction deadline required')
    proof_pin=native['proof_identity'];proof=rt.pinned(folder/'proof.json',proof_pin,MAX_REPORT)
    original=ROOT/'jobs'/revision/'run_hocap_boots_track/code';sb=proof['source_binding']
    require({'infra/hocap_boots_track.py','infra/run_hocap_boots_track.sh',PROTOCOL,
        'infra/robotap_boots_infer.py','infra/tracker_noise_experiment.py'}<=set(sb['helpers']),'Real tracker source helpers required')
    require({v.name for v in original.parent.iterdir()}=={'code','revision','source-sha256'} and
        rt.source(ROOT,original,revision,'run_hocap_boots_track',tuple(sb['helpers']))==sb and digest(sb)==native['source_binding_sha256'],
        'Original whole tracker source/markers differ; no original code executed')
    manifest_pin={k:p['input']['manifest'][k] for k in ('bytes','sha256')}
    require(proof['retention_protocol_identity']==PROTOCOL_PIN and proof['public_manifest_identity']==manifest_pin
        and native['public_manifest_identity']==manifest_pin,'Original public/scientific input pins differ')
    old_protocol=rt.pinned(original/'configs/hocap_boots_protocol_v1.json',proof['protocol_identity'],16384)
    require(native['native_parameters']==old_protocol['native'] and old_protocol['retention_protocol']==PROTOCOL_PIN
        and old_protocol['frames']==[c['frames'] for c in p['cohort']['clips']],'Original native numerical policy differs')
    files={str(folder/'report.json'):native_pin,str(folder/'host.json'):host_pin,str(folder/'proof.json'):proof_pin,
        str(code/PROTOCOL):PROTOCOL_PIN,str(original/'configs/hocap_boots_protocol_v1.json'):proof['protocol_identity']}
    public=rt.pinned(Path(p['input']['manifest']['path']),manifest_pin,1<<20)
    require(public['subject']==p['cohort']['subject'] and public['clips']==[dict(clip=c['clip'],camera=p['cohort']['camera'],num_frames=c['frames']) for c in p['cohort']['clips']], 'Original full public timeline differs')
    expected=[(c['clip'],i)for c in p['cohort']['clips']for i in range(c['frames'])]
    require([(r['clip'],r['frame_position'])for r in public['images']]==expected
        and all(r['source_frame_id']==r['frame_position'] and r['camera']==p['cohort']['camera']for r in public['images']), 'Full original source-frame continuity required')
    files[p['input']['manifest']['path']]=manifest_pin
    bank=proof['bank'];bdir=rt.canonical(ROOT/BANK)
    br=rt.pinned(bdir/'report.json',bank['report'],MAX_REPORT);bh=rt.pinned(bdir/'host.json',bank['host'],MAX_REPORT)
    safe=rt.pinned(bdir/'sanitized_input_proof.json',bank['sanitized'],MAX_REPORT)
    facts(br,dict(status='pass',phase='complete',stage='hocap_native_all_mask_bank',all_native_returned_masks=True,
        private_labels_read=False,ground_truth_used=False,amg_calls=10,anchors_retained=10,original_source_public_models_rehashed_after=True))
    facts(bh,dict(status='pass',stage='hocap_amg_host',native_report_identity=bank['report'],owned_cleanup_verified=True,source_public_rehashed_after=True))
    require(native['bank_report_identity']==bank['report'] and br['banks']==bank['banks'] and len(br['banks'])==10
        and safe['source_binding']==bank['source_binding'] and digest(safe['source_binding'])==br['source_binding_sha256'], 'Frozen automatic bank ancestry differs')
    files.update({str(bdir/'report.json'):bank['report'],str(bdir/'host.json'):bank['host'],str(bdir/'sanitized_input_proof.json'):bank['sanitized']})
    for row in bank['banks']:
        name=row['file'];require(re.fullmatch(r'bank/20231027_(112303|113202)_anchor_[0-9]{6}\.npz',name),'Bounded public bank filename required')
        pin={k:row[k] for k in ('bytes','sha256')};require(rt.identity(bdir/name,MAX_PRED)==pin,'Original bank bytes differ');files[str(bdir/name)]=pin
    rp_path=code/'configs/robotap_boots_inference_pins.json';rp_pin=rt.identity(rp_path,100000)
    require(rp_pin==rt.identity(original/'configs/robotap_boots_inference_pins.json',100000), 'Original runtime config unchanged')
    rp=rt.pinned(rp_path,rp_pin,100000)['runtime'];files[str(rp_path)]=rp_pin
    rpath=ROOT/rp['report_path'];rpin={k:rp[k] for k in ('bytes','sha256')};rr=rt.pinned(rpath,rpin,MAX_REPORT)
    facts(rr,dict(status='pass',phase='complete',stage='bootstapir_runtime_verify',child_image_id=IMAGE,source_rehashed_after=True,
        gpu_execution=False,checkpoint_read=False,rgb_or_labels_read=False))
    require(proof['runtime']['runtime']==rpin and proof['runtime']['native_sources']==rr['native_sources']
        and rr['cpu_import']['numpy']=='1.26.3' and rr['cpu_import']['cuda_initialized']is False,'Genuine same CPU/native runtime evidence required')
    files[str(rpath)]=rpin
    rows=native.get('predictions');require(type(rows)is list and len(rows)==2,'Both predictions required before reference opening')
    for row,clip in zip(rows,p['cohort']['clips']):
        require(row['clip']==clip['clip'] and row['frames']==clip['frames'] and row['file']==clip['clip']+'_tracks.npz'
            and type(row['queries'])is int and 0<=row['queries']<=4096,'Full original fixed query clip required')
        pin={k:row[k] for k in ('bytes','sha256')};require(rt.identity(folder/row['file'],MAX_PRED)==pin,'Both complete prediction byte seals required');files[str(folder/row['file'])]=pin
    return dict(protocol=p,native=native,host=host,proof=proof,files=files,original=original,folder=folder,runtime_path=rpath)


def prediction_arrays(np,path,row,banks):
    with np.load(path,allow_pickle=False)as z:a={n:z[n] for n in z.files}
    q,t,m=row['queries'],row['frames'],row['seeds']
    shapes={'tracks':(np.float32,(q,t,2)),'tracks_256':(np.float32,(q,t,2)),
        'occlusion':(np.float32,(q,t)),'expected_dist':(np.float32,(q,t)),'visible':(np.bool_,(q,t)),
        'query_points':(np.float64,(q,3)),'point_indices':(np.int64,(q,)),'frame_index':(np.int64,(t,)),
        'query_owner_ids':(np.dtype('U128'),(q,)),'query_owner_indices':(np.int64,(q,)),
        'query_birth_frame_indices':(np.int64,(q,)),'seed_ids':(np.dtype('U128'),(m,)),
        'seed_birth_frame_indices':(np.int64,(m,)),'seed_mask_sha256':(np.dtype('U64'),(m,)),
        'seed_native_mask_indices':(np.int64,(m,)),'seed_query_offsets':(np.int64,(m+1,)),
        'seed_anchor_offsets':(np.int64,(6,)),'anchor_frame_indices':(np.int64,(5,)),'image_size':(np.int64,(2,))}
    require(set(a)==set(shapes),'Exact public prediction fields required')
    for n,(dtype,shape)in shapes.items():
        require(a[n].dtype==dtype and a[n].shape==shape and not np.ma.isMaskedArray(a[n]),'Lossless original prediction dtype/shape required')
        if a[n].dtype.kind=='f':require(np.isfinite(a[n]).all(),'Finite original raw predictions required')
    h,w=a['image_size'];off=a['seed_query_offsets'];owners=a['query_owner_indices'];birth=a['query_birth_frame_indices']
    require(min(h,w)>0 and np.array_equal(a['frame_index'],np.arange(t,dtype=np.int64))
        and np.array_equal(a['point_indices'],np.arange(q,dtype=np.int64)) and off[0]==0 and off[-1]==q and np.all(np.diff(off)>=0)
        and np.array_equal(owners,np.repeat(np.arange(m),np.diff(off))) and np.array_equal(a['query_owner_ids'],a['seed_ids'][owners])
        and np.array_equal(birth,a['seed_birth_frame_indices'][owners]) and np.array_equal(a['query_points'][:,0],birth)
        and np.array_equal(a['anchor_frame_indices'],np.arange(5)*(t-1)//4)
        and np.all((birth>=0)&(birth<t)) and row['zero_query_seeds']==int((np.diff(off)==0).sum()),'Original complete seeds/births/timeline differ')
    require(np.array_equal(a['tracks'],a['tracks_256']*np.array([w,h],np.float32)/np.array([256,256],np.float32)), 'Native original-image conversion differs')
    # Public original bank arrays bind every query/seed, including empty seeds.
    fields={'query_points':'query_points','seed_ids':'mask_ids','seed_birth_frame_indices':'query_birth_frame_indices',
        'seed_mask_sha256':'mask_sha256','seed_native_mask_indices':'native_mask_indices'}
    loaded=[]
    for path,bank in banks:
        with np.load(path,allow_pickle=False)as z:
            loaded.append({k:z[k] for k in ('query_points','mask_ids','mask_sha256','native_mask_indices','anchor_frame_index','query_offsets')})
    require(len(loaded)==5,'All five original anchor banks required')
    for target,key in fields.items():
        if target=='seed_birth_frame_indices':v=np.concatenate([np.full(len(b['mask_ids']),b['anchor_frame_index'],np.int64)for b in loaded])
        else:v=np.concatenate([b[key]for b in loaded])
        require(np.array_equal(a[target],v),'Prediction queries/seeds changed from original automatic bank')
    require(np.array_equal(a['seed_anchor_offsets'],np.r_[0,np.cumsum([len(b['mask_ids'])for b in loaded])])
        and np.array_equal(off,np.r_[0,np.cumsum(np.concatenate([np.diff(b['query_offsets'])for b in loaded]))]), 'Empty original seed/query offsets changed')
    for value in a.values():value.setflags(write=False)
    return a


class ReferenceError(ValueError):pass


def label_arrays(np,seg,indices,names,shape,catalog=CATALOG):
    """Factory's sorted positive SEG -> corresponding class rows (NOT class+1)."""
    def valid(ok,why):
        if not ok:raise ReferenceError(why)
    valid(type(seg)is np.ndarray and seg.dtype.kind in 'iu' and seg.ndim==2 and seg.shape==shape,'SEG integer/original-grid schema')
    valid(type(indices)is np.ndarray and indices.dtype.kind in 'iu' and indices.ndim==1,'Class integer row schema')
    valid(type(names)is np.ndarray and names.dtype.kind in 'US' and names.shape==indices.shape,'Class names non-pickle fixed-text schema')
    valid(not np.ma.isMaskedArray(seg) and not np.ma.isMaskedArray(indices) and not np.ma.isMaskedArray(names),'Unmasked native labels required')
    valid(np.all(seg>=0),'Negative SEG')
    positive=np.unique(seg);positive=positive[positive>0]
    valid(len(positive)==len(indices),'Positive SEG/class row cardinalities differ')
    text=[]
    for n in names:
        try:v=bytes(n).decode('utf-8')if names.dtype.kind=='S'else str(n)
        except UnicodeError as e:raise ReferenceError('Class name decoding')from e
        text.append(v)
    # The frozen numerical protocol explicitly requires unique semantic rows.
    # Repeated classes are therefore INCONCLUSIVE, never a quality rejection.
    valid(len(set(int(x)for x in indices))==len(indices),'Duplicate semantic rows: frozen reference qualification inconclusive')
    valid(all(0<=int(i)<len(catalog) and catalog[int(i)]==n for i,n in zip(indices,text)),'Unknown/conflicting primary catalog index/name')
    return {int(s):int(c)for s,c in zip(positive,indices)}


def read_label(np,archive,name,shape):
    item=archive.getinfo(name)
    if item.is_dir() or item.file_size>64<<20 or item.flag_bits&1:raise ReferenceError('Missing/unsafe/bounded label member')
    try:
        with archive.open(item)as stream, np.load(stream,allow_pickle=False)as z:
            if not LABEL_KEYS<=set(z.files):raise ReferenceError('Missing permitted reference fields')
            seg,indices,names=(z[k]for k in ('seg_mask','obj_class_inds','obj_class_names'))
    except (OSError,EOFError,zipfile.BadZipFile)as e:raise ReferenceError('Unreadable permitted reference arrays')from e
    return seg,label_arrays(np,seg,indices,names,shape)


def sample_classes(np,seg,mapping,xy):
    finite=np.isfinite(xy).all(axis=1);inside=finite&(xy[:,0]>=0)&(xy[:,0]<seg.shape[1])&(xy[:,1]>=0)&(xy[:,1]<seg.shape[0])
    result=np.full(len(xy),-2,np.int64);ii=np.flatnonzero(inside)
    pixels=np.floor(xy[ii]).astype(np.int64);s=seg[pixels[:,1],pixels[:,0]]
    result[ii]=np.array([mapping.get(int(x),-1)if x else -1 for x in s],np.int64)
    return result,inside


def qualify_clip(np,a,read,*,deadline):
    """Reference-only streaming pass; NO scores until the whole cohort qualifies."""
    t=len(a['frame_index']);q=len(a['point_indices']);m=len(a['seed_ids']);birth=a['query_birth_frame_indices']
    ref=np.full(q,-3,np.int64);errors=[];catalogs=[];shape=tuple(a['image_size'])
    for frame in range(t):
        check(deadline)
        try:
            seg,mapping=read(frame,shape);catalogs.append(set(mapping.values()))
            ix=np.flatnonzero(birth==frame)
            if len(ix):ref[ix]=sample_classes(np,seg,mapping,a['query_points'][ix][:,[2,1]])[0]
        except (ReferenceError,ValueError,KeyError,zipfile.BadZipFile,OSError,EOFError)as e:
            errors.append({'frame_index':frame,'reason':str(e)[:120]});catalogs.append(set())
    return dict(reference_schema_qualified=not errors,reference_error_count=len(errors),reference_errors=errors,
                frames=t,queries=q,seeds=m,reference=ref,catalogs=catalogs)


def evaluate_clip(np,a,read,*,deadline,qualified=None):
    """All future slots, never missing-visibility/class/frame denominator dropping."""
    t=len(a['frame_index']);q=len(a['point_indices']);m=len(a['seed_ids']);birth=a['query_birth_frame_indices']
    qualified=qualify_clip(np,a,read,deadline=deadline)if qualified is None else qualified
    if not qualified['reference_schema_qualified']:return {k:v for k,v in qualified.items()if k not in ('reference','catalogs')}
    ref=qualified['reference'];catalogs=qualified['catalogs'];shape=tuple(a['image_size'])
    require(np.all(ref!=-3),'All original seed reference births must be processed')
    object_ref=(ref>=0)&(ref<len(CATALOG)-2);possible=np.maximum(t-1-birth,0)
    counts={branch:{k:np.zeros(q,np.int64)for k in ('possible','correct','wrong_object','background','HAND','hidden','outgrid','reference_absent','supported')}for branch in ('A','B')}
    perframe=[];longest=np.zeros(q,np.int64);runs=np.zeros(q,np.int64)
    for frame in range(t):
        check(deadline);seg,mapping=read(frame,shape);active=object_ref&(frame>birth);idx=np.flatnonzero(active)
        fr={'frame_index':frame,'all_query_future_possible':int((frame>birth).sum()),'object_reference_possible':len(idx)}
        future=frame>birth;visible_all=a['visible'][future,frame]
        fr.update(native_visible_all_future=int(visible_all.sum()),native_visible_all_future_fraction=float(visible_all.mean())if len(visible_all)else None,
            native_visible_object_future=int(a['visible'][idx,frame].sum()),native_visible_object_future_fraction=float(a['visible'][idx,frame].mean())if len(idx)else None)
        for branch in ('A','B'):
            xy=a['query_points'][idx][:,[2,1]] if branch=='A'else a['tracks'][idx,frame]
            visibility=np.ones(len(idx),bool)if branch=='A'else a['visible'][idx,frame]
            cls,inside=sample_classes(np,seg,mapping,xy);support=visibility&inside
            right=support&(cls==ref[idx]);other=support&(cls>=0)&(cls<len(CATALOG)-2)&(cls!=ref[idx])
            flags=dict(possible=np.ones(len(idx),bool),correct=right,wrong_object=other,background=support&(cls==-1),
                HAND=support&(cls>=len(CATALOG)-2),hidden=~visibility,outgrid=visibility&~inside,
                reference_absent=np.array([int(ref[i])not in catalogs[frame]for i in idx]),supported=support)
            for k,v in flags.items():counts[branch][k][idx]+=v.astype(np.int64);fr[k+'_'+branch]=int(v.sum())
            if branch=='B':
                runs[idx]=np.where(right,0,runs[idx]+1);longest[idx]=np.maximum(longest[idx],runs[idx])
        perframe.append(fr)
    denom=int(possible[object_ref].sum());require(all(int(v['possible'].sum())==denom for v in counts.values()),'No possible object slot may disappear')
    branches={b:{k:int(v.sum())for k,v in c.items()}for b,c in counts.items()}
    for b in branches:
        branches[b]['correct_fraction']=branches[b]['correct']/denom if denom else None
        branches[b]['wrong_object_fraction']=branches[b]['wrong_object']/denom if denom else None
    query_rows=[dict(point_index=i,seed_id=str(a['query_owner_ids'][i]),birth_frame_index=int(birth[i]),reference_class_index=int(ref[i]),
        reference_class_name=CATALOG[ref[i]]if ref[i]>=0 else 'BACKGROUND'if ref[i]==-1 else 'OUTGRID',future_possible=int(possible[i]),
        object_reference=bool(object_ref[i]),A={k:int(v[i])for k,v in counts['A'].items()},B={k:int(v[i])for k,v in counts['B'].items()},
        longest_B_incorrect_gap=int(longest[i]))for i in range(q)]
    seeds=[]
    for i,name in enumerate(a['seed_ids']):
        ix=np.flatnonzero(a['query_owner_indices']==i);d=int(possible[ix[object_ref[ix]]].sum())
        seeds.append(dict(seed_id=str(name),anchor=int(a['seed_birth_frame_indices'][i]),queries=len(ix),
            status='NO_QUERIES'if not len(ix) else 'NO_FUTURE'if int(a['seed_birth_frame_indices'][i])==t-1 else 'OBSERVED',possible=d,
            A_correct_fraction=int(counts['A']['correct'][ix].sum())/d if d else None,
            B_correct_fraction=int(counts['B']['correct'][ix].sum())/d if d else None,
            reference_histogram={CATALOG[r]if r>=0 else 'BACKGROUND'if r==-1 else 'OUTGRID':int((ref[ix]==r).sum())for r in np.unique(ref[ix])}))
    macro={b:float(np.mean([s[b+'_correct_fraction']for s in seeds if s['possible']]))if any(s['possible']for s in seeds)else None for b in ('A','B')}
    anchors=[]
    for anchor in a['anchor_frame_indices']if 'anchor_frame_indices'in a else np.unique(a['seed_birth_frame_indices']):
        ix=np.flatnonzero(object_ref&(birth==anchor));d=int(possible[ix].sum())
        anchors.append(dict(anchor=int(anchor),possible=d,**{b+'_correct_fraction':int(counts[b]['correct'][ix].sum())/d if d else None for b in ('A','B')}))
    anchor_macro={b:float(np.mean([r[b+'_correct_fraction']for r in anchors if r['possible']]))if any(r['possible']for r in anchors)else None for b in ('A','B')}
    classes=[]
    for cls in np.unique(ref[object_ref]):
        ix=np.flatnonzero(ref==cls);d=int(possible[ix].sum())
        classes.append(dict(class_index=int(cls),class_name=CATALOG[cls],queries=len(ix),possible=d,
            **{b:{k:int(v[ix].sum())for k,v in counts[b].items()}for b in ('A','B')}))
    minima={b:min((f['correct_'+b]/f['object_reference_possible']for f in perframe if f['object_reference_possible']),default=None)for b in ('A','B')}
    return dict(reference_schema_qualified=True,reference_error_count=0,frames=t,queries=q,seeds=m,object_reference_queries=int(object_ref.sum()),
        initial_BACKGROUND_queries=int((ref==-1).sum()),initial_HAND_queries=int((ref>=len(CATALOG)-2).sum()),initial_outgrid_queries=int((ref==-2).sum()),
        all_query_future_possible=int(possible.sum()),possible=denom,A=branches['A'],B=branches['B'],seed_macro=macro,
        anchor_macro=anchor_macro,anchor_diagnostics=anchors,class_coverage=classes,
        per_frame_correct_fraction_minimum=minima,per_frame=perframe,per_query=query_rows,seed_diagnostics=seeds)


def decision(clips):
    if len(clips)!=2 or any(not c['reference_schema_qualified'] or c['possible']==0 for c in clips):return 'INCONCLUSIVE'
    return 'QUALIFIED_SEMANTIC_MASK_POINT_RETENTION'if all(c['B']['correct']>c['A']['correct'] and c['B']['wrong_object']<=c['A']['wrong_object']for c in clips)else 'REJECT'


def evaluate(rt,code,proof,deadline):
    import numpy as np
    require(np.__version__=='1.26.3' and not any(n in sys.modules for n in ('torch','mediapipe','sam2','tapnet')),'Original CPU-only numerical runtime, no model imports')
    p=proof['protocol'];arrays=[]
    for row in proof['native']['predictions']:
        banks=[(ROOT/BANK/r['file'],r)for r in proof['proof']['bank']['banks']if r['clip']==row['clip']]
        arrays.append(prediction_arrays(np,proof['folder']/row['file'],row,banks));check(deadline)
    # This is the FIRST access to any private reference path, after both complete
    # prediction arrays/source/bank/runtime have passed all public checks.
    private=p['private_evaluation']['archive'];archive=rt.canonical(Path(private['path']));pin={k:private[k]for k in ('bytes','sha256')}
    # A leaf-only bind has a synthetic container parent. Original parent700 is
    # authenticated on host; never require/mutate its container surrogate.
    require(stat.S_IMODE(archive.stat().st_mode)==0o400 and rt.identity(archive,2<<30)==pin,
        'Independently frozen original private archive/permissions differ')
    proof['files'][str(archive)]=pin;clips=[]
    with zipfile.ZipFile(archive)as z:
        names=[v.filename for v in z.infolist()]
        if len(names)!=len(set(names)):
            return dict(clips=[dict(clip=r['clip'],frames=r['frames'],queries=r['queries'],reference_schema_qualified=False,
                reference_error_count=1,reference_errors=[{'reason':'Duplicate ZIP member names are ambiguous'}])for r in proof['native']['predictions']],
                decision='INCONCLUSIVE',private_archive_identity=pin,private_keys_accessed=[],semantic_class_not_physical_identity=True,
                query_reference_rematching=False,static_A_is_diagnostic_not_motion_prediction=True)
        qualified=[];readers=[]
        for a,row in zip(arrays,proof['native']['predictions']):
            prefix=f"{p['cohort']['subject']}/{row['clip']}/{p['cohort']['camera']}/"
            expected={prefix+f'label_{t:06d}.npz'for t in range(row['frames'])}
            selected={n for n in names if n.startswith(prefix) and re.fullmatch(re.escape(prefix)+r'label_[0-9]{6}\.npz',n)}
            reader=lambda t,shape,prefix=prefix:read_label(np,z,prefix+f'label_{t:06d}.npz',shape)
            readers.append(reader);info=qualify_clip(np,a,reader,deadline=deadline)
            if selected!=expected:
                info['reference_schema_qualified']=False;info['reference_error_count']+=1
                info['reference_errors'].append({'reason':'Exact selected full original label-member inventory differs'})
            qualified.append(info)
        # Both schemas are closed before computing a single metric. If either
        # fails, retain every clip/reference failure, not scores for the other.
        all_qualified=all(v['reference_schema_qualified']for v in qualified)
        for a,row,reader,info in zip(arrays,proof['native']['predictions'],readers,qualified):
            result=evaluate_clip(np,a,reader,deadline=deadline,qualified=info)if all_qualified else {
                k:v for k,v in info.items()if k not in ('reference','catalogs')}
            result['clip']=row['clip'];clips.append(result)
    return dict(clips=clips,decision=decision(clips),private_archive_identity=pin,private_keys_accessed=sorted(LABEL_KEYS),
        semantic_class_not_physical_identity=True,query_reference_rematching=False,static_A_is_diagnostic_not_motion_prediction=True)


def command(argv,seconds=10):
    r=subprocess.run(argv,capture_output=True,timeout=seconds,check=False)
    require(len(r.stdout)<=32768 and len(r.stderr)<=32768,'Bounded Docker control output required');return r


def publish(rt,path,value,deadline):
    raw=(json.dumps(value,sort_keys=True,allow_nan=False)+'\n').encode();require(len(raw)<=32<<20,'Bounded diagnostic JSON required')
    with path.open('xb')as stream:
        try:
            stream.write(raw);stream.flush();os.fsync(stream.fileno());check(deadline);os.fchmod(stream.fileno(),0o444)
        except Exception:
            value={**value,'status':'fail','phase':'receipt_sealing','error_type':'PublicationDeadlineOrWriteFailure'}
            stream.seek(0);stream.truncate();stream.write((json.dumps(value,sort_keys=True,allow_nan=False)+'\n').encode())
            stream.flush();os.fsync(stream.fileno());os.fchmod(stream.fileno(),0o444);raise


def cleanup(out,name,revision):
    cidfile=out/'container.cid';rt=runtime(Path(os.environ['WR_CODE']))
    rt.identity(cidfile,65,readonly=False);raw=cidfile.read_bytes();require(re.fullmatch(b'[0-9a-f]{64}\n?',raw),'Exact owned CID required');cid=raw.decode().strip()
    r=command(['docker','inspect',cid,'--format','{{.Id}}|{{.Image}}|{{.Name}}|{{index .Config.Labels "world_reward.retention.owner"}}'])
    absent=lambda x:x.returncode==1 and x.stdout.strip()in(b'',b'[]') and x.stderr.strip()in(
        (f'Error: No such object: {cid}'.encode(),f'error: no such object: {cid}'.encode()))
    if not absent(r):
        require(r.returncode==0 and r.stderr==b'' and r.stdout.decode().strip()==cid+'|'+IMAGE+'|/'+name+'|'+revision,'Only exact owned container can be removed')
        removed=command(['docker','rm','-f',cid],15);require(removed.returncode==0 and removed.stdout.decode().strip()==cid,'Owned removal failed')
    require(absent(command(['docker','inspect',cid])),'Independent exact CID absence required')
    cidfile.chmod(0o444)


def main(argv=None):
    ap=argparse.ArgumentParser(allow_abbrev=False);mode=ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--dispatch',action='store_true');mode.add_argument('--native',action='store_true')
    for name in ('prediction-report','prediction-host'):
        ap.add_argument('--'+name+'-bytes',type=int,required=True);ap.add_argument('--'+name+'-sha256',required=True)
    args=ap.parse_args(argv);code=Path(os.environ['WR_CODE']);revision=os.environ['WR_CODE_REVISION'];rt=runtime(code)
    require(sys.platform=='linux' and os.geteuid()==0 and os.environ.get('WR_ROOT')==str(ROOT),'Owned VM02 Linux CPU required')
    started=time.monotonic();deadline=float(os.environ['WR_RETENTION_DEADLINE'])if args.native else started+BUDGET
    def expired(*_):raise TimeoutError('Inclusive CPU evaluation expired')
    signal.signal(signal.SIGALRM,expired);signal.signal(signal.SIGTERM,expired);signal.alarm(max(1,math.ceil(deadline-started)))
    original=source(rt,code,revision);npin=dict(bytes=args.prediction_report_bytes,sha256=args.prediction_report_sha256)
    hpin=dict(bytes=args.prediction_host_bytes,sha256=args.prediction_host_sha256)
    proof=authenticate(rt,code,npin,hpin);out=ROOT/'results'/('hocap-point-retention-'+revision)
    if args.native:
        require(os.environ.get('CUDA_VISIBLE_DEVICES')=='-1' and os.environ.get('WR_IMAGE_ID')==IMAGE
            and {p.name for p in Path('/sys/class/net').iterdir()}=={'lo'} and out.is_dir()
            and {p.name for p in out.iterdir()}=={'container.cid'},'Fresh offline CPU native output required')
        result=evaluate(rt,code,proof,deadline);recheck(rt,proof['files']);require(source(rt,code,revision)==original,'Source changed after reference evaluation');check(deadline)
        result.update(schema='world_reward.hocap_point_retention_eval.v1',stage='hocap_point_retention_native',status='pass',phase='complete',
            producer_revision=revision,source_binding=original,prediction_report_identity=npin,prediction_host_identity=hpin,
            source_prediction_reference_rehashed_after=True,gpu_used=False,model_executed=False,quality_3D_verified=False,adoption=False,
            input_bindings=proof['files'],image_id=IMAGE,
            budget_seconds=BUDGET,elapsed_seconds=BUDGET-(deadline-time.monotonic()))
        publish(rt,out/'native.json',result,deadline);signal.alarm(0);return result
    require(os.uname().nodename=='world-reward-ncc-h100-02' and not out.exists(),'Fresh exact VM02 namespace required');out.mkdir(mode=0o700)
    owner=(out.stat().st_dev,out.stat().st_ino,out.stat().st_uid);name='world-reward-retention-'+revision[:12];failure=None;removed=False;native_exit=None;stderr_tail=None;private=None;private_pin=None
    try:
        image=command(['docker','image','inspect',IMAGE,'--format','{{.Id}}']);require(image.returncode==0 and image.stdout.decode().strip()==IMAGE,'Qualified existing CPU image required, no pull')
        private=Path(proof['protocol']['private_evaluation']['archive']['path'])
        private=rt.canonical(private);private_pin={k:proof['protocol']['private_evaluation']['archive'][k]for k in ('bytes','sha256')}
        require(stat.S_IMODE(private.parent.stat().st_mode)==0o700 and stat.S_IMODE(private.stat().st_mode)==0o400
            and rt.identity(private,2<<30)==private_pin,'Original host-only quarantine permissions and opaque archive hash differ')
        paths=[code.parent,proof['original'].parent,proof['folder'],ROOT/BANK/'report.json',ROOT/BANK/'host.json',
            ROOT/BANK/'sanitized_input_proof.json',ROOT/BANK/'bank',proof['runtime_path'],Path(proof['protocol']['input']['manifest']['path']),private]
        require(len(paths)==len(set(paths)) and not any(','in str(p) or '\n'in str(p) for p in paths),'Exact narrow mounts required')
        cmd=['docker','run','--rm','--name',name,'--cidfile',str(out/'container.cid'),'--label','world_reward.retention.owner='+revision,
            '--network','none','--read-only','--user','0:0','--cap-drop','ALL','--cap-add','DAC_READ_SEARCH','--security-opt','no-new-privileges',
            '--memory','8g','--cpus','4','--tmpfs','/tmp:rw,noexec,nosuid,size=256m','--entrypoint','/usr/bin/env']
        for path in paths:cmd+=['--mount',f'type=bind,src={path},dst={path},readonly']
        cmd+=['--mount',f'type=bind,src={out},dst={out}',IMAGE,'-i','PATH=/opt/conda/bin:/usr/bin:/bin','HOME=/tmp',
            'CUDA_VISIBLE_DEVICES=-1','WR_IMAGE_ID='+IMAGE,'WR_CODE='+str(code),'WR_CODE_REVISION='+revision,'WR_ROOT='+str(ROOT),
            'WR_RETENTION_DEADLINE='+format(deadline,'.17g'),
            'PYTHONDONTWRITEBYTECODE=1','OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1','/opt/conda/bin/python','-I','-B',str(code/HELPERS[0]),
            '--native','--prediction-report-bytes',str(npin['bytes']),'--prediction-report-sha256',npin['sha256'],
            '--prediction-host-bytes',str(hpin['bytes']),'--prediction-host-sha256',hpin['sha256']]
        r=subprocess.run(cmd,capture_output=True,timeout=max(.1,deadline-time.monotonic()),check=False)
        native_exit=r.returncode;stderr_tail=r.stderr[-4096:].decode('utf-8','replace')if r.returncode else None
        require(r.returncode==0 and len(r.stdout)<=8192 and len(r.stderr)<=32768,'Native semantic evaluator failed')
        native=rt.pinned(out/'native.json',rt.identity(out/'native.json',32<<20),32<<20)
        facts(native,dict(stage='hocap_point_retention_native',status='pass',phase='complete',producer_revision=revision,
            source_binding=original,prediction_report_identity=npin,prediction_host_identity=hpin,source_prediction_reference_rehashed_after=True,
            image_id=IMAGE,gpu_used=False,model_executed=False,quality_3D_verified=False,adoption=False))
        require(native['decision']in ('INCONCLUSIVE','REJECT','QUALIFIED_SEMANTIC_MASK_POINT_RETENTION') and len(native['clips'])==2
            and (native['private_keys_accessed']==sorted(LABEL_KEYS)or native['decision']=='INCONCLUSIVE'and native['private_keys_accessed']==[]),
            'Complete exact diagnostic decision/reference schema required')
        check(deadline)
    except subprocess.TimeoutExpired as e:
        stderr_tail=(e.stderr or b'')[-4096:].decode('utf-8','replace');failure=e
    except Exception as e:failure=e
    finally:
        signal.alarm(GRACE)
        try:
            require(rt.canonical(out)==out and (out.stat().st_dev,out.stat().st_ino,out.stat().st_uid)==owner,'Owned namespace replaced; no cleanup/publication')
            if (out/'container.cid').exists():cleanup(out,name,revision);removed=True
            require((out.stat().st_dev,out.stat().st_ino,out.stat().st_uid)==owner,'Owned output namespace changed')
            recheck(rt,proof['files']);require(authenticate(rt,code,npin,hpin)['files']==proof['files'] and source(rt,code,revision)==original,'Host original proof changed')
            if (out/'native.json').exists():
                sealed=rt.strict((out/'native.json').read_bytes());private=Path(proof['protocol']['private_evaluation']['archive']['path'])
                require(rt.identity(private,2<<30)==sealed['private_archive_identity'],'Host original private archive posthash differs')
            if private_pin is not None:
                require(rt.identity(private,2<<30)==private_pin and stat.S_IMODE(private.parent.stat().st_mode)==0o700
                    and stat.S_IMODE(private.stat().st_mode)==0o400,'Host original quarantine posthash/permissions differ')
            if failure is None:check(deadline)
        except Exception as e:failure=failure or e
        report=dict(stage='hocap_point_retention_host',status='fail'if failure else 'pass',producer_revision=revision,
            native_exit_status=native_exit,owned_cleanup_verified=removed,source_prediction_rehashed_after=failure is None,
            native_report_identity=rt.identity(out/'native.json',32<<20)if (out/'native.json').exists()else None,
            error_type=type(failure).__name__ if failure else None,error=str(failure)[:400]if failure else None,
            native_stderr_tail=stderr_tail,gpu_used=False,model_executed=False,adoption=False)
        require(rt.canonical(out)==out and (out.stat().st_dev,out.stat().st_ino,out.stat().st_uid)==owner,'Foreign namespace not published')
        if failure:rt.write(out/'report.json',(json.dumps(report,sort_keys=True)+'\n').encode(),0o444)
        else:publish(rt,out/'report.json',report,deadline)
        out.chmod(0o555);signal.alarm(0)
    require(failure is None,'CPU evaluation failed; original predictions and failed receipt preserved');return report


if __name__=='__main__':
    try:main()
    except Exception as e:print(json.dumps({'status':'fail','error_type':type(e).__name__,'error':str(e)[:400]}),flush=True);raise SystemExit(1)from None
