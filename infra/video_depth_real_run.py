"""Blind RGB-only GPU predictions, then separately mounted private sensor scoring."""
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
from video_depth_real_acquire import ROOT, CONFIG, pin


def public_sequences(base,config_pin):
    manifest=json.loads((base/'inputs/manifest.json').read_text())
    if manifest['schema']!='world_reward.video_depth_public.v1' or manifest['config_pin']!=config_pin:
        raise ValueError('Exact frozen RGB-only manifest required')
    return manifest['sequences']


def score_depth(depth,valid,truth,dt,sensor_frames):
    sensor=truth>0; support=sensor & valid
    if depth.shape!=truth.shape or valid.shape!=truth.shape:raise ValueError('Original full-T sensor grid required')
    coverage=float(support.sum()/sensor.sum())
    if not np.isfinite(depth[support]).all() or (depth[support]<=0).any():raise ValueError('Invalid supported prediction')
    # Equal frame weighting. Unknown sensor observations remain unknown.
    errors=[float(np.mean(np.abs(depth[i][support[i]]-truth[i][support[i]])/truth[i][support[i]]))
            for i in range(len(depth)) if support[i].any()]
    temporal=[]
    for i in range(1,len(depth)):
        if sensor_frames[i] is None or sensor_frames[i-1] is None or sensor_frames[i]==sensor_frames[i-1]:continue
        mask=support[i]&support[i-1]
        if mask.any():temporal.append(float(np.mean(np.abs((depth[i]-depth[i-1])[mask]-(truth[i]-truth[i-1])[mask]))/dt[i-1]))
    if not errors or not temporal:raise ValueError('No synchronized sensor support')
    perframe=[float(support[i].sum()/sensor[i].sum()) for i in range(len(depth)) if sensor[i].any()]
    return dict(absrel=float(np.mean(errors)),temporal_eulerian_error_m_s=float(np.mean(temporal)),
        temporal_definition='same sensor pixel temporal depth-change error including camera/occlusion motion, not material flow',
        coverage=coverage,minimum_frame_coverage=min(perframe),scored_frames=len(errors),temporal_pairs=len(temporal))


def infer(code,cfg,base,out,report):
    if (base/'eval_private').exists():raise ValueError('Private sensor data must not be mounted into predictor')
    import torch
    import video_depth_assets as assets
    from world_reward.video_depth import load_metric_small,infer_metric_video
    from PIL import Image
    asset_report=json.loads((ROOT/'results/video-depth-assets-v2.json').read_text())
    if asset_report['status']!='pass':raise ValueError('Pinned VDA assets required')
    asset_cfg=json.loads((code/'configs/video_depth_assets_v1.json').read_text())
    source=ROOT/assets.SOURCE_DIR;weights=ROOT/assets.WEIGHTS_DIR
    for row in asset_cfg['source_files']:
        if pin(source/row['file'])!={k:row[k] for k in ['bytes','sha256']}:raise ValueError('Pinned source differs')
    checkpoint=weights/asset_cfg['model_file']
    if pin(checkpoint)!=dict(bytes=asset_cfg['model_bytes'],sha256=asset_cfg['model_sha256']):raise ValueError('Pinned VDA checkpoint differs')
    sequences=public_sequences(base,report['config_pin']);decoded=[]
    for seq in sequences:
        frames=[]
        for row in seq['frames']:
            path=base/'inputs'/row['file']
            if pin(path)!={k:row[k] for k in ['bytes','sha256']}:raise ValueError('Original public PNG differs')
            with Image.open(path) as im:rgb=np.asarray(im).copy()
            if rgb.shape!=(480,640,3) or rgb.dtype!=np.uint8:raise ValueError('Original RGB grid differs')
            frames.append(rgb)
        decoded.append(np.stack(frames))
    network=load_metric_small(source,checkpoint)
    vdas=[]
    for seq,frames in zip(sequences,decoded):
        stamps=np.array([float(r['timestamp']) for r in seq['frames']]);fps=float(1/np.median(np.diff(stamps)))
        start=time.monotonic();z=infer_metric_video(network,frames,fps,fp32=False)
        vdas.append(z);report['timings'].append(dict(sequence=seq['name'],model='vda_metric_small',seconds=time.monotonic()-start))
    del network;gc.collect();torch.cuda.empty_cache()
    import object_synthetic_observations as native
    model_path,_,_=native.model_asset(ROOT)
    from moge.model.v2 import MoGeModel
    model=MoGeModel.from_pretrained(str(model_path)).cuda().eval()
    fov=float(np.degrees(2*np.arctan(640/(2*np.hypot(640,480)))))
    for sid,(seq,frames,vda) in enumerate(zip(sequences,decoded,vdas)):
        zz=[];valid=[];start=time.monotonic()
        for frame in frames:
            tensor=torch.from_numpy(frame).cuda().permute(2,0,1).float()/255
            with torch.inference_mode():pred=model.infer(tensor[None],fov_x=fov)
            zz.append(pred['depth'][0].cpu().numpy());valid.append(pred['mask'][0].cpu().numpy().astype(bool))
        target=out/f'sequence_{sid:02d}.npz'
        np.savez_compressed(target,vda=vda,moge=np.stack(zz),moge_valid=np.stack(valid),
            frame_index=np.arange(200,296),timestamps=np.array([float(r['timestamp']) for r in seq['frames']]))
        target.chmod(0o444);report['predictions'][target.name]=pin(target)
        report['timings'].append(dict(sequence=seq['name'],model='moge_original_fixed_RGB_fov',seconds=time.monotonic()-start))
    report.update(status='sealed_predictions',private_values_read=False,models=['metric_vda_small','moge_original'],
        model_overlap_verified=False,production_adopted=False,full_original_frame_indices_preserved=True)


def evaluate(code,cfg,base,out,report):
    from PIL import Image
    predictions=ROOT/'validation/video_depth_real_output_v1/predictions';sealed=json.loads((predictions/'report.json').read_text())
    if sealed['status']!='sealed_predictions' or sealed['config_pin']!=report['config_pin']:raise ValueError('Sealed inference before sensor decoding required')
    acquisition=json.loads((base/'acquisition-report.json').read_text())
    rows=[]
    for sid,seq in enumerate(cfg['sequences']):
        path=predictions/f'sequence_{sid:02d}.npz'
        if pin(path)!=sealed['predictions'][path.name]:raise ValueError('Sealed predictions changed')
        with np.load(path,allow_pickle=False) as a:arrays={k:a[k] for k in a.files}
        original_stamps=np.array([float(Path(r['rgb']['path']).stem) for r in seq['frames']])
        if (not np.array_equal(arrays['frame_index'],[r['rank'] for r in seq['frames']])
                or not np.array_equal(arrays['timestamps'],original_stamps)):
            raise ValueError('Sealed original RGB chronology differs')
        truth=np.zeros(arrays['moge'].shape,np.float32);names=[]
        for i,frame in enumerate(seq['frames']):
            record=frame['depth'];names.append(record['path'] if record else None)
            if record is None:continue
            file=base/'eval_private'/f'{sid:02d}_{Path(record["path"]).name}'
            if pin(file)!=acquisition['private_files'][file.name]:raise ValueError('Private sensor byte identity differs')
            with Image.open(file) as im:raw=np.asarray(im)
            if raw.shape!=(480,640) or raw.dtype.kind not in 'iu':raise ValueError('Original registered uint16 depth required')
            truth[i]=raw.astype(np.float32)/5000
        # This scores changes on the original RGB timeline after nearest-sensor
        # association, not exact sensor-time velocity or material-flow TAE.
        dt=np.diff(arrays['timestamps'])
        if (dt<=0).any():raise ValueError('Original monotonic timestamps required')
        candidate_valid=np.isfinite(arrays['vda'])&(arrays['vda']>0)
        paired=arrays['moge_valid']&candidate_valid
        scores={name:score_depth(arrays[name],paired,truth,dt,names) for name in ['moge','vda']}
        scores['vda']['native_sensor_coverage']=float((candidate_valid&(truth>0)).sum()/(truth>0).sum())
        rows.append(dict(sequence=seq['name'],metrics=scores,
            absrel_gain=1-scores['vda']['absrel']/scores['moge']['absrel'],
            temporal_gain=1-scores['vda']['temporal_eulerian_error_m_s']/scores['moge']['temporal_eulerian_error_m_s'],
            synchronized_fraction=sum(x is not None for x in names)/len(names),
            temporal_timestamps='original_RGB_deltas_with_nearest_registered_sensor_offsets_max20ms',
            unique_sensor_frames=len(set(x for x in names if x is not None))))
    limits=cfg['gates'];gates=dict(
        median_absrel_gain=float(np.median([r['absrel_gain'] for r in rows]))>=limits['median_absrel_gain_min'],
        worst_absrel_nonregression=min(r['absrel_gain'] for r in rows)>=-limits['worst_sequence_absrel_regression_max'],
        median_temporal_gain=float(np.median([r['temporal_gain'] for r in rows]))>=limits['median_temporal_error_gain_min'],
        worst_temporal_nonregression=min(r['temporal_gain'] for r in rows)>=-limits['worst_sequence_temporal_error_regression_max'],
        support=all(min(r['metrics']['vda']['native_sensor_coverage'],r['metrics']['vda']['coverage'],
            r['metrics']['vda']['minimum_frame_coverage'])>=limits['candidate_sensor_valid_coverage_min'] for r in rows),
        synchronization=all(r['synchronized_fraction']>=limits['synchronized_rgb_fraction_min'] for r in rows))
    report.update(status='completed_sensor_evaluation',rows=rows,gates=gates,passed=all(gates.values()),
        private_values_read=True,private_values_used_for_prediction=False,
        scene_generalization_verified=False,full_hoi_accuracy_verified=False,production_adopted=False,
        sealed_prediction_report_pin=pin(predictions/'report.json'))


def run():
    mode=sys.argv[1];code=Path(os.environ['WR_CODE']);rev=os.environ['WR_CODE_REVISION'];start=time.monotonic()
    if mode not in ['infer','evaluate'] or code!=ROOT/'jobs'/rev/('run_video_depth_real_'+mode)/'code':raise ValueError('Exact staged producer required')
    cfg=json.loads((code/CONFIG).read_text());base=ROOT/cfg['namespace']
    out=ROOT/'validation/video_depth_real_output_v1'/('predictions' if mode=='infer' else 'evaluation')
    out.mkdir(exist_ok=False)
    report=dict(status='fail',producer_revision=rev,config_pin=pin(code/CONFIG),private_values_read=False,
        challenge_inputs_used=False,production_adopted=False,timings=[],predictions={})
    try:
        from mediapipe_cpu_runtime_verify import source
        helpers=('infra/video_depth_real_run.py','infra/run_video_depth_real_'+mode+'.sh',CONFIG)
        binding=source(ROOT,code,rev,'run_video_depth_real_'+mode,helpers)
        (infer if mode=='infer' else evaluate)(code,cfg,base,out,report)
        if source(ROOT,code,rev,'run_video_depth_real_'+mode,helpers)!=binding:raise ValueError('Immutable staged source changed')
        report['source_binding']=binding
    except Exception as exc:report.update(error_type=type(exc).__name__,error=str(exc)[:300])
    report['elapsed_seconds']=time.monotonic()-start
    (out/'report.json').write_text(json.dumps(report,sort_keys=True,allow_nan=False)+'\n');(out/'report.json').chmod(0o444)
    print(json.dumps({k:report.get(k) for k in ['status','error','elapsed_seconds','gates','passed','rows']}))
    if report['status']=='fail':raise SystemExit(1)


if __name__=='__main__':run()
