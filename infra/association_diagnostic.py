"""Read-only, same-bank baseline association diagnostics for every packed clip.

This is video-only self-consistency, not identity truth or challenge accuracy.
It cannot detect a wrong actor/object pair whose masks and meshes agree. No
threshold is learned from these challenge records; nothing is fitted or edited.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import time

import numpy as np

import reconstruction_preview as preview

ROOT = preview.ROOT


def mask_statistics(mask):
    if mask.ndim != 2 or mask.dtype != np.bool_:
        raise ValueError('Explicit 2D boolean mask required')
    yy, xx = np.where(mask)
    return dict(pixels=len(xx), bbox_xyxy=[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)]
        if len(xx) else None, centroid_xy=[float(xx.mean()),float(yy.mean())] if len(xx) else None)


def comparison(prediction, observation):
    if prediction.shape != observation.shape:
        raise ValueError('Common viewport required')
    p, o = mask_statistics(prediction), mask_statistics(observation)
    intersection = int(np.count_nonzero(prediction & observation))
    union = p['pixels'] + o['pixels'] - intersection
    # No observation is absent evidence, never a perfect zero/zero match.
    return dict(prediction=p, observation=o, intersection_pixels=intersection,
        iou=intersection/union if o['pixels'] else None,
        observation_coverage=intersection/o['pixels'] if o['pixels'] else None,
        prediction_precision=intersection/p['pixels'] if p['pixels'] and o['pixels'] else None,
        centroid_distance_px=float(np.linalg.norm(np.asarray(p['centroid_xy'])-o['centroid_xy']))
            if p['pixels'] and o['pixels'] else None)


def main():
    if platform.system() != 'Linux' or {p.name for p in Path('/sys/class/net').iterdir()} != {'lo'}:
        raise RuntimeError('Azure-only offline CPU diagnostic required')
    code = Path(os.environ['WR_CODE']); revision = os.environ['WR_CODE_REVISION']
    out = Path(os.environ['WR_DIAGNOSTIC_OUTPUT'])
    if out.resolve() != out or not out.is_dir() or any(out.iterdir()):
        raise ValueError('Fresh exclusive output required')
    from PIL import Image
    begin = time.perf_counter(); rows = []; all_sources = {}
    pins = sorted((code/'configs').glob('cari_clip_*_shared_export_pins.json'))
    if not pins:
        raise ValueError('No frozen complete exports')
    def authenticated(path, pin=None):
        actual = preview.identity(path)
        if pin is not None and actual != pin:
            raise ValueError('Original baseline bytes differ before diagnostics')
        all_sources[str(path)] = actual
        return path
    for pin_path in pins:
        pin = json.loads(authenticated(pin_path).read_text()); spec = pin['clip_spec']
        episode, total = spec['episode_index'], spec['total_frames']
        base = ROOT/f'outputs/episode_{episode:06d}'
        export = base/'cari_shared_export_v1'; prepare = base/'cari_shared_prepare_v1'
        for name, value in pin['export_files'].items(): authenticated(export/name, value)
        input_path = code/f'configs/cari_clip_{episode:06d}_input_pins.json'
        inputs = json.loads(authenticated(input_path).read_text())
        body_path = base/'body_full/report.json'
        body = json.loads(authenticated(body_path, inputs['source_files'][str(body_path.relative_to(ROOT))]).read_text())
        expected = dict(stage='sam3d_body_full_video_initializer', status='pass', episode_index=episode,
            input_track='track_1', ground_truth_used=False, hand_labeled_test=False, oracle_modes=[])
        if any(type(body.get(k)) is not type(v) or body[k] != v for k,v in expected.items()):
            raise ValueError('Original automatic Body provenance required')
        body_array_path = base/'body_full/predictions.npz'
        authenticated(body_array_path)
        if all_sources[str(body_array_path)]['sha256'] != body['predictions_sha256']:
            raise ValueError('Body output hash differs')
        masks_path = base/'automatic_masks/report.json'
        prompts_path = base/'automatic_masks/prompts.json'
        masks = json.loads(authenticated(masks_path).read_text())
        prompts = json.loads(authenticated(prompts_path).read_text())
        if all_sources[str(masks_path)]['sha256'] != body['mask_report_sha256'] \
                or all_sources[str(prompts_path)]['sha256'] != body['prompts_sha256']:
            raise ValueError('Automatic seed/mask ancestor differs')
        prepare_pin_path = code/f'configs/cari_clip_{episode:06d}_shared_prepare_pins.json'
        prepare_pin = json.loads(authenticated(prepare_pin_path).read_text())
        authenticated(prepare/'target.npy', prepare_pin['prepare_files']['target.npy'])
        shared = np.load(prepare/'target.npy', mmap_mode='r', allow_pickle=False)
        final = np.load(export/'target.npy', mmap_mode='r', allow_pickle=False)
        selected = preview.frame_indices(total)
        with np.load(body_array_path, allow_pickle=False) as archive:
            if not np.array_equal(archive['frame_index'], np.arange(total)):
                raise ValueError('Full original Body indices required')
            raw = archive['vertices_camera_m'][selected].copy(); raw_faces = archive['faces']
            focal = archive['focal_length'][selected].copy()
        with np.load(export/'native_parameters.npz', allow_pickle=False) as archive:
            final_faces = archive['human_faces']
        with np.load(export/'trajectory.npz', allow_pickle=False) as archive:
            obj_v, obj_f, rotations, translations, K = [archive[k] for k in
                ('object_vertices','object_faces','object_rotation','object_translation','camera_K')]
        if shared.shape != final.shape or final.shape != (total,18439,3) or not np.array_equal(raw_faces,final_faces):
            raise ValueError('Whole unchanged native topology/timeline required')
        K = K.copy(); K[0] *= preview.WIDTH/spec['width']; K[1] *= preview.HEIGHT/spec['height']
        frame_rows=[]
        body_records={v['frame_index']:v for v in body['frames']}
        for pos,index in enumerate(selected):
            observed=[]; original=[]
            for entity in (0,1):
                path=base/f'automatic_masks/masks/{entity}/{index:06d}.png'
                authenticated(path)
                if entity==0 and all_sources[str(path)]['sha256']!=body_records[index]['mask_sha256']:
                    raise ValueError('Body conditioned mask changed')
                with Image.open(path) as image:
                    array=np.asarray(image)
                    if array.shape!=(spec['height'],spec['width']) or not np.isin(array,[0,255]).all():
                        raise ValueError('Original binary automatic masks required')
                    original.append(mask_statistics(array>0))
                    observed.append(np.asarray(image.resize((preview.WIDTH,preview.HEIGHT),Image.Resampling.NEAREST))>0)
            raw_K=K.copy(); raw_K[0,0]=focal[pos]*preview.WIDTH/spec['width']; raw_K[1,1]=focal[pos]*preview.HEIGHT/spec['height']
            raw_z=preview.raster_depth(raw[pos],raw_faces,raw_K,preview.WIDTH,preview.HEIGHT)
            shared_z=preview.raster_depth(shared[index],final_faces,K,preview.WIDTH,preview.HEIGHT)
            final_z=preview.raster_depth(final[index],final_faces,K,preview.WIDTH,preview.HEIGHT)
            obj=obj_v@rotations[index].T+translations[index]
            object_z=preview.raster_depth(obj,obj_f,K,preview.WIDTH,preview.HEIGHT)
            frame_rows.append(dict(frame_index=index, original_masks=original,
                raw_body_vs_mask=comparison(np.isfinite(raw_z),observed[0]),
                shared_body_vs_mask=comparison(np.isfinite(shared_z),observed[0]),
                final_body_vs_mask=comparison(np.isfinite(final_z),observed[0]),
                final_visible_body_vs_mask=comparison(np.isfinite(final_z)&(final_z<=object_z),observed[0]),
                final_visible_object_vs_mask=comparison(np.isfinite(object_z)&(object_z<final_z),observed[1])))
        rows.append(dict(episode_index=episode,total_frames=total,frame_indices=selected,frames=frame_rows,
            automatic_actor=masks.get('actor_track_id'), automatic_actor_scores=masks.get('actor_scores'),
            actor_seed_observations=masks.get('actor_seed_observations'), seed_frame=masks['seed_frame'],
            automatic_prompts=prompts['prompts'], body_mask_conditioning=body.get('decoder_mask_config')))
        del raw,shared,final
    if any(preview.identity(Path(p))!=v for p,v in all_sources.items()):
        raise ValueError('Readonly baseline changed during diagnostics')
    result=dict(schema='world_reward.association_diagnostic.v1',status='pass',producer_revision=revision,
        episodes=len(rows),sampling='first_middle_last_for_every_committed_complete_export',
        viewport_hw=[preview.HEIGHT,preview.WIDTH],mask_resize='nearest_preview_only',
        input_track='track_1',ground_truth_used=False,predictions_modified=False,model_execution=False,
        gpu_used=False,fitting=False,quality_verified=False,semantic_identity_verified=False,
        thresholds_calibrated=False,adoption=False,rows=rows,sources=all_sources,
        elapsed_seconds=time.perf_counter()-begin)
    path=out/'report.json';path.write_text(json.dumps(result,sort_keys=True)+'\n');path.chmod(0o444)
    print('ASSOCIATION_DIAGNOSTIC_PASS',len(rows),preview.identity(path))


if __name__=='__main__': main()
