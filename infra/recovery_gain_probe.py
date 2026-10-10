"""Azure CPU saved-only mask recovery QA; coverage and proxies are NOT accuracy.

No model calls, GT, mask edits or per-episode selection. Literal source masks and
native absence survive; deterministic gap-boundary comparisons stay on Azure.
"""
import hashlib
import io
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw
from mediapipe_cpu_runtime_verify import identity, require, source, strict, write
from task_grounding_pilot import ROOT, inputs, views
import sam31_recovery as saved
from qwen4d_masks import _inventory

ENTRY = 'run_recovery_gain_probe'
RECOVERY = '0826d17bd1a300220a986b961fcf942d0036e2b4'
HOST_PIN = dict(bytes=2400, sha256='667227c17b2ce656e6c8c86a4bf7dc22bf7deea0823ac7e01037dee14cd44a64')
NATIVE_PIN = dict(bytes=23843, sha256='029ac05ae0e411d4dfc5b21cd32dd73ed2c58b89549011dd011561191aa2cfa7')
HELPERS = ('infra/recovery_gain_probe.py', 'infra/run_recovery_gain_probe.sh', *saved.HELPERS)


def gaps(empty):
    a = np.asarray(empty)
    require(a.ndim == 1 and a.dtype == bool, 'Literal full-T missing observations required')
    edges = np.flatnonzero(np.diff(np.r_[False, a, False].astype(int)))
    return [dict(start=int(s), end=int(e-1), length=int(e-s)) for s, e in edges.reshape(-1, 2)]


def qa_indices(empty, limit=12):
    """Largest gap middle/start/end and boundaries, then uniform remaining RGB."""
    require(type(limit) is int and 1 <= limit <= 12 and len(empty) > 0, 'Bounded global QA selection required')
    chosen = []
    proposals = []
    for row in sorted(gaps(empty), key=lambda r: (-r['length'], r['start'])):
        s, e = row['start'], row['end']; proposals.extend(((s+e)//2, s, e, s-1, e+1))
    proposals.extend(np.linspace(0, len(empty)-1, min(limit, len(empty)), dtype=int).tolist())
    for i in proposals:
        if 0 <= i < len(empty) and i not in chosen: chosen.append(i)
        if len(chosen) == min(limit, len(empty)): break
    return sorted(chosen)


def summary(values):
    a = np.asarray(values, float)
    require(a.ndim == 1 and np.isfinite(a).all(), 'Finite diagnostic observations required')
    return dict(count=len(a), minimum=None if not len(a) else float(a.min()),
                median=None if not len(a) else float(np.median(a)),
                p10=None if not len(a) else float(np.quantile(a, .1)),
                p95=None if not len(a) else float(np.quantile(a, .95)))


def mask_metrics(mask, person, baseline_area):
    from scipy.ndimage import label
    require(mask.dtype == person.dtype == bool and mask.ndim == 2 and mask.shape == person.shape
            and mask.any() and np.isfinite(baseline_area) and baseline_area > 0, 'Literal nonempty mask/grid required')
    y, x = np.nonzero(mask); area = int(mask.sum())
    labels, count = label(mask, structure=np.ones((3,3),bool)); sizes=np.bincount(labels.ravel())[1:]
    return dict(area_pixels=area, area_vs_forward_visible_median=float(area/baseline_area),
        image_area_fraction=float(area/mask.size), bbox_fill=float(area/((x.max()-x.min()+1)*(y.max()-y.min()+1))),
        components=count, largest_component_fraction=float(sizes.max()/area),
        person_mask_overlap_fraction=float((mask & person).sum()/area))


def verify_frame(forward, chosen, reverse, record, native, old_presence):
    """Exact-native continuity, not a judgment of visibility or segmentation truth."""
    choice = record['choice']; selected = choice.startswith('reverse_')
    require(choice in ('saved_forward_native','reverse_rgb_recovery','reverse_native_semantic_anchor_unverified',
                      'unresolved_native_absence'), 'Only the literal frozen native choices are allowed')
    require(type(old_presence) is bool and type(native['native_presence'][1]) is bool,
            'Literal native signs required; no confidence fabrication')
    evaluated = record['native_reverse_evaluated']; logits = record['reverse_logits']
    require(type(evaluated) is bool and (logits is not None) == evaluated, 'Actual evaluated/unknown distinction required')
    if evaluated:
        raw = logits['raw_presence_logit']
        require(type(raw) in (int, float) and np.isfinite(raw) and 0 <= logits['positive_pixels'] <= reverse.size,
                'Finite literal native reverse logits required')
        require(int(reverse.sum()) == (logits['positive_pixels'] if raw > 0 else 0), 'Native absence cannot be overridden')
    else: require(not reverse.any(), 'Unevaluated reverse is unknown, not an invented observation')
    if selected:
        require(not forward.any() and evaluated and logits['raw_presence_logit'] > 0 and reverse.any()
                and np.array_equal(chosen, reverse) and native['native_presence'][1] is True,
                'Recovery must be a positive literal reverse prediction replacing only an empty observation')
    else:
        require(np.array_equal(chosen, forward) and native['native_presence'][1] == old_presence,
                'All original non-recovered observations and native signs must survive exactly')
    return selected


def comparison(rgb, person, forward, recovered, index):
    require(rgb.shape[:2] == person.shape == forward.shape == recovered.shape, 'One common RGB/mask grid required')
    height = max(1, round(rgb.shape[0]*240/rgb.shape[1])); tile = Image.new('RGB', (720, height+38), 'white')
    draw = ImageDraw.Draw(tile)
    for column, mask in enumerate((None, forward, recovered)):
        pixels = rgb.copy()
        if mask is not None:
            pixels[person] = (pixels[person]*.6 + np.array([0,180,230])*.4).astype('uint8')
            pixels[mask] = (pixels[mask]*.4 + np.array([255,140,0])*.6).astype('uint8')
        tile.paste(Image.fromarray(pixels).resize((240,height)), (column*240,38))
        label = ('RGB', 'frozen forward', 'recovery: unverified')[column]
        draw.text((column*240+4,3), f'f{index} {label}', fill='black')
        if mask is not None and not mask.any(): draw.text((column*240+4,19), 'unknown observation', fill='black')
    return tile


def write_qa(path, tiles, maximum=180000):
    require(1 <= len(tiles) <= 12 and maximum == 180000, 'Bounded small diagnostic QA required')
    canvas = Image.new('RGB',(max(t.width for t in tiles),sum(t.height for t in tiles)), 'white'); y=0
    for tile in tiles: canvas.paste(tile,(0,y)); y+=tile.height
    for shrink in (1., .8, .6):
        image = canvas if shrink == 1 else canvas.resize((round(canvas.width*shrink),round(canvas.height*shrink)))
        for quality in (80,65,50,35):
            stream=io.BytesIO(); image.save(stream,format='JPEG',quality=quality,optimize=True)
            if stream.tell() <= maximum:
                write(path,stream.getvalue(),mode=0o444)
                return dict(file=path.name, **identity(path,maximum), panel_width=image.width,
                            scope='automatic_predictions_not_labels_or_accuracy')
    raise ValueError('Bounded QA encoding exceeded')


def recovery_source(code):
    base = ROOT/'results'/('sam31-recovery-'+RECOVERY)
    require(identity(base/'report.json',200000)==HOST_PIN and identity(base/'native-report.json',200000)==NATIVE_PIN,
            'Independently pinned completed recovery receipts required')
    host=strict((base/'report.json').read_bytes()); native=strict((base/'native-report.json').read_bytes())
    binding=source(ROOT,ROOT/'jobs'/RECOVERY/saved.ENTRY/'code',RECOVERY,saved.ENTRY,tuple(host['source_binding']['helpers']))
    require(host['producer_revision']==native['producer_revision']==RECOVERY and host['source_binding']==binding
            and host['native_report']==NATIVE_PIN and native['status']=='complete_diagnostic_not_quality_pass'
            and native['ground_truth_used'] is False and native['manual_labels'] is False
            and native['baseline_modified'] is False and native['original_frame_grid_preserved'] is True
            and [r['episode_index'] for r in native['episodes']]==[9,1,14,7], 'Same original source-bound native cohort required')
    return base,native,binding


def run():
    started=time.monotonic(); revision=os.environ['WR_CODE_REVISION']; code=Path(os.environ['WR_CODE'])
    require(ROOT==Path(os.environ['WR_ROOT']) and code==ROOT/'jobs'/revision/ENTRY/'code', 'Exact Azure-only source required')
    out=ROOT/'results'/('recovery-gain-'+revision)
    require(out.is_dir() and {p.name for p in out.iterdir()} <= {'.container.cid'},
            'Only one fresh owned diagnostic output is allowed')
    report=dict(status='fail',producer_revision=revision,ground_truth_used=False,manual_labels=False,
        model_calls=0,GPU_requested=False,baseline_modified=False,quality_verified=False,
        scope='saved_prediction_coverage_and_consistency_not_accuracy_or_complete_4D',episodes=[])
    bound={}; binding=source(ROOT,code,revision,ENTRY,HELPERS)
    try:
        cfg,_=saved.settings(code); selected=inputs(cfg['episodes'])
        original,old_bound,forward_binding=saved.saved_forward(code,cfg,selected); bound.update(old_bound)
        base,native,recovery_binding=recovery_source(code)
        bound.update({base/'report.json':HOST_PIN,base/'native-report.json':NATIVE_PIN})
        import cv2
        cv2.setNumThreads(2)
        for item,ar in zip(selected,native['episodes']):
            ep=item['episode']; old=original[ep]; dest=base/f'episode_{ep:06d}/automatic_masks'; n=item['total']
            require(ar['frames']==n and ar['automatic_masks']==str(dest.relative_to(base)), 'Literal full-T source namespace required')
            pin=identity(dest/'report.json'); require(pin==ar['report'], 'Pinned source clip report differs')
            bound[dest/'report.json']=pin; metadata=strict((dest/'report.json').read_bytes())
            inventory,inventory_summary,raw=_inventory(dest/'masks',n)
            require(inventory_summary==metadata['mask_inventory'] and raw==(dest/'mask-inventory.json').read_bytes(), 'Recovered mask inventory differs')
            for name in ('tracking.json','mask-inventory.json','recovery.json','reverse-mask-inventory.json'):
                bound[dest/name]=identity(dest/name,16<<20)
            require(bound[dest/'recovery.json']==metadata['recovery'], 'Pinned source recovery choices differ')
            tracking=strict((dest/'tracking.json').read_bytes()); recovery=strict((dest/'recovery.json').read_bytes())
            reverse_pins=strict((dest/'reverse-mask-inventory.json').read_bytes())
            require(tracking['original_frame_indices']==recovery['original_frame_indices']==list(range(n))
                    and len(tracking['records'])==len(recovery['records'])==n and recovery['ground_truth_used'] is False
                    and recovery['manual_labels'] is False and recovery['mask_interpolation'] is False
                    and recovery['quality_verified'] is False and set(reverse_pins)=={f'{i:06d}.png' for i in range(n)}, 'Automatic full-T unverified predictions required')
            with Image.open(old['directory']/'masks/1/000000.png') as image: w,h=image.size
            old_areas=np.asarray(old['tracking']['areas']['1']); new_areas=[]; metrics=[]; agreements=[]; recovered=[]
            native_counts=dict(evaluated=0,absent=0,positive_but_empty=0,not_evaluated_unknown=0)
            baseline=float(np.median(old_areas[old_areas>0])); indices=qa_indices(old_areas==0) if (old_areas==0).any() else []
            rgb=views(item['video'],n,indices) if indices else []; samples=dict(zip(indices,rgb)); tiles=[]
            for i in range(n):
                masks=[]
                for role in (0,1):
                    key=f'{role}/{i:06d}.png'; path=dest/'masks'/key; bound[path]=inventory[key]
                    m=saved.read_mask(path,inventory[key],h,w)
                    require(int(m.sum())==tracking['areas'][str(role)][i], 'Recorded chosen native areas disagree')
                    if role==0 or old_areas[i]>0:
                        require(inventory[key]==old['inventory'][key], 'Original person/visible object bytes changed')
                    masks.append(m)
                forward=saved.read_mask(old['directory']/f'masks/1/{i:06d}.png',old['inventory'][f'1/{i:06d}.png'],h,w)
                path=dest/f'reverse_masks/1/{i:06d}.png'; bound[path]=reverse_pins[path.name]
                reverse=saved.read_mask(path,reverse_pins[path.name],h,w); record=recovery['records'][i]; timeline=tracking['records'][i]
                require(record['frame_index']==timeline['frame_index']==i
                        and timeline['decoded_rgb_sha256']==old['tracking']['records'][i]['decoded_rgb_sha256'], 'Original RGB/frame indices differ')
                require(timeline['native_presence'][0]==old['tracking']['records'][i]['native_presence'][0], 'Original person sign changed')
                chosen=verify_frame(forward,masks[1],reverse,record,timeline,old['tracking']['records'][i]['native_presence'][1])
                if record['native_reverse_evaluated']:
                    native_counts['evaluated']+=1
                    if record['reverse_logits']['raw_presence_logit']<=0: native_counts['absent']+=1
                    elif not reverse.any(): native_counts['positive_but_empty']+=1
                else: native_counts['not_evaluated_unknown']+=1
                new_areas.append(int(masks[1].sum()))
                if chosen: recovered.append(i); metrics.append(mask_metrics(masks[1],masks[0],baseline))
                if forward.any() and reverse.any(): agreements.append(float((forward&reverse).sum()/(forward|reverse).sum()))
                if i in samples:
                    pixels=np.asarray(samples[i]); require(hashlib.sha256(pixels.tobytes()).hexdigest()==timeline['decoded_rgb_sha256'], 'QA seek RGB differs')
                    tiles.append(comparison(pixels,masks[0],forward,masks[1],i))
            require(len(recovered)==ar['recovered_frames'] and recovered==recovery['recovered_frames'], 'Recovery count/choice receipts disagree')
            empty=np.asarray(new_areas)==0
            report['episodes'].append(dict(episode_index=ep,frames=n,originally_missing=int((old_areas==0).sum()),
                recovered=len(recovered),residual_missing=int(empty.sum()),gaps_before=gaps(old_areas==0),gaps_after=gaps(empty),
                original_person_and_visible_object_byte_exact=True,native_absence_preserved=True,
                native_reverse_observation_counts=native_counts,
                forward_reverse_iou_both_nonempty=summary(agreements),recovered_mask_proxies={k:summary([m[k] for m in metrics]) for k in metrics[0]} if metrics else {},
                person_overlap_is_not_hand_or_contact_truth=True,semantic_anchor_predictions_unverified=True,
                qa_indices=indices,qa=None if not tiles else write_qa(out/f'episode_{ep:06d}-qa.jpg',tiles),
                recovered_mask_inventory=inventory_summary,source_clip_report=pin))
        totals={k:sum(r[k] for r in report['episodes']) for k in ('originally_missing','recovered','residual_missing')}
        require(totals==dict(originally_missing=372,recovered=329,residual_missing=43), 'Pinned historical coverage receipt mismatch')
        require(inputs(cfg['episodes'])==selected and all(identity(p,max(pin['bytes'],1))==pin for p,pin in bound.items()), 'Readonly source inputs changed')
        require(source(ROOT,code,revision,ENTRY,HELPERS)==binding and recovery_source(code)[2]==recovery_binding, 'Immutable source changed')
        require(source(ROOT,ROOT/'jobs'/cfg['saved_forward_producer']/saved.FORWARD_ENTRY/'code',
                       cfg['saved_forward_producer'],saved.FORWARD_ENTRY,tuple(forward_binding['helpers']))==forward_binding,
                'Frozen original source changed')
        report.update(status='complete_diagnostic_not_quality_pass',totals=totals,source_binding=binding,
            saved_forward_source_binding=forward_binding,recovery_source_binding=recovery_binding,
            source_host_report=HOST_PIN,source_native_report=NATIVE_PIN,inputs_rehashed_after=True)
    except Exception as exc:
        report.update(status='fail',error_type=type(exc).__name__,error=str(exc)[:400])
    finally:
        report['elapsed_seconds']=time.monotonic()-started
        write(out/'report.json',(json.dumps(report,sort_keys=True,allow_nan=False)+'\n').encode(),mode=0o444)
    print(json.dumps({k:report.get(k) for k in ('status','totals','error_type','error','elapsed_seconds')}),flush=True)
    if report['status']=='fail': raise SystemExit(1)


if __name__=='__main__': run()
