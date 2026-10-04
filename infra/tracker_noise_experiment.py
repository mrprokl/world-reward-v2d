"""Fresh RoboTAP tracker-noise null experiment; no contact, HOI or adoption."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import stat
import sys
import time

import robotap_boots_acquire as acquisition
import robotap_boots_infer as native

ROOT = acquisition.ROOT
BASE = 'validation/tracker_noise_v1'
ORIGINAL = 'validation/robotap_boots_v1'
JOB = 'run_tracker_noise_experiment'
PROTOCOL = 'configs/tracker_noise_protocol_v1.json'
PINS = 'configs/tracker_noise_stage_pins.json'  # Root writes ACTUAL producer seals after completed stages.
ACQUISITION_PINS = 'configs/robotap_boots_acquisition_pins.json'
NATIVE_PINS = 'configs/robotap_boots_inference_pins.json'
STAGES = ('public', 'infer', 'evaluate')
IMAGES = {'public': 'sha256:7ebfff18ba3b76dd919485c19115597d7531dfd3233f69461f1dce3f28a6c6d3',
          'infer': native.IMAGE, 'evaluate': 'sha256:7ebfff18ba3b76dd919485c19115597d7531dfd3233f69461f1dce3f28a6c6d3'}
SPLITS = tuple(f'eval_private/pickles/robotap/robotap_split{i}.pkl' for i in (3, 4))
EXPECTED = dict(schema='world_reward.tracker_noise_protocol.v1', namespace=BASE,
                selection=[{'split': 3, 'lex_positions': [0], 'role': 'fit'},
                           {'split': 4, 'lex_positions': [0, 1], 'role': 'test'}],
                prefix_original_query_indices=32, all_original_frames=True, refill_unavailable=False,
                native=dict(pyramid_level=1, resolution=[256, 256], query_chunk_size=32, compute_dtype='float32',
                            AMP=False, deterministic_algorithms=False, warn_only=False, TF32=False, seed=0),
                error_coordinates='original pixel XY; native original-resolution tracks minus external annotations',
                interval_rule='both interval endpoints strictly after the original query frame',
                fit='single split3 clip; one Gaussian4 adjacent-error-increment model; no jitter',
                null='shift odd-point interval indices by floor(T/3), no wrap; separate full denominators and matched-anchor contrast',
                decision='synchronous mean log-likelihood gain strictly positive with support on each of two heldout clips; matched contrast descriptive only',
                budgets_seconds={'public': 120, 'infer': 900, 'evaluate': 120},
                host_control_seconds=135, host_lifecycle_seconds={'public':900,'infer':1350,'evaluate':900},
                rights={'dataset': 'RoboTAP CC-BY-4.0', 'native_code_checkpoint': 'Apache-2.0',
                        'training_overlap_verified': False, 'challenge_overlap_verified': False},
                claims={'contact': False, 'causal_camera_noise': False, 'full_hoi': False, 'adoption': False,
                        'automatic_track1_queries': False}, initial_queries_external_oracles=True)


def require(ok, reason):
    if not ok: raise ValueError(reason)


def clip_key(split,key):
    value=Path(split).name+':'+key
    require(type(key) is str and len(value)<=256 and key.isascii() and key and
            not any(ord(c)<32 or ord(c)==127 for c in key),'Bounded original split-qualified clip identity required')
    return value


def identity(path, empty=False):
    path = acquisition.canonical(Path(path)); before = path.stat()
    require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and not before.st_mode & 0o222 and
            (empty or before.st_size > 0), 'Readonly single-link source/input required')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''): digest.update(block)
    after = path.stat()
    require(acquisition.FIELDS(before) == acquisition.FIELDS(after), 'Input changed while hashed')
    return dict(bytes=before.st_size, sha256=digest.hexdigest())


def checked(path, pin):
    native.pin(pin)
    result = identity(path); require(result == pin, 'Frozen byte identity differs'); return result


def source_binding(code, revision):
    require(re.fullmatch('[0-9a-f]{40}', revision) and code == ROOT/'jobs'/revision/JOB/'code' and
            Path(__file__).resolve() == code/'infra/tracker_noise_experiment.py', 'Actual immutable experiment source required')
    files = {}
    for p in (code, *sorted(code.rglob('*'))):
        acquisition.canonical(p)
        require(not p.stat().st_mode & 0o222, 'Complete code closure must be readonly')
        if p.is_file(): files[p.relative_to(code).as_posix()] = identity(p, empty=True)
        else: require(p.is_dir(), 'Regular code closure required')
    markers = {n: identity(code.parent/n) for n in ('revision', 'source-sha256')}
    require((code.parent/'revision').read_bytes() == (revision+'\n').encode() and
            re.fullmatch(b'[0-9a-f]{64}\n', (code.parent/'source-sha256').read_bytes()), 'Source dispatch markers differ')
    p = native.strict_json((code/PROTOCOL).read_bytes())
    require(json.dumps(p, sort_keys=True) == json.dumps(EXPECTED, sort_keys=True), 'Frozen noise experiment protocol differs')
    return dict(files=files, markers=markers, producer_revision=revision)


def acquisition_evidence(code, hash_pickles):
    pins = native.strict_json((code/ACQUISITION_PINS).read_bytes())
    require(type(pins) is dict and set(pins)=={'schema','report','retention_receipt','pickles','protocol'} and
            pins['schema']=='world-reward-robotap-boots-acquisition-pins-v1' and
            set(pins['pickles'])=={f'eval_private/pickles/robotap/robotap_split{i}.pkl' for i in range(5)},
            'Original complete acquisition seals required')
    native.pin(pins['report'],True);native.pin(pins['retention_receipt']);native.pin(pins['protocol'])
    require(pins['protocol']==dict(bytes=acquisition.PROTOCOL_BYTES,sha256=acquisition.PROTOCOL_SHA256), 'Original source protocol differs')
    for value in pins['pickles'].values():native.pin(value)
    original = ROOT/ORIGINAL
    report_pin = {k: pins['report'][k] for k in ('bytes', 'sha256')}
    checked(original/'report.json', report_pin)
    checked(original/'eval_private/retention-receipt.json', pins['retention_receipt'])
    report = native.strict_json((original/'report.json').read_bytes())
    retention = native.strict_json((original/'eval_private/retention-receipt.json').read_bytes())
    require(report['status']=='pass' and report['stage']==acquisition.STAGE and
            report['producer_revision']==pins['report']['producer_revision'] and
            report['script_sha256']==pins['report']['script_sha256'] and
            report['pickle_or_rgb_or_gt_decoded'] is False and report['original_pickles_unmodified'] is True and
            report['source_and_protocol_after_reverified'] is True and retention['no_unpickle_or_decode'] is True,
            'Original opaque acquisition must remain qualified')
    require(all(report[k] is False for k in ('inference_performed','evaluation_performed','challenge_inputs_used','gpu_used')),
            'Original acquisition must remain opaque, non-challenge and CPU-only')
    for name in pins['pickles']:
        require(report['retained_files'][name] == retention['retained_files'][name] == pins['pickles'][name],
                'Original complete five-split retention inventory differs')
    selected = {name: checked(original/name, pins['pickles'][name]) for name in SPLITS} if hash_pickles else {}
    return pins, dict(report=report_pin, retention=pins['retention_receipt'], selected_pickles=selected)


def runtime_evidence(code, runtime_path=None):
    pins = native.strict_json((code/NATIVE_PINS).read_bytes()); native.validate_pins(pins)
    protocol = acquisition.read_protocol(code/'configs/robotap_boots_protocol.json')
    assets = ROOT/ORIGINAL/'assets'
    sources = {n: checked(assets/'tapnet_source'/n, {k:r[k] for k in ('bytes','sha256')})
               for n,r in protocol['source']['files'].items()}
    checkpoint = checked(assets/protocol['checkpoint']['file'], {k:protocol['checkpoint'][k] for k in ('bytes','sha256')})
    rpin = {k:pins['runtime'][k] for k in ('bytes','sha256')}
    path = runtime_path or ROOT/pins['runtime']['report_path']; checked(path, rpin)
    report = native.strict_json(path.read_bytes()); cpu = report['cpu_import']
    require(report['stage']=='bootstapir_runtime_verify' and report['status']=='pass' and report['phase']=='complete' and
            report['producer_revision']==pins['runtime']['producer_revision'] and report['child_image_id']==IMAGES['infer'] and
            report['native_source_revision']==protocol['source']['revision'] and report['native_sources']==sources and
            report['source_rehashed_after'] is True and report['gpu_execution'] is False and
            report['checkpoint_read'] is False and report['rgb_or_labels_read'] is False and
            cpu['source_native_einshape_cpu_verified'] is True and cpu['native_tapir_modules_imported'] is True and
            cpu['cuda_initialized'] is False and cpu['models_instantiated'] is False and
            cpu['python']=='3.11' and cpu['torch']=='2.5.1+cu124' and cpu['numpy']=='1.26.3' and
            cpu['versions']=={'dm-tree':'0.1.10','einshape':'1.0','absl-py':'2.5.0','attrs':'26.1.0','wrapt':'1.17.3'} and
            report['source_helpers']['infra/run_bootstapir_runtime_verify.sh']['sha256']==pins['runtime']['script_sha256'],
            'Independent source-native CPU runtime verification required')
    return pins, dict(native_sources=sources, checkpoint=checkpoint, runtime=rpin)


def stage_pins(code, stage):
    data = native.strict_json((code/PINS).read_bytes())
    require(type(data) is dict and set(data) in ({'schema','public'},{'schema','public','infer'}) and
            data['schema']=='world_reward.tracker_noise_stage_pins.v1', 'Actual previous-stage seals required')
    row = data[stage]; require(type(row) is dict and set(row)=={'producer_revision','report','files'}, 'Exact stage producer seal required')
    require(re.fullmatch('[0-9a-f]{40}', row['producer_revision']), 'Actual stage revision required'); native.pin(row['report'])
    report_path = ROOT/BASE/stage/'report.json'; checked(report_path, row['report'])
    report = native.strict_json(report_path.read_bytes())
    require(report['stage']=='tracker_noise_'+stage+'_v1' and report['status']=='pass' and report['phase']=='complete' and
            report['producer_revision']==row['producer_revision'] and report['source_before']==report['source_after'] and
            report['source_before']['producer_revision']==row['producer_revision'] and
            report['script_sha256']==report['source_before']['files']['infra/tracker_noise_experiment.py']['sha256'] and
            report['image_id']==IMAGES[stage] and report['gpu_used'] is (stage=='infer') and
            report['all_inputs_rehashed_after'] is True and report['outputs']==row['files'] and
            all(report[k] is False for k in ('challenge_inputs_used','contact_verified','causal_camera_noise_verified','full_hoi_verified','adoption')),
            'Completed frozen prior stage required')
    return row, report


def public_records(code, pinned=True, arrays=True):
    row, report = stage_pins(code,'public') if pinned else (None,None)
    directory = ROOT/BASE/'public/inputs'
    if row: checked(directory/'manifest.json', row['files']['manifest.json'])
    manifest = native.strict_json((directory/'manifest.json').read_bytes())
    require(set(manifest)=={'schema','selection','videos','initial_queries_external_oracles','future_labels_public','serialization_decoder'} and
            manifest['schema']=='world_reward.tracker_noise_public.v1' and manifest['initial_queries_external_oracles'] is True and
            manifest['future_labels_public'] is False, 'Fresh public-only experiment manifest required')
    decoder=manifest['serialization_decoder']
    require(type(decoder) is dict and decoder.get('source_sha256')=='279aa5b1c1cf1d5b2e2025f76c8594df6312fdc65e9431636448926271eccca2' and
            decoder.get('allowed_global')=='mediapy._VideoArray' and decoder.get('mediapy_package_imported') is False, 'Qualified lossless pickle wrapper decoder required')
    rows = manifest['videos']; require(len(rows)==3 and len(manifest['selection'])==3, 'Three complete selected clips required')
    names = {'manifest.json', *(f'video_{i:03d}.npz' for i in range(3))}
    require({p.name for p in directory.iterdir()}==names and (not row or set(row['files'])==names), 'Exact public inventory required')
    for i,r in enumerate(rows):
        require(r['file']==f'video_{i:03d}.npz' and r['role']==('fit' if i==0 else 'test') and
                r['source_pickle']==SPLITS[0 if i==0 else 1] and r['lex_position']==(0 if i<2 else 1) and
                type(r['lex_position']) is int and r['all_original_frames_retained'] is True and
                all(type(r[k]) is int and r[k]>0 for k in ('frames','height','width','query_count')) and
                manifest['selection'][i]=={k:r[k] for k in ('source_pickle','video_key','lex_position','role')}, 'Frozen split/lex positions differ')
        checked(directory/r['file'], row['files'][r['file']] if row else {k:r[k] for k in ('bytes','sha256')})
        if arrays:
            import numpy as np
            with np.load(directory/r['file'], allow_pickle=False) as data: native.validate_video({k:data[k] for k in data.files},r)
    require(len({(r['source_pickle'],r['video_key']) for r in rows})==3, 'Distinct source identities required; no replacement')
    return rows, row, report


def publish(code, out, deadline):
    import numpy as np
    import robotap_boots_public as public
    pins, before = acquisition_evidence(code,True)
    selected = []
    # Select EVERY key before inspecting points/visibility for any clip.
    cache = []
    for split, positions, role in ((SPLITS[0],(0,),'fit'),(SPLITS[1],(0,1),'test')):
        acquisition.check_deadline(deadline); data=public.read_private(ROOT/ORIGINAL/split,pins['pickles'][split])
        public.first_key(data); keys=sorted(data)
        require(len(keys)>max(positions),'Selected lexical position missing; no replacement')
        for position in positions:
            clip_key(split,keys[position])
            selected.append(dict(source_pickle=split,video_key=keys[position],lex_position=position,role=role))
            cache.append(data[keys[position]])
        del data; gc.collect()
    require(len({(r['source_pickle'],r['video_key']) for r in selected})==3,'Duplicate selected identities; no fallback')
    acquisition.save_bytes(out/'selection.json',(json.dumps(selected,sort_keys=True)+'\n').encode(),0o444)
    (out/'inputs').mkdir(mode=0o755); outputs={}; rows=[]
    for i,(selection,example) in enumerate(zip(selected,cache)):
        acquisition.check_deadline(deadline); video,queries,indices,unavailable=public.initial_queries(example)
        target=out/'inputs'/f'video_{i:03d}.npz'
        with acquisition.private_writer(target) as stream: np.savez(stream,video=video,query_points=queries,point_indices=indices)
        target.chmod(0o444); record=dict(file=target.name,**identity(target),**selection,frames=int(video.shape[0]),
            height=int(video.shape[1]),width=int(video.shape[2]),point_indices=indices.tolist(),
            unavailable_original_indices=unavailable,query_count=len(queries),all_original_frames_retained=True)
        with np.load(target,allow_pickle=False) as saved:
            require(set(saved.files)=={'video','query_points','point_indices'} and all(np.array_equal(saved[k],v)
                    for k,v in (('video',video),('query_points',queries),('point_indices',indices))), 'Public saved arrays differ')
        rows.append(record); outputs[target.name]=identity(target); cache[i]=None; gc.collect()
    manifest=dict(schema='world_reward.tracker_noise_public.v1',selection=selected,videos=rows,
                  initial_queries_external_oracles=True,future_labels_public=False,serialization_decoder=public.MEDIAPY_DECODER_SOURCE)
    acquisition.save_bytes(out/'inputs/manifest.json',(json.dumps(manifest,sort_keys=True)+'\n').encode(),0o444)
    outputs['manifest.json']=identity(out/'inputs/manifest.json')
    return dict(outputs=outputs, selection=selected, videos=rows, acquisition_before=before)


def predict(code,out,deadline,report):
    import random
    import numpy as np
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    random.seed(0); np.random.seed(0)
    import torch
    require(torch.cuda.is_available() and torch.__version__=='2.5.1+cu124' and np.__version__=='1.26.3','Exact native CUDA runtime required')
    torch.manual_seed(0);torch.cuda.manual_seed_all(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(False,warn_only=False);torch.backends.cudnn.benchmark=False
    runtime_evidence(code); rows,_,_=public_records(code)
    tapir,utils=native.native_modules(ROOT/ORIGINAL/'assets/tapnet_source')
    model=native.load_model(torch,tapir,ROOT/ORIGINAL/'assets/bootstapir_checkpoint_v2.pt')
    (out/'predictions').mkdir(mode=0o755); outputs={}; videos=[]
    for r in rows:
        acquisition.check_deadline(deadline); begin=time.monotonic()
        with np.load(ROOT/BASE/'public/inputs'/r['file'],allow_pickle=False) as saved: data={k:saved[k] for k in saved.files}
        native.validate_video(data,r); arrays=native.native_prediction(torch,utils,model,data,report)
        torch.cuda.synchronize();native.validate_prediction(arrays,r,data['query_points'],data['point_indices'])
        target=out/'predictions'/r['file']
        with acquisition.private_writer(target) as stream:np.savez(stream,**arrays)
        target.chmod(0o444)
        with np.load(target,allow_pickle=False) as saved:require(set(saved.files)==set(arrays) and
            all(np.array_equal(saved[k],v) for k,v in arrays.items()),'Native saved predictions differ')
        report['native_calls_completed']+=1;outputs[r['file']]=identity(target)
        videos.append(dict(file=r['file'],frames=r['frames'],query_count=r['query_count'],output=outputs[r['file']],elapsed_seconds=time.monotonic()-begin))
        del arrays,data;gc.collect()
    require(report['native_calls_attempted']==report['native_calls_returned']==report['native_calls_completed']==3,'All three native calls required')
    return dict(outputs=outputs,videos=videos,actual_native_inference=True,private_pickles_read=False)


def evaluate(code,out,deadline):
    import numpy as np
    import robotap_boots_public as public
    import robotap_boots_evaluate as validator
    from world_reward.tracker_noise_null import TrackerErrorClip, fit_tracker_noise_null, evaluate_tracker_noise_null
    rows,_,_=public_records(code); ipins,ireport=stage_pins(code,'infer')
    require(ireport['actual_native_inference'] is True and all(ireport[k]==3 for k in
            ('native_calls_attempted','native_calls_returned','native_calls_completed')), 'Actual complete native predictions required before labels')
    require(set(ipins['files'])=={r['file'] for r in rows}, 'All three frozen prediction artifacts required')
    predictions={}
    for r in rows:
        path=ROOT/BASE/'infer/predictions'/r['file'];checked(path,ipins['files'][r['file']])
        with np.load(ROOT/BASE/'public/inputs'/r['file'],allow_pickle=False) as data:
            queries,indices=data['query_points'].copy(),data['point_indices'].copy()
            rgb_sha=public.array_digest(data['video'])
        with np.load(path,allow_pickle=False) as data: pred={k:data[k] for k in data.files}
        validator.validate_prediction(pred,queries,indices,r)
        predictions[r['file']]=(pred,queries,indices,rgb_sha)
    pins,provenance=acquisition_evidence(code,True)
    model=None;results=[];model_identity=None;metadata_identity=None
    # Split3 first; fit model is persisted/frozen BEFORE opening split4 labels.
    for split, selected in ((SPLITS[0],rows[:1]),(SPLITS[1],rows[1:])):
        if split==SPLITS[1]:
            require(model is not None and identity(out/'model.npz')==model_identity and
                    identity(out/'model-metadata.json')==metadata_identity,'Frozen fit model required before opening heldout labels')
        acquisition.check_deadline(deadline); data=public.read_private(ROOT/ORIGINAL/split,pins['pickles'][split]);public.first_key(data)
        keys=sorted(data)
        for r in selected:
            require(keys[r['lex_position']]==r['video_key'],'Frozen original lexical key changed')
            example=data[r['video_key']];video,queries,indices,unavailable=public.initial_queries(example)
            pred,frozen_queries,frozen_indices,rgb_sha=predictions[r['file']]
            require(np.array_equal(queries,frozen_queries) and np.array_equal(indices,frozen_indices) and
                    unavailable==r['unavailable_original_indices'] and public.array_digest(video)==rgb_sha,
                    'Original RGB/query association differs')
            gt=example['points'][indices].astype(np.float64)*np.array([r['width'],r['height']],np.float64)
            annotation_visible=~example['occluded'][indices]
            clip=TrackerErrorClip(clip_key(r['source_pickle'],r['video_key']),np.arange(r['frames'],dtype=np.int64),indices.copy(),
                                  queries[:,0].astype(np.int64),pred['tracks'].astype(np.float64)-gt,
                                  pred['visible'] & annotation_visible)
            if r['role']=='fit':
                require(model is None,'Only one fit clip permitted')
                model=fit_tracker_noise_null((clip,))
                target=out/'model.npz'
                with acquisition.private_writer(target) as stream:np.savez(stream,mean=model.mean,covariance=model.covariance,
                    independent_covariance=model.independent_covariance)
                target.chmod(0o444);model_identity=identity(target)
                acquisition.save_bytes(out/'model-metadata.json',(json.dumps(dict(fit_clip_keys=list(model.fit_clip_keys),
                    fit_sample_counts=model.fit_sample_counts,fit_supported_pair_counts=model.fit_supported_pair_counts))+'\n').encode(),0o444)
                metadata_identity=identity(out/'model-metadata.json')
            else:
                require(model is not None and identity(out/'model.npz')==model_identity,'Frozen fit model required before heldout labels')
                results.append(evaluate_tracker_noise_null(model,clip))
        del data;gc.collect()
    require(len(results)==2 and identity(out/'model.npz')==model_identity and
            identity(out/'model-metadata.json')==metadata_identity,'Two heldout results and unchanged fit model required')
    outputs={name:identity(out/name) for name in ('model.npz','model-metadata.json')}
    return dict(outputs=outputs,heldout_results=results,private_labels_read=True,fit_model_frozen_before_test_labels=True,
                decision=noise_decision(results),acquisition_before=provenance,
                interpretation='tracker/annotation nuisance null only; not causal camera, contact or identity evidence')


def noise_decision(results):
    require(type(results) is list and len(results)==2,'Both fixed heldout clips must be reported')
    synchronous=[r['synchronous'] for r in results]
    if any(r['supported_pairs']==0 or r['complete_samples']==0 or r['mean_log_likelihood_gain'] is None for r in synchronous):
        return 'INCONCLUSIVE'
    gains=[r['mean_log_likelihood_gain'] for r in synchronous]
    require(all(type(g) is float and math.isfinite(g) for g in gains),'Finite synchronous gains required')
    return 'QUALIFIED_COMPOSITE_PREDICTIVE_NOISE' if all(g>0 for g in gains) else 'REJECT'


def stage_evidence(code,stage,runtime_path=None):
    if stage=='public': return acquisition_evidence(code,True)[1]
    if stage=='infer':
        _,runtime=runtime_evidence(code,runtime_path);rows,row,report=public_records(code,arrays=False)
        return dict(runtime=runtime,public=row,public_manifest=identity(ROOT/BASE/'public/inputs/manifest.json'))
    pub,_=stage_pins(code,'public');pred,_=stage_pins(code,'infer')
    evidence=acquisition_evidence(code,True)[1]
    for phase,row,subdir in (('public',pub,'inputs'),('infer',pred,'predictions')):
        for name,pin in row['files'].items(): checked(ROOT/BASE/phase/subdir/name,pin)
    return dict(acquisition=evidence,public=pub,infer=pred)


def closed_cohort_evidence(code):
    pins=native.strict_json((code/NATIVE_PINS).read_bytes());native.validate_pins(pins)
    path=ROOT/native.PUBLIC/'inputs/manifest.json';pin=pins['public']['files']['manifest.json'];checked(path,pin)
    manifest=native.strict_json(path.read_bytes());rows=manifest['videos']
    require(len(rows)==3 and {r['source_pickle'] for r in rows}==
            {f'eval_private/pickles/robotap/robotap_split{i}.pkl' for i in range(3)} and
            all(r['source_pickle'] not in SPLITS for r in rows), 'Fresh split3/4 must not reuse closed split0/1/2')
    require(manifest['selection']==[{'pickle_file':r['source_pickle'],'video_key':r['video_key']} for r in rows],
            'Original closed-cohort selection association differs')
    return dict(manifest=pin,source_identities=[{k:r[k] for k in ('source_pickle','video_key')} for r in rows],
                new_source_splits_disjoint=True)


def host_control(code,revision,stage,mode):
    require(os.geteuid()==0 and os.uname().sysname=='Linux', 'Owned Linux control host required')
    before=source_binding(code,revision);evidence=stage_evidence(code,stage)
    evidence=dict(stage=evidence,closed_cohort=closed_cohort_evidence(code))
    original=ROOT/ORIGINAL
    mirror=ROOT/'results'/('tracker-noise-runtime-proof-'+revision)/'report.json'
    runtime_pins=native.strict_json((code/NATIVE_PINS).read_bytes()) if stage=='infer' else None
    if mode=='mirror':
        require(stage=='infer', 'Only public CPU runtime receipt may be mirrored')
        path=ROOT/runtime_pins['runtime']['report_path'];raw=path.read_bytes()
        checked(path,{k:runtime_pins['runtime'][k] for k in ('bytes','sha256')})
        mirror.parent.mkdir(mode=0o700);acquisition.save_bytes(mirror,raw,0o444)
        checked(mirror,{k:runtime_pins['runtime'][k] for k in ('bytes','sha256')})
        require(dict(stage=stage_evidence(code,stage),closed_cohort=closed_cohort_evidence(code))==evidence,
                'Original CPU receipt or closed cohort changed during mirror')
        print(mirror);return
    if mode=='mounts':
        paths=[code,code.parent/'revision',code.parent/'source-sha256']
        if stage in ('public','evaluate'):
            paths += [original/'report.json',original/'eval_private/retention-receipt.json',*[original/n for n in SPLITS]]
        if stage in ('infer','evaluate'):
            paths += [ROOT/BASE/'public/report.json',ROOT/BASE/'public/inputs/manifest.json',
                      *[ROOT/BASE/'public/inputs'/f'video_{i:03d}.npz' for i in range(3)]]
        if stage=='infer':
            protocol=acquisition.read_protocol(code/'configs/robotap_boots_protocol.json')
            paths += [original/'assets/tapnet_source'/n for n in protocol['source']['files']]
            paths += [original/'assets'/protocol['checkpoint']['file']]
            checked(mirror,{k:runtime_pins['runtime'][k] for k in ('bytes','sha256')})
            for path in paths:print(str(path)+'\t'+str(path))
            print(str(mirror)+'\t'+str(ROOT/runtime_pins['runtime']['report_path']));return
        if stage=='evaluate':
            paths += [ROOT/BASE/'infer/report.json',*[ROOT/BASE/'infer/predictions'/f'video_{i:03d}.npz' for i in range(3)]]
        for path in paths:print(str(path)+'\t'+str(path))
        return
    protocol=acquisition.read_protocol(code/'configs/robotap_boots_protocol.json')
    acquisition.azure_vm02_identity(protocol)
    print(hashlib.sha256(json.dumps(dict(source=before,evidence=evidence),sort_keys=True).encode()).hexdigest())


def main(argv=None):
    parser=argparse.ArgumentParser(allow_abbrev=False);parser.add_argument('--stage',choices=STAGES,required=True)
    parser.add_argument('--control',choices=('preflight','mirror','mounts'))
    args=parser.parse_args(argv);stage=args.stage
    code,revision=Path(os.environ['WR_CODE']),os.environ['WR_CODE_REVISION']
    if args.control:
        return host_control(code,revision,stage,args.control)
    require(os.uname().sysname=='Linux' and os.geteuid()==1000 and os.environ.get('WR_IMAGE_ID')==IMAGES[stage] and
            os.environ.get('WR_VM02_VERIFIED')=='1' and {p.name for p in Path('/sys/class/net').iterdir()}=={'lo'}, 'Offline verified VM02 stage required')
    out=ROOT/BASE/stage
    acquisition.canonical(out)
    require(out.is_dir() and not any(out.iterdir()) and out.stat().st_uid==1000 and stat.S_IMODE(out.stat().st_mode)==0o700,
            'Fresh sole RW stage leaf required')
    start=time.monotonic();budget=EXPECTED['budgets_seconds'][stage];deadline=start+budget
    def expired(*_): raise TimeoutError('Fixed whole-stage deadline')
    handlers={s:signal.signal(s,expired) for s in (signal.SIGALRM,signal.SIGTERM)};signal.alarm(budget)
    before=None;evidence=None;report=dict(stage='tracker_noise_'+stage+'_v1',status='fail',phase='preflight',producer_revision=revision,
        budget_seconds=budget,image_id=IMAGES[stage],gpu_used=stage=='infer',
        native_calls_attempted=0,native_calls_returned=0,native_calls_completed=0,
        challenge_inputs_used=False,contact_verified=False,causal_camera_noise_verified=False,full_hoi_verified=False,adoption=False,
        private_labels_read=False,private_pickles_read=stage!='infer',initial_queries_external_oracles=True,outputs={})
    report['closed_source_splits_not_reused']=True
    try:
        before=source_binding(code,revision);report['source_before']=before
        report['script_sha256']=before['files']['infra/tracker_noise_experiment.py']['sha256']
        evidence=stage_evidence(code,stage);report['inputs_before']=evidence
        report['phase']=stage;sys.path.insert(0,str(code/'src'))
        result=publish(code,out,deadline) if stage=='public' else predict(code,out,deadline,report) if stage=='infer' else evaluate(code,out,deadline)
        report.update(result);report.update(status='pass',phase='complete')
    except Exception as error:report['failure_type']=type(error).__name__
    finally:
        try:
            require(before is not None and source_binding(code,revision)==before,'Stage source changed')
            report['source_after']=before
            require(evidence is not None and stage_evidence(code,stage)==evidence,'Stage input/runtime changed')
            for name,pin in report['outputs'].items():
                subdir='inputs' if stage=='public' else 'predictions' if stage=='infer' else ''
                checked(out/subdir/name,pin)
            report['all_inputs_rehashed_after']=True
        except Exception as error:report.update(status='fail',posthash_failure_type=type(error).__name__)
        report['elapsed_seconds']=time.monotonic()-start
        if report['elapsed_seconds']>budget:report.update(status='fail',failure_type='InclusiveDeadline')
        signal.alarm(0)
        acquisition.save_bytes(out/'report.json',(json.dumps(report,sort_keys=True,allow_nan=False)+'\n').encode(),0o444)
        for s,handler in handlers.items():signal.signal(s,handler)
    return 0 if report['status']=='pass' else 1


if __name__=='__main__':
    try: raise SystemExit(main())
    except Exception as error:
        print('tracker_noise_experiment FAIL: '+type(error).__name__,file=sys.stderr);raise SystemExit(1)
