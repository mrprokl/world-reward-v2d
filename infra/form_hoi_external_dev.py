"""Four FORM DEV RGB-only inputs; authenticated first-archive reuse, no GT decode.

Acquisition is allowed without an alias bank, but inference readiness is not.
No model runs here. References are copied as opaque bytes to evaluator quarantine.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
import time
import urllib.request

import form_hoi_external_acquire as acquisition

ROOT = acquisition.ROOT
DATA = Path('/srv/world-reward-data/form_hoi_external_dev_v1')
ENTRY = 'run_form_hoi_external_dev'
CONFIG = 'configs/form_hoi_external_dev_v1.json'
HELPERS = ('infra/form_hoi_external_dev.py', 'infra/run_form_hoi_external_dev.sh',
           'infra/form_hoi_external_acquire.py', 'infra/mediapipe_cpu_runtime_verify.py',
           'src/world_reward/form_hoi_protocol.py', CONFIG, acquisition.PROTOCOL)
require, check, canonical, identity, seal = (acquisition.require, acquisition.check,
    acquisition.canonical, acquisition.identity, acquisition.seal)


def acquire_alias_bank(rt, cfg):
    """Metadata-only private Azure transfer; never any RGB or credential output."""
    pin = cfg['alias_guard']['actual_bank_identity']
    if pin is None: return
    producer=cfg['alias_guard']['producer_revision']
    require(type(producer) is str and re.fullmatch('[0-9a-f]{40}',producer),'Exact RGB bank producer required')
    path = canonical(cfg['alias_guard']['path'])
    if path.exists():
        require(rt.identity(path,512<<10)==pin,'Existing frozen alias bank differs');return
    # Same fixed private metadata route as the CPU bank's authenticated publisher.
    from form_track1_alias_bank import PrivateBank
    client=PrivateBank();client.require_private()
    name=f'research-audit-{producer}/track1-rgb-alias-bank.json'
    with client.request('GET',name) as response:
        require(response.status==200 and int(response.headers.get('Content-Length','-1'))==pin['bytes']
            and response.headers.get('x-ms-meta-sha256')==pin['sha256'],'Pinned metadata bank required')
        raw=response.read((512<<10)+1)
    require(len(raw)==pin['bytes'] and hashlib.sha256(raw).hexdigest()==pin['sha256'],
            'Private alias bank transport differs')
    path.parent.mkdir(mode=0o755,parents=True,exist_ok=True)
    with path.open('xb') as stream:
        stream.write(raw);stream.flush();os.fsync(stream.fileno());os.fchmod(stream.fileno(),0o444)
    require(rt.identity(path,512<<10)==pin,'Sealed private metadata bank differs')


def runtime(code):
    rt = acquisition.runtime(code)
    require(Path(acquisition.__file__).resolve() == Path(code)/'infra/form_hoi_external_acquire.py',
            'Actual acquisition helper origin required')
    return rt


def source_binding(rt, code, revision):
    source = rt.source(ROOT, code, revision, ENTRY, HELPERS)
    require(Path(__file__).resolve() == code/'infra/form_hoi_external_dev.py', 'Actual DEV adapter source required')
    cfg = rt.strict((code/CONFIG).read_bytes()); config_pin = rt.identity(code/CONFIG, 16 << 10)
    cohort = rt.pinned(code/acquisition.PROTOCOL, acquisition.PROTOCOL_PIN, 16 << 10)
    require(cfg['schema'] == 'world_reward.form_hoi_external_dev.v1' and
            cfg['source_revision'] == cohort['dataset_revision'] and
            cfg['cohort_protocol_identity'] == acquisition.PROTOCOL_PIN and
            cfg['contiguous_prefix_frames'] == cfg['minimum_length'] == 96 and
            (cfg['fps'], cfg['width'], cfg['height']) == (30,1536,1152) and
            cfg['native_camera'] == 'front_stereo_camera_left' and cfg['reserved_acquired'] == 0 and
            cfg['training_overlap_verified'] is False, 'Frozen four-DEV first96 RGB-only contract required')
    require(cfg['native_semantic_paths']==dict(object_prompt='object.prompt',action='action_desc',
            object_id='object.id',person_id='person.id',full_frames='frame_count') and
            cfg['native_metadata']=='hoi_metadata.yaml' and
            cfg['first_acquisition']['producer_revision']=='658c156c9fcc075e9aa5b659f6cc5e8d5dfd9021' and
            cfg['alias_guard']['frames_per_video']==96 and
            cfg['alias_guard']['decoded_format']=='native_rgb24_sha256_no_resize' and
            cfg['alias_guard']['schema']=='world_reward.track1_rgb_alias_bank.v1' and
            cfg['challenge_manifest']==dict(bytes=13081,sha256='3df960ce0f594b8f51675b21bb070925de7aa87a583332674eb89b0e90fc6263'),
            'Frozen original text selectors and permitted challenge RGB guard required')
    return cfg, cohort, source, config_pin


def authenticate_first(rt, cfg, cohort):
    p = cfg['first_acquisition']; revision = p['producer_revision']
    oldcode = ROOT/'jobs'/revision/'run_form_hoi_external_acquire/code'
    original = rt.source(ROOT, oldcode, revision, 'run_form_hoi_external_acquire',
                         acquisition.HELPERS)
    prior = rt.pinned(oldcode/acquisition.PROTOCOL, acquisition.PROTOCOL_PIN, 16 << 10)
    require(prior == cohort, 'Original and current frozen cohort differ')
    base = acquisition.DATA/f'inventory_first-{revision}'
    report = rt.pinned(base/'report.json', p['report'], 64 << 10)
    first = acquisition.cohort(cohort, 'inventory_first')[0]
    receipt = rt.pinned(base/first['sequence_id']/'receipt.json', p['receipt'], 32 << 10)
    require(report['status'] == 'pass' and report['stage'] == 'inventory_first' and
            report['producer_revision'] == revision and report['source_before'] == original and
            report['source_after'] == original and report['protocol_identity'] == acquisition.PROTOCOL_PIN and
            report['reference_arrays_decoded'] is False and report['reference_masks_decoded'] is False and
            report['calibration_values_decoded'] is False and report['reserved_acquired'] == 0 and
            len(report['sequences']) == 1 and report['sequences'][0]['receipt'] == p['receipt'] and
            receipt['producer_revision'] == revision and receipt['sequence_id'] == first['sequence_id'] and
            receipt['split'] == 'development' and receipt['inference_ready'] is False and
            receipt['annotations_loaded'] is False and receipt['source_revision'] == cfg['source_revision'],
            'Original complete immutable first-archive acquisition PASS required')
    archive = canonical(base/first['sequence_id']/'source.tar')
    pin = dict(bytes=first['archive_size'], sha256=first['archive_sha256'])
    require(not archive.stat().st_mode & 0o222 and identity(archive, first['archive_size']) == pin and
            all(receipt['archive'][k] == pin[k] for k in pin), 'Original publisher-authenticated readonly archive required')
    return archive, dict(producer_revision=revision, report=p['report'], receipt=p['receipt'],
                         source_closure_sha256=original['closure_sha256'], archive=pin)


def native_semantics(raw, cfg):
    """Plain YAML lexical scalars only. Never construct tags, arrays or labels."""
    require(0 < len(raw) <= acquisition.MAX_META, 'Bounded native metadata required')
    text = raw.decode('utf-8'); targets = set(cfg['native_semantic_paths'].values())
    found = {}; nesting = []
    for line in text.splitlines():
        match = re.match(r'^( *)([A-Za-z_][A-Za-z_0-9-]*):(?:[ \t]*(.*))?$', line)
        if not match: continue
        indent, key, scalar = len(match[1]), match[2], match[3] or ''
        while nesting and nesting[-1][0] >= indent: nesting.pop()
        path = '.'.join([k for _,k in nesting]+[key])
        if not scalar or scalar.startswith('#'):
            nesting.append((indent,key)); continue
        if path not in targets: continue
        require(path not in found and not scalar.startswith(('!','&','*','[','{','|','>')),
                'Unique plain native semantic scalar required')
        if scalar.startswith('"'):
            value = json.loads(scalar)
        elif scalar.startswith("'"):
            require(scalar.endswith("'"), 'Malformed quoted native semantic scalar')
            value = scalar[1:-1].replace("''", "'")
        else: value = scalar.split(' #',1)[0].strip()
        require(type(value) in (str,int) and 0 < len(str(value)) <= 1024,
                'Finite bounded native text/ID scalar required')
        found[path] = str(value)
    require(set(found) == targets, 'Required native object/action/person/frame scalar missing')
    result = {key:found[path] for key,path in cfg['native_semantic_paths'].items()}
    require(re.fullmatch('[1-9][0-9]*', result['full_frames']) and
            int(result['full_frames']) >= cfg['minimum_length'], 'Native clip shorter than frozen96; no replacement')
    result['full_frames'] = int(result['full_frames'])
    for key in ('object_prompt','action','object_id','person_id'):
        require(result[key].strip() == result[key] and result[key] and
                all(ord(c) >= 32 and ord(c) != 127 for c in result[key]), 'Nonempty plain native semantics required')
    return result


def probe(video, cfg, full_frames, deadline):
    executable = shutil.which('ffprobe'); require(executable, 'ffprobe required; no invented fps/length')
    result = subprocess.run([executable,'-v','error','-select_streams','v:0','-show_entries',
                            'stream=width,height,nb_frames,r_frame_rate,avg_frame_rate','-of','json',str(video)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
                            timeout=min(30,max(1,deadline-time.monotonic())))
    require(len(result.stdout) <= 64 << 10, 'Bounded ffprobe metadata required')
    value = json.loads(result.stdout); streams = value.get('streams')
    require(type(streams) is list and len(streams) == 1, 'One original RGB video stream required')
    s = streams[0]
    require(s.get('width') == cfg['width'] and s.get('height') == cfg['height'] and
            s.get('nb_frames') == str(full_frames), 'Original native dimensions/frame_count differ')
    for key in ('r_frame_rate','avg_frame_rate'):
        require(type(s.get(key)) is str and re.fullmatch('[1-9][0-9]*/[1-9][0-9]*', s[key]),
                'Positive original rational frame rate required')
        n,d = map(int,s[key].split('/')); require(n == cfg['fps']*d, 'Original native fps differs')
    return dict(width=cfg['width'],height=cfg['height'],fps=cfg['fps'],full_frames=full_frames)


def decode_frame_hashes(video, cfg, deadline):
    """Original first96 native RGB24 pixels, streaming; no media returned locally."""
    executable = shutil.which('ffmpeg'); require(executable, 'ffmpeg native RGB decoder required')
    frame_bytes = cfg['width']*cfg['height']*3; hashes = []
    with subprocess.Popen([executable,'-v','error','-nostdin','-threads','2','-i',str(video),'-map','0:v:0',
                           '-frames:v',str(cfg['contiguous_prefix_frames']),'-vsync','0',
                           '-pix_fmt','rgb24','-f','rawvideo','pipe:1'], stdout=subprocess.PIPE,
                           stderr=subprocess.DEVNULL) as process:
        try:
            for _ in range(cfg['contiguous_prefix_frames']):
                check(deadline); frame = bytearray()
                while len(frame) < frame_bytes:
                    data = process.stdout.read(frame_bytes-len(frame))
                    require(bool(data), 'Original first96 RGB decoder truncated; no padding')
                    frame.extend(data); check(deadline)
                hashes.append(hashlib.sha256(frame).hexdigest())
            require(not process.stdout.read(1), 'Unexpected decoder frame count')
            require(process.wait(timeout=min(15,max(1,deadline-time.monotonic()))) == 0, 'RGB decoder failed')
        except BaseException:
            process.kill(); process.wait(); raise
    return hashes


def alias_guard(rt, cfg, *, decoded_hashes, video_pin):
    pin = cfg['alias_guard']['actual_bank_identity']; path = canonical(Path(cfg['alias_guard']['path']))
    if pin is None or not path.exists():
        return dict(qualified=False, status='PENDING_PINNED_CHALLENGE_RGB_ALIAS_BANK',
                    exact_content_duplicate_check_passed=False, near_alias_absence_verified=False)
    bank = rt.pinned(path, pin, 512 << 10)
    require(bank['schema'] == cfg['alias_guard']['schema'] and
            bank['dataset_revision'] == cfg['challenge_revision'] and
            bank['input_manifest'] == cfg['challenge_manifest'] and
            bank['frames_per_video'] == cfg['contiguous_prefix_frames'] and
            bank['decoded_format'] == cfg['alias_guard']['decoded_format'] and
            len(bank['videos']) == 30 and {r['episode'] for r in bank['videos']} == set(range(30)),
            'Exact all-thirty permitted Track1 RGB alias bank required')
    matches = []
    for r in bank['videos']:
        require(type(r['bytes']) is int and r['bytes'] > 0 and re.fullmatch('[0-9a-f]{64}',r['sha256']) and
                len(r['frame_rgb_sha256']) == 96 and all(re.fullmatch('[0-9a-f]{64}',h) for h in r['frame_rgb_sha256']),
                'Original RGB hash/decoded-first96 bank contract differs')
        shared = sorted(set(decoded_hashes).intersection(r['frame_rgb_sha256']))
        if video_pin['sha256'] == r['sha256'] or shared:
            matches.append(dict(episode=r['episode'], byte_duplicate=video_pin['sha256'] == r['sha256'],
                                exact_shared_pixel_frames=len(shared)))
    require(not matches, 'Challenge RGB exact content duplicate/alias detected; inference forbidden')
    return dict(qualified=True, status='QUALIFIED_EXACT_CONTENT_GUARD_ONLY', bank_identity=pin,
                all_challenge_videos=30, exact_content_duplicate_check_passed=True,
                near_alias_absence_verified=False, full_recording_offset_alias_absence_verified=False)


def selected_members(members, sequence, cfg):
    prefix = sequence+'/'; byname = {r['name']:r for r in members if not r['directory']}
    required = {prefix+'videos/'+cfg['native_camera']+'.mp4': 'rgb',
                prefix+cfg['native_metadata']: 'native_metadata'}
    for name in required: require(name in byname, 'Exact native single-front RGB/metadata layout absent')
    selected = dict(required)
    for name,r in byname.items():
        if PurePosixPath(name).name in cfg['reference_member_names']:
            selected[name] = 'reference'
        if PurePosixPath(name).name == cfg['reference_mask_camera']+'.h5' and any(
                p in PurePosixPath(name).parts for p in ('human_masks','object_masks')):
            selected[name] = 'reference'
    # No depth, other cameras, ground plane/symmetry or trim metadata are retained.
    for name in selected:
        require(name.startswith(prefix), 'Selected external member escaped frozen sequence')
    require(any(PurePosixPath(n).name == 'poses.npy' for n in selected) and
            any(PurePosixPath(n).name == 'output_aligned.glb' for n in selected) and
            any(PurePosixPath(n).name == 'mhr_params_mv.pt' for n in selected) and
            any(PurePosixPath(n).name == 'edex' for n in selected), 'Complete evaluator human/object/camera reference bytes required')
    return [(byname[n],role) for n,role in sorted(selected.items())]


def validate_public_package(package, cfg, *, require_ready=False):
    """Exact public adapter schema: no native IDs, GT, oracle or implicit grid."""
    fields={'schema','sequence_id','dataset','dataset_revision','split','video','video_pin',
            'total','full_source_frames','camera','height','width','fps','original_frame_indices',
            'object_prompt','action','inference_ready','reference_inputs_present'}
    require(type(package) is dict and set(package)==fields,'Public package has missing/forbidden inputs')
    require(package['schema']=='world_reward.external_rgb_input.v1' and package['dataset']=='nvidia/form-hoi' and
            package['dataset_revision']==cfg['source_revision'] and package['split']=='development' and
            package['total']==96 and package['original_frame_indices']==list(range(96)) and
            all(type(i) is int for i in package['original_frame_indices']) and
            type(package['full_source_frames']) is int and package['full_source_frames']>=96 and
            (package['camera'],package['height'],package['width'],package['fps'])==
            (cfg['native_camera'],cfg['height'],cfg['width'],cfg['fps']) and
            package['reference_inputs_present'] is False and type(package['inference_ready']) is bool,
            'Original96 single-RGB public package contract differs')
    video=PurePosixPath(package['video'])
    require(video.is_absolute() and video.name=='rgb.mp4' and video.parent.name=='inputs' and
            not any(p in ('..','eval_private','gt','masks','depth','calibration') for p in video.parts),
            'Public RGB route must not expose private references')
    require(type(package['video_pin']) is dict and set(package['video_pin'])=={'bytes','sha256'} and
            type(package['video_pin']['bytes']) is int and package['video_pin']['bytes']>0 and
            re.fullmatch('[0-9a-f]{64}',package['video_pin']['sha256']), 'Public original RGB SHA/bytes required')
    for key in ('object_prompt','action'):
        require(type(package[key]) is str and package[key].strip()==package[key] and package[key],
                'Native text-only public conditioning required')
    if require_ready: require(package['inference_ready'] is True,'Public inference still blocked')


def extract_one(archive, row, destination, cfg, rt, deadline):
    members = acquisition.inventory(archive,row,deadline,qualify_rgb=False)
    selected = selected_members(members,row['sequence_id'],cfg)
    public = destination/'inputs'; private = destination/'eval_private'
    public.mkdir(mode=0o755); private.mkdir(mode=0o700); kept = []; semantics = None
    with tarfile.open(archive,'r:') as saved:
        entries = {m.name:m for m in saved}
        for r,role in selected:
            check(deadline); m = entries[r['name']]; require(m.size == r['bytes'], 'Selected source member changed')
            relative = 'inputs/rgb.mp4' if role == 'rgb' else 'eval_private/'+r['name'][len(row['sequence_id'])+1:]
            target = destination/relative; target.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
            h = hashlib.sha256(); count = 0; small = bytearray()
            require(role != 'native_metadata' or 0 < m.size <= acquisition.MAX_META, 'Native metadata byte cap')
            with saved.extractfile(m) as src, target.open('xb') as out:
                for block in iter(lambda:src.read(1 << 20),b''):
                    check(deadline); count += len(block); require(count <= m.size, 'Selected member byte cap')
                    h.update(block); out.write(block)
                    if role == 'native_metadata': small.extend(block)
                require(count == m.size and count > 0, 'Selected source truncated/empty')
                out.flush(); os.fsync(out.fileno()); os.fchmod(out.fileno(),0o444 if role == 'rgb' else 0o400)
            pin = dict(bytes=count,sha256=h.hexdigest()); require(identity(target,count) == pin,'Extracted bytes differ')
            kept.append(dict(source=r['name'],file=relative,role=role,**pin))
            if role == 'native_metadata': semantics = native_semantics(bytes(small),cfg)
    require(semantics is not None, 'Native task semantics absent')
    video = public/'rgb.mp4'; metadata = probe(video,cfg,semantics['full_frames'],deadline)
    hashes = decode_frame_hashes(video,cfg,deadline)
    video_pin = next({k:r[k] for k in ('bytes','sha256')} for r in kept if r['role'] == 'rgb')
    guard = alias_guard(rt,cfg,decoded_hashes=hashes,video_pin=video_pin)
    package = dict(schema='world_reward.external_rgb_input.v1',sequence_id=row['sequence_id'],
                   dataset='nvidia/form-hoi',dataset_revision=cfg['source_revision'],split='development',
                   video=str(video),video_pin=video_pin,total=96,full_source_frames=semantics['full_frames'],
                   camera=cfg['native_camera'],height=metadata['height'],width=metadata['width'],fps=metadata['fps'],
                   original_frame_indices=list(range(96)),object_prompt=semantics['object_prompt'],action=semantics['action'],
                   inference_ready=guard['qualified'],reference_inputs_present=False)
    validate_public_package(package,cfg)
    input_pin = seal(public/'input.json',package)
    return dict(sequence_id=row['sequence_id'],split='development',retained=kept,input=input_pin,
                native_ids={k:semantics[k] for k in ('person_id','object_id')},native_video=metadata,
                frame_rgb_sha256=hashes,alias_guard=guard,inference_ready=guard['qualified'],
                reference_arrays_decoded=False,reference_masks_decoded=False,source_calibration_decoded=False,
                manual_prompt=False,interaction_trim_used=False,object_person_heldout_claim=False)


def run(rt,code,revision,*,reservation=None,opener=None):
    started=time.monotonic();cfg,cohort,source,config_pin=source_binding(rt,code,revision)
    deadline=started+cfg['budget_seconds']; target=canonical(DATA/revision)
    require(target.parent.is_dir() and shutil.disk_usage(target.parent).free >= cfg['minimum_free_bytes'],
            'Existing Azure data namespace and available6GiB required')
    if reservation is None:
        require(not target.exists(),'Fresh four-DEV namespace required');target.mkdir(mode=0o700)
    else:
        s=target.lstat();require(reservation==dict(device=s.st_dev,inode=s.st_ino,source_sha256=source['closure_sha256']) and
            s.st_uid==os.getuid() and stat.S_IMODE(s.st_mode)==0o700 and not tuple(target.iterdir()),
            'Exact fresh owned DEV reservation required')
    report=dict(schema='world_reward.form_hoi_external_dev_preparation.v1',status='fail',producer_revision=revision,
        source_before=source,config_identity=config_pin,sequences=[],reserved_acquired=0,models_loaded=False,gpu_used=False,
        reference_arrays_decoded=False,reference_masks_decoded=False,source_calibration_decoded=False,
        inference_ready=False,object_person_heldout_verified=False,training_overlap_verified=False,
        original_contiguous_frames=list(range(96)),external_reference_kind='multiview_reconstructed_pseudo_GT')
    active=None; archive=None
    try:
        first,report['first_archive_provenance']=authenticate_first(rt,cfg,cohort)
        opener=opener or urllib.request.build_opener(urllib.request.ProxyHandler({}),acquisition.PublisherRedirect())
        for i,row in enumerate(acquisition.cohort(cohort,'acquire_dev')):
            check(deadline);active=target/row['sequence_id'];active.mkdir(mode=0o700)
            if i==0:archive=first;download=dict(reused_authenticated_first_archive=True)
            else:
                archive=active/'source.tar';download=acquisition.fetch(row,cfg['source_revision'],archive,opener,deadline)
            receipt=extract_one(archive,row,active,cfg,rt,deadline)
            receipt.update(producer_revision=revision,download=download,source_archive_sha256=row['archive_sha256'])
            check(deadline);pin=seal(active/'receipt.json',receipt)
            if i:archive.unlink()
            archive=None;report['sequences'].append(dict(sequence_id=row['sequence_id'],receipt=pin,
                inference_ready=receipt['inference_ready'],native_ids=receipt['native_ids'],
                video=receipt['native_video'],alias_guard=receipt['alias_guard']))
            print(json.dumps(dict(stage='form_DEV_preparation',completed_DEV=i+1,total_DEV=4,
                                  inference_ready=receipt['inference_ready'],reserved_acquired=0)),flush=True)
            active=None
        report['native_object_ids']=[r['native_ids']['object_id'] for r in report['sequences']]
        require(len(set(report['native_object_ids']))==4,'DEV object IDs unexpectedly aliased; no reroll')
        report['source_after']=rt.source(ROOT,code,revision,ENTRY,HELPERS)
        require(report['source_after']==source,'Immutable source changed')
        check(deadline);report['status']='pass';report['inference_ready']=all(r['inference_ready'] for r in report['sequences'])
        report['decision']='QUALIFIED_DEV_RGB_INPUTS' if report['inference_ready'] else 'ACQUIRED_SEALED_PENDING_ALIAS_BANK'
    except Exception as error:
        report['error_type']=type(error).__name__
        if isinstance(error,ValueError):report['error_context']=str(error)
    finally:
        if active is not None:
            require(active.parent==target and not active.is_symlink() and active.stat().st_uid==os.getuid(),
                    'Refuse foreign active DEV cleanup');shutil.rmtree(active)
        report['elapsed_seconds']=time.monotonic()-started
        report['first_archive_preserved']=True
        report['owned_other_raw_archives_removed']=not any(target.glob('*/source.tar*'))
        seal(target/'report.json',report)
    return report


def main():
    require(sys.platform=='linux' and os.environ.get('WR_ROOT')==str(ROOT),'Azure Linux only')
    require(len(sys.argv)==1,'No cohort/frame override allowed')
    code=canonical(os.environ['WR_CODE']);revision=os.environ['WR_CODE_REVISION'];rt=runtime(code)
    require(re.fullmatch('[0-9a-f]{40}',revision),'Pinned source revision required')
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Inclusive DEV deadline')))
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(TimeoutError('DEV terminated')))
    cfg,_,_,_=source_binding(rt,code,revision);signal.alarm(cfg['budget_seconds'])
    result=run(rt,code,revision,reservation=rt.strict(os.environ['WR_FORM_DEV_NAMESPACE_LEASE']))
    signal.alarm(0);print(json.dumps({k:result[k] for k in ('status','inference_ready','elapsed_seconds')}),flush=True)
    return 0 if result['status']=='pass' else 1


if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as error:
        print(json.dumps(dict(stage='form_DEV_preparation',status='fail',error_type=type(error).__name__)),flush=True)
        raise SystemExit(1)
