"""Fresh16 original GDI+OWL banks, no labels, selection or model changes.

Old closed caller and its model/qualification helpers remain byte-identical.
The new transport preserves acquisition holes and exposes public RGB only.
"""
import argparse
import hashlib
import math
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import time

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1]/'src')]
import rgb_endpoint_bank as original
from world_reward import rgb_bank_inputs as rgb_inputs

rt, owl = original.rt, original.owl
ROOT = rt.ROOT
ENTRY = 'run_vcoco_pilot_endpoint_bank'
CONFIG = 'configs/vcoco_pilot_endpoint_bank_v1.json'
DATA = Path('/srv/world-reward-data/vcoco_role_pilot_v1/inputs')
OUTPUT = ROOT/'results/vcoco-pilot-endpoint-banks-v1'
IMAGE = original.IMAGE
NATIVE_FILES = tuple(dict.fromkeys(('infra/vcoco_pilot_endpoint_bank.py', CONFIG,
    'src/world_reward/rgb_bank_inputs.py', *original.NATIVE_FILES)))
HELPERS = tuple(dict.fromkeys((*NATIVE_FILES, 'infra/run_vcoco_pilot_endpoint_bank.sh', *original.HELPERS)))
encode, write, error = original.encode, original.write, original.error
MAX_OUTPUT = 16*(16 << 20)
BANK_KEYS = frozenset(('person_raw_boxes','person_raw_scores','person_raw_labels','person_retained_boxes',
    'person_retained_scores','person_retained_raw_slots','person_retained_ids','person_model_pred_boxes','person_model_logits',
    'person_model_input_ids','person_model_attention_mask','owl_patch_ids','owl_boxes_padded_normalized_cxcywh',
    'owl_objectness_logits','owl_boxes_original_xyxy','image_size','original_frame_index'))


def source_state(code):
    rows={str(p.relative_to(code)):[s.st_dev,s.st_ino,s.st_mode,s.st_size,s.st_nlink,s.st_uid,s.st_gid,s.st_mtime_ns,s.st_ctime_ns]
        for p in(code,*sorted(code.rglob('*')))for s in(p.lstat(),)}
    return hashlib.sha256(encode(rows)).hexdigest()


def check(deadline):
    if not math.isfinite(deadline) or time.monotonic() >= deadline:
        raise TimeoutError('Inclusive fresh bank deadline')


def configuration(code, source):
    p = rt.pinned(code/CONFIG, source['helpers'][CONFIG], 16 << 10)
    fixed = dict(schema='world_reward.vcoco_pilot_endpoint_bank.v1', image_id=IMAGE,
        output=str(OUTPUT), input_directory=str(DATA), maximum_images=16,
        maximum_input_bytes=MAX_OUTPUT, maximum_image_bytes=16 << 20,
        maximum_decoded_pixels=16 << 20, maximum_bank_bytes=16 << 20,
        maximum_output_bytes=MAX_OUTPUT, budget_seconds=600, native_cleanup_seconds=20,
        outer_seconds=615, cpu_threads=4, maximum_host_memory_bytes=8 << 30, maximum_container_memory_bytes=8 << 30,
        model_loads=2, person_query='person.', confidence=.3, text_threshold=.25,
        nms_iou=.7, owl_patches=3600, ground_truth_used=False, ownership_verified=False,
        quality_verified=False, adoption=False)
    rt.require(set(p) == set(fixed) | {'original_recipe', 'reused_helper_pins'}
        and all(type(p.get(k)) is type(v) and p[k] == v for k, v in fixed.items()), 'Frozen fresh16 recipe required')
    rt.require(set(p['reused_helper_pins']) == set(original.HELPERS)|{'src/world_reward/rgb_bank_inputs.py'}
        and all(source['helpers'][n] == wanted for n, wanted in p['reused_helper_pins'].items()),
        'Complete original helper byte pins required')
    rt.require(Path(original.__file__).resolve() == code/'infra/rgb_endpoint_bank.py'
        and Path(rgb_inputs.__file__).resolve() == code/'src/world_reward/rgb_bank_inputs.py',
        'Imported original/common helper origin differs')
    for module,name in ((rt,'infra/mediapipe_cpu_runtime_verify.py'),(owl,'infra/owlv2_native_qualify.py'),
                        (original.gdi,'infra/openimages_joint_pair_gdi.py')):
        rt.require(Path(module.__file__).resolve()==code/name,'Actual original qualification helper import required')
    rt.require(p['original_recipe'] == source['helpers'][original.CONFIG], 'Original closed recipe pin differs')
    rt.pinned(code/original.CONFIG, p['original_recipe'], 16 << 10)
    old = original.configuration(code, source)
    rt.require(all(old[k] == p[k] for k in ('image_id', 'model_loads', 'person_query',
        'confidence', 'text_threshold', 'nms_iou', 'owl_patches')), 'Original model policy unchanged')
    return p, old


def public_inputs(pin, count, *, native_mounts=False):
    # Individual readonly mounts have synthetic parent-directory mode0755 in
    # the read-only container; host directory is sealed500, every leaf remains RO.
    return rgb_inputs.read_inputs(DATA, pin, count, identity=rt.identity, pinned=rt.pinned,
                                  readonly_directory=not native_mounts, maximum_slots=16)


def observe(inputs, detect, model, operations, deadline, *, decode=None, saver=None,
            records=None, completed=None):
    """One full original inference per acquired image; original holes stay holes."""
    decode = decode or (lambda row: rgb_inputs.decode_rgb(DATA, row, identity=rt.identity))
    saver = saver or original.save_bank
    rows = [] if records is None else records
    for ordinal, row in enumerate(inputs['images']):
        check(deadline); slot = rgb_inputs.original_slot(row,16)
        arrays, metadata = original.bank_arrays(decode(row), row, ordinal, detect, model, operations)
        rt.require(set(arrays)==BANK_KEYS,'Every17 original native array required')
        operations.tensor_ops.cuda.synchronize()
        metadata.update(original_slot=slot, acquired_ordinal=ordinal)
        metadata.update(saver(OUTPUT, slot, arrays)); rows.append(metadata)
        if completed is not None: completed()
    return rows


def native(code, revision, p, pin, deadline):
    started = time.monotonic(); proof = rt.pinned(OUTPUT/'proof.json', pin, 2 << 20)
    report = dict(schema=p['schema'], stage='native_vcoco_pilot_endpoint_banks', status='fail',
        phase='authentication', producer_revision=revision, image_id=IMAGE, proof_identity=pin,
        model_loads=0, person_forward_calls=0, image_embed_calls=0, objectness_calls=0,
        box_calls=0, images=[], person_query='person.', confidence=.3, text_threshold=.25,
        nms_iou=.7, all_patches_retained=True, ground_truth_used=False,
        reference_metadata_read=False, split_metadata_read=False, challenge_inputs_used=False,
        actor_selection_performed=False, ownership_verified=False, quality_verified=False,
        adoption=False, dwpose_calls=0, sam_calls=0, hoi_calls=0, tracking_calls=0, network='none')
    validated=False;runtime=None
    try:
        rt.require(os.environ['WR_IMAGE_ID'] == IMAGE and {x.name for x in Path('/sys/class/net').iterdir()} == {'lo'}, 'Exact offline native image required')
        rt.require(set(proof) == {'source', 'native_files', 'assets', 'owl_runtime', 'inputs_identity', 'images', 'image_id'}
            and proof['source']['producer_revision'] == revision and proof['image_id'] == IMAGE
            and {str(x.relative_to(code)) for x in code.rglob('*') if x.is_file()} == set(NATIVE_FILES)
            and set(proof['native_files']) == set(NATIVE_FILES), 'Individual native source whitelist required')
        for n, wanted in proof['native_files'].items(): rt.require(rt.identity(code/n, 2 << 20) == wanted, 'Native mounted source differs')
        for n, wanted in proof['source']['markers'].items(): rt.require(rt.identity(code.parent/n, 100) == wanted, 'Dispatch marker differs')
        configuration(code, proof['source'])
        inputs = public_inputs(proof['inputs_identity'], proof['images'], native_mounts=True)
        policy = rt.pinned(code/owl.CONFIG, proof['native_files'][owl.CONFIG], 16 << 10)
        expected, card = original.expected_asset_files(policy)
        rt.require(set(proof['assets']) == set(expected) | {card}
            and all(proof['assets'][n] == wanted for n, wanted in expected.items()), 'Original model/config/card leaves required')
        for n, wanted in proof['assets'].items(): rt.require(rt.identity(Path(n), 1 << 30) == wanted, 'Original asset differs')
        validated=True
        runtime = owl.installed(policy)
        rt.require(runtime == proof['owl_runtime'], 'Qualified native runtime differs')
        grounding = original.installed_grounding(runtime['wheel_RECORD_identities']['transformers'])
        report.update(runtime_identity=runtime, grounding_source_identity=grounding, phase='models'); check(deadline)
        detect, model, operations, keepalive = original.load_models(policy); report['model_loads'] = 2
        report['phase'] = 'native_forward'
        def completed():
            for k in ('person_forward_calls', 'image_embed_calls', 'objectness_calls', 'box_calls'): report[k] += 1
        observe(inputs, detect, model, operations, deadline, records=report['images'], completed=completed)
        del model, keepalive, detect
        operations.tensor_ops.cuda.empty_cache(); operations.tensor_ops.cuda.synchronize(); del operations
        report.update(status='pass',phase='complete')
    except BaseException as exc: report.update(status='fail', error_type=error(exc))
    finally:
        try:
            rt.require(validated,'No authenticated native evidence to rehash')
            rt.require(public_inputs(proof['inputs_identity'],proof['images'],native_mounts=True)==inputs,'Native RGB projection changed')
            for n,wanted in proof['native_files'].items():rt.require(rt.identity(code/n,2 << 20)==wanted,'Native source changed')
            for n,wanted in proof['source']['markers'].items():rt.require(rt.identity(code.parent/n,100)==wanted,'Dispatch marker changed')
            for n,wanted in proof['assets'].items():rt.require(rt.identity(Path(n),1 << 30)==wanted,'Native asset changed')
            if runtime is not None:
                rt.require(owl.installed(policy)==runtime and original.installed_grounding(runtime['wheel_RECORD_identities']['transformers'])==grounding,
                    'Native installed source/runtime changed')
            for row in report['images']:rt.require(rt.identity(OUTPUT/row['file'],16 << 20)==row['identity'],'Saved full bank changed')
            report['source_inputs_runtime_assets_rehashed_after']=True;check(deadline)
        except BaseException as exc:report.update(status='fail',post_error_type=error(exc))
    report['elapsed_seconds']=time.monotonic()-started
    with (OUTPUT/'native.json').open('xb')as stream:
        os.fchmod(stream.fileno(),0o400)
        def update():
            stream.seek(0);stream.write(encode(report));stream.truncate();stream.flush();os.fsync(stream.fileno())
        try:
            update();rt.require(rt.identity(OUTPUT/'native.json',2 << 20)==
                dict(bytes=len(encode(report)),sha256=hashlib.sha256(encode(report)).hexdigest()),'Native report bytes differ');check(deadline)
        except BaseException as exc:report.update(status='fail',publication_error_type=error(exc));update()
    return report


def command(args, deadline):
    check(deadline)
    result = subprocess.run(args, env=dict(PATH='/usr/bin:/bin', HOME='/nonexistent', DOCKER_HOST='unix://'+str(ROOT/'docker.sock')),
        capture_output=True, timeout=min(10, max(.001, deadline-time.monotonic())), check=False)
    rt.require(result.returncode == 0 and len(result.stdout) <= 1 << 20, 'Bounded lifecycle command failed')
    return result.stdout.decode().strip()


def cleanup(path, name, revision, deadline):
    rt.identity(path, 65, readonly=False); raw = path.read_bytes()
    rt.require(re.fullmatch(b'[0-9a-f]{64}\n?', raw), 'Owned original CID required'); cid = raw.decode().strip()
    present = command(['docker', 'ps', '-aq', '--no-trunc', '--filter', 'id='+cid], deadline)
    rt.require(present in ('', cid), 'Ambiguous owned CID')
    if present:
        actual = command(['docker', 'inspect', cid, '--format', '{{.Image}}|{{.Name}}|{{index .Config.Labels "world-reward.job"}}|{{index .Config.Labels "world-reward.revision"}}'], deadline)
        rt.require(actual == IMAGE+'|/'+name+'|'+ENTRY+'|'+revision, 'Foreign container cannot be removed')
        command(['docker', 'rm', '-f', cid], deadline)
    rt.require(not command(['docker','ps','-aq','--no-trunc','--filter','id='+cid],deadline),'Owned CID survives cleanup')
    rt.require(not command(['docker', 'ps', '-aq', '--no-trunc', '--filter', 'name=^/'+name+'$'], deadline), 'Owned name survives')
    path.chmod(0o400)


def validate_report(report, p, revision, proof_pin, inputs):
    rt.require(report['schema'] == p['schema'] and report['stage'] == 'native_vcoco_pilot_endpoint_banks'
        and report['status'] == 'pass' and report['phase'] == 'complete' and report['producer_revision'] == revision
        and report['image_id'] == IMAGE and report['proof_identity'] == proof_pin
        and type(report['model_loads']) is int and report['model_loads'] == 2
        and report['source_inputs_runtime_assets_rehashed_after'] is report['all_patches_retained'] is True
        and report['person_query'] == 'person.' and (report['confidence'], report['text_threshold'], report['nms_iou']) == (.3, .25, .7)
        and all(report[k] is False for k in ('ground_truth_used', 'reference_metadata_read', 'split_metadata_read',
            'challenge_inputs_used', 'actor_selection_performed', 'ownership_verified', 'quality_verified', 'adoption'))
        and all(type(report[k]) is int and report[k] == 0 for k in ('dwpose_calls', 'sam_calls', 'hoi_calls', 'tracking_calls'))
        and all(type(report[k]) is int and report[k] == len(inputs['images']) for k in ('person_forward_calls', 'image_embed_calls', 'objectness_calls', 'box_calls'))
        and len(report['images']) == len(inputs['images']), 'Complete original native bank census required')
    for ordinal, (row, image) in enumerate(zip(report['images'], inputs['images'])):
        slot = rgb_inputs.original_slot(image,16)
        rt.require(row['image_id'] == image['image_id'] and row['original_frame_index'] == 0
            and row['bank_index'] == row['acquired_ordinal'] == ordinal and row['original_slot'] == slot
            and row['image_size'] == [image['height'], image['width']] and row['input_file'] == image['file']
            and row['input_identity'] == {k: image[k] for k in ('bytes', 'sha256')}
            and row['file'] == f'image_{slot:06d}.npz' and row['owl_patches'] == 3600 and row['person_native_queries'] == 900
            and type(row['person_retained_rows']) is int and 0 <= row['person_retained_rows'] <= row['person_postprocessor_rows'] <= 900
            and len(row['person_ids']) == len(set(row['person_ids'])) == row['person_retained_rows']
            and rt.identity(OUTPUT/row['file'], 16 << 20) == row['identity'], 'Original slot/grid/fullbank identity required')
        arrays=row['arrays'];rt.require(set(arrays)==BANK_KEYS and all(set(v)=={'shape','dtype','sha256'}
            and type(v['shape'])is list and all(type(n)is int and n>=0 for n in v['shape'])
            and type(v['dtype'])is str and re.fullmatch('[0-9a-f]{64}',v['sha256'])for v in arrays.values()),
            'Every saved native array identity required')
        shapes={'person_model_pred_boxes':[1,900,4],'person_model_logits':[1,900,256],'owl_patch_ids':[3600],
            'owl_boxes_padded_normalized_cxcywh':[3600,4],'owl_objectness_logits':[3600],
            'owl_boxes_original_xyxy':[3600,4],'image_size':[2],'original_frame_index':[]}
        rt.require(all(arrays[k]['shape']==v for k,v in shapes.items()),'All900/3600 native array shapes required')


def dispatch(code, revision, input_pin, count):
    import fcntl
    started = time.monotonic(); deadline = started+600
    source = rt.source(ROOT, code, revision, ENTRY, HELPERS); source_stats=source_state(code);p, recipe = configuration(code, source)
    inputs = public_inputs(input_pin, count); prior = original.qualifications(code, recipe, live=True)
    policy = rt.pinned(code/owl.CONFIG, source['helpers'][owl.CONFIG], 16 << 10)
    assets = original.asset_files(prior, policy)
    rt.require(not OUTPUT.exists() and OUTPUT.parent.is_dir(), 'Fresh fixed pilot bank output required')
    lock = rt.canonical(ROOT/'jobs/.world-reward-h100.lock'); info = lock.lstat()
    rt.require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, 'Single-link cooperative H100 lock required')
    fd = os.open(lock, os.O_RDONLY | os.O_NOFOLLOW); name = 'world-reward-vcoco-pilot-endpoint-'+revision[:12]; created = False
    host = dict(schema=p['schema'], stage='vcoco_pilot_endpoint_bank_host', status='fail', producer_revision=revision,
        source_binding=source, original_qualification=prior, original_recipe_identity=p['original_recipe'],
        public_inputs_identity=input_pin, acquired_images=count, frozen_population_slots=16, image_id=IMAGE,
        ground_truth_used=False, quality_verified=False, ownership_verified=False, adoption=False,
        training_overlap_verified=False,challenge_overlap_verified=False,license_eligibility_verified=False,
        reference_metadata_read=False,split_metadata_read=False,FIT_performed=False,
        owned_cleanup_verified=False, outputs_sealed=False)
    try:
        rt.require((os.fstat(fd).st_dev, os.fstat(fd).st_ino) == (info.st_dev, info.st_ino), 'Lock inode differs')
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rt.require(not command(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], deadline)
            and not command(['docker', 'ps', '-aq', '--no-trunc', '--filter', 'name=^/'+name+'$'], deadline), 'Other GPU work/occupied owned name')
        rt.require(command(['docker','image','inspect',IMAGE,'--format','{{.Id}}'],deadline)==IMAGE,'Exact qualified live Docker image required')
        OUTPUT.mkdir(mode=0o700); created = True; owner = OUTPUT.lstat()
        proof_pin = write(OUTPUT/'proof.json', dict(source=source, native_files={n: source['helpers'][n] for n in NATIVE_FILES},
            assets=assets, owl_runtime=prior['owl']['native_runtime'], inputs_identity=input_pin, images=count, image_id=IMAGE))
        mounts = [code/n for n in NATIVE_FILES]+[code.parent/n for n in ('revision', 'source-sha256')]
        mounts += [Path(n) for n in assets]+[DATA/'manifest.json']+[DATA/r['file'] for r in inputs['images']]
        cmd = ['docker', 'run', '--rm', '--name', name, '--cidfile', str(OUTPUT/'.container.cid'),
            '--label', 'world-reward.job='+ENTRY, '--label', 'world-reward.revision='+revision,
            '--gpus', 'device=0', '--network', 'none', '--user', '0:0', '--memory', '8g', '--cpus', '4',
            '--pids-limit', '256', '--read-only', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
            '--tmpfs', '/tmp:rw,noexec,nosuid,size=512m', '--entrypoint', '/usr/bin/env']
        for path in mounts:
            rt.canonical(path); rt.require(path.is_file() and ',' not in str(path) and '\n' not in str(path), 'Individual readonly literal mounts only')
            cmd += ['--mount', f'type=bind,src={path},dst={path},readonly']
        cmd += ['--mount', f'type=bind,src={OUTPUT},dst={OUTPUT}', IMAGE, '-i', 'PATH=/opt/conda/bin:/usr/bin:/bin',
            'HOME=/tmp', 'HF_HUB_OFFLINE=1', 'TRANSFORMERS_OFFLINE=1', 'OMP_NUM_THREADS=4', 'OPENBLAS_NUM_THREADS=4', 'MKL_NUM_THREADS=4',
            'WR_CODE='+str(code), 'WR_CODE_REVISION='+revision, 'WR_IMAGE_ID='+IMAGE,
            'WR_VCOCO_BANK_DEADLINE='+format(deadline-20, '.17g'), '/opt/conda/bin/python', '-I', '-B', str(code/NATIVE_FILES[0]),
            '--native', '--proof-bytes', str(proof_pin['bytes']), '--proof-sha256', proof_pin['sha256']]
        result = subprocess.run(cmd, env=dict(PATH='/usr/bin:/bin', HOME='/nonexistent', DOCKER_HOST='unix://'+str(ROOT/'docker.sock')),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=max(.001, deadline-time.monotonic()-20), check=False)
        host['native_exit_status'] = result.returncode
        host['native_report_identity'] = rt.identity(OUTPUT/'native.json', 2 << 20)
        report = rt.pinned(OUTPUT/'native.json', host['native_report_identity'], 2 << 20)
        rt.require(result.returncode == 0, 'Native process failed'); validate_report(report, p, revision, proof_pin, inputs)
        rt.require(report['runtime_identity'] == prior['owl']['native_runtime'], 'Original qualified native runtime required')
        host['native_images'] = report['images']
    except BaseException as exc: host['error_type'] = error(exc)
    finally:
        try:
            if created:
                s = OUTPUT.lstat(); rt.require((s.st_dev, s.st_ino, s.st_uid) == (owner.st_dev, owner.st_ino, owner.st_uid), 'Owned output replaced')
                cleanup_deadline=min(deadline+15,time.monotonic()+10)
                if (OUTPUT/'.container.cid').exists(): cleanup(OUTPUT/'.container.cid', name, revision, cleanup_deadline)
                else: rt.require(not command(['docker', 'ps', '-aq', '--no-trunc', '--filter', 'name=^/'+name+'$'], cleanup_deadline), 'Owned container survives without CID')
                host['owned_cleanup_verified'] = True
        except BaseException as exc:host.update(status='fail',cleanup_error_type=error(exc))
        try:rt.require(rt.source(ROOT,code,revision,ENTRY,HELPERS)==source and source_state(code)==source_stats,'Complete source changed')
        except BaseException as exc:host.update(status='fail',source_post_error_type=error(exc))
        try:rt.require(public_inputs(input_pin,count)==inputs,'Public source images changed')
        except BaseException as exc:host.update(status='fail',inputs_post_error_type=error(exc))
        try:rt.require(original.qualifications(code,recipe,live=True)==prior,'Original model/runtime qualification changed')
        except BaseException as exc:host.update(status='fail',runtime_post_error_type=error(exc))
        try:
            rt.require(not any(k in host for k in ('source_post_error_type','inputs_post_error_type','runtime_post_error_type'))
                and configuration(code, source) == (p, recipe)and (lock.lstat().st_dev, lock.lstat().st_ino) == (info.st_dev, info.st_ino),
                'Complete source/input/model/runtime changed')
            rt.require(command(['docker','image','inspect',IMAGE,'--format','{{.Id}}'],deadline)==IMAGE,'Qualified live image changed')
            rt.require(not command(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], deadline), 'GPU process survives')
            if 'native_images' in host:
                rt.require({x.name for x in OUTPUT.iterdir()} == {'proof.json', 'native.json', '.container.cid', *[r['file'] for r in host['native_images']]}
                    and sum(x.stat().st_size for x in OUTPUT.iterdir()) <= MAX_OUTPUT-(2 << 20)
                    and rt.identity(OUTPUT/'proof.json', 2 << 20) == proof_pin
                    and rt.identity(OUTPUT/'native.json', 2 << 20) == host['native_report_identity'], 'Exclusive complete output census differs')
                validate_report(report, p, revision, proof_pin, inputs)
            host['source_inputs_runtime_assets_rehashed_after'] = True; check(deadline)
            if 'error_type'not in host and 'cleanup_error_type'not in host and host.get('native_exit_status')==0:host['status']='pass'
        except BaseException as exc: host.update(status='fail', post_error_type=error(exc))
        os.close(fd)
        if created:
            s = rt.canonical(OUTPUT).lstat()
            rt.require((s.st_dev, s.st_ino, s.st_uid) == (owner.st_dev, owner.st_ino, owner.st_uid), 'Refuse foreign output publication')
            allowed = {'host.json', 'proof.json', 'native.json', '.container.cid'} | {f'image_{rgb_inputs.original_slot(r,16):06d}.npz' for r in inputs['images']}
            with (OUTPUT/'host.json').open('xb') as stream:
                os.fchmod(stream.fileno(),0o400);host_inode=os.fstat(stream.fileno())
                def update():
                    stream.seek(0);stream.write(encode(host));stream.truncate();stream.flush();os.fsync(stream.fileno())
                try:
                    for leaf in OUTPUT.iterdir():
                        rt.canonical(leaf);s=leaf.lstat()
                        rt.require(leaf.name in allowed and stat.S_ISREG(s.st_mode)and s.st_nlink==1 and s.st_uid==owner.st_uid,
                            'Only owned output leaves');leaf.chmod(0o400)
                    leaves={leaf.name:rt.identity(leaf,16 << 20)for leaf in OUTPUT.iterdir()if leaf.name!='host.json'}
                    OUTPUT.chmod(0o500);host.update(outputs_sealed=True,elapsed_seconds=time.monotonic()-started);update()
                    now=(OUTPUT/'host.json').lstat();s=OUTPUT.lstat()
                    rt.require((now.st_dev,now.st_ino)==(host_inode.st_dev,host_inode.st_ino)
                        and (s.st_dev,s.st_ino,s.st_uid)==(owner.st_dev,owner.st_ino,owner.st_uid)
                        and stat.S_IMODE(s.st_mode)==0o500 and {x.name for x in OUTPUT.iterdir()}==set(leaves)|{'host.json'}
                        and all(rt.identity(OUTPUT/n,16 << 20)==v for n,v in leaves.items())
                        and rt.identity(OUTPUT/'host.json',2 << 20)==dict(bytes=len(encode(host)),sha256=hashlib.sha256(encode(host)).hexdigest()),
                        'Final sealed output evidence differs');check(deadline)
                except BaseException as exc:host.update(status='fail',publication_error_type=error(exc));update()
    return host


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--native', action='store_true'); parser.add_argument('--manifest-bytes', type=int)
    parser.add_argument('--manifest-sha256'); parser.add_argument('--images', type=int)
    parser.add_argument('--proof-bytes', type=int); parser.add_argument('--proof-sha256')
    rt.require(all(sys.argv.count(n) <= 1 for n in ('--native', '--manifest-bytes', '--manifest-sha256', '--images', '--proof-bytes', '--proof-sha256')), 'Duplicate CLI forbidden')
    a = parser.parse_args(); code = Path(os.environ['WR_CODE']); revision = os.environ['WR_CODE_REVISION']
    rt.require(sys.platform == 'linux' and os.geteuid() == 0 and re.fullmatch('[0-9a-f]{40}', revision)
        and code == ROOT/'jobs'/revision/ENTRY/'code' and Path(__file__).resolve() == code/NATIVE_FILES[0], 'Actual immutable Azure source namespace required')
    if a.native:
        rt.require(a.manifest_bytes is a.manifest_sha256 is a.images is None, 'Native input only from host proof')
        pin = dict(bytes=a.proof_bytes, sha256=a.proof_sha256); proof = rt.pinned(OUTPUT/'proof.json', pin, 2 << 20)
        # Native whitelist lacks host-only qualification leaves; its whole
        # immutable byte proof was validated before transport by the host.
        p = rt.pinned(code/CONFIG, proof['native_files'][CONFIG], 16 << 10)
        deadline = float(os.environ['WR_VCOCO_BANK_DEADLINE']); check(deadline)
        def expired(*_): raise TimeoutError('Inclusive native deadline')
        old = {s: signal.signal(s, expired) for s in (signal.SIGTERM, signal.SIGALRM)}
        signal.setitimer(signal.ITIMER_REAL, max(.001, deadline-time.monotonic()))
        try: result = native(code, revision, p, pin, deadline)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            for s, handler in old.items(): signal.signal(s, handler)
    else:
        rt.require(os.uname().nodename == 'world-reward-ncc-h100-02' and a.proof_bytes is a.proof_sha256 is None, 'VM02 public manifest arguments required')
        def expired(*_):raise TimeoutError('Inclusive host endpoint deadline')
        old={s:signal.signal(s,expired)for s in(signal.SIGTERM,signal.SIGALRM)};signal.setitimer(signal.ITIMER_REAL,600)
        try:result=dispatch(code,revision,dict(bytes=a.manifest_bytes,sha256=a.manifest_sha256),a.images)
        finally:
            signal.setitimer(signal.ITIMER_REAL,0)
            for s,handler in old.items():signal.signal(s,handler)
    print(encode(dict(stage=result['stage'], status=result['status'], quality_verified=False)).decode().strip())
    return 0 if result['status'] == 'pass' else 1


if __name__ == '__main__':
    try: raise SystemExit(main())
    except Exception as exc:
        print(encode(dict(status='fail', error_type=error(exc))).decode().strip()); raise SystemExit(1) from None
