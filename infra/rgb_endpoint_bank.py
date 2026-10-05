"""Complete person-specific GDI + image-only OWLv2 banks on public original RGB.

No reference metadata, actor selection, SAM/DWPose, ranking or ownership. Original
runtime/model qualification remains separate from this fresh caller's source.
"""
import argparse
import base64
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
import mediapipe_cpu_runtime_verify as rt
import owlv2_native_qualify as owl
import openimages_joint_pair_gdi as gdi

ROOT = rt.ROOT
ENTRY = 'run_rgb_endpoint_bank'
CONFIG = 'configs/rgb_endpoint_bank_v2.json'
DATA = Path('/srv/world-reward-data/coco_endpoint_v2/inputs')
OUTPUT = ROOT/'results/coco-endpoint-banks-v2'
IMAGE = owl.IMAGE
NATIVE_FILES = ('infra/rgb_endpoint_bank.py', CONFIG, 'infra/mediapipe_cpu_runtime_verify.py',
    'infra/owlv2_native_qualify.py', 'configs/owlv2_native_qualification_v1.json',
    'infra/openimages_joint_pair_gdi.py', 'infra/bridge_frontend_bindings.py',
    'infra/frontend_selected_assets.py', 'src/world_reward/__init__.py',
    'src/world_reward/prompt_selection.py', 'src/world_reward/owlv2_object_observations.py')
HELPERS = (*NATIVE_FILES, 'infra/run_rgb_endpoint_bank.sh', 'infra/frontend_sam2_kernel_gate.py',
    'configs/frontend_grounding_source_pins.json', 'configs/frontend_asset_archive_pins.json', *owl.ACQUIRE_HELPERS)
GDI_SOURCES = ('transformers/models/grounding_dino/configuration_grounding_dino.py',
    'transformers/models/grounding_dino/modeling_grounding_dino.py',
    'transformers/models/grounding_dino/processing_grounding_dino.py',
    'transformers/models/grounding_dino/image_processing_grounding_dino.py')


def encode(value):
    import json
    return (json.dumps(value, sort_keys=True, allow_nan=False)+'\n').encode()


def write(path, value):
    raw = encode(value); rt.require(len(raw) <= 2 << 20, 'Bounded scalar receipt required')
    rt.write(path, raw); return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def check(deadline):
    if not math.isfinite(deadline) or time.monotonic() >= deadline:
        raise TimeoutError('Inclusive endpoint-bank deadline')


def error(exc):
    return type(exc).__name__ if type(exc) in (ValueError, RuntimeError, TimeoutError, OSError,
        KeyError, ImportError, ModuleNotFoundError) else 'other'


def configuration(code, source):
    p = rt.pinned(code/CONFIG, source['helpers'][CONFIG], 16 << 10)
    fixed = dict(schema='world_reward.rgb_endpoint_bank.v2', image_id=IMAGE, output=str(OUTPUT),
        input_directory=str(DATA), maximum_images=64, maximum_input_bytes=1 << 30,
        maximum_image_bytes=16 << 20, maximum_decoded_pixels=16 << 20,
        maximum_bank_bytes=16 << 20, maximum_output_bytes=1 << 30, budget_seconds=600,
        person_query='person.', confidence=.3, text_threshold=.25, nms_iou=.7,
        owl_patches=3600, model_loads=2, ground_truth_used=False, ownership_verified=False,
        quality_verified=False, adoption=False)
    rt.require(set(p) == set(fixed) | {'helper_pins', 'owl_qualification'}
        and all(type(p.get(k)) is type(v) and p[k] == v for k, v in fixed.items()), 'Fixed prospective bank policy required')
    for name, pin in p['helper_pins'].items():
        rt.require(name in NATIVE_FILES and source['helpers'][name] == pin, 'Original reused native helper changed')
    return p


def public_inputs(pin, count):
    """Only the generic complete public RGB projection; never a cohort/ref file."""
    rt.require(type(count) is int and 0 < count <= 64, 'Explicit acquired original-image count required')
    value = rt.pinned(DATA/'manifest.json', pin, 1 << 20)
    rt.require(type(value) is dict and set(value) == {'schema', 'images'}
        and value['schema'] == 'world_reward.rgb_proposal_inputs.v1'
        and type(value['images']) is list and len(value['images']) == count, 'Complete public-only RGB manifest required')
    ids = set(); names = set(); previous = -1; total = 0
    for row in value['images']:
        rt.require(type(row) is dict and set(row) == {'image_id', 'file', 'bytes', 'sha256', 'width', 'height'}
            and type(row['image_id']) is str and re.fullmatch('[0-9a-f]{32}', row['image_id'])
            and type(row['file']) is str and re.fullmatch(r'image_0000[0-5][0-9]\.jpg|image_00006[0-3]\.jpg', row['file']),
            'Opaque IDs and original bounded slot basenames required')
        slot = int(row['file'][6:12])
        rt.require(slot > previous and row['image_id'] not in ids and row['file'] not in names
            and type(row['bytes']) is int and 0 < row['bytes'] <= 16 << 20
            and type(row['sha256']) is str and re.fullmatch('[0-9a-f]{64}', row['sha256'])
            and all(type(row[k]) is int and row[k] > 0 for k in ('width', 'height'))
            and row['width']*row['height'] <= 16 << 20, 'Complete ordered original RGB slots required')
        previous = slot; ids.add(row['image_id']); names.add(row['file']); total += row['bytes']
        rt.require(rt.identity(DATA/row['file'], 16 << 20) == {k: row[k] for k in ('bytes', 'sha256')}, 'Original public RGB changed')
    rt.require(total <= 1 << 30 and {x.name for x in DATA.iterdir()} == {'manifest.json'} | names,
        'Exclusive public-only input inventory and byte budget required')
    return value


def qualifications(code, p, *, live=False):
    """Actual old procedural PASS + full source/runtime/assets, no old execution."""
    wanted = p['owl_qualification']; revision = wanted['producer_revision']
    old = ROOT/'jobs'/revision/owl.ENTRY/'code'
    source = rt.source(ROOT, old, revision, owl.ENTRY, owl.HELPERS)
    host = rt.pinned(owl.OUTPUT/'host.json', wanted['host'], 2 << 20)
    native = rt.pinned(owl.OUTPUT/'native.json', wanted['native'], 1 << 20)
    policy = rt.pinned(code/owl.CONFIG, source['helpers'][owl.CONFIG], 16 << 10)
    rt.require(rt.identity(code/owl.CONFIG) == source['helpers'][owl.CONFIG]
        and rt.identity(code/'infra/owlv2_native_qualify.py') == source['helpers']['infra/owlv2_native_qualify.py'],
        'Original qualified model implementation/config required')
    runtime = owl.runtime_proof(code, live=live); grounding = gdi.runtime_identity(code, live=live)
    acquisition = host['acquisition_binding']
    rt.require(owl.acquisition(acquisition['producer_revision'], acquisition['report_identity'], policy) == acquisition,
        'Actual original four-asset acquisition required')
    rt.require(host['schema'] == policy['schema'] and host['stage'] == 'owlv2_procedural_qualification_host'
        and host['status'] == 'pass' and host['producer_revision'] == revision and host['source_binding'] == source
        and host['runtime_binding'] == runtime and host['native_report_identity'] == wanted['native']
        and host['native_exit_status'] == 0 and host['image_id'] == IMAGE
        and host['owned_cleanup_verified'] is host['outputs_sealed'] is host['source_runtime_assets_rehashed_after'] is True
        and native['schema'] == policy['schema'] and native['stage'] == 'native_owlv2_procedural'
        and native['status'] == 'pass' and native['phase'] == 'complete' and native['producer_revision'] == revision
        and native['image_id'] == IMAGE and native['model_loads'] == 1
        and native['image_embed_calls'] == native['objectness_calls'] == native['box_calls'] == 3
        and native['source_runtime_assets_rehashed_after'] is True
        and native['dataset_read'] is native['quality_verified'] is native['ownership_verified'] is native['adopted'] is False,
        'Genuine complete original OWL qualification required')
    proof = rt.pinned(owl.OUTPUT/'proof.json', native['proof_identity'], 2 << 20)
    rt.require(proof['source'] == source and proof['image_id'] == IMAGE and proof['assets'] == policy['assets']
        and proof['native_files'] == {n: source['helpers'][n] for n in owl.NATIVE_FILES}
        and host['native_frames'] == native['frames'] and len(native['frames']) == 3,
        'Original full procedural source/output proof required')
    for i, row in enumerate(native['frames']):
        rt.require(row['file'] == f'procedural_{i:02d}.npz' and row['original_frame_index'] == i and row['patches'] == 3600
            and rt.identity(owl.OUTPUT/row['file'], 1 << 20) == row['identity'], 'Original procedural bank differs')
    rt.require({x.name for x in owl.OUTPUT.iterdir()} == {'host.json', 'native.json', 'proof.json', '.container.cid',
        *[f'procedural_{i:02d}.npz' for i in range(3)]}, 'Original complete qualification inventory required')
    return dict(runtime=runtime, grounding=grounding, acquisition=acquisition,
        owl=dict(producer_revision=revision, host=wanted['host'], native=wanted['native'],
            source_binding=source, native_runtime=native['runtime_identity']))


def asset_files(proof, policy):
    files = {str(gdi.binding.DEST/'weights'/name): dict(bytes=pin['bytes'], sha256=pin['sha256'])
        for name, pin in proof['grounding']['assets'].items()}
    files[str(gdi.binding.DEST/'weights/grounding_dino/README.md')] = proof['grounding']['model_card_identity']
    files.update({str(owl.ASSETS/name): pin for name, pin in policy['assets'].items()})
    return files


def expected_asset_files(policy):
    """Native validates fixed public leaves, not arbitrary host-provided paths."""
    files = {str(gdi.binding.DEST/'weights'/name): dict(bytes=size, sha256=digest)
        for name, (digest, size) in gdi.ASSETS.items()}
    files.update({str(owl.ASSETS/name): pin for name, pin in policy['assets'].items()})
    return files, str(gdi.binding.DEST/'weights/grounding_dino/README.md')


def installed_grounding(record_pin):
    """Full RECORD pinned by actual OWL qualification, not version-only trust."""
    from importlib import metadata
    dist = metadata.distribution('transformers'); records = [x for x in dist.files or () if str(x).endswith('.dist-info/RECORD')]
    rt.require(len(records) == 1 and rt.identity(Path(dist.locate_file(records[0])), 2 << 20, readonly=False) == record_pin,
        'Original qualified Transformers wheel RECORD differs')
    result = {}
    for name in GDI_SOURCES:
        entries = [x for x in dist.files if str(x) == name]
        rt.require(len(entries) == 1 and entries[0].hash is not None and entries[0].hash.mode == 'sha256', 'Original GDI RECORD entry required')
        entry = entries[0]; path = Path(dist.locate_file(entry))
        rt.require(path.is_relative_to('/opt/conda/lib/python3.11/site-packages') and type(entry.size) is int and 0 < entry.size < 1 << 20,
            'Original bounded installed GDI source required')
        actual = rt.identity(path, 1 << 20, readonly=False)
        rt.require(actual['bytes'] == entry.size and base64.urlsafe_b64encode(bytes.fromhex(actual['sha256'])).decode().rstrip('=') == entry.hash.value,
            'Original GDI implementation differs from qualified RECORD')
        result[name] = actual
    return result


def load_models(policy):
    """Unchanged native architectures/state; no prefix guessing or partial load."""
    import torch
    from safetensors.torch import load_file
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, Owlv2Config, Owlv2ForObjectDetection, Owlv2ImageProcessor
    from transformers.image_transforms import center_to_corners_format
    from transformers.models.owlv2.image_processing_owlv2 import _scale_boxes
    from world_reward.owlv2_object_observations import NativeOwlv2Operations
    rt.require(torch.cuda.is_available() and torch.cuda.device_count() == 1 and 'H100' in torch.cuda.get_device_name(), 'One actual H100 required')
    torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    rt.require(not torch.is_autocast_enabled() and not torch.is_autocast_enabled('cpu'), 'No ambient AMP')
    directory = gdi.binding.DEST/'weights/grounding_dino'
    processor = AutoProcessor.from_pretrained(directory, local_files_only=True)
    model, info = AutoModelForZeroShotObjectDetection.from_pretrained(directory, local_files_only=True, output_loading_info=True)
    rt.require(all(info.get(n) == [] for n in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs'))
        and all(not x.is_floating_point() or x.dtype == torch.float32 for x in model.state_dict().values()), 'Complete original GDI FP32 weights required')
    model = model.to('cuda').eval()
    object_model = Owlv2ForObjectDetection(Owlv2Config.from_json_file(str(owl.ASSETS/'config.json')))
    state = load_file(str(owl.ASSETS/'model.safetensors'), device='cpu'); owl.strict_state(object_model, state, torch); del state
    object_model = object_model.to('cuda').eval()
    op = Owlv2ImageProcessor(**rt.strict((owl.ASSETS/'preprocessor_config.json').read_bytes()))
    operations = NativeOwlv2Operations(op, torch, 'cuda', center_to_corners_format, _scale_boxes)
    def detect(rgb):
        from PIL import Image
        value = processor(images=Image.fromarray(rgb), text='person.', return_tensors='pt').to('cuda')
        with torch.inference_mode():
            prediction = model(**value)
            result = processor.post_process_grounded_object_detection(prediction, value.input_ids,
                threshold=.3, text_threshold=.25, target_sizes=[rgb.shape[:2]])[0]
        return (result['boxes'].cpu().numpy(), result['scores'].cpu().numpy(), result['text_labels'],
            prediction.pred_boxes.cpu().numpy(), prediction.logits.cpu().numpy(),
            value.input_ids.cpu().numpy(), value.attention_mask.cpu().numpy(), model.config.max_text_len)
    return detect, object_model, operations, (model, processor)


def decode(row):
    import numpy as np
    from PIL import Image
    rt.require(rt.identity(DATA/row['file'], 16 << 20) == {k: row[k] for k in ('bytes', 'sha256')}, 'Original RGB changed before decode')
    with Image.open(DATA/row['file']) as image:
        rt.require(image.format == 'JPEG' and image.mode == 'RGB' and image.size == (row['width'], row['height']), 'Original JPEG/grid required, no conversion')
        return np.array(image, copy=True)


def bank_arrays(rgb, row, index, detect, object_model, operations):
    import numpy as np
    from world_reward.owlv2_object_observations import infer_owlv2_object_frame
    digest = hashlib.sha256(rgb.tobytes()).hexdigest()
    boxes, scores, labels, raw_boxes, logits, ids, mask, length = detect(rgb)
    padding = gdi.validate_native_text_logits(raw_boxes, logits, ids, mask, length)
    person = gdi.retained_bank(boxes, scores, labels, row['image_id'], 'person', row['width'], row['height'])
    arrays = {'person_'+k: v for k, v in person.items()}
    arrays.update(person_model_pred_boxes=raw_boxes, person_model_logits=logits,
        person_model_input_ids=ids, person_model_attention_mask=mask)
    observation = infer_owlv2_object_frame(object_model, rgb, 0, operations)
    arrays.update(owl_patch_ids=observation.patch_ids, owl_boxes_padded_normalized_cxcywh=observation.boxes_padded_normalized_cxcywh,
        owl_objectness_logits=observation.objectness_logits, owl_boxes_original_xyxy=observation.boxes_original_xyxy,
        image_size=np.asarray(observation.image_size, dtype=np.int64), original_frame_index=np.asarray(0, dtype=np.int64))
    rt.require(hashlib.sha256(rgb.tobytes()).hexdigest() == digest, 'Native inference changed original RGB')
    metadata = dict(image_id=row['image_id'], original_frame_index=0, bank_index=index,
        image_size=[row['height'], row['width']], width=row['width'], height=row['height'],
        decoded_rgb_sha256=digest, input_file=row['file'], input_identity={k: row[k] for k in ('bytes', 'sha256')},
        person_native_queries=900, person_postprocessor_rows=len(boxes), person_retained_rows=len(person['retained_ids']),
        person_ids=person['retained_ids'].tolist(), native_text_padding=padding, owl_patches=3600)
    return arrays, metadata


def save_bank(out, index, arrays):
    import numpy as np
    path = out/f'image_{index:06d}.npz'
    with path.open('xb') as stream:
        os.fchmod(stream.fileno(), 0o400); np.savez(stream, **arrays); stream.flush(); os.fsync(stream.fileno())
    pin = rt.identity(path, 16 << 20)
    with np.load(path, allow_pickle=False) as saved:
        rt.require(set(saved.files) == set(arrays) and all(saved[n].dtype == a.dtype and saved[n].shape == a.shape
            and saved[n].tobytes() == a.tobytes() for n, a in arrays.items()), 'Lossless complete native array publication required')
    return dict(file=path.name, identity=pin, arrays={n: dict(shape=list(a.shape), dtype=a.dtype.str,
        sha256=hashlib.sha256(a.tobytes()).hexdigest()) for n, a in arrays.items()})


def native(code, revision, p, pin, deadline):
    started = time.monotonic(); proof = rt.pinned(OUTPUT/'proof.json', pin, 2 << 20)
    report = dict(schema=p['schema'], stage='native_all_person_owl_banks', status='fail', phase='authentication',
        producer_revision=revision, image_id=IMAGE, proof_identity=pin, model_loads=0, person_forward_calls=0,
        image_embed_calls=0, objectness_calls=0, box_calls=0, images=[], person_query='person.',
        confidence=.3, text_threshold=.25, nms_iou=.7, all_patches_retained=True,
        ground_truth_used=False, reference_metadata_read=False, split_metadata_read=False, challenge_inputs_used=False,
        actor_selection_performed=False, ownership_verified=False, quality_verified=False, adoption=False,
        dwpose_calls=0, sam_calls=0, hoi_calls=0, tracking_calls=0, network='none')
    try:
        rt.require(os.environ['WR_IMAGE_ID'] == IMAGE and {x.name for x in Path('/sys/class/net').iterdir()} == {'lo'}, 'Exact offline native image required')
        rt.require(set(proof) == {'source', 'native_files', 'assets', 'owl_runtime', 'inputs_identity', 'images', 'image_id'}
            and proof['source']['producer_revision'] == revision and proof['image_id'] == IMAGE
            and {str(x.relative_to(code)) for x in code.rglob('*') if x.is_file()} == set(NATIVE_FILES), 'Native individual-source whitelist required')
        for name, wanted in proof['native_files'].items(): rt.require(rt.identity(code/name, 2 << 20) == wanted, 'Mounted source differs')
        rt.require(set(proof['native_files']) == set(NATIVE_FILES), 'Complete native source census required')
        for name, wanted in proof['source']['markers'].items(): rt.require(rt.identity(code.parent/name, 100) == wanted, 'Dispatch marker differs')
        inputs = public_inputs(proof['inputs_identity'], proof['images']); policy = rt.pinned(code/owl.CONFIG, proof['native_files'][owl.CONFIG], 16 << 10)
        expected, card = expected_asset_files(policy)
        rt.require(set(proof['assets']) == set(expected) | {card}
            and all(proof['assets'][name] == wanted for name, wanted in expected.items()), 'Exact original model/config/card leaves required')
        for name, wanted in proof['assets'].items(): rt.require(rt.identity(Path(name), 1 << 30) == wanted, 'Original mounted model/card differs')
        runtime = owl.installed(policy)
        rt.require(runtime == proof['owl_runtime'], 'Actual qualified native runtime differs')
        grounding = installed_grounding(runtime['wheel_RECORD_identities']['transformers'])
        report.update(runtime_identity=runtime, grounding_source_identity=grounding, phase='models'); check(deadline)
        detect, model, operations, keepalive = load_models(policy); report['model_loads'] = 2
        for index, row in enumerate(inputs['images']):
            report['phase'] = 'native_forward'; check(deadline)
            rgb = decode(row); arrays, metadata = bank_arrays(rgb, row, index, detect, model, operations)
            operations.tensor_ops.cuda.synchronize()
            metadata.update(save_bank(OUTPUT, index, arrays)); report['images'].append(metadata)
            for key in ('person_forward_calls', 'image_embed_calls', 'objectness_calls', 'box_calls'): report[key] += 1
        del model, keepalive, detect
        operations.tensor_ops.cuda.empty_cache(); operations.tensor_ops.cuda.synchronize(); del operations
        report['phase'] = 'posthash'
        rt.require(owl.installed(policy) == runtime and installed_grounding(runtime['wheel_RECORD_identities']['transformers']) == grounding
            and public_inputs(proof['inputs_identity'], proof['images']) == inputs, 'Native source/runtime/original RGB changed')
        for name, wanted in proof['native_files'].items(): rt.require(rt.identity(code/name, 2 << 20) == wanted, 'Native mounted source changed')
        for name, wanted in proof['assets'].items(): rt.require(rt.identity(Path(name), 1 << 30) == wanted, 'Original mounted model/card changed')
        for row in report['images']: rt.require(rt.identity(OUTPUT/row['file'], 16 << 20) == row['identity'], 'Saved full bank changed')
        check(deadline); report.update(status='pass', phase='complete', source_inputs_runtime_assets_rehashed_after=True)
    except BaseException as exc: report.update(status='fail', error_type=error(exc))
    report['elapsed_seconds'] = time.monotonic()-started; write(OUTPUT/'native.json', report)
    return report


def command(args, deadline):
    check(deadline)
    result = subprocess.run(args, env=dict(PATH='/usr/bin:/bin', HOME='/nonexistent', DOCKER_HOST='unix://'+str(ROOT/'docker.sock')),
        capture_output=True, timeout=min(10, max(.001, deadline-time.monotonic())), check=False)
    rt.require(result.returncode == 0 and len(result.stdout) <= 1 << 20, 'Bounded lifecycle command failed')
    return result.stdout.decode().strip()


def cleanup(path, name, revision, deadline):
    rt.identity(path, 65, readonly=False); raw = path.read_bytes()
    rt.require(re.fullmatch(b'[0-9a-f]{64}\n?', raw), 'Original owned CID required'); cid = raw.decode().strip()
    present = command(['docker', 'ps', '-aq', '--no-trunc', '--filter', 'id='+cid], deadline)
    rt.require(present in ('', cid), 'Ambiguous owned CID')
    if present:
        actual = command(['docker', 'inspect', cid, '--format', '{{.Image}}|{{.Name}}|{{index .Config.Labels "world-reward.job"}}|{{index .Config.Labels "world-reward.revision"}}'], deadline)
        rt.require(actual == IMAGE+'|/'+name+'|'+ENTRY+'|'+revision, 'Foreign container cannot be removed')
        command(['docker', 'rm', '-f', cid], deadline)
    rt.require(not command(['docker', 'ps', '-aq', '--no-trunc', '--filter', 'name=^/'+name+'$'], deadline), 'Owned name survives')
    path.chmod(0o400)


def validate_report(report, p, revision, proof_pin, inputs):
    rt.require(report['schema'] == p['schema'] and report['stage'] == 'native_all_person_owl_banks'
        and report['status'] == 'pass' and report['phase'] == 'complete' and report['producer_revision'] == revision
        and report['image_id'] == IMAGE and report['proof_identity'] == proof_pin and type(report['model_loads']) is int and report['model_loads'] == 2
        and report['source_inputs_runtime_assets_rehashed_after'] is True and report['all_patches_retained'] is True
        and report['person_query'] == 'person.' and (report['confidence'], report['text_threshold'], report['nms_iou']) == (.3, .25, .7)
        and all(report[k] is False for k in ('ground_truth_used', 'reference_metadata_read', 'split_metadata_read',
            'challenge_inputs_used', 'actor_selection_performed', 'ownership_verified', 'quality_verified', 'adoption'))
        and all(type(report[k]) is int and report[k] == 0 for k in ('dwpose_calls', 'sam_calls', 'hoi_calls', 'tracking_calls'))
        and all(type(report[k]) is int and report[k] == len(inputs['images']) for k in ('person_forward_calls', 'image_embed_calls', 'objectness_calls', 'box_calls'))
        and len(report['images']) == len(inputs['images']), 'Actual complete no-selection native bank census required')
    for index, (row, image) in enumerate(zip(report['images'], inputs['images'])):
        rt.require(row['image_id'] == image['image_id'] and row['original_frame_index'] == 0 and row['bank_index'] == index
            and row['image_size'] == [image['height'], image['width']] and row['input_file'] == image['file']
            and row['input_identity'] == {k: image[k] for k in ('bytes', 'sha256')}
            and row['file'] == f'image_{index:06d}.npz' and row['owl_patches'] == 3600 and row['person_native_queries'] == 900
            and type(row['person_retained_rows']) is int and 0 <= row['person_retained_rows'] <= row['person_postprocessor_rows'] <= 900
            and len(row['person_ids']) == len(set(row['person_ids'])) == row['person_retained_rows']
            and rt.identity(OUTPUT/row['file'], 16 << 20) == row['identity'], 'Every original full endpoint bank required')


def dispatch(code, revision, input_pin, count):
    import fcntl
    started = time.monotonic(); deadline = started+600
    source = rt.source(ROOT, code, revision, ENTRY, HELPERS); p = configuration(code, source)
    inputs = public_inputs(input_pin, count); prior = qualifications(code, p, live=True)
    policy = rt.pinned(code/owl.CONFIG, source['helpers'][owl.CONFIG], 16 << 10); assets = asset_files(prior, policy)
    rt.require(not OUTPUT.exists() and OUTPUT.parent.is_dir(), 'Fresh fixed endpoint output required')
    lock = rt.canonical(ROOT/'jobs/.world-reward-h100.lock'); info = lock.lstat()
    rt.require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, 'Single-link cooperative H100 lock required')
    fd = os.open(lock, os.O_RDONLY | os.O_NOFOLLOW); name = 'world-reward-endpoint-bank-'+revision[:12]; created = False
    host = dict(schema=p['schema'], stage='rgb_endpoint_bank_host', status='fail', producer_revision=revision,
        source_binding=source, original_qualification=prior, public_inputs_identity=input_pin, acquired_images=count,
        image_id=IMAGE, ground_truth_used=False, quality_verified=False, ownership_verified=False, adoption=False,
        owned_cleanup_verified=False, outputs_sealed=False)
    try:
        rt.require((os.fstat(fd).st_dev, os.fstat(fd).st_ino) == (info.st_dev, info.st_ino), 'Lock inode differs')
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rt.require(not command(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], deadline)
            and not command(['docker', 'ps', '-aq', '--no-trunc', '--filter', 'name=^/'+name+'$'], deadline), 'Other GPU work or occupied owned name')
        OUTPUT.mkdir(mode=0o700); created = True; owner = OUTPUT.lstat()
        safe = dict(source=source, native_files={n: source['helpers'][n] for n in NATIVE_FILES}, assets=assets,
            owl_runtime=prior['owl']['native_runtime'], inputs_identity=input_pin, images=count, image_id=IMAGE)
        proof_pin = write(OUTPUT/'proof.json', safe)
        mounts = [code/n for n in NATIVE_FILES]+[code.parent/n for n in ('revision', 'source-sha256')]
        mounts += [Path(n) for n in assets]+[DATA/'manifest.json']+[DATA/r['file'] for r in inputs['images']]
        cmd = ['docker', 'run', '--rm', '--name', name, '--cidfile', str(OUTPUT/'.container.cid'),
            '--label', 'world-reward.job='+ENTRY, '--label', 'world-reward.revision='+revision, '--gpus', 'device=0',
            '--network', 'none', '--user', '0:0', '--memory', '64g', '--cpus', '4', '--pids-limit', '256', '--read-only',
            '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges', '--tmpfs', '/tmp:rw,noexec,nosuid,size=512m', '--entrypoint', '/usr/bin/env']
        for path in mounts:
            rt.canonical(path); rt.require(path.is_file() and ',' not in str(path) and '\n' not in str(path), 'Individual pinned readonly files only')
            cmd += ['--mount', f'type=bind,src={path},dst={path},readonly']
        cmd += ['--mount', f'type=bind,src={OUTPUT},dst={OUTPUT}', IMAGE, '-i', 'PATH=/opt/conda/bin:/usr/bin:/bin', 'HOME=/tmp',
            'HF_HUB_OFFLINE=1', 'TRANSFORMERS_OFFLINE=1', 'OMP_NUM_THREADS=4', 'OPENBLAS_NUM_THREADS=4', 'MKL_NUM_THREADS=4',
            'WR_CODE='+str(code), 'WR_CODE_REVISION='+revision, 'WR_IMAGE_ID='+IMAGE,
            'WR_ENDPOINT_DEADLINE='+format(deadline-20, '.17g'), '/opt/conda/bin/python', '-I', '-B', str(code/NATIVE_FILES[0]),
            '--native', '--proof-bytes', str(proof_pin['bytes']), '--proof-sha256', proof_pin['sha256']]
        result = subprocess.run(cmd, env=dict(PATH='/usr/bin:/bin', HOME='/nonexistent', DOCKER_HOST='unix://'+str(ROOT/'docker.sock')),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=max(.001, deadline-time.monotonic()-20), check=False)
        host['native_exit_status'] = result.returncode
        host['native_report_identity'] = rt.identity(OUTPUT/'native.json', 2 << 20)
        report = rt.pinned(OUTPUT/'native.json', host['native_report_identity'], 2 << 20)
        rt.require(result.returncode == 0, 'Native endpoint process failed')
        validate_report(report, p, revision, proof_pin, inputs)
        rt.require(report['runtime_identity'] == prior['owl']['native_runtime'], 'Native runtime must equal original actual qualification')
        host['native_images'] = report['images']
    except BaseException as exc: host['error_type'] = error(exc)
    finally:
        try:
            if created:
                current = OUTPUT.lstat(); rt.require((current.st_dev, current.st_ino, current.st_uid) == (owner.st_dev, owner.st_ino, owner.st_uid), 'Owned output replaced')
                cleanup(OUTPUT/'.container.cid', name, revision, time.monotonic()+10); host['owned_cleanup_verified'] = True
            rt.require(rt.source(ROOT, code, revision, ENTRY, HELPERS) == source and configuration(code, source) == p
                and public_inputs(input_pin, count) == inputs and qualifications(code, p, live=True) == prior
                and (lock.lstat().st_dev, lock.lstat().st_ino) == (info.st_dev, info.st_ino), 'Complete original source/input/model/runtime changed')
            rt.require(not command(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], deadline), 'GPU process survives')
            if 'native_images' in host:
                rt.require({x.name for x in OUTPUT.iterdir()} == {'proof.json', 'native.json', '.container.cid', *[r['file'] for r in host['native_images']]}
                    and sum(x.stat().st_size for x in OUTPUT.iterdir()) <= 1 << 30
                    and rt.identity(OUTPUT/'proof.json', 2 << 20) == proof_pin
                    and rt.identity(OUTPUT/'native.json', 2 << 20) == host['native_report_identity'], 'Exclusive full output census/bytes changed')
                validate_report(report, p, revision, proof_pin, inputs)
            host['source_inputs_runtime_assets_rehashed_after'] = True; check(deadline)
            if 'error_type' not in host and host.get('native_exit_status') == 0: host['status'] = 'pass'
        except BaseException as exc: host.update(status='fail', post_error_type=error(exc))
        os.close(fd)
        if created:
            current = rt.canonical(OUTPUT).lstat()
            rt.require((current.st_dev, current.st_ino, current.st_uid) == (owner.st_dev, owner.st_ino, owner.st_uid), 'Refuse publication into replaced output namespace')
            with (OUTPUT/'host.json').open('xb') as stream:
                os.fchmod(stream.fileno(), 0o400)
                for leaf in OUTPUT.iterdir():
                    rt.canonical(leaf); s = leaf.lstat()
                    allowed = {'host.json', 'proof.json', 'native.json', '.container.cid'} | {f'image_{i:06d}.npz' for i in range(count)}
                    rt.require(leaf.name in allowed and stat.S_ISREG(s.st_mode) and s.st_nlink == 1 and s.st_uid == owner.st_uid, 'Only owned single-link output leaves')
                    leaf.chmod(0o400)
                OUTPUT.chmod(0o500); host['outputs_sealed'] = True; host['elapsed_seconds'] = time.monotonic()-started
                if time.monotonic() >= deadline: host.update(status='fail', post_error_type='TimeoutError')
                stream.write(encode(host)); stream.flush(); os.fsync(stream.fileno())
                if time.monotonic() >= deadline and host['status'] == 'pass':
                    host.update(status='fail', post_error_type='TimeoutError'); stream.seek(0); stream.write(encode(host)); stream.truncate(); stream.flush(); os.fsync(stream.fileno())
    return host


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--native', action='store_true'); parser.add_argument('--manifest-bytes', type=int)
    parser.add_argument('--manifest-sha256'); parser.add_argument('--images', type=int)
    parser.add_argument('--proof-bytes', type=int); parser.add_argument('--proof-sha256')
    rt.require(all(sys.argv.count(n) <= 1 for n in ('--native', '--manifest-bytes', '--manifest-sha256', '--images', '--proof-bytes', '--proof-sha256')), 'Duplicate CLI inputs forbidden')
    a = parser.parse_args(); code = Path(os.environ['WR_CODE']); revision = os.environ['WR_CODE_REVISION']
    rt.require(sys.platform == 'linux' and os.geteuid() == 0 and re.fullmatch('[0-9a-f]{40}', revision)
        and code == ROOT/'jobs'/revision/ENTRY/'code' and Path(__file__).resolve() == code/NATIVE_FILES[0], 'Original Azure source namespace required')
    if a.native:
        rt.require(a.manifest_bytes is a.manifest_sha256 is a.images is None, 'Native public inputs come only from sealed host proof')
        pin = dict(bytes=a.proof_bytes, sha256=a.proof_sha256); proof = rt.pinned(OUTPUT/'proof.json', pin, 2 << 20)
        p = configuration(code, proof['source']); deadline = float(os.environ['WR_ENDPOINT_DEADLINE']); check(deadline)
        def expired(*_): raise TimeoutError('Inclusive native bank deadline')
        old = {s: signal.signal(s, expired) for s in (signal.SIGTERM, signal.SIGALRM)}
        signal.setitimer(signal.ITIMER_REAL, max(.001, deadline-time.monotonic()))
        try: result = native(code, revision, p, pin, deadline)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            for s, handler in old.items(): signal.signal(s, handler)
    else:
        rt.require(os.uname().nodename == 'world-reward-ncc-h100-02' and a.proof_bytes is a.proof_sha256 is None, 'VM02 host-only public manifest args required')
        result = dispatch(code, revision, dict(bytes=a.manifest_bytes, sha256=a.manifest_sha256), a.images)
    print(encode(dict(stage=result['stage'], status=result['status'], quality_verified=False)).decode().strip())
    return 0 if result['status'] == 'pass' else 1


if __name__ == '__main__':
    try: raise SystemExit(main())
    except Exception as exc:
        print(encode(dict(status='fail', error_type=error(exc))).decode().strip()); raise SystemExit(1) from None
