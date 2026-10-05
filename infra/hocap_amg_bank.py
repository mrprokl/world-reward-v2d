"""Native class-agnostic HO-Cap AMG bank, RGB-only and no association claim.

Host authenticates the opaque extraction receipt but sends only a sanitized
public proof to inference. Child decodes EVERY original JPEG before model load,
retaining ten anchor RGB arrays only. All native returned masks/zero-query seeds
are preserved; fixed resource limits abort, never truncate the candidate bank.
"""
from __future__ import annotations

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
import warnings

import bridge_frontend_bindings as binding

ROOT = binding.ROOT
ENTRY = 'run_hocap_amg_bank'
PROTOCOL = 'configs/hocap_amg_protocol_v1.json'
PROTOCOL_PIN = dict(bytes=2172,sha256='5f63a6c82e3594cda799a4e167c296be2471b8c75bb2090b2c3401d1301de1d3')
HELPERS = ('infra/hocap_amg_bank.py', 'infra/run_hocap_amg_bank.sh', PROTOCOL,
    'infra/bridge_frontend_bindings.py', 'infra/frontend_selected_assets.py',
    'infra/frontend_sam2_kernel_gate.py', binding.CONFIG, binding.selected.PINS)
EXTRACTION_HELPERS = ('infra/hocap_extract.py', 'infra/run_hocap_extract.sh',
    'infra/hocap_acquire.py', 'infra/mediapipe_cpu_runtime_verify.py',
    'configs/hocap_extraction_protocol_v1.json')
require = binding.require


def check(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError('Inclusive AMG bank deadline exhausted')


def write_json(path, value):
    import json
    raw = (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()
    with Path(path).open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno()); os.fchmod(stream.fileno(), 0o444)
    return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def json_digest(value):
    import json
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        allow_nan=False, default=str).encode()).hexdigest()


def source_binding(code, revision):
    code = binding.canonical(code)
    require(Path(__file__).resolve() == code/'infra/hocap_amg_bank.py'
        and set(HELPERS) <= {str(p.relative_to(code)) for p in code.rglob('*') if p.is_file()},
        'Required helpers in complete frozen original source closure required')
    require(binding.identity(code/'infra/frontend_sam2_kernel_gate.py', 200000)
        == binding.KERNEL_SOURCE_PIN, 'Original measured kernel helper required')
    return binding.kernel_helper().closure(code, revision, ENTRY, HELPERS)


def protocol(code):
    require(PROTOCOL_PIN is not None, 'Frozen protocol byte pin missing')
    return binding.pinned(Path(code)/PROTOCOL, PROTOCOL_PIN, 16384)


def frontend_proof(code, p, *, live=False):
    """Reuse genuine old build/kernel/source proofs, never spoof bridge ENTRIES."""
    kernel = binding.kernel_helper()
    build = binding.pinned(binding.BUILD_REPORT, binding.BUILD_PIN)
    receipt = binding.pinned(binding.KERNEL_REPORT, binding.KERNEL_PIN)
    old = kernel.closure(binding.BUILD_CODE, binding.BUILD_REV,
        'run_frontend_grounding_build', kernel.BUILD_HELPERS)
    gate = kernel.closure(binding.KERNEL_CODE, binding.KERNEL_REV, 'run_frontend_sam2_kernel_gate',
        ('infra/frontend_sam2_kernel_gate.py', 'infra/run_frontend_sam2_kernel_gate.sh', binding.CONFIG))
    require(old['helpers']['infra/frontend_grounding_build.py']['sha256'] == binding.BUILD_SHA
        and gate['helpers']['infra/frontend_sam2_kernel_gate.py'] == binding.KERNEL_SOURCE_PIN
        and build.get('source_binding') == old and receipt.get('source_binding') == gate
        and binding.identity(Path(code)/binding.CONFIG) == old['helpers'][binding.CONFIG]
        == gate['helpers'][binding.CONFIG], 'Measured source/build/kernel ancestry differs')
    facts = dict(schema='world_reward.frontend_grounding_build.v6', stage='frontend_grounding_build',
        status='pass', phase='complete', producer_revision=binding.BUILD_REV, offline_build_exit_code=0,
        child_probe_exit_code=0, parent_unchanged_verified=True, source_rechecked_before_and_after=True,
        extension_import_verified=True, original_grounding_image_parity_claimed=False,
        CUDA_execution_verified=False, replica_ready=False, license_eligibility_verified=False,
        training_overlap_verified=False)
    require(all(type(build.get(k)) is type(v) and build[k] == v for k,v in facts.items()),
        'Genuine complete CPU build required')
    facts = dict(schema='world_reward.frontend_sam2_kernel_gate.v1', stage='frontend_sam2_kernel_gate',
        status='pass', producer_revision=binding.KERNEL_REV,
        script_sha256=binding.KERNEL_SOURCE_PIN['sha256'], image_id=binding.IMAGE, models_loaded=False,
        challenge_data_read=False, CUDA_operator_execution_verified=True,
        build_report_identity=binding.BUILD_PIN, replica_ready=False)
    require(all(type(receipt.get(k)) is type(v) and receipt[k] == v for k,v in facts.items())
        and receipt.get('operator_source_identities', {}).get('extension') == binding.EXTENSION_PIN,
        'Genuine measured CUDA operator required')
    child, parent = build['child_image'], build['parent_image']
    owner = hashlib.sha256((binding.BUILD_REV+old['closure_sha256']).encode()).hexdigest()
    require(child['Id'] == binding.IMAGE == p['image_id'] and parent['Id'] == kernel.BASE
        and child['Architecture'] == 'amd64' and child['Os'] == 'linux'
        and len(parent['RootFS']['Layers']) == 44
        and child['RootFS']['Layers'][:44] == parent['RootFS']['Layers']
        and build.get('owner') == owner, 'Original source-owned frontend image/rootfs differs')
    probe = build.get('private_child_probe_log', {})
    require(probe.get('relative_path') == 'results/frontend-grounding-build-v6/child-CPU-probe.log'
        and binding.identity(ROOT/probe['relative_path'], 128*1024)
        == {k:probe[k] for k in ('bytes','sha256')}, 'Original CPU import log differs')
    cfg = binding.strict_json((Path(code)/binding.CONFIG).read_bytes())
    sam = [r for r in cfg['repositories'] if r['repo'] == 'facebookresearch/sam2']
    require(len(sam) == 1 and sam[0]['revision'] == p['sam2_source_revision'], 'Original SAM2 source revision differs')
    wanted = p['installed_config']; source_name = 'sam2/sam2/'+wanted['path']
    require({k:build['source_files'].get(source_name, {}).get(k) for k in ('bytes','sha256')}
        == {k:wanted[k] for k in ('bytes','sha256')}, 'Original installed configuration differs')
    selected = binding.selected.load_contract(ROOT, binding.MANIFEST)
    cp = p['checkpoint']
    assets = binding.selected_assets(selected, {cp['relative_path']:(cp['sha256'], cp['bytes'])})
    acquisition = binding.strict_json((binding.DEST/'results/weights-acquisition.json').read_bytes())
    sam_rows = [r for r in acquisition.get('assets', []) if r.get('repo_id') == 'facebook/sam2.1-hiera-large']
    require(len(sam_rows) == 1 and sam_rows[0].get('revision') == p['sam2_checkpoint_revision']
        and sam_rows[0].get('path') == str(ROOT/'weights/sam2'), 'Original selected checkpoint revision differs')
    proof = dict(child_image=child, parent_image=parent, owner=owner, source_files=build['source_files'],
        build_report_identity=binding.BUILD_PIN, kernel_report_identity=binding.KERNEL_PIN,
        selected_manifest_identity={k:selected['manifest_identity'][k] for k in ('bytes','sha256')},
        checkpoint=assets, extension_identity=binding.EXTENSION_PIN)
    if live: kernel.validate_live_image(proof)
    return proof


def input_receipt(path, pin, p):
    """Independent bytes are verified BEFORE JSON. Opaque fields are not exported."""
    require(binding.canonical(path) == Path(p['input_report']), 'Exact original extraction receipt path required')
    require(stat.S_IMODE(Path(path).lstat().st_mode) == 0o444, 'Original sealed extraction receipt mode444 required')
    value = binding.pinned(path, pin, p['maximum_input_report_bytes'])
    facts = dict(schema='world_reward.hocap_extraction.v1', stage='hocap_saved_rgb_private_metadata_extraction',
        status='pass', phase='complete', public_inventory_qualified=True,
        source_archives_public_rehashed_after=True, network_used=False, label_values_parsed=False,
        calibration_values_parsed=False, model_loaded=False, gpu_used=False, challenge_inputs_used=False,
        image_decoder_qualified=False, reference_continuity_qualified=False, adoption=False)
    require(all(type(value.get(k)) is type(v) and value[k] == v for k,v in facts.items()),
        'Genuine complete untouched extraction PASS required')
    revision = value.get('producer_revision')
    require(type(revision) is str and re.fullmatch('[0-9a-f]{40}', revision), 'Original extraction producer required')
    old_code = ROOT/'jobs'/revision/'run_hocap_extract/code'
    # Its original digest algorithm is intentionally reused, not redefined.
    helper = old_code/'infra/mediapipe_cpu_runtime_verify.py'
    import importlib.util
    require(binding.identity(helper, 200000) == value.get('source_before', {}).get('helpers', {}).get(
        'infra/mediapipe_cpu_runtime_verify.py'), 'Source-bound original CPU helper required before import')
    spec = importlib.util.spec_from_file_location('hocap_amg_original_extraction_source', helper)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    old = module.source(ROOT, old_code, revision, 'run_hocap_extract', EXTRACTION_HELPERS)
    require(old == value.get('source_before') and module.identity(old_code/'configs/hocap_extraction_protocol_v1.json', 16384)
        == value.get('protocol_identity'), 'Actual original extraction source/config/markers differ')
    require(type(value.get('frames')) is int and value['frames'] > 0
        and type(value.get('clips')) is list and len(value['clips']) == 2,
        'Exact full public extraction clip/frame inventory required')
    return dict(schema='world_reward.hocap_amg_public_input_proof.v1', input_report_identity=pin,
        extraction_producer_revision=revision, extraction_source_sha256=old['closure_sha256'],
        public_manifest_identity=value['public_manifest'], frames=value['frames'], clips=value['clips'],
        private_labels_read=False, calibration_read=False, opaque_metadata_exported=False,
        challenge_inputs_used=False)


def validate_sanitized(value, input_pin, p):
    fields = {'schema','input_report_identity','extraction_producer_revision','extraction_source_sha256',
        'public_manifest_identity','frames','clips','private_labels_read','calibration_read',
        'opaque_metadata_exported','challenge_inputs_used','source_binding','frontend_proof_sha256','protocol_identity'}
    require(type(value) is dict and set(value) == fields
        and value['schema'] == 'world_reward.hocap_amg_public_input_proof.v1'
        and value['input_report_identity'] == input_pin and value['protocol_identity'] == PROTOCOL_PIN
        and all(value[k] is False for k in ('private_labels_read','calibration_read','opaque_metadata_exported','challenge_inputs_used'))
        and type(value['frames']) is int and value['frames'] > 0
        and type(value['clips']) is list and len(value['clips']) == 2
        and re.fullmatch('[0-9a-f]{40}',str(value['extraction_producer_revision']))
        and all(re.fullmatch('[0-9a-f]{64}',str(value[k])) for k in ('extraction_source_sha256','frontend_proof_sha256')),
        'Exact sanitized public-only proof fields required')
    for expected,clip in zip(p['clips'],value['clips']):
        require(type(clip) is dict and set(clip) == {'clip','camera','num_frames'}
            and clip['clip'] == expected and clip['camera'] == p['camera']
            and type(clip['num_frames']) is int and 5 <= clip['num_frames'] <= p['maximum_frames_per_clip'],
            'Only exact public clip/camera/frame counts may be exported')
    require(value['frames'] == sum(c['num_frames'] for c in value['clips']), 'Sanitized full frame count differs')


def public_inputs(directory, manifest_pin, p, deadline):
    """Hash all original RGB; only exact safe public fields/paths are accepted."""
    directory = binding.canonical(directory)
    require(directory == Path(p['inputs']), 'Exact selected RGB-only extraction directory required')
    value = binding.pinned(directory/'manifest.json', manifest_pin, p['maximum_manifest_bytes'])
    fields = {'schema','subject','clips','images','raw_metadata_public','annotations_public',
        'calibration_public','timestamps_verified','image_decoder_qualified','reference_continuity_qualified'}
    require(type(value) is dict and set(value) == fields and value['schema'] == 'world_reward.hocap_rgb_inventory.v1'
        and value['subject'] == p['subject'] and all(value[k] is False for k in fields
        - {'schema','subject','clips','images'}), 'Strict original RGB-only manifest required')
    clips = value['clips']; images = value['images']
    require(type(clips) is list and len(clips) == 2 and type(images) is list,
        'Two fixed original full clips required')
    records = []; wanted_files = {'manifest.json'}; offset = 0
    for expected, clip in zip(p['clips'], clips):
        require(type(clip) is dict and set(clip) == {'clip','camera','num_frames'}
            and clip['clip'] == expected and clip['camera'] == p['camera']
            and type(clip['num_frames']) is int and 5 <= clip['num_frames'] <= p['maximum_frames_per_clip'],
            'Fixed original clip/camera and full frame count required')
        rows = []
        for t in range(clip['num_frames']):
            check(deadline); require(offset < len(images), 'Missing original RGB frame')
            row = images[offset]; offset += 1
            name = expected+f'/color_{t:06d}.jpg'
            require(type(row) is dict and set(row) == {'clip','camera','frame_position','source_frame_id','file','bytes','sha256'}
                and row['clip'] == expected and row['camera'] == p['camera']
                and type(row['frame_position']) is type(row['source_frame_id']) is int
                and row['frame_position'] == row['source_frame_id'] == t and row['file'] == name,
                'Original RGB indices/order/file grid required')
            pin = {k:row[k] for k in ('bytes','sha256')}
            binding.validate_file_pins({name:pin}, {name}, p['maximum_rgb_bytes'])
            require(binding.identity(directory/name, p['maximum_rgb_bytes']) == pin, 'Original RGB bytes changed')
            wanted_files.add(name); rows.append({**row, 'path':directory/name})
        records.append({**clip, 'records':rows})
    require(offset == len(images), 'Extra original RGB record')
    actual_files = set()
    for path in directory.rglob('*'):
        binding.canonical(path); s = path.lstat()
        require(not s.st_mode & 0o222 and (stat.S_ISDIR(s.st_mode) or stat.S_ISREG(s.st_mode)),
            'Readonly public files only; no symlink/special path')
        relative = str(path.relative_to(directory))
        if path.is_dir(): require(relative in p['clips'], 'Unexpected private public directory')
        else: actual_files.add(relative)
    require(actual_files == wanted_files, 'Exact public JPEG/manifest inventory, no private payload')
    return records, value


def decode_anchors(clips, p, deadline, progress):
    """Native Pillow JPEG RGB only; decode all T, retain the ten fixed anchors."""
    import numpy as np
    from PIL import Image
    selected = []; grid = None; progress['rgb_decodes'] = 0
    for clip in clips:
        count = clip['num_frames']; anchor_ids = tuple(k*(count-1)//4 for k in range(5))
        require(len(set(anchor_ids)) == 5, 'Five unique original anchors required')
        for row in clip['records']:
            check(deadline)
            with warnings.catch_warnings():
                warnings.simplefilter('error')
                with Image.open(row['path']) as image:
                    require(image.format == 'JPEG' and image.mode == 'RGB', 'Native original RGB JPEG required; no conversion')
                    width, height = image.size
                    require(width > 0 and height > 0 and width*height <= p['maximum_image_pixels'], 'Bounded original JPEG grid required')
                    if grid is None: grid = (height,width)
                    require(grid == (height,width), 'Every original JPEG must share one native grid')
                    image.load(); rgb = np.asarray(image)
                    require(rgb.dtype == np.uint8 and rgb.shape == (height,width,3), 'Native complete RGB bytes required')
                    if row['source_frame_id'] in anchor_ids:
                        selected.append(dict(clip=clip['clip'], frame_index=row['source_frame_id'],
                            frames=count, width=width, height=height, rgb=rgb.copy(),
                            rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest()))
            progress['rgb_decodes'] += 1
            del rgb
    require(progress['rgb_decodes'] == sum(c['num_frames'] for c in clips) and len(selected) == 10,
        '100% original JPEG decode before model load required')
    progress.update(image_size=list(grid), anchors_retained=10, all_original_jpegs_decoded_before_model=True)
    return selected


def bank_arrays(records, row, p):
    """Preserve ALL native outputs, packed masks and fixed automatic query seeds."""
    import numpy as np
    require(type(records) is list and len(records) <= p['maximum_masks_per_anchor'],
        'Native list or fixed resource bound failed; never truncate')
    height, width = row['height'], row['width']; count = len(records)
    packed = np.empty((count, (height*width+7)//8), np.uint8)
    boxes = np.empty((count,4), np.float64); points = np.empty((count,2), np.float64)
    crops = np.empty((count,4), np.float64); scores = np.empty(count, np.float64)
    stability = np.empty(count, np.float64); areas = np.empty(count, np.int64)
    ids = []; hashes = []; queries = []; offsets = [0]
    keys = {'segmentation','bbox','area','predicted_iou','point_coords','stability_score','crop_box'}
    for i, record in enumerate(records):
        require(type(record) is dict and set(record) == keys, 'Exact native AMG record fields required')
        mask = record['segmentation']
        require(type(mask) is np.ndarray and mask.dtype == np.bool_ and mask.shape == (height,width),
            'Native binary original-grid AMG mask required')
        require(type(record['area']) is int and record['area'] == int(mask.sum()), 'Native mask area changed')
        for name, shape in (('bbox',(4,)),('crop_box',(4,)),('point_coords',(1,2))):
            value = np.asarray(record[name])
            require(not np.ma.isMaskedArray(record[name]) and value.shape == shape
                and value.dtype.kind in 'fiu' and np.isfinite(value).all(), 'Finite native AMG coordinates required')
        boxes[i], crops[i], points[i] = record['bbox'], record['crop_box'], record['point_coords'][0]
        require(np.array_equal(crops[i], [0,0,width,height]) and np.all(boxes[i,2:] >= 0),
            'Frozen crop0/native nonnegative XYWH boxes required')
        for name in ('predicted_iou','stability_score'):
            require(type(record[name]) in (float,int) and math.isfinite(record[name])
                and 0 <= record[name] <= 1, 'Finite raw native quality score required')
        scores[i], stability[i], areas[i] = record['predicted_iou'], record['stability_score'], record['area']
        packed[i] = np.packbits(mask.ravel(), bitorder=p['packed_mask_bitorder'])
        digest = hashlib.sha256(mask.tobytes()).hexdigest(); hashes.append(digest)
        ids.append(f"{row['clip']}:{row['frame_index']:06d}:{i:06d}:{digest}")
        x0,y0,bw,bh = boxes[i]
        require(np.isfinite([x0+bw,y0+bh]).all(), 'Finite native XYWH endpoints required')
        x0,x1 = np.clip([x0,x0+bw],0,width); y0,y1 = np.clip([y0,y0+bh],0,height)
        q = np.empty((0,3), np.float64)
        if x1 > x0 and y1 > y0:
            side = p['query_grid_side']
            xx = np.floor(x0+(np.arange(side)+.5)*(x1-x0)/side).astype(np.int64)
            yy = np.floor(y0+(np.arange(side)+.5)*(y1-y0)/side).astype(np.int64)
            x,y = np.meshgrid(xx,yy); pixels = np.unique(np.column_stack((y.ravel(),x.ravel())),axis=0)
            # Floating endpoints may round onto the image boundary. Exclude
            # those unavailable pixels, never move/refill them onto the mask.
            pixels = pixels[(pixels[:,0]>=0)&(pixels[:,0]<height)&(pixels[:,1]>=0)&(pixels[:,1]<width)]
            pixels = pixels[mask[pixels[:,0],pixels[:,1]]]
            q = np.column_stack((np.full(len(pixels),row['frame_index']),pixels.astype(np.float64)+.5))
        queries.append(q); offsets.append(offsets[-1]+len(q))
    identifiers = np.array(ids, dtype='U128'); offsets = np.array(offsets,np.int64)
    q = np.concatenate(queries) if queries else np.empty((0,3),np.float64)
    owners = np.repeat(np.arange(count,dtype=np.int64),np.diff(offsets))
    arrays = dict(frame_index=np.arange(row['frames'],dtype=np.int64), anchor_frame_index=np.array(row['frame_index'],np.int64),
        image_size=np.array([height,width],np.int64), packed_masks=packed, mask_ids=identifiers,
        mask_sha256=np.array(hashes,dtype='U64'), native_mask_indices=np.arange(count,dtype=np.int64),
        native_bbox_xywh=boxes, native_point_coords_xy=points, native_crop_box_xywh=crops,
        predicted_iou=scores, stability_score=stability, mask_area=areas, query_points=q,
        query_offsets=offsets, query_owner_indices=owners, query_owner_ids=identifiers[owners],
        query_birth_frame_indices=np.full(len(q),row['frame_index'],np.int64))
    return arrays, dict(native_masks=count, queries=len(q), zero_query_seeds=int((np.diff(offsets)==0).sum()),
        duplicate_mask_bytes=count-len(set(hashes)), packed_mask_bytes=packed.nbytes,
        original_seed_ids_retained=True, physical_identities_certified=False)


def observe(anchors, out, report, p, deadline, generate, persist):
    import numpy as np
    require(len(anchors) == 10, 'Exactly ten original fixed anchor images required')
    folder = out/'bank'; require(not folder.exists(), 'Fresh bank folder required'); folder.mkdir(mode=0o700)
    used = 0; report.update(banks=[], amg_attempts=0, amg_calls=0)
    for row in anchors:
        check(deadline); report.update(phase='native_amg', current_clip=row['clip'], current_anchor=row['frame_index'])
        report['amg_attempts'] += 1; persist(); started = time.monotonic()
        records = generate(row['rgb']); report['amg_calls'] += 1
        arrays, audit = bank_arrays(records,row,p)
        require(used+sum(a.nbytes for a in arrays.values())+65536 <= p['maximum_bank_bytes'],
            'All-bank byte resource cap exceeded; never truncate masks/queries')
        name = f"{row['clip']}_anchor_{row['frame_index']:06d}.npz"; path = folder/name
        with path.open('xb') as stream:
            np.savez(stream,**arrays); stream.flush(); os.fsync(stream.fileno()); os.fchmod(stream.fileno(),0o444)
        pin = binding.identity(path,p['maximum_bank_bytes']); used += pin['bytes']
        require(used <= p['maximum_bank_bytes'], 'Actual total bank bytes exceeded fixed cap')
        report['banks'].append(dict(clip=row['clip'],anchor_frame_index=row['frame_index'],frames=row['frames'],
            width=row['width'],height=row['height'],decoded_rgb_sha256=row['rgb_sha256'],file='bank/'+name,
            **pin,**audit,elapsed_seconds=time.monotonic()-started))
        report['bank_bytes'] = used; persist(); del arrays, records
    folder.chmod(0o555)
    require(report['amg_calls'] == report['amg_attempts'] == len(report['banks']) == 10,
        'Every fixed native AMG call and complete returned bank required')


def mounts(code, p):
    """Whitelist code/safe frontend proofs/checkpoint/public JPEGs ONLY."""
    paths = [Path(code).parent,*binding.control_paths(),binding.DEST/'weights'/p['checkpoint']['relative_path'],Path(p['inputs'])]
    require(len(paths) == len(set(paths)), 'Nonduplicate source/public mount whitelist required')
    forbidden = (Path(p['input_report']).parent, ROOT/'data', ROOT/'vendor')
    for path in paths:
        binding.canonical(path); require(path.exists() and ',' not in str(path)
            and '\n' not in str(path) and (not path.is_relative_to(forbidden[0]) or path == Path(p['inputs']))
            and not any(path.is_relative_to(f) for f in forbidden[1:])
            and 'metadata_private' not in path.parts and 'eval_private' not in path.parts,
            'No opaque extraction receipt, label, calibration or challenge mount allowed')
    return paths


def host_preflight(code, revision, p, pin, deadline):
    source = source_binding(code,revision); model = frontend_proof(code,p,live=True)
    safe = input_receipt(Path(p['input_report']),pin,p)
    clips, _ = public_inputs(Path(p['inputs']),safe['public_manifest_identity'],p,deadline)
    require(safe['clips'] == [{k:c[k] for k in ('clip','camera','num_frames')} for c in clips]
        and safe['frames'] == sum(c['num_frames'] for c in clips), 'Extraction/public manifest inventory differs')
    safe.update(source_binding=source, frontend_proof_sha256=json_digest(model), protocol_identity=PROTOCOL_PIN)
    validate_sanitized(safe,pin,p)
    return safe


def _command(args, seconds=15):
    result = subprocess.run(args, capture_output=True, timeout=seconds, check=False)
    require(result.returncode == 0, 'Bounded native control failed')
    return result.stdout.decode().strip()


def cleanup(cidfile, name, revision, image):
    require(cidfile.exists(), 'Owned CID receipt missing')
    path = binding.canonical(cidfile); s = path.lstat()
    require(stat.S_ISREG(s.st_mode) and s.st_nlink == 1 and s.st_uid == 0 and 0 < s.st_size <= 65,
        'Owned regular CID required')
    raw = path.read_bytes(); require(re.fullmatch(b'[0-9a-f]{64}\n?',raw), 'Exact CID bytes required')
    cid = raw.decode().strip()
    found = _command(['docker','ps','-aq','--no-trunc','--filter','id='+cid],5)
    require(found in ('',cid), 'Ambiguous CID query')
    if found:
        actual = _command(['docker','inspect',cid,'--format',
            '{{.Image}}|{{.Name}}|{{index .Config.Labels "world-reward.job"}}|{{index .Config.Labels "world-reward.revision"}}'],5)
        require(actual == image+'|/'+name+'|'+ENTRY+'|'+revision, 'Foreign container cannot be removed')
        _command(['docker','rm','-f',cid],15)
    require(not _command(['docker','ps','-aq','--no-trunc','--filter','name=^/'+name+'$'],5), 'Owned cleanup incomplete')
    path.chmod(0o444)


def dispatch(args, code, revision, p):
    import fcntl
    started = time.monotonic(); deadline = started+p['budget_seconds']
    safe = host_preflight(code,revision,p,args.input_pin,deadline)
    out = binding.canonical(Path(p['output'])); require(not out.exists() and out.parent.is_dir(), 'Fresh fixed output required')
    paths = mounts(code,p); name = 'world-reward-hocap-amg-'+revision[:12]
    require(not _command(['docker','ps','-aq','--no-trunc','--filter','name=^/'+name+'$'],5), 'Container name occupied')
    lock = binding.canonical(ROOT/'jobs/.world-reward-h100.lock'); s = lock.lstat()
    require(stat.S_ISREG(s.st_mode) and s.st_nlink == 1, 'Existing cooperative lock required')
    fd = os.open(lock,os.O_RDONLY|os.O_NOFOLLOW); owned = False; failure = None
    host = dict(stage='hocap_amg_host',status='fail',producer_revision=revision,image_id=p['image_id'],
        budget_seconds=p['budget_seconds'],opaque_input_receipt_mounted=False,private_labels_read=False,
        quality_verified=False,owned_cleanup_verified=False,source_public_rehashed_after=False)
    try:
        require((os.fstat(fd).st_dev,os.fstat(fd).st_ino) == (s.st_dev,s.st_ino), 'Lock inode changed')
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        require(not _command(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],5), 'Other GPU compute present')
        require(host_preflight(code,revision,p,args.input_pin,deadline) == safe, 'Pre-model public/source proof changed')
        out.mkdir(mode=0o700); owned = True
        proof_pin = write_json(out/'sanitized_input_proof.json',safe)
        host['sanitized_input_proof_identity'] = proof_pin
        cmd = ['docker','run','--rm','--name',name,'--cidfile',str(out/'.container.cid'),
            '--label','world-reward.job='+ENTRY,'--label','world-reward.revision='+revision,
            '--gpus','all','--network','none','--user','0:0','--memory','64g','--cpus','4',
            '--read-only','--cap-drop','ALL','--security-opt','no-new-privileges',
            '--tmpfs','/tmp:rw,noexec,nosuid,size=512m','--entrypoint','/usr/bin/env']
        for path in paths: cmd.extend(('--mount',f'type=bind,src={path},dst={path},readonly'))
        cmd.extend(('--mount',f'type=bind,src={out},dst={out}',p['image_id'],'-i',
            'PATH=/opt/conda/bin:/usr/bin:/bin','HOME=/tmp','XDG_CACHE_HOME=/tmp',
            'PYTHONPATH='+str(code/'infra')+':'+str(code/'src'),'WR_ROOT='+str(ROOT),'WR_CODE='+str(code),
            'WR_CODE_REVISION='+revision,'WR_IMAGE_ID='+p['image_id'],'WR_AMG_DEADLINE='+format(deadline,'.17g'),
            'PYTHONDONTWRITEBYTECODE=1','HF_HUB_OFFLINE=1','TRANSFORMERS_OFFLINE=1','WANDB_MODE=disabled',
            'OMP_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4','MKL_NUM_THREADS=4',
            '/opt/conda/bin/python','-B',str(code/'infra/hocap_amg_bank.py'),'--run',
            '--input-report-bytes',str(args.input_pin['bytes']),'--input-report-sha256',args.input_pin['sha256'],
            '--proof-bytes',str(proof_pin['bytes']),'--proof-sha256',proof_pin['sha256']))
        with (out/'native.log').open('xb') as log:
            os.fchmod(log.fileno(),0o400)
            result = subprocess.run(cmd,stdout=log,stderr=log,timeout=max(1,deadline-time.monotonic()),check=False)
        host['native_exit_status'] = result.returncode; require(result.returncode == 0, 'Native AMG bank failed')
        receipt = binding.strict_json((out/'report.json').read_bytes())
        require(receipt.get('status') == 'pass' and receipt.get('phase') == 'complete'
            and receipt.get('producer_revision') == revision and receipt.get('sanitized_input_proof_identity') == proof_pin
            and receipt.get('all_original_jpegs_decoded_before_model') is True
            and receipt.get('amg_calls') == 10 and receipt.get('original_source_public_models_rehashed_after') is True,
            'Complete native sealed bank receipt required')
        host['native_report_identity'] = binding.identity(out/'report.json',p['maximum_input_report_bytes'])
    except Exception as error:
        failure = error; host['error_type'] = type(error).__name__
    finally:
        signal.alarm(p['cleanup_grace_seconds'])
        try:
            if owned:
                cleanup(out/'.container.cid',name,revision,p['image_id']); host['owned_cleanup_verified'] = True
                require(not _command(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],5), 'GPU cleanup incomplete')
                require(binding.identity(out/'sanitized_input_proof.json',p['maximum_input_report_bytes']) == proof_pin,
                    'Sanitized source proof changed')
            after = host_preflight(code,revision,p,args.input_pin,time.monotonic()+p['cleanup_grace_seconds'])
            require(after == safe and (lock.lstat().st_dev,lock.lstat().st_ino) == (s.st_dev,s.st_ino),
                'Original post-run source/public/model/lock changed')
            host['source_public_rehashed_after'] = True
        except Exception as error:
            failure = failure or error; host['post_error_type'] = type(error).__name__
        finally:
            os.close(fd); signal.alarm(0)
            if owned:
                host.update(status='fail' if failure else 'pass',elapsed_seconds=time.monotonic()-started)
                write_json(out/'host.json',host)
                if not failure: (out/'native.log').unlink()
    require(failure is None, 'AMG wrapper failed; immutable owned receipts retained')
    return host


def native(args, code, revision, p):
    started = time.monotonic(); deadline = float(os.environ['WR_AMG_DEADLINE']); check(deadline)
    out = binding.canonical(Path(p['output'])); pin = args.proof_pin
    report = dict(schema='world_reward.hocap_amg_bank.v1',stage='hocap_native_all_mask_bank',
        status='fail',phase='public_authentication',producer_revision=revision,image_id=p['image_id'],
        sanitized_input_proof_identity=pin,private_labels_read=False,calibration_read=False,
        challenge_inputs_used=False,ground_truth_used=False,oracle_modes=[],network='none',
        opaque_metadata_read=False,training_overlap_verified=False,quality_verified=False,adoption=False,
        boots_executed=False,association_executed=False,all_native_returned_masks=True,
        model_loads=0,rgb_decodes=0,amg_calls=0,original_source_public_models_rehashed_after=False)
    source = model = safe = clips = installed = None; failure = None
    def persist(): check(deadline)
    try:
        require(os.environ.get('WR_IMAGE_ID') == p['image_id'], 'Actual original frontend image required')
        require({f.name for f in out.iterdir()} == {'sanitized_input_proof.json','.container.cid','native.log'},
            'Fresh child output allows only exact sanitized proof and owned CID')
        require(not Path(p['input_report']).exists() and not (Path(p['input_report']).parent/'metadata_private').exists()
            and not (ROOT/'data').exists() and not (ROOT/'vendor').exists(), 'Private/challenge mounts forbidden')
        safe = binding.pinned(out/'sanitized_input_proof.json',pin,p['maximum_input_report_bytes'])
        validate_sanitized(safe,args.input_pin,p)
        source = source_binding(code,revision); model = frontend_proof(code,p)
        require(safe.get('source_binding') == source and safe.get('frontend_proof_sha256') == json_digest(model)
            and safe.get('protocol_identity') == PROTOCOL_PIN, 'Host/source/native model proof differs')
        clips,_ = public_inputs(Path(p['inputs']),safe['public_manifest_identity'],p,deadline)
        anchors = decode_anchors(clips,p,deadline,report)
        require(report['rgb_decodes'] == safe['frames'] and safe['clips'] == [{k:c[k] for k in ('clip','camera','num_frames')} for c in clips],
            'Original decode count/extraction inventory differs')
        report.update(phase='native_model_load',public_manifest_identity=safe['public_manifest_identity'],
            source_binding_sha256=json_digest(source),frontend_proof_sha256=json_digest(model),
            inference_precision='native_BF16_autocast_FP32_model',amg_parameters=p['amg'],
            build_apply_postprocessing=p['build_apply_postprocessing'],model_asset=p['checkpoint'],
            installed_config=p['installed_config'])
        import torch
        require(torch.cuda.is_available() and 'H100' in torch.cuda.get_device_name(), 'Actual H100 CUDA required')
        torch.manual_seed(0); torch.cuda.manual_seed_all(0)
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        installed = binding.installed_sam2(model)
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        with torch.inference_mode():
            sam = build_sam2(p['installed_config']['path'],str(binding.DEST/'weights'/p['checkpoint']['relative_path']),
                device='cuda',mode='eval',apply_postprocessing=p['build_apply_postprocessing'])
            report['model_loads'] += 1
            require(not sam.training and not hasattr(sam.image_encoder.forward,'_torchdynamo_orig_callable'),
                'Uncompiled native evaluation model required')
            require(sam.sam_mask_decoder.dynamic_multimask_via_stability is True
                and sam.sam_mask_decoder.dynamic_multimask_stability_delta == .05
                and sam.sam_mask_decoder.dynamic_multimask_stability_thresh == .98,
                'Original native dynamic multimask policy required')
            generator = SAM2AutomaticMaskGenerator(sam,**p['amg'])
            require(generator.predictor._transforms.max_hole_area == generator.predictor._transforms.max_sprinkle_area == 0,
                'AMG native hole/sprinkle0 required; no fallback branch')
            def generate(rgb):
                check(deadline)
                with warnings.catch_warnings(),torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                    warnings.simplefilter('error'); result = generator.generate(rgb)
                torch.cuda.synchronize(); return result
            observe(anchors,out,report,p,deadline,generate,persist)
        torch.cuda.synchronize(); del anchors,generator,sam
        require(report['model_loads'] == 1, 'Exactly one native model load required')
    except Exception as error:
        failure = error; report['error_type'] = type(error).__name__
    finally:
        try:
            require(source is not None and source_binding(code,revision) == source and protocol(code) == p
                and model is not None and frontend_proof(code,p) == model
                and safe is not None and binding.pinned(out/'sanitized_input_proof.json',pin,p['maximum_input_report_bytes']) == safe,
                'Original source/proof/checkpoint changed')
            public_inputs(Path(p['inputs']),safe['public_manifest_identity'],p,deadline)
            if installed is not None: require(binding.installed_sam2(model) == installed, 'Installed SAM2 sources/config/extension changed')
            for row in report.get('banks',[]): require(binding.identity(out/row['file'],p['maximum_bank_bytes'])
                == {k:row[k] for k in ('bytes','sha256')}, 'Original bank changed')
            report['original_source_public_models_rehashed_after'] = True; check(deadline)
        except Exception as error:
            failure = failure or error; report['post_error_type'] = type(error).__name__
        report.update(status='fail' if failure else 'pass',phase=report['phase'] if failure else 'complete',
            elapsed_seconds=time.monotonic()-started,budget_seconds=p['budget_seconds'])
        write_json(out/'report.json',report)
    require(failure is None, 'Native AMG bank failed; inspect immutable receipt')
    return report


class SingleUseParser(argparse.ArgumentParser):
    def parse_args(self, args=None, namespace=None):
        args = list(sys.argv[1:] if args is None else args); seen = set()
        for word in args:
            if word.startswith('--'):
                option = word.partition('=')[0]
                if option in seen: self.error('Duplicate option: '+option)
                seen.add(option)
        return super().parse_args(args,namespace)


def parser():
    parser = SingleUseParser(allow_abbrev=False)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--dispatch',action='store_true');mode.add_argument('--run',action='store_true')
    parser.add_argument('--input-report-bytes',type=int,required=True);parser.add_argument('--input-report-sha256',required=True)
    parser.add_argument('--proof-bytes',type=int);parser.add_argument('--proof-sha256')
    return parser


def main(argv=None):
    args = parser().parse_args(argv); code = Path(os.environ['WR_CODE']); revision = os.environ['WR_CODE_REVISION']
    require(sys.platform == 'linux' and os.geteuid() == 0 and Path(os.environ['WR_ROOT']) == ROOT
        and re.fullmatch('[0-9a-f]{40}',revision), 'Owned Linux root immutable dispatch required')
    p = protocol(code); args.input_pin = dict(bytes=args.input_report_bytes,sha256=args.input_report_sha256)
    binding.validate_file_pins({'receipt':args.input_pin},{'receipt'},p['maximum_input_report_bytes'])
    if args.dispatch:
        require(args.proof_bytes is args.proof_sha256 is None and os.uname().nodename == 'world-reward-ncc-h100-02',
            'Real VM02 host dispatch required, no supplied sanitized proof')
    else:
        args.proof_pin = dict(bytes=args.proof_bytes,sha256=args.proof_sha256)
        binding.validate_file_pins({'proof':args.proof_pin},{'proof'},p['maximum_input_report_bytes'])
    def expired(*_): raise TimeoutError('Inclusive AMG bank/cleanup bound exhausted')
    signal.signal(signal.SIGALRM,expired);signal.signal(signal.SIGTERM,expired)
    signal.alarm(p['budget_seconds'] if args.dispatch else max(1,math.ceil(float(os.environ['WR_AMG_DEADLINE'])-time.monotonic())))
    try: result = dispatch(args,code,revision,p) if args.dispatch else native(args,code,revision,p)
    finally: signal.alarm(0)
    import json
    print(json.dumps(dict(stage=result['stage'],status=result['status'],quality_verified=False)),flush=True)


if __name__ == '__main__':
    try: main()
    except Exception as error:
        import json
        print(json.dumps(dict(stage='hocap_amg_bank',status='fail',error_type=type(error).__name__)),flush=True)
        raise SystemExit(1) from None
