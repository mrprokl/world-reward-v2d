"""Four NEW public RGB anchors: unchanged full Body and MoGe2 predictions.

Engineering only. No renderer/truth import, alignment, temporal fitting, contact
claim or metric-accuracy gate. Invalid MoGe pixels are preserved, never filled.
"""
from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import signal
import stat
import sys
import time

# The isolated host preflight may import only its dispatched sibling sources.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bridge_frontend_bindings as binding

ENTRY = 'run_bridge_rgb_anchor_infer'
BASE = 'validation/bridge_rgb_anchor_v1'
OUTPUT = BASE + '/observations_anchor_v1'
MASK_PINS = 'configs/bridge_rgb_anchor_mask_pins.json'
WIDTH, HEIGHT, FOCAL, BUDGET = 640, 480, 800., 180
BODY_REL = 'weights/cari4d/sam3d_body/checkpoints/sam-3d-body-dinov3'
DINO_REL = 'weights/cari4d/sam3d_body/torch_home/hub/facebookresearch_dinov3_main'
BODY_SOURCE = 'vendor/video_to_data/reconstruction/modules/v2d_sam3d_body/lib/sam_3d_body'
BODY_SHA = 'b5a2f9d305dd02626b967aa2e86021fba07065df66ce7a7e00ffb9664f150abf'
MHR_SHA = '352e271a6c42729c68554ceaea0c955e866970160c31e35506d782dc0f7377bc'
MOGE_REV = 'b135031bae30b5ac2ae141a0e68717795ce38340'
MOGE_SHA = '280741fd09bc3f403ccff9967784c2a391b52d2c0742ae3efdb21d9f90cc1a01'
MOGE_BYTES = 1323815904
XET_SHA = '9f4c4857a8203605fd29a80f0e81e9ed52fc1654c1e657d437ab29b73d8db37c'
GEOMETRY_SHA = '2f8d5de7d671af16d25fe13c555fd73855d8447bc086bb23aa42e859b303fb05'
require = binding.require


def moge_path(root):
    return root/'weights/cari4d/hf_home/hub/blobs'/XET_SHA[:2]/XET_SHA


def moge_asset(root):
    """Actual canonical Xet file, content SHA distinct from its storage name."""
    receipt_path = root/'results/weights-acquisition.json'
    receipt = binding.identity(receipt_path,2_000_000)
    acquisition = binding.strict_json(receipt_path.read_bytes())
    rows = [r for r in acquisition['assets'] if r.get('repo_id') == 'Ruicheng/moge-2-vitl-normal']
    require(len(rows) == 1 and rows[0].get('revision') == MOGE_REV
        and rows[0].get('cache_dir') == str(root/'weights/cari4d/hf_home/hub'), 'Original exact MoGe2 acquisition revision/cache required')
    path = moge_path(root); asset = binding.identity(path)
    require(asset == dict(bytes=MOGE_BYTES,sha256=MOGE_SHA), 'Independent canonical MoGe2 content bytes differ')
    return path, receipt, asset


def host_moge_chain(root):
    """Inspect only the two known relative links; no cache scan or alias creation."""
    import posixpath
    cache = root/'weights/cari4d/hf_home/hub'
    snapshot = cache/'models--Ruicheng--moge-2-vitl-normal/snapshots'/MOGE_REV/'model.pt'
    repo_blob = cache/'models--Ruicheng--moge-2-vitl-normal/blobs'/MOGE_SHA
    xet = moge_path(root); current = snapshot; result = {}
    for _ in range(3):
        binding.canonical(current.parent); s = current.lstat()
        if stat.S_ISLNK(s.st_mode):
            target = os.readlink(current)
            require(target and not target.startswith('/') and '\\' not in target and '\0' not in target, 'Original MoGe relative link required')
            destination = Path(posixpath.normpath(str(current.parent/target)))
            require((current == snapshot and destination in (repo_blob,xet)) or (current == repo_blob and destination == xet), 'Original MoGe link escapes independently known graph')
            after=current.lstat()
            require(all(getattr(s,k)==getattr(after,k) for k in ('st_dev','st_ino','st_mode','st_size','st_mtime_ns','st_ctime_ns')), 'Original MoGe link changed while read')
            result[str(current)] = dict(link=target,target=str(destination)); current = destination
        else:
            require(current == xet, 'Actual independently known canonical Xet file required')
            result[str(current)] = binding.identity(current)
            require(snapshot.resolve()==xet, 'Actual original MoGe snapshot resolution differs')
            return result
    raise ValueError('MoGe link graph exceeds two links')


def bbox(rgb, mask):
    import numpy as np
    require(rgb.shape == (HEIGHT, WIDTH, 3) and rgb.dtype == np.uint8
        and mask.shape == (HEIGHT, WIDTH) and mask.dtype == np.uint8
        and np.isin(mask, [0, 255]).all(), 'Original RGB/binary automatic person mask required')
    y, x = np.where(mask > 0)
    require(len(x) >= 20, 'Automatic person mask absent; no bbox fallback')
    box = np.array([x.min(), y.min(), x.max()+1, y.max()+1], np.float32)
    require((box[2:]-box[:2]).min() > 1, 'Automatic mask bbox is degenerate')
    return box, (mask > 0).astype(np.uint8)[..., None]


def named_metadata(names, controls, parents, transform):
    import numpy as np
    require(type(names) is list and len(names) == 127 and type(controls) is list and len(controls) == 249
        and all(type(n) is str and n for n in names+controls)
        and len(set(names)) == 127 and len(set(controls)) == 249, 'Actual unique MHR 127/249 names required')
    p, m = np.asarray(parents), np.asarray(transform)
    require(p.shape == (127,) and p.dtype.kind in 'iu' and ((p >= -1) & (p < 127)).all()
        and (p == -1).sum() == 1, 'Actual one-root MHR parent tree required')
    for index in range(127):
        visited = set()
        while index != -1:
            require(index not in visited, 'MHR parent cycle forbidden')
            visited.add(index); index = int(p[index])
    require(m.shape == (889, 249) and m.dtype.kind == 'f' and np.isfinite(m).all(), 'Actual finite MHR 889x249 transform required')
    return dict(joint_names=names, parameter_names=controls, joint_parents=p.tolist(),
        parameter_transform_sha256=hashlib.sha256(m.tobytes()).hexdigest(),
        parameter_transform_dtype=str(m.dtype), parameter_transform_shape=list(m.shape),
        metadata_source='same_SHA_bound_bundled_MHR_asset_load', hand_anatomy_accuracy_verified=False)


def load_body(root, torch, selected_manifest):
    """One checkpoint load; preserve official strict state and local DINO loader."""
    import numpy as np
    import body_smoke as body
    source = body._source_identity(root, selected_manifest=selected_manifest)
    directory, assets = body._body_assets(root)
    contract = binding.selected.load_contract(root,selected_manifest)
    for name,actual in assets.items():
        wanted = contract['entries'][BODY_REL+'/'+name]
        require(wanted.get('type') == 'file' and {k:wanted.get(k) for k in ('bytes','sha256')} == actual,
            'Every Body model/config/license asset must match authenticated selected manifest')
    require(assets['model.ckpt'] == dict(bytes=2109129346, sha256=BODY_SHA)
        and assets['assets/mhr_model.pt'] == dict(bytes=696110248, sha256=MHR_SHA), 'Exact independent Body/MHR bytes required')
    original_hub, calls = body._install_local_dinov3_loader(torch, root/DINO_REL, selected_manifest=selected_manifest)
    loading = {}
    try:
        import sam_3d_body
        from sam_3d_body import build_models
        require(Path(sam_3d_body.__file__).resolve() == Path('/workspace/v2d_sam3d_body/lib/sam_3d_body/__init__.py'), 'Original installed Body package required')
        asset = torch.jit.load(str(directory/'assets/mhr_model.pt'), map_location='cpu')
        reference = named_metadata(asset.get_joint_names(), asset.get_parameter_names(),
            asset.character_torch.skeleton.joint_parents.detach().cpu().numpy(),
            asset.get_parameter_transform().detach().cpu().numpy())
        asset_state = asset.state_dict()
        original_loader = build_models.load_state_dict
        def strict_loader(module, state_dict, strict=False, logger=None):
            loading.update(body._load_checkpoint_with_asset_buffers(module, state_dict, original_loader, torch,
                explicit_asset_state=asset_state))
            prepare = module.backbone.encoder.prepare_tokens_with_masks
            def unmasked(x, masks=None):
                require(masks is None, 'Retained zero DINO token permitted only for unmasked RGB')
                return prepare(x, masks=None)
            module.backbone.encoder.prepare_tokens_with_masks = unmasked
        build_models.load_state_dict = strict_loader
        try:
            model, config = sam_3d_body.load_sam_3d_body(checkpoint_path=str(directory/'model.ckpt'),
                device='cuda', mhr_path=str(directory/'assets/mhr_model.pt'))
        finally: build_models.load_state_dict = original_loader
    finally: torch.hub.load = original_hub
    require(len(calls) == 1 and config.MODEL.BACKBONE.TYPE == calls[0], 'Exactly one local DINO backbone load required')
    require(loading.get('mode') == 'strict_network_and_head_state_with_explicit_asset_buffer_retention'
        and loading.get('parameter_tensors_loaded') == 1101 and loading.get('unexpected_keys') == []
        and len(loading.get('retained_mhr_asset_buffer_names', [])) == 113, 'Original strict network/asset retention ABI required')
    names = model.head_pose.mhr.get_joint_names()
    require(names == reference['joint_names'], 'Loaded native head joint names differ from same exact asset')
    character = model.head_pose.mhr.character_torch
    actual = named_metadata(list(character.skeleton.joint_names), list(character.parameter_transform.parameter_names),
        character.skeleton.joint_parents.detach().cpu().numpy(),
        character.parameter_transform.parameter_transform.detach().cpu().numpy())
    require(actual == reference, 'Loaded native head names/parents/transform differ from bundled asset')
    del asset, asset_state
    model.eval(); model.requires_grad_(False)
    estimator = sam_3d_body.SAM3DBodyEstimator(model, config)
    faces = np.asarray(estimator.faces)
    require(faces.shape == (36874, 3) and faces.dtype.kind in 'iu' and faces.min() >= 0 and faces.max() < 18439
        and not np.any((faces[:,0] == faces[:,1]) | (faces[:,1] == faces[:,2]) | (faces[:,0] == faces[:,2])), 'Native face topology ABI required')
    return model, estimator, faces.astype(np.int64), dict(body_assets=assets, inference_source_identity=source,
        checkpoint_loading=loading, local_backbone_loads=1, named_metadata=reference,
        body_revision=body.BODY_REVISION, dinov3_revision=body.DINOV3_REVISION)


def body_projection(prediction, values):
    import numpy as np
    cpu = lambda x: x.detach().float().cpu().numpy()
    require(np.isclose(float(values['focal_length']), FOCAL, rtol=1e-6, atol=1e-4), 'RGB-size-only focal800 required')
    points = cpu(prediction['pred_keypoints_3d']) + values['pred_cam_t']
    require(points.shape == (70, 3) and np.isfinite(points).all() and (points[:,2] > 0).all(), 'Fresh positive camera keypoints required')
    pixels = FOCAL*points[:,:2]/points[:,2:] + [WIDTH/2, HEIGHT/2]
    error = float(np.linalg.norm(pixels-cpu(prediction['pred_keypoints_2d']), axis=1).max())
    require(np.isfinite(error) and error <= .05, 'Native model image-keypoint projection failed')
    return dict(max_projection_error_px=error, focal_length=FOCAL, independent_RGB_hand_support_verified=False)


def depth_contract(depth, points, valid, normalized, person, object_mask):
    import numpy as np
    from world_reward.pointmap import validate_camera_pointmap
    require(depth.shape == (HEIGHT, WIDTH) and depth.dtype == np.float32 and points.shape == (HEIGHT, WIDTH, 3)
        and points.dtype == np.float32 and valid.shape == (HEIGHT, WIDTH) and valid.dtype == np.bool_, 'Original MoGe float32/boolean grids required')
    require(person.shape == valid.shape and object_mask.shape == valid.shape and person.dtype == np.bool_
        and object_mask.dtype == np.bool_, 'Original automatic support grids required')
    K = np.array([[FOCAL,0,WIDTH/2],[0,FOCAL,HEIGHT/2],[0,0,1]], np.float64)
    checks = validate_camera_pointmap(depth, points, valid, normalized, K)
    return K, checks


def checked_depth(torch, module, model, tensor, diagnostics, persist, clip):
    """Observe original native nearest64 support before rejecting its fallback."""
    original = module.recover_focal_shift; before = len(diagnostics)
    def checked(points, mask=None, focal=None, downsample_size=(64,64)):
        require(downsample_size == (64,64) and torch.is_tensor(mask) and mask.dtype == torch.bool
            and tuple(mask.shape) == (1,HEIGHT,WIDTH) and tuple(points.shape) == (1,HEIGHT,WIDTH,3), 'Original native focal solver ABI required')
        sampled = torch.nn.functional.interpolate(mask.float().unsqueeze(1), (64,64), mode='nearest').squeeze(1) > 0
        row = dict(clip_id=clip, nearest64_valid_pixels=int(sampled.sum().item()),
            focal_prior_supplied=focal is not None, original_returned=False)
        diagnostics.append(row); persist()
        require(row['nearest64_valid_pixels'] >= 2, 'Native MoGe focal solver lacks two sampled valid pixels; no fallback')
        result = original(points, mask, focal=focal, downsample_size=downsample_size)
        row['original_returned'] = True; persist(); return result
    module.recover_focal_shift = checked
    try:
        import math
        fov = math.degrees(2*math.atan(WIDTH/(2*FOCAL)))
        with torch.inference_mode(): result = model.infer(tensor[None], fov_x=fov, apply_mask=False)
        require(len(diagnostics) == before+1, 'Exactly one native focal solver return per MoGe call required')
        return result
    finally: module.recover_focal_shift = original


def read_frame(record, mask_row, root):
    import numpy as np
    from PIL import Image
    with Image.open(record['path']) as png:
        require(png.format == 'PNG' and png.mode == 'RGB' and png.size == (WIDTH,HEIGHT), 'Original public RGB PNG required')
        rgb = np.asarray(png).copy()
    masks = []
    for label in ('person','object'):
        part = mask_row[label]
        with Image.open(root/BASE/'automatic_masks'/part['mask_file']) as png:
            require(png.format == 'PNG' and png.mode == 'L' and png.size == (WIDTH,HEIGHT), 'Original automatic L PNG required')
            value = np.asarray(png).copy()
        require(np.isin(value,[0,255]).all() and int(np.count_nonzero(value)) == part['mask_pixels'], 'Frozen automatic mask area/values differ')
        masks.append(value)
    return rgb, *masks


def controls(root, code, live=False):
    proof = binding.authenticate(root, code, ENTRY, live=live)
    records, public = binding.public_inputs(root, code/binding.INPUT_PINS)
    masks, mask_files = binding.load_masks(root, code/MASK_PINS, records)
    receipt = binding.DEST/'results/weights-acquisition.json'
    row = proof['selected_contract']['entries']['results/weights-acquisition.json']
    require(binding.identity(receipt,2_000_000) == {k:row[k] for k in ('bytes','sha256')}, 'Exact exported safe acquisition receipt required')
    if live:host_moge_chain(root)
    return proof, records, masks, public | mask_files


def preflight_summary(root,code):
    proof,_,_,frozen=controls(root,code,live=True)
    return dict(image_id=proof['child_image']['Id'],source_binding=proof['source_binding'],
        public_files={str(p):v for p,v in frozen.items()},original_MoGe_link_graph=host_moge_chain(root))


def readonly_mounts(root, code):
    """Exact public code/proofs, selected model/source paths, never parent caches."""
    paths = [(p,p) for p in (code.parent,*binding.control_paths(),binding.DEST/BODY_SOURCE,binding.DEST/DINO_REL)]
    paths += [(binding.DEST/DINO_REL,root/DINO_REL), (binding.DEST/'results/weights-acquisition.json',root/'results/weights-acquisition.json')]
    for name in ('model.ckpt','model_config.yaml','assets/mhr_model.pt','LICENSE'):
        source = binding.DEST/BODY_REL/name
        paths += [(source,source),(source,root/BODY_REL/name)]
    paths += [(moge_path(root),moge_path(root))]
    paths += [(root/BASE/name,root/BASE/name) for name in ('inputs','automatic_masks')]
    for source, destination in paths:
        binding.canonical(source); binding.canonical(destination.parent)
        require(source.exists(), 'Required narrow readonly inference mount absent')
    return list(dict.fromkeys((str(a),str(b)) for a,b in paths))


def run(root, code, out, report, persist):
    import numpy as np
    import body_smoke as body
    import hand_synthetic_infer as hand
    import object_synthetic_observations as depth_helper
    proof, records, masks, frozen = controls(root, code)
    report.update(source_binding=proof['source_binding'], build_report_identity=proof['build_report_identity'],
        kernel_report_identity=proof['kernel_report_identity'], image_id=proof['child_image']['Id'],
        selected_manifest_identity=proof['selected_contract']['manifest_identity'],
        extraction_receipt_identity=proof['selected_contract']['extraction_receipt_identity'],
        public_input_pins={str(p):v for p,v in frozen.items()}, phase='model_load')
    original_chain=binding.strict_json(os.environ['WR_MOGE_ORIGINAL_CHAIN'])
    require(type(original_chain) is dict and str(moge_path(root)) in original_chain
        and original_chain[str(moge_path(root))] == dict(bytes=MOGE_BYTES,sha256=MOGE_SHA), 'Launcher-bound actual original MoGe graph required')
    report.update(original_MoGe_link_graph=original_chain,MoGe_load_path=str(moge_path(root)),
        MoGe_load_mode='canonical_original_content_blob_no_HF_alias_or_cache_mount')
    persist()
    require('torch' not in sys.modules, 'CUDA environment must precede Torch import')
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import torch
    import moge
    from moge.model import v2 as module
    from moge.model.v2 import MoGeModel
    from moge.utils import geometry_torch
    require(torch.cuda.is_available() and str(torch.__version__) == '2.5.1+cu124' and torch.version.cuda == '12.4', 'Actual pinned CUDA runtime required')
    torch.manual_seed(0); torch.cuda.manual_seed_all(0); torch.set_num_threads(4)
    torch.use_deterministic_algorithms(False,warn_only=False)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False; torch.backends.cudnn.benchmark=False
    model, estimator, faces, body_meta = load_body(root,torch,binding.MANIFEST)
    report.update(body_model=body_meta, body_model_loads=1, torch=str(torch.__version__), CUDA=torch.version.cuda)
    path, acquisition, asset = moge_asset(root)
    distribution = metadata.distribution('moge'); direct = distribution.read_text('direct_url.json')
    source = depth_helper.installed_source(Path(moge.__file__).parent, json.loads(direct) if direct else {})
    require(depth_helper.identity(Path(geometry_torch.__file__))['sha256'] == GEOMETRY_SHA, 'Original MoGe focal solver source required')
    depth_model = MoGeModel.from_pretrained(str(path)).cuda().eval()
    report.update(MoGe_model_loads=1, MoGe_model_asset=asset, MoGe_acquisition_receipt=acquisition, MoGe_source=source, phase='four_original_predictions')
    persist()
    for record, mask_row in zip(records,masks):
        require(not torch.are_deterministic_algorithms_enabled() and not torch.is_deterministic_algorithms_warn_only_enabled()
            and not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32, 'Explicit ordinary CUDA/noTF32 execution contract changed')
        rgb,person,object_mask = read_frame(record,mask_row,root); box,prompt = bbox(rgb,person)
        row = dict(clip_id=record['clip_id'],frame_id=0,rgb_sha256=record['sha256'],
            decoded_RGB_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),person_pixels=int(np.count_nonzero(person)),
            object_pixels=int(np.count_nonzero(object_mask)),bbox_xyxy=box.tolist(),completed=False)
        report['frames'].append(row);report['body_attempts']+=1;persist()
        with torch.inference_mode(): predictions = estimator.process_one_image(img=rgb,bboxes=box[None],masks=prompt,cam_int=None,inference_type='full')
        report['body_returns']+=1;persist()
        require(type(predictions) is list and len(predictions)==1, 'Exactly one automatic full Body actor required')
        values,parity = hand.decode_prediction(torch,model,predictions[0],'full');report['native_decode_returns']+=1
        row.update(native_decode_errors=parity,projection=body_projection(predictions[0],values));persist()
        tensor = torch.from_numpy(rgb).cuda().permute(2,0,1).float()/255
        report['MoGe_attempts']+=1;persist()
        predicted=checked_depth(torch,module,depth_model,tensor,report['native_focal_solver'],persist,record['clip_id'])
        torch.cuda.synchronize();report['MoGe_returns']+=1
        depth,points,valid,normalized = [predicted[k][0].detach().cpu().numpy() for k in ('depth','points','mask','intrinsics')]
        row.update(valid_depth_pixels=int(np.count_nonzero(valid)),person_valid_depth_pixels=int(np.count_nonzero((person>0)&valid)),
            object_valid_depth_pixels=int(np.count_nonzero((object_mask>0)&valid)))
        persist()  # Support diagnostics precede every numerical pointmap gate.
        K,checks=depth_contract(depth,points,valid,normalized,person>0,object_mask>0);row['pointmap_checks']=checks
        arrays = dict(values);arrays['joint_global_rotations_native']=arrays.pop('joint_global_rotations')
        arrays.update(faces=faces,depth=depth,points=points,depth_validity=valid,intrinsics_normalized=normalized,camera_K=K,
            person_mask=person>0,object_mask=object_mask>0,clip_id=np.array(record['clip_id'],np.int64),frame_id=np.array(0,np.int64))
        target=out/(Path(record['file']).stem+'.npz')
        with target.open('xb') as stream:np.savez_compressed(stream,**arrays)
        target.chmod(0o400);row.update(output=dict(file=target.name,**binding.identity(target)),completed=True);persist()
        del predictions,predicted,tensor
    binding.recheck(frozen)
    require(binding.authenticate(root,code,ENTRY)==proof, 'Inference source/proofs changed during predictions')
    require(body._source_identity(root,selected_manifest=binding.MANIFEST)==body_meta['inference_source_identity']
        and body._body_assets(root)[1]==body_meta['body_assets'], 'Selected Body source/model bytes changed')
    require(moge_asset(root)[1:]==(acquisition,asset), 'Actual MoGe model bytes changed')
    require(depth_helper.installed_source(Path(moge.__file__).parent,json.loads(direct) if direct else {})==source
        and depth_helper.identity(Path(geometry_torch.__file__))['sha256']==GEOMETRY_SHA, 'Actual MoGe source changed')
    require({p.name for p in out.iterdir()} == {'report.json',*[r['output']['file'] for r in report['frames']]}, 'Exact four prediction artifacts and report required')
    for row in report['frames']:
        require(binding.identity(out/row['output']['file'])=={k:row['output'][k] for k in ('bytes','sha256')}, 'Frozen prediction artifact changed')
    require(all(report[k]==4 for k in ('body_attempts','body_returns','native_decode_returns','MoGe_attempts','MoGe_returns'))
        and len(report['native_focal_solver'])==4 and all(r['original_returned'] for r in report['native_focal_solver']), 'Exactly four original completed model observations required')
    report.update(status='pass',phase='complete',sources_and_inputs_rechecked=True)


def persist_run(out,report,operation):
    """Own receipt sealed on success AND failure; never overwrite another run."""
    started=time.monotonic();path=out/'report.json'
    with path.open('x') as stream:
        def persist():
            report['elapsed_seconds']=time.monotonic()-started;stream.seek(0);json.dump(report,stream,allow_nan=False)
            stream.write('\n');stream.truncate();stream.flush();os.fsync(stream.fileno())
        def expired(*_):raise TimeoutError('Four-anchor model budget exceeded')
        signal.signal(signal.SIGALRM,expired);signal.signal(signal.SIGTERM,expired);signal.alarm(BUDGET)
        try:persist();operation(persist)
        except Exception as error:
            import re
            detail=str(error)
            if re.search(r'hf_[A-Za-z0-9]{20,}|Bearer\s+\S+|AccountKey=|(?i:password|api_key|authorization)\s*[=:]',detail):detail='Sensitive upstream diagnostic omitted'
            report.update(status='fail',error_type=type(error).__name__,error=detail[:250]);raise
        finally:
            signal.alarm(0)
            try:persist()
            finally:path.chmod(0o400)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    modes=parser.add_mutually_exclusive_group(required=True)
    for mode in ('preflight','mounts','verify','run'):modes.add_argument('--'+mode,action='store_true')
    args=parser.parse_args(argv);root=binding.ROOT;code=Path(os.environ['WR_CODE'])
    require(Path(__file__).resolve()==code/'infra/bridge_rgb_anchor_infer.py', 'Actual immutable inference source required')
    if not args.run:
        if args.mounts:
            controls(root,code,live=True)
            for source,destination in readonly_mounts(root,code):print(source+'\t'+destination)
        else:print(json.dumps(preflight_summary(root,code),sort_keys=True,separators=(',',':')))
        return
    require(sys.platform=='linux' and os.geteuid()==0 and {p.name for p in Path('/sys/class/net').iterdir()}=={'lo'}, 'Root-owned offline Linux GPU container required')
    require(os.environ.get('WR_IMAGE_ID')==binding.IMAGE, 'Authenticated new runtime image required')
    out=binding.canonical(root/OUTPUT);require(out.is_dir() and not any(out.iterdir()), 'Fresh exclusively reserved observation output required')
    report=dict(schema='world_reward.bridge_rgb_anchor_observations.v1',stage='bridge_rgb_anchor_model_observations',status='fail',phase='integrity',
        producer_revision=code.parent.parent.name,script_sha256=binding.identity(Path(__file__),200000)['sha256'],
        budget_seconds=BUDGET,frames=[],body_model_loads=0,MoGe_model_loads=0,body_attempts=0,body_returns=0,
        native_decode_returns=0,MoGe_attempts=0,MoGe_returns=0,native_focal_solver=[],network='none',
        ground_truth_used=False,challenge_inputs_used=False,hand_labeled_test=False,oracle_modes=[],
        geometry_frame='opencv_x_right_y_down_z_forward',joint_rotation_frame='native_MHR_local_axes_in_native_global_frame_NOT_camera',
        focal_prior='RGB_hypot_640_480_800_not_source_calibration',model_parameters_modified=False,
        human_identity_clip_constant=False,metric_scale_accuracy_verified=False,independent_RGB_hand_support_verified=False,
        predicted_silhouette_QA_performed=False,contact_verified=False,bridge_adoption_performed=False,
        original_grounding_image_parity_claimed=False,challenge_performance_verified=False,submission_eligible=False,
        deterministic_algorithms=False,TF32=False,bit_exact_replay_claimed=False)
    persist_run(out,report,lambda persist:run(root,code,out,report,persist))
    print(json.dumps(dict(stage=report['stage'],status=report['status'],frames=len(report['frames']),elapsed_seconds=report['elapsed_seconds'],submission_eligible=False)))


if __name__=='__main__':main()
