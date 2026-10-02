"""Tiny public-only contracts/stubs; no model, private truth or CUDA reads."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


@pytest.fixture
def infer(monkeypatch):
    infra=Path(__file__).resolve().parents[1]/'infra';monkeypatch.syspath_prepend(str(infra))
    spec=importlib.util.spec_from_file_location('test_hand_infer',infra/'hand_synthetic_infer.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def public_fixture(infer,root):
    base=root/'validation/hands_rgb_v1';inputs=base/'inputs';masks=base/'automatic_masks'
    inputs.mkdir(parents=True);masks.mkdir()
    images=[];records=[]
    for index in range(6):
        name=f'case_{index:03d}.png';rgb=inputs/name;mask=masks/name
        rgb.write_bytes(b'ownfakeRGB'+bytes([index]));mask.write_bytes(b'ownfakemask'+bytes([index]))
        images.append({'file':name,'sha256':infer.sha256(rgb),'width':1024,'height':768})
        records.append({'frame_index':index,'file':name,'mask_file':name,'rgb_sha256':infer.sha256(rgb),
                        'mask_sha256':infer.sha256(mask),'width':1024,'height':768})
    manifest={'schema':'world-reward-hands-rgb-inputs-v1','images':images}
    path=inputs/'manifest.json';path.write_text(json.dumps(manifest))
    report={'schema':'world-reward-hands-automatic-masks-v1','stage':'hand_synthetic_automatic_person_masks',
            'status':'pass','frames':6,'ground_truth_used':False,'challenge_inputs_used':False,'hand_labeled_test':False,
            'oracle_modes':[],'person_query':'person.','images':records,
            'input_manifest':{'sha256':infer.sha256(path),'bytes':path.stat().st_size,'path':'descriptive_only'}}
    report_path=masks/'report.json';report_path.write_text(json.dumps(report))
    return manifest,path,report,report_path


def test_public_whitelist_six_images_masks_no_private(infer,tmp_path):
    public_fixture(infer,tmp_path)
    private=tmp_path/'validation/hands_rgb_v1/eval_private';private.mkdir();(private/'truth.json').write_text('NOTREAD')
    records,hashes=infer.public_inputs(tmp_path)
    assert len(records)==6 and [r['frame_index'] for r in records]==list(range(6))
    assert set(hashes)=={'input_manifest','mask_report'}
    assert all('eval_private' not in str(x) for r in records for x in r.values())


@pytest.mark.parametrize('kind',['abstain','boolcount','manual','rgbhash','maskhash','order','inputextra','schemaextra','wrongsize','symlink'])
def test_public_contract_abstention_hash_manual_or_extra_inputs_fail(infer,tmp_path,kind):
    manifest,path,report,report_path=public_fixture(infer,tmp_path)
    if kind=='abstain':report['status']='abstain'
    if kind=='boolcount':report['frames']=True
    if kind=='manual':report['hand_labeled_test']=True
    if kind=='rgbhash':report['images'][0]['rgb_sha256']='a'*64
    if kind=='maskhash':report['images'][0]['mask_sha256']='a'*64
    if kind=='order':report['images'][0]['frame_index']=1
    if kind=='inputextra':(path.parent/'truth.json').write_text('{}')
    if kind=='schemaextra':manifest['camera_truth']=[];path.write_text(json.dumps(manifest));report['input_manifest']['sha256']=infer.sha256(path);report['input_manifest']['bytes']=path.stat().st_size
    if kind=='wrongsize':manifest['images'][0]['width']=1;path.write_text(json.dumps(manifest));report['input_manifest']['sha256']=infer.sha256(path);report['input_manifest']['bytes']=path.stat().st_size
    if kind=='symlink':p=path.parent/'case_000.png';p.unlink();p.symlink_to(path.parent/'case_001.png')
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError):infer.public_inputs(tmp_path)


def test_derived_mask_bbox_binary_original_grid_no_fallback(infer):
    rgb=np.zeros((768,1024,3),np.uint8);mask=np.zeros((768,1024),np.uint8)
    mask[20:30,40:60]=255
    box,prompt=infer.derived_bbox(rgb,mask)
    assert np.array_equal(box,[40.,20.,60.,30.]) and box.dtype==np.float32
    assert prompt.shape==(768,1024,1) and prompt.dtype==np.uint8 and set(np.unique(prompt))=={0,1}
    with pytest.raises(ValueError):infer.derived_bbox(rgb,np.zeros_like(mask))
    mask[0,0]=1
    with pytest.raises(ValueError):infer.derived_bbox(rgb,mask)
    with pytest.raises(ValueError):infer.derived_bbox(rgb[:2],mask)


def offline_main(infer,monkeypatch,tmp_path,reserved=False):
    public_fixture(infer,tmp_path)
    monkeypatch.setenv('WR_ROOT',str(tmp_path));monkeypatch.setenv('WR_CODE_REVISION','c'*40)
    monkeypatch.setenv('WR_IMAGE_ID','sha256:'+'d'*64)
    monkeypatch.setattr(infer.platform,'system',lambda:'Linux')
    original=Path.iterdir
    monkeypatch.setattr(Path,'iterdir',lambda p:iter([Path('lo')]) if str(p)=='/sys/class/net' else original(p))
    monkeypatch.setattr(sys,'argv',[infer.__file__])
    path=tmp_path/'validation/hands_rgb_v1/predictions-v2/report.json'
    if reserved:path.parent.mkdir();monkeypatch.setenv('WR_HAND_OUTPUT_RESERVED','1')
    return path


def test_parent180s_partial_frozen_outputs_no_private_read(infer,monkeypatch,tmp_path):
    path=offline_main(infer,monkeypatch,tmp_path,reserved=True)
    def run(args,*,check,timeout,env):
        assert timeout==180. and check is False and len(env['WR_HAND_INFER_NONCE'])==64
        report=json.loads(path.read_text());report.update(current_case=0,current_mode='full');path.write_text(json.dumps(report))
        raise subprocess.TimeoutExpired(args,timeout)
    monkeypatch.setattr(infer.subprocess,'run',run)
    with pytest.raises(subprocess.TimeoutExpired):infer.main()
    r=json.loads(path.read_text())
    assert r['status']=='fail' and r['current_mode']=='full' and r['private_truth_read'] is False and r['raw266_used'] is False
    before=path.read_bytes()
    with pytest.raises(FileExistsError):infer.main()
    monkeypatch.setattr(sys,'argv',[infer.__file__,'--worker'])
    with pytest.raises(RuntimeError,match='immutable'):infer.main()
    assert path.read_bytes()==before


def test_empty_output_without_wrapper_reservation_is_not_resumed(infer,monkeypatch,tmp_path):
    path=offline_main(infer,monkeypatch,tmp_path);path.parent.mkdir()
    with pytest.raises(FileExistsError):infer.main()


def test_wrapper_mounts_exact_public_inputs_and_prediction_only_no_groundtruth():
    p=Path(__file__).resolve().parents[1]/'infra/run_hand_synthetic_infer.sh';s=p.read_text()
    assert 'src=$BASE/inputs,dst=$BASE/inputs,readonly' in s
    assert 'src=$BASE/automatic_masks,dst=$BASE/automatic_masks,readonly' in s
    assert 'src=$BASE/predictions-v2,dst=$BASE/predictions-v2"' in s
    assert 'src=$ROOT/validation,dst=' not in s and 'eval_private' not in s
    assert 'src=$ROOT/data' not in s and 'src=$ROOT/outputs' not in s
    assert 'weights-acquisition.json' in s and '--network none' in s and '--gpus all' in s
    assert 'timeout --signal=TERM --kill-after=10s 190s' in s
    assert 'chown "$(id -u scenesmith):$(id -g scenesmith)" "$BASE/predictions-v2"' in s
    assert 'chown -R' not in s
    assert s.index('mkdir "$BASE/predictions-v2"') < s.index('chown ') < s.index('docker run')


def test_strict_helper_no266_or_cached_joint_rotation_decoder(infer):
    source=Path(infer.__file__).read_text()
    assert 'body._source_identity(root)' in source and 'body._body_assets(root)' in source
    assert 'body._load_checkpoint_with_asset_buffers' in source and 'body._native_forward_from_blocks' in source
    assert "joint_global_rotations=rotation_values" in source
    assert "private_truth_read':False" in source and "cam_int=None,inference_type=mode" in source
    assert "BODY_BYTES = 2109129346" in source and infer.MAX_SECONDS==180.
