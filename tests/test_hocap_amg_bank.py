"""Tiny synthetic JPEG/native-callback fixtures only; no models or real data."""
import ast
import copy
from contextlib import nullcontext
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import stat
import sys
from types import SimpleNamespace
from types import ModuleType

import numpy as np
from PIL import Image
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'infra'))
spec = importlib.util.spec_from_file_location('wr_hocap_amg_test',REPO/'infra/hocap_amg_bank.py')
d = importlib.util.module_from_spec(spec); spec.loader.exec_module(d)


def seal(path, raw):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(raw);path.chmod(0o444)
    return dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


def mutate(path,raw):
    path.chmod(0o600);return seal(path,raw)


def jpeg(mode='RGB',size=(19,13)):
    stream=io.BytesIO();Image.new(mode,size).save(stream,format='JPEG');return stream.getvalue()


def cohort(tmp_path, frames=(6,7)):
    p=json.loads((REPO/d.PROTOCOL).read_bytes());p['inputs']=str(tmp_path/'inputs')
    p['output']=str(tmp_path/'output');p['input_report']=str(tmp_path/'extract/report.json')
    directory=Path(p['inputs']);directory.mkdir();rows=[];clips=[]
    for clip,count in zip(p['clips'],frames):
        folder=directory/clip;folder.mkdir()
        for t in range(count):
            name=f'{clip}/color_{t:06d}.jpg';pin=seal(directory/name,jpeg())
            rows.append(dict(clip=clip,camera=p['camera'],frame_position=t,source_frame_id=t,file=name,**pin))
        folder.chmod(0o555)
        clips.append(dict(clip=clip,camera=p['camera'],num_frames=count))
    manifest=dict(schema='world_reward.hocap_rgb_inventory.v1',subject=p['subject'],clips=clips,images=rows,
        raw_metadata_public=False,annotations_public=False,calibration_public=False,timestamps_verified=False,
        image_decoder_qualified=False,reference_continuity_qualified=False)
    pin=seal(directory/'manifest.json',json.dumps(manifest).encode());directory.chmod(0o555)
    return p,pin,manifest


def native_record(mask,bbox=None,score=.9):
    h,w=mask.shape
    return dict(segmentation=mask,bbox=[0.,0.,float(w-1),float(h-1)] if bbox is None else bbox,
        area=int(mask.sum()),predicted_iou=score,stability_score=.99,point_coords=[[.25,.25]],crop_box=[0.,0.,float(w),float(h)])


def test_frozen_native_protocol_and_source_parse():
    raw=(REPO/d.PROTOCOL).read_bytes();assert d.PROTOCOL_PIN==dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    p=json.loads(raw)
    assert p['amg']['min_mask_region_area']==p['amg']['crop_n_layers']==0
    assert p['amg']['points_per_side']==32 and p['amg']['points_per_batch']==64
    assert p['amg']['pred_iou_thresh']==.8 and p['amg']['stability_score_thresh']==.95
    assert p['amg']['multimask_output'] is True and p['amg']['use_m2m'] is False
    assert p['budget_seconds']==900 and p['cleanup_grace_seconds']==60
    assert p['maximum_bank_bytes']==1<<30 and p['maximum_masks_per_anchor']==3072
    ast.parse((REPO/'infra/hocap_amg_bank.py').read_text())


def test_public_full_t_hash_decode_infers_grid_retains_only_five_each(tmp_path):
    p,pin,manifest=cohort(tmp_path)
    clips,value=d.public_inputs(Path(p['inputs']),pin,p,float('inf'))
    progress={};anchors=d.decode_anchors(clips,p,float('inf'),progress)
    assert value==manifest and progress['rgb_decodes']==13
    assert progress['image_size']==[13,19] and progress['anchors_retained']==10
    assert progress['all_original_jpegs_decoded_before_model'] is True
    assert [a['frame_index'] for a in anchors]==[0,1,2,3,5,0,1,3,4,6]
    assert all(a['rgb'].shape==(13,19,3) and a['rgb'].dtype==np.uint8 for a in anchors)
    assert len(clips[0]['records'])==6 and len(clips[1]['records'])==7


@pytest.mark.parametrize('bad',['extra','indices','boolindex','clip','frames','privatefield','timestamp','rgbchange','missing','symlink'])
def test_public_manifest_or_payload_mutations_rejected(tmp_path,bad):
    p,pin,value=cohort(tmp_path);directory=Path(p['inputs'])
    if bad=='extra':directory.chmod(0o755);seal(directory/'meta.yaml',b'PRIVATE')
    elif bad=='indices':value['images'][1]['source_frame_id']=0
    elif bad=='boolindex':value['images'][0]['frame_position']=False
    elif bad=='clip':value['clips'][0]['clip']='other'
    elif bad=='frames':value['clips'][0]['num_frames']=5
    elif bad=='privatefield':value['images'][0]['annotation']='PRIVATE'
    elif bad=='timestamp':value['timestamps_verified']=True
    elif bad=='rgbchange':mutate(directory/value['images'][0]['file'],b'CHANGED')
    elif bad=='missing':
        path=directory/value['images'][0]['file'];path.parent.chmod(0o755);path.unlink()
    elif bad=='symlink':
        path=directory/value['images'][0]['file'];path.parent.chmod(0o755);path.unlink();path.symlink_to(directory/value['images'][1]['file'])
    if bad not in ('rgbchange','missing','symlink'):
        pin=mutate(directory/'manifest.json',json.dumps(value).encode());directory.chmod(0o555)
    with pytest.raises((ValueError,FileNotFoundError)):
        d.public_inputs(directory,pin,p,float('inf'))


@pytest.mark.parametrize('bad',['gray','grid','png','truncated'])
def test_every_frame_native_jpeg_validation_before_anchor_models(tmp_path,bad):
    p,pin,value=cohort(tmp_path)
    # Frame4 is NOT an anchor in the six-frame clip; it still must be decoded.
    path=Path(p['inputs'])/value['images'][4]['file']
    raw=jpeg('L') if bad=='gray' else jpeg(size=(20,13)) if bad=='grid' else b'bad'
    if bad=='png':
        stream=io.BytesIO();Image.new('RGB',(19,13)).save(stream,format='PNG');raw=stream.getvalue()
    if bad=='truncated':raw=jpeg()[:-12]
    value['images'][4].update(mutate(path,raw));pin=mutate(Path(p['inputs'])/'manifest.json',json.dumps(value).encode())
    clips,_=d.public_inputs(Path(p['inputs']),pin,p,float('inf'))
    with pytest.raises((ValueError,OSError)):
        d.decode_anchors(clips,p,float('inf'),{})


def test_complete_all_returned_masks_duplicate_overlap_empty_queries_packed_exact():
    p=json.loads((REPO/d.PROTOCOL).read_bytes());row=dict(clip=p['clips'][0],frame_index=7,frames=20,height=9,width=11)
    full=np.ones((9,11),bool);empty=np.zeros_like(full);one=empty.copy();one[8,10]=True
    records=[native_record(full),native_record(full),native_record(empty),native_record(one,bbox=[10.,8.,0.,0.])]
    arrays,audit=d.bank_arrays(records,row,p)
    assert audit['native_masks']==4 and audit['duplicate_mask_bytes']==1 and audit['zero_query_seeds']==2
    assert len(set(arrays['mask_ids']))==4 and arrays['native_mask_indices'].tolist()==[0,1,2,3]
    unpacked=np.unpackbits(arrays['packed_masks'],axis=1,bitorder='little')[:,:99].reshape(4,9,11).astype(bool)
    np.testing.assert_array_equal(unpacked,np.stack([r['segmentation'] for r in records]))
    np.testing.assert_array_equal(arrays['frame_index'],np.arange(20))
    assert arrays['query_offsets'].tolist()==[0,16,32,32,32]
    assert arrays['query_points'].shape==(32,3) and (arrays['query_points'][:,0]==7).all()
    assert arrays['query_birth_frame_indices'].tolist()==[7]*32
    assert arrays['query_owner_ids'].tolist()==list(np.repeat(arrays['mask_ids'][:2],16))
    assert arrays['native_point_coords_xy'][0].tolist()==[.25,.25]
    assert audit['physical_identities_certified'] is False


def test_empty_native_return_preserves_no_mask_bank():
    p=json.loads((REPO/d.PROTOCOL).read_bytes());row=dict(clip=p['clips'][0],frame_index=0,frames=5,height=3,width=4)
    arrays,audit=d.bank_arrays([],row,p)
    assert arrays['packed_masks'].shape==(0,2) and arrays['mask_ids'].shape==(0,)
    assert arrays['query_points'].shape==(0,3) and arrays['query_offsets'].tolist()==[0]
    assert audit['native_masks']==audit['queries']==0


def test_offimage_native_bbox_clamps_queries_not_mask_metadata_or_refills():
    p=json.loads((REPO/d.PROTOCOL).read_bytes());row=dict(clip=p['clips'][0],frame_index=1,frames=5,height=4,width=4)
    mask=np.ones((4,4),bool)
    arrays,_=d.bank_arrays([native_record(mask,bbox=[-2.,-1.,8.,7.]),native_record(mask,bbox=[8.,8.,1.,1.])],row,p)
    np.testing.assert_array_equal(arrays['native_bbox_xywh'],[[-2.,-1.,8.,7.],[8.,8.,1.,1.]])
    assert arrays['query_offsets'].tolist()==[0,16,16]
    assert (arrays['query_points'][:,1:]>=.5).all() and (arrays['query_points'][:,1:]<4).all()


@pytest.mark.parametrize('bad',['maskdtype','maskshape','nanbbox','negativewidth','score','crop','area','extra','masked','cap'])
def test_native_invalid_record_abort_never_filter(bad):
    p=json.loads((REPO/d.PROTOCOL).read_bytes());row=dict(clip=p['clips'][0],frame_index=0,frames=5,height=4,width=4)
    record=native_record(np.ones((4,4),bool));records=[record]
    if bad=='maskdtype':record['segmentation']=record['segmentation'].astype(np.uint8)
    elif bad=='maskshape':record['segmentation']=np.ones((3,4),bool)
    elif bad=='nanbbox':record['bbox'][0]=np.nan
    elif bad=='negativewidth':record['bbox'][2]=-1
    elif bad=='score':record['predicted_iou']=np.inf
    elif bad=='crop':record['crop_box']=[0,0,3,4]
    elif bad=='area':record['area']=15
    elif bad=='extra':record['target']='manual'
    elif bad=='masked':record['bbox']=np.ma.array(record['bbox'],mask=False)
    elif bad=='cap':p['maximum_masks_per_anchor']=0
    with pytest.raises(ValueError):d.bank_arrays(records,row,p)


def test_actual_callback_ten_calls_no_topk_or_manual_labels(tmp_path):
    p,pin,_=cohort(tmp_path);clips,_=d.public_inputs(Path(p['inputs']),pin,p,float('inf'))
    progress={};anchors=d.decode_anchors(clips,p,float('inf'),progress)
    out=Path(p['output']);out.mkdir();calls=[]
    def generate(rgb):
        calls.append(rgb.shape);mask=np.ones(rgb.shape[:2],bool)
        return [native_record(mask),native_record(mask,score=.85)]
    d.observe(anchors,out,progress,p,float('inf'),generate,lambda:None)
    assert len(calls)==progress['amg_calls']==progress['amg_attempts']==10
    assert len(progress['banks'])==10 and all(r['native_masks']==2 and r['queries']==32 for r in progress['banks'])
    assert (out/'bank').stat().st_mode&0o777==0o555
    for row in progress['banks']:
        assert d.binding.identity(out/row['file'])=={k:row[k] for k in ('bytes','sha256')}
        with np.load(out/row['file'],allow_pickle=False) as saved:
            assert saved['packed_masks'].shape[0]==2
            assert saved['mask_ids'].dtype.kind=='U' and saved['query_points'].dtype==np.float64


def test_resource_abort_without_truncated_saved_bank(tmp_path):
    p,pin,_=cohort(tmp_path);clips,_=d.public_inputs(Path(p['inputs']),pin,p,float('inf'))
    anchors=d.decode_anchors(clips,p,float('inf'),{});out=Path(p['output']);out.mkdir()
    p['maximum_bank_bytes']=1;calls=[]
    def generate(rgb):calls.append(1);return [native_record(np.ones(rgb.shape[:2],bool))]
    with pytest.raises(ValueError,match='resource cap'):
        d.observe(anchors,out,{},p,float('inf'),generate,lambda:None)
    assert len(calls)==1 and not tuple((out/'bank').iterdir())


def extraction_fixture(tmp_path,monkeypatch):
    p,manifest_pin,manifest=cohort(tmp_path)
    rev='a'*40;root=tmp_path/'root';old=root/'jobs'/rev/'run_hocap_extract/code'
    helper=old/'infra/mediapipe_cpu_runtime_verify.py'
    seal(helper,(REPO/'infra/mediapipe_cpu_runtime_verify.py').read_bytes())
    for name in d.EXTRACTION_HELPERS:
        if name!='infra/mediapipe_cpu_runtime_verify.py':seal(old/name,b'tiny-source')
    seal(old.parent/'revision',(rev+'\n').encode());seal(old.parent/'source-sha256',b'b'*64+b'\n')
    for folder in sorted([old,*[x for x in old.rglob('*') if x.is_dir()]],key=lambda x:len(x.parts),reverse=True):folder.chmod(0o555)
    hspec=importlib.util.spec_from_file_location('hocap_test_original_cpu',helper);h=importlib.util.module_from_spec(hspec);hspec.loader.exec_module(h)
    source=h.source(root,old,rev,'run_hocap_extract',d.EXTRACTION_HELPERS)
    receipt=dict(schema='world_reward.hocap_extraction.v1',stage='hocap_saved_rgb_private_metadata_extraction',
        status='pass',phase='complete',public_inventory_qualified=True,source_archives_public_rehashed_after=True,
        network_used=False,label_values_parsed=False,calibration_values_parsed=False,model_loaded=False,gpu_used=False,
        challenge_inputs_used=False,image_decoder_qualified=False,reference_continuity_qualified=False,adoption=False,
        producer_revision=rev,frames=len(manifest['images']),clips=manifest['clips'],public_manifest=manifest_pin,
        source_before=source,protocol_identity=h.identity(old/'configs/hocap_extraction_protocol_v1.json'),
        opaque_metadata={'never_export':{'private_value':'DO_NOT_EXPORT'}},original_ancestry={'private_path':'DO_NOT_EXPORT'})
    pin=seal(Path(p['input_report']),json.dumps(receipt).encode());monkeypatch.setattr(d,'ROOT',root)
    return p,pin,receipt,old


def test_host_receipt_pins_precede_json_and_sanitization_excludes_opaque(tmp_path,monkeypatch):
    p,pin,receipt,_=extraction_fixture(tmp_path,monkeypatch)
    safe=d.input_receipt(Path(p['input_report']),pin,p)
    assert safe['input_report_identity']==pin and safe['public_manifest_identity']==receipt['public_manifest']
    assert 'DO_NOT_EXPORT' not in json.dumps(safe) and 'opaque_metadata' not in safe
    assert safe['opaque_metadata_exported'] is False
    bad=dict(pin,sha256='0'*64)
    original=d.binding.strict_json
    monkeypatch.setattr(d.binding,'strict_json',lambda *_:pytest.fail('Must hash before parse'))
    with pytest.raises(ValueError):d.input_receipt(Path(p['input_report']),bad,p)
    monkeypatch.setattr(d.binding,'strict_json',original)


@pytest.mark.parametrize('field,value',[('status','fail'),('phase','public_extract'),('source_archives_public_rehashed_after',False),
    ('gpu_used',True),('label_values_parsed',True),('calibration_values_parsed',True),('challenge_inputs_used',True),('model_loaded',True)])
def test_host_original_receipt_fail_or_oracle_refused(tmp_path,monkeypatch,field,value):
    p,_,receipt,_=extraction_fixture(tmp_path,monkeypatch);receipt[field]=value
    pin=mutate(Path(p['input_report']),json.dumps(receipt).encode())
    with pytest.raises(ValueError):d.input_receipt(Path(p['input_report']),pin,p)


def test_original_source_mutation_refused_even_resealed_input_receipt(tmp_path,monkeypatch):
    p,pin,_,old=extraction_fixture(tmp_path,monkeypatch)
    mutate(old/'infra/hocap_extract.py',b'changed-source')
    with pytest.raises(ValueError):d.input_receipt(Path(p['input_report']),pin,p)


def test_cli_unknown_abbreviation_duplicate_and_missing_pins_refused():
    for argv in (['--dispatch'],['--disp','--input-report-bytes','1','--input-report-sha256','a'*64],
        ['--dispatch','--run','--input-report-bytes','1','--input-report-sha256','a'*64],
        ['--dispatch','--dispatch','--input-report-bytes','1','--input-report-sha256','a'*64],
        ['--dispatch','--input-report-bytes','1','--input-report-bytes=2','--input-report-sha256','a'*64],
        ['--run','--input-report-bytes','1','--input-report-sha256','a'*64,'--proof-bytes','1',
            '--proof-sha256','b'*64,'--proof-sha256=c']):
        with pytest.raises(SystemExit):d.parser().parse_args(argv)


def test_native_reader_order_and_original_backend_are_explicit_source_contract():
    source=(REPO/'infra/hocap_amg_bank.py').read_text();tree=ast.parse(source)
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='native')
    calls={}
    for n in ast.walk(fn):
        if isinstance(n,ast.Call):
            name=ast.unparse(n.func);calls.setdefault(name,[]).append(n.lineno)
    assert min(calls['decode_anchors'])<min(calls['build_sam2'])<min(calls['SAM2AutomaticMaskGenerator'])
    imports=[n.lineno for n in ast.walk(fn) if isinstance(n,ast.Import) and any(x.name=='torch' for x in n.names)]
    assert min(calls['decode_anchors'])<min(imports)
    assert 'binding.authenticate(' not in source and 'AutoModel' not in source and 'build_sam2_hf' not in source
    assert 'boots.native_prediction' not in source and 'partial_mask_assignment' not in source
    assert 'weights_only' not in source  # Original native SAM2 builder, not custom checkpoint loading.
    assert 'max_hole_area == generator.predictor._transforms.max_sprinkle_area == 0' in source
    assert "warnings.simplefilter('error')" in source
    wrapper=(REPO/'infra/run_hocap_amg_bank.sh').read_text()
    assert '960s' in wrapper and '--dispatch' in wrapper and 'set +x' in wrapper


def test_child_mounts_do_not_include_extraction_report_private_or_entire_models(tmp_path,monkeypatch):
    p=json.loads((REPO/d.PROTOCOL).read_bytes());p['input_report']=str(tmp_path/'extract/report.json')
    p['inputs']=str(tmp_path/'extract/inputs');Path(p['inputs']).mkdir(parents=True)
    code=tmp_path/'root/jobs'/('a'*40)/d.ENTRY/'code';code.mkdir(parents=True)
    safe=tmp_path/'safe';safe.mkdir();checkpoint=safe/'weights/sam2/sam2.1_hiera_large.pt';seal(checkpoint,b'tiny-model')
    monkeypatch.setattr(d.binding,'DEST',safe);monkeypatch.setattr(d.binding,'control_paths',lambda:())
    paths=d.mounts(code,p)
    assert paths==[code.parent,checkpoint,Path(p['inputs'])]
    assert Path(p['input_report']) not in paths and Path(p['input_report']).parent not in paths
    assert safe not in paths and safe/'weights/sam2' not in paths
    monkeypatch.setattr(d.binding,'control_paths',lambda:(Path(p['input_report']).parent,))
    with pytest.raises(ValueError):d.mounts(code,p)


@pytest.mark.parametrize('corrupt_frame',[False,True])
def test_complete_native_child_validates_all_jpegs_before_one_model_load(tmp_path,monkeypatch,corrupt_frame):
    p,manifest_pin,manifest=cohort(tmp_path);root=tmp_path/'root';root.mkdir()
    code=tmp_path/'code';code.mkdir();out=Path(p['output']);out.mkdir()
    if corrupt_frame:
        path=Path(p['inputs'])/manifest['images'][4]['file']
        manifest['images'][4].update(mutate(path,jpeg('L')))
        manifest_pin=mutate(Path(p['inputs'])/'manifest.json',json.dumps(manifest).encode())
    source={'closure_sha256':'b'*64};model={};input_pin=dict(bytes=5940,sha256='c'*64)
    safe=dict(schema='world_reward.hocap_amg_public_input_proof.v1',input_report_identity=input_pin,
        extraction_producer_revision='a'*40,extraction_source_sha256='d'*64,public_manifest_identity=manifest_pin,
        frames=len(manifest['images']),clips=manifest['clips'],private_labels_read=False,calibration_read=False,
        opaque_metadata_exported=False,challenge_inputs_used=False,source_binding=source,
        frontend_proof_sha256=d.json_digest(model),protocol_identity=d.PROTOCOL_PIN)
    proof_pin=seal(out/'sanitized_input_proof.json',json.dumps(safe).encode())
    seal(out/'.container.cid',b'e'*64+b'\n');seal(out/'native.log',b'native')
    monkeypatch.setattr(d,'ROOT',root);monkeypatch.setattr(d,'source_binding',lambda *_:copy.deepcopy(source))
    monkeypatch.setattr(d,'frontend_proof',lambda *_a,**_k:copy.deepcopy(model))
    monkeypatch.setattr(d,'protocol',lambda *_:copy.deepcopy(p))
    monkeypatch.setattr(d.binding,'installed_sam2',lambda *_:dict(verified=True))
    monkeypatch.setenv('WR_AMG_DEADLINE','1e20');monkeypatch.setenv('WR_IMAGE_ID',p['image_id'])
    events=[];torch=ModuleType('torch')
    torch.cuda=SimpleNamespace(is_available=lambda:True,get_device_name=lambda:'H100',
        manual_seed_all=lambda *_:None,synchronize=lambda:None)
    torch.backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
        cudnn=SimpleNamespace(allow_tf32=True))
    torch.manual_seed=lambda *_:None;torch.bfloat16='bfloat16'
    torch.inference_mode=lambda:nullcontext();torch.autocast=lambda *_a,**_k:nullcontext()
    monkeypatch.setitem(sys.modules,'torch',torch)
    build=ModuleType('sam2.build_sam');amg=ModuleType('sam2.automatic_mask_generator')
    def build_model(config,checkpoint,**kwargs):
        assert config==p['installed_config']['path'] and kwargs==dict(device='cuda',mode='eval',apply_postprocessing=True)
        assert not corrupt_frame;events.append('model')
        return SimpleNamespace(training=False,image_encoder=SimpleNamespace(forward=lambda:None),
            sam_mask_decoder=SimpleNamespace(dynamic_multimask_via_stability=True,
                dynamic_multimask_stability_delta=.05,dynamic_multimask_stability_thresh=.98))
    class Generator:
        def __init__(self,model,**kwargs):
            assert kwargs==p['amg']
            self.predictor=SimpleNamespace(_transforms=SimpleNamespace(max_hole_area=0,max_sprinkle_area=0))
        def generate(self,rgb):events.append('AMG');return [native_record(np.ones(rgb.shape[:2],bool))]
    build.build_sam2=build_model;amg.SAM2AutomaticMaskGenerator=Generator
    monkeypatch.setitem(sys.modules,'sam2',ModuleType('sam2'))
    monkeypatch.setitem(sys.modules,'sam2.build_sam',build);monkeypatch.setitem(sys.modules,'sam2.automatic_mask_generator',amg)
    args=SimpleNamespace(input_pin=input_pin,proof_pin=proof_pin)
    if corrupt_frame:
        with pytest.raises(ValueError):d.native(args,code,'f'*40,p)
        assert not events
        report=json.loads((out/'report.json').read_bytes());assert report['status']=='fail' and report['model_loads']==0
    else:
        report=d.native(args,code,'f'*40,p)
        assert report['status']=='pass' and report['rgb_decodes']==13 and report['model_loads']==1
        assert report['all_original_jpegs_decoded_before_model'] and report['original_source_public_models_rehashed_after']
        assert events==['model']+['AMG']*10
        assert not report['boots_executed'] and not report['association_executed']
        assert (out/'report.json').stat().st_mode&0o777==0o444


def test_cleanup_requires_exact_cid_name_image_revision_not_name_only(tmp_path,monkeypatch):
    cid='a'*64;revision='b'*40;path=tmp_path/'CID';seal(path,(cid+'\n').encode())
    calls=[]
    def command(args,seconds=15):
        calls.append(args)
        if args[1]=='ps':return cid if 'id='+cid in args else ''
        if args[1]=='inspect':return d.binding.IMAGE+'|/correct|'+d.ENTRY+'|'+revision
        return ''
    monkeypatch.setattr(d,'_command',command)
    if os.getuid()!=0:
        with pytest.raises(ValueError,match='Owned regular CID'):d.cleanup(path,'correct',revision,d.binding.IMAGE)
        assert not calls
    else:
        d.cleanup(path,'correct',revision,d.binding.IMAGE)
        assert any(c[1:3]==['rm','-f'] for c in calls)
    source=(REPO/'infra/hocap_amg_bank.py').read_text()
    assert 'Foreign container cannot be removed' in source and "s.st_uid == 0" in source


def test_same_sanitized_proof_rejects_extra_private_fields(tmp_path):
    p,pin,manifest=cohort(tmp_path)
    safe=dict(schema='world_reward.hocap_amg_public_input_proof.v1',input_report_identity=pin,
        extraction_producer_revision='a'*40,extraction_source_sha256='d'*64,public_manifest_identity=pin,
        frames=len(manifest['images']),clips=manifest['clips'],private_labels_read=False,calibration_read=False,
        opaque_metadata_exported=False,challenge_inputs_used=False,source_binding={},
        frontend_proof_sha256='b'*64,protocol_identity=d.PROTOCOL_PIN)
    d.validate_sanitized(safe,pin,p)
    safe['opaque_metadata']={'any':'PRIVATE'}
    with pytest.raises(ValueError):d.validate_sanitized(safe,pin,p)


def test_complete_source_closure_allows_extra_public_code_and_hashes_it(tmp_path,monkeypatch):
    root=tmp_path/'root';revision='a'*40;code=root/'jobs'/revision/d.ENTRY/'code'
    for name in d.HELPERS:seal(code/name,(REPO/name).read_bytes())
    extra=code/'configs/another_public_protocol.json';seal(extra,b'{"public":1}')
    seal(code.parent/'revision',(revision+'\n').encode());seal(code.parent/'source-sha256',b'b'*64+b'\n')
    for folder in sorted([code,*[x for x in code.rglob('*') if x.is_dir()]],key=lambda x:len(x.parts),reverse=True):
        folder.chmod(0o555)
    spec=importlib.util.spec_from_file_location('hocap_original_kernel_fixture',REPO/'infra/frontend_sam2_kernel_gate.py')
    kernel=importlib.util.module_from_spec(spec);spec.loader.exec_module(kernel)
    monkeypatch.setattr(kernel,'ROOT',root);monkeypatch.setattr(d,'__file__',str(code/'infra/hocap_amg_bank.py'))
    monkeypatch.setattr(d.binding,'kernel_helper',lambda:kernel)
    before=d.source_binding(code,revision)
    assert set(before['helpers'])==set(d.HELPERS)
    mutate(extra,b'{"public":2}')
    after=d.source_binding(code,revision)
    assert before['helpers']==after['helpers'] and before['closure_sha256']!=after['closure_sha256']


@pytest.mark.parametrize('foreign',['none','image','name','job','revision'])
def test_owned_cid_cleanup_controls_never_remove_foreign_container(monkeypatch,foreign):
    cid='a'*64;revision='b'*40;name='owned';image=d.binding.IMAGE;calls=[];sealed=[]
    path=SimpleNamespace(exists=lambda:True,lstat=lambda:SimpleNamespace(st_mode=stat.S_IFREG|0o600,
        st_nlink=1,st_uid=0,st_size=65),read_bytes=lambda:(cid+'\n').encode(),chmod=sealed.append)
    monkeypatch.setattr(d.binding,'canonical',lambda p:p)
    identity=[image,'/'+name,d.ENTRY,revision]
    if foreign!='none':identity[['image','name','job','revision'].index(foreign)]='foreign'
    def command(args,seconds=15):
        calls.append(args)
        if args[1]=='ps':return cid if 'id='+cid in args else ''
        if args[1]=='inspect':return '|'.join(identity)
        return ''
    monkeypatch.setattr(d,'_command',command)
    if foreign=='none':
        d.cleanup(path,name,revision,image)
        assert ['docker','rm','-f',cid] in calls and sealed==[0o444]
    else:
        with pytest.raises(ValueError,match='Foreign container'):d.cleanup(path,name,revision,image)
        assert not any(c[1]=='rm' for c in calls) and not sealed
