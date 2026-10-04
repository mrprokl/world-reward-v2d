"""Tiny manufactured PNGs and injectable native call signatures, never model runs."""
import ast
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest

REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO/'infra'),str(REPO/'src')]
spec=importlib.util.spec_from_file_location('wr_ycbv_masks_test',REPO/'infra/ycbv_point_masks.py')
gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)


def png(mode='RGB',value=20):
    buffer=io.BytesIO();Image.new(mode,(640,480),value if mode=='L' else (value,30,40)).save(buffer,format='PNG')
    return buffer.getvalue()


def cohort(tmp_path):
    directory=tmp_path/gate.BASE/'inputs';directory.mkdir(parents=True);raw=png();rows=[]
    for scene in gate.public.SCENES:
        for frame in range(96):
            name=f'scene_{scene:06d}_frame_{frame:06d}.png';path=directory/name;path.write_bytes(raw);path.chmod(0o444)
            rows.append(dict(scene_id=scene,frame_id=frame,file=name,sha256=hashlib.sha256(raw).hexdigest(),width=640,height=480))
    manifest=dict(schema=gate.public.SCHEMA,revision=gate.public.REVISION,license='MIT',selection=gate.public.SELECTION,
        attribution=gate.public.ATTRIBUTION,images=rows)
    path=directory/'manifest.json';path.write_text(json.dumps(manifest));path.chmod(0o444)
    pins=dict(schema=gate.public.PINS_SCHEMA,manifest=gate.binding.identity(path),
        acquisition_report=dict(bytes=10,sha256='a'*64,producer_revision='b'*40,script_sha256='c'*64))
    return gate.public.public_inputs(directory,pins)[0],pins


def report():
    return dict(detector_attempts=0,detector_calls=0,sam2_attempts=0,sam2_calls=0,frames_completed=0,seeds=[],masks=[])


def callbacks(events,*,fault=None):
    def detect(rgb,query):
        events.append(('detect',query));return (gate.BoxDetection((10.,12.,100.,120.),.9),)
    def propagate(stage,prompt,masks,weights):
        events.append(('propagate',stage));folder=Path(stage)
        assert [p.name for p in sorted(folder.iterdir())]==[f'{i:06d}.png' for i in range(96)]
        data=json.loads(Path(prompt).read_text())
        assert data=={'prompts':[{'frame_index':0,'object_id':1,'points':None,'point_labels':None,
            'box':{'x0':10.,'y0':12.,'x1':100.,'y1':120.},'mask_path':None}]}
        assert weights==str(gate.binding.DEST/'weights/sam2')
        output=Path(masks)/'1';output.mkdir(parents=True)
        values=np.zeros((480,640),np.uint8);values[10:13,10:13]=255
        for frame in range(96):
            if fault=='missing' and frame==95:continue
            value=values.copy()
            if fault=='empty' and frame==95:value[:]=0
            if fault=='gray' and frame==95:value[0,0]=7
            image=Image.fromarray(value,mode='L')
            if fault=='grid' and frame==95:image=image.resize((639,480))
            if fault=='rgb' and frame==95:image=image.convert('RGB')
            image.save(output/f'{frame:06d}.png')
        if fault=='extra':(output/'private.json').write_text('{}')
        if fault=='raise':raise RuntimeError('native fail')
    return detect,propagate


def test_exact_full_reader_no_private(tmp_path,monkeypatch):
    records,pins=cohort(tmp_path);original=Path.open;directory=records[0]['path'].parent
    def public_only(path,*args,**kwargs):
        assert path.parent==directory;return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',public_only)
    actual,proof=gate.public.public_inputs(directory,pins)
    assert len(actual)==len(proof['RGB_identities'])==288
    assert [r['scene_id'] for r in actual[::96]]==[48,49,50]
    assert [r['frame_id'] for r in actual[:96]]==list(range(96))


def test_three_detections_before_sam_all_native_masks_and_bytecopies(tmp_path):
    records,_=cohort(tmp_path);out=tmp_path/'out';out.mkdir(mode=0o700);events=[];r=report()
    detect,propagate=callbacks(events)
    def checked(*args):
        for index,path in enumerate(sorted(Path(args[0]).iterdir())):
            original=records[(len(events)-3)*96+index]['path']
            assert path.read_bytes()==original.read_bytes() and path.stat().st_ino!=original.stat().st_ino
        return propagate(*args)
    gate.observe(records,out,r,lambda:None,detect=detect,propagate=checked)
    assert events[:3]==[('detect','object.')]*3
    assert all(item[0]=='propagate' for item in events[3:])
    assert r['detector_attempts']==r['detector_calls']==r['sam2_attempts']==r['sam2_calls']==3
    assert r['frames_completed']==len(r['masks'])==288 and r['all_three_detections_passed_before_SAM2']
    assert not list(out.glob('.rgb-stage-*'))
    for row in r['masks']:
        assert row['mask_pixels']==9 and gate.binding.identity(out/row['file'])=={k:row[k] for k in ('bytes','sha256')}
        assert (out/row['file']).stat().st_mode&0o777==0o444


@pytest.mark.parametrize('fault',['missing','ambiguous','low','invalid'])
def test_detection_failure_never_calls_sam_or_creates_stage(tmp_path,fault):
    records,_=cohort(tmp_path);out=tmp_path/'out';out.mkdir();r=report();events=[];ordinary,propagate=callbacks(events)
    def detect(rgb,query):
        if r['detector_attempts']<3:return ordinary(rgb,query)
        if fault=='missing':return ()
        if fault=='ambiguous':return (gate.BoxDetection((0.,0.,10.,10.),.9),gate.BoxDetection((100.,100.,110.,110.),.89))
        if fault=='low':return (gate.BoxDetection((10.,12.,100.,120.),.29),)
        return (gate.BoxDetection((-1.,12.,100.,120.),.9),)
    with pytest.raises(ValueError):gate.observe(records,out,r,lambda:None,detect=detect,propagate=propagate)
    assert r['sam2_attempts']==r['sam2_calls']==0 and not list(out.iterdir())


@pytest.mark.parametrize('fault',['missing','empty','gray','grid','rgb','extra','raise'])
def test_native_mask_failure_not_rescued_and_owned_rgb_stage_removed(tmp_path,fault):
    records,_=cohort(tmp_path);out=tmp_path/'out';out.mkdir();r=report();events=[];detect,propagate=callbacks(events,fault=fault)
    with pytest.raises((ValueError,RuntimeError)):gate.observe(records,out,r,lambda:None,detect=detect,propagate=propagate)
    assert r['frames_completed']==0 and r['detector_calls']==3 and r['sam2_attempts']==1
    assert not list(out.glob('.rgb-stage-*')) and (out/'scene_000048/prompts.json').exists()


def test_disposable_ownership_no_unknown_delete(tmp_path):
    records,_=cohort(tmp_path);directory=tmp_path/'stage';owner,frozen=gate.stage_frames(records[:96],directory)
    extra=directory/'unexpected.txt';extra.write_text('user work')
    with pytest.raises(ValueError):gate.remove_stage(directory,owner,frozen)
    assert extra.read_text()=='user work' and len(list(directory.iterdir()))==97


def test_disposable_changed_or_replaced_file_retained(tmp_path):
    records,_=cohort(tmp_path);directory=tmp_path/'stage';owner,frozen=gate.stage_frames(records[:96],directory)
    path=directory/'000000.png';path.chmod(0o644);path.write_bytes(b'changed');path.chmod(0o444)
    with pytest.raises(ValueError):gate.remove_stage(directory,owner,frozen)
    assert path.read_bytes()==b'changed'


def test_no_overwrite_stage_or_output(tmp_path):
    records,_=cohort(tmp_path);directory=tmp_path/'stage';directory.mkdir();sentinel=directory/'old';sentinel.write_bytes(b'old')
    with pytest.raises(FileExistsError):gate.stage_frames(records[:96],directory)
    assert sentinel.read_bytes()==b'old'
    out=tmp_path/'out';out.mkdir();(out/'scene_000048').mkdir();events=[];detect,propagate=callbacks(events)
    with pytest.raises(FileExistsError):gate.observe(records,out,report(),lambda:None,detect=detect,propagate=propagate)
    assert not any(e[0]=='propagate' for e in events)


def test_rgb_original_png_mode_and_hash(tmp_path):
    records,_=cohort(tmp_path);assert gate.read_rgb(records[0]).shape==(480,640,3)
    path=records[0]['path'];path.chmod(0o644);path.write_bytes(png('L'));path.chmod(0o444)
    with pytest.raises(ValueError):gate.read_rgb(records[0])
    records[0]['sha256']=gate.binding.identity(path)['sha256']
    with pytest.raises(ValueError):gate.read_rgb(records[0])


def test_model_provenance_checks_precede_imports_and_no_old_entry_spoof():
    tree=ast.parse(Path(gate.__file__).read_text());run=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='run')
    source=ast.unparse(run)
    assert source.index('source_binding(')<source.index('import torch')
    assert source.index('public.public_inputs(')<source.index('import torch')
    assert source.index('selected_model_assets(')<source.index('import torch')
    assert source.index('installed_sources(')<source.index('AutoProcessor.from_pretrained(')
    assert 'binding.authenticate(' not in Path(gate.__file__).read_text()
    assert 'validate_assets(' not in source and 'torch.use_deterministic_algorithms(False, warn_only=False)' in source
    assert gate.BUDGET==600 and gate.IMAGE==gate.binding.IMAGE
    assert gate.NATIVE_MASK_SHA=='5193404292cfc7e66053e261049e58b76d3484f92ac1124c481f9951cc4907ec'


def test_source_binding_uses_real_new_namespace_and_original_kernel(tmp_path,monkeypatch):
    root=tmp_path;revision='a'*40;code=root/'jobs'/revision/gate.ENTRY/'code';(code/'infra').mkdir(parents=True)
    script=code/gate.HELPERS[0];script.write_text('own source');script.chmod(0o444)
    monkeypatch.setattr(gate,'ROOT',root);monkeypatch.setattr(gate,'__file__',str(script))
    monkeypatch.setattr(gate.binding,'identity',lambda p,*a:gate.binding.KERNEL_SOURCE_PIN)
    calls=[];monkeypatch.setattr(gate.kernel,'closure',lambda c,r,e,h:calls.append((c,r,e,h))or {'source':'own'})
    assert gate.source_binding(root,code,revision)=={'scope':'complete_host_dispatch','source':'own'}
    assert calls[0][2]==gate.ENTRY and calls[0][3]==gate.HELPERS
    with pytest.raises(ValueError):gate.source_binding(root,root/'jobs'/revision/'run_bridge_rgb_anchor_masks/code',revision)


def test_wrapper_firewall_lock_cleanup_and_real_dependency_closure():
    path=REPO/'infra/run_ycbv_point_masks.sh';source=path.read_text()
    assert subprocess.run(['bash','-n',str(path)]).returncode==0
    assert 'exec 9<"$LOCK"' in source and 'flock --nonblock 9' in source and 'lock_identity fd' in source
    assert '--user 0:0' in source and '--cap-drop ALL' in source and '--security-opt no-new-privileges' in source
    assert '--read-only' in source and '--network none' in source and '630s docker run' in source
    assert '--label world-reward.job=run_ycbv_point_masks' in source and 'docker inspect "$cid"' in source
    assert '--mount "type=bind,src=$BASE/inputs,dst=$BASE/inputs,readonly"' in source
    assert 'eval_private' not in source and 'chown' not in source and 'chmod -R' not in source
    assert '$DEST/weights/$relative' in source and '"$DEST/weights/grounding_dino"' not in source
    import azure_job
    files={str(p.relative_to(REPO)):p.read_bytes() for directory in ('infra','src','configs') for p in (REPO/directory).rglob('*') if p.is_file()and p.suffix in ('.py','.sh','.json','.toml')}
    files['pyproject.toml']=(REPO/'pyproject.toml').read_bytes()
    # Future independent actual pins must be committed before dispatch; no production fallback.
    files.setdefault(gate.PIN_FILE,b'{}')
    selected=azure_job.runtime_bundle_paths(files,'infra/run_ycbv_point_masks.sh')
    assert set(gate.HELPERS)<=set(selected)
    assert 'infra/ycbv_point_depth.py' in selected and 'infra/frontend_sam2_kernel_gate.py' in selected


def write_readonly(path,raw):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():path.chmod(0o600)
    path.write_bytes(raw);path.chmod(0o400)
    return gate.binding.identity(path)


@pytest.fixture
def frontend_fixture(tmp_path,monkeypatch):
    root=tmp_path/'root';code=root/'jobs'/('a'*40)/gate.ENTRY/'code';code.mkdir(parents=True)
    monkeypatch.setattr(gate,'ROOT',root)
    for key,path in (('ROOT',root),('BUILD_CODE',root/'build/code'),('KERNEL_CODE',root/'kernel/code'),
        ('BUILD_REPORT',root/'build.json'),('KERNEL_REPORT',root/'kernel.json')):
        monkeypatch.setattr(gate.binding,key,path)
    cfg=dict(bytes=1,sha256='c'*64)
    old=dict(closure_sha256='b'*64,helpers={gate.binding.CONFIG:cfg,
        'infra/frontend_grounding_build.py':dict(bytes=27520,sha256=gate.binding.BUILD_SHA)})
    kernel=dict(closure_sha256='d'*64,helpers={gate.binding.CONFIG:copy.deepcopy(cfg),
        'infra/frontend_sam2_kernel_gate.py':gate.binding.KERNEL_SOURCE_PIN})
    closures={gate.binding.BUILD_CODE:old,gate.binding.KERNEL_CODE:kernel}
    monkeypatch.setattr(gate.kernel,'closure',lambda c,*a:copy.deepcopy(closures[c]))
    probe=write_readonly(root/'results/frontend-grounding-build-v6/child-CPU-probe.log',b'own CPU source log')
    parent=dict(Id=gate.kernel.BASE,RootFS=dict(Layers=['same']*44))
    child=dict(Id=gate.IMAGE,Architecture='amd64',Os='linux',RootFS=dict(Layers=['same']*44+['child']))
    build=dict(schema='world_reward.frontend_grounding_build.v6',stage='frontend_grounding_build',status='pass',phase='complete',
        producer_revision=gate.binding.BUILD_REV,offline_build_exit_code=0,child_probe_exit_code=0,
        parent_unchanged_verified=True,source_rechecked_before_and_after=True,extension_import_verified=True,
        original_grounding_image_parity_claimed=False,CUDA_execution_verified=False,replica_ready=False,
        license_eligibility_verified=False,training_overlap_verified=False,source_binding=old,source_files={},
        child_image=child,parent_image=parent,owner=hashlib.sha256((gate.binding.BUILD_REV+old['closure_sha256']).encode()).hexdigest(),
        private_child_probe_log=dict(relative_path='results/frontend-grounding-build-v6/child-CPU-probe.log',**probe))
    receipt=dict(schema='world_reward.frontend_sam2_kernel_gate.v1',stage='frontend_sam2_kernel_gate',status='pass',
        producer_revision=gate.binding.KERNEL_REV,script_sha256=gate.binding.KERNEL_SOURCE_PIN['sha256'],image_id=gate.IMAGE,
        models_loaded=False,challenge_data_read=False,CUDA_operator_execution_verified=True,replica_ready=False,
        source_binding=kernel,operator_source_identities=dict(extension=copy.deepcopy(gate.binding.EXTENSION_PIN)))
    original_identity=gate.binding.identity
    monkeypatch.setattr(gate.binding,'identity',lambda p,*a:cfg if Path(p)==code/gate.binding.CONFIG else original_identity(p,*a))
    contract=dict(entries={},manifest_identity={})
    monkeypatch.setattr(gate.binding.selected,'load_contract',lambda r,m:contract)
    def seal():
        pin=write_readonly(gate.binding.BUILD_REPORT,json.dumps(build).encode());monkeypatch.setattr(gate.binding,'BUILD_PIN',pin)
        receipt['build_report_identity']=pin
        pin=write_readonly(gate.binding.KERNEL_REPORT,json.dumps(receipt).encode());monkeypatch.setattr(gate.binding,'KERNEL_PIN',pin)
    seal()
    return root,code,build,receipt,closures,contract,seal


def test_genuine_build_kernel_selected_contract_no_old_entry_execution(frontend_fixture,monkeypatch):
    root,code,build,receipt,closures,contract,seal=frontend_fixture
    live=[];monkeypatch.setattr(gate.kernel,'validate_live_image',lambda proof:live.append(proof['child_image']['Id']))
    proof=gate.frontend_proof(code,live=True)
    assert proof['selected_contract'] is contract and proof['child_image']['Id']==gate.IMAGE
    assert live==[gate.IMAGE] and proof['extension_identity']==gate.binding.EXTENSION_PIN


@pytest.mark.parametrize('fault',['status','bool','source','script','config','owner','image','layer','platform','kernel','extension','probe'])
def test_actual_measured_provenance_fails_closed(frontend_fixture,fault):
    root,code,build,receipt,closures,contract,seal=frontend_fixture
    if fault=='status':build['status']='fail'
    elif fault=='bool':build['offline_build_exit_code']=False
    elif fault=='source':closures[gate.binding.BUILD_CODE]['closure_sha256']='e'*64
    elif fault=='script':closures[gate.binding.BUILD_CODE]['helpers']['infra/frontend_grounding_build.py']['sha256']='e'*64
    elif fault=='config':closures[gate.binding.KERNEL_CODE]['helpers'][gate.binding.CONFIG]['bytes']=2
    elif fault=='owner':build['owner']='e'*64
    elif fault=='image':build['child_image']['Id']=gate.kernel.BASE
    elif fault=='layer':build['child_image']['RootFS']['Layers'][0]='other'
    elif fault=='platform':build['child_image']['Architecture']='arm64'
    elif fault=='kernel':receipt['CUDA_operator_execution_verified']=False
    elif fault=='extension':receipt['operator_source_identities']['extension']['sha256']='e'*64
    else:write_readonly(root/'results/frontend-grounding-build-v6/child-CPU-probe.log',b'changed')
    seal()
    with pytest.raises(ValueError):gate.frontend_proof(code)


def test_selected_exact_principal_model_receipt_revisions(tmp_path,monkeypatch):
    monkeypatch.setattr(gate.binding,'DEST',tmp_path)
    original=dict(assets=[dict(repo_id='IDEA-Research/grounding-dino-base',revision=gate.detector_policy.DETECTOR_REVISION,
        path=str(gate.ROOT/'weights/grounding_dino')),
        dict(repo_id='facebook/sam2.1-hiera-large',revision=gate.detector_policy.SAM2_REVISION,path=str(gate.ROOT/'weights/sam2'))])
    path=tmp_path/'results/weights-acquisition.json';write_readonly(path,json.dumps(original).encode())
    monkeypatch.setattr(gate.binding,'selected_assets',lambda c,a:{'pinned_nine_models':True})
    assert gate.selected_model_assets({'selected_contract':{}})=={'pinned_nine_models':True}
    original['assets'][0]['revision']='main';write_readonly(path,json.dumps(original).encode())
    with pytest.raises(ValueError):gate.selected_model_assets({'selected_contract':{}})


def test_narrow_source_marks_scope_and_forbids_current_acquisition_recipe(tmp_path,monkeypatch):
    revision='a'*40;code=tmp_path/'jobs'/revision/gate.ENTRY/'code'
    for name in gate.HELPERS:write_readonly(code/name,('own'+name).encode())
    write_readonly(code.parent/'revision',(revision+'\n').encode());write_readonly(code.parent/'source-sha256',('b'*64+'\n').encode())
    monkeypatch.setattr(gate,'ROOT',tmp_path);monkeypatch.setattr(gate,'__file__',str(code/gate.HELPERS[0]))
    for module,name in ((gate.binding,'bridge_frontend_bindings'),(gate.kernel,'frontend_sam2_kernel_gate'),
        (gate.detector_policy,'hand_synthetic_masks'),(gate.public,'ycbv_point_depth'),
        (gate.binding.selected,'frontend_selected_assets'),(gate.public.files,'tudl_holdout_inputs')):
        monkeypatch.setattr(module,'__file__',str(code/f'infra/{name}.py'))
    monkeypatch.setattr(gate.binding,'KERNEL_SOURCE_PIN',gate.binding.identity(code/'infra/frontend_sam2_kernel_gate.py'))
    result=gate.source_binding(tmp_path,code,revision,container=True)
    assert result['scope']=='narrow_container_source' and set(result['helpers'])==set(gate.HELPERS)
    assert 'closure_sha256' not in result
    write_readonly(code/'configs/ycbv_point_protocol.json',b'acquisition recipe, not permitted in GPU code')
    with pytest.raises(ValueError):gate.source_binding(tmp_path,code,revision,container=True)


def test_run_failure_rehashes_source_public_models_and_installed_code_before_receipt(tmp_path,monkeypatch):
    from types import SimpleNamespace
    events=[];out=tmp_path/'out';out.mkdir()
    monkeypatch.setattr(gate,'source_binding',lambda *a,**k:events.append('source')or {'bound':'narrow'})
    monkeypatch.setattr(gate,'frontend_proof',lambda *a,**k:events.append('proof')or {'original':'proof'})
    monkeypatch.setattr(gate.binding,'strict_json',lambda raw:{'manifest':{'sha256':'a'*64,'bytes':1},'acquisition_report':{}})
    pinpath=tmp_path/gate.PIN_FILE;pinpath.parent.mkdir();pinpath.write_text('{}')
    monkeypatch.setattr(gate.public,'public_inputs',lambda *a:events.append('public')or ([],{'RGB_identities':{}}))
    monkeypatch.setattr(gate,'selected_model_assets',lambda *a:events.append('models')or {})
    monkeypatch.setattr(gate,'installed_sources',lambda *a:events.append('installed')or {})
    def fail(*a,**k):events.append('observe');raise RuntimeError('native failure')
    monkeypatch.setattr(gate,'observe',fail)
    cuda=SimpleNamespace(is_available=lambda:True,get_device_name=lambda:'own H100 fake',manual_seed_all=lambda x:None,empty_cache=lambda:None)
    torch=SimpleNamespace(cuda=cuda,manual_seed=lambda x:None,use_deterministic_algorithms=lambda *a,**k:None,
        backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False)),cudnn=SimpleNamespace(allow_tf32=False)))
    model=SimpleNamespace(to=lambda device:SimpleNamespace(eval=lambda:object()))
    fake=SimpleNamespace(from_pretrained=lambda *a,**k:model)
    monkeypatch.setitem(sys.modules,'torch',torch)
    monkeypatch.setitem(sys.modules,'transformers',SimpleNamespace(AutoProcessor=fake,AutoModelForZeroShotObjectDetection=fake))
    monkeypatch.setitem(sys.modules,'v2d.sam2.lib.video_to_masks',SimpleNamespace(video_to_masks=lambda *a:None))
    r=report()
    with pytest.raises(RuntimeError,match='native failure'):gate.run(tmp_path,tmp_path,'a'*40,out,r,lambda:None)
    assert r['original_inputs_sources_assets_rehashed_after'] is True
    assert events[-5:]==['source','proof','public','models','installed']
    assert r.get('status')!='pass'
