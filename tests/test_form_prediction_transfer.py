import copy
import hashlib
import io
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'infra'))
import form_prediction_transfer as t

PRED='a'*40;DEV='b'*40;REV='c'*40;SEQS=['one','two','three','four'];DATASET='d'*40


def digest(raw):return dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


def stage(sid,variant,ip,gp,binding):
    return dict(schema='world_reward.form_external_prediction_stage.v1',status='complete',stage=t.STAGES[variant],
        producer_revision=PRED,sequence_id=sid,dataset='nvidia/form-hoi',input_pin=ip,original_frame_indices=list(range(96)),
        requested_steps=300,full_predictions_sealed_before_evaluation=True,source_rehashed_after=True,oracle_modes=[],
        ground_truth_used=False,private_truth_read=False,hand_labeled_test=False,reference_inputs_mounted=False,
        training_overlap_verified=False,production_adopted=False,artifacts={'eval_geometry.npz':gp},source_binding=binding)


def manifests(tmp_path,monkeypatch):
    root=tmp_path/'root';(root/'results').mkdir(parents=True);monkeypatch.setattr(t,'ROOT',root)
    monkeypatch.setattr(t.dev,'DATA',tmp_path/'dev')
    monkeypatch.setattr(t,'cohorts',lambda code:(SEQS,DATASET))
    base=root/'results'/('form-hoi-external-predict-'+PRED)
    bind=dict(producer_revision=PRED,markers={'rev':1},closure_sha256='e'*64)
    manifest=dict(schema='world_reward.form_hoi_external_prediction_set.v1',producer_revision=PRED,dataset_revision=DATASET,
        split='development',ground_truth_used=False,private_truth_read=False,original_frame_indices=list(range(96)),sequences=[])
    package=dict(schema='world_reward.form_prediction_transfer.v1',prediction_revision=PRED,dev_revision=DEV,
        dataset_revision=DATASET,sequences=SEQS,files=[],all_four_paired_predictions_sealed=True,ground_truth_read=False,
        reference_bytes_included=False,RGB_included=False,models_included=False,private_container_verified=True,
        provenance={'native_source_binding':bind})
    blobs={};runtime={}
    for filename,asset in t.RUNTIME_ASSETS.items():
        raw=b'opaque-fixed-runtime-'+filename.encode();record=dict(path=asset['path'],**digest(raw))
        runtime[filename]=record
        name=t.prefix(PRED,DEV)+'/runtime/'+filename;blobs[name]=raw
    monkeypatch.setattr(t,'RUNTIME_ASSETS',runtime)
    package['runtime_files']=[dict(file=filename,name=t.prefix(PRED,DEV)+'/runtime/'+filename,etag='"etag"',**{k:asset[k] for k in ('bytes','sha256')}) for filename,asset in runtime.items()]
    for sid in SEQS:
        ip=digest(b'publicinput');row=dict(sequence_id=sid,input=dict(path=str(t.dev.DATA/DEV/sid/'inputs/input.json'),pin=ip),variants={})
        for variant,folder in t.STAGES.items():
            raw=b'opaque-no-decode-npz-'+sid.encode()+variant.encode();gp=digest(raw)
            report=(json.dumps(stage(sid,variant,ip,gp,bind),sort_keys=True)+'\n').encode();rp=digest(report)
            row['variants'][variant]=dict(geometry=dict(path=str(base/sid/folder/'eval_geometry.npz'),pin=gp),report=dict(path=str(base/sid/folder/'report.json'),pin=rp))
            for filename,payload in [('eval_geometry.npz',raw),('report.json',report)]:
                name=t.prefix(PRED,DEV)+'/'+sid+'/'+folder+'/'+filename;record=dict(name=name,etag='"etag"',sequence_id=sid,variant=variant,file=filename,**digest(payload))
                package['files'].append(record);blobs[name]=payload
        manifest['sequences'].append(row)
    raw=(json.dumps(manifest,sort_keys=True)+'\n').encode();name=t.prefix(PRED,DEV)+'/manifest.json'
    package['prediction_manifest']=dict(name=name,etag='"etag"',**digest(raw));blobs[name]=raw
    raw=(json.dumps(package,sort_keys=True)+'\n').encode();blobs[t.prefix(PRED,DEV)+'/package.json']=raw
    return root,manifest,package,blobs,digest(raw),bind


def test_fixed_prediction_only_routes_no_media_models_private_queries():
    for file in ('package.json','manifest.json','one/fit_A/report.json','one/fit_B/eval_geometry.npz'):
        assert t.allowed_blob(t.prefix(PRED,DEV)+'/'+file)
    for file in ('one/fit_A/refined.pth','one/rgb.mp4','eval_private/poses.npy','one/fit_A/report.json?sig=secret',
                 'one/fit_C/eval_geometry.npz','../manifest.json','one/fit_A/target.npy'):
        assert not t.allowed_blob(t.prefix(PRED,DEV)+'/'+file)
    with pytest.raises(ValueError):t.revision('main')


def test_prediction_package_and_safe_manifest_relocation_preserve_pins_input_producer(tmp_path,monkeypatch):
    root,m,p,blobs,pp,bind=manifests(tmp_path,monkeypatch)
    t.package_contract(p,PRED,DEV,SEQS,DATASET)
    target=root/'results/package';local=t.relocate_manifest(m,PRED,DEV,SEQS,DATASET,target)
    assert local['producer_revision']==m['producer_revision']
    for a,b in zip(m['sequences'],local['sequences']):
        assert a['input']==b['input']
        for variant in ('A','B'):
            for role in ('geometry','report'):
                assert a['variants'][variant][role]['pin']==b['variants'][variant][role]['pin']
                assert Path(b['variants'][variant][role]['path']).is_relative_to(target)
    assert m['sequences'][0]['variants']['A']['geometry']['path']!=local['sequences'][0]['variants']['A']['geometry']['path']


@pytest.mark.parametrize('fault',['reference','RGB','missing','duplicate','huge','escape','etag','revision','variant','manifest_escape'])
def test_package_fail_closed_no_arbitrary_payloads(tmp_path,monkeypatch,fault):
    root,m,p,blobs,pp,bind=manifests(tmp_path,monkeypatch)
    if fault=='reference':p['reference_bytes_included']=True
    elif fault=='RGB':p['RGB_included']=True
    elif fault=='missing':p['files'].pop()
    elif fault=='duplicate':p['files'][1]=p['files'][0]
    elif fault=='huge':p['files'][0]['bytes']=t.MAX_GEOMETRY+1
    elif fault=='escape':p['files'][0]['name']='../../private'
    elif fault=='etag':p['files'][0]['etag']='secret'
    elif fault=='revision':p['prediction_revision']='e'*40
    elif fault=='variant':p['files'][0]['variant']='C'
    else:p['prediction_manifest']['name']='other/manifest.json'
    with pytest.raises(ValueError):t.package_contract(p,PRED,DEV,SEQS,DATASET)


@pytest.mark.parametrize('fault',['truth','notsealed','steps','source','input','geometry','missingB','route'])
def test_native_stage_or_allfour_manifest_cannot_be_missing_or_false(tmp_path,monkeypatch,fault):
    root,m,p,blobs,pp,bind=manifests(tmp_path,monkeypatch)
    r=stage(SEQS[0],'A',m['sequences'][0]['input']['pin'],m['sequences'][0]['variants']['A']['geometry']['pin'],bind)
    if fault=='truth':r['private_truth_read']=True
    elif fault=='notsealed':r['full_predictions_sealed_before_evaluation']=False
    elif fault=='steps':r['requested_steps']=299
    elif fault=='source':r['source_binding']={}
    elif fault=='input':r['input_pin']=digest(b'otherinput')
    elif fault=='geometry':r['artifacts']['eval_geometry.npz']=digest(b'othergeometry')
    elif fault=='missingB':del m['sequences'][-1]['variants']['B']
    else:m['sequences'][0]['variants']['A']['geometry']['path']=str(root/'eval_private/poses.npy')
    with pytest.raises(ValueError):
        t.stage_contract(r,PRED,SEQS[0],'A',m['sequences'][0]['input']['pin'],m['sequences'][0]['variants']['A']['geometry']['pin'],bind)
        t.prediction_manifest_contract(m,PRED,DEV,SEQS,DATASET,root/'results'/('form-hoi-external-predict-'+PRED))


class Response(io.BytesIO):
    def __init__(self,raw,etag='"etag"'):
        super().__init__(raw);self.status=200;self.headers={'Content-Length':str(len(raw)),'x-ms-meta-sha256':digest(raw)['sha256'],'ETag':etag}


class Client:
    def __init__(self,blobs):self.blobs=blobs;self.calls=[]
    def require_private(self):self.calls.append(('private',))
    def request(self,method,name,headers=None):
        assert method=='GET' and t.allowed_blob(name);self.calls.append((method,name,headers));return Response(self.blobs[name])


def test_fetch_all16_original_pins_no_npz_decode_and_final_local_manifest(tmp_path,monkeypatch):
    root,m,p,blobs,pp,bind=manifests(tmp_path,monkeypatch);client=Client(blobs)
    monkeypatch.setattr(t,'source',lambda *args:bind)
    result=t.fetch(root,PRED,DEV,pp,REV,bind,client=client)
    target=root/'results'/('form-hoi-prediction-package-'+PRED+'-'+DEV)
    local=json.loads((target/'manifest.json').read_bytes())
    t.prediction_manifest_contract(local,PRED,DEV,SEQS,DATASET,target)
    assert result['status']=='pass' and result['local_manifest']['pin']==t.identity(target/'manifest.json')
    assert result['payload_pins_unchanged'] and result['input_routes_unchanged'] and not result['private_references_transferred']
    assert len(list(target.glob('*/fit_*/eval_geometry.npz')))==8
    assert all(not f.stat().st_mode&0o222 for f in target.rglob('*') if f.is_file())
    assert len([x for x in client.calls if x[0]=='GET'])==22
    assert result['runtime_files_count']==4 and not result['runtime_array_values_decoded']
    with pytest.raises(ValueError,match='overwrite'):t.fetch(root,PRED,DEV,pp,REV,bind,client=client)


def test_bad_package_sha_or_manifest_payload_mismatch_no_local_creation(tmp_path,monkeypatch):
    root,m,p,blobs,pp,bind=manifests(tmp_path,monkeypatch);client=Client(blobs)
    bad=dict(pp,sha256='0'*64)
    with pytest.raises(ValueError):t.fetch(root,PRED,DEV,bad,REV,bind,client=client)
    assert not (root/'results'/('form-hoi-prediction-package-'+PRED+'-'+DEV)).exists()
    p['files'][0]['sha256']='0'*64
    raw=(json.dumps(p,sort_keys=True)+'\n').encode();blobs[t.prefix(PRED,DEV)+'/package.json']=raw
    with pytest.raises(ValueError,match='package/manifest'):t.fetch(root,PRED,DEV,digest(raw),REV,bind,client=client)
    assert not (root/'results'/('form-hoi-prediction-package-'+PRED+'-'+DEV)).exists()


def test_preserve_verified_files_no_manifest_after_midfetch_corrupt(tmp_path,monkeypatch):
    root,m,p,blobs,pp,bind=manifests(tmp_path,monkeypatch)
    blobs[p['files'][3]['name']]=b'corrupt'
    with pytest.raises(ValueError):t.fetch(root,PRED,DEV,pp,REV,bind,client=Client(blobs))
    target=root/'results'/('form-hoi-prediction-package-'+PRED+'-'+DEV)
    assert target.exists() and not (target/'manifest.json').exists()
    assert len(list(target.glob('*/fit_*/eval_geometry.npz')))>0
    assert not list(target.glob('**/*.part'))


def test_atomic_sealed_output_no_baseline_overwrite(tmp_path):
    path=tmp_path/'receipt.json';t.seal_bytes(path,b'original')
    with pytest.raises(ValueError):t.seal_bytes(path,b'other')
    assert path.read_bytes()==b'original' and not path.stat().st_mode&0o222


def test_closure_cpu_only_no_secrets_gt_numpy_model_imports():
    root=Path(__file__).resolve().parents[1]
    assert all((root/x).is_file() for x in t.HELPERS)
    script=(root/'infra/form_prediction_transfer.py').read_text();wrapper=(root/'infra/run_form_prediction_transfer.sh').read_text()
    assert 'import numpy' not in script and 'import torch' not in script and 'pickle' not in script
    assert '--gpus' not in wrapper and 'env -i' in wrapper
    assert 'form-hoi-prediction-package-' in script and 'If-Match' in script


def test_assemble_all8_final_stages_before_network_and_no_private_reads(tmp_path,monkeypatch):
    root,m,p,blobs,pp,bind=manifests(tmp_path,monkeypatch);original=root/'results'/('form-hoi-external-predict-'+PRED)
    public=t.dev.DATA/DEV;public.mkdir(parents=True)
    t.seal_json(public/'public-transfer.json',dict(status='pass',dev_revision=DEV,sequences=SEQS,
        private_references_transferred=False,heavy_data_local=False,manifest_identity=digest(b'prior')))
    code=tmp_path/'code';(code/'configs').mkdir(parents=True)
    (code/'configs/form_hoi_external_dev_v1.json').write_text('{}')
    monkeypatch.setattr(t.dev,'validate_public_package',lambda *args,**kwargs:None)
    monkeypatch.setattr(t,'source',lambda *args:bind)
    for row in m['sequences']:
        sid=row['sequence_id'];inp=Path(row['input']['path']);inp.parent.mkdir(parents=True)
        raw=(json.dumps(dict(sequence_id=sid,dataset_revision=DATASET,video=str(inp.parent/'rgb.mp4')))+'\n').encode()
        ip=t.seal_bytes(inp,raw)
        for variant,stage_name in t.STAGES.items():
            d=original/sid/stage_name;d.mkdir(parents=True)
            raw=blobs[t.prefix(PRED,DEV)+'/'+sid+'/'+stage_name+'/eval_geometry.npz']
            gp=t.seal_bytes(d/'eval_geometry.npz',raw)
            t.seal_json(d/'report.json',stage(sid,variant,ip,gp,bind))
    actual,files,tracked,provenance=t.assemble(code,PRED,DEV)
    assert len(files)==16 and len(tracked)==21 and len(actual['sequences'])==4
    assert provenance['native_source_binding']==bind
    assert all('eval_private' not in f['path'].parts for f in files)
    # Missing final B on the last DEV means no manifest / no automatic omission.
    (original/SEQS[-1]/'fit_B/eval_geometry.npz').unlink()
    with pytest.raises(FileNotFoundError):t.assemble(code,PRED,DEV)


class PublishClient(Client):
    def upload(self,name,raw,mime,revision):
        assert t.allowed_blob(name) and name not in self.blobs
        self.calls.append(('PUT',name));self.blobs[name]=raw
        return dict(name=name,mime=mime,etag='"etag"',**digest(raw))
    def head(self,name,pin,etag):
        assert digest(self.blobs[name])=={k:pin[k] for k in ('bytes','sha256')} and etag=='"etag"'
        self.calls.append(('HEAD',name))


def test_publish_package_commit_marker_last_and_source_rehashed(tmp_path,monkeypatch):
    root,m,p,blobs,pp,bind=manifests(tmp_path,monkeypatch);files=[];tracked=[]
    for row in p['files']:
        path=tmp_path/'sealed'/row['sequence_id']/t.STAGES[row['variant']]/row['file'];path.parent.mkdir(parents=True,exist_ok=True)
        payload=blobs[row['name']];pin=t.seal_bytes(path,payload);tracked.append((path,pin))
        files.append(dict(sequence_id=row['sequence_id'],variant=row['variant'],file=row['file'],path=path,pin=pin,mime='application/octet-stream'))
    for filename,asset in t.RUNTIME_ASSETS.items():
        path=root/asset['path'];path.parent.mkdir(parents=True,exist_ok=True)
        t.seal_bytes(path,blobs[t.prefix(PRED,DEV)+'/runtime/'+filename])
    monkeypatch.setattr(t,'assemble',lambda *args:(m,files,tracked,{'native_source_binding':bind}))
    monkeypatch.setattr(t,'source',lambda *args:bind);client=PublishClient({})
    package,receipt=t.publish(root,PRED,DEV,REV,bind,client=client)
    puts=[x[1] for x in client.calls if x[0]=='PUT']
    assert len(puts)==22 and puts[-2:]==[t.prefix(PRED,DEV)+'/manifest.json',t.prefix(PRED,DEV)+'/package.json']
    assert receipt['bytes']==len(client.blobs[puts[-1]])
    assert len(package['files'])==16 and package['all_four_paired_predictions_sealed']
    assert t.identity(root/'results'/('form-prediction-transfer-publish-'+REV+'.json'))['bytes']>0


def test_publish_source_modified_before_network_never_upload(tmp_path,monkeypatch):
    root,m,p,blobs,pp,bind=manifests(tmp_path,monkeypatch);f=tmp_path/'changed';t.seal_bytes(f,b'actual')
    monkeypatch.setattr(t,'runtime_source_files',lambda:[])
    monkeypatch.setattr(t,'assemble',lambda *args:(m,[],[(f,digest(b'wrong'))],{}));client=PublishClient({})
    with pytest.raises(ValueError,match='source changed'):t.publish(root,PRED,DEV,REV,bind,client=client)
    assert not client.calls


def test_exact_public_runtime_pins_match_evaluator_contract():
    import form_hoi_external_eval as evaluator
    for filename in ('mhr_metrics.py','mhr_submission.py','mesh_common.py'):
        asset=t.RUNTIME_ASSETS[filename]
        assert {k:asset[k] for k in ('bytes','sha256')}==evaluator.OFFICIAL['v2dlb/'+filename]
    c=json.loads((Path(__file__).resolve().parents[1]/evaluator.CONFIG).read_bytes())
    assert {k:t.RUNTIME_ASSETS['hand_spec.npz'][k] for k in ('bytes','sha256')}=={k:c['hand_spec'][k] for k in ('bytes','sha256')}
    assert sum(a['bytes'] for a in t.RUNTIME_ASSETS.values())==99306


@pytest.mark.parametrize('fault',['missing','wrongSHA','giant','unknown','duplicate','modelroute'])
def test_runtime_contract_only_four_tiny_exact_public_files(tmp_path,monkeypatch,fault):
    root,m,p,blobs,pp,bind=manifests(tmp_path,monkeypatch)
    r=p['runtime_files']
    if fault=='missing':r.pop()
    elif fault=='wrongSHA':r[0]['sha256']='0'*64
    elif fault=='giant':r[0]['bytes']=1 << 30
    elif fault=='unknown':r[0]['file']='mhr_model.pt'
    elif fault=='duplicate':r[1]=r[0]
    else:r[0]['name']=t.prefix(PRED,DEV)+'/runtime/mhr_model.pt'
    with pytest.raises(ValueError):t.package_contract(p,PRED,DEV,SEQS,DATASET)


def test_runtime_install_can_reuse_exact_immutable_only(tmp_path,monkeypatch):
    root,m,p,blobs,pp,bind=manifests(tmp_path,monkeypatch);client=Client(blobs)
    first=t.install_runtime_files(client,p['runtime_files']);calls=len(client.calls)
    second=t.install_runtime_files(client,p['runtime_files'])
    assert first==second and len(client.calls)==calls
    path=first[0][0];path.chmod(0o644);path.write_bytes(b'different');path.chmod(0o444)
    with pytest.raises(ValueError,match='Existing public runtime'):t.install_runtime_files(client,p['runtime_files'])
    assert path.read_bytes()==b'different'


def test_runtime_sources_must_match_pins_before_network(tmp_path,monkeypatch):
    root,m,p,blobs,pp,bind=manifests(tmp_path,monkeypatch)
    for filename,record in t.RUNTIME_ASSETS.items():
        path=root/record['path'];path.parent.mkdir(parents=True,exist_ok=True)
        t.seal_bytes(path,blobs[t.prefix(PRED,DEV)+'/runtime/'+filename])
    assert len(t.runtime_source_files())==4
    path=root/t.RUNTIME_ASSETS['mhr_metrics.py']['path'];path.chmod(0o644)
    with pytest.raises(ValueError):t.runtime_source_files()


def test_eval_wrapper_uses_only_each_host_existing_qualified_image():
    wrapper=(Path(__file__).resolve().parents[1]/'infra/run_form_hoi_external_eval.sh').read_text()
    assert 'world-reward-ncc-h100-02) IMAGE=sha256:7ebfff18ba3b76dd919485c19115597d7531dfd3233f69461f1dce3f28a6c6d3' in wrapper
    assert 'scenesmith-ncc-h100-01) IMAGE=sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7' in wrapper
    assert 'docker image inspect' in wrapper and 'docker pull' not in wrapper and 'docker tag' not in wrapper
