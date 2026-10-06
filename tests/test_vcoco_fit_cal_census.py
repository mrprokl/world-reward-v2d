"""Manufactured lineage payloads, no original role values/RGB/Azure/Torch."""
from copy import deepcopy
import base64
import hashlib
import json
import os
from pathlib import Path
import stat
import time

import pytest
import vcoco_fit_cal_census as p


def save(path,value):
    raw=value if type(value)is bytes else p.c.encode(value)
    path.write_bytes(raw);path.chmod(0o400);return p.c.pin(raw)


def fixture(tmp_path,monkeypatch,*,proof=None):
    root=tmp_path/'pilot';root.mkdir(mode=0o700);root.chmod(0o700)
    metadata=root/'metadata';private=root/'eval_private';inputs=root/'inputs'
    for path in (metadata,private,inputs):path.mkdir(mode=0o700)
    monkeypatch.setattr(p.prep,'DATA',root)
    if proof is None:proof=dict(source_binding=dict(producer_revision=p.PREPARE['revision']),census_binding=dict(current=dict(binding=True)),
        source_stat_identity=dict(bytes=7,sha256='b'*64),census_report_identity=dict(bytes=8,sha256='c'*64),
        census_report_state=[1,2],census_directory_state=[3,4])
    rows=[];records=[];mapping=[];public=[];identities={};reference_pins={}
    for slot in range(16):
        iid=1000+slot;name=f'COCO_val2014_{iid:012d}.jpg';photo=str(iid)
        image=dict(id=iid,width=16,height=12,file_name=name,partition='val2014',publisher_image_url=f'http://images.cocodataset.org/val2014/{name}',
            flickr_url=f'http://farm3.staticflickr.com/8/{photo}_abcdef.jpg',photo_id=photo,license=4,license_url=p.c.GRANT,creator_identity='UNKNOWN')
        reference=f'reference_{slot:06d}.json';ref=save(metadata/reference,b'{"UNOPENED_ROLE_POISON":[1e999]}')
        row=dict(slot=slot,split='DEV'if slot<8 else'RESERVED',official_split='val'if slot<8 else'test',image_id=iid,
            image=image,reference_file=reference,reference_identity=ref);rows.append(row);reference_pins[reference]=ref
        blob=b'authored original byte identity '+str(slot).encode();file=f'image_{slot:06d}.jpg';pin=save(inputs/file,blob)
        opaque=hashlib.sha256((p.prep.NAMESPACE+f'{iid:012d}').encode()).hexdigest()[:32]
        status=dict(slot=slot,split=row['split'],image_id=iid,status='acquired',image_pin=pin,
            original_md5=base64.b64encode(hashlib.md5(blob).digest()).decode());records.append(status)
        mapping.append(dict(slot=slot,public_image_id=opaque,public_file=file,image_pin=pin))
        public.append(dict(image_id=opaque,file=file,**pin,width=16,height=12));identities[file]=pin
    cohort=dict(schema='world_reward.vcoco_role_pilot_cohort.v1',producer_revision=p.PREPARE['revision'],source_binding=proof['source_binding'],
        census_report_identity=proof['census_report_identity'],namespace=p.prep.NAMESPACE,retry_count=0,replacement_count=0,records=rows)
    cp=save(metadata/'cohort.json',cohort)
    selection=dict(status='pass',phase='select',source_binding=proof['source_binding'],producer_revision=p.PREPARE['revision'],
        input_proof=proof,cohort_identity=cp,source_and_inputs_rehashed_after=True,outputs_sealed=True,
        artifact_identities={'cohort.json':cp,**reference_pins})
    sp=save(metadata/'report.json',selection)
    pp=save(inputs/'manifest.json',dict(schema='world_reward.rgb_proposal_inputs.v1',images=public));identities['manifest.json']=pp
    acquisition=dict(status='pass',phase='acquire',producer_revision=p.PREPARE['revision'],source_binding=proof['source_binding'],
        input_proof=proof,decision='READY_PENDING_SEPARATE_RGB_BANK',selected_slots=16,selection_report_identity=sp,cohort_identity=cp,
        public_inputs_identity=pp,availability_gate_passed=True,source_and_inputs_rehashed_after=True,outputs_sealed=True,public_outputs_sealed=True,
        RGB_decoded=False,GPU_used=False,models_loaded=False,FIT_performed=False,predictions_read=False,selection_performed=False,
        fresh_reference_values_consulted=False,records=records,public_mappings=mapping,public_artifact_identities=identities,
        counts=dict(slots=16,acquired=16,missing=0,acquired_DEV=8,acquired_RESERVED=8))
    ap=save(private/'report.json',acquisition)
    for path in (metadata,private,inputs):path.chmod(0o500)
    monkeypatch.setattr(p,'PINS',dict(selection=sp,cohort=cp,acquisition=ap,public=pp))
    # Root-only runtime policy is a caller contract; fake ONLY uid/gid projection.
    original_lstat=Path.lstat
    def root_stat(path,*args,**kwargs):
        s=original_lstat(path,*args,**kwargs)
        if path==root or root in path.parents:
            class S:pass
            value=S()
            for n in ('st_dev','st_ino','st_size','st_mode','st_nlink','st_uid','st_gid','st_mtime_ns','st_ctime_ns'):setattr(value,n,getattr(s,n))
            value.st_uid=value.st_gid=0;return value
        return s
    monkeypatch.setattr(Path,'lstat',root_stat)
    excluded=dict(photos={str(i)for i in range(1,433)},authors={'known'},md5={base64.b64encode(b'h'*16).decode()})
    return root,proof,excluded,dict(selection=selection,cohort=cohort,acquisition=acquisition,public=dict(schema='world_reward.rgb_proposal_inputs.v1',images=public))


def update(root,values,label):
    path=root/{'selection':'metadata/report.json','cohort':'metadata/cohort.json','acquisition':'eval_private/report.json','public':'inputs/manifest.json'}[label]
    path.chmod(0o600);pin=save(path,values[label]);p.PINS[label]=pin


def test_frozen_protocol_current_helpers_and_no_test_profile_mutation():
    root=Path(p.__file__).resolve().parents[1]
    assert p.RECIPE['population_source']==p.c.pin((root/'infra/vcoco_population_census.py').read_bytes())
    assert p.RECIPE['minimum_photos']==dict(train=32,val=16)
    assert p.RECIPE['budget_seconds']==1200 and p.RECIPE['outer_seconds']==1215
    assert p.PREPARE['files']==303 and p.PREPARE['entries']==308
    assert p.prep.DATA==Path('/srv/world-reward-data/vcoco_role_pilot_v1')
    assert p.c.ENTRY=='run_vcoco_role_census'


def test_actual_lineage_shapes_all16_ref_hashes_not_decoded(tmp_path,monkeypatch):
    root,proof,excluded,_=fixture(tmp_path,monkeypatch);before=deepcopy(excluded)
    strict=p.rt.strict
    def decode(raw):
        assert b'UNOPENED_ROLE_POISON'not in raw,'Reference payload decoded'
        return strict(raw)
    monkeypatch.setattr(p.rt,'strict',decode)
    result,states=p.pilot(excluded,proof,lambda:None)
    assert excluded==before and len(result['photos'])==448 and len(result['md5'])==17
    assert len(states['references'])==19  # Directory + report/cohort/all16 reference files.
    assert str(root/'inputs/image_000015.jpg')in states['pilot_states']
    assert p.pilot(excluded,proof,lambda:None)==(result,states)


@pytest.mark.parametrize('fault',['source','input_proof','producer','no_ref_flag','slots','mapping','sha','md5','split','counts','public_keys'])
def test_original_pilot_actual_fields_not_mock_boolean_auth(tmp_path,monkeypatch,fault):
    root,proof,excluded,values=fixture(tmp_path,monkeypatch);a=values['acquisition']
    if fault=='source':a['source_binding']['producer_revision']='e'*40
    elif fault=='input_proof':a['input_proof']['census_report_identity']['sha256']='f'*64
    elif fault=='producer':a['producer_revision']='e'*40
    elif fault=='no_ref_flag':a['fresh_reference_values_consulted']=True
    elif fault=='slots':a['selected_slots']=15
    elif fault=='mapping':a['public_mappings'][0]['slot']=1
    elif fault=='sha':a['records'][0]['image_pin']['sha256']='0'*64
    elif fault=='md5':a['records'][0]['original_md5']=base64.b64encode(b'x'*16).decode()
    elif fault=='split':a['records'][0]['split']='RESERVED'
    elif fault=='counts':a['counts']['acquired_DEV']=7
    elif fault=='public_keys':values['public']['images'][0]['label']='forbidden';update(root,values,'public');a['public_inputs_identity']=p.PINS['public']
    update(root,values,'acquisition')
    with pytest.raises(ValueError):p.pilot(excluded,proof,lambda:None)


@pytest.mark.parametrize('fault',['photo','md5','parent_mode','foreign_leaf','reference_pin'])
def test_conservative_alias_privacy_and_reference_byte_guards(tmp_path,monkeypatch,fault):
    root,proof,excluded,values=fixture(tmp_path,monkeypatch)
    if fault=='photo':excluded['photos'].remove('1');excluded['photos'].add('1000')
    elif fault=='md5':excluded['md5'].add(values['acquisition']['records'][0]['original_md5'])
    elif fault=='parent_mode':root.chmod(0o755)
    elif fault=='foreign_leaf':(root/'inputs').chmod(0o700);save(root/'inputs/foreign.txt',b'x');(root/'inputs').chmod(0o500)
    elif fault=='reference_pin':
        leaf=root/'metadata/reference_000015.json';leaf.chmod(0o600);leaf.write_bytes(b'changed');leaf.chmod(0o400)
    with pytest.raises(ValueError):p.pilot(excluded,proof,lambda:None)


def test_canonical_json_proof_roundtrip_and_tamper():
    value=dict(mapping={i:dict(slot=i)for i in range(16)},source=dict(sha256='a'*64))
    parsed=p.normalized(value);assert p.normalized(value)==p.normalized(parsed)
    changed=deepcopy(parsed);changed['mapping']['10']['slot']=11
    assert p.normalized(value)!=p.normalized(changed)


def test_original_source_reused_helper_and_archive_checks(tmp_path,monkeypatch):
    code=tmp_path/'current';old=tmp_path/'jobs'/('a'*40)/'entry/code';code.mkdir();old.mkdir(parents=True)
    helpers=('helper.py',);pin=save(code/'helper.py',b'unchanged')
    marker=old.parent/'source-sha256';marker.write_text('b'*64+'\n');marker.chmod(0o400)
    save(old/'helper.py',b'unchanged');monkeypatch.setattr(p,'ROOT',tmp_path)
    binding=dict(entries=2,closure_sha256='c'*64,helpers={'helper.py':pin})
    monkeypatch.setattr(p.c,'source',lambda *a:dict(binding=binding,stat_identity=pin))
    spec=dict(entries=2,closure_sha256='c'*64,files=1,archive_xz_sha256='b'*64)
    assert p.original(code,'a'*40,'entry',helpers,spec)[0]==old
    marker.chmod(0o600);marker.write_text('d'*64+'\n');marker.chmod(0o400)
    with pytest.raises(ValueError,match='original source'):p.original(code,'a'*40,'entry',helpers,spec)


def test_full_authenticate_reconstructs_actual_prepare_proof_fields(tmp_path,monkeypatch):
    census=tmp_path/'census';census.mkdir(mode=0o700);before=dict(current=dict(binding='native'),historical={},inputs={})
    config=tmp_path/'old-config.json';config_pin=save(config,dict(authored_cfg=True))
    spec=dict(producer_revision='3f445f2d9ee4b2e9849ff15f5dd2cd31cbed964c',configuration=dict(path=str(config),pin=config_pin),
        report=dict(path=str(census/'report.json'),pin=None))
    report=dict(status='pass',stage='complete',capacity_gate_passed=True,source_binding=before,
        configuration_identity=config_pin,outputs_sealed=True,source_and_inputs_rehashed_after=True,historical_slots=432,eligibility_inventory_rows=561)
    spec['report']['pin']=save(census/'report.json',report);census.chmod(0o500)
    prepare_source=dict(binding=dict(producer_revision=p.PREPARE['revision']),stat_identity=dict(bytes=7,sha256='b'*64))
    proof=dict(source_binding=prepare_source['binding'],census_binding=before,source_stat_identity=prepare_source['stat_identity'],
        census_report_identity=spec['report']['pin'],census_report_state=p.c.state(census/'report.json'),census_directory_state=p.c.state(census))
    for key in ('census_report_state','census_directory_state'):proof[key][5:7]=[0,0]
    _,_,excluded,_=fixture(tmp_path,monkeypatch,proof=proof)
    old_cfg=dict(history_names=['h'],inputs={})
    for n in ('h','ownership_cohort','ownership_acquired','coco32_acquired','coco64_acquired'):
        path=tmp_path/n;pin=save(path,dict(identity_only=n));old_cfg['inputs'][n]=dict(path=str(path),pin=pin)
    code=Path(p.__file__).resolve().parents[1]
    monkeypatch.setattr(p.c,'DATA',census)
    monkeypatch.setattr(p.c,'source',lambda *a:dict(binding=dict(helpers={'infra/vcoco_population_census.py':p.RECIPE['population_source']})))
    def original(_code,revision,entry,helpers,_spec):
        assert _code==code
        if entry==p.prep.ENTRY:return tmp_path,prepare_source
        assert revision==spec['producer_revision']and helpers==p.c.HELPERS
        return config.parents[0],dict()
    monkeypatch.setattr(p,'original',original)
    monkeypatch.setattr(p.prep,'configuration',lambda code,source:dict(census=spec))
    monkeypatch.setattr(p.c,'CONFIG',config.name)
    monkeypatch.setattr(p.c,'configuration',lambda cfg:old_cfg)
    monkeypatch.setattr(p.c,'authenticate',lambda cfg,*args:before if cfg==old_cfg else pytest.fail('Wrong old cfg'))
    monkeypatch.setattr(p.c,'exclusions',lambda *args:deepcopy(excluded))
    # Real pinned JSON/report checks plus real pilot16/ref/JPEG verification;
    # source/process authenticity boundary alone is represented by tiny fixtures.
    original_lstat=Path.lstat
    def root_stat(path,*args,**kwargs):
        s=original_lstat(path,*args,**kwargs)
        if path==census or census in path.parents:
            class S:pass
            value=S()
            for n in ('st_dev','st_ino','st_size','st_mode','st_nlink','st_uid','st_gid','st_mtime_ns','st_ctime_ns'):setattr(value,n,getattr(s,n))
            value.st_uid=value.st_gid=0;return value
        return s
    monkeypatch.setattr(Path,'lstat',root_stat)
    actual=p.authenticate(code,'a'*40,lambda:None)
    assert actual[0]==old_cfg and len(actual[1]['photos'])==448
    assert actual[2]['prepare_proof']==proof and actual[2]['original_census']==before


def test_owned_same_fd_publication_late_failure_demotes_not_overwrites(tmp_path,monkeypatch):
    out=tmp_path/'out';out.mkdir(mode=0o700);out.chmod(0o700);owned=out.lstat();expected={}
    p.prep.raw_write(out/'inventory.json',p.c.encode([dict(split='train',image_id=1,photo_id='1')]),expected)
    report=dict(status='pass',decision='CAPACITY_METADATA_ONLY_NO_SELECTION',outputs_sealed=False)
    p.prep.publish(out,report,time.monotonic()-1,time.monotonic(),owned,expected)
    saved=p.rt.strict((out/'report.json').read_bytes())
    assert saved['status']=='fail'and saved['publication_failed']is True
    assert stat.S_IMODE(out.stat().st_mode)==0o500 and stat.S_IMODE((out/'report.json').stat().st_mode)==0o400
    assert set(x.name for x in out.iterdir())=={'inventory.json','report.json'}


@pytest.mark.parametrize('capacity',[False,True])
def test_run_counts_only_outputs_and_final_auth_even_failure(tmp_path,monkeypatch,capsys,capacity):
    out=tmp_path/'census';code=Path(p.__file__).resolve().parents[1];proof=dict(frozen=True);calls=[]
    monkeypatch.setattr(p,'OUTPUT',out);monkeypatch.setenv('WR_CODE',str(code));monkeypatch.setenv('WR_CODE_REVISION','a'*40)
    monkeypatch.setattr(p.os,'geteuid',lambda:0);monkeypatch.setattr(p.sys,'platform','linux')
    monkeypatch.setattr(p.os,'uname',lambda:type('Host',(),dict(nodename='world-reward-ncc-h100-02'))())
    monkeypatch.setattr(p.resource,'getrusage',lambda *_:type('Usage',(),dict(ru_maxrss=1000))())
    before=(dict(),dict(photos=set()),proof)
    def authenticate(*args):calls.append(1);return deepcopy(before)
    monkeypatch.setattr(p,'authenticate',authenticate)
    rows=[dict(split='train',image_id=1,photo_id='1')]
    def census(*args,**kwargs):
        assert kwargs['roles']==p.RECIPE['roles']and kwargs['minimum_photos']==dict(train=32,val=16)
        return dict(capacity_gate_passed=capacity,eligibility_inventory_identity=p.c.pin(p.c.encode(rows))),rows
    monkeypatch.setattr(p.population,'census_population',census)
    result=p.run(time.monotonic(),time.monotonic()+10)
    assert result['status']==('pass'if capacity else'fail')and result['source_and_inputs_rehashed_after']is True
    assert result['FIT_performed']is result['selection_performed']is result['TEST_role_values_read']is False
    assert len(calls)==2 and set(x.name for x in out.iterdir())=={'inventory.json','report.json'}
    assert 'image_id'not in capsys.readouterr().out


def test_source_no_numpy_torch_or_old_producer_execution():
    import ast
    tree=ast.parse(Path(p.__file__).read_bytes());imports=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Import):imports.extend(n.name for n in node.names)
        elif isinstance(node,ast.ImportFrom):imports.append(node.module)
        if isinstance(node,ast.Call)and isinstance(node.func,ast.Attribute):assert node.func.attr not in ('reproduce','select_artifacts','acquire','census')
    assert not any(n.split('.')[0]in('torch','numpy','transformers','onnxruntime')for n in imports)
