import copy
import io
import json
from pathlib import Path
import zipfile
import pytest
import visual_genome_ownership_census as vg


def raw(value): return io.BytesIO(json.dumps(value,allow_nan=False).encode())
def object_(iid,name,x=0): return dict(object_id=iid,names=[name],synsets=[],x=x,y=0,w=10,h=10)
def banks():
    values=[object_(1,'person'),object_(2,'woman',20),object_(3,'cup',40),object_(4,'chair',60)]
    objects=[dict(image_id=7,objects=values)]
    relations=[dict(image_id=7,relationships=[dict(relationship_id=9,predicate='holding',subject=values[0],object=values[2])])]
    bank={7:dict(image_id=7,coco_id=70,photo_id='700',width=100,height=100,license_url=vg.GRANT,creator_identity='UNKNOWN')}
    return objects,relations,bank


def test_complete_crowded_positive_is_metadata_only():
    objects,relations,bank=banks(); out,count=vg.semantic_census(raw(objects),raw(relations),bank,{'cup','chair'})
    assert out==[bank[7]] and count['eligible_images']==1 and count['distinct_eligible_photos']==1
    assert count['person_ids']==2 and count['target_ids']==2 and count['published_direct_hold_edges']==1
    assert not any(k in out[0] for k in ('objects','relationships','bbox','predicate','split','slot'))


def test_old_and_rights_unknown_semantics_not_decoded(monkeypatch):
    objects,relations,bank=banks(); originals=vg.js.strict_decode; decoded=[]
    def record(value):
        decoded.append(value); return originals(value)
    monkeypatch.setattr(vg.js,'strict_decode',record)
    poisoned=b'[{"objects":[{"marker":"CLOSED_VALUE","x":1e999}],"image_id":999},'+json.dumps(objects[0]).encode()+b']'
    out,_=vg.semantic_census(io.BytesIO(poisoned),raw(relations),bank,{'cup','chair'})
    assert len(out)==1 and not any(b'CLOSED_VALUE' in value for value in decoded)


@pytest.mark.parametrize('change', ['duplicate_object','duplicate_relation','endpoint_box','endpoint_names','endpoint_synsets',
    'missing_endpoint','bad_box','both_names','missing_synsets','name_type','missing_global','nonfinite','ambiguous_class'])
def test_native_conflicts_reject_image_without_repair(change):
    objects,relations,bank=banks(); values=objects[0]['objects']; pair=relations[0]['relationships'][0]
    if change=='duplicate_object': values.append(copy.deepcopy(values[0]))
    elif change=='duplicate_relation': relations[0]['relationships'].append(copy.deepcopy(pair))
    elif change=='endpoint_box': pair['subject']=dict(pair['subject'],x=1)
    elif change=='endpoint_names': pair['subject']=dict(pair['subject'],names=['man'])
    elif change=='endpoint_synsets': pair['subject']=dict(pair['subject'],synsets=['person.n.01'])
    elif change=='missing_endpoint': pair['subject']=dict(pair['subject'],object_id=99)
    elif change=='bad_box': values[3]['x']=99
    elif change=='both_names': values[3]['name']='chair'
    elif change=='missing_synsets': del values[3]['synsets']
    elif change=='name_type': values[3]['names']='chair'
    elif change=='missing_global': objects=[]
    elif change=='nonfinite':
        source=json.dumps(objects).replace('"x": 0','"x": 1e999',1).encode()
    elif change=='ambiguous_class': values[0]['names']=['person','cup']
    source=io.BytesIO(source) if change=='nonfinite' else raw(objects)
    out,count=vg.semantic_census(source,raw(relations),bank,{'cup','chair'})
    assert out==[] and count['rejected_images']==1


def test_names_variant_exact_predicate_unknown_and_no_reverse():
    objects,relations,bank=banks(); p=relations[0]['relationships'][0]
    for r in objects[0]['objects']: r['name']=r.pop('names')[0]
    assert vg.semantic_census(raw(objects),raw(relations),bank,{'cup','chair'})[1]['eligible_images']==1
    p['predicate']='Holding'
    assert vg.semantic_census(raw(objects),raw(relations),bank,{'cup','chair'})[1]['eligible_images']==0
    p['predicate']='holds'; p['subject'],p['object']=p['object'],p['subject']
    assert vg.semantic_census(raw(objects),raw(relations),bank,{'cup','chair'})[1]['eligible_images']==0
    p['subject'],p['object']=p['object'],p['subject']; objects[0]['objects'][3]['name']='unknown_native'
    assert vg.semantic_census(raw(objects),raw(relations),bank,{'cup','chair'})[1]['eligible_images']==0


def test_duplicate_structural_image_id_closes():
    objects,relations,bank=banks()
    with pytest.raises(ValueError): vg.semantic_census(raw(objects*2),raw(relations),bank,{'cup','chair'})
    with pytest.raises(ValueError): vg.semantic_census(raw(objects),raw(relations*2),bank,{'cup','chair'})


def test_empty_missing_and_raw_identical_boxes_distinct_ids():
    objects,relations,bank=banks(); values=objects[0]['objects']; values[1]['x']=values[0]['x']
    assert vg.semantic_census(raw(objects),raw(relations),bank,{'cup','chair'})[1]['person_ids']==2
    relations[0]['relationships']=[]
    out,count=vg.semantic_census(raw(objects),raw(relations),bank,{'cup','chair'}); assert out==[] and count['insufficient_positive_crowd']==1
    out,count=vg.semantic_census(raw([]),raw([]),bank,{'cup','chair'}); assert out==[] and count['missing_object_or_relation_row']==1
    assert vg.semantic_census(raw([]),raw([]),{},set())[0]==[]


def catalog():
    return dict(images=[dict(id=70,width=100,height=100,license=4,file_name='000000000070.jpg',
        coco_url='http://images.cocodataset.org/val2017/000000000070.jpg',flickr_url='http://farm1.staticflickr.com/1/700_ab.jpg')],
        licenses=[dict(id=4,url=vg.GRANT)],categories=[dict(id=i,name=n) for i,n in enumerate(['person']+['n'+str(i) for i in range(79)])],
        annotations=[dict(marker='OLD_REFERENCE_MUST_NOT_DECODE',bbox=[1,2,3,4])])


def test_coco_metadata_skip_all_annotations_and_historical_photos(monkeypatch):
    decoded=[]; original=vg.js.strict_decode
    def record(value): decoded.append(value); return original(value)
    monkeypatch.setattr(vg.js,'strict_decode',record)
    found,targets=vg.coco_catalog(raw(catalog()),set())
    assert len(targets)==79 and found[70]['creator_identity']=='UNKNOWN'
    assert not any(b'OLD_REFERENCE' in value for value in decoded)
    assert vg.coco_catalog(raw(catalog()),{'700'})[0]=={}
    bad=catalog();bad['licenses'][0]['url']='https://example.com'
    with pytest.raises(ValueError): vg.coco_catalog(raw(bad),set())


def test_image_crosslinks_only_unique_structural_ids_and_grids():
    cat,_=vg.coco_catalog(raw(catalog()),set()); image=dict(image_id=7,coco_id=70,flickr_id=700,width=100,height=100)
    found,count=vg.image_index(raw([image]),cat);assert found[7]['photo_id']=='700'
    for field,value in [('flickr_id',701),('coco_id',999),('width',99),('flickr_id','700x')]:
        bad=dict(image,**{field:value});assert vg.image_index(raw([bad]),cat)[0]=={}
    with pytest.raises(ValueError): vg.image_index(raw([image,image]),cat)
    assert vg.image_index(raw([image,dict(image,image_id=8)]),cat)[0] == {}


def test_zip_only_expected_member_and_full_crc_stream(tmp_path):
    path=tmp_path/'one.zip'
    content=b'[{"image_id":7}]'
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED) as z:z.writestr('image_data.json',content)
    with zipfile.ZipFile(path) as z:crc=f'{z.getinfo("image_data.json").CRC:08x}'
    spec=dict(path=str(path),kind='vg',member='image_data.json',expanded_bytes=len(content),crc32=crc)
    archive,stream=vg.zip_stream(spec,lambda:None)
    assert list(vg.js.iter_array(stream,id_key='image_id'))[0][0]==7 and stream.size==len(content)
    stream.handle.close();archive.close()
    with zipfile.ZipFile(path,'a') as z:z.writestr('foreign.json',b'{}')
    with pytest.raises(ValueError):vg.zip_stream(spec,lambda:None)


def test_primary_requests_exact_one_pinned_text(monkeypatch):
    value=b'holding,holds\n'; row=dict(url='https://homes.cs.washington.edu/~ranjay/x.txt',pin=vg.pin(value))
    calls=[]
    class Response:
        status=200;headers={}
        def __enter__(self):return self
        def __exit__(self,*_):pass
        def geturl(self):return row['url']
        def read(self,n):return value[:n]
    class Opener:
        def open(self,request,timeout):calls.append((request.full_url,timeout));return Response()
    monkeypatch.setattr(vg.urllib.request,'build_opener',lambda *args:Opener())
    assert vg.primary_text(row,vg.time.monotonic()+30)==value and len(calls)==1
    row['pin']=vg.pin(b'other')
    with pytest.raises(ValueError):vg.primary_text(row,vg.time.monotonic()+30)
    row['url']='http://homes.cs.washington.edu/x'
    with pytest.raises(ValueError):vg.primary_text(row,vg.time.monotonic()+30)


def test_frozen_config_no_existing_producer_execution_or_media():
    root=Path(__file__).resolve().parents[1]; cfg=json.loads((root/vg.CONFIG).read_text())
    assert cfg['minimum_distinct_photos']==96 and cfg['budget_seconds']==600 and len(cfg['target_names'])==79
    assert set(cfg['person_names'])==vg.PERSON and set(cfg['predicates'])==vg.PREDICATES
    assert len(cfg['inputs'])==20 and len(cfg['history_names'])==6 and len(cfg['primary_texts'])==3
    assert not any('reference_' in row['path'] or '/videos/' in row['path'] for row in cfg['inputs'].values())
    for name in cfg['helper_pins']:assert vg.pin((root/name).read_bytes())==cfg['helper_pins'][name]
    wrapper=(root/vg.HELPERS[1]).read_text(); assert 'python3 -I -B' in wrapper and 'CUDA_VISIBLE_DEVICES=-1' in wrapper and '615s' in wrapper
    assert vg.configuration(cfg) == cfg
    for key,value in [('minimum_distinct_photos',95),('budget_seconds',1800),('selection_performed',True)]:
        altered=dict(cfg,**{key:value})
        with pytest.raises(ValueError):vg.configuration(altered)


def test_authenticate_distinct_original_receipt_abis_and_all_rechecks(tmp_path,monkeypatch):
    """Tiny fabricated provenance only; no image/annotation archive is consulted."""
    root=Path(__file__).resolve().parents[1];cfg=json.loads((root/vg.CONFIG).read_text());cfg=copy.deepcopy(cfg)
    proofs={}
    for name,s in cfg['sources'].items():
        code=tmp_path/'jobs'/s['revision']/s['entry']/'code';code.mkdir(parents=True)
        (code/'leaf').write_bytes(b'x');(code.parent/'source-sha256').write_text(s['archive_xz_sha256']+'\n')
        s['files']=1
        b=dict(producer_revision=s['revision'],closure_sha256=s['closure_sha256'],entries=s['entries'])
        proofs[name]=dict(binding=b,modes_identity={'bytes':1,'sha256':'a'*64},stat_identity={'bytes':1,'sha256':'b'*64})
    own=proofs['ownership']['binding'];selected=dict(status='pass',phase='select',source_binding=own,producer_revision=cfg['sources']['ownership']['revision'],
        cohort_identity=cfg['inputs']['ownership_cohort']['pin'],source_and_inputs_rehashed_after=True,outputs_sealed=True,
        input_proof=dict(frozen_inputs={r['path']:dict(pin=r['pin']) for r in cfg['inputs'].values()},
            original_closed_failure=dict(report_identity=cfg['inputs']['original_closed_failure']['pin'])))
    values=dict(vg_report=dict(status='pass',source_binding={k:proofs['vg'][k] for k in ('binding','modes_identity')},
        artifact_identities={n:cfg['inputs'][n]['pin'] for n in cfg['vg_files']},source_rehashed_after=True,outputs_sealed=True),ownership_report=selected,
        ownership_cohort=dict(source_binding=own,producer_revision=own['producer_revision'],reference_values_exposed=False),
        ownership_acquired=dict(source_binding=own,producer_revision=own['producer_revision'],phase='acquire',status='fail',
            decision='CLOSED_ACQUISITION_CAPACITY_INCONCLUSIVE',source_and_inputs_rehashed_after=True,outputs_sealed=True),
        coco_report=dict(status='pass',source_binding=proofs['coco']['binding'],producer_revision=proofs['coco']['binding']['producer_revision'],
            phase='census',archive_identity=cfg['inputs']['coco_archive']['pin'],member_identity=dict(bytes=cfg['streams']['coco']['expanded_bytes'],
                sha256=cfg['streams']['coco']['expanded_sha256']),source_and_inputs_rehashed_after=True,outputs_sealed=True))
    pinmap={}
    for name,r in cfg['inputs'].items():
        path=tmp_path/name;path.write_bytes(vg.encode(values.get(name,{})));pinmap[str(path)]=r['pin']
        if r['path'] in selected['input_proof']['frozen_inputs']:
            selected['input_proof']['frozen_inputs'][str(path)]=selected['input_proof']['frozen_inputs'].pop(r['path'])
        r['path']=str(path)
    (tmp_path/'ownership_report').write_bytes(vg.encode(selected))
    monkeypatch.setattr(vg,'ROOT',tmp_path)
    def sourced(code,revision,entry,helpers):
        return next((p for p in proofs.values() if p['binding']['producer_revision']==revision),dict(binding={}))
    monkeypatch.setattr(vg,'source',sourced)
    monkeypatch.setattr(vg.rt,'identity',lambda path,*args,**kwargs:pinmap[str(path)])
    class MD5:
        def update(self,_):pass
        def hexdigest(self):return 'f4bbac642086de4f52a3fdda2de5fa2c'
    monkeypatch.setattr(vg.hashlib,'md5',MD5)
    result=vg.authenticate(cfg,tmp_path/'current','c'*40)
    assert len(result['inputs'])==20 and result['historical']==proofs
    selected['cohort_identity']={'bytes':1,'sha256':'0'*64};(tmp_path/'ownership_report').write_bytes(vg.encode(selected))
    with pytest.raises(ValueError,match='Closed96'):vg.authenticate(cfg,tmp_path/'current','c'*40)


@pytest.mark.parametrize('failure',['deadline','directory_sync','interrupt'])
def test_late_publication_cannot_leave_pass(tmp_path,monkeypatch,failure):
    report=dict(status='pass',decision='CAPACITY_METADATA_ONLY',outputs_sealed=False)
    deadline=vg.time.monotonic()+30
    if failure=='deadline':monkeypatch.setattr(vg,'check',lambda _: (_ for _ in ()).throw(TimeoutError()))
    elif failure=='directory_sync':monkeypatch.setattr(vg.identities,'sync_directory',lambda _: (_ for _ in ()).throw(OSError()))
    else:monkeypatch.setattr(vg,'check',lambda _: (_ for _ in ()).throw(KeyboardInterrupt()))
    vg.publish_report(tmp_path,report,deadline,vg.time.monotonic())
    assert json.loads((tmp_path/'report.json').read_bytes())['status']=='fail'
    assert (tmp_path/'report.json').stat().st_mode&0o777==0o400
    assert report['publication_failed'] is True
