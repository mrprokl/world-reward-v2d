"""Tiny manufactured metadata only: no original labels, RGB, network or Torch."""
from collections import Counter
from copy import deepcopy
import io
import json
from pathlib import Path
import struct
import zipfile

import pytest
import vcoco_population_census as p


ROLES = {n:f'data__vcoco__vcoco_{n}.json' for n in ('train','val')}
MINIMUM = dict(train=32,val=16)


def raw(value): return json.dumps(value,allow_nan=False).encode()


def fixture(tmp_path, *, exclude_poison=False, duplicate_photo=False, extra_context=False):
    cfg = dict(inputs={},archive={},splits={},roles={'val':ROLES['val'],'test':'data__vcoco__vcoco_test.json'},
        max_expanded_bytes=1 << 20,max_row_bytes=1 << 16,max_role_bytes=1 << 16,max_field_bytes=1 << 15,
        minimum_noncrowd_people=2,minimum_nonperson_objects=2,minimum_localized_positive_pairs=1,
        minimum_distinct_photos=dict(val=1,test=1))
    def save(key,content):
        path=tmp_path/key;path.write_bytes(content);path.chmod(0o400)
        cfg['inputs'][key]=dict(path=str(path),pin=p.c.pin(content));return path
    images = {}; annotations = {}
    for iid,partition in ((101,'train2014'),(202,'val2014'),(303,'val2014')):
        filename=f'COCO_{partition}_{iid:012d}.jpg'
        photo=101 if duplicate_photo and iid==202 else iid
        images[iid]=dict(id=iid,width=80,height=60,license=4,file_name=filename,
            coco_url=f'http://images.cocodataset.org/{partition}/{filename}',
            flickr_url=f'http://farm3.staticflickr.com/8/{photo}_abcdef.jpg')
        annotations[iid]=[dict(id=iid*10+j,image_id=iid,category_id=1 if j<3 else 17,
            iscrowd=0,bbox=[-1,1,2,2],area=4) for j in range(1,5)]
    if extra_context:
        annotations[101].extend([
            dict(id=1015,image_id=101,category_id=17,iscrowd=1,bbox=[1,1,2,2],area=4),
            dict(id=1016,image_id=101,category_id=17,iscrowd=0,bbox=[1,1,0,2],area=0)])
    categories=[dict(id=1,name='person')]+[dict(id=i,name=f'category_{i}') for i in range(2,81)]
    contents = {}
    for partition,iids in (('train2014',[101]),('val2014',[202,303])):
        value=dict(images=[images[i] for i in iids],annotations=sum((annotations[i] for i in iids),[]),
            licenses=[dict(id=4,url=p.c.GRANT)],categories=categories)
        content=raw(value)
        if exclude_poison and partition=='val2014':
            content=content.replace(raw(annotations[303][0]),b'{"bbox":[1e999],"marker":"TEST_POISON","image_id":303}')
        contents[f'annotations/instances_{partition}.json']=content
    archive=tmp_path/'original.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for name,content in contents.items():z.writestr(name,content)
    with zipfile.ZipFile(archive) as z:
        catalogue=[dict(member=i.filename,compressed_bytes=i.compress_size,expanded_bytes=i.file_size,
            crc32=f'{i.CRC:08x}',external_attr=i.external_attr,compression=i.compress_type,flags=i.flag_bits) for i in z.infolist()]
    cfg['archive']=dict(member_catalogue=catalogue,catalogue_tail_identity=p.c.pin(archive.read_bytes()[-64:]),
        instances_members={n:f'annotations/instances_{n}.json' for n in ('train2014','val2014')})
    cfg['inputs']['coco_archive']=dict(path=str(archive),pin=p.c.pin(archive.read_bytes()));archive.chmod(0o400)
    save('metadata_report',raw(dict(archive_members=[dict(member=n,expanded_bytes=len(b),
        expanded_sha256=p.c.pin(b)['sha256']) for n,b in contents.items()])))
    for name,ids in dict(train=[101],val=[202],test=[303],trainval=[101,202],all=[101,202,303]).items():
        key=f'split_{name}';save(key,(''.join(f'{i}\n' for i in ids)).encode());cfg['splits'][name]=key
    for name,iid in (('train',101),('val',202),('test',303)):
        action=dict(action_name='authored_action',role_name=['agent','obj','instr'],ann_id=[iid*10+1,iid*10+2],
            image_id=[iid,iid],label=[1,1],role_object_id=[iid*10+1,iid*10+2,iid*10+3,iid*10+4,0,0])
        actions=[action]
        if extra_context and name=='train':
            actions.append(dict(action_name='general_roles',role_name=['agent','obj','instr'],ann_id=[1011,1012],
                image_id=[101,101],label=[1,1],role_object_id=[1011,1012,1015,1016,1011,1012]))
        save(f'data__vcoco__vcoco_{name}.json',raw(actions))
    return cfg


def run(cfg,excluded=()):
    return p.census_population(cfg,dict(photos=set(excluded)),roles=ROLES,minimum_photos=MINIMUM)


def rewrite(cfg,key,value):
    path=Path(cfg['inputs'][key]['path']);path.chmod(0o600);path.write_bytes(raw(value));path.chmod(0o400)
    cfg['inputs'][key]['pin']=p.c.pin(path.read_bytes())


def test_explicit_populations_return_private_metadata_not_selection_or_labels(tmp_path):
    cfg=fixture(tmp_path);before=deepcopy(cfg);report,inventory=run(cfg)
    assert cfg==before and cfg['roles']==dict(val=ROLES['val'],test='data__vcoco__vcoco_test.json')
    assert inventory==[dict(split='train',image_id=101,photo_id='101'),dict(split='val',image_id=202,photo_id='202')]
    assert all(set(row)=={'split','image_id','photo_id'} for row in inventory)
    assert report['eligibility_inventory_identity']==p.c.pin(p.c.encode(inventory))
    assert not report['capacity_gate_passed'] and report['minimum_distinct_photos']==MINIMUM
    assert set(report['splits'])=={'train','val'} and 'references' not in report and 'instances' not in report


def test_same_val_behavior_as_original_census_not_modified(tmp_path):
    cfg=fixture(tmp_path);before=deepcopy(cfg);original=p.c.census(cfg,dict(photos=set()));report,_=run(cfg)
    assert report['splits']['val']==original['splits']['val']
    assert report['expanded_instance_members']==original['expanded_instance_members'] and cfg==before


def test_test_role_never_opened_and_test_annotation_semantics_never_decoded(tmp_path,monkeypatch):
    cfg=fixture(tmp_path,exclude_poison=True);test=cfg['inputs']['data__vcoco__vcoco_test.json']['path']
    decoded=[];original=p.c.js.strict_decode;original_open=Path.open
    def decode(value):decoded.append(value);return original(value)
    def opening(path,*args,**kwargs):
        assert str(path)!=test,'TEST role file opened'
        return original_open(path,*args,**kwargs)
    monkeypatch.setattr(p.c.js,'strict_decode',decode);monkeypatch.setattr(Path,'open',opening)
    report,inventory=run(cfg)
    assert report['eligibility_inventory_rows']==2 and len(inventory)==2
    assert not any(b'TEST_POISON' in value or b'1e999' in value for value in decoded)


def test_all_same_image_instance_context_unscorable_roles_not_removed(tmp_path,monkeypatch):
    cfg=fixture(tmp_path,extra_context=True);original=p.c.parse_vcoco_role_reference;seen=[]
    def parse(actions,instances,images):seen.append(deepcopy(instances));return original(actions,instances,images)
    monkeypatch.setattr(p.c,'parse_vcoco_role_reference',parse)
    report,inventory=run(cfg)
    assert all({1015,1016} <= {r['id'] for r in rows} for rows in seen)
    assert report['splits']['train']['unscorable_positive_roles']==4
    assert report['splits']['train']['localized_positive_pairs']==2
    assert report['catalog_counts']['crowd_instances']==1
    assert report['catalog_counts']['invalid_or_nonpositive_geometry_instances']==1
    assert len(inventory)==2 and seen[0][0]['bbox']==[-1,1,2,2]  # No clipping/rescue.


def test_known_historical_or_pilot_photo_excluded_before_role_semantics(tmp_path):
    cfg=fixture(tmp_path);actions=json.loads(Path(cfg['inputs'][ROLES['train']]['path']).read_bytes())
    actions[0]['label']=['OLD_POISON','OLD_POISON'];rewrite(cfg,ROLES['train'],actions)
    report,inventory=run(cfg,{'101'})
    assert inventory==[dict(split='val',image_id=202,photo_id='202')]
    assert report['splits']['train']['projected_rows']==0


def test_duplicate_fresh_photos_all_rejected_not_counted_twice(tmp_path):
    cfg=fixture(tmp_path,duplicate_photo=True);report,inventory=run(cfg)
    assert not inventory and not report['capacity_gate_passed']
    assert report['catalog_counts']['duplicate_fresh_photo_images_rejected']==2


def test_multiple_action_records_same_pair_do_not_multiply_pair_weight(tmp_path):
    cfg=fixture(tmp_path);actions=json.loads(Path(cfg['inputs'][ROLES['val']]['path']).read_bytes())
    other=deepcopy(actions[0]);other['action_name']='another_authored_action';actions.append(other);rewrite(cfg,ROLES['val'],actions)
    report,_=run(cfg)
    assert report['splits']['val']['positive_action_rows']==4 and report['splits']['val']['localized_positive_pairs']==2


def test_missing_roles_not_off_or_synthetic_positive_pairs(tmp_path):
    cfg=fixture(tmp_path);actions=json.loads(Path(cfg['inputs'][ROLES['train']]['path']).read_bytes())
    actions[0]['role_object_id'][2:]=[0,0,0,0];rewrite(cfg,ROLES['train'],actions)
    report,inventory=run(cfg)
    assert report['splits']['train']['missing_positive_role_ids']==4
    assert report['splits']['train']['localized_positive_pairs']==0 and all(r['split']=='val' for r in inventory)


@pytest.mark.parametrize('roles',[{},dict(train='wrong',val=ROLES['val']),dict(ROLES,test='data__vcoco__vcoco_test.json')])
def test_wrong_or_implicit_role_population_rejected_before_io(roles,monkeypatch):
    monkeypatch.setattr(Path,'open',lambda *a,**k:pytest.fail('I/O before population check'))
    with pytest.raises(ValueError,match='TRAIN/VAL'):p.census_population({},dict(photos=set()),roles=roles,minimum_photos=MINIMUM)


@pytest.mark.parametrize('minimum',[dict(train=True,val=16),dict(train=0,val=16),dict(train=32),dict(train=32,val=16,test=8)])
def test_invalid_unfrozen_minimum_rejected_before_io(minimum,monkeypatch):
    monkeypatch.setattr(Path,'open',lambda *a,**k:pytest.fail('I/O before threshold check'))
    with pytest.raises(ValueError,match='minimums'):p.census_population({},dict(photos=set()),roles=ROLES,minimum_photos=minimum)


@pytest.mark.parametrize('key',['minimum_noncrowd_people','minimum_nonperson_objects','minimum_localized_positive_pairs'])
def test_unchanged_original_predicates_required(tmp_path,key):
    cfg=fixture(tmp_path);cfg[key]=1 if cfg[key]==2 else 0
    with pytest.raises(ValueError,match='predicates'):run(cfg)


def test_source_pin_checked_before_each_role_open(tmp_path,monkeypatch):
    cfg=fixture(tmp_path);calls=Counter();identity=p.c.rt.identity
    def changed(path,*args,**kwargs):
        calls[str(path)]+=1
        if str(path)==cfg['inputs'][ROLES['train']]['path'] and calls[str(path)]==2:return dict(bytes=1,sha256='0'*64)
        return identity(path,*args,**kwargs)
    monkeypatch.setattr(p.c.rt,'identity',changed)
    with pytest.raises(ValueError,match='each open'):run(cfg)


def test_expansion_verified_against_original_receipt_not_just_between_passes(tmp_path):
    cfg=fixture(tmp_path);native=json.loads(Path(cfg['inputs']['metadata_report']['path']).read_bytes())
    native['archive_members'][0]['expanded_sha256']='0'*64;rewrite(cfg,'metadata_report',native)
    with pytest.raises(ValueError,match='complete member SHA'):run(cfg)


def test_crc_failure_exhausted_even_excluded_annotation_rows(tmp_path):
    cfg=fixture(tmp_path);path=Path(cfg['inputs']['coco_archive']['path']);value=bytearray(path.read_bytes())
    offset=value.index(b'PK\x01\x02');crc=struct.unpack_from('<I',value,offset+16)[0]
    struct.pack_into('<I',value,offset+16,crc^1);path.chmod(0o600);path.write_bytes(value);path.chmod(0o400)
    with zipfile.ZipFile(path) as z:
        cfg['archive']['member_catalogue'][0]['crc32']=f'{z.infolist()[0].CRC:08x}'
    cfg['archive']['catalogue_tail_identity']=p.c.pin(path.read_bytes()[-64:])
    with pytest.raises(zipfile.BadZipFile):run(cfg,{'101'})


def test_no_torch_numpy_or_model_import_in_seam():
    import ast
    module=ast.parse(Path(p.__file__).read_bytes());imports=[]
    for node in ast.walk(module):
        if isinstance(node,ast.Import):imports.extend(i.name for i in node.names)
        elif isinstance(node,ast.ImportFrom):imports.append(node.module)
    assert not any(n.split('.')[0] in {'torch','numpy','transformers','onnxruntime'} for n in imports)
    assert not any(isinstance(n,ast.Attribute) and n.attr in {'authenticate','run','configuration'} for n in ast.walk(module))
