"""Manufactured metadata fixtures only; no historical values, RGB or network."""
import base64
from copy import deepcopy
import importlib
import json
from pathlib import Path
import socket
import urllib.request

import pytest
import ownership_pair_census as p


def metadata(i, author=None):
    return dict(ImageID=f'{i:016x}', OriginalLandingURL=f'https://www.flickr.com/photos/a{i}/{100000+i}/',
                OriginalURL=f'https://c1.staticflickr.com/{i}_o.jpg', OriginalSize='123',
                OriginalMD5=base64.b64encode(i.to_bytes(16, 'big')).decode(), Rotation='0.0',
                Author=f'Author{i}', AuthorProfileURL=f'https://www.flickr.com/photos/{author or "a"+str(i)}/', Title=f'Title{i}')


def historical():
    groups = []
    for start, count in ((1, 16), (17, 128)):
        rows = [metadata(i) for i in range(start, start+count)]
        groups.append(dict(selected_ids=[r['ImageID'] for r in rows], public_metadata=rows))
    for start, count, schema, tail in ((145, 64, 'world_reward.openimages_joint_pair_cohort.v1', 'TEST'),
                                       (209, 32, 'world_reward.proposal_external_cohort.v1', 'RESERVED')):
        groups.append(dict(schema=schema, no_replacements=True, records=[dict(slot=s,
            split='DEV' if s < count//2 else tail, publisher_metadata=metadata(start+s)) for s in range(count)]))
    for start, count, schema in ((1, 32, 'world_reward.coco_proposal_cohort.v1'),
                                 (33, 64, 'world_reward.coco_endpoint_cohort.v2')):
        records = []
        for slot in range(count):
            iid = start+slot; photo = str(200000+iid)
            records.append(dict(slot=slot, split='DEV' if slot < count//2 else 'RESERVED', image_id=iid,
                image=dict(id=iid, license=4, flickr_url=f'http://farm1.staticflickr.com/42/{photo}_abc123.jpg'),
                photo_id=photo, reference_file=f'reference_{slot:06d}.json', reference_identity=dict(bytes=42,sha256='0'*64)))
        groups.append(dict(schema=schema, freeze_before_rgb=True, no_replacements=True, records=records))
    return groups


def records(n=100):
    return [dict(image_id=f'{i:016x}', eligible=True, publisher_metadata=metadata(i)) for i in range(1000, 1000+n)]


def empty():
    return {k:set() for k in ('ids', 'authors', 'photos', 'md5', 'urls', 'coco_ids')}


def test_all336_slots_including_missing_and_reserved_no_mutation():
    values = historical(); before = deepcopy(values)
    out, counts = p._historical_exclusions(values)
    assert values == before and len(out['ids']) == 240 and len(out['coco_ids']) == 96
    assert counts == dict(inspected_historical_slots=336, excluded_oi_ids=240, excluded_coco_ids=96,
                         excluded_author_profiles=240, excluded_unique_photos=336, excluded_known_md5=240,
                         historical_missing_md5=96, historical_unknown_author_slots=96)


def test_cross_oi_coco_flickr_routes_share_photo_identity_without_false_slot_count():
    values = historical()
    values[4]['records'][0]['image']['flickr_url'] = 'https://live.staticflickr.com/42/100001_abcdef_b.jpg'
    values[4]['records'][0]['photo_id'] = '100001'
    values[0]['public_metadata'][0]['OriginalLandingURL'] = 'https://www.flickr.com/a1/100001/in/set-123/'
    out, counts = p._historical_exclusions(values)
    assert counts['inspected_historical_slots'] == 336 and counts['excluded_unique_photos'] == 335
    row = records(1)[0]; row['publisher_metadata']['OriginalLandingURL'] = 'https://www.flickr.com/photos/another/200002/'
    assert p._capacity([row], out)['independent_slots_lower_bound'] == 0


def test_historical_missing_md5_is_explicit_not_guessed_and_coco_authors_unknown():
    values = historical(); values[0]['public_metadata'][0]['OriginalMD5'] = ''
    out, counts = p._historical_exclusions(values)
    assert counts['historical_missing_md5'] == 97 and len(out['md5']) == 239
    assert counts['historical_unknown_author_slots'] == 96


@pytest.mark.parametrize('bad', ['', 'UNKNOWN', 'https://example.com/people/x/', 'https://www.flickr.com/photos/x/?secret=SECRET'])
def test_historical_oi_author_unknown_or_untrusted_fails(bad):
    values = historical(); values[0]['public_metadata'][0]['AuthorProfileURL'] = bad
    with pytest.raises(ValueError, match='author'): p._historical_exclusions(values)


def test_author_people_photos_route_alias_excluded():
    values = historical(); values[0]['public_metadata'][0]['AuthorProfileURL'] = 'http://www.flickr.com/people/shared'
    out, _ = p._historical_exclusions(values); fresh = records(1)
    fresh[0]['publisher_metadata']['AuthorProfileURL'] = 'https://www.flickr.com/photos/shared/'
    assert p._capacity(fresh, out)['eligible_after_all_exclusions'] == 0


@pytest.mark.parametrize('group', range(6))
def test_any_missing_historical_slot_rejected(group):
    values = historical()
    if group < 2: values[group]['public_metadata'].pop()
    else: values[group]['records'].pop()
    with pytest.raises(ValueError): p._historical_exclusions(values)


def test_duplicate_native_oi_or_coco_ids_rejected_not_union():
    values = historical(); values[2]['records'][0]['publisher_metadata'] = deepcopy(values[0]['public_metadata'][0])
    with pytest.raises(ValueError, match='Disjoint'): p._historical_exclusions(values)
    values = historical(); values[5]['records'][0]['image_id'] = 1; values[5]['records'][0]['image']['id'] = 1
    with pytest.raises(ValueError, match='COCO'): p._historical_exclusions(values)


@pytest.mark.parametrize('field,key', [('ImageID','ids'), ('AuthorProfileURL','authors'), ('OriginalLandingURL','photos'),
                                      ('OriginalMD5','md5'), ('OriginalURL','urls')])
def test_each_historical_identity_excludes_before_capacity(field, key):
    row = records(1)[0]; m = row['publisher_metadata']; author, photo, digest, url = p._metadata_identity(m)
    out = empty(); out[key].add(dict(ids=m['ImageID'], authors=author, photos=photo, md5=digest, urls=url)[key])
    assert p._capacity([row], out)['independent_slots_lower_bound'] == 0


def test_capacity_is_full_count_not96_cohort_deterministic_without_mutation():
    rows = records(110); before = deepcopy(rows); a = p._capacity(rows, empty())
    assert a['independent_slots_lower_bound'] == 110 and rows == before
    assert p._capacity(list(reversed(rows)), empty()) == a
    assert set(a) == {'eligible_after_all_exclusions', 'source_authors', 'unique_photos', 'unique_md5',
                      'independent_slots_lower_bound', 'metadata_inventory_identity'}


@pytest.mark.parametrize('field', ['AuthorProfileURL','OriginalLandingURL','OriginalMD5','OriginalURL'])
def test_automatic_metadata_aliases_do_not_inflate_independent_capacity(field):
    rows = records(96); rows[1]['publisher_metadata'][field] = rows[0]['publisher_metadata'][field]
    assert p._capacity(rows, empty())['independent_slots_lower_bound'] == 95


def test_missing_new_author_or_md5_not_usable_and_bad_rotation_not_repaired():
    rows = records(3); rows[0]['publisher_metadata']['AuthorProfileURL'] = ''; rows[1]['publisher_metadata']['OriginalMD5'] = ''
    assert p._capacity(rows, empty())['independent_slots_lower_bound'] == 1
    rows[2]['publisher_metadata']['Rotation'] = '90.0'
    with pytest.raises(ValueError, match='orientation'): p._capacity(rows, empty())


def test_unknown_record_eligibility_or_duplicate_id_rejected():
    rows = records(2); rows[0]['eligible'] = 1
    with pytest.raises(ValueError, match='Unique'): p._capacity(rows, empty())
    rows = records(1); rows *= 2
    with pytest.raises(ValueError, match='Unique'): p._capacity(rows, empty())


def test_import_and_metadata_helpers_no_network_models_or_old_selection(monkeypatch):
    def forbidden(*_a, **_k): raise AssertionError('Forbidden')
    monkeypatch.setattr(urllib.request, 'urlopen', forbidden); monkeypatch.setattr(socket, 'socket', forbidden)
    importlib.reload(p)
    assert p._capacity(records(), p._historical_exclusions(historical())[0])['independent_slots_lower_bound'] == 100
    src = Path(p.__file__).read_text()
    assert 'census.image_census(boxes[iid], relations[iid], human, parts)' in src
    assert 'coco.select(' not in src and 'identities.select_cohort(' not in src and "row['reference_identity']" not in src


def test_original_geometry_binding_and_group_ambiguity_preserved(monkeypatch):
    row = metadata(1000); iid = row['ImageID']; human = {'p'}; parts = set()
    def box(cls, coords):
        return dict(ImageID=iid, LabelName=cls, IsGroupOf='0', IsDepiction='0', IsInside='0',
                    **dict(zip(('XMin','YMin','XMax','YMax'), map(str,coords))))
    boxes = [box('p',(0,0,.2,.8)), box('p',(.8,0,1,.8)), box('o',(.1,.8,.2,1)), box('o',(.8,.8,.9,1))]
    rel = dict(ImageID=iid, RelationshipLabel='holds', LabelName1='p', LabelName2='o')
    rel.update({k+'1':v for k,v in boxes[0].items() if k in ('XMin','YMin','XMax','YMax')})
    rel.update({k+'2':v for k,v in boxes[2].items() if k in ('XMin','YMin','XMax','YMax')})
    data = dict(triplets=[rel], relations=[rel], boxes=boxes, metadata=[row]); checks=[]
    monkeypatch.setattr(p.census, 'csv_rows', lambda path:iter(data[path]))
    cfg = dict(human_classes=list(human),body_part_classes=[])
    counts = p.collect({k:k for k in data},cfg,empty(),lambda:checks.append(1))
    assert counts['independent_slots_lower_bound'] == 1 and checks
    duplicate = dict(boxes[0],IsGroupOf='1'); data['boxes'] = boxes+[duplicate]
    counts = p.collect({k:k for k in data},cfg,empty(),lambda:None)
    assert counts['independent_slots_lower_bound'] == 0 and counts['unscorable_positive_pairs'] == 1


def test_config_pins_exact_four_cached_csv_six_ledgers_and_helper_sources():
    root = Path(__file__).resolve().parents[1]; cfg=json.loads((root/p.CONFIG).read_text())
    assert cfg['minimum_independent_slots'] == 96 and cfg['selection'] == 'none_feasibility_only'
    assert cfg['historical_slots'] == 336 and len(cfg['historical']) == 6 and cfg['budget_seconds'] == 180
    assert cfg['historical'][-1]['pin'] == dict(bytes=41315,sha256='5b2e6e99f3979f02586b7aa67ccef74a913f836bdc2b7314bcf94d06777562c6')
    prior=json.loads((root/'configs/proposal_external_census_v1.json').read_text())
    assert all(cfg['files'][k]['pin'] == prior['files'][k]['pin'] for k in cfg['files'])
    assert all(p.pin((root/k).read_bytes()) == v for k,v in cfg['reused_helper_pins'].items())


def fake_run(monkeypatch, tmp_path, *, enough=True, changed=False, failing=False):
    root=Path(__file__).resolve().parents[1]; cfg=json.loads((root/p.CONFIG).read_text())
    target=tmp_path/'root'; (target/'results').mkdir(parents=True)
    code=tmp_path/'immutable-code'; (code/'infra').mkdir(parents=True)
    driver=code/p.HELPERS[0]; driver.write_bytes((root/p.HELPERS[0]).read_bytes()); driver.chmod(0o444)
    files=[]
    for name in cfg['files']:
        path=tmp_path/(name+'.csv'); path.write_bytes(b'METADATA\n'); path.chmod(0o444)
        cfg['files'][name]=dict(path=str(path),pin=p.pin(path.read_bytes()),readonly=True); files.append(path)
    for row,value in zip(cfg['historical'],historical()):
        path=tmp_path/(row['name']+'.json'); path.write_bytes(p.encode(value)); path.chmod(0o444)
        row.update(path=str(path),pin=p.pin(path.read_bytes()),readonly=True); files.append(path)
    monkeypatch.setattr(p,'ROOT',target); monkeypatch.setattr(p,'__file__',str(driver))
    monkeypatch.setenv('WR_CODE',str(code)); monkeypatch.setenv('WR_CODE_REVISION','0'*40)
    monkeypatch.setattr(p.sys,'platform','linux'); monkeypatch.setattr(p.os,'geteuid',lambda:0)
    monkeypatch.setattr(p.os,'uname',lambda:type('U',(),{'nodename':'world-reward-ncc-h100-02'})())
    source={'helpers':{p.CONFIG:dict(bytes=1,sha256='0'*64)}}; calls=[]
    def source_read(*_):
        calls.append('source'); return {'changed':True} if changed and calls.count('source')>1 else source
    monkeypatch.setattr(p.rt,'source',source_read); monkeypatch.setattr(p,'configuration',lambda *_:cfg)
    monkeypatch.setattr(p.signal,'setitimer',lambda *_:None)
    def collect(*_):
        calls.append('collect')
        if failing: raise ValueError('SECRET SHOULD NEVER APPEAR')
        return p._capacity(records(96 if enough else 95),empty())
    monkeypatch.setattr(p,'collect',collect)
    return target/cfg['output'],calls


@pytest.mark.parametrize('enough,decision', [(True,'FEASIBLE_PENDING_FROZEN_NEW_STUDY'),(False,'CLOSED_CAPACITY_INCONCLUSIVE')])
def test_actual_lifecycle_pass_is_technical_no_cohort_and_exact_seals(monkeypatch,tmp_path,capsys,enough,decision):
    output,calls=fake_run(monkeypatch,tmp_path,enough=enough)
    report=p.run(); stdout=capsys.readouterr().out; saved=json.loads((output/'report.json').read_bytes())
    assert report == saved and report['status']=='pass' and report['decision']==decision
    assert report['capacity_gate_passed'] is enough
    assert report['source_and_inputs_rehashed_after'] and report['outputs_sealed'] and calls==['source','collect','source']
    assert set(x.name for x in output.iterdir())=={'report.json'}
    assert output.stat().st_mode&0o777==0o555 and (output/'report.json').stat().st_mode&0o777==0o444
    assert report['selection_performed'] is False and report['historical_reference_files_opened'] is False
    assert 'publisher_metadata' not in stdout and 'SECRET' not in stdout


def test_failure_is_sanitized_and_posthash_attempted(monkeypatch,tmp_path,capsys):
    output,calls=fake_run(monkeypatch,tmp_path,failing=True); report=p.run(); capsys.readouterr()
    assert report['status']=='fail' and report['error_type']=='ValueError'
    assert report['source_and_inputs_rehashed_after'] and calls==['source','collect','source']
    assert 'SECRET' not in (output/'report.json').read_text()


def test_source_change_closes_not_published_pass(monkeypatch,tmp_path,capsys):
    output,_=fake_run(monkeypatch,tmp_path,changed=True); report=p.run(); capsys.readouterr()
    assert report['status']=='fail' and not report['source_and_inputs_rehashed_after']
    assert json.loads((output/'report.json').read_bytes())['status']=='fail'


@pytest.mark.parametrize('stage',['seal','write','hash'])
def test_late_publication_after_seal_write_or_identity_cannot_leave_pass(monkeypatch,tmp_path,stage):
    output=tmp_path/'out'; output.mkdir(); report=dict(status='pass',outputs_sealed=False); clock=[0.]
    monkeypatch.setattr(p.time,'monotonic',lambda:clock[0])
    if stage=='seal':
        real=Path.chmod
        def chmod(path,*a,**kw):
            value=real(path,*a,**kw)
            if path==output: clock[0]=181.
            return value
        monkeypatch.setattr(Path,'chmod',chmod)
    elif stage=='write':
        real=p.os.fsync; calls=[0]
        def fsync(fd):
            value=real(fd); calls[0]+=1
            if calls[0]==2: clock[0]=181.
            return value
        monkeypatch.setattr(p.os,'fsync',fsync)
    else:
        real=p.rt.identity
        def identity(*a,**kw):
            value=real(*a,**kw); clock[0]=181.; return value
        monkeypatch.setattr(p.rt,'identity',identity)
    receipt=p._publish(output,report,180.,0.)
    raw=(output/'report.json').read_bytes()
    assert receipt==p.pin(raw) and json.loads(raw)['status']=='fail' and report['error_type']=='TimeoutError'


def test_fresh_output_namespace_not_reused(monkeypatch,tmp_path):
    output,_=fake_run(monkeypatch,tmp_path); output.mkdir()
    with pytest.raises(ValueError,match='Fresh'): p.run()
