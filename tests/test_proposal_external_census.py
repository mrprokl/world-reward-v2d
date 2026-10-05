"""Manufactured metadata only: disjointness, frozen selection and lifecycle."""
from copy import deepcopy
import base64
import hashlib
import importlib
import json
from pathlib import Path
import socket
import urllib.request

import pytest

import proposal_external_census as p


def metadata(i, author=None):
    return dict(ImageID=f'{i:016x}', OriginalLandingURL=f'https://www.flickr.com/photos/a{i}/{100000+i}/',
                OriginalURL=f'https://c1.staticflickr.com/{i}_o.jpg', OriginalSize='123',
                OriginalMD5=base64.b64encode(i.to_bytes(16, 'big')).decode(), Rotation='0.0',
                Author=f'Author{i}', AuthorProfileURL=f'https://www.flickr.com/photos/{author or "a"+str(i)}/', Title=f'Title{i}')


def old_groups():
    groups = []
    for start, count in ((1, 16), (17, 128)):
        rows = [metadata(i) for i in range(start, start+count)]
        groups.append(dict(selected_ids=[r['ImageID'] for r in rows], public_metadata=rows))
    rows = [dict(slot=i, split='DEV' if i < 32 else 'TEST', publisher_metadata=metadata(145+i)) for i in range(64)]
    groups.append(dict(schema='world_reward.openimages_joint_pair_cohort.v1', records=rows,
                       no_replacements=True, challenge_inputs_used=False))
    return groups


def empty_exclusions():
    return {k: set() for k in ('ids', 'authors', 'md5', 'photos', 'urls')}


def records(count=40):
    return [dict(image_id=f'{i:016x}', eligible=True, publisher_metadata=metadata(i)) for i in range(1000, 1000+count)]


def test_all208_including_missing_old64_slots_are_excluded_immutable():
    old = old_groups(); before = deepcopy(old); out = p.exclusions(*old)
    assert old == before and len(out['ids']) == len(out['authors']) == len(out['photos']) == len(out['md5']) == 208
    assert metadata(208)['ImageID'] in out['ids']


def test_author_route_and_trailing_slash_are_canonicalized():
    old = old_groups(); old[0]['public_metadata'][0]['AuthorProfileURL'] = 'http://www.flickr.com/people/shared'
    out = p.exclusions(*old); fresh = records()
    fresh[0]['publisher_metadata']['AuthorProfileURL'] = 'https://www.flickr.com/photos/shared/'
    selected, counts = p.choose(fresh, out)
    assert fresh[0]['image_id'] not in {r['publisher_metadata']['ImageID'] for r in selected}
    assert counts['eligible_after_all_exclusions'] == 39


@pytest.mark.parametrize('bad', ['', 'https://example.com/people/a/', 'https://www.flickr.com/photos/a/?token=SECRET'])
def test_missing_or_noncanonical_historical_author_fails(bad):
    old = old_groups(); old[0]['public_metadata'][0]['AuthorProfileURL'] = bad
    with pytest.raises(ValueError, match='author'):
        p.exclusions(*old)


def test_old16_128_64_overlap_is_error_not_silent_union():
    old = old_groups(); old[2]['records'][0]['publisher_metadata'] = deepcopy(old[0]['public_metadata'][0])
    with pytest.raises(ValueError, match='208'):
        p.exclusions(*old)


def test_old64_must_include_all_slots_and_original_splits():
    for field in ('slot', 'split'):
        old = old_groups(); old[2]['records'][63][field] = 0 if field == 'slot' else 'DEV'
        with pytest.raises(ValueError, match='old64'):
            p.exclusions(*old)


@pytest.mark.parametrize('field, excluded_key', [('ImageID', 'ids'), ('AuthorProfileURL', 'authors'),
                                              ('OriginalMD5', 'md5'), ('OriginalURL', 'urls'),
                                              ('OriginalLandingURL', 'photos')])
def test_each_exclusion_identity_blocks_new_image(field, excluded_key):
    rows = records(); old = metadata(300)
    rows[0]['publisher_metadata'][field] = old[field]
    if field == 'ImageID':
        rows[0]['image_id'] = old[field]
    author, photo, digest, url = p.candidate_identity(old)
    out = empty_exclusions(); out[excluded_key].add(dict(ids=old['ImageID'], authors=author, photos=photo, md5=digest, urls=url)[excluded_key])
    selected, counts = p.choose(rows, out)
    assert rows[0]['image_id'] not in {r['publisher_metadata']['ImageID'] for r in selected}
    assert counts['eligible_after_all_exclusions'] == 39


def test_flickr_direct_set_alias_old_photo_never_becomes_new_photo():
    old = old_groups(); old[0]['public_metadata'][0]['OriginalLandingURL'] = 'https://www.flickr.com/a1/100001/in/set-456/'
    excluded = p.exclusions(*old); rows = records()
    rows[0]['publisher_metadata']['OriginalLandingURL'] = 'https://www.flickr.com/photos/newaccount/100001/'
    _, counts = p.choose(rows, excluded)
    assert counts['eligible_after_all_exclusions'] == 39


def test_exact32_sorted_before_http_no_reference_geometry_exposed_or_mutation():
    rows = records(); before = deepcopy(rows)
    selected, counts = p.choose(rows, empty_exclusions())
    expected = sorted(rows, key=lambda r: (hashlib.sha256((p.HASH_NAMESPACE+r['image_id']).encode()).hexdigest(), r['image_id']))[:32]
    assert [r['publisher_metadata']['ImageID'] for r in selected] == [r['image_id'] for r in expected]
    assert [r['split'] for r in selected] == ['DEV']*16+['RESERVED']*16 and [r['slot'] for r in selected] == list(range(32))
    assert all(set(r) == {'slot', 'split', 'publisher_metadata'} for r in selected)
    assert rows == before and counts['unique_author_photo_md5_slots'] == 40
    assert p.choose(list(reversed(rows)), empty_exclusions()) == (selected, counts)


@pytest.mark.parametrize('field', ['AuthorProfileURL', 'OriginalMD5', 'OriginalLandingURL', 'OriginalURL'])
def test_new_aliases_count_once_and_insufficient32_returns_no_partial_cohort(field):
    rows = records(32)
    rows[1]['publisher_metadata'][field] = rows[0]['publisher_metadata'][field]
    selected, counts = p.choose(rows, empty_exclusions())
    assert selected == [] and counts['unique_author_photo_md5_slots'] == 31


def test_no_accessibility_filter_miss_replacement_or_model_score_in_selection():
    rows = records(32)
    for r in rows:
        r['publisher_metadata']['OriginalSize'] = '123'
        r['arbitrary_prediction'] = -999
    selected, _ = p.choose(rows, empty_exclusions())
    assert len(selected) == 32 and all('arbitrary_prediction' not in r for r in selected)
    rows[0]['eligible'] = False
    assert p.choose(rows, empty_exclusions())[0] == []


def test_unknown_new_identity_is_excluded_before_frozen_selection():
    rows = records(33); rows[0]['publisher_metadata']['OriginalMD5'] = ''
    selected, counts = p.choose(rows, empty_exclusions())
    assert len(selected) == 32 and counts['eligible_after_all_exclusions'] == 32


def test_bad_or_duplicate_new_metadata_not_repaired():
    rows = records(); rows.append(deepcopy(rows[0]))
    with pytest.raises(ValueError, match='Unique'):
        p.choose(rows, empty_exclusions())
    rows = records(); rows[0]['publisher_metadata']['Rotation'] = 'nan'
    with pytest.raises(ValueError, match='orientation'):
        p.choose(rows, empty_exclusions())


def test_geometry_reuses_exact_original_function_not_changed_matching():
    assert p.census.image_census.__module__ == 'openimages_joint_pair_census'
    src = Path(p.__file__).read_text()
    assert 'census.image_census(boxes[iid], relations[iid], human, parts)' in src
    assert 'acquire_slot(' not in src and 'acquire_cohort(' not in src


def test_no_network_or_model_loading_even_during_import_and_selection(monkeypatch):
    def forbidden(*_a, **_k):
        raise AssertionError('Forbidden network call')
    monkeypatch.setattr(urllib.request, 'urlopen', forbidden)
    monkeypatch.setattr(socket, 'socket', forbidden)
    importlib.reload(p)
    assert len(p.choose(records(), p.exclusions(*old_groups()))[0]) == 32


def test_config_exact_preregistered_pins_and_no_scope_rescue():
    root = Path(__file__).resolve().parents[1]; cfg = json.loads((root/p.CONFIG).read_text())
    assert cfg['budget_seconds'] == 180 and cfg['old_ids_excluded'] == 208
    assert cfg['slots'] == 32 and cfg['dev'] == cfg['reserved'] == 16 and cfg['network_allowed'] is False
    assert cfg['files']['old64']['pin'] == dict(bytes=31043, sha256='441bdc57e101cb4e8291ca6bbebc554ca397619e2a892469d9e410a16359a456')
    for f, pin in cfg['reused_helper_pins'].items():
        assert p.pin((root/f).read_bytes()) == pin
    assert cfg['selection']['no_replacements'] is True


def fake_run(monkeypatch, tmp_path, *, late=False, changed=False, enough=True):
    root = Path(__file__).resolve().parents[1]; cfg = json.loads((root/p.CONFIG).read_text())
    fake_root = tmp_path/'root'; (fake_root/'results').mkdir(parents=True)
    paths = {}
    for k, value in zip(('old16', 'old128', 'old64'), old_groups()):
        file = tmp_path/(k+'.json'); file.write_bytes(p.encode(value)); file.chmod(0o444)
        paths[k] = file
    for k in ('boxes', 'relations', 'triplets', 'metadata'):
        file = tmp_path/(k+'.csv'); file.write_bytes(b'METADATA\n'); file.chmod(0o444); paths[k] = file
    cfg['files'] = {k:dict(path=str(v), pin=p.pin(v.read_bytes())) for k,v in paths.items()}
    monkeypatch.setattr(p, 'ROOT', fake_root); monkeypatch.setenv('WR_CODE_REVISION', '0'*40)
    monkeypatch.setenv('WR_CODE', str(root)); monkeypatch.setattr(p.sys, 'platform', 'linux')
    monkeypatch.setattr(p.os, 'geteuid', lambda: 0)
    monkeypatch.setattr(p.os, 'uname', lambda: type('U', (), {'nodename':'world-reward-ncc-h100-02'})())
    source = {'helpers':{p.CONFIG: {'bytes':1, 'sha256':'0'*64}}}
    calls = []
    def source_read(*_):
        calls.append('source')
        return {'changed':True} if changed and len(calls) > 1 else source
    monkeypatch.setattr(p.rt, 'source', source_read); monkeypatch.setattr(p, 'configuration', lambda *_:cfg)
    actual_identity = p.rt.identity
    def identity(*a, **kw):
        calls.append('identity'); return actual_identity(*a, **kw)
    monkeypatch.setattr(p.rt, 'identity', identity)
    clock = [0.]
    monkeypatch.setattr(p.time, 'monotonic', lambda:clock[0])
    def collect(*args):
        calls.append('collect')
        selected, counts = p.choose(records(40 if enough else 31), p.exclusions(*old_groups()))
        if late:
            clock[0] = 181.
        return selected, counts
    monkeypatch.setattr(p, 'collect', collect)
    return fake_root/p.configuration(None,None)['output'], calls


def test_actual_lifecycle_posthash_before_sealed_cohort_and_compact_report(monkeypatch, tmp_path, capsys):
    output, calls = fake_run(monkeypatch, tmp_path)
    report = p.run(); capsys.readouterr()
    assert report['status'] == 'pass' and report['source_and_inputs_rehashed_after']
    assert calls.count('identity') == 15 and calls.count('source') == 2
    cohort = json.loads((output/'cohort.json').read_bytes())
    assert len(cohort['records']) == 32 and cohort['reference_geometry_exposed'] is False
    assert report['cohort_identity'] == p.pin((output/'cohort.json').read_bytes())
    assert set(x.name for x in output.iterdir()) == {'report.json', 'cohort.json'}
    assert all(not x.stat().st_mode & 0o222 for x in (output, *output.iterdir()))


def test_deadline_fail_does_not_publish_cohort(monkeypatch, tmp_path, capsys):
    output, _ = fake_run(monkeypatch, tmp_path, late=True)
    with pytest.raises(ValueError, match='deadline'):
        p.run()
    capsys.readouterr(); report = json.loads((output/'report.json').read_bytes())
    assert report['status'] == 'fail' and report['error_type'] == 'TimeoutError'
    assert not (output/'cohort.json').exists()


def test_source_change_closes_before_cohort_no_payload_in_report(monkeypatch, tmp_path, capsys):
    output, _ = fake_run(monkeypatch, tmp_path, changed=True)
    with pytest.raises(ValueError, match='Source changed'):
        p.run()
    capsys.readouterr(); report = json.loads((output/'report.json').read_bytes())
    assert report['status'] == 'fail' and not (output/'cohort.json').exists()
    assert 'Source changed' not in (output/'report.json').read_text()


def test_insufficient_metadata_closes_without_partial_cohort(monkeypatch, tmp_path, capsys):
    output, _ = fake_run(monkeypatch, tmp_path, enough=False)
    report = p.run(); capsys.readouterr()
    assert report['status'] == 'pass' and report['decision'] == 'INCONCLUSIVE_CLOSED_NO_RGB'
    assert report['counts']['unique_author_photo_md5_slots'] == 31 and not (output/'cohort.json').exists()


def test_late_directory_seal_does_not_leave_pass_report(monkeypatch, tmp_path, capsys):
    output, _ = fake_run(monkeypatch, tmp_path)
    original = Path.chmod; clock = [0.]
    monkeypatch.setattr(p.time, 'monotonic', lambda:clock[0])
    def chmod(path, *a, **kw):
        result = original(path, *a, **kw)
        if path == output:
            clock[0] = 181.
        return result
    monkeypatch.setattr(Path, 'chmod', chmod)
    report = p.run(); capsys.readouterr()
    stored = json.loads((output/'report.json').read_bytes())
    assert report['status'] == stored['status'] == 'fail' and stored['error_type'] == 'TimeoutError'
    assert stored['elapsed_seconds'] == 181. and not output.stat().st_mode & 0o222


def test_final_receipt_fsync_late_cannot_leave_pass(monkeypatch, tmp_path, capsys):
    output, _ = fake_run(monkeypatch, tmp_path)
    original = p.os.fsync; clock = [0.]; calls = [0]
    monkeypatch.setattr(p.time, 'monotonic', lambda:clock[0])
    def fsync(fd):
        result = original(fd); calls[0] += 1
        # Cohort sync, initial report sync, then final report sync.
        if calls[0] == 3:
            clock[0] = 181.
        return result
    monkeypatch.setattr(p.os, 'fsync', fsync)
    report = p.run(); capsys.readouterr()
    assert report['status'] == json.loads((output/'report.json').read_bytes())['status'] == 'fail'


def test_collect_tiny_publisher_csv_preserves_geometry_and_excludes_ids(tmp_path):
    import csv
    boxes = []
    def box(i, cls, xy):
        return dict(ImageID=f'{i:016x}', LabelName=cls, XMin=str(xy[0]), YMin=str(xy[1]), XMax=str(xy[2]), YMax=str(xy[3]),
                    IsGroupOf='0', IsDepiction='0', IsInside='0')
    relations = []
    for i in range(1000, 1033):
        rows = [box(i, '/m/man', (0.,0.,.3,1.)), box(i, '/m/man', (.6,0.,1.,1.)),
                box(i, '/m/cup', (.1,.1,.2,.2)), box(i, '/m/book', (.7,.1,.8,.2))]
        boxes.extend(rows)
        rel = dict(ImageID=f'{i:016x}', LabelName1='/m/man', LabelName2='/m/cup', RelationshipLabel='holds')
        for suffix, row in (('1', rows[0]), ('2', rows[2])):
            rel.update({k+suffix:row[k] for k in ('XMin','YMin','XMax','YMax')})
        relations.append(rel)
    values = dict(boxes=boxes, relations=relations,
                  triplets=[dict(LabelName1='/m/man', LabelName2='/m/cup', RelationshipLabel='holds')],
                  metadata=[metadata(i) for i in range(1000,1033)])
    paths = {}
    for name, rows in values.items():
        path = tmp_path/(name+'.csv'); paths[name] = path
        with path.open('w',newline='') as stream:
            writer = csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    excluded = empty_exclusions(); excluded['ids'].add(f'{1000:016x}')
    calls = []
    selected, counts = p.collect(paths, dict(human_classes=['/m/man'], body_part_classes=['/m/hand']), excluded,
                                lambda:calls.append('check'))
    assert counts['new_relation_images'] == counts['geometry_orientation_eligible'] == 32 and len(selected) == 32
    assert f'{1000:016x}' not in {r['publisher_metadata']['ImageID'] for r in selected} and calls
    assert all(set(r) == {'slot','split','publisher_metadata'} for r in selected)


def test_configuration_rejects_changed_helpers_and_network_scope(monkeypatch):
    root = Path(__file__).resolve().parents[1]; cfg = json.loads((root/p.CONFIG).read_text())
    source = {'helpers':{p.CONFIG:p.pin((root/p.CONFIG).read_bytes()), **cfg['reused_helper_pins']}}
    def local_pinned(path, expected, maximum):
        raw = Path(path).read_bytes(); assert len(raw) <= maximum and p.pin(raw) == expected
        return p.rt.strict(raw)
    monkeypatch.setattr(p.rt, 'pinned', local_pinned)
    assert p.configuration(root, source)['network_allowed'] is False
    bad = deepcopy(source); bad['helpers'][p.HELPERS[4]]['sha256'] = '0'*64
    with pytest.raises(ValueError, match='helper bytes'):
        p.configuration(root, bad)
    original = p.rt.pinned
    def bad_scope(path, expected, maximum):
        value = original(path, expected, maximum); value['network_allowed'] = True; return value
    monkeypatch.setattr(p.rt, 'pinned', bad_scope)
    with pytest.raises(ValueError, match='metadata-only'):
        p.configuration(root, source)
