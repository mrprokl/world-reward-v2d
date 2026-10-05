"""Procedural publisher metadata/HTTP controls; no real images or accuracy."""
from copy import deepcopy
import base64
import hashlib
import json
from pathlib import Path
import stat
import time

import pytest

import openimages_joint_pair_acquire as acq


def jpeg(width=16, height=12):
    # Header-only synthetic bytes, deliberately not a decoded real image.
    frame = bytes([8])+height.to_bytes(2, 'big')+width.to_bytes(2, 'big')+bytes([3])+bytes([1, 0x11, 0, 2, 0x11, 0, 3, 0x11, 0])
    return b'\xff\xd8\xff\xe0\x00\x04AB\xff\xc0\x00\x11'+frame+b'\xff\xd9'


def metadata(index, raw=None):
    raw = jpeg() if raw is None else raw
    return dict(ImageID=f'{index:016x}', OriginalLandingURL=f'https://www.flickr.com/photos/author{index}/{index+100}/',
        OriginalURL=f'https://live.staticflickr.com/10/{index}_unique.jpg', OriginalSize=str(len(raw)),
        OriginalMD5=base64.b64encode(hashlib.md5(raw+str(index).encode()).digest()).decode(), Rotation='0.0',
        Author=f'author{index}', AuthorProfileURL=f'https://www.flickr.com/people/author{index}/', Title='Original title')


def census(count=80):
    return [dict(image_id=r['ImageID'], eligible=True, publisher_metadata=r,
                 positive_pairs=999, countable_person_boxes=999) for r in (metadata(i) for i in range(count))]


def empty_exclusions():
    return {k: set() for k in ('ids', 'md5', 'url', 'photo')}


def cfg(slots=1):
    return dict(slots=slots, workers=4, max_creator_page_bytes=2 << 20, max_image_bytes=16 << 20,
                max_total_image_bytes=1 << 30, request_timeout=15, max_decoded_pixels=16 << 20)


def rights(row, **change):
    obj = dict(**{'@type': 'ImageObject'}, acquireLicensePage=row['OriginalLandingURL'], license=acq.LICENSE_URL,
               author=dict(name=row['Author']))
    obj.update(change)
    return ('<script type="application/ld+json">'+json.dumps(obj)+'</script>').encode()


def slot(row, slot=0, split='DEV'):
    return dict(slot=slot, split=split, publisher_metadata=row)


def test_selection_deterministic_and_only_publisher_metadata():
    rows = census(); original = deepcopy(rows)
    a = acq.select_cohort(rows, empty_exclusions()); b = acq.select_cohort(list(reversed(rows)), empty_exclusions())
    assert a == b and rows == original
    assert len(a) == 64 and sum(r['split'] == 'DEV' for r in a) == 32
    assert [r['slot'] for r in a] == list(range(64))
    assert all(set(r) == {'slot', 'split', 'publisher_metadata'} for r in a)
    assert all(set(r['publisher_metadata']) == set(acq.METADATA_KEYS) for r in a)
    assert 'positive_pairs' not in json.dumps(a)
    ranks = [hashlib.sha256((acq.HASH_NAMESPACE+r['publisher_metadata']['ImageID']).encode()).hexdigest() for r in a]
    assert ranks == sorted(ranks)


@pytest.mark.parametrize('duplicate', ['author', 'md5', 'url', 'photo'])
def test_duplicate_source_not_another_slot(duplicate):
    rows = census(4); first, other = rows[0]['publisher_metadata'], rows[1]['publisher_metadata']
    key = dict(author='AuthorProfileURL', md5='OriginalMD5', url='OriginalURL', photo='OriginalLandingURL')[duplicate]
    other[key] = first[key]
    if duplicate == 'author':
        other[key] = first[key].replace('/people/', '/photos/').rstrip('/')
    if duplicate == 'photo':
        other[key] = first[key].replace('/author0/', '/different/')
    selected = acq.select_cohort(rows, empty_exclusions(), slots=3, dev=1)
    ids = {r['publisher_metadata']['ImageID'] for r in selected}
    assert not {first['ImageID'], other['ImageID']} <= ids


@pytest.mark.parametrize('identity', ['ids', 'md5', 'url', 'photo'])
def test_old_photo_cannot_reenter_with_new_image_id(identity):
    rows = census(4); row = rows[0]['publisher_metadata']; excluded = empty_exclusions()
    excluded[identity].add(dict(ids=row['ImageID'], md5=row['OriginalMD5'], url=row['OriginalURL'],
                                photo=acq.photo_identity(row['OriginalLandingURL']))[identity])
    selected = acq.select_cohort(rows, excluded, slots=3, dev=1)
    assert row['ImageID'] not in {r['publisher_metadata']['ImageID'] for r in selected}


def test_old_exclusions_use_only_original_selected_metadata():
    rows = [metadata(200), metadata(201)]
    result = acq.old_exclusions([(2, dict(selected_ids=[r['ImageID'] for r in rows], public_metadata=rows,
                                         reference_boxes='never needed'))])
    assert len(result['ids']) == len(result['photo']) == len(result['md5']) == 2


@pytest.mark.parametrize('change', ['empty_author', 'empty_md5', 'ineligible'])
def test_unusable_selection_does_not_fetch_or_replace(change):
    rows = census(3)
    if change == 'ineligible':
        rows[0]['eligible'] = False
    else:
        rows[0]['publisher_metadata'][dict(empty_author='AuthorProfileURL', empty_md5='OriginalMD5')[change]] = ''
    with pytest.raises(ValueError, match='Insufficient'):
        acq.select_cohort(rows, empty_exclusions(), slots=3, dev=1)


@pytest.mark.parametrize('change', ['duplicate_id', 'wrong_id', 'rotation', 'bad_md5', 'bad_author'])
def test_invalid_original_identity_rejected(change):
    rows = census(3)
    if change == 'duplicate_id':
        rows.append(deepcopy(rows[0]))
    elif change == 'wrong_id':
        rows[0]['publisher_metadata']['ImageID'] = '0'*16
        rows[0]['image_id'] = '1'*16
    elif change == 'rotation':
        rows[0]['publisher_metadata']['Rotation'] = '-1.0'
    elif change == 'bad_md5':
        rows[0]['publisher_metadata']['OriginalMD5'] = 'not-md5'
    else:
        rows[0]['publisher_metadata']['AuthorProfileURL'] = 'https://www.flickr.com.evil/people/a/'
    with pytest.raises(ValueError):
        acq.select_cohort(rows, empty_exclusions(), slots=2, dev=1)


@pytest.mark.parametrize('width,height', [(16, 12), (1, 1), (4096, 4096)])
def test_header_dimension_bound_without_decode_claim(width, height):
    result = acq.jpeg_header(jpeg(width, height), 16 << 20)
    assert result == dict(width=width, height=height, channels=3, header_only=True, decoded_content_verified=False)


@pytest.mark.parametrize('raw', [b'', b'not-jpeg', b'\xff\xd8\xff\xda', jpeg(0, 5), jpeg(4097, 4096),
                               b'\xff\xd8\xff\xe0\xff\xffshort'])
def test_invalid_or_oversized_header_not_repaired(raw):
    with pytest.raises(ValueError):
        acq.jpeg_header(raw, 16 << 20)


def test_rights_exact_license_author_and_landing():
    row = metadata(1); original = deepcopy(row)
    result = acq.creator_rights(rights(row), row)
    assert result['individual_creator_declaration_verified'] and result['license'] == 'CC-BY-2.0'
    assert row == original and result['creator_page']['sha256'] == acq.pin(rights(row))['sha256']
    assert 'creator_ld_json' not in result


@pytest.mark.parametrize('change', [dict(license='https://creativecommons.org/licenses/by-nc/4.0/'),
    dict(author=dict(name='another')), dict(acquireLicensePage='https://www.flickr.com/photos/other/123/')])
def test_rights_mismatch_is_not_access_permission(change):
    row = metadata(1)
    with pytest.raises(ValueError):
        acq.creator_rights(rights(row, **change), row)


def test_ambiguous_creator_declarations_fail_closed():
    row = metadata(1)
    with pytest.raises(ValueError):
        acq.creator_rights(rights(row)*2, row)


def test_acquired_slot_exact_raw_and_readonly(tmp_path, capsys):
    raw = jpeg(); row = metadata(10, raw); row['OriginalMD5'] = base64.b64encode(hashlib.md5(raw).digest()).decode()
    before = deepcopy(row); calls = []
    def request(url, maximum, deadline, timeout):
        calls.append((url, maximum, timeout))
        return rights(row) if url == row['OriginalLandingURL'] else raw
    result = acq.acquire_slot(slot(row), tmp_path, cfg(), time.monotonic()+10, request=request)
    folder = tmp_path/row['ImageID']
    assert result['status'] == 'acquired' and len(calls) == 2 and row == before
    assert (folder/'rgb.jpg').read_bytes() == raw and result['image_pin'] == acq.pin(raw)
    assert (folder.stat().st_mode & 0o777) == 0o555
    assert all(stat.S_IMODE(p.stat().st_mode) == 0o444 for p in folder.iterdir())
    stored = json.loads((folder/'record.json').read_bytes())
    assert stored == {k: v for k, v in result.items() if k != 'record_pin'}
    assert capsys.readouterr().out == ''


@pytest.mark.parametrize('failure', ['rights', 'missing', 'md5', 'header', 'deadline', 'oversize', 'url', 'rotation'])
def test_failed_slot_is_retained_without_retry_or_rgb_substitution(tmp_path, failure):
    raw = jpeg(); row = metadata(10, raw); row['OriginalMD5'] = base64.b64encode(hashlib.md5(raw).digest()).decode(); calls = []
    if failure == 'oversize':
        row['OriginalSize'] = str((16 << 20)+1)
    if failure == 'url':
        row['OriginalURL'] = 'https://live.staticflickr.com.evil/a.jpg'
    if failure == 'rotation':
        row['Rotation'] = '-1.0'
    if failure == 'header':
        raw = b'not-jpeg'; row['OriginalSize'] = str(len(raw)); row['OriginalMD5'] = base64.b64encode(hashlib.md5(raw).digest()).decode()
    def request(url, *_args):
        calls.append(url)
        if failure == 'missing':
            raise OSError('scripted offline fixture')
        if url == row['OriginalLandingURL']:
            return rights(row, license='unknown') if failure == 'rights' else rights(row)
        return b'X'*len(raw) if failure == 'md5' else raw
    deadline = time.monotonic()-1 if failure == 'deadline' else time.monotonic()+10
    result = acq.acquire_slot(slot(row), tmp_path, cfg(), deadline, request=request)
    folder = tmp_path/row['ImageID']
    assert result['status'] == 'unavailable' and 'record.json' in {p.name for p in folder.iterdir()}
    assert not (folder/'rgb.jpg').exists() and len(calls) <= 2 and len(calls) == len(set(calls))
    assert not folder.stat().st_mode & 0o222 and 'error_type' in result


def test_sealed_cohort_before_first_request_and_complete_failed_slots(tmp_path):
    cohort_dir = tmp_path/'selection'; cohort_dir.mkdir(); cohort_path = cohort_dir/'cohort.json'
    selected = [slot(metadata(i), i, 'DEV' if i == 0 else 'TEST') for i in range(2)]
    cohort_pin = acq.publish(cohort_path, dict(records=selected)); cohort_dir.chmod(0o555)
    output = tmp_path/'output'; output.mkdir(); calls = []
    def request(url, *_args):
        assert cohort_path.exists() and acq.rt.identity(cohort_path) == cohort_pin
        assert not cohort_dir.stat().st_mode & 0o222
        calls.append(url); raise OSError('scripted missing')
    result = acq.acquire_cohort(selected, output, cfg(2), time.monotonic()+10, cohort_path, cohort_pin, request=request)
    assert [r['slot'] for r in result] == [0, 1] and len(calls) == 2
    assert all(r['status'] == 'unavailable' for r in result)


def test_unsealed_or_tampered_cohort_blocks_all_requests(tmp_path):
    cohort_path = tmp_path/'cohort.json'; cohort_pin = acq.publish(cohort_path, dict(records=[])); calls = []
    def request(*args):
        calls.append(args); raise AssertionError('must not run')
    with pytest.raises(ValueError, match='Sealed cohort'):
        acq.acquire_cohort([], tmp_path, cfg(), time.monotonic()+10, cohort_path, cohort_pin, request=request)
    assert calls == []


def test_wrapper_no_gpu_and_strict_single_fresh_mode():
    wrapper = (Path(__file__).resolve().parents[1]/'infra/run_openimages_joint_pair_acquire.sh').read_text()
    assert '310s' in wrapper and '--kill-after=5s' in wrapper and 'env -i' in wrapper
    assert 'python3 -I -B' in wrapper and 'world-reward-ncc-h100-02' in wrapper
    assert 'docker' not in wrapper and 'set +x' in wrapper


def test_current_config_and_frozen_lineage_are_exact():
    code = Path(__file__).resolve().parents[1]
    # Working-tree config is prospective and writable; no immutable-source claim.
    original = acq.rt.identity
    def working_identity(path, maximum, **kwargs):
        return original(path, maximum, readonly=False)
    from unittest.mock import patch
    with patch.object(acq.rt, 'identity', working_identity):
        value, _ = acq.configuration(code)
    assert value['slots'] == 64 and value['dev'] == value['test'] == 32
    assert value['census']['sha256'] == acq.CENSUS_PIN['sha256'] and value['census']['producer_revision'] == acq.CENSUS_REV
    assert value['no_replacements'] is True and value['quality_verified'] is False


def test_config_retuning_rejected(tmp_path):
    source = Path(__file__).resolve().parents[1]/acq.CONFIG
    target = tmp_path/acq.CONFIG; target.parent.mkdir()
    value = json.loads(source.read_bytes()); value['slots'] = 63
    target.write_text(json.dumps(value)); target.chmod(0o444)
    with pytest.raises(ValueError, match='Exact predeclared'):
        acq.configuration(tmp_path)


@pytest.mark.parametrize('mismatch', [None, 'source', 'after', 'old_pin'])
def test_authentication_requires_actual_old_closure_and_metadata_pins(tmp_path, monkeypatch, mismatch):
    revision = 'a'*40; code = tmp_path/'jobs'/revision/acq.ENTRY/'code'
    source = dict(producer_revision=revision, helpers={})
    old_code = tmp_path/'jobs'/acq.CENSUS_REV/acq.CENSUS_ENTRY/'code'
    old_source = dict(producer_revision=acq.CENSUS_REV, helpers={acq.CENSUS_CONFIG: dict(bytes=10, sha256='c'*64)})
    old_paths = [tmp_path/'openimages_holds_census_v1/census_v2.json',
                 tmp_path/'openimages_holds_census_fresh128_v1/census.json']
    old_files = {}
    for path, start, count in zip(old_paths, (1000, 2000), (16, 128)):
        path.parent.mkdir(); rows = [metadata(start+i) for i in range(count)]
        raw = acq.encode(dict(selected_ids=[r['ImageID'] for r in rows], public_metadata=rows))
        path.write_bytes(raw); path.chmod(0o444); old_files[str(path)] = acq.pin(raw)
    old_config = dict(files=old_files)
    census_report = dict(schema='world_reward.openimages_joint_pair_census.v1', status='pass',
        producer_revision=acq.CENSUS_REV, source_binding=deepcopy(old_source), frozen_inputs=old_files,
        selection_performed=False, challenge_inputs_used=False, rgb_read=False, models_loaded=False,
        gpu_used=False, network_used=False, source_and_inputs_rehashed_after=True,
        stage='new_joint_pair_metadata_feasibility', counts=dict(old_ids_excluded=144), records=census())
    if mismatch == 'source':
        census_report['source_binding']['producer_revision'] = 'b'*40
    if mismatch == 'after':
        census_report['source_and_inputs_rehashed_after'] = False
    if mismatch == 'old_pin':
        old_paths[0].chmod(0o644); old_paths[0].write_bytes(b'changed'); old_paths[0].chmod(0o444)
    config = dict(census=dict(path='results/report.json')); config_pin = dict(bytes=20, sha256='d'*64)
    monkeypatch.setattr(acq, 'configuration', lambda _: (config, config_pin))
    calls = []
    def source_read(root, actual_code, rev, entry, helpers):
        calls.append((actual_code, entry, helpers))
        return source if actual_code == code else old_source
    monkeypatch.setattr(acq.rt, 'source', source_read)
    def pinned(path, expected, maximum=1 << 20):
        if path == tmp_path/'results/report.json':
            assert expected == acq.CENSUS_PIN
            return census_report
        assert path == old_code/acq.CENSUS_CONFIG and expected == old_source['helpers'][acq.CENSUS_CONFIG]
        return old_config
    monkeypatch.setattr(acq.rt, 'pinned', pinned)
    if mismatch:
        with pytest.raises(ValueError):
            acq.authenticate(tmp_path, code, revision)
    else:
        cfg_out, records, excluded, current, historical, frozen = acq.authenticate(tmp_path, code, revision)
        assert current == source and historical == old_source and records == census_report['records']
        assert len(excluded['ids']) == 144 and len(frozen) == 4 and cfg_out == config
        assert calls == [(code, acq.ENTRY, acq.HELPERS), (old_code, acq.CENSUS_ENTRY, acq.CENSUS_HELPERS)]


def test_fetch_disabled_proxy_and_redirect_and_exact_bound(monkeypatch):
    calls = []
    class Response:
        status = 200
        url = 'https://www.flickr.com/photos/a/1/'
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return None
        def read(self, maximum):
            calls.append(maximum)
            return b'x'*(maximum if len(calls) == 1 else 0)
    class Opener:
        def open(self, request, timeout):
            assert timeout <= 15 and request.full_url == Response.url
            return Response()
    def build(*handlers):
        assert handlers[0].proxies == {} and isinstance(handlers[1], acq.ExactRedirect)
        return Opener()
    monkeypatch.setattr(acq.urllib.request, 'build_opener', build)
    with pytest.raises(ValueError, match='byte bound'):
        acq.fetch(Response.url, 10, time.monotonic()+30)
    with pytest.raises(ValueError, match='Redirect'):
        acq.ExactRedirect().redirect_request(None)
