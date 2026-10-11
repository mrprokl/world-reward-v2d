"""Tiny manufactured receipts and loopback byte ranges; no Azure/media files."""
from io import BytesIO
import json
import os
from pathlib import Path
import sys
import threading
import time
import urllib.error
import urllib.request

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'infra'))
import end2end_preview_stream as stream
import full4d_stream as legacy


def receipt():
    rev = 'a' * 40
    identity = dict(bytes=20, sha256='b' * 64)
    binding = dict(producer_revision=rev, entries=12, closure_sha256='c' * 64,
        helpers={name: identity.copy() for name in stream.HELPERS},
        markers={name: dict(bytes=size, sha256='d' * 64)
                 for name, size in [('revision', 41), ('source-sha256', 65)]})
    states = [dict(episode=9, status='complete', QA_passed=True),
              dict(episode=1, status='not_reconstructed_in_this_ablation', reason='upstream full-pose unsupported'),
              dict(episode=14, status='complete', QA_passed=False),
              dict(episode=7, status='failed', reason='automatic pose unavailable', phase='native_initialization')]
    rows = [dict(name=f'full4d-{rev}/episode_{ep:06d}.{ext}', mime=mime, etag='"0xABC123"',
                 episode_index=ep, **identity)
            for ep in stream.COMPLETE for ext, mime in [('mp4', 'video/mp4'), ('jpg', 'image/jpeg')]]
    value = dict(schema=stream.SCHEMA, status='pass', producer_revision=rev, render_revision='e' * 40,
        candidate_revision='f' * 40, baseline_revision=stream.BASELINE, render_report_pin=identity.copy(),
        cohort=stream.COHORT.copy(), states=states, files=rows, source_binding=binding,
        endpoint=legacy.ENDPOINT, private_container_verified=True, public_access_changed=False,
        account_keys_used=False, quality_verified=False, heavy_data_uploaded=False)
    return value, rev


def continuation_receipt():
    value, rev = receipt(); value.update(schema=stream.CONTINUATION_SCHEMA, candidate_kind='continuation')
    value['states'][0]['candidate_status'] = 'accepted_native_continuation'
    value['states'][2]['candidate_status'] = 'dynamic_A_fallback_no_improvement'
    return value, rev


def test_strict_v2_native_continuation_and_fallback_labels_preserve_legacy_v1():
    value, rev = continuation_receipt()
    states, allowed = stream.records(value, rev, candidate_kind='continuation')
    assert len(allowed) == 4 and len(states) == 4
    page = stream.index_html(states).decode()
    assert 'C contact-continuation' in page and 'Baseline fallback / no gain' in page and 'Native joint' not in page
    assert 'non vérifiés' in page
    with pytest.raises(ValueError): stream.records(value, rev, candidate_kind='joint')
    old, _ = receipt()
    stream.records(old, rev, candidate_kind='joint')
    with pytest.raises(ValueError): stream.records(old, rev, candidate_kind='continuation')


@pytest.mark.parametrize('bad', ['kind', 'missing_kind', 'extra', 'missing_status', 'false_status', 'legacy_inject'])
def test_v2_continuation_fields_are_not_arbitrary_optional_metadata(bad):
    value, rev = continuation_receipt()
    if bad == 'kind': value['candidate_kind'] = 'oracle'
    elif bad == 'missing_kind': del value['candidate_kind']
    elif bad == 'extra': value['gain_verified'] = True
    elif bad == 'missing_status': del value['states'][0]['candidate_status']
    elif bad == 'false_status': value['states'][0]['candidate_status'] = 'complete'
    elif bad == 'legacy_inject': value['schema'] = stream.SCHEMA
    with pytest.raises(ValueError): stream.records(value, rev)


@pytest.mark.parametrize('mutation', ['schema', 'producer', 'render', 'candidate', 'baseline', 'endpoint',
    'private', 'public', 'keys', 'quality', 'heavy', 'extra_field', 'cohort', 'order', 'missing_state',
    'filter_failure', 'complete_extra', 'qa_not_bool', 'unknown_status', 'reason_unbounded',
    'missing_pair', 'duplicate', 'extra_manifest', 'wrong_name', 'secret_query', 'wrong_mime',
    'video_cap', 'poster_cap', 'bytes_bool', 'sha', 'etag', 'episode_bool', 'render_pin',
    'binding_revision', 'binding_hash', 'helper_missing', 'helper_pin', 'marker_size'])
def test_invalid_receipts_rejected_before_any_reader(mutation):
    v, rev = receipt()
    if mutation == 'schema': v['schema'] = 'world_reward.full4d_publish.v1'
    elif mutation == 'producer': v['producer_revision'] = '0' * 40
    elif mutation == 'render': v['render_revision'] = '0' * 40
    elif mutation == 'candidate': v['candidate_revision'] = '0' * 40
    elif mutation == 'baseline': v['baseline_revision'] = '0' * 40
    elif mutation == 'endpoint': v['endpoint'] += '?sig=manufactured-secret'
    elif mutation == 'private': v['private_container_verified'] = False
    elif mutation == 'public': v['public_access_changed'] = True
    elif mutation == 'keys': v['account_keys_used'] = True
    elif mutation == 'quality': v['quality_verified'] = True
    elif mutation == 'heavy': v['heavy_data_uploaded'] = True
    elif mutation == 'extra_field': v['accessToken'] = 'manufactured-secret'
    elif mutation == 'cohort': v['cohort'][1] = 2
    elif mutation == 'order': v['states'].reverse()
    elif mutation == 'missing_state': v['states'].pop()
    elif mutation == 'filter_failure': v['states'][2] = dict(episode=14, status='failed', reason='QA failed')
    elif mutation == 'complete_extra': v['states'][1] = dict(episode=1, status='complete', QA_passed=True)
    elif mutation == 'qa_not_bool': v['states'][0]['QA_passed'] = 1
    elif mutation == 'unknown_status': v['states'][1]['status'] = 'pending'
    elif mutation == 'reason_unbounded': v['states'][1]['reason'] = 'a' * 513
    elif mutation == 'missing_pair': v['files'].pop()
    elif mutation == 'duplicate': v['files'][3] = v['files'][0].copy()
    elif mutation == 'extra_manifest': v['files'].append(dict(name=f'full4d-{rev}/manifest.json'))
    elif mutation == 'wrong_name': v['files'][0]['name'] = v['files'][0]['name'].replace(rev, 'f' * 40)
    elif mutation == 'secret_query': v['files'][0]['name'] += '?sig=manufactured-secret'
    elif mutation == 'wrong_mime': v['files'][0]['mime'] = 'application/octet-stream'
    elif mutation == 'video_cap': v['files'][0]['bytes'] = legacy.MAX_VIDEO + 1
    elif mutation == 'poster_cap': v['files'][1]['bytes'] = legacy.MAX_POSTER + 1
    elif mutation == 'bytes_bool': v['files'][0]['bytes'] = True
    elif mutation == 'sha': v['files'][0]['sha256'] = 'x' * 64
    elif mutation == 'etag': v['files'][0]['etag'] = '"bad\r\nheader"'
    elif mutation == 'episode_bool': v['files'][0]['episode_index'] = True
    elif mutation == 'render_pin': v['render_report_pin']['bytes'] = (4 << 20) + 1
    elif mutation == 'binding_revision': v['source_binding']['producer_revision'] = 'f' * 40
    elif mutation == 'binding_hash': v['source_binding']['closure_sha256'] = 'x' * 64
    elif mutation == 'helper_missing': v['source_binding']['helpers'].pop('infra/end2end_preview.py')
    elif mutation == 'helper_pin': v['source_binding']['helpers']['infra/end2end_preview.py']['bytes'] = 0
    elif mutation == 'marker_size': v['source_binding']['markers']['revision']['bytes'] = 40
    with pytest.raises(ValueError):
        stream.records(v, rev, render_revision='e' * 40, candidate_revision='f' * 40)


def test_receipt_protocols_separate_and_qa_failure_never_filtered():
    value, rev = receipt()
    states, allowed = stream.records(value, rev)
    assert states == value['states'] and states[2]['QA_passed'] is False
    assert set(allowed) == {'/episode-9.mp4', '/episode-9.jpg', '/episode-14.mp4', '/episode-14.jpg'}
    with pytest.raises(ValueError): legacy.records(value, rev)
    with pytest.raises(ValueError): legacy.partial_records(value, rev)


def test_only_tiny_canonical_sealed_receipt_read(tmp_path):
    value, rev = receipt()
    path = tmp_path / 'publication.json'
    path.write_text(json.dumps(value)); path.chmod(0o444)
    assert stream.load_receipt(path, rev)[:2] == stream.records(value, rev)
    assert len(stream.load_receipt(path, rev)[2]) == 64
    path.chmod(0o644)
    with pytest.raises(ValueError): stream.load_receipt(path, rev)
    path.chmod(0o444)
    link = tmp_path / 'link.json'; link.symlink_to(path)
    with pytest.raises(ValueError): stream.load_receipt(link, rev)
    link.unlink(); os.link(path, link)
    with pytest.raises(ValueError): stream.load_receipt(path, rev)
    assert {p.name for p in tmp_path.iterdir()} == {'publication.json', 'link.json'}


@pytest.mark.parametrize('raw', [b'{"schema":1,"schema":2}', b'{"x":NaN}', b'x' * (stream.MAX_RECEIPT + 1)])
def test_bad_or_oversize_metadata_rejected(tmp_path, raw):
    path = tmp_path / 'publication.json'; path.write_bytes(raw); path.chmod(0o444)
    with pytest.raises(ValueError): stream.load_receipt(path, 'a' * 40)


class Reader:
    def __init__(self): self.heads = []; self.opens = []
    def verify_head(self, row): self.heads.append(row['name'])
    def open(self, row, selected):
        self.verify_head(row); self.opens.append((row['name'], selected))
        value = b'0123456789abcdefghij'
        if selected is not None: value = value[selected[0]:selected[1] + 1]
        return BytesIO(value), len(value)


def test_loopback_html_three_columns_all_statuses_and_no_background_reads(tmp_path):
    value, rev = receipt(); states, allowed = stream.records(value, rev)
    reader = Reader(); server = stream.make_server(states, allowed, reader=reader)
    worker = threading.Thread(target=server.serve_forever, daemon=True); worker.start()
    url = f'http://127.0.0.1:{server.server_port}'
    try:
        with urllib.request.urlopen(url + '/') as response:
            page = response.read().decode()
            assert response.headers['Content-Security-Policy'].startswith("default-src 'none'")
        assert page.count('<section class="clip-summary">') == 4
        assert '<video' not in page and '<source' not in page and '<img' not in page
        assert 'Un clip par page' in page and 'ORIGINAL' in page and 'BASELINE' in page and 'CORRECTIF' in page
        assert 'QA PASS — GT pending' in page and 'QA FAIL — not adopted' in page
        assert '2/4' in page and 'non vérifiés' in page and 'Épisode 01' in page and 'Épisode 07' in page
        assert 'poster=' not in page and '<img' not in page and '<script' not in page and 'autoplay' not in page
        assert '/episode-1.mp4' not in page and '/episode-7.mp4' not in page
        assert 'blob.core' not in page and 'sig=' not in page and 'Authorization' not in page
        assert reader.heads == reader.opens == []
        for ep in stream.COMPLETE:
            for focus in (None, 'original', 'baseline', 'candidate'):
                with urllib.request.urlopen(url + stream.view_path(ep, focus)) as response:
                    clip = response.read().decode()
                assert clip.count('<video ') == clip.count('preload="none"') == clip.count('<source ') == 1
                assert f'/episode-{ep}.mp4' in clip
                assert all(f'/episode-{other}.mp4' not in clip for other in stream.COHORT if other != ep)
                assert '<script' not in clip and 'autoplay' not in clip and 'poster=' not in clip and '<img' not in clip
                assert 'tremblements' in clip and 'contact main–objet' in clip and 'mouvement conservé' in clip
                assert 'QA PASS — GT pending' in clip if ep == 9 else 'QA FAIL — not adopted' in clip
                assert clip.count('aria-current="page"') == 1
                assert reader.heads == reader.opens == []
        for ep in (1, 7):
            with urllib.request.urlopen(url + stream.view_path(ep)) as response:
                incomplete = response.read().decode()
            assert '<video' not in incomplete and '<source' not in incomplete
            assert 'Cause enregistrée' in incomplete and 'sans clip remplacé' in incomplete
        assert reader.heads == reader.opens == []
        for path in ('/episode-1.mp4', '/episode-7.mp4', '/episode-9.mp4?sig=manufactured-secret',
                     '/view-episode-1/original', '/view-episode-9/unknown', '/view-episode-9?x=1'):
            with pytest.raises(urllib.error.HTTPError): urllib.request.urlopen(url + path)
        assert reader.opens == []
        request = urllib.request.Request(url + '/episode-14.mp4', headers={'Range': 'bytes=4-8'})
        with urllib.request.urlopen(request) as response:
            assert response.status == 206 and response.read() == b'45678'
            assert response.headers['Content-Range'] == 'bytes 4-8/20'
            assert response.headers['Cache-Control'] == 'no-store, private'
        assert len(reader.opens) == 1 and reader.opens[0][1] == (4, 8)
        request = urllib.request.Request(url + '/episode-9.jpg', method='HEAD')
        with urllib.request.urlopen(request) as response:
            assert response.status == 200 and response.read() == b''
        assert len(reader.opens) == 1 and len(reader.heads) == 2
        assert not tuple(tmp_path.iterdir())
    finally:
        server.shutdown(); server.server_close(); worker.join(timeout=5)


def test_foreign_origin_mutation_expired_viewer_and_bad_range_never_read_azure():
    value, rev = receipt(); states, allowed = stream.records(value, rev)
    reader = Reader(); server = stream.make_server(states, allowed, reader=reader)
    worker = threading.Thread(target=server.serve_forever, daemon=True); worker.start()
    url = f'http://127.0.0.1:{server.server_port}'
    try:
        requests = [urllib.request.Request(url + '/', headers={'Origin': 'https://foreign.example'}),
                    urllib.request.Request(url + '/episode-9.mp4', method='POST', data=b'no'),
                    urllib.request.Request(url + '/episode-9.mp4', headers={'Range': 'bytes=0-1,3-4'}),
                    urllib.request.Request(url + '/', headers={'Authorization': 'Bearer manufactured-secret'})]
        for request in requests:
            with pytest.raises(urllib.error.HTTPError): urllib.request.urlopen(request)
        server.deadline = time.monotonic() - 1
        for path in ('/', '/episode-9.mp4'):
            with pytest.raises(urllib.error.HTTPError): urllib.request.urlopen(url + path)
        assert reader.opens == reader.heads == []
    finally:
        server.shutdown(); server.server_close(); worker.join(timeout=5)


def sqp_receipt():
    value,rev=continuation_receipt();value.update(schema=stream.SQP_SCHEMA,candidate_kind='sqp')
    value['states'][0]['candidate_status']='accepted_native_pose_sqp'
    return value,rev


def test_strict_v3_sparse_SQP_uses_same_tiny_media_routes_and_honest_labels():
    v,rev=sqp_receipt();states,allowed=stream.records(v,rev,candidate_kind='sqp')
    assert len(allowed)==4 and len(states)==4
    page=stream.index_html(states).decode()
    assert 'C sparse pose-SQP' in page and 'Baseline fallback / no gain' in page
    assert 'C contact-continuation' not in page and 'Native joint' not in page
    assert 'non vérifiés' in page and '<video' not in page
    for wrong in ['joint','continuation']:
        with pytest.raises(ValueError):stream.records(v,rev,candidate_kind=wrong)


@pytest.mark.parametrize('bad',['schema','kind','status','missing_kind','missing_status','claim'])
def test_SQP_metadata_cannot_mislabel_continuation_or_claim_physical_gain(bad):
    v,rev=sqp_receipt()
    if bad=='schema':v['schema']=stream.CONTINUATION_SCHEMA
    elif bad=='kind':v['candidate_kind']='continuation'
    elif bad=='status':v['states'][0]['candidate_status']='accepted_native_continuation'
    elif bad=='missing_kind':del v['candidate_kind']
    elif bad=='missing_status':del v['states'][0]['candidate_status']
    elif bad=='claim':v['quality_verified']=True
    with pytest.raises(ValueError):stream.records(v,rev,candidate_kind='sqp')


def smoothing_receipt():
    value, rev = continuation_receipt()
    value.update(schema=stream.SMOOTHING_SCHEMA, candidate_kind='smoothing')
    value['states'][0]['candidate_status'] = 'accepted_rigid_projection_pending_independent_QA'
    return value, rev


def test_strict_v4_smoothing_is_a_diagnostic_not_native_articulation_or_physical_validation():
    value, rev = smoothing_receipt()
    states, allowed = stream.records(value, rev, candidate_kind='smoothing')
    assert len(allowed) == 4 and len(states) == 4
    page = stream.index_html(states).decode()
    assert 'Lissage contraint — diagnostic, GT à vérifier' in page
    assert 'Baseline fallback / no gain' in page and 'Native joint' not in page and 'sparse pose-SQP' not in page
    assert 'Qualité 4D et gain face à CARI4D non vérifiés' in page
    assert 'La couleur identifie la méthode, pas sa qualité' in page
    for kind in ('joint', 'continuation', 'sqp'):
        with pytest.raises(ValueError): stream.records(value, rev, candidate_kind=kind)
    for kind, factory in [('joint', receipt), ('continuation', continuation_receipt), ('sqp', sqp_receipt)]:
        other, rev = factory()
        with pytest.raises(ValueError): stream.records(other, rev, candidate_kind='smoothing')
        stream.records(other, rev, candidate_kind=kind)


@pytest.mark.parametrize('bad', ['schema', 'kind', 'missing_kind', 'missing_status', 'false_status', 'qa', 'claim'])
def test_v4_smoothing_has_exact_source_method_and_no_quality_claim(bad):
    value, rev = smoothing_receipt()
    if bad == 'schema': value['schema'] = stream.SQP_SCHEMA
    elif bad == 'kind': value['candidate_kind'] = 'sqp'
    elif bad == 'missing_kind': del value['candidate_kind']
    elif bad == 'missing_status': del value['states'][0]['candidate_status']
    elif bad == 'false_status': value['states'][0]['candidate_status'] = 'accepted_native_pose_sqp'
    elif bad == 'qa': value['states'][0]['QA_passed'] = 'PASS'
    elif bad == 'claim': value['quality_verified'] = True
    with pytest.raises(ValueError): stream.records(value, rev, candidate_kind='smoothing')


def test_focused_pages_use_exact_single_file_crop_without_retimming_or_geometry_changes():
    value, rev = smoothing_receipt(); states, _ = stream.records(value, rev)
    for focus, position in [('original', 'left center'), ('baseline', 'center center'), ('candidate', 'right center')]:
        page = stream.episode_html(states, 9, focus=focus).decode()
        assert f'class="focus-view focus-{focus}"' in page
        assert f'.focus-{focus} video{{object-position:{position}}}' in page
        assert 'aspect-ratio:8/7;object-fit:cover' in page
        assert page.count('src="/episode-9.mp4"') == 1 and '/episode-14.mp4' not in page
        assert 'sans nouvelle reconstruction ni retiming' in page and 'La couleur' not in page
        assert 'Lissage contraint — diagnostic, GT à vérifier' in page and 'pas encore adopté' in page
        assert '<script' not in page and 'autoplay' not in page and 'preload="none"' in page
    with pytest.raises(ValueError): stream.episode_html(states, 9, focus='oracle')
    with pytest.raises(ValueError): stream.episode_html(states, 29)


def test_recorded_failure_reason_is_escaped_and_all_clips_remain_navigable():
    value, rev = receipt()
    value['states'][1]['reason'] = '<script>alert("not trusted")</script>'
    states, _ = stream.records(value, rev)
    page = stream.episode_html(states, 1).decode()
    assert '<script' not in page and '&lt;script&gt;' in page and 'état est figé' in page
    assert '/episode-1.mp4' not in page
    page = stream.episode_html(states, 9).decode()
    assert 'href="/view-episode-7"' in page and 'href="/view-episode-1"' in page


def test_small_screen_legend_keeps_big_readable_labels_and_explicit_column_positions():
    value, rev = receipt(); states, _ = stream.records(value, rev)
    page = stream.episode_html(states, 9).decode()
    assert '@media(max-width:700px)' in page
    assert '.columns{grid-template-columns:1fr;gap:10px}' in page
    assert '.column-label h2{font-size:20px;margin:0}' in page
    assert '.column-label .position{display:inline}' in page
    for position in ('gauche', 'centre', 'droite'):
        assert f'<span class="position">· {position}</span>' in page
    assert 'max-width:1600px' in page and 'min-height:44px' in page
    assert '<nav class="view-nav" aria-label="Agrandir une vue">' in page


def test_all_new_pages_enforce_foreign_origin_expiry_and_same_existing_CSP_without_reader():
    value, rev = smoothing_receipt(); states, allowed = stream.records(value, rev)
    reader = Reader(); server = stream.make_server(states, allowed, reader=reader)
    worker = threading.Thread(target=server.serve_forever, daemon=True); worker.start()
    url = f'http://127.0.0.1:{server.server_port}'
    try:
        for path in ('/view-episode-9', '/view-episode-14/candidate', '/view-episode-1'):
            request = urllib.request.Request(url + path, method='HEAD')
            with urllib.request.urlopen(request) as response:
                assert response.read() == b''
                assert response.headers['Cache-Control'] == 'no-store, private'
                assert response.headers['Content-Security-Policy'] == (
                    "default-src 'none'; style-src 'unsafe-inline'; img-src 'self'; media-src 'self'; frame-ancestors 'none'")
            request = urllib.request.Request(url + path, headers={'Origin': 'https://foreign.example'})
            with pytest.raises(urllib.error.HTTPError): urllib.request.urlopen(request)
        server.deadline = time.monotonic() - 1
        for path in ('/view-episode-9', '/view-episode-14/candidate', '/view-episode-1'):
            with pytest.raises(urllib.error.HTTPError): urllib.request.urlopen(url + path)
        assert reader.opens == reader.heads == []
    finally:
        server.shutdown(); server.server_close(); worker.join(timeout=5)
