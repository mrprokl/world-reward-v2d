"""Publish only bounded full4D previews to the existing private Azure container.

Run on the Azure host, after the offline saved-render stage. Managed identity
is held in memory, never printed or persisted. There are no account keys, SAS
tokens, public ACL changes, servers, source videos, meshes or checkpoints.
The receipt contains nonsensitive blob names/ETags/SHA identities only. A
separate viewer may mint short-lived read-only user-delegation SAS in memory.
"""
from __future__ import annotations

import argparse
import base64
from email.utils import formatdate
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import time
import urllib.error
import urllib.parse
import urllib.request


ROOT = Path('/srv/scenesmith/world-reward')
ENDPOINT = 'https://stworldrewardresearch26.blob.core.windows.net/qa-previews'
MAX_VIDEO, MAX_POSTER, MAX_MANIFEST = 2_000_000, 100_000, 16_384
MAX_PARTIAL_RECEIPT = 64 << 10
SCHEMA = 'world_reward.full4d_publish.v1'
PARTIAL_SCHEMA = 'world_reward.full4d_partial_publish.v1'
COHORT = [9, 1, 14, 7]
PUBLISH_ENTRY, NUMERICAL_ENTRY = 'run_full4d_publish', 'run_gemini_full4d'
EXPORT_FILES = {'report.json', 'trajectory.npz', 'native_parameters.npz', 'target.npy', 'object_aligned.glb'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(path):
    path = Path(path)
    require(path.is_absolute() and path.resolve() == path
        and not any(p.is_symlink() for p in (path, *path.parents)), 'Canonical ordinary paths required')
    return path


def identity(path, maximum):
    path = canonical(path)
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1
        and not before.st_mode & 0o222 and 0 < before.st_size <= maximum,
        'Bounded read-only single-link preview artifact required')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    after = path.lstat()
    require(all(getattr(before, k) == getattr(after, k) for k in
        ('st_dev', 'st_ino', 'st_mode', 'st_nlink', 'st_size', 'st_mtime_ns', 'st_ctime_ns')),
        'Preview changed while hashing')
    return dict(bytes=before.st_size, sha256=digest.hexdigest())


def strict(raw):
    def pairs(rows):
        result = {}
        for key, value in rows:
            require(key not in result, 'Duplicate manifest field')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite metadata')))


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise RuntimeError('Private preview redirects forbidden')


class PrivatePreviews:
    """A fixed container only; bearer tokens and error bodies never escape."""
    def __init__(self):
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
        self.token = None
        self.expiry = 0

    def authorization(self):
        if time.time() + 300 >= self.expiry:
            try:
                url = ('http://169.254.169.254/metadata/identity/oauth2/token'
                    '?api-version=2018-02-01&resource=https%3A%2F%2Fstorage.azure.com%2F')
                request = urllib.request.Request(url, headers={'Metadata': 'true'})
                with self.opener.open(request, timeout=10) as response:
                    raw = response.read(32769)
                require(len(raw) <= 32768, 'Bounded managed identity reply required')
                value = strict(raw)
                require(value['token_type'].lower() == 'bearer'
                    and type(value['access_token']) is str and 0 < len(value['access_token']) < 16384,
                    'Azure managed identity required')
                self.token = value['access_token']
                self.expiry = int(value['expires_on'])
            except Exception:
                raise RuntimeError('Private Azure managed identity unavailable') from None
        return {'Authorization': 'Bearer '+self.token}

    def request(self, method, name=None, data=None, headers=None, *, container_acl=False):
        require(method in {'PUT', 'HEAD', 'GET', 'DELETE'}, 'Restricted preview REST method required')
        require(container_acl is False or method == 'HEAD' and name is None,
            'Only read-only container properties validation allowed')
        if container_acl:
            url = ENDPOINT+'?restype=container'
        else:
            require(type(name) is str and re.fullmatch(
                r'full4d-[0-9a-f]{40}/(?:episode_[0-9]{6}\.(?:mp4|jpg)|manifest\.json)', name),
                'Only revision-bound lightweight preview blobs allowed')
            url = ENDPOINT+'/'+urllib.parse.quote(name, safe='/')
        try:
            request = urllib.request.Request(url, data=data, method=method,
                headers={'x-ms-version': '2023-11-03', 'x-ms-date': formatdate(usegmt=True),
                    **self.authorization(), **(headers or {})})
            return self.opener.open(request, timeout=60)
        except urllib.error.HTTPError as error:
            raise RuntimeError(f'Private preview operation refused (HTTP{error.code})') from None
        except Exception:
            raise RuntimeError('Private preview operation failed') from None

    def require_private(self):
        with self.request('HEAD', container_acl=True) as response:
            require(response.status == 200 and not response.headers.get('x-ms-blob-public-access'),
                'Existing preview container must remain private')

    def head(self, name, expected, etag=None):
        with self.request('HEAD', name) as response:
            require(response.status == 200
                and int(response.headers.get('Content-Length', '-1')) == expected['bytes']
                and response.headers.get('x-ms-meta-sha256') == expected['sha256']
                and (etag is None or response.headers.get('ETag') == etag),
                'Private immutable preview HEAD identity differs')
            return response.headers['ETag']

    def upload(self, name, raw, mime, revision):
        pin = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        md5 = base64.b64encode(hashlib.md5(raw, usedforsecurity=False).digest()).decode()
        with self.request('PUT', name, raw, {'x-ms-blob-type': 'BlockBlob', 'If-None-Match': '*',
                'Content-Type': mime, 'Content-MD5': md5, 'x-ms-blob-content-type': mime,
                'x-ms-blob-content-disposition': 'inline', 'x-ms-blob-cache-control': 'private, no-store',
                'x-ms-meta-sha256': pin['sha256'], 'x-ms-meta-worldreward_revision': revision}) as response:
            require(response.status == 201, 'Exclusive private preview creation required')
            etag = response.headers['ETag']
        return dict(name=name, etag=etag, mime=mime, **pin)

    def delete_owned(self, row):
        self.head(row['name'], row, row['etag'])
        with self.request('DELETE', row['name'], headers={'If-Match': row['etag']}) as response:
            require(response.status == 202, 'Owned conditional preview deletion required')


def _preview_records(root, revision, episodes):
    """Shared saved-render checks; each public protocol separately binds cohort."""
    experiment = canonical(root)/'experiments'/f'full4d-v1-{revision}'
    files, reports = [], []
    for ep in episodes:
        directory = canonical(experiment/'videos'/f'episode_{ep:06d}')
        report_path = directory/'report.json'
        report_pin = identity(report_path, 262144)
        report = strict(report_path.read_bytes())
        expected = dict(schema='world_reward.full4d_video.v1', status='pass', producer_revision=revision,
            episode_index=ep, input_track='track_1', ground_truth_used=False, hand_labeled_test=False,
            oracle_modes=[], model_execution=False, optimizer_execution=False, metric_evaluation=False,
            quality_verified=False, original_geometry_unchanged=True, per_frame_alignment=False,
            per_frame_camera=False, per_frame_centring=False, source_rehashed_after=True, fps=30)
        require(all(type(report.get(k)) is type(v) and report[k] == v for k, v in expected.items())
            and type(report.get('original_frames')) is int and report['original_frames'] >= 3
            and report.get('frames_encoded') == report['original_frames']
            and report.get('original_frame_indices') == list(range(report['original_frames'])),
            'Complete saved-only full4D preview provenance required')
        reports.append(dict(episode_index=ep, path=str(report_path), identity=report_pin,
            original_frames=report['original_frames']))
        for extension, key, maximum, mime in (
                ('mp4', 'video', MAX_VIDEO, 'video/mp4'), ('jpg', 'poster', MAX_POSTER, 'image/jpeg')):
            path = directory/f'episode_{ep:06d}.{extension}'
            pin = identity(path, maximum)
            require(pin == report[key], 'Frozen full4D preview output bytes differ')
            files.append(dict(episode_index=ep, path=str(path),
                name=f'full4d-{revision}/episode_{ep:06d}.{extension}', mime=mime, **pin))
    return experiment, files, reports


def preview_records(root, revision, episodes):
    require(type(revision) is str and re.fullmatch('[0-9a-f]{40}', revision), 'Exact producer revision required')
    require(type(episodes) is list and 3 <= len(episodes) <= 4 and len(set(episodes)) == len(episodes)
        and all(type(ep) is int and 0 <= ep < 30 for ep in episodes), 'Original3or4 distinct selected episodes required')
    return _preview_records(root, revision, episodes)


def _metadata(path, *, maximum=8 << 20, readonly=True):
    if readonly:
        before = identity(path, maximum)
    else:
        from mediapipe_cpu_runtime_verify import identity as mutable_identity
        before = mutable_identity(path, maximum, readonly=False)
    raw = Path(path).read_bytes(); value = strict(raw)
    after = identity(path, maximum) if readonly else mutable_identity(path, maximum, readonly=False)
    require(after == before, 'Source metadata changed during snapshot read')
    return value, before, raw


def _frame_indices(value, total):
    return type(value) is list and value == list(range(total)) and all(type(index) is int for index in value)


def _producer_status(row):
    status = row.get('producer_status', row.get('status'))
    if status == 'complete_full4d_visual_diagnostic_not_quality_pass': return 'complete'
    if status in {'pending', 'running'}: return 'pending'
    if status in {'fail', 'fail_upstream_pose_unsupported_native_absence', 'not_run_upstream_scout_failure'}: return 'failed'
    raise ValueError('Only actual known numerical producer states may be displayed')


def rendered_source_pins(root, numerical_code, sources):
    """Rehash saved-render sources, excluding the active mutable pilot report."""
    from mediapipe_cpu_runtime_verify import identity as artifact_identity
    root, numerical_code = canonical(root), canonical(numerical_code)
    require(type(sources) is dict and 1 <= len(sources) <= 128, 'Bounded actual saved-render source ledger required')
    result = {}
    for name, pin in sources.items():
        path = canonical(Path(name))
        require(path.is_relative_to(root) and path != root
                and not (path.parent.parent == root/'experiments' and path.name == 'report.json')
                and type(pin) is dict and set(pin) == {'bytes', 'sha256'}
                and type(pin['bytes']) is int and 0 < pin['bytes'] <= 2 << 30
                and re.fullmatch('[0-9a-f]{64}', str(pin['sha256'])),
                'Only original source payload pins; never active experiment status metadata')
        require(artifact_identity(path, 2 << 30, readonly=path.is_relative_to(numerical_code)) == pin,
                'Saved-render bound source payload changed')
        result[name] = pin
    return result


def partial_receipt_contract(receipt, publication_revision):
    """Pure tiny receipt contract shared with the loopback viewer."""
    require(type(publication_revision) is str and re.fullmatch('[0-9a-f]{40}', publication_revision),
            'Actual partial publication revision required')
    require(type(receipt) is dict and receipt.get('schema') == PARTIAL_SCHEMA and receipt.get('status') == 'pass'
            and receipt.get('producer_revision') == publication_revision
            and receipt.get('publication_source_entry') == PUBLISH_ENTRY
            and receipt.get('numerical_source_entry') == NUMERICAL_ENTRY
            and re.fullmatch('[0-9a-f]{40}', str(receipt.get('numerical_producer_revision')))
            and receipt['numerical_producer_revision'] != publication_revision
            and receipt.get('endpoint') == ENDPOINT
            and receipt.get('cohort') == COHORT and receipt.get('cohort_denominator') == 4
            and receipt.get('statuses_frozen_at_capture') is True and receipt.get('live_status_claimed') is False
            and all(receipt.get(key) is False for key in ('quality_verified', 'challenge_performance_verified',
                'account_keys_used', 'public_access_changed', 'sas_tokens_persisted', 'heavy_data_uploaded',
                'original_source_videos_uploaded'))
            and receipt.get('private_container_verified') is True and receipt.get('source_rehashed_after') is True,
            'Explicit private partial snapshot protocol required; not the legacy full cohort protocol')
    for key, revision in (('publication_source_binding', publication_revision),
                          ('numerical_source_binding', receipt['numerical_producer_revision'])):
        binding = receipt.get(key)
        require(type(binding) is dict and binding.get('producer_revision') == revision
                and re.fullmatch('[0-9a-f]{64}', str(binding.get('closure_sha256')))
                and type(binding.get('helpers')) is dict and binding['helpers'],
                'Distinct actual numeric/publication immutable source bindings required')
    snapshot = receipt.get('source_report_snapshot')
    require(type(snapshot) is dict and type(snapshot.get('bytes')) is int and 0 < snapshot['bytes'] <= 8 << 20
            and re.fullmatch('[0-9a-f]{64}', str(snapshot.get('sha256'))), 'Bounded actual numerical-report snapshot required')
    states = receipt.get('cohort_statuses')
    require(type(states) is list and len(states) == 4 and all(type(row) is dict for row in states)
            and [row.get('episode_index') for row in states] == COHORT
            and all(type(row.get('episode_index')) is int and type(row.get('original_frames')) is int
                    and type(row.get('producer_status')) is str
                    and row['original_frames'] >= 3 and row.get('status') == _producer_status(row)
                    for row in states), 'All four source-bound original cohort statuses required')
    complete = [row['episode_index'] for row in states if row['status'] == 'complete']
    require(1 <= len(complete) <= 2 and receipt.get('episodes') == complete,
            'Publish all and only the currently complete one/two original clips; never cherry-pick')
    reports = receipt.get('reports')
    require(type(reports) is list and all(type(row) is dict for row in reports)
            and [row.get('episode_index') for row in reports] == complete
            and all(row.get('original_frames') == next(state['original_frames'] for state in states
                if state['episode_index'] == row['episode_index']) for row in reports),
            'Partial media must retain each full original frame count')
    for report in reports:
        for key in ('identity', 'export_report', 'export_pins'):
            pin = report.get(key)
            require(type(pin) is dict and set(pin) == {'bytes', 'sha256'}
                    and type(pin['bytes']) is int and 0 < pin['bytes'] <= 4 << 20
                    and re.fullmatch('[0-9a-f]{64}', str(pin['sha256'])),
                    'Full-T renderer/export/source pins required for every complete preview')
    return complete, states


def partial_preview_records(root, numerical_revision, publication_code, publication_revision):
    """Readonly actual full-T export/render checks, not new inference or fitting."""
    from mediapipe_cpu_runtime_verify import source
    root, publication_code = canonical(root), canonical(publication_code)
    require(type(numerical_revision) is str and re.fullmatch('[0-9a-f]{40}', numerical_revision)
            and numerical_revision != publication_revision
            and Path(__file__).resolve() == publication_code/'infra/full4d_publish.py',
            'Separate actual immutable partial publisher required')
    publisher = source(root, publication_code, publication_revision, PUBLISH_ENTRY,
        ('infra/full4d_publish.py', 'infra/run_full4d_publish.sh', 'infra/mediapipe_cpu_runtime_verify.py'))
    experiment = root/'experiments'/f'full4d-v1-{numerical_revision}'
    report, report_pin, raw = _metadata(experiment/'report.json', readonly=False)
    binding = report.get('source_binding', {})
    require(type(binding.get('helpers')) is dict and binding['helpers'], 'Actual numerical source helper ledger required')
    numerical_code = root/'jobs'/numerical_revision/NUMERICAL_ENTRY/'code'
    numeric = source(root, numerical_code, numerical_revision, NUMERICAL_ENTRY, tuple(binding['helpers']))
    require(report.get('schema') == 'world_reward.gemini_full4d.v1'
            and report.get('producer_revision') == numerical_revision and binding == numeric
            and report.get('status') in {'running', 'fail', 'complete_diagnostic_not_quality_pass'}
            and report.get('sampling') == dict(population=30, seed=20261008, count=4, replacement=False)
            and all(report.get(key) is False for key in ('ground_truth_used', 'hand_labeled_test',
                'quality_verified', 'challenge_performance_verified', 'jitter_fix_claimed', 'baseline_modified'))
            and report.get('oracle_modes') == [], 'Genuine no-oracle full4D numerical producer report required')
    rows = report.get('episodes')
    require(type(rows) is list and len(rows) == 4 and all(type(row) is dict for row in rows)
            and [row.get('episode') for row in rows] == COHORT
            and all(type(row.get('episode')) is int and type(row.get('original_frames')) is int
                    and row['original_frames'] >= 3 for row in rows), 'Original ordered four-clip denominator required')
    states = [dict(episode_index=row['episode'], original_frames=row['original_frames'],
                   status=_producer_status(row), producer_status=row['status']) for row in rows]
    complete = [row['episode_index'] for row in states if row['status'] == 'complete']
    require(1 <= len(complete) <= 2, 'Partial protocol requires exactly one/two actual complete clips')
    for ep in complete:
        directory=experiment/'videos'/f'episode_{ep:06d}'
        require(directory.is_dir() and {path.name for path in directory.iterdir()} ==
                {'report.json', f'episode_{ep:06d}.mp4', f'episode_{ep:06d}.jpg'},
                'Exact completed three-file saved render inventory required')
    _, files, renders = _preview_records(root, numerical_revision, complete)
    evidence = []
    for state, render in ((state, render) for state in states for render in renders
                           if state['episode_index'] == render['episode_index']):
        ep, total = state['episode_index'], state['original_frames']
        directory = experiment/'videos'/f'episode_{ep:06d}'
        require({p.name for p in directory.iterdir()} == {'report.json', f'episode_{ep:06d}.mp4', f'episode_{ep:06d}.jpg'},
                'Exact completed three-file saved render inventory required')
        video, video_pin, _ = _metadata(directory/'report.json', maximum=262144)
        numerical_row = next(row for row in rows if row['episode'] == ep)
        require(render['original_frames'] == total and _frame_indices(video.get('original_frame_indices'), total)
                and numerical_row.get('video_report') == video_pin,
                'Actual complete original render timeline required')
        export = experiment/'outputs'/f'episode_{ep:06d}'/'cari_shared_export_v1'
        require({p.name for p in export.iterdir()} == EXPORT_FILES, 'Exact five-file full native export required')
        export_report, export_pin, _ = _metadata(export/'report.json', maximum=4 << 20)
        expected = dict(stage='world_reward_native_cari_shared_full_video_direct_export', status='pass',
            phase='complete', producer_revision=numerical_revision, episode_index=ep, frames=total,
            input_track='track_1', ground_truth_used=False, ground_truth_read=False, private_truth_read=False,
            hand_labeled_test=False, oracle_modes=[], unchanged_refined_predictions_verified=True,
            full_original_native_export_verified=True, source_inputs_assets_rehashed=True, source_helpers_rehashed=True)
        clip_spec = dict(episode_index=ep, total_frames=total, camera_name='front_stereo_camera_left', height=1152, width=1536)
        require(all(type(export_report.get(key)) is type(value) and export_report[key] == value for key, value in expected.items())
                and _frame_indices(export_report.get('original_frame_indices'), total)
                and export_report.get('clip_spec') == clip_spec
                and export_report.get('script_sha256') == identity(numerical_code/'infra/cari_full_export.py', 2 << 20)['sha256'],
                'Genuine full-T no-oracle native export PASS required before any partial publication')
        pin_path = experiment/'pins'/f'cari_clip_{ep:06d}_shared_export_pins.json'
        pins, pin_id, _ = _metadata(pin_path, maximum=4 << 20)
        require(pins.get('schema') == 'world-reward-cari-shared-export-pins-v1'
                and pins.get('clip_spec') == clip_spec
                and pins.get('export') == export_pin | dict(producer_revision=numerical_revision,
                    script_sha256=export_report['script_sha256']) and set(pins.get('export_files', {})) == EXPORT_FILES,
                'Actual independent sealed export pin manifest required')
        payloads = {name: identity(export/name, 2 << 30) for name in EXPORT_FILES}
        require(payloads == pins['export_files'] and payloads['report.json'] == export_pin
                and export_report.get('output_files') == {name: pin for name, pin in payloads.items() if name != 'report.json'}
                and video.get('sources', {}).get(str(pin_path)) == pin_id
                and all(video.get('sources', {}).get(str(export/name)) == pin for name, pin in payloads.items())
                and video.get('sources', {}).get(str(numerical_code/'infra/full4d_video.py')) ==
                    identity(numerical_code/'infra/full4d_video.py', 1 << 20),
                'Frozen full native export and exact saved-render source/artifact ancestry required')
        evidence.append(render | dict(export_report=export_pin, export_pins=pin_id))
        rendered_source_pins(root, numerical_code, video['sources'])
    # No active numerical file is changed, and no later status is invented.
    require(source(root, publication_code, publication_revision, PUBLISH_ENTRY, tuple(publisher['helpers'])) == publisher
            and source(root, numerical_code, numerical_revision, NUMERICAL_ENTRY, tuple(numeric['helpers'])) == numeric,
            'Numeric/publication immutable source changed during readonly preflight')
    return dict(experiment=experiment, files=files, reports=evidence, raw_snapshot=raw,
        source_report_snapshot=report_pin, cohort_statuses=states, episodes=complete,
        publication_source_binding=publisher, numerical_source_binding=numeric)


def publish_partial(root, numerical_revision, publication_code, publication_revision, *, client=None):
    """New immutable private publication namespace, distinct from active numeric job."""
    proof = partial_preview_records(root, numerical_revision, publication_code, publication_revision)
    directory = canonical(root)/'results'/f'full4d-partial-{publication_revision}'
    require(not directory.exists(), 'Fresh separate partial publication namespace required')
    directory.mkdir(mode=0o755)
    snapshot = directory/'numerical-report.json'
    with snapshot.open('xb') as writer: writer.write(proof['raw_snapshot'])
    snapshot.chmod(0o444)
    require(identity(snapshot, 8 << 20) == proof['source_report_snapshot'], 'Exact immutable report snapshot required')
    client = client or PrivatePreviews(); client.require_private()
    result = dict(schema=PARTIAL_SCHEMA, status='fail', producer_revision=publication_revision,
        numerical_producer_revision=numerical_revision, publication_source_entry=PUBLISH_ENTRY,
        numerical_source_entry=NUMERICAL_ENTRY, cohort=COHORT, cohort_denominator=4,
        cohort_statuses=proof['cohort_statuses'], episodes=proof['episodes'],
        source_report_snapshot=proof['source_report_snapshot'], statuses_frozen_at_capture=True, live_status_claimed=False,
        publication_source_binding=proof['publication_source_binding'], numerical_source_binding=proof['numerical_source_binding'],
        reports=proof['reports'], endpoint=ENDPOINT, files=[], private_container_verified=True,
        quality_verified=False, challenge_performance_verified=False, account_keys_used=False,
        public_access_changed=False, sas_tokens_persisted=False, heavy_data_uploaded=False, original_source_videos_uploaded=False)
    started = time.monotonic()
    try:
        numerical_code = canonical(root)/'jobs'/numerical_revision/NUMERICAL_ENTRY/'code'
        for record in proof['reports']:
            video, _, _ = _metadata(Path(record['path']), maximum=262144)
            rendered_source_pins(root, numerical_code, video['sources'])
        for record in proof['files']:
            path = Path(record['path']); require(identity(path, MAX_VIDEO) == {key: record[key] for key in ('bytes', 'sha256')},
                'Completed rendered preview changed before progressive publication')
            name = f'full4d-{publication_revision}/'+record['name'].split('/')[-1]
            uploaded = client.upload(name, path.read_bytes(), record['mime'], publication_revision)
            result['files'].append(dict(episode_index=record['episode_index'], **uploaded))
            client.head(uploaded['name'], uploaded, uploaded['etag'])
        manifest = {key: value for key, value in result.items() if key not in {'reports', 'publication_source_binding', 'numerical_source_binding'}}
        manifest['status'] = 'frozen_partial_preview_snapshot_not_quality_pass'
        raw = (json.dumps(manifest, sort_keys=True, allow_nan=False)+'\n').encode()
        require(len(raw) <= MAX_MANIFEST, 'Bounded partial status/media manifest required')
        uploaded = client.upload(f'full4d-{publication_revision}/manifest.json', raw, 'application/json', publication_revision)
        result['files'].append(uploaded); client.head(uploaded['name'], uploaded, uploaded['etag'])
        for record in proof['files']:
            require(identity(Path(record['path']), MAX_VIDEO) == {key: record[key] for key in ('bytes', 'sha256')},
                    'Completed rendered preview changed during progressive publication')
        for record in proof['reports']:
            require(identity(Path(record['path']), 262144) == record['identity'], 'Full-T saved renderer report changed')
            video, _, _ = _metadata(Path(record['path']), maximum=262144)
            rendered_source_pins(root, numerical_code, video['sources'])
        require(identity(snapshot, 8 << 20) == proof['source_report_snapshot'], 'Frozen numerical report snapshot changed')
        from mediapipe_cpu_runtime_verify import source
        for revision, entry, key in ((publication_revision, PUBLISH_ENTRY, 'publication_source_binding'),
                                     (numerical_revision, NUMERICAL_ENTRY, 'numerical_source_binding')):
            require(source(root, root/'jobs'/revision/entry/'code', revision, entry,
                tuple(proof[key]['helpers'])) == proof[key], 'Numerical/publication source changed during upload')
        result.update(status='pass', source_rehashed_after=True,
            mp4_total_bytes=sum(row['bytes'] for row in result['files'] if row['mime'] == 'video/mp4'))
        partial_receipt_contract(result, publication_revision)
        require(len(json.dumps(result, sort_keys=True, allow_nan=False).encode())+128 <= MAX_PARTIAL_RECEIPT,
                'Explicit bounded64KiB partial metadata receipt required')
    except Exception as error:
        result.update(status='fail', error_type=type(error).__name__); raise
    finally:
        result['elapsed_seconds'] = time.monotonic()-started
        with (directory/'report.json').open('x') as writer:
            json.dump(result, writer, sort_keys=True, allow_nan=False); writer.write('\n')
        (directory/'report.json').chmod(0o444)
    print('FULL4D_PARTIAL_PUBLISH_PASS', len(result['episodes']), result['mp4_total_bytes'])
    return result


def publish(root, revision, episodes, *, client=None):
    experiment, files, reports = preview_records(root, revision, episodes)
    receipt = experiment/'published.json'
    require(not receipt.exists() and not receipt.is_symlink(), 'New private preview publication only')
    client = client or PrivatePreviews()
    client.require_private()
    result = dict(schema=SCHEMA, status='fail', producer_revision=revision, episodes=episodes,
        endpoint=ENDPOINT, private_container_verified=True, files=[], reports=reports,
        account_keys_used=False, public_access_changed=False, sas_tokens_persisted=False,
        heavy_data_uploaded=False, original_source_videos_uploaded=False)
    started = time.monotonic()
    try:
        for record in files:
            path = Path(record['path'])
            require(identity(path, MAX_VIDEO) == {k: record[k] for k in ('bytes', 'sha256')},
                'Frozen preview changed before publication')
            uploaded = client.upload(record['name'], path.read_bytes(), record['mime'], revision)
            result['files'].append(dict(episode_index=record['episode_index'], **uploaded))
            # Record successful PUT/ETag before HEAD so a partial verification
            # failure cannot orphan an unrecorded owned preview blob.
            client.head(uploaded['name'], uploaded, uploaded['etag'])
        manifest = dict(schema=SCHEMA, producer_revision=revision, episodes=episodes,
            private_preview_only=True, files=result['files'],
            playback=dict(fps=30, comparison='original_RGB_left_saved_4D_right',
                scale='unchanged_inferred_metric_scale_not_ground_truth', virtual_floor='fixed_assumed_background'))
        raw = (json.dumps(manifest, sort_keys=True, allow_nan=False)+'\n').encode()
        require(len(raw) <= MAX_MANIFEST, 'Tiny private preview manifest required')
        uploaded = client.upload(f'full4d-{revision}/manifest.json', raw, 'application/json', revision)
        result['files'].append(uploaded)
        client.head(uploaded['name'], uploaded, uploaded['etag'])
        for record in files:
            require(identity(Path(record['path']), MAX_VIDEO) == {k: record[k] for k in ('bytes', 'sha256')},
                'Read-only rendered preview changed during upload')
        for record in reports:
            require(identity(Path(record['path']), 262144) == record['identity'], 'Saved QA receipt changed during upload')
        result.update(status='pass', source_rehashed_after=True,
            mp4_total_bytes=sum(r['bytes'] for r in result['files'] if r['mime'] == 'video/mp4'))
    except Exception as error:
        result['error_type'] = type(error).__name__
        # No foreign object or existing blob is overwritten/deleted. Preserve an
        # exact partial-publication receipt so cleanup can use original ETags.
        raise
    finally:
        result['elapsed_seconds'] = time.monotonic()-started
        with receipt.open('x') as stream:
            json.dump(result, stream, sort_keys=True, allow_nan=False); stream.write('\n')
        receipt.chmod(0o444)
    print('FULL4D_PUBLISH_PASS', len(episodes), result['mp4_total_bytes'])
    return result


def partial_cleanup_records(receipt, publication_revision):
    """Pure exact owner namespace for PASS or interrupted successful-PUT receipts."""
    require(type(publication_revision) is str and re.fullmatch('[0-9a-f]{40}', publication_revision)
            and type(receipt) is dict and receipt.get('schema') == PARTIAL_SCHEMA
            and receipt.get('status') in {'pass', 'fail'} and receipt.get('producer_revision') == publication_revision
            and receipt.get('publication_source_entry') == PUBLISH_ENTRY and receipt.get('endpoint') == ENDPOINT
            and receipt.get('cohort') == COHORT and receipt.get('cohort_denominator') == 4
            and all(receipt.get(key) is False for key in ('account_keys_used','public_access_changed','sas_tokens_persisted')),
            'Original partial publication receipt and owner required for cleanup')
    binding = receipt.get('publication_source_binding', {})
    require(binding.get('producer_revision') == publication_revision
            and re.fullmatch('[0-9a-f]{64}', str(binding.get('closure_sha256'))),
            'Actual partial publication source ownership required')
    episodes = receipt.get('episodes')
    require(type(episodes) is list and 1 <= len(episodes) <= 2 and len(set(episodes)) == len(episodes)
            and all(type(ep) is int and ep in COHORT for ep in episodes),
            'Only original actual published partial episodes may be deleted')
    allowed = {f'full4d-{publication_revision}/episode_{ep:06d}.{ext}':(mime,limit,ep)
        for ep in episodes for ext,mime,limit in (('mp4','video/mp4',MAX_VIDEO),('jpg','image/jpeg',MAX_POSTER))}
    allowed[f'full4d-{publication_revision}/manifest.json']=('application/json',MAX_MANIFEST,None)
    rows=receipt.get('files'); seen=set()
    require(type(rows) is list and len(rows) <= len(allowed), 'Bounded original successful-PUT cleanup inventory required')
    for row in rows:
        require(type(row) is dict and row.get('name') in allowed and row['name'] not in seen,
                'No foreign, extra or duplicate partial publication blob may be deleted')
        mime,limit,episode=allowed[row['name']]
        require(row.get('mime') == mime and type(row.get('bytes')) is int and 0 < row['bytes'] <= limit
                and re.fullmatch('[0-9a-f]{64}', str(row.get('sha256')))
                and re.fullmatch(r'"[A-Za-z0-9-]{1,100}"', str(row.get('etag')))
                and (episode is None or type(row.get('episode_index')) is int and row['episode_index'] == episode),
                'Exact owned original PUT ETag/bytes/SHA/mime binding required')
        seen.add(row['name'])
    if receipt['status']=='pass':
        partial_receipt_contract(receipt,publication_revision)
        require(seen==set(allowed), 'PASS cleanup requires the entire exact original publication inventory')
    return rows


def delete_partial(root, publication_revision, *, publication_code, client=None):
    """Conditional ETag cleanup only in this original publisher's namespace."""
    from mediapipe_cpu_runtime_verify import source
    root,publication_code=canonical(root),canonical(publication_code)
    require(Path(__file__).resolve() == publication_code/'infra/full4d_publish.py', 'Actual owned cleanup source required')
    directory=root/'results'/f'full4d-partial-{publication_revision}'
    receipt,receipt_pin,_=_metadata(directory/'report.json',maximum=MAX_PARTIAL_RECEIPT)
    rows=partial_cleanup_records(receipt,publication_revision)
    binding=receipt['publication_source_binding']
    require(source(root,publication_code,publication_revision,PUBLISH_ENTRY,tuple(binding['helpers']))==binding,
            'Only actual original partial publisher may delete its owned previews')
    client=client or PrivatePreviews();client.require_private()
    for row in rows:client.delete_owned(row)
    require(identity(directory/'report.json',MAX_PARTIAL_RECEIPT)==receipt_pin, 'Original cleanup receipt changed')
    print('FULL4D_PARTIAL_PRIVATE_PREVIEWS_DELETED',len(rows))


def delete(root, revision):
    require(re.fullmatch('[0-9a-f]{40}', revision), 'Exact producer revision required')
    receipt = canonical(root)/'experiments'/f'full4d-v1-{revision}'/'published.json'
    identity(receipt, MAX_MANIFEST)
    result = strict(receipt.read_bytes())
    require(result.get('schema') == SCHEMA and result.get('producer_revision') == revision
        and result.get('endpoint') == ENDPOINT and result.get('public_access_changed') is False,
        'Exact original private publication receipt required')
    client = PrivatePreviews(); client.require_private()
    for row in result['files']:
        require(row['name'].startswith(f'full4d-{revision}/'), 'Only original owned publication namespace allowed')
        client.delete_owned(row)
    print('FULL4D_PRIVATE_PREVIEWS_DELETED', len(result['files']))


def main():
    require(os.uname().sysname == 'Linux' and os.geteuid() == 0
        and os.uname().nodename == 'scenesmith-ncc-h100-01', 'Azure VM01 managed-identity host only')
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--episodes', type=int, nargs='+')
    parser.add_argument('--delete', action='store_true')
    parser.add_argument('--partial', action='store_true')
    args = parser.parse_args()
    if args.partial:
        require(args.episodes is None, 'Partial protocol derives its frozen cohort from actual source report')
        if args.delete:
            require(args.revision == os.environ['WR_CODE_REVISION'],
                    'Partial cleanup revision must be this exact original publication owner')
            delete_partial(ROOT,args.revision,publication_code=Path(os.environ['WR_CODE']))
        else:
            publish_partial(ROOT, args.revision, Path(os.environ['WR_CODE']), os.environ['WR_CODE_REVISION'])
    elif args.delete:
        require(args.episodes is None, 'Deletion uses original immutable receipt only')
        delete(ROOT, args.revision)
    else:
        publish(ROOT, args.revision, args.episodes)


if __name__ == '__main__':
    main()
