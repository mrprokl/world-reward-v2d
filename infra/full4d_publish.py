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
SCHEMA = 'world_reward.full4d_publish.v1'


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
        require(container_acl is False or method == 'GET' and name is None,
            'Only read-only container ACL validation allowed')
        if container_acl:
            url = ENDPOINT+'?restype=container&comp=acl'
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
        with self.request('GET', container_acl=True) as response:
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
                'x-ms-meta-sha256': pin['sha256'], 'x-ms-meta-worldreward-revision': revision}) as response:
            require(response.status == 201, 'Exclusive private preview creation required')
            etag = response.headers['ETag']
        return dict(name=name, etag=etag, mime=mime, **pin)

    def delete_owned(self, row):
        self.head(row['name'], row, row['etag'])
        with self.request('DELETE', row['name'], headers={'If-Match': row['etag']}) as response:
            require(response.status == 202, 'Owned conditional preview deletion required')


def preview_records(root, revision, episodes):
    require(type(revision) is str and re.fullmatch('[0-9a-f]{40}', revision), 'Exact producer revision required')
    require(type(episodes) is list and 3 <= len(episodes) <= 4 and len(set(episodes)) == len(episodes)
        and all(type(ep) is int and 0 <= ep < 30 for ep in episodes), 'Original3or4 distinct selected episodes required')
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
    args = parser.parse_args()
    if args.delete:
        require(args.episodes is None, 'Deletion uses original immutable receipt only')
        delete(ROOT, args.revision)
    else:
        publish(ROOT, args.revision, args.episodes)


if __name__ == '__main__':
    main()
