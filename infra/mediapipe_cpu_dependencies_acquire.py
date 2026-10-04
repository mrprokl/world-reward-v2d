"""Azure CPU-only exact wheel acquisition; no install, execution, or dataset."""
import email.parser
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import pwd
import re
import signal
import stat
import sys
import time
import urllib.request
import zipfile

import mediapipe_hands_acquire as mp

ROOT = mp.ROOT
JOB = 'run_mediapipe_cpu_dependencies_acquire'
EVIDENCE = 'vendor/research/mediapipe_cpu_dependencies_v2'
DOWNLOAD_RESULT = 'results/mediapipe-cpu-dependencies-acquire-v2'
RESULT = 'results/mediapipe-cpu-dependencies-acquire-v3'
MANIFEST = 'configs/mediapipe_cpu_dependencies_v1.json'
PRIOR_PINS = 'configs/mediapipe_hands_acquire_pins.json'
DOWNLOAD_PINS = 'configs/mediapipe_cpu_dependencies_download_pins.json'
MANIFEST_PIN = dict(bytes=59268, sha256='efb697fd458156a93520db3add265b75b3506f84e62b681920ce6f7c75375c81')
BUDGET = 300
require, canonical, identity, state = mp.require, mp.canonical, mp.identity, mp.state
HELPERS = ('infra/mediapipe_cpu_dependencies_acquire.py', 'infra/run_mediapipe_cpu_dependencies_acquire.sh',
           'infra/mediapipe_hands_acquire.py', MANIFEST, PRIOR_PINS)


def source_binding(root, code, revision, *, old=False, verify_acquired=False):
    if not old:
        require(Path(__file__) == Path(code)/HELPERS[0] and Path(mp.__file__) == Path(code)/'infra/mediapipe_hands_acquire.py', 'Actual imported helper paths differ')
        return dependency_source(root, code, revision, verify_acquired=verify_acquired)
    code = canonical(code); entry = mp.JOB if old else JOB
    require(re.fullmatch('[0-9a-f]{40}', revision) and code == root/'jobs'/revision/entry/'code', 'Exact source job required')
    markers = {n: identity(code.parent/n) for n in ('revision', 'source-sha256')}
    require((code.parent/'revision').read_bytes() == (revision+'\n').encode() and
            re.fullmatch(b'[0-9a-f]{64}\n', (code.parent/'source-sha256').read_bytes()), 'Dispatch markers differ')
    closure = {}
    for path in (code, *sorted(code.rglob('*'))):
        canonical(path); before = state(path)
        require(not before[2] & 0o222, 'Readonly complete source closure required')
        closure[str(path.relative_to(code))] = {'directory': True} if stat.S_ISDIR(before[2]) else identity(path, empty=True)
    helpers = ('infra/mediapipe_hands_acquire.py', 'infra/run_mediapipe_hands_acquire.sh') if old else HELPERS
    require(all(n in closure and 'sha256' in closure[n] for n in helpers), 'Source/config closure incomplete')
    return dict(producer_revision=revision, markers=markers, entries=len(closure), helpers={n: closure[n] for n in helpers},
                closure_sha256=hashlib.sha256(json.dumps(closure, sort_keys=True).encode()).hexdigest())


def dependency_source(root, code, revision, *, verify_acquired=False):
    """Pure historical snapshot binding, without pretending to run its entry."""
    code = canonical(code)
    require(re.fullmatch('[0-9a-f]{40}', revision) and code == root/'jobs'/revision/JOB/'code', 'Exact dependency snapshot required')
    markers = {n: identity(code.parent/n) for n in ('revision', 'source-sha256')}
    require((code.parent/'revision').read_bytes() == (revision+'\n').encode() and
            re.fullmatch(b'[0-9a-f]{64}\n', (code.parent/'source-sha256').read_bytes()), 'Dispatch markers differ')
    closure = {}
    for path in (code, *sorted(code.rglob('*'))):
        canonical(path); require(not state(path)[2] & 0o222, 'Readonly historical source required')
        closure[str(path.relative_to(code))] = {'directory': True} if path.is_dir() else identity(path, empty=True)
    helpers = HELPERS + ((DOWNLOAD_PINS,) if verify_acquired else ())
    require(all(n in closure and 'sha256' in closure[n] for n in helpers), 'Historical dependency closure incomplete')
    return dict(producer_revision=revision, markers=markers, entries=len(closure), helpers={n: closure[n] for n in helpers},
                closure_sha256=hashlib.sha256(json.dumps(closure, sort_keys=True).encode()).hexdigest())


def pin(value):
    require(type(value) is dict and set(value) == {'bytes', 'sha256'} and type(value['bytes']) is int and
            0 < value['bytes'] <= 256 << 20 and re.fullmatch('[0-9a-f]{64}', value['sha256']), 'Exact bounded byte pin required')
    return value


def verify_prior(root, pins_path):
    pins_identity = identity(pins_path); pins = mp.strict_json(pins_path)
    require(set(pins) == {'schema', 'producer_revision', 'report', 'helper'} and
            pins['schema'] == 'world_reward.mediapipe_hands_acquire_pins.v1' and
            re.fullmatch('[0-9a-f]{40}', pins['producer_revision']), 'Independent prior acquisition pin schema differs')
    require(identity(Path(mp.__file__)) == pin(pins['helper']), 'Imported acquisition helper differs from original pin')
    code = root/'jobs'/pins['producer_revision']/mp.JOB/'code'
    source = source_binding(root, code, pins['producer_revision'], old=True)
    require(source['helpers']['infra/mediapipe_hands_acquire.py'] == pins['helper'], 'Original source helper bytes differ')
    path = root/mp.RESULT/'report.json'
    require(identity(path) == pin(pins['report']), 'Independent prior receipt bytes differ')
    receipt = mp.strict_json(path)
    expected = dict(stage='mediapipe_hands_source_model_acquisition', status='pass', source_binding=source,
                    source_rehashed_after=True, artifacts_rehashed_after=True, owned_partials_removed=True,
                    models_loaded=False, model_nodes_decoded=False, packages_installed=False, gpu_used=False,
                    dataset_read=False, private_values_read=False, quality_claim=False,
                    license_eligibility_verified=False, task_constituent_license_verified=False,
                    training_overlap_verified=False, challenge_overlap_verified=False)
    require(all(type(receipt.get(k)) is type(v) and receipt[k] == v for k, v in expected.items()), 'Genuine unchanged prior acquisition PASS required')
    require(type(receipt.get('elapsed_seconds')) in (int, float) and 0 <= receipt['elapsed_seconds'] <= 300,
            'Prior acquisition budget differs')
    rows = receipt.get('artifacts'); require(type(rows) is list and len(rows) == len(mp.ASSETS), 'All seven original artifacts required')
    files = {row['file']: row for row in rows}; require(len(files) == len(rows), 'Duplicate prior artifact records')
    actual = {}
    for asset in mp.ASSETS:
        name = asset['folder']+'/'+asset['name']; row = files.get(name, {})
        wanted = pin({k: row.get(k) for k in ('bytes', 'sha256')})
        require(wanted['bytes'] == asset['bytes'] and (not asset.get('sha256') or wanted['sha256'] == asset['sha256']), 'Prior asset publisher/audit pin differs')
        path = canonical(root/name); require(identity(path) == wanted, 'Prior readonly artifact changed')
        if asset.get('md5'):
            digest = hashlib.md5()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(mp.BLOCK), b''): digest.update(block)
            require(digest.hexdigest() == asset['md5'], 'Original model publisher MD5 differs')
        actual[name] = wanted
    return dict(pins_identity=pins_identity, report_identity=pins['report'], source_binding=source, artifacts=actual)


def load_manifest(path):
    require(identity(path) == MANIFEST_PIN, 'Independently frozen dependency manifest differs')
    value = mp.strict_json(path); rows = value.get('packages')
    require(value.get('schema') == 'world_reward.mediapipe_cpu_dependencies.v1' and type(rows) is list and
            len(rows) == value.get('package_count') == 26 and value.get('new_wheel_bytes', 1 << 40) <= 1_000_000_000,
            'Frozen dependency manifest scope differs')
    for row in rows:
        require(row['filename'].endswith('.whl') and re.fullmatch('[A-Za-z0-9_.+-]+', row['filename']) and
                row['url'].startswith('https://files.pythonhosted.org/packages/') and row['url'].endswith('/'+row['filename']) and
                row['metadata']['url'] == row['url']+'.metadata', 'Exact public wheel/metadata URLs required')
        pin({k: row[k] for k in ('bytes', 'sha256')}); pin({k: row['metadata'][k] for k in ('bytes', 'sha256')})
    return value


def wheel_record(path, metadata_path, row, deadline):
    inventory = mp.zip_inventory(path, deadline, wheel=False); entries = inventory.pop('inventory')
    by_name = {r['name']: r for r in entries}; prefix = row['filename'].split('-', 2)
    dist_info = prefix[0]+'-'+prefix[1]+'.dist-info/'; metadata_name = dist_info+'METADATA'
    require(metadata_name in by_name and by_name[metadata_name]['bytes'] == row['metadata']['bytes'] and
            by_name[metadata_name]['sha256'] == row['metadata']['sha256'], 'Wheel embedded METADATA differs from PEP658 pin')
    raw = metadata_path.read_bytes(); parsed = email.parser.BytesParser().parsebytes(raw)
    normalize = lambda s: re.sub('[-_.]+', '-', s).lower()
    license = row['metadata']['license']
    headers = parsed.get_all('License') or []
    require(len(headers) <= 1, 'Duplicate License metadata header')
    text = str(headers[0]).encode('utf-8') if headers else b''
    require(len(text) == license['declared_text_bytes'] and
            (hashlib.sha256(text).hexdigest() if headers else None) == license['declared_text_sha256'] and
            (license['declared'] is None or str(headers[0]) == license['declared']),
            'Wheel License text byte/hash differs')
    require(normalize(parsed.get('Name', '')) == row['name'] and parsed.get_all('Version') == [row['version']] and
            (parsed.get_all('Requires-Dist') or []) == row['metadata']['requires_dist'] and
            (parsed.get_all('License-File') or []) == license['license_files'] and
            parsed.get_all('License-Expression', []) == ([] if license['expression'] is None else [license['expression']]),
            'Wheel dependency/license metadata differs')
    notices = [r for r in entries if '/licenses/' in r['name'].lower() or re.match(
        r'(?i)^(licen[sc]e|copying|notice|authors|copyright)(?:[._-]|$)', PurePosixPath(r['name']).name)]
    mapping, unresolved = {}, []
    for declared in license['license_files']:
        # Raw legacy metadata may declare ../LICENSE. Only safe archive member
        # names from the strict inventory are considered; never extract/join it.
        basename = PurePosixPath(declared).name
        matches = [r['name'] for r in notices if PurePosixPath(r['name']).name == basename]
        mapping[declared] = matches
        if not matches: unresolved.append(declared)
    require(not unresolved, 'Declared wheel license notice missing from safe inventory')
    return dict(name=row['name'], version=row['version'], file=str(path), metadata_file=str(metadata_path),
                bytes=row['bytes'], sha256=row['sha256'], entries=inventory['entries'],
                expanded_bytes=inventory['expanded_bytes'], inventory_sha256=hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest(),
                license_metadata_files=[by_name[metadata_name], *notices], declared_license_files=mapping,
                unresolved_license_files=unresolved, license_eligibility_verified=False)


def assets(rows):
    result = []
    for row in rows:
        meta = row['metadata']; result.append(dict(name=row['filename']+'.metadata', folder=EVIDENCE,
            url=meta['url'], bytes=meta['bytes'], sha256=meta['sha256'],
            mime=('application/octet-stream', 'binary/octet-stream', 'text/plain')))
        if row['acquisition'] == 'new_Azure_wheel_only':
            result.append(dict(name=row['filename'], folder=EVIDENCE, url=row['url'], bytes=row['bytes'],
                sha256=row['sha256'], mime=('application/octet-stream', 'binary/octet-stream', 'application/zip')))
        else: require(row['name'] == 'mediapipe' and row['acquisition'] == 'reference_existing_mediapipe_hands_acquire_v1', 'Only original MediaPipe wheel may be referenced')
    require(len({r['name'] for r in result}) == len(result), 'Dependency publication filename collision')
    return result


def verify_downloads(root, code):
    """Authenticate the independently pinned, closed v2 metadata-parser FAIL.

    It is not a generic failed-run resume: all 51 publisher-pinned bytes must
    already exist readonly, with the original source/prior/posthash intact.
    """
    pins_path = code/DOWNLOAD_PINS; pins_identity = identity(pins_path); pins = mp.strict_json(pins_path)
    require(set(pins) == {'schema', 'producer_revision', 'report', 'helper'} and
            pins['schema'] == 'world_reward.mediapipe_cpu_dependencies_download_pins.v1' and
            re.fullmatch('[0-9a-f]{40}', pins['producer_revision']), 'Independent verified-download pin schema differs')
    original = root/'jobs'/pins['producer_revision']/JOB/'code'
    source = dependency_source(root, original, pins['producer_revision'])
    require(source['helpers'][HELPERS[0]] == pin(pins['helper']), 'Original download producer differs')
    path = root/DOWNLOAD_RESULT/'report.json'; require(identity(path) == pin(pins['report']), 'Closed original download receipt differs')
    receipt = mp.strict_json(path); prior = verify_prior(root, code/PRIOR_PINS); manifest = load_manifest(code/MANIFEST)
    expected = dict(schema='world_reward.mediapipe_cpu_dependencies_acquisition.v1', stage='mediapipe_cpu_dependencies_acquisition',
        status='fail', error_type='AcquisitionError', reason='Wheel dependency/license metadata differs', source_binding=source,
        prior_acquisition=prior, manifest_identity=MANIFEST_PIN, budget_seconds=300,
        source_rehashed_after=True, prior_rehashed_after=True, artifacts_rehashed_after=True, owned_partials_removed=True,
        models_loaded=False, packages_installed=False, gpu_used=False, dataset_read=False, private_values_read=False,
        quality_claim=False, license_eligibility_verified=False, training_overlap_verified=False, challenge_overlap_verified=False)
    require(all(type(receipt.get(k)) is type(v) and receipt[k] == v for k, v in expected.items()), 'Only exact unchanged metadata-parser FAIL may supply downloaded bytes')
    require(type(receipt.get('elapsed_seconds')) in (int, float) and 0 < receipt['elapsed_seconds'] <= 300, 'Original download budget differs')
    requested = assets(manifest['packages']); records = receipt.get('artifacts', [])
    wanted = {r['folder']+'/'+r['name']: r for r in requested}
    require(type(records) is list and len(records) == len(wanted) == 51 and
            {r['file'] for r in records} == set(wanted), 'All exact original downloads required')
    directory = canonical(root/EVIDENCE)
    require(stat.S_IMODE(directory.stat().st_mode) == 0o555 and {p.name for p in directory.iterdir()} == {r['name'] for r in requested},
            'Exclusive readonly original download directory required')
    for record in records:
        row = wanted[record['file']]; wanted_pin = {k: row[k] for k in ('bytes', 'sha256')}
        require({k: record[k] for k in ('bytes', 'sha256')} == wanted_pin and identity(root/record['file']) == wanted_pin and
                record.get('publisher_sha256_verified') is True and record.get('publisher_md5_verified') is False and
                record.get('hash_basis') == ('publisher_PEP658_sha256' if row['name'].endswith('.metadata') else 'publisher_sha256'),
                'Original publisher-pinned readonly download differs')
    partial = receipt.get('wheels', [])
    require(type(partial) is list and len(partial) == 3 and
            [r['name'] for r in partial] == [r['name'] for r in manifest['packages'][:3]], 'Original inventory failure location differs')
    return dict(pins_identity=pins_identity, report_identity=pins['report'], source_binding=source,
                prior_acquisition=prior, manifest_identity=MANIFEST_PIN, artifacts=records,
                original_status=receipt['status'], original_reason=receipt['reason'])


def acquire(root, code, revision, *, opener=None, namespace_lease=None, verify_acquired=False):
    started = time.monotonic(); root, code = canonical(root), canonical(code)
    directory, out = canonical(root/EVIDENCE), canonical(root/RESULT)
    require((verify_acquired or namespace_lease is not None or not directory.exists()) and not out.exists() and directory.parent.is_dir() and out.parent.is_dir(), 'Fresh namespaces required; no resume')
    out.mkdir(mode=0o700); out.chmod(0o700)
    owned, artifacts, wheels, before, prior, failure, created, downloads = [], [], [], None, None, None, None, None
    report = dict(schema='world_reward.mediapipe_cpu_dependencies_acquisition.v1', stage='mediapipe_cpu_dependencies_acquisition',
        status='fail', budget_seconds=BUDGET, budget_scope='acquisition_checks_public_sealing_posthash', receipt_publication_outer_seconds=320,
        artifacts=artifacts, wheels=wheels, models_loaded=False, packages_installed=False, gpu_used=False, dataset_read=False,
        private_values_read=False, quality_claim=False, license_eligibility_verified=False, training_overlap_verified=False, challenge_overlap_verified=False)
    old_urls = mp.URLS
    try:
        before = source_binding(root, code, revision, verify_acquired=verify_acquired); report['source_binding'] = before
        if verify_acquired:
            require(opener is None and namespace_lease is None, 'Verify-only accepts no network or namespace lease')
            report.update(mode='verify_acquired_bytes_and_notices', network_used=False, new_downloads=0, stage_adoption=False,
                          budget_scope='source_download_checks_inventory_posthash')
            downloads = verify_downloads(root, code); report['prior_verified_downloads'] = downloads
            prior = downloads['prior_acquisition']; artifacts.extend(downloads['artifacts'])
        elif namespace_lease is not None:
            lease = mp.validate_namespace_lease(namespace_lease, [directory], before['closure_sha256'])
            created = lease[0][1]; report['namespace_lease'] = namespace_lease
        prior = prior or verify_prior(root, code/PRIOR_PINS); report['prior_acquisition'] = prior
        manifest = load_manifest(code/MANIFEST); report['manifest_identity'] = MANIFEST_PIN
        rows = manifest['packages']; requested = assets(rows)
        # Scoped reuse of the pinned original downloader; immutable manifest
        # defines every URL, never arbitrary command-line URLs or credentials.
        if not verify_acquired: mp.URLS = frozenset(r['url'] for r in requested)
        if not verify_acquired and namespace_lease is None:
            directory.mkdir(mode=0o700); created = state(directory)[:2]; directory.chmod(0o700)
        if not verify_acquired: opener = opener or urllib.request.build_opener(urllib.request.ProxyHandler({}), mp.PublicRedirect())
        for item in ([] if verify_acquired else requested):
            record = mp.fetch(item, root, opener, started+BUDGET, owned)
            record.update(hash_basis='publisher_PEP658_sha256' if item['name'].endswith('.metadata') else 'publisher_sha256', publisher_sha256_verified=True)
            artifacts.append(record)
        for row in rows:
            path = root/mp.EVIDENCE/row['filename'] if row['name'] == 'mediapipe' else directory/row['filename']
            wheels.append(wheel_record(path, directory/(row['filename']+'.metadata'), row, started+BUDGET))
        report['status'] = 'pass'
    except Exception as exc:
        failure = exc; report['error_type'] = type(exc).__name__
        if isinstance(exc, mp.AcquisitionError): report['reason'] = str(exc)
    finally:
        if signal.getsignal(signal.SIGALRM) is cancel: signal.setitimer(signal.ITIMER_REAL, 0)
        mp.URLS = old_urls
        for part, inode in owned:
            if part.exists() and not part.is_symlink() and state(part)[:2] == inode: part.unlink()
        if created is not None and directory.exists() and not directory.is_symlink() and state(directory)[:2] == created: directory.chmod(0o555)
        try: report['source_rehashed_after'] = before is not None and source_binding(root, code, revision, verify_acquired=verify_acquired) == before
        except Exception: report['source_rehashed_after'] = False
        try: report['prior_rehashed_after'] = prior is not None and verify_prior(root, code/PRIOR_PINS) == prior
        except Exception: report['prior_rehashed_after'] = False
        try: report['artifacts_rehashed_after'] = all(identity(root/r['file']) == {k: r[k] for k in ('bytes', 'sha256')} for r in artifacts)
        except Exception: report['artifacts_rehashed_after'] = False
        if verify_acquired:
            try: report['downloads_rehashed_after'] = downloads is not None and verify_downloads(root, code) == downloads
            except Exception: report['downloads_rehashed_after'] = False
        report['elapsed_seconds'] = time.monotonic()-started; report['owned_partials_removed'] = all(not p.exists() for p, _ in owned)
        if not all(report[k] for k in ('source_rehashed_after', 'prior_rehashed_after', 'artifacts_rehashed_after')) or (verify_acquired and not report['downloads_rehashed_after']) or report['elapsed_seconds'] > BUDGET:
            report['status'] = 'fail'; failure = failure or mp.AcquisitionError('Source, prior artifacts, or inclusive budget failure')
        with (out/'report.json').open('x') as stream:
            json.dump(report, stream, indent=2, allow_nan=False); stream.write('\n')
            stream.flush(); os.fsync(stream.fileno()); os.fchmod(stream.fileno(), 0o444)
    if failure: raise mp.AcquisitionError('MediaPipe dependency acquisition failed; inspect sealed receipt') from None
    return report


def cancel(signum, frame):
    raise mp.AcquisitionError('Dependency acquisition cancelled or budget expired')


def main():
    require(platform.system() == 'Linux' and os.environ.get('WR_ROOT') == str(ROOT) and
            os.getuid() == pwd.getpwnam('scenesmith').pw_uid, 'Azure CPU scenesmith runtime required')
    require(sys.argv[1:] == ['--verify-acquired'], 'Only explicit verify-acquired entry enabled; no download replay')
    signal.signal(signal.SIGTERM, cancel); signal.signal(signal.SIGALRM, cancel); signal.setitimer(signal.ITIMER_REAL, BUDGET+5)
    try:
        require('WR_NAMESPACE_LEASE' not in os.environ, 'Verify-only cannot bootstrap an existing namespace')
        report = acquire(ROOT, Path(os.environ['WR_CODE']), os.environ['WR_CODE_REVISION'], verify_acquired=True)
        print(json.dumps({k: report[k] for k in ('stage', 'status', 'elapsed_seconds')}))
    finally: signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == '__main__':
    try: main()
    except Exception as exc:
        print(json.dumps(dict(stage='mediapipe_cpu_dependencies_acquisition', status='fail', error_type=type(exc).__name__)))
        raise SystemExit(1) from None
