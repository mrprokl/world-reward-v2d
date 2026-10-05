"""Azure-only MMHOI archive census: names/lengths, zero member payloads.

No retries, complete archive, directory-container ZIP, labels, images or model.
Unknown Range support must fail before body reads if the server returns200.
"""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys
import time
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mediapipe_cpu_runtime_verify as rt
from zip_inventory import read_directory, parse_directory

ROOT = Path('/srv/scenesmith/world-reward')
ENTRY = 'run_mmhoi_inventory'
CONFIG = 'configs/mmhoi_inventory_v1.json'
CONFIG_V2 = 'configs/mmhoi_inventory_v2.json'
CONFIG_V3 = 'configs/mmhoi_inventory_v3.json'
HELPERS = ('infra/mmhoi_inventory.py', 'infra/zip_inventory.py',
           'infra/run_mmhoi_inventory.sh', 'infra/mediapipe_cpu_runtime_verify.py', CONFIG)


def pin(raw):
    return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise ValueError('Unexpected publisher redirect; no fallback download')


class Ranges:
    def __init__(self, cfg, deadline):
        self.cfg, self.deadline = cfg, deadline
        self.bytes, self.proofs, self.rejected_headers = 0, [], None
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())

    def check(self):
        rt.require(time.monotonic() < self.deadline, 'Inclusive census deadline reached')

    def get(self, start, end):
        self.check()
        cfg = self.cfg
        rt.require(0 <= start <= end < cfg['archive_bytes']
                   and self.bytes + end - start + 1 <= cfg['max_total_range_bytes'], 'Original range budget')
        parts = []
        for first in range(start, end + 1, cfg['max_request_bytes']):
            self.check()
            last = min(first + cfg['max_request_bytes'] - 1, end); n = last - first + 1
            req = urllib.request.Request(cfg['archive_url'], headers={
                'Range': f'bytes={first}-{last}', 'Accept-Encoding': 'identity',
                'If-Unmodified-Since': cfg['last_modified'],
            })
            with self.opener.open(req, timeout=min(cfg['request_timeout_seconds'], self.deadline - time.monotonic())) as response:
                h = response.headers
                wanted_range = f'bytes {first}-{last}/{cfg["archive_bytes"]}'
                if not (response.status == 206 and response.geturl() == cfg['archive_url']
                        and h.get('Content-Range') == wanted_range and h.get('Content-Length') == str(n)
                        and h.get('Content-Encoding', 'identity').lower() == 'identity'
                        and (h.get('Last-Modified') == cfg['last_modified']
                             or (cfg.get('range_last_modified_policy') == 'exact_or_absent_with_pinned_metadata_before_after'
                                 and h.get('Last-Modified') is None))):
                    self.rejected_headers = dict(status=response.status, requested_bytes=n,
                        header_sha256=hashlib.sha256(json.dumps({k:h.get(k) for k in
                            ('Content-Range','Content-Length','Content-Encoding','Last-Modified')}, sort_keys=True).encode()).hexdigest(),
                        body_read=False)
                    raise ValueError('Range identity/header firewall rejected before body read')
                raw = response.read(n + 1)
                self.bytes += len(raw)
                rt.require(len(raw) == n, 'Exact original range body required')
            self.proofs.append(dict(start=first, end=last, **pin(raw),
                                   publisher_last_modified_present=h.get('Last-Modified') is not None))
            parts.append(raw)
        self.check()
        return b''.join(parts)


def acquire_metadata(ranges):
    cfg = ranges.cfg; ranges.check()
    request = urllib.request.Request(cfg['metadata_url'], headers={'Accept-Encoding': 'identity'})
    with ranges.opener.open(request, timeout=cfg['request_timeout_seconds']) as response:
        rt.require(response.status == 200 and response.geturl() == cfg['metadata_url']
                   and response.headers.get('Content-Encoding', 'identity').lower() == 'identity', 'Exact publisher metadata response')
        raw = response.read(cfg['metadata_pin']['bytes'] + 1)
    rt.require(pin(raw) == cfg['metadata_pin'], 'Independently pinned publisher metadata changed')
    value = rt.strict(raw)
    rt.require(value.get('size') == cfg['archive_bytes']
               and value.get('checksum') == 'md5:' + cfg['publisher_md5']
               and value.get('version_id') == cfg['version_id']
               and value.get('file_id') == cfg['file_id']
               and value.get('bucket_id') == cfg['bucket_id']
               and value.get('links', {}).get('content') == cfg['archive_url'], 'Original version/file/archive identity required')
    return raw


def summarize(rows, cfg):
    extensions, sessions = {}, {}
    camera_zero, csv_count, excluded_count = 0, 0, 0
    for row in rows:
        if row['directory']:
            continue
        p = PurePosixPath(row['name']); suffix = p.suffix.lower()
        extensions[suffix] = extensions.get(suffix, 0) + 1
        if row['name'].startswith(cfg['exclude_schema_read_scenario']):
            excluded_count += 1
        parts = p.parts
        if len(parts) > 4 and parts[:2] == ('MMHOI', 'sequences'):
            key = '/'.join(parts[:4])
            session = sessions.setdefault(key, dict(files=0, camera0_jpegs=0, csv_files=0))
            session['files'] += 1
            if suffix == '.jpg' and p.name.startswith('0_'):
                session['camera0_jpegs'] += 1; camera_zero += 1
            if suffix == '.csv':
                session['csv_files'] += 1; csv_count += 1
    return dict(files=sum(not r['directory'] for r in rows), directories=sum(r['directory'] for r in rows),
                extensions=extensions, sessions=len(sessions), session_names_sha256=hashlib.sha256(
                    json.dumps(sorted(sessions), separators=(',', ':')).encode()).hexdigest(),
                camera0_JPEG_filename_count=camera_zero, sequence_CSV_filename_count=csv_count,
                schema_read_excluded_scenario_files=excluded_count), sessions


def run(config_name=CONFIG):
    started = time.monotonic()
    rt.require(os.geteuid() == 0 and os.uname().sysname == 'Linux'
               and os.uname().nodename == 'world-reward-ncc-h100-02', 'Exact Azure CPU host only')
    revision, code = os.environ['WR_CODE_REVISION'], Path(os.environ['WR_CODE'])
    rt.require(config_name in (CONFIG, CONFIG_V2, CONFIG_V3), 'Only frozen explicit inventory versions')
    helpers = (*HELPERS, config_name)
    source = rt.source(ROOT, code, revision, ENTRY, helpers)
    cfg = rt.strict((code / config_name).read_bytes())
    rt.require(cfg['schema'] in ('world_reward.mmhoi_inventory.v1', 'world_reward.mmhoi_inventory.v2',
                                'world_reward.mmhoi_inventory.v3') and cfg['entry'] == ENTRY
               and cfg['no_member_payload_reads'] is cfg['no_retries'] is cfg['no_directory_container_downloads'] is True,
               'Frozen metadata-only/no-retry scope')
    original_failure = None
    if config_name in (CONFIG_V2, CONFIG_V3):
        rt.require(cfg['schema'] == ('world_reward.mmhoi_inventory.v2' if config_name == CONFIG_V2 else 'world_reward.mmhoi_inventory.v3')
                   and cfg['publisher_metadata_required_before_after'] is True
                   and cfg['range_last_modified_policy'] == 'exact_or_absent_with_pinned_metadata_before_after',
                   'Explicit original-metadata-bound transport-v2 policy required')
        first = rt.strict((code / CONFIG).read_bytes())
        original_failure = Path(first['output']) / 'report.json'
        expected = {k:cfg['original_failure'][k] for k in ('bytes','sha256')}
        old = rt.pinned(original_failure, expected, 1 << 20)
        rt.require(old['status'] == 'fail' and old['producer_revision'] == cfg['original_failure']['producer_revision']
                   and old['range_bytes'] == 0 and old['member_payload_read'] is False
                   and old['RGB_read'] is False and old['outputs'] == {}, 'Original zero-body failure must remain closed')
    directory_failure = None
    if config_name == CONFIG_V3:
        second = rt.strict((code / CONFIG_V2).read_bytes())
        directory_failure = Path(second['output']) / 'report.json'
        directory_pin = {k:cfg['directory_budget_failure'][k] for k in ('bytes','sha256')}
        old = rt.pinned(directory_failure, directory_pin, 1 << 20)
        rt.require(old['status'] == 'fail' and old['producer_revision'] == cfg['directory_budget_failure']['producer_revision']
                   and old['range_bytes'] == 98 and old['member_payload_read'] is False
                   and old['CSV_values_read'] is False and old['RGB_read'] is False and old['outputs'] == {},
                   'Original pre-directory budget failure must remain closed')
    output = rt.canonical(cfg['output'])
    rt.require(not output.exists(), 'Exclusive new census namespace required')
    output.mkdir(mode=0o700)
    ranges = Ranges(cfg, started + cfg['budget_seconds'])
    report = dict(schema=cfg['schema'], producer_revision=revision, source_binding=source,
        status='fail', stage='source_metadata', configuration_identity=source['helpers'][config_name],
        scope=cfg['reference_scope'], archive_bytes=cfg['archive_bytes'], publisher_md5=cfg['publisher_md5'],
        version_id=cfg['version_id'], file_id=cfg['file_id'], budget_seconds=cfg['budget_seconds'],
        whole_archive_SHA_verified=False, member_CRC_verified=False, member_payload_read=False,
        CSV_values_read=False, RGB_read=False, calibration_read=False, mesh_read=False,
        GPU_used=False, models_loaded=False, cohort_selected=False, quality_verified=False,
        adopted=False, local_heavy_transfer=False, outputs={})
    try:
        metadata = acquire_metadata(ranges)
        report['publisher_metadata_identity'] = pin(metadata)
        report['stage'] = 'central_directory'
        raw, layout = read_directory(ranges.get, cfg['archive_bytes'], cfg['max_central_bytes'], cfg['max_members'],
            expected_layout=cfg['expected_trailer_layout'] if config_name == CONFIG_V3 else None)
        rows = parse_directory(raw, layout); ranges.check()
        counts, sessions = summarize(rows, cfg)
        # Version/file metadata is independently byte-pinned before AND after
        # ranges. This is not a CAS ETag or whole-archive authenticity proof.
        rt.require(acquire_metadata(ranges) == metadata, 'Publisher file/version metadata changed after ranges')
        report['publisher_metadata_pinned_before_after'] = True
        if original_failure is not None:
            rt.require(rt.identity(original_failure, 1 << 20) == expected, 'Original failure changed during transport-v2')
            report['original_failure_unchanged'] = True
        if directory_failure is not None:
            rt.require(rt.identity(directory_failure, 1 << 20) == directory_pin, 'Original directory-budget failure changed')
            report['directory_budget_failure_unchanged'] = True
        payload = dict(schema=cfg['schema'] + '.members', layout=layout, rows=rows)
        for name, data in (('publisher_file_metadata.json', metadata), ('central_directory.bin', raw),
                           ('members.json', (json.dumps(payload, sort_keys=True, separators=(',', ':')) + '\n').encode()),
                           ('sessions.json', (json.dumps(sessions, sort_keys=True, separators=(',', ':')) + '\n').encode())):
            ranges.check(); rt.write(output / name, data, 0o444)
            report['outputs'][name] = pin(data)
        rt.require(rt.source(ROOT, code, revision, ENTRY, helpers) == source, 'Original complete source changed')
        for name, expected in report['outputs'].items():
            rt.require(rt.identity(output / name, 512 << 20) == expected, 'Saved original census bytes changed')
        ranges.check()
        report.update(status='pass', stage='complete_metadata_census', counts=counts, layout=layout,
                      decision='INVENTORY_ONLY_SCHEMA_AND_RGB_ALIGNMENT_UNQUALIFIED', source_rehashed_after=True)
    except BaseException as exc:
        report['error_type'] = type(exc).__name__
        report['decision'] = 'CLOSED_METADATA_GATE_NO_RGB_OR_MODEL'
        raise
    finally:
        report.update(range_bytes=ranges.bytes, range_proofs=ranges.proofs,
                      rejected_headers=ranges.rejected_headers, elapsed_seconds=time.monotonic() - started)
        rt.write(output / 'report.json', (json.dumps(report, sort_keys=True, allow_nan=False) + '\n').encode(), 0o444)
        output.chmod(0o555)
        print(json.dumps({k:report[k] for k in ('status','stage','decision','range_bytes','elapsed_seconds')}, sort_keys=True))


if __name__ == '__main__':
    modes = {():CONFIG, ('--metadata-bound-range-v2',):CONFIG_V2,
             ('--dimensioned-directory-v3',):CONFIG_V3}
    rt.require(tuple(sys.argv[1:]) in modes, 'Only explicit frozen transport modes')
    run(modes[tuple(sys.argv[1:])])
