"""Saved-only HO-Cap RGB extraction; immutable failed acquisition stays FAIL.

No network, inference or reference-value decoding. The independently measured
whole archives and exact central-directory census authorize this new namespace,
not a relaxation or rerun of the old acquisition. Raw metadata stays private.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
import re
import shutil
import signal
import stat
import sys
import time

ROOT = Path('/srv/scenesmith/world-reward')
ENTRY = 'run_hocap_extract'
PROTOCOL = 'configs/hocap_extraction_protocol_v1.json'
PROTOCOL_PIN = dict(bytes=2826, sha256='16605fa6a938fd4f7273743ee68f999db644d346464627e5edca434be827448c')
HELPERS = ('infra/hocap_extract.py', 'infra/run_hocap_extract.sh',
           'infra/hocap_acquire.py', 'infra/mediapipe_cpu_runtime_verify.py', PROTOCOL)
OLD_HELPERS = ('infra/hocap_acquire.py', 'infra/run_hocap_acquire.sh',
               'infra/mediapipe_cpu_runtime_verify.py', 'configs/hocap_acquisition_protocol_v1.json')


def runtime(code):
    sys.path.insert(0, str(Path(code)/'infra'))
    import hocap_acquire as h
    import mediapipe_cpu_runtime_verify as rt
    rt.require(Path(h.__file__).resolve() == code/'infra/hocap_acquire.py'
        and Path(rt.__file__).resolve() == code/'infra/mediapipe_cpu_runtime_verify.py',
        'Actual extraction helper origins required')
    return rt, h


def source_binding(rt, code, revision):
    own = rt.source(ROOT, code, revision, ENTRY, HELPERS)
    rt.require(Path(__file__).resolve() == code/'infra/hocap_extract.py', 'Actual caller source required')
    return rt.pinned(code/PROTOCOL, PROTOCOL_PIN, 16<<10), own


def authenticate(rt, h, code, p):
    """Hash old whole source/archives/primary text, without any ZIP payload read."""
    base = rt.canonical(Path(p['original'])); revision = p['producer_revision']
    original = ROOT/'jobs'/revision/'run_hocap_acquire/code'
    source = rt.source(ROOT, original, revision, 'run_hocap_acquire', OLD_HELPERS)
    rt.require(source['closure_sha256'] == p['source_closure_sha256'], 'Original complete source closure differs')
    old = rt.pinned(base/'report.json', p['report'], 64<<10)
    facts = dict(schema='world_reward.hocap_acquisition.v1', stage='hocap_rgb_private_reference_acquisition',
        status='fail', phase='safe_inventory', producer_revision=revision,
        error_type='ValueError', error_context='ZIP member cap exceeded', download_complete=True,
        source_archive_public_rehashed_after=True, public_inventory_qualified=False,
        label_values_parsed=False, calibration_values_parsed=False, model_loaded=False,
        gpu_used=False, challenge_inputs_used=False, adoption=False)
    rt.require(all(type(old.get(k)) is type(v) and old[k] == v for k,v in facts.items())
        and old.get('source_before') == source and old.get('protocol_identity') == p['original_protocol'],
        'Exact closed acquisition failure/ancestry required, never relabelled')
    prior = rt.pinned(original/'configs/hocap_acquisition_protocol_v1.json', p['original_protocol'], 16<<10)
    rt.require(prior['output'] == str(base) and all(prior[k] == p[k] for k in ('subject','camera','clips'))
        and p['maximum_zip_members'] == max(r['members'] for r in p['archives'].values()), 'Frozen source/selectors/census differ')
    # Reuse actual current helper bytes only when identical to the old operators.
    for name in ('infra/hocap_acquire.py','infra/mediapipe_cpu_runtime_verify.py'):
        rt.require(rt.identity(code/name, 2_000_000) == source['helpers'][name], 'Reused safety/metadata operators changed')
    files = {str(base/'report.json'): p['report']}
    rt.require(old.get('primary_sources') == p['primary_sources'], 'Original primary source ledger differs')
    for name,pin in p['primary_sources'].items():
        path = base/'primary'/name; rt.require(rt.identity(path, 64<<10) == pin, 'Primary source changed'); files[str(path)] = pin
    text = (base/'primary/publisher.html').read_text()
    rt.require('Creative Commons Attribution 4.0 International License (CC BY 4.0)' in text
        and 'creativecommons.org/licenses/by/4.0/' in text, 'Original explicit dataset grant required')
    pins = {name:{k:r[k] for k in ('bytes','sha256')} for name,r in p['archives'].items()}
    rt.require(old.get('archives') == pins and {r['file']:r['bytes'] for r in prior['archives']}
        == {n:r['bytes'] for n,r in pins.items()}, 'All five original archive identities required')
    for name,pin in pins.items():
        path = base/'quarantine'/name; s = path.lstat()
        rt.require(s.st_uid == p['source_owner_uid'] and stat.S_IMODE(s.st_mode) == 0o400
            and rt.identity(path, pin['bytes']) == pin, 'Original opaque archive mode/identity changed')
        files[str(path)] = pin
    return prior, dict(original_source=source, original_files=files, original_failure_preserved=True)


def extract(rt, h, code, revision, target, *, reservation=None, watchdog=False):
    started = time.monotonic(); p, source = source_binding(rt, code, revision)
    deadline = started+p['budget_seconds']; target = rt.canonical(Path(target))
    if reservation is None: rt.require(not target.exists(), 'Fresh extraction output only')
    else:
        s = target.lstat()
        rt.require(reservation == dict(device=s.st_dev,inode=s.st_ino,source_sha256=source['closure_sha256'])
            and stat.S_ISDIR(s.st_mode) and s.st_uid == os.getuid() and stat.S_IMODE(s.st_mode) == 0o700
            and not tuple(target.iterdir()), 'Exact fresh owned namespace lease required')
    rt.require(shutil.disk_usage(target.parent).free >= p['minimum_free_bytes'], 'Insufficient selected-extraction disk')
    if reservation is None: target.mkdir(mode=0o700)
    public = target/'inputs'; private = target/'metadata_private'
    public.mkdir(mode=0o755); private.mkdir(mode=0o700)
    r = dict(schema='world_reward.hocap_extraction.v1', stage='hocap_saved_rgb_private_metadata_extraction',
        status='fail', phase='original_authentication', producer_revision=revision, source_before=source,
        protocol_identity=PROTOCOL_PIN, archive_inventories={}, opaque_metadata={},
        public_inventory_qualified=False, network_used=False, label_values_parsed=False,
        calibration_values_parsed=False, model_loaded=False, gpu_used=False, challenge_inputs_used=False,
        image_decoder_qualified=False, reference_continuity_qualified=False, adoption=False)
    before = None; manifest = None; error = None
    if watchdog: signal.alarm(p['budget_seconds'])
    try:
        prior, before = authenticate(rt,h,code,p); r['original_ancestry'] = before; h.check(deadline)
        local = copy.deepcopy(prior); local['maximum_zip_members'] = p['maximum_zip_members']
        r['phase'] = 'safe_inventory'; selected = None
        for name,census in p['archives'].items():
            r['active_archive'] = name
            entries,audit = h.inventory(Path(p['original'])/'quarantine'/name, local, deadline)
            actual = dict(members=audit['members'],expanded_bytes=audit['expanded_bytes'],
                maximum_member_bytes=max((m.file_size for m in entries.values()),default=0))
            rt.require(actual == {k:census[k] for k in actual}, 'Exact independently frozen ZIP census differs')
            r['archive_inventories'][name] = audit
            if name == p['subject']+'.zip': selected = entries
            else: del entries
        r['phase'] = 'public_extract'
        manifest,_ = h.extract_public(Path(p['original'])/'quarantine'/(p['subject']+'.zip'), selected,
            local,public,private,deadline,metadata=r['opaque_metadata'])
        h.check(deadline); h.write_report(public/'manifest.json',manifest,0o444)
        r.update(public_inventory_qualified=True, frames=len(manifest['images']), clips=manifest['clips'],
            public_manifest=rt.identity(public/'manifest.json',16<<20))
    except BaseException as caught:
        error = caught; r.update(error_type=type(caught).__name__,error_context=str(caught)[:240]
            if isinstance(caught,(ValueError,TimeoutError)) else 'Saved extraction failed')
    finally:
        if watchdog: signal.alarm(p['cleanup_grace_seconds'])
        try:
            rt.require(source_binding(rt,code,revision) == (p,source), 'Extraction source changed')
            rt.require(before is not None and authenticate(rt,h,code,p)[1] == before, 'Original failed source/archives changed')
            for name,pin in r['opaque_metadata'].items():
                rt.require(rt.identity(private/name,64<<10) == pin, 'Retained raw metadata changed')
            if r['public_inventory_qualified']:
                for row in manifest['images']:
                    rt.require(rt.identity(public/row['file'],row['bytes']) == {k:row[k] for k in ('bytes','sha256')}, 'Selected original RGB changed')
                rt.require(rt.identity(public/'manifest.json',16<<20) == r['public_manifest'], 'Public manifest changed')
            r['source_archives_public_rehashed_after'] = True; h.check(deadline)
        except BaseException as caught:
            error = error or caught; r['post_error_type'] = type(caught).__name__
        r.update(status='pass' if error is None else 'fail',phase='complete' if error is None else r['phase'],
            elapsed_seconds=time.monotonic()-started,budget_seconds=p['budget_seconds'])
        for folder in public.iterdir():
            if folder.is_dir(): folder.chmod(0o555)
        public.chmod(0o555); target.chmod(0o755)
        h.write_report(target/'report.json',r,0o444,deadline=deadline)
        if watchdog: signal.alarm(0)
    return r


def main():
    import json
    if len(sys.argv) != 1 or sys.platform != 'linux' or os.geteuid() != 1000: raise ValueError('Azure UID1000 no-argument CPU caller only')
    code = Path(os.environ['WR_CODE']); revision = os.environ['WR_CODE_REVISION']
    if Path(os.environ['WR_ROOT']) != ROOT or not re.fullmatch('[0-9a-f]{40}',revision): raise ValueError('Frozen canonical root/revision required')
    rt,h = runtime(code)
    def expired(*_): raise TimeoutError('Inclusive600s extraction budget exhausted')
    signal.signal(signal.SIGALRM,expired); signal.signal(signal.SIGTERM,expired)
    p,_ = source_binding(rt,code,revision)
    r = extract(rt,h,code,revision,Path(p['output']),reservation=rt.strict(os.environ['WR_HOCAP_EXTRACTION_LEASE']),watchdog=True)
    print(json.dumps({k:r[k] for k in ('stage','status','public_inventory_qualified')}),flush=True)
    return 0 if r['status'] == 'pass' else 1


if __name__ == '__main__':
    try: raise SystemExit(main())
    except Exception as error:
        import json
        print(json.dumps(dict(stage='hocap_saved_extraction',status='fail',error_type=type(error).__name__)),flush=True)
        raise SystemExit(1) from None
