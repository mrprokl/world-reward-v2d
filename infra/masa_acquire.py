"""Bounded Azure acquisition of an authenticated MASA subset and opaque R50.

No upstream module, package manager, model, image or dataset is executed.
Apache declarations are evidence, not eligibility or training-overlap clearance.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import pwd
import re
import shutil
import signal
import stat
import sys
import time
import urllib.parse
import urllib.request

ROOT = Path('/srv/scenesmith/world-reward')
BASE = Path('/srv/world-reward-data/masa_native_v1')
ENTRY = 'run_masa_acquire'
MANIFEST = 'configs/masa_acquisition_v1.json'
REPORT = 'results/masa-acquisition-v1.json'
BUDGET, OUTER, MAXIMUM, TEXT_MAXIMUM, FREE, BLOCK = 600, 610, 600000000, 500000, 4 << 30, 1 << 20
SOURCE_REVISION = 'c5472b9c7615f35abdf1188cb1a0c5408fe50d66'
PUBLISHER_REVISION = '25ed372c47f2c46cf36fd446d1b657b656bc7ea9'
CHECKPOINT = dict(bytes=528391980, sha256='082670efc6e8820eff8257f78ea14dfb52d6cdbe2910ecccf0901a74f4a0fd76')
HELPERS = ('infra/masa_acquire.py', 'infra/run_masa_acquire.sh', MANIFEST,
           'infra/mediapipe_cpu_runtime_verify.py', 'infra/mediapipe_hands_acquire.py')
HELPER_PINS = {
 'infra/mediapipe_cpu_runtime_verify.py': dict(bytes=23559, sha256='936ad97c5ffca3b7f86e3462a600a247b6fa540d9b669e748892ff45e12aedf2'),
 'infra/mediapipe_hands_acquire.py': dict(bytes=21381, sha256='8293510c02777cd5b844865285736ff1a653631a8803849a9047ee9030196570')}
MANIFEST_PIN = dict(bytes=7148,sha256='488cbf89b15d1843ad1a6b578d77fbb12d7742a550a77f6885f26d337875974d')


def helpers(code):
    """Authenticate bytes and origin before importing the two stdlib helpers."""
    modules = []
    for name, pin in HELPER_PINS.items():
        p = Path(code)/name
        s = p.lstat()
        if (not p.is_absolute() or p.resolve() != p or any(q.is_symlink() for q in (p, *p.parents))
            or not stat.S_ISREG(s.st_mode) or s.st_nlink != 1 or s.st_mode & 0o222
            or s.st_size != pin['bytes'] or hashlib.sha256(p.read_bytes()).hexdigest() != pin['sha256']):
            raise ValueError('Pinned acquisition helper differs')
        after = p.lstat()
        if stable(s) != stable(after): raise ValueError('Pinned acquisition helper changed')
        spec = importlib.util.spec_from_file_location('wr_masa_'+p.stem, p)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        if Path(module.__file__) != p: raise ValueError('Imported helper origin differs')
        modules.append(module)
    return tuple(modules)


def stable(s):
    return tuple(getattr(s, k) for k in ('st_dev','st_ino','st_uid','st_gid','st_mode','st_nlink',
                                        'st_size','st_mtime_ns','st_ctime_ns'))


def binding(rt, root, code, revision):
    rt.require(rt.canonical(root) == ROOT and Path(__file__).resolve() == code/HELPERS[0], 'Actual acquisition entry required')
    rt.require(set(p.name for p in code.parent.iterdir()) == {'code','revision','source-sha256'}, 'Exact source snapshot parent required')
    proof = rt.source(root, code, revision, ENTRY, HELPERS)
    rt.require(all(proof['helpers'][n] == pin for n, pin in HELPER_PINS.items()), 'Accepted helper identity differs')
    return proof


def endpoint(url, *, initial=False, rows=()):
    try:
        p = urllib.parse.urlsplit(url); host = p.hostname or ''
        allowed = (p.scheme == 'https' and not p.username and not p.password and p.port in (None,443)
                   and not p.fragment and (host in ('huggingface.co','raw.githubusercontent.com')
                   or host.endswith(('.hf.co','.huggingface.co'))))
        if not allowed or (initial and (p.query or url not in {r['url'] for r in rows})):
            raise ValueError('Unlisted endpoint')
    except (ValueError, TypeError): raise ValueError('Unlisted public HTTPS endpoint') from None
    return url


class PublicRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        endpoint(newurl)
        if (urllib.parse.urlsplit(req.full_url).hostname == 'raw.githubusercontent.com'
            or urllib.parse.urlsplit(newurl).hostname == 'raw.githubusercontent.com'):
            raise ValueError('Cross-publisher/source redirect forbidden')
        if any(k.lower() in ('authorization','cookie') for k in req.headers):
            raise ValueError('Authenticated transport forbidden')
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def manifest(rt, code):
    p = code/MANIFEST
    pin = rt.identity(p, TEXT_MAXIMUM)
    rt.require(pin == MANIFEST_PIN, 'Frozen acquisition config differs')
    c = rt.strict(p.read_bytes()); rows = c.get('assets', [])
    rt.require(c.get('schema') == 'world_reward.masa_acquisition.v1'
               and c.get('source_revision') == SOURCE_REVISION and c.get('publisher_revision') == PUBLISHER_REVISION
               and c.get('source_scope') == 'authenticated_selected_native_R50_subset_not_whole_git_tree'
               and c.get('limits') == dict(budget_seconds=BUDGET,outer_seconds=OUTER,maximum_total_bytes=MAXIMUM,
                  maximum_text_bytes=TEXT_MAXIMUM,minimum_free_bytes=FREE,network_attempts_per_asset=1,parallel_downloads=1),
               'Frozen acquisition policy differs')
    rt.require(type(rows) is list and len(rows) >= 3 and len({r['file'] for r in rows}) == len(rows), 'Exact asset inventory required')
    source_url = 'https://raw.githubusercontent.com/siyuanliii/masa/'+SOURCE_REVISION+'/'
    publisher_url = 'https://huggingface.co/dereksiyuanli/masa/resolve/'+PUBLISHER_REVISION+'/'
    for r in rows:
        rt.require(type(r) is dict and set(r) == {'file','bytes','sha256','url','kind'}, 'Exact asset row required')
        p = PurePosixPath(r['file'])
        rt.require(not p.is_absolute() and '..' not in p.parts and p.as_posix() == r['file']
                   and len(p.parts) > 1 and p.parts[0] in ('source','publisher','weights')
                   and type(r['bytes']) is int and 0 < r['bytes'] <= (MAXIMUM if r['kind'] == 'opaque_checkpoint' else TEXT_MAXIMUM)
                   and re.fullmatch('[0-9a-f]{64}',r['sha256']), 'Bounded canonical asset row required')
        endpoint(r['url'], initial=True, rows=rows)
        rt.require(r['url'] == (source_url+'/'.join(p.parts[1:]) if p.parts[0] == 'source'
                               else publisher_url+'/'.join(p.parts[1:])), 'Pinned publisher path differs')
    rt.require(rows[0]['file'] == 'source/LICENSE' and rows[1]['file'] == 'publisher/README.md'
               and rows[-1]['file'] == 'weights/masa_r50.pth' and rows[-1]['kind'] == 'opaque_checkpoint'
               and {k:rows[-1][k] for k in ('bytes','sha256')} == CHECKPOINT
               and sum(r['bytes'] for r in rows) <= MAXIMUM
               and sum(r['kind'] == 'opaque_checkpoint' for r in rows) == 1
               and {'source/'+n for n in c['native_source_files']} <= {r['file'] for r in rows}, 'Notice-first native subset/weight identity differs')
    return c, pin


def fetch(rt, mp, base, row, rows, opener, deadline, partials):
    rt.require(time.monotonic() < deadline, 'Inclusive acquisition deadline')
    target = rt.canonical(base/row['file']); part = target.with_name(target.name+'.part')
    req = urllib.request.Request(endpoint(row['url'], initial=True, rows=rows), headers={'Accept-Encoding':'identity'})
    with opener.open(req, timeout=min(20,max(.001,deadline-time.monotonic()))) as response:
        endpoint(response.geturl())
        rt.require(response.status == 200 and response.headers.get('Content-Encoding','identity').lower() == 'identity'
                   and response.headers.get_content_type().lower() not in ('text/html','application/xhtml+xml'), 'Publisher transport differs')
        if row['file'].startswith('source/'):
            rt.require(response.geturl() == row['url'], 'Source redirect forbidden')
        length = response.headers.get('Content-Length')
        rt.require(length is None or re.fullmatch('[0-9]+',length) and int(length) == row['bytes'], 'Publisher length differs')
        h = hashlib.sha256(); count = 0
        with part.open('xb') as f:
            os.fchmod(f.fileno(),0o600); s = part.lstat(); partials.append((part,(s.st_dev,s.st_ino,s.st_uid)))
            while True:
                rt.require(time.monotonic() < deadline, 'Inclusive acquisition deadline')
                block = response.read(min(BLOCK,row['bytes']-count+1))
                rt.require(count+len(block) <= row['bytes'] and len(block) <= BLOCK, 'Publisher body overflow')
                if not block: break
                if count == 0:
                    rt.require(not block.lstrip().lower().startswith((b'<html',b'<!doctype html'))
                               and not block.startswith(b'version https://git-lfs.github.com/spec/'), 'HTML or LFS pointer rejected')
                f.write(block); h.update(block); count += len(block)
            rt.require(count == row['bytes'] and h.hexdigest() == row['sha256'], 'Publisher byte/SHA differs')
            f.flush(); os.fsync(f.fileno()); os.fchmod(f.fileno(),0o444)
        rt.require(time.monotonic() < deadline, 'Inclusive acquisition deadline')
        mp.publish(part,target)
    return dict(file=row['file'],bytes=count,sha256=h.hexdigest(),kind=row['kind'],
                hash_basis='publisher_git_lfs_sha256' if row['kind'] == 'opaque_checkpoint' else 'independent_primary_text_sha256')


def notices(rt, base):
    license_text = (base/'source/LICENSE').read_bytes()
    card = (base/'publisher/README.md').read_text(encoding='utf-8')
    rt.require(b'Apache License' in license_text and b'Version 2.0' in license_text
               and card.startswith('---\n') and 'license: apache-2.0' in card.split('---',2)[1]
               and 'library_name: masa' in card.split('---',2)[1], 'Pinned source/card declarations differ')
    return dict(source='Apache-2.0',publisher_card='apache-2.0',eligibility_verified=False,training_overlap_status='unknown')


def posthash(rt, root, code, revision, before, completed, completed_stats, report):
    rt.require(binding(rt,root,code,revision) == before, 'Source changed after acquisition')
    report['source_rehashed_after'] = True
    rt.require(all(stable(p.lstat()) == completed_stats[p] and rt.identity(p,MAXIMUM) == pin
                   and stable(p.lstat()) == completed_stats[p] for p,pin in completed.items()), 'Acquired artifacts changed')
    report['artifacts_rehashed_after'] = True


def acquire(root, code, revision, *, opener=None, namespace_lease=None, started=None):
    now = time.monotonic(); started = now if started is None else started
    if not isinstance(started,(int,float)) or not 0 < started <= now or not now-started < BUDGET:
        raise ValueError('Original bounded acquisition start required')
    deadline = started+BUDGET
    rt, mp = helpers(code); root = rt.canonical(root); code = rt.canonical(code)
    before = binding(rt,root,code,revision); config, config_pin = manifest(rt,code); rows = config['assets']
    base = rt.canonical(BASE); out = rt.canonical(root/REPORT)
    rt.require(base.parent.is_dir() and out.parent.is_dir() and not out.exists()
               and (namespace_lease is not None or not base.exists()), 'Fresh namespace/receipt only; no resume')
    report = dict(schema='world_reward.masa_acquisition_receipt.v1',stage='masa_native_r50_source_weight_acquisition',
      status='fail',phase='namespace',producer_revision=revision,source_binding=before,config_identity=config_pin,
      source_scope=config['source_scope'],budget_seconds=BUDGET,outer_seconds=OUTER,
      budget_scope='acquisition_checks_public_sealing_posthash',receipt_publication_grace_seconds=OUTER-BUDGET,
      assets=[],source_rehashed_after=False,artifacts_rehashed_after=False,public_sealed=False,owned_partials_removed=False,
      models_loaded=False,weights_decoded=False,packages_installed=False,rgb_or_datasets_used=False,ground_truth_used=False,
      gpu_used=False,challenge_inputs_used=False,oracle_modes=[],quality_evaluated=False,adopted=False,
      license_eligibility_verified=False,training_overlap_verified=False)
    dirs = []; partials = []; completed = {}; completed_stats = {}; failure = None
    try:
        rt.require(shutil.disk_usage(base.parent).free >= FREE, 'Minimum free space unavailable')
        if namespace_lease is not None:
            dirs = [(p,(di[0],di[1],os.getuid())) for p,di in mp.validate_namespace_lease(namespace_lease,[base],before['closure_sha256'])]
            report['namespace_lease'] = namespace_lease
        else:
            base.mkdir(mode=0o700); s = base.lstat(); dirs.append((base,(s.st_dev,s.st_ino,s.st_uid)))
        for row in rows:
            p = base/row['file']; parents = []
            while p.parent != base:
                p = p.parent; parents.append(p)
            for parent in reversed(parents):
                if parent.exists(): continue
                parent.mkdir(mode=0o700); s = parent.lstat(); dirs.append((parent,(s.st_dev,s.st_ino,s.st_uid)))
        opener = opener or urllib.request.build_opener(urllib.request.ProxyHandler({}),PublicRedirect())
        report['phase'] = 'download'
        for row in rows:
            if row['kind'] == 'opaque_checkpoint': report['upstream_license_declarations'] = notices(rt,base)
            asset = fetch(rt,mp,base,row,rows,opener,deadline,partials)
            report['assets'].append(asset); completed[base/row['file']] = {k:asset[k] for k in ('bytes','sha256')}
            completed_stats[base/row['file']] = stable((base/row['file']).lstat())
        report['status'] = 'pass'; report['phase'] = 'complete'
    except BaseException as exc:
        failure = 'deadline_or_cancellation' if isinstance(exc,(TimeoutError,KeyboardInterrupt)) else 'acquisition_check_failed'
    finally:
        # Prevent SIGALRM interrupting integrity checks/sealing; elapsed still gates PASS.
        signal.setitimer(signal.ITIMER_REAL,0)
        try:
            for p, key in partials:
                if not p.exists() and not p.is_symlink(): continue
                s = p.lstat()
                rt.require(rt.canonical(p) == p and stat.S_ISREG(s.st_mode) and s.st_nlink == 1
                           and (s.st_dev,s.st_ino,s.st_uid) == key, 'Owned partial identity differs')
                p.unlink()
            report['owned_partials_removed'] = True
            allowed = {p for p,_ in dirs} | set(completed)
            for p,key in dirs:
                s = p.lstat()
                rt.require(rt.canonical(p) == p and stat.S_ISDIR(s.st_mode) and (s.st_dev,s.st_ino,s.st_uid) == key
                           and set(p.iterdir()) <= allowed, 'Owned namespace changed')
            for p,_ in reversed(dirs): p.chmod(0o555)
            report['public_sealed'] = bool(dirs)
        except BaseException:
            failure = 'cleanup_or_posthash_failed'
        try:
            posthash(rt,root,code,revision,before,completed,completed_stats,report)
        except BaseException:
            failure = 'cleanup_or_posthash_failed'
        elapsed = time.monotonic()-started
        if elapsed >= BUDGET: failure = 'inclusive_deadline_exceeded'
        report['elapsed_seconds'] = elapsed
        if failure is not None: report.update(status='fail',error=failure)
        raw = (json.dumps(report,sort_keys=True,allow_nan=False)+'\n').encode()
        rt.require(len(raw) <= 100000, 'Bounded acquisition receipt required')
        rt.write(out,raw,0o444)
    if report['status'] != 'pass': raise ValueError('Acquisition failed; bounded immutable receipt retained') from None
    return report


def cancelled(*_): raise TimeoutError('Bounded acquisition cancellation')


def main():
    if (len(sys.argv) != 1 or sys.platform != 'linux' or os.geteuid() != 1000
        or pwd.getpwnam('scenesmith').pw_uid != 1000 or os.environ.get('WR_ROOT') != str(ROOT)):
        raise ValueError('Exact Azure unprivileged acquisition required')
    code = Path(os.environ['WR_CODE']); revision = os.environ['WR_CODE_REVISION']
    started = float(os.environ['WR_ACQUISITION_STARTED']); remaining = started+BUDGET-time.monotonic()
    if not 0 < remaining <= BUDGET: raise ValueError('Original acquisition deadline required')
    signal.signal(signal.SIGALRM,cancelled); signal.signal(signal.SIGTERM,cancelled); signal.signal(signal.SIGINT,cancelled)
    signal.setitimer(signal.ITIMER_REAL,remaining)
    try:
        rt,_ = helpers(code)
        lease = rt.strict(os.environ['WR_NAMESPACE_LEASE']) if 'WR_NAMESPACE_LEASE' in os.environ else None
        result = acquire(ROOT,code,revision,namespace_lease=lease,started=started)
        print(json.dumps({k:result[k] for k in ('stage','status','elapsed_seconds')}))
    finally: signal.setitimer(signal.ITIMER_REAL,0)


if __name__ == '__main__':
    try: main()
    except BaseException:
        print(json.dumps(dict(stage='masa_native_r50_source_weight_acquisition',status='fail',error='bounded_acquisition_failure')))
        raise SystemExit(1) from None
