"""Narrow Azure-only replica: original six person banks, two RGBs and source.

No model/masks/calibration/GT or local media transit. Private Azure identity is
RAM-only. Original files are authenticated before/after; imports never replace
existing files. This is byte replication, not an inference/quality experiment.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys
import tarfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mediapipe_cpu_runtime_verify as rt
from articulated_runtime_transfer import Blob, NoRedirect
import urllib.request

ROOT = Path('/srv/scenesmith/world-reward')
PRODUCER = '5e4255d192a657bec3428806ded272f457804a01'
ENTRY = 'run_person_bank_transfer'
REPORT = dict(bytes=24928, sha256='0d8227caf7d35df87547c3a7cd92cb89686d08372f22adc8ee3fd8b940aadac6')
BASE = 'results/person-pose-bank-probe-'+PRODUCER
OLD = 'jobs/'+PRODUCER+'/run_person_pose_bank_probe'
EXPECTED = ((9,0,2),(9,27,3),(9,55,3),(26,0,2),(26,26,2),(26,53,2))
MAXIMUM = 40 << 20
BUDGET = 300


class PersonBlob(Blob):
    def __init__(self, revision):
        rt.require(re.fullmatch('[0-9a-f]{40}', revision), 'Exact private transfer revision required')
        self.url = 'https://stworldrewardresearch26.blob.core.windows.net/runtime-transfers/person-banks-'+revision+'.tar'
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
        self.managed_identity = True; self.token = None; self.token_expiry = 0


def identity(path, maximum=MAXIMUM):
    return rt.identity(path, maximum, empty=True, readonly=False)


def original_files():
    path = ROOT/BASE/'report.json'; report = rt.pinned(path, REPORT, 32 << 10)
    rt.require(report['status'] == 'pass' and report['producer_revision'] == PRODUCER
        and report['stage'] == 'automatic_all_person_dwpose_diagnostic'
        and report['ground_truth_used'] is False and report['hand_labeled_test'] is False
        and report['source_inputs_assets_rehashed_after'] is True
        and report['all_recorded_person_banks_retained'] is True
        and tuple((x['episode'],x['frame_index'],x['persons']) for x in report['banks']) == EXPECTED,
        'Actual original automatic banks required')
    old = ROOT/OLD/'code'; source = report['source_binding']
    rt.require(rt.source(ROOT, old, PRODUCER, 'run_person_pose_bank_probe', tuple(source['helpers'])) == source,
               'Original full source/modes differ')
    rows = {BASE+'/report.json': REPORT}
    scope = rt.strict((old/'configs/person_pose_bank_probe_v1.json').read_bytes())
    for episode in scope['episodes']:
        name = episode['video']; pin = episode['files'][name]
        rt.require(report['input_pins'][name] == pin and identity(ROOT/name) == pin, 'Original Track1 RGB differs')
        rows[name] = pin
    for bank in report['banks']:
        name = BASE+'/'+bank['prediction_file']
        rt.require(identity(ROOT/name, 1 << 20) == bank['prediction'], 'Original bank differs')
        rows[name] = bank['prediction']
    for path in sorted((ROOT/OLD).rglob('*')):
        rt.canonical(path)
        s = path.lstat()
        rt.require(not s.st_mode & 0o222 and (stat.S_ISREG(s.st_mode) or stat.S_ISDIR(s.st_mode)), 'Original readonly source-only parent required')
        if path.is_file(): rows[str(path.relative_to(ROOT))] = identity(path, 2 << 20)
    rt.require(sum(x['bytes'] for x in rows.values()) < MAXIMUM-2*(1 << 20), 'Bounded two-RGB replica required')
    return rows, source


def write_receipt(out, value):
    rt.write(out/'report.json', (json.dumps(value,sort_keys=True,allow_nan=False)+'\n').encode(), 0o444)
    out.chmod(0o555)


def run(action, revision, expected=None, export_revision=None):
    started = time.monotonic()
    rt.require(os.geteuid() == 0 and os.uname().sysname == 'Linux'
        and os.uname().nodename == ('scenesmith-ncc-h100-01' if action == 'export' else 'world-reward-ncc-h100-02'), 'Exact Azure replica host required')
    code = Path(os.environ['WR_CODE'])
    own = rt.source(ROOT,code,revision,ENTRY,('infra/person_bank_transfer.py','infra/run_person_bank_transfer.sh','infra/articulated_runtime_transfer.py','infra/mediapipe_cpu_runtime_verify.py'))
    out = ROOT/('results/person-bank-transfer-'+action+'-'+revision)
    rt.require(not out.exists(), 'Fresh transfer namespace required'); out.mkdir(mode=0o700)
    archive_path = out/'replica.tar'
    value = dict(schema='world_reward.person_bank_replica.v1',status='fail',action=action,
        producer_revision=revision,source_binding=own,qualification_or_model_execution=False,
        local_heavy_transfer=False,ground_truth_used=False)
    try:
        if action == 'export':
            rows, source = original_files()
            manifest = dict(schema=value['schema'],original_producer_revision=PRODUCER,original_source_binding=source,files=rows,
                directories=[str(p.relative_to(ROOT)) for p in sorted((ROOT/OLD).rglob('*')) if p.is_dir()])
            raw = json.dumps(manifest,sort_keys=True,allow_nan=False).encode()
            rt.require(len(raw) <= 256 << 10, 'Bounded full replica manifest required')
            with tarfile.open(archive_path,'x',format=tarfile.USTAR_FORMAT) as tar:
                info=tarfile.TarInfo('manifest.json');info.size=len(raw);info.mode=0o444;tar.addfile(info,io.BytesIO(raw))
                for name,pin in sorted(rows.items()):
                    info=tarfile.TarInfo(name);info.size=pin['bytes'];info.mode=0o444
                    with (ROOT/name).open('rb') as stream:tar.addfile(info,stream)
            archive_path.chmod(0o444); pin=identity(archive_path)
            blob=PersonBlob(revision)
            with blob.request('PUT',archive_path.read_bytes(),headers={'x-ms-blob-type':'BlockBlob','If-None-Match':'*'}) as response:
                rt.require(response.status == 201,'Exclusive private replica upload required')
            rt.require(original_files() == (rows,source), 'Original artifacts changed after export')
            value.update(status='pass',archive=pin,files=len(rows),original_source_binding=source,original_files_unchanged=True)
        else:
            rt.require(type(expected) is dict and 0 < expected['bytes'] <= MAXIMUM and re.fullmatch('[0-9a-f]{64}', expected['sha256']), 'Independent exported archive pin required')
            blob=PersonBlob(export_revision)
            with blob.request('GET') as response,archive_path.open('xb') as stream:
                rt.require(int(response.headers['Content-Length']) == expected['bytes'], 'Exact private replica length required')
                total=0
                while True:
                    block=response.read(1 << 20)
                    if not block:break
                    total+=len(block);rt.require(total <= expected['bytes'] and time.monotonic()-started < BUDGET,'Bounded replica download required');stream.write(block)
                stream.flush();os.fsync(stream.fileno())
            archive_path.chmod(0o444);rt.require(identity(archive_path) == expected,'Independent exact archive differs')
            stage=out/'stage';stage.mkdir(mode=0o700)
            with tarfile.open(archive_path,'r:') as tar:
                members=tar.getmembers();rt.require(0 < len(members) <= 600 and len({m.name for m in members}) == len(members),'Bounded unique archive census required')
                first=members[0];rt.require(first.name == 'manifest.json' and first.isreg() and first.size <= 256 << 10,'First full replica manifest required')
                manifest=rt.strict(tar.extractfile(first).read());rows=manifest['files']
                rt.require(manifest['schema'] == value['schema'] and manifest['original_producer_revision'] == PRODUCER and rows[BASE+'/report.json'] == REPORT
                    and {m.name for m in members[1:]} == set(rows),'Exact original full archive inventory required')
                for m in members[1:]:
                    p=PurePosixPath(m.name)
                    rt.require(m.isreg() and not m.pax_headers and not p.is_absolute() and '..' not in p.parts
                        and (m.name.startswith(OLD+'/') or m.name.startswith(BASE+'/') or m.name in [f'data/track_1/videos/chunk-000/observation.images.exo_camera/episode_{e:06d}.mp4' for e in (9,26)])
                        and m.size == rows[m.name]['bytes'], 'Only original source/banks/twoRGB regular files allowed')
                    dst=stage/m.name;dst.parent.mkdir(parents=True,exist_ok=True)
                    with tar.extractfile(m) as src,dst.open('xb') as target:shutil.copyfileobj(src,target,1 << 20)
                    dst.chmod(0o444);rt.require(identity(dst,2 << 20 if m.name.startswith(OLD+'/') else MAXIMUM) == rows[m.name],'Every replica byte differs')
            for name in manifest.get('directories', []):
                p=PurePosixPath(name)
                rt.require(type(name) is str and p.as_posix() == name and not p.is_absolute() and '..' not in p.parts
                    and name.startswith(OLD+'/'), 'Only exact original source directories allowed')
                (stage/name).mkdir(parents=True,exist_ok=True)
            # Before publication, reject ALL collisions; never overwrite a user's file.
            rt.require(not (ROOT/OLD).exists() and not (ROOT/BASE).exists()
                and all(not (ROOT/name).exists() for name in rows),'Replica destinations must all be absent')
            for folder in (stage/OLD,stage/BASE):
                for p in sorted(folder.rglob('*'),reverse=True):
                    if p.is_dir():p.chmod(0o555)
                folder.chmod(0o555)
            for name in (OLD,BASE):
                dst=ROOT/name;rt.canonical(dst);dst.parent.mkdir(parents=True,exist_ok=True);os.rename(stage/name,dst)
            for name in rows:
                if name.startswith('data/'):
                    dst=ROOT/name;rt.canonical(dst);dst.parent.mkdir(parents=True,exist_ok=True);os.rename(stage/name,dst)
            actual_rows,source=original_files()
            rt.require(actual_rows == rows and source == manifest['original_source_binding'], 'Published full original source/banks/RGB differ')
            with blob.request('DELETE') as response:rt.require(response.status == 202,'Private transfer blob cleanup required')
            for p in sorted(stage.rglob('*'),reverse=True):
                rt.require(p.is_dir() and not p.is_symlink(),'Only owned empty staging directories removed');p.rmdir()
            stage.rmdir()
            value.update(status='pass',archive=expected,files=len(rows),original_source_binding=source,
                original_files_unchanged=True,private_transfer_blob_removed=True,owned_staging_removed=True)
        rt.require(rt.source(ROOT,code,revision,ENTRY,tuple(own['helpers'])) == own and time.monotonic()-started < BUDGET,'Bounded original transfer source required')
        archive_path.unlink();value['owned_archive_removed']=True
    except Exception as error:
        value.update(status='fail', error_type=type(error).__name__)
        if type(error) is ValueError and len(error.args)==1:value['requirement']=str(error.args[0])[:160]
    value['elapsed_seconds']=time.monotonic()-started
    if value['elapsed_seconds'] >= BUDGET: value.update(status='fail', requirement='Inclusive replica deadline exceeded')
    write_receipt(out,value)
    rt.require(value['status'] == 'pass','Immutable replica receipt failed')
    return value


def main():
    p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('action',choices=('export','import'))
    p.add_argument('--export-revision');p.add_argument('--archive-bytes',type=int);p.add_argument('--archive-sha256');a=p.parse_args()
    value=run(a.action,os.environ['WR_CODE_REVISION'],dict(bytes=a.archive_bytes,sha256=a.archive_sha256) if a.action=='import' else None,a.export_revision)
    print(json.dumps({k:value[k] for k in ('status','archive','files','elapsed_seconds')}))


if __name__=='__main__':
    try:main()
    except Exception as error:
        print(json.dumps(dict(status='fail',error_type=type(error).__name__)));raise SystemExit(1) from None
