"""Two fresh EPECK/ASan processes; a numerical diagnostic, never mesh adoption."""
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import selectors
import shutil
import signal
import stat
import subprocess
import sys
import time

ROOT = Path('/srv/scenesmith/world-reward')
ENTRY = 'run_cgal_sum_control'
PINS = 'configs/certified_solid_qualification_pins.json'
HELPERS = ('infra/cgal_sum_control.py', 'infra/run_cgal_sum_control.sh', 'infra/cgal_sum_control.cpp',
           'infra/certified_solid_build.py', 'infra/certified_solid_source_job.py', PINS,
           'configs/certified_solid_protocol_v1.json')
FLAGS = ['-O1', '-g', '-std=c++17', '-fsanitize=address', '-fno-omit-frame-pointer',
         '-fno-optimize-sibling-calls', '-ffp-contract=off', '-frounding-math', '-fno-fast-math']
TERMS, TOTAL, COMPILE, ARM = 131072, 600, 180, 60
PHASES = ('before_build', 'before_exact_force', 'exact_force_complete', 'before_teardown', 'teardown_complete')
HEADER_PINS = {'CGAL/Lazy_exact_nt.h':dict(bytes=46566,sha256='a02d707ddf05cdd14126dca87f74181ebad8398c7c7f6a25a18ede49c9739576'),
    'CGAL/Exact_predicates_exact_constructions_kernel.h':dict(bytes=2647,sha256='98a9963c9460978f735edcb055fa1a511023d3125663721bbc1542bca72638ff')}
RIGHTS = dict(glue_spdx='Apache-2.0',linked_cgal_scope='GPL-3.0-or-later OR commercial license',
              binary_is_apache_only=False,competition_eligibility_verified=False)


def require(ok, message):
    if not ok: raise ValueError(message)


def helpers(code):
    require(code.is_absolute() and code.resolve() == code and not any(p.is_symlink() for p in (code, *code.parents)), 'Canonical source required')
    sys.path.insert(0, str(code/'infra'))
    import certified_solid_build as build
    import certified_solid_source_job as certificate
    require(Path(build.__file__).resolve() == code/'infra/certified_solid_build.py' and
            Path(certificate.__file__).resolve() == code/'infra/certified_solid_source_job.py', 'Actual immutable helpers required')
    return build, certificate


def binding(code, revision, build):
    require(re.fullmatch('[0-9a-f]{40}', revision) and code == ROOT/'jobs'/revision/ENTRY/'code' and
            Path(__file__).resolve() == code/'infra/cgal_sum_control.py', 'Actual new diagnostic entry required')
    rows = {}; ledger = hashlib.sha256()
    for p in (code, *sorted(code.rglob('*'))):
        require(p.resolve() == p and not p.is_symlink() and not p.lstat().st_mode & 0o222, 'Canonical readonly whole source required')
        mode = stat.S_IMODE(p.lstat().st_mode)
        if p.is_dir(): require(mode == 0o555, 'Immutable source directory mode differs'); continue
        require(mode in (0o444,0o555), 'Immutable source file mode differs')
        rows[p.relative_to(code).as_posix()] = build.identity(p, readonly=True, maximum=2 << 20, empty=True)
        ledger.update(p.relative_to(code).as_posix().encode()+b'\0'+str(mode).encode()+b'\0'+bytes.fromhex(rows[p.relative_to(code).as_posix()]['sha256']))
    require(set(HELPERS) <= set(rows) and {p.name for p in code.parent.iterdir()} == {'code','revision','source-sha256'}, 'Complete immutable snapshot required')
    markers = {n: build.identity(code.parent/n, readonly=True, maximum=128) for n in ('revision','source-sha256')}
    require((code.parent/'revision').read_bytes() == (revision+'\n').encode() and
            re.fullmatch(b'[0-9a-f]{64}\n', (code.parent/'source-sha256').read_bytes()), 'Original dispatch markers differ')
    require(build.strict_json((code/build.CONFIG).read_bytes()) == build.EXPECTED, 'Frozen original CGAL acquisition differs')
    return dict(producer_revision=revision, markers=markers, helpers={n: rows[n] for n in HELPERS},
                files=len(rows), files_sha256=hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),readonly_ledger_sha256=ledger.hexdigest())


def left(start, cap=TOTAL):
    value = cap - (time.monotonic()-start); require(value > 0, 'Inclusive numerical diagnostic deadline'); return value


def command(argv, seconds):
    """Drain bounded-memory pipes; retain only a compact sanitizer excerpt."""
    start = time.monotonic(); data = {0:bytearray(),1:bytearray()}; counts = [0,0]; digests = [hashlib.sha256(),hashlib.sha256()]; tail = bytearray()
    process = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    timed_out = False
    try:
        with selectors.DefaultSelector() as selector:
            for i, stream in enumerate((process.stdout,process.stderr)): selector.register(stream,selectors.EVENT_READ,i)
            while selector.get_map():
                if time.monotonic()-start >= seconds and not timed_out:
                    timed_out = True; os.killpg(process.pid,signal.SIGKILL)
                for key,_ in selector.select(.1):
                    block = os.read(key.fileobj.fileno(),8192)
                    if not block: selector.unregister(key.fileobj); continue
                    i = key.data; counts[i] += len(block); digests[i].update(block)
                    capacity = (8192,131072)[i]; data[i].extend(block[:max(0,capacity-len(data[i]))])
                    if i == 1: tail.extend(block); del tail[:-1000]
            returncode = process.wait(timeout=5)
    finally:
        if process.poll() is None: os.killpg(process.pid,signal.SIGKILL); process.wait(timeout=5)
        process.stdout.close(); process.stderr.close()
    stderr = bytes(data[1]).decode('utf-8',errors='replace')
    trace = [line[:400] for line in stderr.splitlines() if 'AddressSanitizer' in line or
             re.search(r'#\d+.*(?:Lazy_exact|Handle_for|Lazy_rep|Handle<)',line)][:72]
    return dict(returncode=returncode,timed_out=timed_out,stdout=bytes(data[0]).decode('utf-8',errors='replace'),
                stdout_bytes=counts[0],stderr_bytes=counts[1],stderr_sha256=digests[1].hexdigest(),
                sanitizer_excerpt=trace,stderr_prefix_truncated=counts[1]>131072,elapsed_seconds=time.monotonic()-start,
                failure_tail=bytes(tail).decode('utf-8',errors='replace') if returncode != 0 else '')


def arm_result(result, arm, build):
    result.pop('failure_tail',None)  # Compile failures only: no full native stderr retained.
    require(arm in ('sequential','balanced') and result['stdout_bytes'] <= 8192 and type(result['returncode']) is int and type(result['timed_out']) is bool, 'Bounded actual process result required')
    rows = [build.strict_json(line) for line in result.pop('stdout').splitlines()]
    for i,row in enumerate(rows):
        require(type(row) is dict and set(row) == {'arm','terms','phase','exact_zero_verified'} and
                row['arm'] == arm and type(row['terms']) is int and row['terms'] == TERMS and
                i < len(PHASES) and row['phase'] == PHASES[i] and type(row['exact_zero_verified']) is bool and
                row['exact_zero_verified'] == (i >= 2), 'Exact ordered native phase/term proof required')
    require(rows, 'Native process never entered the experiment')
    result['phases'] = rows
    result['exact_zero_verified'] = len(rows) == len(PHASES) and result['returncode'] == 0 and not result['timed_out']
    return result


def decision(arms):
    sequential, balanced = arms['sequential'], arms['balanced']; trace = sequential['sanitizer_excerpt']
    overflow = any('AddressSanitizer' in line and 'stack-overflow' in line for line in trace)
    frames = sum(bool(re.search(r'#\d+.*(?:Lazy_exact|Handle_for|Lazy_rep|Handle<)',line)) for line in trace)
    diagnosed = (balanced['exact_zero_verified'] and not sequential['timed_out'] and sequential['returncode'] != 0 and
                 sequential['phases'][-1]['phase'] in ('before_exact_force','before_teardown') and overflow and frames >= 3)
    return dict(status='pass' if diagnosed else 'inconclusive',lazy_exact_stack_overflow_supported=diagnosed,
                observed_sequential_stack_frames=frames,diagnosed_phase=sequential['phases'][-1]['phase'] if diagnosed else None,
                production_failure_explained=False,adoption=False)


def header_ledger(headers, build):
    rows = {}
    for p in (headers,*sorted(headers.rglob('*'))):
        require(p.resolve() == p and not p.is_symlink(), 'Owned header path differs')
        if p.is_dir(): continue
        rows[p.relative_to(headers).as_posix()] = build.identity(p, readonly=True, maximum=16 << 20, empty=True)
    require(all(rows.get(n) == pin for n,pin in HEADER_PINS.items()), 'Pinned primary exact-arithmetic headers differ')
    return hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()


def native(code, revision, work, headers, build):
    start = time.monotonic(); before = binding(code,revision,build)
    require(sys.platform == 'linux' and os.getuid() == 1000 and os.environ.get('WR_NATIVE_NETWORK') == 'none' and
            os.environ.get('CUDA_VISIBLE_DEVICES') == '-1' and {p.name for p in Path('/sys/class/net').iterdir()} == {'lo'} and
            work.resolve() == work and not any(p.is_symlink() for p in (work,*work.parents)) and
            work.stat().st_uid == 1000 and stat.S_IMODE(work.stat().st_mode) == 0o700 and not any(work.iterdir()), 'Fresh offline CPU native context required')
    pins = build.strict_json((code/PINS).read_bytes()); require(os.environ.get('WR_CPU_IMAGE_ID') == pins['child_image_id'], 'Qualified CPU image required')
    h = header_ledger(headers,build); report = dict(stage='cgal_sum_control_native_v1',status='fail',phase='compile',source_binding=before,
        image_id=pins['child_image_id'],terms=TERMS,compiler_flags=FLAGS,budget_seconds=TOTAL,compile_seconds=COMPILE,arm_seconds=ARM,
        primary_headers=HEADER_PINS,rights=RIGHTS,gpu_used=False,challenge_inputs_used=False,gt_used=False,adoption=False,production_failure_explained=False)
    try:
        binary = work/'sum-control'; args = ['c++',*FLAGS,'-I'+str(headers),str(code/'infra/cgal_sum_control.cpp'),'-o',str(binary),'-lgmpxx','-lgmp','-lmpfr']
        compile_result = command(args,min(COMPILE,left(start))); report['compile'] = {k:v for k,v in compile_result.items() if k != 'stdout'}
        require(compile_result['returncode'] == 0 and not compile_result['timed_out'], 'Bounded ASan compilation failed')
        report['binary'] = build.identity(binary); report['phase'] = 'arms'; report['arms'] = {}
        for arm in ('sequential','balanced'):
            report['arms'][arm] = arm_result(command([str(binary),arm],min(ARM,left(start))),arm,build)
        report.update(decision(report['arms']),phase='complete')
        require(build.identity(binary) == report['binary'], 'Owned control binary changed')
    except Exception as error: report['failure_type'] = type(error).__name__
    finally:
        try:
            report['source_binding_after'] = binding(code,revision,build)
            require(report['source_binding_after'] == before and header_ledger(headers,build) == h, 'Source/header bytes changed')
            report.update(source_rehashed_after=True,headers_rehashed_after=True)
        except Exception as error: report.update(status='fail',posthash_failure_type=type(error).__name__)
        report['elapsed_seconds'] = time.monotonic()-start
        if report['elapsed_seconds'] > TOTAL: report.update(status='fail',failure_type='NativeInclusiveDeadline')
        build.write_json(work/'native.json',report)
    return 0 if report['status'] in ('pass','inconclusive') else 1


def cleanup_scratch(scratch, owner):
    info = scratch.lstat(); require((info.st_dev,info.st_ino) == owner and not scratch.is_symlink(), 'Owned scratch root changed')
    nodes = list(scratch.rglob('*'))
    for p in nodes:
        s = p.lstat(); require(not p.is_symlink() and (stat.S_ISDIR(s.st_mode) or stat.S_ISREG(s.st_mode) and s.st_nlink == 1), 'Foreign scratch entry; refuse deletion')
    for p in (scratch,*nodes):
        if p.is_dir(): p.chmod(0o755)
    shutil.rmtree(scratch)


def host(code, revision, build, certificate):
    start = time.monotonic(); before = binding(code,revision,build)
    require(sys.platform == 'linux' and os.getuid() == 0 and os.environ.get('DOCKER_HOST') == 'unix://'+str(ROOT/'docker.sock'), 'Owned Linux Docker CPU host required')
    pins, paths, qualified = certificate.qualification(code,build); image = pins['child_image_id']
    require(build.run(['docker','image','inspect',image,'--format','{{.Id}}'],min(10,left(start))).decode().strip() == image, 'Actual qualified CPU image differs')
    name = 'wr-cgal-sum-'+revision; out = ROOT/'results'/('cgal-sum-control-'+revision)
    require(out.resolve() == out and not any(p.is_symlink() for p in (out,*out.parents)) and out.parent.is_dir() and
            not out.exists() and not build.run(['docker','ps','-aq','--filter','name=^/'+name+'$'],min(10,left(start))).strip(), 'Fresh result/container required')
    out.mkdir(mode=0o755); out.chmod(0o755); scratch = out/'disposable'; scratch.mkdir(mode=0o755); scratch.chmod(0o755)
    owner = (scratch.stat().st_dev,scratch.stat().st_ino); work = scratch/'work'; work.mkdir(mode=0o700); os.chown(work,1000,1000)
    cid = out/'.container.cid'; report = dict(stage='cgal_sum_control_host_v1',status='fail',phase='header_acquisition',producer_revision=revision,
        source_binding=before,qualified_runtime=qualified,image_id=image,budget_seconds=TOTAL,gpu_used=False,gt_used=False,
        primary_headers=HEADER_PINS,rights=RIGHTS,challenge_inputs_used=False,adoption=False,production_failure_explained=False)
    old = {s:signal.signal(s,lambda *_: (_ for _ in ()).throw(TimeoutError('Inclusive CPU deadline'))) for s in (signal.SIGALRM,signal.SIGTERM)}
    signal.alarm(max(1,int(left(start))))
    try:
        archive,sums = scratch/'cgal.tar.xz',scratch/'sha256sum.txt'
        for pin,path in ((build.CGAL,archive),(build.SUM,sums)): build.download(pin,path,min(start+TOTAL,time.monotonic()+120))
        require(sum(row.split() in ([build.CGAL['sha256'],'CGAL-6.0.1-library.tar.xz'],[build.CGAL['sha256'],'*CGAL-6.0.1-library.tar.xz']) for row in sums.read_text().splitlines()) == 1, 'Publisher checksum concordance differs')
        headers = scratch/'headers'; report['headers'] = build.extract_headers(archive,headers,build.EXPECTED['capacities']); ledger = header_ledger(headers,build)
        for p in (headers,*headers.rglob('*')):
            if p.is_dir(): p.chmod(0o555)
        mounts = []
        for p in (code,code.parent/'revision',code.parent/'source-sha256'): mounts += ['--mount',f'type=bind,src={p},dst={p},readonly']
        report['phase'] = 'native_compile_and_arms'
        args = ['docker','run','--name',name,'--cidfile',str(cid),'--label','world_reward.certified_solid.owner='+revision,
            '--network','none','--read-only','--user','1000:1000','--cap-drop','ALL','--security-opt','no-new-privileges',
            '--cpus','4','--memory','16g','--tmpfs','/tmp:rw,nosuid,nodev,noexec,size=256m',*mounts,
            '--mount',f'type=bind,src={headers},dst=/opt/wr-cgal/include,readonly','--mount',f'type=bind,src={work},dst={work}',
            '--entrypoint','/usr/bin/env',image,'-i','PATH=/opt/conda/bin:/usr/local/bin:/usr/bin:/bin','HOME=/tmp','TMPDIR=/tmp',
            'ASAN_OPTIONS=detect_leaks=0:abort_on_error=1:symbolize=1','CUDA_VISIBLE_DEVICES=-1','WR_NATIVE_NETWORK=none',
            'WR_ROOT='+str(ROOT),'WR_CODE='+str(code),'WR_CODE_REVISION='+revision,'WR_CPU_IMAGE_ID='+image,
            'PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','python3','-I','-B',str(code/'infra/cgal_sum_control.py'),'--native']
        build.run(args,left(start),scratch/'native.log')
    except Exception as error: report['failure_type'] = type(error).__name__
    finally:
        signal.alarm(30)
        try:
            build.cleanup_container(name,image,revision,cid); report['owned_container_removed'] = True
            require(binding(code,revision,build) == before and certificate.qualification(code,build)[2] == qualified and
                    build.run(['docker','image','inspect',image,'--format','{{.Id}}'],10).decode().strip() == image, 'Source/qualified runtime changed')
            report.update(source_binding_after=before,source_rehashed_after=True,qualified_runtime_rehashed_after=True)
            if (work/'native.json').is_file():
                raw = (work/'native.json').read_bytes(); require(len(raw) <= 2 << 20, 'Bounded native receipt required'); result = build.strict_json(raw); report['native'] = result
                require(result['source_binding'] == result['source_binding_after'] == before and result['source_rehashed_after'] is result['headers_rehashed_after'] is True, 'Native source/header postproof differs')
                require(result['stage'] == 'cgal_sum_control_native_v1' and result['terms'] == TERMS and result['image_id'] == image, 'Actual native control differs')
                require(header_ledger(headers,build) == ledger and all(build.identity(p) == {k:pin[k] for k in ('bytes','sha256')} for pin,p in ((build.CGAL,archive),(build.SUM,sums))), 'Acquired public headers changed')
                build.seal(out/'native.json',raw)
                require(result['status'] in ('pass','inconclusive') and result['phase'] == 'complete' and 'failure_type' not in report and
                        result['elapsed_seconds'] <= TOTAL and result['compiler_flags'] == FLAGS and
                        all(result[k] is False for k in ('gpu_used','gt_used','challenge_inputs_used','adoption','production_failure_explained')), 'Completed diagnostic proof required')
                require(decision(result['arms'])['status'] == result['status'], 'Attribution decision differs'); report.update(decision(result['arms']),phase='complete')
            else: raise ValueError('Native process produced no receipt')
        except Exception as error: report.update(status='fail',posthash_failure_type=type(error).__name__)
        try:
            cleanup_scratch(scratch,owner); report['owned_scratch_removed'] = True
        except Exception as error: report.update(status='fail',cleanup_failure_type=type(error).__name__)
        report['elapsed_seconds'] = time.monotonic()-start
        if report['elapsed_seconds'] > TOTAL: report.update(status='fail',failure_type='HostInclusiveDeadline')
        build.write_json(out/'report.json',report); out.chmod(0o555)
        signal.alarm(0)
        for s,handler in old.items(): signal.signal(s,handler)
    return 0 if report['status'] in ('pass','inconclusive') else 1


def main():
    p = argparse.ArgumentParser(); p.add_argument('--native',action='store_true'); args = p.parse_args()
    code,revision = Path(os.environ['WR_CODE']),os.environ['WR_CODE_REVISION']
    require(os.environ['WR_ROOT'] == str(ROOT), 'Exact research root required'); build,certificate = helpers(code)
    return native(code,revision,ROOT/'results'/('cgal-sum-control-'+revision)/'disposable/work',Path('/opt/wr-cgal/include'),build) if args.native else host(code,revision,build,certificate)


if __name__ == '__main__':
    try: raise SystemExit(main())
    except Exception as error:
        print('cgal_sum_control FAIL: '+type(error).__name__,file=sys.stderr); raise SystemExit(1)
