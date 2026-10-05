"""One data-free native CPU cost control; never opens dataset rows or media."""
from collections import Counter
import hashlib
import io
import json
import os
from pathlib import Path
import resource
import signal
import stat
import sys
import time
import zipfile

sys.path.insert(0,str(Path(__file__).resolve().parent))
import metadata_json_stream as parser
import mediapipe_cpu_runtime_verify as rt
import visual_genome_metadata_acquire as lifecycle

ROOT = Path('/srv/scenesmith/world-reward')
OUT = Path('/srv/world-reward-data/metadata_json_cost_probe_v1')
ENTRY = 'run_metadata_json_cost_probe'
HELPERS = ('infra/metadata_json_cost_probe.py','infra/run_metadata_json_cost_probe.sh',
    'infra/metadata_json_stream.py','infra/mediapipe_cpu_runtime_verify.py','infra/visual_genome_metadata_acquire.py')
PINS = {'infra/metadata_json_stream.py':dict(bytes=8322,sha256='742b10d5f92e3446d17d4953f1d088281f051577653899ddcc7081f61545bf53'),
    'infra/mediapipe_cpu_runtime_verify.py':dict(bytes=23559,sha256='936ad97c5ffca3b7f86e3462a600a247b6fa540d9b669e748892ff45e12aedf2'),
    'infra/visual_genome_metadata_acquire.py':dict(bytes=15022,sha256='e6f36ee12240b379ddcb52ade7b9e9d413c9309995a73ca8f7df0e3c34695ada')}
BYTES = 64 << 20
CASES = (('dense_excluded',256),('mixed',4096),('long_consulted',65536))
ID_BASE = 1000000000
PROJECTED_BYTES = 1130711325


def selected(kind,index): return kind == 'long_consulted' or (kind == 'mixed' and index%16 == 15)


def record(kind,stride,index):
    keep = selected(kind,index); iid = str(ID_BASE+index).encode()
    number = b'1.25' if keep else b'1e999'
    marker = b'AUTHORED' if keep else b'OLD_POISON'
    semantic = b'"objects":[{"v":'+number+b',"s":[0,1,-2,3.5,true,false,null],"name":"'+marker+b'","u":"'+"é🧠".encode()+b'\\n\\\"","pad":"'
    prefix = b'{"image_id":'+iid+b','+semantic if index%2 == 0 else b'{'+semantic
    suffix = b'"}]}' if index%2 == 0 else b'"}],"image_id":'+iid+b'}'
    padding = stride-1-len(prefix)-len(suffix)
    rt.require(padding >= 0,'Authored fixed row too small')
    return prefix+b'x'*padding+suffix


def fixture(kind,stride,total,checkpoint=lambda:None):
    """Stride includes its comma. Final padding is JSON whitespace, not a row."""
    rt.require(kind in {k for k,_ in CASES} and type(total) is int and total > stride,'Bounded authored fixture required')
    count = (total-1)//stride
    yield b'['
    for index in range(count):
        if index%256 == 0: checkpoint()
        if index: yield b','
        yield record(kind,stride,index)
    yield b']'+b' '*(total-(count*stride+1))


class Reader:
    def __init__(self,chunks):
        self.chunks = iter(chunks); self.pending = bytearray(); self.done = False
        self.digest,self.size = hashlib.sha256(),0
    def read(self,count):
        rt.require(type(count) is int and 0 < count <= 65536,'Bounded authored read required')
        while len(self.pending) < count and not self.done:
            try:self.pending.extend(next(self.chunks))
            except StopIteration:self.done = True
        value = bytes(self.pending[:count]); del self.pending[:count]
        self.digest.update(value); self.size += len(value); return value


def expected(kind,stride,total,checkpoint):
    digest = hashlib.sha256(); size = 0
    for chunk in fixture(kind,stride,total,checkpoint):digest.update(chunk);size += len(chunk)
    rt.require(size == total,'Authored fixture exact byte count differs')
    n = (total-1)//stride; decoded = n if kind == 'long_consulted' else n//16 if kind == 'mixed' else 0
    return dict(bytes=total,sha256=digest.hexdigest(),rows=n,decoded_rows=decoded,excluded_rows=n-decoded,
        first_id=ID_BASE,last_id=ID_BASE+n-1)


def measure(kind,stride,total,deadline):
    checkpoint = lambda:lifecycle.check(deadline)
    want = expected(kind,stride,total,checkpoint)  # Generator/hash only; no parser warm-up.
    started = time.monotonic(); reader = Reader(fixture(kind,stride,total,checkpoint)); counts = Counter()
    for iid,raw in parser.iter_array(reader,id_key='image_id',check=checkpoint,max_bytes=total):
        index = counts['rows']; rt.require(type(iid) is int and iid == ID_BASE+index,'Exact authored row/ID order differs')
        if selected(kind,index):
            value = parser.strict_decode(raw); node = value['objects'][0]
            rt.require(value['image_id'] == iid and node['v'] == 1.25 and node['name'] == 'AUTHORED'
                and node['u'] == 'é🧠\n"','Authored consulted UTF8/control values differ')
            counts['decoded_rows'] += 1
        else:
            rt.require(b'OLD_POISON' in raw and b'1e999' in raw,'Excluded poison absent; never decode this row')
            counts['excluded_rows'] += 1
        counts['rows'] += 1
    elapsed = time.monotonic()-started; checkpoint()
    rt.require(reader.read(1) == b'' and reader.size == want['bytes'] and reader.digest.hexdigest() == want['sha256']
        and all(counts[k] == want[k] for k in ('rows','decoded_rows','excluded_rows')),'Complete authored fixture/hash/count differs')
    return dict(case=kind,stride_bytes=stride,expected=want,observed=dict(counts),sha256=reader.digest.hexdigest(),
        elapsed_seconds=elapsed,seconds_per_byte=elapsed/total,structural_excluded_semantic_decode_calls=0)


def negative_controls():
    bad = [b'[{"image_id":1,"x":1,"x":2}]',b'[{"image_id":1,"x":NaN}]',b'[{"image_id":1,"x":1e999}]',
        b'[{"image_id":1,"x":'+b'['*130+b'0'+b']'*130+b'}]',b'[{"image_id":1}]trailing',b'[{"image_id":1,"x":"\xff"}]']
    for data in bad:
        try:
            for _,raw in parser.iter_array(io.BytesIO(data),id_key='image_id'):parser.strict_decode(raw)
        except (ValueError,UnicodeError):continue
        raise ValueError('Authored invalid JSON accepted')
    original = io.BytesIO()
    with zipfile.ZipFile(original,'w',compression=zipfile.ZIP_DEFLATED) as archive:archive.writestr('authored.json',b'[{"image_id":1}]')
    raw = original.getvalue()
    corrupt = bytearray(raw); central = raw.index(b'PK\x01\x02'); corrupt[central+16] ^= 1
    for data in (bytes(corrupt),raw[:-20]):
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                with archive.open('authored.json') as stream:list(parser.iter_array(stream,id_key='image_id'))
        except (ValueError,zipfile.BadZipFile):continue
        raise ValueError('Authored corrupt ZIP accepted')
    return dict(invalid_json_rejected=len(bad),invalid_zip_rejected=2,controls_after_timed_cases=True)


def peak_memory():
    observed = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    size = int(observed*(1024 if sys.platform == 'linux' else 1))
    rt.require(0 < size <= 16 << 30,'Observed peak resident memory exceeds16GiB')
    return dict(peak_resident_bytes=size,peak_resident_limit_bytes=16 << 30,peak_resident_within_limit=True)


def gate(cases):
    rt.require(len(cases) == 3 and [c['case'] for c in cases] == [c for c,_ in CASES],'All three authored cases required')
    projected = max(c['seconds_per_byte'] for c in cases)*PROJECTED_BYTES+120
    return dict(projected_seconds=projected,projection_bytes=PROJECTED_BYTES,reserve_seconds=120,
                budget_seconds=600,capacity_gate_passed=projected <= 600,projection_is_runtime_guarantee=False)


def source(code,revision):
    binding = rt.source(ROOT,code,revision,ENTRY,HELPERS)
    modes = {str(p.relative_to(code)):[stat.S_IMODE(p.lstat().st_mode),p.lstat().st_uid,p.lstat().st_gid]
        for p in (code,*sorted(code.rglob('*')))}
    return dict(binding=binding,modes_identity=lifecycle.pin(json.dumps(modes,sort_keys=True).encode()))


def run():
    started = time.monotonic(); deadline = started+600
    rt.require(sys.platform == 'linux' and os.geteuid() == 0 and os.uname().nodename == 'world-reward-ncc-h100-02','Exact Azure CPU host required')
    code,revision = Path(os.environ['WR_CODE']),os.environ['WR_CODE_REVISION']
    rt.require(Path(__file__).resolve() == code/HELPERS[0],'Immutable actual entry required')
    before = source(code,revision)
    for module,name in ((parser,HELPERS[2]),(rt,HELPERS[3]),(lifecycle,HELPERS[4])):
        rt.require(Path(module.__file__).resolve() == code/name and before['binding']['helpers'][name] == PINS[name],'Original helper origin/pin differs')
    lifecycle.check(deadline); rt.require(not rt.canonical(OUT).exists() and OUT.parent.is_dir(),'Fresh CPU control namespace only; no retry')
    OUT.mkdir(mode=0o700); lifecycle.sync(OUT.parent)
    report = dict(schema='world_reward.metadata_json_cost_probe.v1',stage='authored_streams',status='fail',producer_revision=revision,
        source_binding=before,cases=[],network_used=False,actual_dataset_values_read=False,RGB_read=False,GPU_used=False,
        models_loaded=False,expanded_files_written=False,parser_warmup=False,retry_count=0,selection_performed=False,
        source_rehashed_after=False,outputs_sealed=False,quality_verified=False,adopted=False)
    def interrupted(*_):raise TimeoutError('Fixed CPU control interrupted')
    handlers = {s:signal.signal(s,interrupted) for s in (signal.SIGTERM,signal.SIGINT,signal.SIGALRM)}
    signal.setitimer(signal.ITIMER_REAL,max(.001,deadline-time.monotonic()))
    try:
        for kind,stride in CASES:report['cases'].append(measure(kind,stride,BYTES,deadline))
        report.update(negative_controls=negative_controls(),**gate(report['cases']),**peak_memory(),stage='complete')
        report.update(status='pass' if report['capacity_gate_passed'] else 'fail',
            decision='PARSER_COST_CONTROL_PASS_PENDING_REAL_CENSUS' if report['capacity_gate_passed'] else 'CLOSED_PARSER_COST_INSUFFICIENT')
    except BaseException as exc:
        report.update(status='fail',decision='CLOSED_PARSER_COST_NO_DATASET',error_type=type(exc).__name__ if type(exc).__name__ in
            ('ValueError','TimeoutError','OSError','BadZipFile') else 'other')
    finally:
        try:rt.require(source(code,revision) == before,'Original code/modes changed after control');report['source_rehashed_after'] = True;lifecycle.check(deadline)
        except BaseException:report.update(status='fail',decision='CLOSED_PARSER_COST_POSTCHECK',postcheck_failed=True)
        try:lifecycle.publish_report(OUT,report,deadline,started=started)
        except BaseException:report['status'] = 'fail'
        finally:
            signal.setitimer(signal.ITIMER_REAL,0)
            for s,h in handlers.items():signal.signal(s,h)
        print(json.dumps(dict(status=report['status'],stage=report['stage'],decision=report['decision'],
            projected_seconds=report.get('projected_seconds'),report_identity=rt.identity(OUT/'report.json',64 << 10)),sort_keys=True))
    return report


if __name__ == '__main__':
    rt.require(len(sys.argv) == 1,'One immutable data-free control only')
    sys.exit(0 if run()['status'] == 'pass' else 1)
