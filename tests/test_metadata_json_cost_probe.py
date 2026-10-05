import hashlib
import io
import json
from pathlib import Path
import pytest
import metadata_json_cost_probe as cost


@pytest.mark.parametrize('kind,stride,total', [('dense_excluded',256,2048),('mixed',256,4352),('long_consulted',512,2048)])
def test_tiny_authored_exact_stream_count_hash_and_ids(kind,stride,total):
    # Deliberately <=10KiB procedural fixtures; no native64MiB execution locally.
    raw=b''.join(cost.fixture(kind,stride,total));want=cost.expected(kind,stride,total,lambda:None)
    assert len(raw)==total and hashlib.sha256(raw).hexdigest()==want['sha256']
    rows=json.loads(raw);assert len(rows)==(total-1)//stride
    assert [r['image_id'] for r in rows]==list(range(cost.ID_BASE,cost.ID_BASE+len(rows)))
    assert want['decoded_rows']==sum(cost.selected(kind,i) for i in range(len(rows)))
    assert all(r['objects'][0]['u']=='é🧠\n"' for r in rows)
    observed=cost.measure(kind,stride,total,cost.time.monotonic()+30)
    assert observed['expected']==want and observed['sha256']==want['sha256']
    assert observed['observed']['rows']==len(rows) and observed['structural_excluded_semantic_decode_calls']==0


def test_mixed_poison_never_whole_row_semantically_decoded(monkeypatch):
    original=cost.parser.strict_decode;calls=[]
    def check(raw):
        assert b'OLD_POISON' not in raw and b'1e999' not in raw
        calls.append(raw);return original(raw)
    monkeypatch.setattr(cost.parser,'strict_decode',check)
    observed=cost.measure('mixed',256,4352,cost.time.monotonic()+30)
    assert observed['expected']['decoded_rows']==1 and observed['expected']['excluded_rows']==15
    assert len([r for r in calls if r.lstrip().startswith(b'{')])==1


def test_id_field_first_last_and_expected_nonstrict_excluded_numbers():
    first=cost.record('dense_excluded',256,0);last=cost.record('dense_excluded',256,1)
    assert first.startswith(b'{"image_id":') and last.endswith(b'"image_id":1000000001}')
    assert len(first)==len(last)==255 and b'OLD_POISON' in first and b'1e999' in first


def test_reader_bounded_and_complete_with_tiny_chunks():
    reader=cost.Reader([b'a'*7,b'b'*13,b'c'*3]);chunks=[]
    while value:=reader.read(5):chunks.append(value)
    assert b''.join(chunks)==b'a'*7+b'b'*13+b'c'*3 and reader.size==23
    assert reader.digest.hexdigest()==hashlib.sha256(b''.join(chunks)).hexdigest()
    for value in (-1,0,65537,True,1.0):
        with pytest.raises(ValueError):reader.read(value)


def test_negative_controls_are_tiny_and_complete():
    assert cost.negative_controls()==dict(invalid_json_rejected=6,invalid_zip_rejected=2,controls_after_timed_cases=True)


def test_peak_rss_linux_units_and_predeclared_bound(monkeypatch):
    class Usage:ru_maxrss=1024
    monkeypatch.setattr(cost.resource,'getrusage',lambda _:Usage())
    monkeypatch.setattr(cost.sys,'platform','linux')
    assert cost.peak_memory()['peak_resident_bytes']==1048576
    Usage.ru_maxrss=(16 << 30)//1024+1
    with pytest.raises(ValueError):cost.peak_memory()


def test_projection_worst_case_fixed_budget_and_missing_cases():
    cases=[dict(case=name,seconds_per_byte=value) for (name,_),value in zip(cost.CASES,(1e-8,2e-8,3e-8))]
    result=cost.gate(cases)
    assert result['projected_seconds']==3e-8*1130711325+120 and result['capacity_gate_passed']
    cases[1]['seconds_per_byte']=1e-6
    assert not cost.gate(cases)['capacity_gate_passed']
    with pytest.raises(ValueError):cost.gate(cases[:2])
    with pytest.raises(ValueError):cost.gate(cases[::-1])


def test_timeout_during_generation_no_warmup_or_retry(monkeypatch):
    count=[]
    def fail(_):count.append(1);raise TimeoutError()
    monkeypatch.setattr(cost.lifecycle,'check',fail)
    monkeypatch.setattr(cost.parser,'iter_array',lambda *_args,**_kwargs:pytest.fail('Parser started after deadline'))
    with pytest.raises(TimeoutError):cost.measure('dense_excluded',256,2048,0)
    assert len(count)==1


def test_native_constants_source_pins_and_no_data_mounts():
    assert cost.BYTES==67108864 and cost.CASES==(('dense_excluded',256),('mixed',4096),('long_consulted',65536))
    root=Path(__file__).resolve().parents[1]
    for name,pin in cost.PINS.items():
        raw=(root/name).read_bytes();assert dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())==pin
    wrapper=(root/cost.HELPERS[1]).read_text()
    assert 'python3 -I -B' in wrapper and 'CUDA_VISIBLE_DEVICES=-1' in wrapper and '615s' in wrapper
    assert not any(word in wrapper for word in ('docker','--gpus','/weights/','/videos/'))


def test_reused_lifecycle_late_pass_demoted(tmp_path,monkeypatch):
    report=dict(status='pass',stage='complete',decision='PARSER_COST_CONTROL_PASS_PENDING_REAL_CENSUS')
    monkeypatch.setattr(cost.lifecycle,'check',lambda _:(_ for _ in ()).throw(TimeoutError()))
    with pytest.raises(TimeoutError):cost.lifecycle.publish_report(tmp_path,report,0,started=cost.time.monotonic())
    saved=json.loads((tmp_path/'report.json').read_bytes())
    assert saved['status']=='fail' and saved['publication_failed'] is True
    assert (tmp_path/'report.json').stat().st_mode&0o777==0o400


def test_fixture_input_rejects_non_authored_kind_or_too_small():
    with pytest.raises(ValueError):list(cost.fixture('unknown',256,2048))
    with pytest.raises(ValueError):list(cost.fixture('mixed',256,128))
    with pytest.raises(ValueError):cost.record('dense_excluded',32,0)
