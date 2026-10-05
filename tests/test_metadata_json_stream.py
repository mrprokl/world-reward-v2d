import io
import json
import pytest
import metadata_json_stream as stream


def rows(raw, **kwargs):
    return list(stream.iter_array(io.BytesIO(raw), **kwargs))


@pytest.mark.parametrize('chunk', [1, 2, 7, 65536])
def test_top_level_id_after_semantics_and_utf8(chunk, monkeypatch):
    raw = '[{"objects":[{"name":"é \\\" \\u263a","value":1e999}],"image_id":3},{"image_id":4,"objects":[]}]'.encode()
    consulted = []; original = stream.json.loads
    def decode(value, **kwargs):
        consulted.append(value)
        return original(value, **kwargs)
    monkeypatch.setattr(stream.json, 'loads', decode)
    found = rows(raw, id_key='image_id', chunk_bytes=chunk)
    assert [r[0] for r in found] == [3, 4]
    assert not any('1e999' in str(v) for v in consulted)
    with pytest.raises(ValueError, match='Nonfinite'):
        stream.strict_decode(found[0][1])


def test_coco_annotations_skipped_without_value_decode(monkeypatch):
    raw = b'{"annotations":[{"bbox":[1e999],"marker":"OLD_POISON"}],"images":[{"id":7}],"licenses":[]}'
    decoded = []; original = stream.json.loads
    def record(value, **kwargs):
        decoded.append(value); return original(value, **kwargs)
    monkeypatch.setattr(stream.json, 'loads', record)
    result = list(stream.iter_object_arrays(io.BytesIO(raw), ('images', 'licenses'), chunk_bytes=3))
    assert result == [('images', b'{"id":7}')]
    assert not any('OLD_POISON' in str(v) or '1e999' in str(v) for v in decoded)


@pytest.mark.parametrize('raw', [b'[{}]tail', b'[{"x":1,}]', b'[{},]', b'[[]]', b'[{"x":01}]',
    b'[{"x":+1}]', b'[{"x":1.}]', b'[{"x":1e}]', b'[{"x":NaN}]', b'[{"x":Infinity}]',
    b'[{"x":trueX}]', b'[{"x":null false}]', b'[{"x":/*a*/1}]', b'[{"x":"a\x01"}]',
    b'[{"x":"\\q"}]', b'[{"x":"\\uZZZZ"}]', b'[{"x":"\xff"}]', b'[{"x":"unfinished}]',
    b'[{"x":[1,2}}]', b'[{"x":1,"x":2}]', b'[{"x":{"a":1,"a":2}}]'])
def test_malformed_skipped_values_rejected(raw):
    with pytest.raises((ValueError, UnicodeError)):
        rows(raw, id_key='image_id', chunk_bytes=2)


def test_consulted_surrogate_nonfinite_duplicates():
    for raw in (b'{"x":"\\ud800"}', b'{"x":1e999}', b'{"x":1,"x":2}'):
        with pytest.raises((ValueError, UnicodeError)): stream.strict_decode(raw)


def test_row_stream_depth_and_token_limits():
    with pytest.raises(ValueError, match='row/token'): rows(b'[{"x":"abcdef"}]', row_bytes=10)
    with pytest.raises(ValueError, match='stream byte'): rows(b'[{},{}]', max_bytes=5)
    with pytest.raises(ValueError, match='nesting'): rows(b'[{"x":'+b'['*130+b'0'+b']'*130+b'}]')
    with pytest.raises(ValueError, match='token'): rows(b'[{"x":'+b'1'*257+b'}]')


def test_root_arrays_keys_and_empty():
    assert rows(b' [] ') == []
    assert list(stream.iter_object_arrays(io.BytesIO(b'{"images":[],"licenses":[]}'), ('images','licenses'))) == []
    for raw in (b'{"images":{},"licenses":[]}', b'{"images":[],"images":[],"licenses":[]}',
                b'{"images":[]} ', b'{"images":[],"licenses":[],}', b'{}', b'{}junk'):
        with pytest.raises(ValueError): list(stream.iter_object_arrays(io.BytesIO(raw), ('images','licenses')))


def test_streaming_bounded_reads_and_callback():
    class Source(io.BytesIO):
        def read(self, count=-1):
            assert 0 < count <= 8
            return super().read(count)
    checks = []
    result = list(stream.iter_array(Source(b'[{"image_id":1},{"image_id":2}]'), id_key='image_id',
                                    chunk_bytes=8, check=lambda: checks.append(True)))
    assert [r[0] for r in result] == [1,2] and len(checks) >= 4


def test_direct_id_not_nested_and_decoding_only_requested_row():
    data = b'[{"objects":[{"image_id":9}],"image_id":2},{"image_id":3}]'
    assert [iid for iid, _ in rows(data, id_key='image_id')] == [2,3]
    assert rows(b'[{"objects":[{"image_id":9}]}]', id_key='image_id')[0][0] is None


@pytest.mark.parametrize('key',['chunk_bytes','row_bytes','max_bytes'])
@pytest.mark.parametrize('value',[-1,0,True,1.5,None])
def test_invalid_limits_never_read_source(key,value):
    class Source:
        def read(self,_):pytest.fail('Invalid limits read the source')
    with pytest.raises(ValueError):list(stream.iter_array(Source(),**{key:value}))
    with pytest.raises(ValueError):list(stream.iter_object_arrays(Source(),(),**{key:value}))


def test_chunk_request_capped_by_total_limit():
    class Source(io.BytesIO):
        def read(self,count):assert count<=2;return super().read(count)
    assert list(stream.iter_array(Source(b'[]'),max_bytes=2,chunk_bytes=65536))==[]


@pytest.mark.parametrize('value', [{"x":[True,None,0,-1,1.5,2e3]}, {"image_id":1,"nested":{"u":"🧠"}}, {}])
def test_roundtrip_valid_values(value):
    raw = json.dumps([value], ensure_ascii=False).encode()
    assert stream.strict_decode(rows(raw)[0]) == value
