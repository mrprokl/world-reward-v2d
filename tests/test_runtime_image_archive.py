"""Manufactured tiny TAR graphs only; no Docker, network or layer extraction."""
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile

import pytest


@pytest.fixture
def archive():
    path = Path(__file__).parents[1]/'infra/runtime_image_archive.py'
    spec = importlib.util.spec_from_file_location('runtime_archive_test', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def encoded(value): return json.dumps(value, separators=(',', ':')).encode()
def digest(raw): return 'sha256:'+hashlib.sha256(raw).hexdigest()
def blob_name(raw): return 'blobs/sha256/'+digest(raw)[7:]


def fixture(archive, count=47, hybrid=False, compressed=False, depth=0, repeat=False):
    layers = [f'manufactured uncompressed layer {i}'.encode() for i in range(count)]
    if repeat: layers = [layers[0]]*count
    diffs = [digest(raw) for raw in layers]
    config = encoded(dict(architecture='amd64', os='linux', rootfs=dict(type='layers', diff_ids=diffs)))
    rows = {}
    if hybrid:
        name = blob_name(config); descriptors = []
        for raw in layers:
            stored = gzip.compress(raw, mtime=0) if compressed else raw
            d = dict(mediaType=archive._GZIP[0] if compressed else archive._RAW[0], digest=digest(stored), size=len(stored))
            rows[blob_name(stored)] = stored; descriptors.append(d)
        names = ['blobs/sha256/'+d['digest'][7:] for d in descriptors]
        platform = encoded(dict(schemaVersion=2, mediaType=archive._MANIFEST[0],
            config=dict(mediaType=archive._CONFIG[0], digest=digest(config), size=len(config)), layers=descriptors))
        rows[blob_name(platform)] = platform
        d = dict(mediaType=archive._MANIFEST[0], digest=digest(platform), size=len(platform), platform=dict(architecture='amd64', os='linux'))
        for _ in range(depth):
            inner = encoded(dict(schemaVersion=2, mediaType=archive._INDEX[0], manifests=[d]))
            rows[blob_name(inner)] = inner
            d = dict(mediaType=archive._INDEX[0], digest=digest(inner), size=len(inner))
        rows['index.json'] = encoded(dict(schemaVersion=2, mediaType=archive._INDEX[0], manifests=[d]))
        rows['oci-layout'] = encoded(dict(imageLayoutVersion='1.0.0'))
    else:
        name = digest(config)[7:]+'.json'
        names = [f'{i:064x}/layer.tar' for i in range(count)]
        if repeat: names = [names[0]]*count
        rows.update(zip(names, layers))
    rows[name] = config
    manifest = dict(Config=name, RepoTags=None, Layers=names)
    if hybrid: manifest['LayerSources'] = dict(zip(diffs, descriptors))
    rows['manifest.json'] = encoded([manifest])
    return rows, (digest(config), hashlib.sha256(encoded(diffs)).hexdigest(), count)


def save(tmp_path, rows, extra=(), format=tarfile.USTAR_FORMAT):
    path = tmp_path/'image.tar'
    with tarfile.open(path, 'w', format=format) as tar:
        for name, raw in [*rows.items(), *extra]:
            if isinstance(raw, tarfile.TarInfo): tar.addfile(raw); continue
            m = tarfile.TarInfo(name); m.size = len(raw); m.mode = 0o644
            tar.addfile(m, io.BytesIO(raw))
    return path.resolve()


def manifest(rows, change):
    value = json.loads(rows['manifest.json']); change(value[0]); rows['manifest.json'] = encoded(value)


def platform_change(rows, change):
    index = json.loads(rows['index.json']); d = index['manifests'][0]
    name = 'blobs/sha256/'+d['digest'][7:]; value = json.loads(rows.pop(name)); change(value)
    raw = encoded(value); rows[blob_name(raw)] = raw
    d.update(digest=digest(raw), size=len(raw)); rows['index.json'] = encoded(index)


@pytest.mark.parametrize('tags', [None, []])
def test_default_complete_classic_identity_without_permission_mutation(archive, tmp_path, tags):
    rows, pins = fixture(archive); manifest(rows, lambda r: r.update(RepoTags=tags))
    rows['repositories'] = b'{}'
    for i, name in enumerate(json.loads(rows['manifest.json'])[0]['Layers']):
        parent = str(Path(name).parent); rows[parent+'/VERSION'] = b'1.0\n'
        rows[parent+'/json'] = encoded(dict(id=Path(parent).name, parent=f'{i-1:064x}' if i else ''))
    path = save(tmp_path, rows); before = path.read_bytes(); mode = path.stat().st_mode
    result = archive.authenticate(path, *pins[:2])
    assert result['representation'] == 'classic' and result['layer_count'] == 47
    assert result['rootfs_diff_ids'] == json.loads(rows[json.loads(rows['manifest.json'])[0]['Config']])['rootfs']['diff_ids']
    assert result['config_id'] == pins[0] and result['ordered_rootfs_sha256'] == pins[1]
    assert result['index_ids'] == [] and result['platform_manifest_id'] is None
    assert result['unique_layer_blobs'] == 47
    assert all(result[p] is False for p in ('archive_whole_sha_verified', 'layers_extracted', 'docker_loaded'))
    assert path.read_bytes() == before and path.stat().st_mode == mode


@pytest.mark.parametrize('depth', [0, 1, 2])
@pytest.mark.parametrize('compressed', [False, True])
def test_hybrid_full_links_config_identity_and_uncompressed_diff_ids(archive, tmp_path, depth, compressed):
    rows, pins = fixture(archive, hybrid=True, compressed=compressed, depth=depth)
    path = save(tmp_path, rows); result = archive.authenticate(path, *pins)
    assert result['config_id'] == pins[0] and result['config_id'] != result['platform_manifest_id']
    assert len(result['index_ids']) == depth+1 and result['index_ids'][0] == digest(rows['index.json'])
    assert result['representation'] == 'hybrid_oci' and result['compressed_layers'] == (47 if compressed else 0)
    assert result['uncompressed_layer_bytes'] == sum(len(f'manufactured uncompressed layer {i}'.encode()) for i in range(47))


def test_partial_layer_sources_are_only_authenticated_links(archive, tmp_path):
    rows, pins = fixture(archive, count=2, hybrid=True)
    manifest(rows, lambda r: r.update(LayerSources=dict(list(r['LayerSources'].items())[:1])))
    assert archive.authenticate(save(tmp_path, rows), *pins)['layer_count'] == 2


@pytest.mark.parametrize('fault', ['config', 'reversed', 'unsupported_codec', 'different_config_type', 'bool_schema', 'unused_blob'])
def test_authenticated_but_inconsistent_platform_document_refused(archive, tmp_path, fault):
    rows, pins = fixture(archive, count=2, hybrid=True)
    def change(node):
        if fault == 'config': node['config']['digest'] = 'sha256:'+'a'*64
        elif fault == 'reversed': node['layers'].reverse()
        elif fault == 'unsupported_codec': node['layers'][0]['mediaType'] = 'application/vnd.oci.image.layer.v1.tar+zstd'
        elif fault == 'different_config_type': node['config']['mediaType'] = archive._RAW[0]
        elif fault == 'bool_schema': node['schemaVersion'] = True
    platform_change(rows, change)
    if fault == 'unused_blob': rows['blobs/sha256/'+'a'*64] = b'unreferenced image payload'
    with pytest.raises(ValueError): archive.authenticate(save(tmp_path, rows), *pins)


@pytest.mark.parametrize('fault', ['tags', 'parent', 'parent_empty', 'unknown', 'sources_null', 'sources_foreign', 'sources_forged', 'sources_urls'])
def test_foreign_legacy_selection_or_unbound_layer_sources_refused(archive, tmp_path, fault):
    rows, pins = fixture(archive, count=2, hybrid=True)
    def change(r):
        if fault == 'tags': r['RepoTags'] = ['foreign:latest']
        elif fault.startswith('parent'): r['Parent'] = '' if fault == 'parent_empty' else 'sha256:'+'a'*64
        elif fault == 'unknown': r['Foreign'] = 'anything'
        elif fault == 'sources_null': r['LayerSources'] = None
        elif fault == 'sources_foreign': r['LayerSources']['sha256:'+'a'*64] = next(iter(r['LayerSources'].values()))
        elif fault == 'sources_forged': next(iter(r['LayerSources'].values()))['size'] += 1
        elif fault == 'sources_urls': next(iter(r['LayerSources'].values()))['urls'] = ['https://foreign.invalid']
    manifest(rows, change); path = save(tmp_path, rows); before = path.read_bytes()
    with pytest.raises(ValueError): archive.authenticate(path, *pins)
    assert path.read_bytes() == before


@pytest.mark.parametrize('fault', ['size', 'bool_size', 'digest', 'multi', 'platform', 'annotations', 'urls', 'media', 'index_depth', 'half_layout', 'half_index'])
def test_oci_descriptor_chain_and_single_platform_are_exact(archive, tmp_path, fault):
    rows, pins = fixture(archive, count=2, hybrid=True, depth=3 if fault == 'index_depth' else 0)
    index = json.loads(rows['index.json']); d = index['manifests'][0]
    if fault == 'size': d['size'] += 1
    elif fault == 'bool_size': d['size'] = True
    elif fault == 'digest': d['digest'] = 'sha256:'+'a'*64
    elif fault == 'multi': index['manifests'].append(d.copy())
    elif fault == 'platform': d['platform']['architecture'] = 'arm64'
    elif fault == 'annotations': d['annotations'] = {'io.containerd.image.name': 'foreign'}
    elif fault == 'urls': d['urls'] = ['https://foreign.invalid']
    elif fault == 'media': d['mediaType'] = 'application/vnd.in-toto+json'
    rows['index.json'] = encoded(index)
    if fault == 'half_layout': del rows['oci-layout']
    if fault == 'half_index': del rows['index.json']
    with pytest.raises(ValueError): archive.authenticate(save(tmp_path, rows), *pins)


@pytest.mark.parametrize('fault', ['layer', 'config', 'rootfs', 'count', 'bool_count', 'malformed_config', 'reordered'])
def test_all_raw_config_and_ordered_layers_bound(archive, tmp_path, fault):
    rows, pins = fixture(archive, count=2)
    config_name = json.loads(rows['manifest.json'])[0]['Config']
    if fault == 'layer': rows[next(k for k in rows if k.endswith('layer.tar'))] += b'changed'
    elif fault == 'config': pins = ('sha256:'+'a'*64, *pins[1:])
    elif fault == 'rootfs': pins = (pins[0], 'a'*64, pins[2])
    elif fault == 'count': pins = (*pins[:2], 3)
    elif fault == 'bool_count': pins = (*pins[:2], True)
    elif fault == 'malformed_config':
        rows[config_name] = b'[]'; pins = (digest(b'[]'), *pins[1:])
    elif fault == 'reordered': manifest(rows, lambda r: r['Layers'].reverse())
    with pytest.raises(ValueError): archive.authenticate(save(tmp_path, rows), *pins)


@pytest.mark.parametrize('raw', [b'[{"Config":"x","Config":"y"}]', b'[{"n":NaN}]', b'[{"n":1e999}]'])
def test_strict_json_duplicate_and_all_nonfinite_forms(archive, tmp_path, raw):
    rows, pins = fixture(archive, count=1); rows['manifest.json'] = raw
    with pytest.raises(ValueError): archive.authenticate(save(tmp_path, rows), *pins)


@pytest.mark.parametrize('name', ['/absolute', '../parent', './dot', 'a//b', 'back\\slash', 'control\x7f', 'unknown', 'manifest.json'])
def test_unsafe_names_unreferenced_files_and_duplicates(archive, tmp_path, name):
    rows, pins = fixture(archive, count=1)
    with pytest.raises(ValueError): archive.authenticate(save(tmp_path, rows, [(name, b'foreign')]), *pins)


@pytest.mark.parametrize('kind', [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.CHRTYPE, tarfile.FIFOTYPE])
def test_alias_and_special_tar_members_refused(archive, tmp_path, kind):
    rows, pins = fixture(archive, count=1)
    m = tarfile.TarInfo('alias'); m.type = kind; m.linkname = '/foreign'
    with pytest.raises(ValueError): archive.authenticate(save(tmp_path, rows, [('alias', m)]), *pins)


@pytest.mark.parametrize('format', [tarfile.PAX_FORMAT, tarfile.GNU_FORMAT])
def test_hidden_tar_extended_name_headers_not_accepted(archive, tmp_path, format):
    rows, pins = fixture(archive, count=1)
    old = json.loads(rows['manifest.json'])[0]['Config']; name = 'long/'+'x'*120+'.json'
    rows[name] = rows.pop(old); manifest(rows, lambda r: r.update(Config=name))
    with pytest.raises(ValueError): archive.authenticate(save(tmp_path, rows, format=format), *pins)


def test_unreferenced_directories_and_fake_huge_pax_stop_before_payload_read(archive, tmp_path):
    rows, pins = fixture(archive, count=1)
    directory = tarfile.TarInfo('unreferenced'); directory.type = tarfile.DIRTYPE
    with pytest.raises(ValueError): archive.authenticate(save(tmp_path, rows, [('unreferenced', directory)]), *pins)
    path = tmp_path/'pax.tar'; pax = tarfile.TarInfo('extension'); pax.type = tarfile.XHDTYPE; pax.size = 2<<30
    path.write_bytes(pax.tobuf(format=tarfile.USTAR_FORMAT)+b'\0'*512)
    with pytest.raises(ValueError, match='nonextended'): archive.authenticate(path.resolve(), *pins)


def test_file_directory_alias_and_nonzero_trailing_payload_refused(archive, tmp_path):
    rows, pins = fixture(archive, count=1)
    rows['unrelated'] = b'file'; rows['unrelated/child'] = b'child'
    with pytest.raises(ValueError): archive.authenticate(save(tmp_path, rows), *pins)
    del rows['unrelated']; del rows['unrelated/child']; path = save(tmp_path, rows)
    with path.open('ab') as stream: stream.write(b'concealed trailer')
    with pytest.raises(ValueError, match='trailing'): archive.authenticate(path, *pins)


def test_repeated_blob_still_counts_full_ordered_expansion(archive, tmp_path, monkeypatch):
    rows, pins = fixture(archive, count=3, hybrid=True, repeat=True)
    result = archive.authenticate(save(tmp_path, rows), *pins)
    assert result['unique_layer_blobs'] == 1 and result['layer_count'] == 3
    assert result['uncompressed_layer_bytes'] == 3*len(b'manufactured uncompressed layer 0')
    monkeypatch.setattr(archive, 'MAX_EXPANDED', result['uncompressed_layer_bytes']-1)
    with pytest.raises(ValueError, match='uncompressed'): archive.authenticate(tmp_path/'image.tar', *pins)


@pytest.mark.parametrize('cap', ['MAX_MEMBERS', 'MAX_JSON', 'MAX_LAYER', 'MAX_ARCHIVE', 'MAX_EXPANDED'])
def test_bounded_resources_fail_closed(archive, tmp_path, monkeypatch, cap):
    rows, pins = fixture(archive, count=2, hybrid=True, compressed=True); path = save(tmp_path, rows)
    monkeypatch.setattr(archive, cap, 1)
    with pytest.raises(ValueError): archive.authenticate(path, *pins)


def test_gzip_crc_and_expansion_bound_not_just_compressed_sha(archive, tmp_path, monkeypatch):
    rows, pins = fixture(archive, count=1, hybrid=True, compressed=True)
    monkeypatch.setattr(archive, 'MAX_EXPANDED', 2)
    with pytest.raises(ValueError, match='cap'): archive.authenticate(save(tmp_path, rows), *pins)


@pytest.mark.parametrize('fault', ['crc', 'truncated'])
def test_bad_gzip_with_authenticated_compressed_blob_still_rejected(archive, tmp_path, fault):
    rows, pins = fixture(archive, count=1, hybrid=True, compressed=True)
    r = json.loads(rows['manifest.json'])[0]; old_name = r['Layers'][0]; stored = rows.pop(old_name)
    if fault == 'crc': stored = stored[:-8]+bytes([stored[-8]^1])+stored[-7:]
    else: stored = stored[:-4]
    name = blob_name(stored); rows[name] = stored
    descriptor = dict(mediaType=archive._GZIP[0], digest=digest(stored), size=len(stored))
    platform_change(rows, lambda node: node['layers'].__setitem__(0, descriptor))
    manifest(rows, lambda r: r.update(Layers=[name], LayerSources={next(iter(r['LayerSources'])): descriptor}))
    with pytest.raises((OSError, EOFError)): archive.authenticate(save(tmp_path, rows), *pins)


def test_legacy_first_layer_parent_cannot_self_reference(archive, tmp_path):
    rows, pins = fixture(archive, count=1)
    parent = str(Path(json.loads(rows['manifest.json'])[0]['Layers'][0]).parent)
    rows[parent+'/json'] = encoded(dict(id=Path(parent).name, parent=Path(parent).name))
    with pytest.raises(ValueError): archive.authenticate(save(tmp_path, rows), *pins)


def test_canonical_single_link_archive_and_post_mutation(archive, tmp_path, monkeypatch):
    rows, pins = fixture(archive, count=1); path = save(tmp_path, rows)
    link = tmp_path/'alias'; link.hardlink_to(path)
    with pytest.raises(ValueError): archive.authenticate(path, *pins)
    link.unlink(); link.symlink_to(path)
    with pytest.raises(ValueError): archive.authenticate(link, *pins)
    link.unlink()
    original = archive._json; changed = False
    def mutate(raw):
        nonlocal changed
        if not changed:
            changed = True; path.chmod(0o600)
        return original(raw)
    monkeypatch.setattr(archive, '_json', mutate)
    with pytest.raises(ValueError, match='changed'): archive.authenticate(path, *pins)


def test_source_only_no_model_network_docker_unpack_or_mutation(archive):
    source = Path(archive.__file__).read_text()
    for forbidden in ('import torch', 'import numpy', 'subprocess', 'urllib', '.extractall(', '.extract(', '.unlink(', '.chmod('):
        assert forbidden not in source
    assert 'O_NOFOLLOW' in source and "mode='r:'" in source


def test_classic_containing_only_structural_parent_directories(archive, tmp_path):
    rows, pins = fixture(archive, count=1)
    parent = str(Path(json.loads(rows['manifest.json'])[0]['Layers'][0]).parent)
    d = tarfile.TarInfo(parent+'/'); d.type = tarfile.DIRTYPE
    assert archive.authenticate(save(tmp_path, rows, [(d.name, d)]), *pins)['layer_count'] == 1


def test_parser_rejects_hidden_extension_before_processing(archive):
    m = archive._ImageTarInfo('malicious'); m.type = tarfile.XHDTYPE; m.size = 2<<30
    with pytest.raises(ValueError, match='Unsupported'): m._proc_member(None)


def legacy_fixture(archive, count=47):
    """Separate manufactured census pins, not evidence about a real Docker save."""
    rows, _ = fixture(archive, count=count, hybrid=True)
    r = json.loads(rows['manifest.json'])[0]; config = json.loads(rows[r['Config']])
    config.update(created='2026-01-02T03:04:05Z', container_config=archive._EMPTY_CONTAINER,
        config={'Cmd': ['unchanged-pinned-command']}, author='manufactured source')
    new = encoded(config); del rows[r['Config']]; rows[blob_name(new)] = new
    platform_change(rows, lambda p: p.update(config=dict(mediaType=archive._CONFIG[0], digest=digest(new), size=len(new))))
    manifest(rows, lambda r: r.update(Config=blob_name(new)))
    pins = {}; previous = None
    for i in range(count):
        fields = dict(created='1970-01-01T00:00:00Z', container_config=archive._EMPTY_CONTAINER, os='linux')
        if i == count-1: fields = {k: config[k] for k in archive._V1_FIELDS if k in config}
        identifier = hashlib.sha256(f'manufactured-legacy-id-{i}'.encode()).hexdigest()
        node = dict(fields, id=identifier)
        if previous: node['parent'] = previous
        data = encoded(node); name = blob_name(data); rows[name] = data
        pins[name] = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest()); previous = identifier
    rootfs = config['rootfs']['diff_ids']
    return rows, (digest(new), hashlib.sha256(encoded(rootfs)).hexdigest(), count), pins


def legacy_change(rows, pins, index, change):
    name = list(pins)[index]; node = json.loads(rows.pop(name)); change(node); pins.pop(name)
    data = encoded(node); name = blob_name(data); rows[name] = data
    pins[name] = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def test_pinned_full_47_moby_compatibility_chain_metadata_is_not_derived_identity(archive, tmp_path):
    rows, pins, metadata = legacy_fixture(archive); path = save(tmp_path, rows); before = path.read_bytes()
    with pytest.raises(ValueError, match='Unreferenced'): archive.authenticate(path, *pins)
    result = archive.authenticate(path, *pins, expected_legacy_metadata=metadata)
    assert result['pinned_legacy_metadata_verified'] is True and result['legacy_ids_derived'] is False
    assert result['legacy_metadata_count'] == 47 and result['config_id'] == pins[0]
    assert result['legacy_metadata_manifest_sha256'] == hashlib.sha256(json.dumps(metadata, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    assert path.read_bytes() == before and result['docker_loaded'] is False
    assert len(list(metadata.values())[1]['sha256']) == 64


@pytest.mark.parametrize('fault', ['missing', 'count', 'wrong_bytes', 'wrong_sha', 'nonblob', 'overlap', 'extra_pin_field', 'bool_bytes'])
def test_independent_metadata_full_pin_contract_cannot_be_forged(archive, tmp_path, fault):
    rows, pins, metadata = legacy_fixture(archive, count=3); name = next(iter(metadata))
    if fault == 'missing': del rows[name]
    elif fault == 'count': metadata.pop(name)
    elif fault == 'wrong_bytes': metadata[name]['bytes'] += 1
    elif fault == 'wrong_sha': metadata[name]['sha256'] = 'a'*64
    elif fault == 'nonblob': metadata['../foreign'] = metadata.pop(name)
    elif fault == 'overlap':
        name = json.loads(rows['manifest.json'])[0]['Config']; metadata.pop(next(iter(metadata)))
        metadata[name] = dict(bytes=len(rows[name]), sha256=hashlib.sha256(rows[name]).hexdigest())
    elif fault == 'extra_pin_field': metadata[name]['arbitrary'] = True
    elif fault == 'bool_bytes': metadata[name]['bytes'] = True
    with pytest.raises(ValueError): archive.authenticate(save(tmp_path, rows), *pins, expected_legacy_metadata=metadata)


@pytest.mark.parametrize('fault', ['command', 'bool_number', 'extra_os', 'foreign_parent', 'cycle', 'branch', 'duplicate_id', 'terminal', 'root_parent', 'unbound_extra'])
def test_even_separately_pinned_legacy_json_has_strict_source_chain_schema(archive, tmp_path, fault):
    rows, pins, metadata = legacy_fixture(archive, count=3)
    records = [json.loads(rows[p]) for p in metadata]
    if fault == 'command': legacy_change(rows, metadata, 0, lambda r: r['container_config'].update(Cmd=['foreign-execution']))
    elif fault == 'bool_number': legacy_change(rows, metadata, 0, lambda r: r['container_config'].update(AttachStdin=0))
    elif fault == 'extra_os': legacy_change(rows, metadata, 0, lambda r: r.update(rootfs={}))
    elif fault == 'foreign_parent': legacy_change(rows, metadata, 1, lambda r: r.update(parent='a'*64))
    elif fault == 'cycle': legacy_change(rows, metadata, 0, lambda r: r.update(parent=records[-1]['id']))
    elif fault == 'branch': legacy_change(rows, metadata, 2, lambda r: r.update(parent=records[0]['id']))
    elif fault == 'duplicate_id': legacy_change(rows, metadata, 1, lambda r: r.update(id=records[0]['id']))
    elif fault == 'terminal': legacy_change(rows, metadata, 2, lambda r: r.update(architecture='arm64'))
    elif fault == 'root_parent': legacy_change(rows, metadata, 0, lambda r: r.update(parent=''))
    elif fault == 'unbound_extra': rows[blob_name(b'not bound')] = b'not bound'
    with pytest.raises(ValueError): archive.authenticate(save(tmp_path, rows), *pins, expected_legacy_metadata=metadata)


def test_compatibility_pin_bytes_bound_precedes_json_use(archive, tmp_path, monkeypatch):
    rows, pins, metadata = legacy_fixture(archive, count=1); path = save(tmp_path, rows)
    name = next(iter(metadata)); rows[name] = rows[name][:-1]+b'x'; path = save(tmp_path, rows)
    parsed = archive._json
    def no_corrupt_parse(raw):
        assert not raw.endswith(b'x')
        return parsed(raw)
    monkeypatch.setattr(archive, '_json', no_corrupt_parse)
    with pytest.raises(ValueError, match='bytes differ'): archive.authenticate(path, *pins, expected_legacy_metadata=metadata)
