"""Tiny streamed archives and HTTP mocks only; no Azure, Docker or model data."""
import base64
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import urllib.parse

import pytest


@pytest.fixture
def transfer():
    path = Path(__file__).parents[1]/'infra/articulated_runtime_transfer.py'
    spec = importlib.util.spec_from_file_location('transfer_test', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def pin(raw): return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


class Response(io.BytesIO):
    def __init__(self, raw=b'', status=201):
        super().__init__(raw); self.status = status; self.headers = {'Content-Length': str(len(raw))}


class FakeBlob:
    def __init__(self): self.blocks = {}; self.archive = None; self.calls = []
    def request(self, method, data=None, query='', headers=None):
        self.calls.append((method, query, headers))
        if method == 'GET': return Response(self.archive, 200)
        fields = urllib.parse.parse_qs(query)
        if fields['comp'] == ['block']: self.blocks[fields['blockid'][0]] = data
        else:
            assert headers['If-None-Match'] == '*'
            assert self.archive is None
            import re
            tokens = re.findall(r'<Latest>([^<]+)</Latest>', data.decode())
            self.archive = b''.join(self.blocks[p] for p in tokens)
        return Response()


def tiny_pack(transfer, tmp_path, monkeypatch):
    monkeypatch.setattr(transfer, 'BLOCK', 1024)
    root = tmp_path/'source'; root.mkdir()
    files = {'validation/robotap_boots_v1/assets/tapnet_source/README.md': pin(b'tiny source'),
             'results/cpu-proof/report.json': pin(b'{}')}
    for name, wanted in files.items():
        path = root/name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'tiny source' if wanted['bytes'] == 11 else b'{}')
    image = tmp_path/'image'; image.write_bytes(b'not a real image')
    blob = FakeBlob(); writer = transfer.BlockWriter(blob)
    manifest = transfer.pack(writer, root, files, image, transfer.identity(image), 'a'*40, {'files': 2, 'sha256': 'b'*64})
    expected = writer.finish()
    return blob, files, expected, manifest


def test_whole_original_protocol_selects_exact_nine_leaves_only(transfer):
    root = Path(__file__).parents[1]; files = transfer.asset_pins(root)
    assert len(files) == 9 and sum(p.endswith('.py') for p in files) == 5
    assert files[transfer.RUNTIME] == transfer.RUNTIME_PIN
    assert files[transfer.BASE+'/bootstapir_checkpoint_v2.pt']['bytes'] == 218886140
    assert not any('public_v2' in p or p.endswith(('.npz', '.mp4', '.pkl')) for p in files)


def test_streamed_exclusive_blocks_download_and_manifest_first_exact_inventory(transfer, tmp_path, monkeypatch):
    blob, files, expected, manifest = tiny_pack(transfer, tmp_path, monkeypatch)
    assert all(len(block) <= 1024 for block in blob.blocks.values())
    assert expected == pin(blob.archive)
    assert len({len(base64.b64decode(p)) for p in blob.blocks}) == 1
    archive = tmp_path/'received.tar'; transfer.download(blob, archive, expected)
    scratch = tmp_path/'items'; scratch.mkdir()
    loaded, extracted = transfer.unpack(archive, scratch, files, 'a'*40)
    assert loaded == manifest and set(extracted) == {*files, 'image.tar'}
    assert all(transfer.identity(extracted[p]) == files[p] for p in files)
    with tarfile.open(archive) as saved:
        assert saved.getnames()[0] == 'manifest.json'


@pytest.mark.parametrize('fault', ['duplicate', 'symlink', 'unexpected', 'wrong_size', 'wrong_hash', 'private', 'manifest_not_first'])
def test_safe_tar_rejects_aliases_unlisted_private_payload_and_mutations(transfer, tmp_path, monkeypatch, fault):
    blob, files, _, _ = tiny_pack(transfer, tmp_path, monkeypatch)
    archive = tmp_path/'tampered.tar'
    with tarfile.open(fileobj=io.BytesIO(blob.archive)) as original, tarfile.open(archive, 'w', format=tarfile.USTAR_FORMAT) as changed:
        members = original.getmembers()
        if fault == 'manifest_not_first': members[0], members[1] = members[1], members[0]
        for m in members:
            raw = original.extractfile(m).read()
            if m.name in files:
                if fault == 'symlink': m.type = tarfile.SYMTYPE; m.linkname = '/foreign'; m.size = 0
                elif fault == 'unexpected': m.name = '../foreign'
                elif fault == 'private': m.name = 'validation/robotap_boots_v1/private/labels.pkl'
                elif fault == 'wrong_size': raw += b'x'; m.size += 1
                elif fault == 'wrong_hash': raw = bytes([raw[0] ^ 1])+raw[1:]
            changed.addfile(m, io.BytesIO(raw))
            if fault == 'duplicate' and m.name == 'manifest.json': changed.addfile(m, io.BytesIO(raw))
    scratch = tmp_path/'items'; scratch.mkdir()
    with pytest.raises(ValueError): transfer.unpack(archive, scratch, files, 'a'*40)


def test_private_sas_never_exposed_in_http_failure_or_redirect(transfer):
    class Fails:
        def open(self, request, timeout): raise RuntimeError('LEAK '+request.full_url)
    url = 'https://ownedaccount.blob.core.windows.net/private/articulated-runtime-'+'a'*40+'.tar?sp=cw&sig=SECRET'
    blob = transfer.Blob(url, 'a'*40, opener=Fails())
    with pytest.raises(RuntimeError) as failure: blob.request('PUT', b'payload')
    assert 'SECRET' not in str(failure.value) and 'blob.core' not in str(failure.value)
    assert failure.value.__suppress_context__ is True
    with pytest.raises(RuntimeError): transfer.NoRedirect().redirect_request(None, None, None, None, None, None)
    for bad in (url.replace('https:', 'http:'), url.replace('/private/', '/other/path/'), url.replace('sig=SECRET', 'none=x')):
        with pytest.raises(ValueError): transfer.Blob(bad, 'a'*40)


def test_managed_identity_unsigned_url_is_fixed_account_only(transfer, monkeypatch):
    url = 'https://stworldrewardresearch26.blob.core.windows.net/runtime-transfers/articulated-runtime-'+'a'*40+'.tar'
    blob = transfer.Blob(url, 'a'*40, managed_identity=True)
    blob.token = 'IN_MEMORY_ONLY'; blob.token_expiry = 10**12
    assert blob.authorization() == {'Authorization': 'Bearer IN_MEMORY_ONLY'}
    with pytest.raises(ValueError): transfer.Blob(url, 'a'*40)
    with pytest.raises(ValueError): transfer.Blob(url.replace('stworldrewardresearch26', 'foreignaccount'), 'a'*40, managed_identity=True)


def test_managed_identity_refresh_failure_never_exposes_token(transfer, monkeypatch):
    class Fails:
        def open(self, *_a, **_k): raise RuntimeError('LEAK TOKEN')
    monkeypatch.setattr(transfer.urllib.request, 'build_opener', lambda *_: Fails())
    url = 'https://stworldrewardresearch26.blob.core.windows.net/runtime-transfers/articulated-runtime-'+'a'*40+'.tar'
    blob = transfer.Blob(url, 'a'*40, managed_identity=True)
    with pytest.raises(RuntimeError) as failure: blob.authorization()
    assert 'TOKEN' not in str(failure.value) and failure.value.__suppress_context__


def test_install_no_overwrite_readonly_leaf_and_owned_failure_rollback(transfer, tmp_path, monkeypatch):
    monkeypatch.setattr(transfer.os, 'chown', lambda *args: None)
    monkeypatch.setattr(transfer.os, 'fchown', lambda *args: None)
    root = tmp_path/'target'; root.mkdir()
    source = tmp_path/'source'; source.write_bytes(b'original')
    files = {'new/assets/source.py': pin(b'original')}
    assert transfer.install({'new/assets/source.py': source}, root, files) == files
    target = root/'new/assets/source.py'; assert target.stat().st_mode & 0o777 == 0o444
    with pytest.raises(ValueError): transfer.install({'new/assets/source.py': source}, root, files)
    assert target.read_bytes() == b'original'
    broken = {'new/assets/fail.py': pin(b'wrong')}
    with pytest.raises(ValueError): transfer.install({'new/assets/fail.py': source}, root, broken)
    assert not (root/'new/assets/fail.py').exists() and target.read_bytes() == b'original'


def test_truncated_or_extra_download_rejected_and_no_foreign_replacement(transfer, tmp_path):
    blob = FakeBlob(); blob.archive = b'abc'; path = tmp_path/'bytes'
    with pytest.raises(ValueError): transfer.download(blob, path, pin(b'abcd'))
    assert path.read_bytes() == b''
    with pytest.raises(FileExistsError): transfer.download(blob, path, pin(b'abc'))


def test_identity_rejects_hardlink_or_symlink(transfer, tmp_path):
    path = tmp_path/'original'; path.write_bytes(b'one')
    other = tmp_path/'alias'; other.hardlink_to(path)
    with pytest.raises(ValueError): transfer.identity(path)
    other.unlink(); other.symlink_to(path)
    with pytest.raises(ValueError): transfer.identity(other)


def test_inner_image_config_and_all_47_layer_bytes_verified_before_load(transfer, tmp_path, monkeypatch):
    layers = [f'uncompressed-layer-{i}'.encode() for i in range(47)]
    digests = ['sha256:'+hashlib.sha256(raw).hexdigest() for raw in layers]
    config = json.dumps({'architecture': 'amd64', 'os': 'linux', 'rootfs': {'type': 'layers', 'diff_ids': digests}}, separators=(',', ':')).encode()
    image = 'sha256:'+hashlib.sha256(config).hexdigest()
    monkeypatch.setattr(transfer, 'IMAGE', image)
    monkeypatch.setattr(transfer, 'LAYERS_SHA', hashlib.sha256(json.dumps(digests, separators=(',', ':')).encode()).hexdigest())
    config_name = image[7:]+'.json'; names = [f'{i}/layer.tar' for i in range(47)]
    for fault in (None, 'tags', 'layer_bytes', 'link', 'extra'):
        path = tmp_path/(str(fault)+'.tar')
        manifest = json.dumps([{'Config': config_name, 'RepoTags': None if fault != 'tags' else ['foreign:tag'], 'Layers': names}]).encode()
        contents = [('manifest.json', manifest), (config_name, config), *zip(names, layers)]
        if fault == 'extra': contents.append(('foreign/private.dat', b'private'))
        with tarfile.open(path, 'w', format=tarfile.USTAR_FORMAT) as saved:
            for name, raw in contents:
                m = tarfile.TarInfo(name); m.mode = 0o444
                if fault == 'layer_bytes' and name == names[0]: raw += b'altered'
                if fault == 'link' and name == names[0]: m.type = tarfile.SYMTYPE; m.linkname = '/foreign'
                else: m.size = len(raw)
                saved.addfile(m, io.BytesIO(raw))
        if fault is None: transfer.verify_image_archive(path)
        else:
            with pytest.raises(ValueError): transfer.verify_image_archive(path)


def test_source_no_model_or_dataset_execution_secret_args_or_broad_cleanup(transfer):
    source = Path(transfer.__file__).read_text()
    assert all(token not in source for token in ('import torch', 'import numpy', 'docker rmi', 'rmtree(', 'shell=True', 'nvidia-smi'))
    assert "os.environ.pop('WR_BOOTSTRAP_BLOB_URL'" in source
    assert "signal.alarm(BUDGET)" in source and 'BUDGET = 7200' in source
    assert 'inspect_image()' in source and "'layers': 47" in source
    assert "docker', 'image', 'save'" in source and "docker', 'image', 'load'" in source
    assert "os.fchmod(stream.fileno(), 0o400)" in source
