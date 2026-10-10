"""Tiny CPU runner contracts; never API, Azure, original video or credentials."""
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

import pytest

import sam31_late_anchor_run as runtime

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_cpu_stage_two_automatic_original_anchors():
    c, rc = runtime.settings(ROOT)
    assert c['episodes'] == [1, 7] and c['original_cohort'] == [9, 1, 14, 7]
    assert c['saved_forward_native_report'] == rc['saved_forward_native_report']
    assert c['concurrency'] == 2 and c['budget_seconds'] == 300
    assert c['cpu_image'] == runtime.BASE and c['model'] == 'gemini-3.5-flash'


@pytest.mark.parametrize('change', [lambda c: c.update(episodes=[9, 1]), lambda c: c.update(ground_truth_used=True),
    lambda c: c.update(model='gemini-pro'), lambda c: c.update(concurrency=8),
    lambda c: c.update(frames_decoded_per_clip=300), lambda c: c.update(manual_labels=True),
    lambda c: c.update(reference_frame=29), lambda c: c.update(anchor_policy='find_best_frame')])
def test_frozen_cpu_contract_not_score_tuning(tmp_path, change):
    (tmp_path / 'configs').mkdir()
    for name in (runtime.CONFIG, 'configs/sam31_recovery_v1.json', 'configs/sam31_runtime_v1.json'):
        content = json.loads((ROOT / name).read_text())
        if name == runtime.CONFIG: change(content)
        (tmp_path / name).write_text(json.dumps(content))
    with pytest.raises(ValueError): runtime.settings(tmp_path)


def test_last_native_visible_frame_not_perclip_selected_for_quality():
    assert runtime.final_visible([1] * 40 + [0, 0]) == 39
    assert runtime.final_visible([0] * 40 + [3, 0]) == 40
    with pytest.raises(ValueError): runtime.final_visible([1] * 29 + [0] * 20)
    with pytest.raises(ValueError): runtime.final_visible([True] * 40)


def test_decode_twoframe_seek_and_rgb_binding_without_full_decode(monkeypatch):
    import numpy as np
    import hashlib
    rgb = np.zeros((4, 5, 3), np.uint8); rgb[:, :, 0] = 42
    rgb_sha = hashlib.sha256(rgb.tobytes()).hexdigest()
    calls = []
    class Cap:
        position = 0
        def __init__(self, video): calls.append(('open', video))
        def isOpened(self): return True
        def set(self, code, index): self.position = index; calls.append(('seek', index)); return True
        def get(self, code): return self.position
        def read(self): self.position += 1; calls.append(('read', 1)); return True, rgb
        def release(self): calls.append(('release', 1))
    cv2 = SimpleNamespace(VideoCapture=Cap, CAP_PROP_POS_FRAMES=1, COLOR_BGR2RGB=2,
                          cvtColor=lambda a, code: a)
    monkeypatch.setitem(sys.modules, 'cv2', cv2)
    png, w, h = runtime.decode_frame('dummy-never-opened.mp4', 99, rgb_sha)
    assert png.startswith(b'\x89PNG\r\n\x1a\n') and (w, h) == (5, 4)
    assert calls == [('open', 'dummy-never-opened.mp4'), ('seek', 99), ('read', 1), ('release', 1)]


def test_decode_wrong_rgb_or_position_never_billable_model_call(monkeypatch):
    import numpy as np
    class Cap:
        def __init__(self, video): pass
        def isOpened(self): return True
        def set(self, code, index): return True
        def get(self, code): return 0
        def read(self): raise AssertionError('must reject wrong frame before read')
        def release(self): pass
    monkeypatch.setitem(sys.modules, 'cv2', SimpleNamespace(VideoCapture=Cap, CAP_PROP_POS_FRAMES=1,
        COLOR_BGR2RGB=2, cvtColor=lambda a, code: a))
    with pytest.raises(ValueError): runtime.decode_frame('dummy', 99, 'a' * 64)


def test_transport_one_request_exact_vertex_url_token_only_in_ram(monkeypatch):
    calls = []
    class Response:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self, maximum): calls.append(('limit', maximum)); return b'{"fake":true}'
    class Opener:
        def open(self, request, timeout):
            assert request.full_url == runtime.ENDPOINT
            assert request.get_header('Authorization') == 'Bearer synthetic-token'
            calls.append(('request', request.data, timeout)); return Response()
    monkeypatch.setattr(runtime.urllib.request, 'build_opener', lambda *args: Opener())
    send = runtime.transport('synthetic-token'); assert send(b'body', 12.) == b'{"fake":true}'
    assert calls == [('request', b'body', 12.), ('limit', 200001)]
    with pytest.raises(ValueError): runtime.transport('bad token')
    with pytest.raises(RuntimeError): runtime.NoRedirect().redirect_request()


def test_private_owned_envelope_cleanup_does_not_touch_replacement(tmp_path, monkeypatch):
    path = tmp_path / 'key.pem'; path.write_bytes(b'unit-private-key-not-real'); path.chmod(0o600)
    monkeypatch.setattr(runtime, 'KEY', path); monkeypatch.setattr(runtime, 'ENVELOPE', tmp_path / 'envelope')
    before = runtime.private_secret(path)
    runtime.remove_owned_secret(path, before); assert not path.exists()
    path.write_bytes(b'unit-private-key-not-real'); path.chmod(0o600); before = runtime.private_secret(path)
    path.write_bytes(b'changed-private-file'); path.chmod(0o600)
    with pytest.raises(ValueError): runtime.remove_owned_secret(path, before)
    assert path.exists()


def test_broad_permissions_and_symlinks_forbidden_on_secret(tmp_path):
    path = tmp_path / 'bad'; path.write_bytes(b'fake'); path.chmod(0o644)
    with pytest.raises(ValueError): runtime.private_secret(path)
    path.chmod(0o600); link = tmp_path / 'linked'; link.symlink_to(path)
    with pytest.raises(ValueError): runtime.private_secret(link)


def test_cpu_only_restricted_source_and_exact_owner_cleanup():
    raw = (ROOT / 'infra/sam31_late_anchor_run.py').read_text()
    wrapper = (ROOT / 'infra/run_sam31_late_anchor.sh').read_text()
    assert "'--memory', '8g', '--cpus', '4'" in raw
    assert "'--network', 'host'" in raw and "'NVIDIA_VISIBLE_DEVICES=void'" in raw
    assert "'--gpus'" not in raw and 'world-reward-h100.lock' not in raw
    assert "actual['Name'] == '/' + name" in raw and 'world_reward.sam31_late_anchor.owner' in raw
    assert "'rsa_oaep_md:sha256'" in raw and 'input=decrypted.stdout' in raw
    assert 'key.pem' in raw and '.enc' in raw and 'set +x' in wrapper
    assert 'run_sam31_late_anchor/code' in wrapper
    assert "write(out / f'vertex-response_" in raw and 'images_saved=False' in raw


def test_host_import_and_settings_no_site_packages_numeric_or_api_imports():
    code = r'''import sys,importlib.abc
from pathlib import Path
r=Path(sys.argv[1]);sys.path[:0]=[str(r/'infra'),str(r/'src')]
blocked={'numpy','scipy','torch','cv2','PIL','sam3'}
class Finder(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname.split('.')[0] in blocked or fullname=='world_reward.occlusion_recovery':
   raise AssertionError('unexpected host numerical import '+fullname)
sys.meta_path.insert(0,Finder())
import sam31_late_anchor_run as m
c,rc=m.settings(r)
assert c['episodes']==[1,7] and not blocked.intersection(sys.modules)
print('stdlib-only')'''
    result = subprocess.run(['rtk', 'proxy', sys.executable, '-I', '-S', '-B', '-c', code, str(ROOT)],
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'stdlib-only'
