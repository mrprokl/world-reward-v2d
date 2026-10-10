import inspect
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'infra'))
import form_track1_alias_bank as bank


def test_bank_does_not_infer_or_decode_private_truth():
    code=inspect.getsource(bank.build)
    assert 'range(30)' in code and 'decode_frame_hashes(p,cfg,deadline)' in code
    assert 'reference_read=False' in code and 'frames_per_video=96' in code
    assert 'readonly=False)==pin' in code
    assert 'track_2' not in code and 'eval_private' not in code


def test_fixed_bank_only_separate_private_transport():
    code=inspect.getsource(bank.PrivateBank)
    assert "{'PUT','HEAD','GET'}" in code
    assert 'track1-rgb-alias-bank.json' in code
    assert 'self.authorization()' in code
    assert 'public_access' not in code and 'sig=' not in code


def test_cpu_only_wrapper():
    wrapper=Path(bank.__file__).with_name('run_form_track1_alias_bank.sh').read_text()
    assert '--gpus' not in wrapper and '--network none' in wrapper
    assert 'videos/chunk-000/observation.images.exo_camera,readonly' in wrapper
