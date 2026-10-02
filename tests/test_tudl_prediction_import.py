from pathlib import Path
import subprocess


def test_import_frozen_predictions_exact_no_inference_private_mount_or_unsafe_extract():
    p=Path(__file__).resolve().parents[1]/'infra/run_tudl_prediction_import.sh';s=p.read_text()
    subprocess.run(['bash','-n',str(p)],check=True)
    assert '--gpus' not in s and 'eval_private,dst' not in s and '.extractall(' not in s
    assert "len(members)==11" in s and "len(files)==10" in s and "out.exists()" in s
    assert 'a187959136b5aeaca57447ac19e696e119df1607ebc40fac8a48a7d1b680682a' in s
    assert s.index('hashlib.sha256(rp)')<s.index('out.mkdir')
    assert 'public_predictions' in s and 'inference_rerun' in s
