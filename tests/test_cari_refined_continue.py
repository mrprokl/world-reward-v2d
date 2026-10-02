"""Explicit own continuation cannot redo acquisition or overwrite readers."""
from pathlib import Path
import subprocess


def test_continuation_no_asset_or_preflight_mutation_and_strict_new_outputs():
    p=Path(__file__).resolve().parents[1]/'infra/run_cari_refined_continue.sh';s=p.read_text()
    subprocess.run(['bash','-n',str(p)],check=True)
    assert 'run_cari_refinement_assets' not in s and 'run_cari_refinement_preflight' not in s
    assert 'cari_refined cari_conversion_refined final_schema_refined' in s
    assert '[[ -z "$WR_WAIT_FOR" ]]' in s and '--bundle-source refined --no-wait' in s
    assert s.index('Never resume or overwrite')<s.index('run_cari_refine.sh')<s.index('run_cari_converter.sh')<s.index('run_final_episode_gate.sh')
