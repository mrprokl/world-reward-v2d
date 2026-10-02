"""Immutable stage order and independent GPU queue, no remote calls."""
from pathlib import Path
import subprocess


def test_refinement_chain_has_distinct_frozen_targets_and_explicit_final_bundle():
    source = (Path(__file__).parents[1] / "infra/run_cari_refined_chain.sh").read_text()
    assert "cari_refined cari_conversion_refined final_schema_refined" in source
    assert 'inactive|failed) break' in source and '43200' in source
    assert 'run_cari_refine.sh" --episode "$WR_EPISODE" --no-wait' in source
    assert 'run_cari_converter.sh" --episode "$WR_EPISODE" --bundle-source refined --no-wait' in source
    assert 'run_final_episode_gate.sh" --episode "$WR_EPISODE" --bundle-source refined' in source
    assert source.index('run_cari_refinement_assets.sh') < source.index('systemctl show') < source.index('run_cari_refine.sh')
    assert 'run_cari_forward.sh' not in source and 'run_cari_prepare.sh' not in source
    for name in ("run_cari_refined_chain.sh", "run_cari_refinement_assets.sh"):
        subprocess.run(["bash", "-n", str(Path(__file__).parents[1] / "infra" / name)], check=True)
