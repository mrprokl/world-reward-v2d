"""CPU-only import preflight isolation and immutable runtime closure."""
import importlib.util
from pathlib import Path
import subprocess


def test_preflight_exposes_no_models_outputs_gpu_or_private_inputs():
    root = Path(__file__).parents[1]
    wrapper = (root / "infra/run_cari_refinement_preflight.sh").read_text()
    source = (root / "infra/cari_refinement_preflight.py").read_text()
    assert "--network none --memory 8g --cpus 4" in wrapper and "--gpus" not in wrapper
    assert "--env CUDA_VISIBLE_DEVICES=''" in wrapper and "60s docker run" in wrapper
    assert "src=$ROOT/outputs" not in wrapper and "src=$ROOT/data" not in wrapper and "src=$ROOT/weights" not in wrapper
    assert "actual_refinement_verified" in source and "torch.cuda.is_initialized()" in source
    assert "MHRLayer.from_mhr_assets" not in source and "torch.load" not in source
    subprocess.run(["bash", "-n", str(root / "infra/run_cari_refinement_preflight.sh")], check=True)


def test_refined_chain_import_closure_includes_all_stages_and_small_assets_acquirer():
    root = Path(__file__).parents[1]
    spec = importlib.util.spec_from_file_location("test_refine_launcher", root / "infra/azure_job.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    files = {str(p.relative_to(root)): p.read_bytes() for base in ("infra", "src", "configs")
             for p in (root / base).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    files['pyproject.toml'] = (root / 'pyproject.toml').read_bytes()
    selected = module.runtime_bundle_paths(files, 'infra/run_cari_refined_chain.sh')
    assert set(selected) >= {'infra/cari_refine.py', 'infra/cari_converter.py', 'infra/cari_wrapper_common.sh',
                             'infra/cari_refinement_preflight.py', 'infra/acquire_cari_refinement_assets.py',
                             'infra/final_episode_gate.py', 'infra/track1_episode_loader.py'}
    assert 'infra/cari_forward.py' not in selected and 'infra/object_pose_smoke.py' not in selected
