"""One independent serial camera hypothesis; no data or GPU execution."""
from pathlib import Path
import subprocess


def test_new_camera_chain_reuses_public_rgb_not_renderer_or_gt():
    path = Path(__file__).parents[1]/"infra/run_joint_rgb_camera_chain.sh"
    source = path.read_text()
    assert 'inactive|failed) break' in source and '43200' in source
    assert 'predictions_camera_v1 quality_camera_v1' in source
    assert 'run_joint_rgb_camera_infer.sh' in source and 'run_joint_rgb_camera_evaluate.sh' in source
    assert 'run_joint_rgb_render.sh' not in source and 'run_joint_rgb_masks.sh' not in source
    assert 'eval_private' not in source
    subprocess.run(["bash", "-n", str(path)], check=True)
