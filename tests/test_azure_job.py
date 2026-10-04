"""Tiny header closure regression: no network, native build or Azure call."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("wr_azure_header_closure", ROOT / "infra/azure_job.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def files():
    return {"infra/run_build.sh": b'python "$CODE/infra/build.py"\n',
            "infra/build.py": b'HEADER="mesh_conditioned_chart_v2.hpp"\n',
            "infra/mesh_conditioned_chart_v2.hpp": b'#include "exact_dyadic.h"\n',
            "infra/exact_dyadic.h": b'#include "generated_or_vendor.hpp"\n',
            "infra/unused.hpp": b"// unrelated not transported\n",
            "configs/build.json": b"{}", "pyproject.toml": b""}


def test_literal_header_and_transitive_quoted_header_are_exact_committed_closure():
    source = files()
    selected = launcher.runtime_bundle_paths(source, "infra/run_build.sh")
    assert selected == sorted(set(source)-{"infra/unused.hpp"})
    assert "infra/generated_or_vendor.hpp" not in selected


@pytest.mark.parametrize("reference", [
    b'from pathlib import Path\np=Path(__file__).with_name("mesh_conditioned_chart_v2.hpp")\n',
    b'helpers={"infra/mesh_conditioned_chart_v2.hpp"}\n',
])
def test_explicit_header_source_refs_require_committed_file(reference):
    source = files(); source["infra/build.py"] = reference
    assert "infra/mesh_conditioned_chart_v2.hpp" in launcher.runtime_bundle_paths(source, "infra/run_build.sh")
    del source["infra/mesh_conditioned_chart_v2.hpp"]
    with pytest.raises(ValueError, match="dependency is not committed"):
        launcher.runtime_bundle_paths(source, "infra/run_build.sh")


def test_shell_and_source_closure_comments_include_both_header_suffixes():
    source = files(); source["infra/build.py"] = b"VALUE=1\n"
    source["infra/run_build.sh"] += (b'# Source closure: /infra/mesh_conditioned_chart_v2.hpp /infra/exact_dyadic.h\n')
    selected = launcher.runtime_bundle_paths(source, "infra/run_build.sh")
    assert {"infra/mesh_conditioned_chart_v2.hpp", "infra/exact_dyadic.h"} <= set(selected)
    assert "infra/unused.hpp" not in selected
    source["infra/run_build.sh"] += b'# Source closure: /infra/missing.hpp\n'
    with pytest.raises(ValueError, match="dependency is not committed"):
        launcher.runtime_bundle_paths(source, "infra/run_build.sh")
