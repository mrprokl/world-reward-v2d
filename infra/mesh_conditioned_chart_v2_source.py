"""Prepare a source-bound v2 backend, without compiling or executing it.

Only the Chart definition and explicit build identity change. The cached
collapse/intersection/volume/serialization body and its main flow stay exact.
The historical backbone and all its qualification/failure records stay intact.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import stat

from world_reward.mesh_conditioning_v2 import POLICY_SHA256

BACKBONE_BYTES = 16162
BACKBONE_SHA256 = "f97d826d8543822c7ebbe11b4ef12c69a29c246a0902a0ea7484515c3f79148b"
BACKBONE_FILE = "mesh_conditioned_qem.cpp"
HEADER_FILE = "mesh_conditioned_chart_v2.hpp"
CHART_START = b"struct Chart {\n"
CHART_END = b"\n\nbool simplify("
BUILD_ID = b'      std::cout<<"{\\"source_sha256\\":\\""<<WR_CONDITIONED_SOURCE_SHA256\n'
NEW_BUILD_ID = (
    b'      std::cout<<"{\\"chart_version\\":2,\\"origin_search_performed\\":false,'
    b'\\"backbone_source_sha256\\":\\""<<WR_CONDITIONED_SOURCE_SHA256\n'
    b'        <<"\\",\\"chart_header_sha256\\":\\""<<WR_CONDITIONED_CHART_V2_SHA256\n'
    b'        <<"\\",\\"chart_policy_sha256\\":\\""<<WR_CONDITIONED_CHART_V2_POLICY_SHA256\n'
    b'        <<"\\",\\"source_sha256\\":\\""<<WR_CONDITIONED_V2_GENERATED_SHA256\n'
)


def _read(path):
    path = Path(path)
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or not 0 < before.st_size <= 100000:
        raise ValueError("Bound regular single-link source required")
    raw = path.read_bytes()
    after = path.lstat()
    fields = ("st_dev", "st_ino", "st_mode", "st_nlink", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, k) != getattr(after, k) for k in fields):
        raise ValueError("Source changed during derivation")
    return raw


def derive_cached_backend(backbone_path, header_path):
    """Return generated C++ bytes and required immutable compiler identities.

    Caller must publish the returned header unchanged, keep all inherited core
    prefix identities, set WR_CONDITIONED_CACHE=1, and qualify this NEW backend.
    This function writes no source/binary/output and claims no native execution.
    """
    if Path(backbone_path).name != BACKBONE_FILE or Path(header_path).name != HEADER_FILE:
        raise ValueError("Explicit backbone and v2 header filenames required")
    raw, header = _read(backbone_path), _read(header_path)
    if len(raw) != BACKBONE_BYTES or hashlib.sha256(raw).hexdigest() != BACKBONE_SHA256:
        raise ValueError("Historical cached backbone identity differs")
    if raw.count(CHART_START) != 1 or raw.count(CHART_END) != 1 or raw.count(BUILD_ID) != 1:
        raise ValueError("Historical source segment ABI differs")
    start, end = raw.index(CHART_START), raw.index(CHART_END)
    if start >= end or not raw[start:end].endswith(b"};"):
        raise ValueError("Explicit historical chart boundary differs")
    derived = raw[:start] + b'#include "mesh_conditioned_chart_v2.hpp"' + raw[end:]
    derived = derived.replace(BUILD_ID, NEW_BUILD_ID, 1)
    guards = (b'#ifndef WR_CONDITIONED_V2_GENERATED_SHA256\n#error Bind exact derived v2 source\n#endif\n'
              b'#if WR_CONDITIONED_CACHE != 1\n#error Chart v2 backend requires the cached backbone\n#endif\n')
    derived = guards + derived
    header_sha = hashlib.sha256(header).hexdigest()
    generated_sha = hashlib.sha256(derived).hexdigest()
    return derived, {
        "schema": "world-reward-mesh-conditioned-chart-v2-derived-source-v1",
        "backbone": {"bytes": len(raw), "sha256": BACKBONE_SHA256},
        "header": {"bytes": len(header), "sha256": header_sha},
        "generated_source": {"bytes": len(derived), "sha256": generated_sha},
        "compiler_macros": {"WR_CONDITIONED_SOURCE_SHA256": BACKBONE_SHA256,
            "WR_CONDITIONED_V2_GENERATED_SHA256": generated_sha,
            "WR_CONDITIONED_CHART_V2_SHA256": header_sha,
            "WR_CONDITIONED_CHART_V2_POLICY_SHA256": POLICY_SHA256,
            "WR_CONDITIONED_CACHE": 1},
        "chart_version": 2, "cached_backbone_body_unchanged": True,
        "native_backend_qualified": False, "adopted": False,
    }
