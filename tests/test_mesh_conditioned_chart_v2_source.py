"""Source derivation/contract tests only: no C++ compilation or native call."""
import hashlib
import json
from pathlib import Path
import sys
import re

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "infra"))
import mesh_conditioned_chart_v2_source as source


def derived():
    return source.derive_cached_backend(ROOT / "infra" / source.BACKBONE_FILE,
                                        ROOT / "infra" / source.HEADER_FILE)


def test_exact_cached_body_mapping_and_main_flow_are_not_copied_or_changed():
    before = (ROOT / "infra" / source.BACKBONE_FILE).read_bytes()
    cpp, receipt = derived()
    marker = b"\n\nbool simplify("
    end = before.index(b"\nint main(")
    assert before[before.index(marker):end] == cpp[cpp.index(marker):cpp.index(b"\nint main(")]
    flow = b"    if(argc==3 && std::string(argv[1])==\"--preflight\")"
    assert before[before.index(flow):] == cpp[cpp.index(flow):]
    assert cpp.count(b'#include "mesh_conditioned_chart_v2.hpp"') == 1
    assert source.CHART_START not in cpp
    assert receipt["cached_backbone_body_unchanged"]
    assert receipt["generated_source"]["sha256"] == hashlib.sha256(cpp).hexdigest()
    assert receipt["compiler_macros"]["WR_CONDITIONED_CACHE"] == 1
    assert receipt["native_backend_qualified"] is False and receipt["adopted"] is False
    assert (ROOT / "infra" / source.BACKBONE_FILE).read_bytes() == before


def test_build_info_discloses_new_derived_source_header_and_historical_backbone():
    cpp, receipt = derived()
    exact = (b'      std::cout<<"{\\"chart_version\\":2,\\"origin_search_performed\\":false,'
             b'\\"backbone_source_sha256\\":\\""<<WR_CONDITIONED_SOURCE_SHA256\n')
    assert exact in cpp
    for macro in ("WR_CONDITIONED_V2_GENERATED_SHA256", "WR_CONDITIONED_CHART_V2_SHA256",
                  "WR_CONDITIONED_CHART_V2_POLICY_SHA256"):
        assert macro.encode() in cpp
        assert len(receipt["compiler_macros"][macro]) == 64
    assert receipt["compiler_macros"]["WR_CONDITIONED_SOURCE_SHA256"] == source.BACKBONE_SHA256
    assert b"#if WR_CONDITIONED_CACHE != 1" in cpp
    # Decode the actual C++ string prefix, not a hand-written surrogate receipt.
    start = cpp.index(b"      std::cout<<")
    block = cpp[start:cpp.index(b"      return 0;", start)].decode()
    parts = []
    for token in re.findall(r'"(?:\\.|[^"\\])*"|\bWR_[A-Z0-9_]+\b', block):
        if token == "WR_SERIALIZATION_SOURCE_SHA256": break
        parts.append(json.loads(token) if token.startswith('"') else receipt["compiler_macros"][token])
    info = json.loads("".join(parts)+'inherited"}')
    assert info["chart_version"] == 2
    assert info["source_sha256"] == receipt["generated_source"]["sha256"]
    assert info["backbone_source_sha256"] == source.BACKBONE_SHA256


def test_native_header_exact_source_rule_range_inverse_and_no_origin_retry():
    header = (ROOT / "infra" / source.HEADER_FILE).read_text()
    select = header.index("midpoint_selected[j]=midpoint[j]!=0 && sterbenz(V,j,midpoint[j]);")
    verify = header.index("roundtrip[j]!=p[j]")
    assert select < verify
    assert "2*double_units(smallest)>=y && double_units(largest)<=2*y" in header
    assert "std::ldexp(double(delta),-exponent)" in header
    assert "std::ldexp(p[j],exponent)" in header
    assert "origin[j]==0 ? p[j] : p[j]-origin[j]" in header
    assert "origin[j]==0 ? double(delta) : origin[j]+delta" in header
    assert "gradual binary64 underflow" in header
    assert "no alternate origin" in header
    assert "for(int v=0;v<V.rows();++v)" in header
    assert "try {" not in header and "catch(" not in header


@pytest.mark.parametrize("change", ["content", "filename", "symlink", "hardlink"])
def test_historical_source_identity_or_regular_file_change_is_rejected(tmp_path, change):
    path = tmp_path / source.BACKBONE_FILE
    original = ROOT / "infra" / source.BACKBONE_FILE
    if change == "symlink": path.symlink_to(original)
    elif change == "hardlink":
        target = tmp_path / "copy.cpp"
        target.write_bytes(original.read_bytes())
        path.hardlink_to(target)
    else:
        path.write_bytes(original.read_bytes() + (b"// changed\n" if change == "content" else b""))
        if change == "filename": path = path.rename(tmp_path / "wrong.cpp")
    with pytest.raises(ValueError):
        source.derive_cached_backend(path, ROOT / "infra" / source.HEADER_FILE)


def test_source_mutation_during_read_fails_before_generated_publication(tmp_path, monkeypatch):
    path = tmp_path / source.BACKBONE_FILE
    path.write_bytes((ROOT / "infra" / source.BACKBONE_FILE).read_bytes())
    read = Path.read_bytes
    def altered(self):
        raw = read(self)
        if self == path: self.write_bytes(raw+b" ")
        return raw
    monkeypatch.setattr(Path, "read_bytes", altered)
    with pytest.raises(ValueError, match="changed"):
        source.derive_cached_backend(path, ROOT / "infra" / source.HEADER_FILE)
