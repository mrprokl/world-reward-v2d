"""Stdlib-only public H99 contract; PNG header fixtures contain no scene truth."""
import ast
from dataclasses import FrozenInstanceError
import importlib.util
import json
from pathlib import Path
import struct
import sys
import zlib

import pytest

REPO=Path(__file__).resolve().parents[1]
INFRA=REPO/"infra"


@pytest.fixture
def protocol(monkeypatch):
    monkeypatch.syspath_prepend(str(INFRA))
    spec=importlib.util.spec_from_file_location("factorial_rgb_protocol_test",INFRA/"factorial_rgb_protocol.py")
    m=importlib.util.module_from_spec(spec);monkeypatch.setitem(sys.modules,spec.name,m);spec.loader.exec_module(m);return m


def png_header(width=1024,height=768,color=2,interlace=0):
    body=struct.pack(">IIBBBBB",width,height,8,color,0,0,interlace)
    return b"\x89PNG\r\n\x1a\n"+struct.pack(">I",13)+b"IHDR"+body+struct.pack(">I",zlib.crc32(b"IHDR"+body)&0xffffffff)


def fixture(protocol,tmp_path):
    directory=tmp_path/"inputs";directory.mkdir();digests=[]
    for i in range(24):
        path=directory/protocol.COHORT.frame_name(i);path.write_bytes(png_header()+f"fixture{i}".encode());path.chmod(0o444)
        digests.append(protocol.identity(path)["sha256"])
    data=protocol.public_manifest(digests)
    def save():
        path=directory/"manifest.json"
        if path.exists():path.chmod(0o644)
        path.write_text(json.dumps(data));path.chmod(0o444)
    save();return directory,data,save


def test_frozen_exact_public_config_no_private_recipe(protocol):
    c=protocol.COHORT
    assert(c.groups,c.frames,c.count,c.width,c.height,c.prior_focal)==(8,3,24,1024,768,1280.)
    assert c.fixed_K==((1280.,0.,512.),(0.,1280.,384.),(0.,0.,1.))
    with pytest.raises(FrozenInstanceError):c.frames=5
    with pytest.raises(ValueError):protocol.PublicCohort(base="../private")
    with pytest.raises(ValueError):protocol.PublicCohort(frames=True)
    tree=ast.parse(Path(protocol.__file__).read_text());imports={n.module for n in tree.body if isinstance(n,ast.ImportFrom)}
    imports|={a.name for n in tree.body if isinstance(n,ast.Import)for a in n.names}
    assert not {"numpy","torch","PIL","factorial_rgb_render"}&imports
    assert "SHAPES"not in protocol.__dict__ and"YAWS"not in protocol.__dict__


@pytest.mark.parametrize("index",[-1,24,True,1.,None])
def test_exact_original_frame_indices(protocol,index):
    with pytest.raises(ValueError):protocol.COHORT.frame_name(index)


def test_all24_public_readonly_inputs_and_receipt(protocol,tmp_path):
    directory,_,_=fixture(protocol,tmp_path);records,receipt=protocol.public_inputs(directory)
    assert len(records)==24 and receipt==protocol.identity(directory/"manifest.json")
    assert [(r["group_index"],r["frame_index"])for r in records]==[(c,f)for c in range(8)for f in range(3)]
    assert all(set(r)=={"file","sha256","width","height","path","group_index","frame_index"}for r in records)


@pytest.mark.parametrize("fault",["schema","topprivate","rowprivate","order","count","booldim","badsha","tamper","writable","extra","symlink","directoryalias"])
def test_public_firewall_no_private_fallback(protocol,tmp_path,fault):
    directory,data,save=fixture(protocol,tmp_path);path=directory/data["images"][0]["file"]
    if fault=="schema":data["schema"]="world-reward-keypoint-rgb-v1"
    elif fault=="topprivate":data["camera_K"]=[[1]]
    elif fault=="rowprivate":data["images"][0]["shape"]=[0]
    elif fault=="order":data["images"].reverse()
    elif fault=="count":data["images"].pop()
    elif fault=="booldim":data["images"][0]["width"]=True
    elif fault=="badsha":data["images"][0]["sha256"]="z"*64
    elif fault=="tamper":path.chmod(0o644);path.write_bytes(png_header()+b"changed");path.chmod(0o444)
    elif fault=="writable":path.chmod(0o644)
    elif fault=="extra":(directory/"truth.npz").write_bytes(b"hidden")
    elif fault=="symlink":path.unlink();path.symlink_to(directory/data["images"][1]["file"])
    else:
        alias=tmp_path/"alias";alias.symlink_to(directory);directory=alias
    save()
    with pytest.raises(ValueError):protocol.public_inputs(directory)


@pytest.mark.parametrize("fault",["signature","crc","truncated","size","gray","RGBA","interlaced"])
def test_native_rgb_png_header_contract(protocol,tmp_path,fault):
    header=png_header()
    if fault=="signature":header=b"bad"+header[3:]
    elif fault=="crc":header=header[:-1]+bytes([header[-1]^1])
    elif fault=="truncated":header=header[:25]
    elif fault=="size":header=png_header(width=1000)
    elif fault=="gray":header=png_header(color=0)
    elif fault=="RGBA":header=png_header(color=6)
    else:header=png_header(interlace=1)
    path=tmp_path/"image.png";path.write_bytes(header)
    with pytest.raises(ValueError):protocol.validate_png_header(path)
