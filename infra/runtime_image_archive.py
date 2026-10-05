"""Authenticate one classic/hybrid Docker save without loading/extracting it.

Only a pinned config and its complete ordered diff-ID chain are accepted. OCI
indices/manifests are authenticated links, not interchangeable Docker identities.
Attestations, foreign URLs/tags, Parent images and unsupported compression abstain.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tarfile

MAX_ARCHIVE = 64 << 30
MAX_EXPANDED = 64 << 30
MAX_LAYER = 32 << 30
MAX_MEMBERS = 512
MAX_JSON = 1 << 20
BLOCK = 1 << 20
_INDEX = ('application/vnd.oci.image.index.v1+json',
          'application/vnd.docker.distribution.manifest.list.v2+json')
_MANIFEST = ('application/vnd.oci.image.manifest.v1+json',
             'application/vnd.docker.distribution.manifest.v2+json')
_CONFIG = ('application/vnd.oci.image.config.v1+json',
           'application/vnd.docker.container.image.v1+json')
_RAW = ('application/vnd.oci.image.layer.v1.tar',
        'application/vnd.docker.image.rootfs.diff.tar')
_GZIP = ('application/vnd.oci.image.layer.v1.tar+gzip',
         'application/vnd.docker.image.rootfs.diff.tar.gzip')
_STABLE = ('st_dev', 'st_ino', 'st_size', 'st_mode', 'st_nlink',
           'st_uid', 'st_gid', 'st_mtime_ns', 'st_ctime_ns')


class _ImageTarInfo(tarfile.TarInfo):
    # Reject hidden extensions even if an archive changes after header precheck.
    def _proc_member(self, tar):
        _require(self.type in (tarfile.REGTYPE, tarfile.AREGTYPE, tarfile.DIRTYPE), 'Unsupported image TAR header')
        return super()._proc_member(tar)


def _require(value, message):
    if not value: raise ValueError(message)


def _sha(value):
    _require(type(value) is str and re.fullmatch('sha256:[0-9a-f]{64}', value), 'SHA256 descriptor required')
    return value


def _name(value):
    _require(type(value) is str and value and len(value) <= 256
        and PurePosixPath(value).as_posix() == value and not value.startswith('/')
        and not any(p in ('.', '..') for p in value.split('/'))
        and not any(ord(c) < 32 or ord(c) == 127 for c in value) and '\\' not in value,
        'Canonical bounded archive-relative name required')
    value.encode('utf-8', errors='strict')
    return value


def _json(raw):
    def pairs(rows):
        out = {}
        for k, v in rows:
            _require(k not in out, 'Duplicate JSON key'); out[k] = v
        return out
    def real(token):
        value = float(token); _require(math.isfinite(value), 'Nonfinite JSON'); return value
    return json.loads(raw, object_pairs_hook=pairs, parse_float=real,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def _descriptor(value, media):
    _require(type(value) is dict and {'digest', 'size', 'mediaType'} <= set(value)
        and set(value) <= {'digest', 'size', 'mediaType', 'platform', 'annotations'}
        and value['mediaType'] in media and type(value['size']) is int
        and 0 < value['size'] <= MAX_LAYER and value.get('annotations', {}) == {},
        'Bounded local descriptor without tags/URLs/data required')
    _sha(value['digest'])
    if 'platform' in value:
        _require(value['platform'] == {'architecture': 'amd64', 'os': 'linux'}, 'Only exact linux/amd64 platform supported')
    return value


def _snapshot(value): return tuple(getattr(value, field) for field in _STABLE)


def authenticate(path, expected_config_id, expected_ordered_rootfs_sha256, expected_layer_count=47):
    """Return authenticated IDs/rootfs; no Docker, network, layer unpack or alias.

    Compressed blob hashes bind OCI descriptors; their bounded uncompressed byte
    hashes independently bind config diff_ids. The containing TAR is checked
    for stable inode/content metadata, NOT claimed publisher-SHA authenticated.
    Caller provides the wall-clock deadline and independently pins archive bytes.
    """
    _sha(expected_config_id)
    _require(type(expected_ordered_rootfs_sha256) is str and re.fullmatch('[0-9a-f]{64}', expected_ordered_rootfs_sha256)
        and type(expected_layer_count) is int and 1 <= expected_layer_count <= 256, 'Exact rootfs pin/count required')
    p = Path(path)
    _require(p.is_absolute() and p.resolve() == p and not any(q.is_symlink() for q in (p, *p.parents)), 'Canonical nonsymlink archive required')
    before = p.lstat()
    _require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and 0 < before.st_size <= MAX_ARCHIVE, 'Bounded single-link regular archive required')
    with os.fdopen(os.open(p, os.O_RDONLY | os.O_NOFOLLOW), 'rb') as source:
        _require(_snapshot(os.fstat(source.fileno())) == _snapshot(before), 'Archive replaced before open')
        # Reject extension headers before tarfile can allocate their payload.
        offset = 0; headers = 0
        while offset < before.st_size:
            source.seek(offset); header = source.read(512)
            _require(len(header) == 512, 'Truncated image TAR header')
            if header == b'\0'*512: break
            info = tarfile.TarInfo.frombuf(header, 'utf-8', 'strict'); headers += 1
            _require(headers <= MAX_MEMBERS and info.type in (tarfile.REGTYPE, tarfile.AREGTYPE, tarfile.DIRTYPE)
                and 0 <= info.size <= MAX_LAYER and (not info.isdir() or info.size == 0), 'Bounded nonextended image TAR header required')
            offset += 512+((info.size+511)//512)*512
            _require(offset <= before.st_size, 'Truncated image TAR member')
        source.seek(0)
        with tarfile.open(fileobj=source, mode='r:', tarinfo=_ImageTarInfo) as tar:
            table = {}; total = 0; end = 0
            for m in tar:
                _require(len(table) < MAX_MEMBERS and not m.pax_headers
                    and m.type in (tarfile.REGTYPE, tarfile.AREGTYPE, tarfile.DIRTYPE)
                    and m.offset_data == m.offset+512, 'Bounded unaliased non-PAX image TAR required')
                name = _name(m.name.rstrip('/') if m.isdir() else m.name)
                _require(name not in table and type(m.size) is int and 0 <= m.size <= MAX_LAYER
                    and (not m.isdir() or m.size == 0), 'Duplicate or oversized image member')
                total += m.size; _require(total <= MAX_ARCHIVE, 'Image member-byte cap exceeded')
                end = m.offset_data+((m.size+511)//512)*512
                table[name] = m
            for name in table:
                _require(all(str(q) not in table or table[str(q)].isdir()
                    for q in PurePosixPath(name).parents if str(q) != '.'), 'File/directory image alias refused')
            source.seek(end)
            for block in iter(lambda: source.read(BLOCK), b''):
                _require(not any(block), 'Nonzero trailing image TAR bytes')
            used = set(); layer_rows = []
            def raw(name, maximum=None):
                if maximum is None: maximum = MAX_JSON
                name = _name(name); m = table.get(name)
                _require(m is not None and m.isreg() and 0 < m.size <= maximum, 'Missing/bounded regular image metadata required')
                used.add(name)
                with tar.extractfile(m) as stream: value = stream.read(maximum+1)
                _require(len(value) == m.size, 'Truncated image metadata')
                return value
            def blob(d, media):
                _descriptor(d, media); name = 'blobs/sha256/'+d['digest'][7:]
                value = raw(name)
                _require(len(value) == d['size'] and 'sha256:'+hashlib.sha256(value).hexdigest() == d['digest'], 'OCI metadata descriptor digest/size differs')
                return value
            legacy = _json(raw('manifest.json', 65536))
            _require(type(legacy) is list and len(legacy) == 1 and type(legacy[0]) is dict, 'One Docker image required')
            row = legacy[0]
            _require({'Config', 'RepoTags', 'Layers'} <= set(row) and set(row) <= {'Config', 'RepoTags', 'Layers', 'LayerSources'}
                and row['RepoTags'] in (None, [])
                and type(row['Layers']) is list and len(row['Layers']) == expected_layer_count, 'Only one untagged image without parent allowed')
            config_raw = raw(row['Config']); config = _json(config_raw)
            _require('sha256:'+hashlib.sha256(config_raw).hexdigest() == expected_config_id
                and type(config) is dict and config.get('architecture') == 'amd64' and config.get('os') == 'linux'
                and type(config.get('rootfs')) is dict and config['rootfs'].get('type') == 'layers', 'Original pinned linux/amd64 config differs')
            diff_ids = config['rootfs'].get('diff_ids')
            _require(type(diff_ids) is list and len(diff_ids) == expected_layer_count, 'Full original diff-ID list required')
            for diff_id in diff_ids: _sha(diff_id)
            rootfs_sha = hashlib.sha256(json.dumps(diff_ids, separators=(',', ':')).encode()).hexdigest()
            _require(rootfs_sha == expected_ordered_rootfs_sha256, 'Ordered original rootfs differs')
            for name in row['Layers']: _name(name)
            hybrid = 'oci-layout' in table or 'index.json' in table
            platform_id = None; index_ids = []; codecs = [_RAW[0]]*expected_layer_count
            if hybrid:
                _require(_json(raw('oci-layout', 65536)) == {'imageLayoutVersion': '1.0.0'}, 'Exact OCI layout1 required')
                index_raw = raw('index.json'); index = _json(index_raw)
                index_ids.append('sha256:'+hashlib.sha256(index_raw).hexdigest())
                for depth in range(3):
                    _require(type(index) is dict and set(index) <= {'schemaVersion', 'mediaType', 'manifests'}
                        and type(index.get('schemaVersion')) is int and index['schemaVersion'] == 2
                        and index.get('mediaType', _INDEX[0]) in _INDEX
                        and type(index.get('manifests')) is list and len(index['manifests']) == 1, 'One-platform bounded OCI index required; no attestations')
                    d = index['manifests'][0]; _descriptor(d, (*_INDEX, *_MANIFEST))
                    value = blob(d, (*_INDEX, *_MANIFEST)); node = _json(value)
                    if d['mediaType'] in _MANIFEST:
                        platform_id = d['digest']; break
                    _require(depth < 2, 'OCI index depth cap exceeded')
                    _require(type(node) is dict and node.get('mediaType') == d['mediaType'], 'OCI index descriptor media type differs')
                    index_ids.append(d['digest']); index = node
                _require(type(node) is dict and set(node) <= {'schemaVersion', 'mediaType', 'config', 'layers'}
                    and type(node.get('schemaVersion')) is int and node['schemaVersion'] == 2
                    and node.get('mediaType') == d['mediaType'] and type(node.get('layers')) is list
                    and len(node['layers']) == expected_layer_count, 'Exact OCI platform manifest required')
                cd = _descriptor(node.get('config'), _CONFIG)
                _require(cd['digest'] == expected_config_id and blob(cd, _CONFIG) == config_raw
                    and row['Config'] == 'blobs/sha256/'+cd['digest'][7:], 'Classic/OCI config selection differs')
                for name, d in zip(row['Layers'], node['layers']):
                    _descriptor(d, (*_RAW, *_GZIP)); _require(name == 'blobs/sha256/'+d['digest'][7:], 'Classic/OCI ordered layer selection differs')
                    layer_rows.append(d); codecs[len(layer_rows)-1] = d['mediaType']
            sources = row.get('LayerSources')
            _require('LayerSources' not in row or type(sources) is dict, 'LayerSources must be a descriptor map')
            if sources:
                _require(hybrid and set(sources) <= set(diff_ids), 'LayerSources requires authenticated OCI correspondence')
                for diff, d in zip(diff_ids, layer_rows):
                    if diff not in sources: continue
                    actual = _descriptor(sources[diff], (*_RAW, *_GZIP))
                    _require(actual == d, 'LayerSources descriptor not identical to authenticated graph')
            expanded_total = 0; verified = {}
            for i, (name, diff, codec) in enumerate(zip(row['Layers'], diff_ids, codecs)):
                m = table.get(name); _require(m is not None and m.isreg() and 0 < m.size <= MAX_LAYER, 'All original regular image layers required')
                used.add(name)
                if hybrid: _require(m.size == layer_rows[i]['size'], 'OCI layer blob size differs')
                if name in verified:
                    old_diff, old_codec, size = verified[name]
                    _require((old_diff, old_codec) == (diff, codec), 'Repeated blob conflicts with ordered rootfs')
                    expanded_total += size; _require(expanded_total <= MAX_EXPANDED, 'Full uncompressed image cap exceeded'); continue
                digest = hashlib.sha256(); raw_size = 0
                with tar.extractfile(m) as stream:
                    for block in iter(lambda: stream.read(BLOCK), b''): digest.update(block); raw_size += len(block)
                _require(raw_size == m.size, 'Truncated layer blob')
                if hybrid:
                    d = layer_rows[i]
                    _require(m.size == d['size'] and 'sha256:'+digest.hexdigest() == d['digest'], 'OCI layer blob digest/size differs')
                if codec in _GZIP:
                    digest = hashlib.sha256(); size = 0; limit = min(MAX_LAYER, MAX_EXPANDED-expanded_total)
                    with tar.extractfile(m) as stream, gzip.GzipFile(fileobj=stream, mode='rb') as decoded:
                        while True:
                            block = decoded.read(min(BLOCK, limit-size+1))
                            if not block: break
                            size += len(block); _require(size <= limit, 'Uncompressed layer cap exceeded'); digest.update(block)
                else: size = m.size
                expanded_total += size; _require(expanded_total <= MAX_EXPANDED, 'Full uncompressed image cap exceeded')
                _require('sha256:'+digest.hexdigest() == diff, 'Original uncompressed layer diff-ID differs')
                verified[name] = (diff, codec, size)
            for i, name in enumerate(row['Layers']):
                if name.startswith('blobs/'): continue
                parent = str(PurePosixPath(name).parent)
                version = parent+'/VERSION'; metadata = parent+'/json'
                if version in table: _require(raw(version, 16) in (b'1.0', b'1.0\n'), 'Classic layer VERSION differs')
                if metadata in table:
                    previous = PurePosixPath(row['Layers'][i-1]).parent.name if i else ''
                    info = _json(raw(metadata)); _require(type(info) is dict
                        and info.get('id') == PurePosixPath(parent).name
                        and info.get('parent', '') in ('', previous), 'Classic layer metadata parent/ID differs')
            if 'repositories' in table: _require(_json(raw('repositories', 65536)) == {}, 'No repository tags may be imported')
            _require(all(name in used or m.isdir() and any(p.startswith(name+'/') for p in used)
                for name, m in table.items()), 'Unreferenced image graph/member refused')
            receipt = dict(schema='world_reward.runtime_image_archive.v1', representation='hybrid_oci' if hybrid else 'classic',
                config_id=expected_config_id, platform_manifest_id=platform_id, index_ids=index_ids,
                ordered_rootfs_sha256=rootfs_sha, rootfs_diff_ids=diff_ids, layer_count=expected_layer_count,
                unique_layer_blobs=len(verified), uncompressed_layer_bytes=expanded_total,
                compressed_layers=sum(codec in _GZIP for codec in codecs), archive_bytes=before.st_size,
                archive_whole_sha_verified=False, layers_extracted=False, docker_loaded=False)
        _require(_snapshot(os.fstat(source.fileno())) == _snapshot(before) == _snapshot(p.lstat()), 'Archive changed during authentication')
    return receipt
