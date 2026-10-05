"""Bounded ZIP/ZIP64 central-directory metadata, never member payloads.

This intentionally accepts only single-disk, uncommented archives. A supplied
range callable must enforce the remote identity/206 firewall before body reads.
CRC fields are metadata, NOT evidence of verified member or whole-archive bytes.
"""
from pathlib import PurePosixPath
import stat
import struct


def require(value, message):
    if not value:
        raise ValueError(message)


def read_directory(get_range, archive_bytes, maximum_bytes, maximum_members, *, expected_layout=None):
    require(type(archive_bytes) is int and archive_bytes >= 98,
            'Bounded original archive length required')
    require(type(maximum_bytes) is int and maximum_bytes > 0
            and type(maximum_members) is int and maximum_members > 0,
            'Predeclared central-directory budgets required')
    end = archive_bytes - 22
    eocd = get_range(end, archive_bytes - 1)
    require(len(eocd) == 22 and eocd[:4] == b'PK\x05\x06',
            'Uncommented ZIP EOCD required; no tail scan into member data')
    _, disk, cd_disk, disk_count, count, length, offset, comment = struct.unpack('<4s4H2IH', eocd)
    require(disk == cd_disk == comment == 0 and disk_count == count, 'Single disk/comment-free ZIP required')
    boundary = end
    zip64 = count == 65535 or length == 4294967295 or offset == 4294967295
    if zip64:
        locator = get_range(end - 20, end - 1)
        require(len(locator) == 20 and locator[:4] == b'PK\x06\x07', 'Original ZIP64 locator required')
        _, disk, where, disks = struct.unpack('<4sIQI', locator)
        require(disk == 0 and disks == 1 and 0 <= where <= end - 76, 'Single-disk ZIP64 bounds required')
        record = get_range(where, where + 55)
        require(len(record) == 56 and record[:4] == b'PK\x06\x06', 'Original ZIP64 EOCD required')
        _, record_size, _, _, disk, cd_disk, disk_count, count, length, offset = struct.unpack('<4sQ2H2I4Q', record)
        require(record_size == 44 and where + 56 == end - 20
                and disk == cd_disk == 0 and disk_count == count, 'No ZIP64 extensible data or extra disks')
        boundary = where
    require(0 < count <= maximum_members and 46 * count <= length <= maximum_bytes
            and offset >= 0 and offset + length == boundary, 'Central-directory census bounds exceeded')
    layout = dict(members=count, bytes=length, offset=offset, zip64=zip64,
                  archive_bytes=archive_bytes, trailer_boundary=boundary)
    require(expected_layout is None or layout == expected_layout, 'Original diagnostic ZIP64 metadata changed')
    raw = get_range(offset, offset + length - 1)
    require(len(raw) == length, 'Complete central-directory bytes required')
    return raw, layout


def parse_directory(raw, layout):
    require(type(raw) is bytes and len(raw) == layout['bytes'], 'Original central-directory bytes required')
    result, names, pos = [], set(), 0
    for _ in range(layout['members']):
        require(pos + 46 <= len(raw), 'Truncated central header')
        row = struct.unpack_from('<4s6H3I5H2I', raw, pos)
        require(row[0] == b'PK\x01\x02', 'Original central header signature required')
        flags, compression = row[3:5]
        crc32, compressed, size = row[7:10]
        n, x, c, disk = row[10:14]
        attrs, where = row[15:17]
        stop = pos + 46 + n + x + c
        require(n > 0 and stop <= len(raw), 'Complete bounded filename/extras required')
        name = raw[pos + 46:pos + 46 + n].decode('utf-8' if flags & 2048 else 'cp437')
        extra = raw[pos + 46 + n:pos + 46 + n + x]
        chunks, i = {}, 0
        while i < len(extra):
            require(i + 4 <= len(extra), 'Truncated extra-field header')
            tag, length = struct.unpack_from('<HH', extra, i); i += 4
            require(i + length <= len(extra) and tag not in chunks, 'Duplicate/truncated extra field')
            chunks[tag] = extra[i:i + length]; i += length
        values = [size, compressed, where, disk]
        sentinels, widths = [4294967295] * 3 + [65535], [8, 8, 8, 4]
        if any(v == sentinel for v, sentinel in zip(values, sentinels)):
            z, i = chunks.get(1, b''), 0
            for j, (sentinel, width) in enumerate(zip(sentinels, widths)):
                if values[j] == sentinel:
                    require(i + width <= len(z), 'Required ZIP64 extra field missing')
                    values[j] = int.from_bytes(z[i:i + width], 'little'); i += width
        size, compressed, where, disk = values
        require(disk == 0 and 0 <= where and where + 30 + compressed <= layout['offset'],
                'Member offsets must precede the central directory')
        p, directory = PurePosixPath(name), name.endswith('/')
        canonical = name.rstrip('/') if directory else name
        kind = stat.S_IFMT(attrs >> 16)
        require(canonical and not p.is_absolute() and str(p) == canonical
                and '\\' not in name and not any(ord(a) < 32 or ord(a) == 127 for a in name)
                and all(a not in ('', '.', '..') for a in canonical.split('/'))
                and canonical not in names, 'Unsafe or duplicate original member name')
        require(kind in ((0, stat.S_IFDIR) if directory else (0, stat.S_IFREG))
                and not flags & (1 | 64) and compression in (0, 8)
                and (not directory or size == 0), 'Regular unencrypted member/directory metadata required')
        names.add(canonical)
        result.append(dict(name=name, directory=directory, bytes=size, compressed_bytes=compressed,
                           local_header_offset=where, compression=compression, flags=flags,
                           external_attributes=attrs, crc32=f'{crc32:08x}'))
        pos = stop
    require(pos == len(raw), 'Trailing or undeclared central-directory bytes')
    # Prevent a file path from serving as a directory ancestor, without payloads.
    files = {r['name'] for r in result if not r['directory']}
    require(all(not any(str(parent) in files for parent in PurePosixPath(r['name']).parents)
                for r in result), 'File/directory ancestor collision')
    return result
