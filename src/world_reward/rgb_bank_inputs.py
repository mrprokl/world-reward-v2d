"""Public six-field original RGB inputs shared by label-free native callers.

SHA authenticates bytes, not publisher MD5/rights or prediction quality. Those
remain outside the native boundary. Original slots survive missing acquisition.
Only decode imports Pillow/NumPy; metadata authentication is stdlib-only.
"""
from pathlib import Path
import re

KEYS = frozenset(('image_id', 'file', 'bytes', 'sha256', 'width', 'height'))
SCHEMA = 'world_reward.rgb_proposal_inputs.v1'
MAX_IMAGE_BYTES = 16 << 20
MAX_PIXELS = 16 << 20


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def original_slot(row, maximum_slots=96):
    name = row.get('file')
    _require(type(maximum_slots) is int and 0 < maximum_slots <= 1000000
             and type(name) is str and re.fullmatch(r'image_[0-9]{6}\.jpg', name),
             'Original bounded JPEG slot basename required')
    slot = int(name[6:12])
    _require(slot < maximum_slots, 'JPEG slot exceeds frozen population')
    return slot


def read_inputs(directory, pin, count, *, identity, pinned, maximum_slots=96,
                readonly_directory=True):
    """Authenticate complete acquired projection, never private slot metadata."""
    directory = Path(directory)
    _require(directory.is_absolute() and directory.resolve() == directory
             and not any(p.is_symlink() for p in (directory, *directory.parents))
             and directory.is_dir() and (not readonly_directory or not directory.stat().st_mode & 0o222),
             'Canonical readonly public directory required')
    _require(type(maximum_slots) is int and 0 < maximum_slots <= 1000000
             and type(readonly_directory) is bool and type(count) is int and 0 < count <= maximum_slots,
             'Explicit acquired image count required')
    value = pinned(directory/'manifest.json', pin, 1 << 20)
    _require(type(value) is dict and set(value) == {'schema', 'images'}
             and value['schema'] == SCHEMA and type(value['images']) is list
             and len(value['images']) == count, 'Complete public RGB projection required')
    ids, names, previous, total = set(), set(), -1, 0
    for row in value['images']:
        _require(type(row) is dict and set(row) == KEYS, 'Only six public RGB fields allowed')
        slot = original_slot(row, maximum_slots)
        _require(type(row['image_id']) is str and re.fullmatch('[0-9a-f]{32}', row['image_id'])
                 and slot > previous and row['image_id'] not in ids and row['file'] not in names
                 and type(row['bytes']) is int and 0 < row['bytes'] <= MAX_IMAGE_BYTES
                 and type(row['sha256']) is str and re.fullmatch('[0-9a-f]{64}', row['sha256'])
                 and all(type(row[k]) is int and row[k] > 0 for k in ('width', 'height'))
                 and row['width']*row['height'] <= MAX_PIXELS,
                 'Ordered unique original image identities/grid required')
        _require(identity(directory/row['file'], MAX_IMAGE_BYTES)
                 == {k: row[k] for k in ('bytes', 'sha256')}, 'Original public JPEG changed')
        ids.add(row['image_id']); names.add(row['file']); previous = slot; total += row['bytes']
    _require(total <= maximum_slots*MAX_IMAGE_BYTES
             and {p.name for p in directory.iterdir()} == {'manifest.json'} | names,
             'Exclusive complete public directory/byte budget required')
    return value


def decode_rgb(directory, row, *, identity):
    """Original RGB JPEG/grid only: no EXIF rotation, conversion, crop or repair."""
    import numpy as np
    from PIL import Image
    path = Path(directory)/row['file']
    _require(identity(path, MAX_IMAGE_BYTES) == {k: row[k] for k in ('bytes', 'sha256')},
             'Original JPEG changed before decode')
    with Image.open(path) as image:
        _require(image.format == 'JPEG' and image.mode == 'RGB'
                 and not getattr(image, 'is_animated', False)
                 and image.size == (row['width'], row['height']), 'Original JPEG RGB/grid required')
        rgb = np.array(image, copy=True)
    _require(identity(path, MAX_IMAGE_BYTES) == {k: row[k] for k in ('bytes', 'sha256')},
             'Original JPEG changed during decode')
    return rgb
