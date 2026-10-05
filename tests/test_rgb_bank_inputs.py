"""Authored tiny JPEGs only; metadata/decode has no model or network calls."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

import mediapipe_cpu_runtime_verify as rt
from world_reward import rgb_bank_inputs as p


def seal(path, raw):
    path.write_bytes(raw); path.chmod(0o400)
    return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def fixture(tmp_path):
    data = tmp_path/'inputs'; data.mkdir(); rows = []
    for slot in (0, 95):
        path = data/f'image_{slot:06d}.jpg'
        Image.fromarray(np.full((7, 9, 3), slot, dtype=np.uint8)).save(path, format='JPEG')
        path.chmod(0o400)
        rows.append(dict(image_id=f'{slot+1:032x}', file=path.name, width=9, height=7,
                         **rt.identity(path, p.MAX_IMAGE_BYTES)))
    value = dict(schema=p.SCHEMA, images=rows)
    pin = seal(data/'manifest.json', (json.dumps(value)+'\n').encode()); data.chmod(0o500)
    return data, value, pin


def read(data, pin, count=2, **kwargs):
    return p.read_inputs(data, pin, count, identity=rt.identity, pinned=rt.pinned, **kwargs)


def test_original_slot95_holes_and_grid_decode(tmp_path):
    data, value, pin = fixture(tmp_path)
    assert read(data, pin) == value
    assert [p.original_slot(r) for r in value['images']] == [0, 95]
    for row in value['images']:
        rgb = p.decode_rgb(data, row, identity=rt.identity)
        assert rgb.shape == (7, 9, 3) and rgb.dtype == np.uint8
    assert rt.identity(data/'manifest.json') == pin


@pytest.mark.parametrize('fault', ['count','boolcount','slot96','split','md5','reorder','duplicate',
    'sha','bytes','boolwidth','pixels','schema','extra','writable','symlink','unknownid'])
def test_private_invalid_or_changed_input_fails_before_decode(tmp_path, fault):
    data, value, pin = fixture(tmp_path); count = 2
    data.chmod(0o700)
    if fault == 'count': count = 1
    elif fault == 'boolcount': count = True
    elif fault == 'extra': seal(data/'references.json', b'not a public field')
    elif fault == 'writable': (data/value['images'][0]['file']).chmod(0o600)
    elif fault == 'symlink':
        leaf = data/value['images'][0]['file']; raw = leaf.read_bytes(); leaf.unlink()
        seal(tmp_path/'outside', raw); leaf.symlink_to(tmp_path/'outside')
    else:
        row = value['images'][0]
        if fault == 'slot96': row['file'] = 'image_000096.jpg'
        elif fault == 'split': row['split'] = 'FIT'
        elif fault == 'md5': row['original_md5'] = 'publisher metadata forbidden here'
        elif fault == 'reorder': value['images'].reverse()
        elif fault == 'duplicate': value['images'][1]['image_id'] = row['image_id']
        elif fault == 'sha': row['sha256'] = '0'*64
        elif fault == 'bytes': row['bytes'] = p.MAX_IMAGE_BYTES+1
        elif fault == 'boolwidth': row['width'] = True
        elif fault == 'pixels': row['width'] = p.MAX_PIXELS
        elif fault == 'schema': value['schema'] = 'private.cohort'
        elif fault == 'unknownid': row['image_id'] = 'rawphoto'
        (data/'manifest.json').chmod(0o600)
        pin = seal(data/'manifest.json', json.dumps(value).encode())
    data.chmod(0o500)
    with pytest.raises((ValueError, FileNotFoundError)): read(data, pin, count)


def test_native_parent_mode_exception_never_relaxes_leaf_bytes(tmp_path):
    data, value, pin = fixture(tmp_path); data.chmod(0o755)
    with pytest.raises(ValueError): read(data, pin)
    assert read(data, pin, readonly_directory=False) == value
    (data/value['images'][0]['file']).chmod(0o600)
    with pytest.raises(ValueError): read(data, pin, readonly_directory=False)


@pytest.mark.parametrize('fault', ['shape', 'grayscale', 'png', 'sha'])
def test_no_image_conversion_or_grid_repair(tmp_path, fault):
    data, value, _ = fixture(tmp_path); row = dict(value['images'][0]); path = data/row['file']
    if fault == 'shape': row['height'] = 8
    elif fault == 'sha': row['sha256'] = '0'*64
    else:
        path.chmod(0o600)
        Image.new('L' if fault == 'grayscale' else 'RGB', (9, 7)).save(path, format='JPEG' if fault == 'grayscale' else 'PNG')
        path.chmod(0o400); row.update(rt.identity(path))
    with pytest.raises(ValueError): p.decode_rgb(data, row, identity=rt.identity)


def test_metadata_module_has_no_heavy_import_or_global_data_path():
    import ast
    tree = ast.parse(Path(p.__file__).read_bytes())
    imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert all(not isinstance(n, ast.ImportFrom) or n.module in ('pathlib',) for n in imports)
    assert all(not isinstance(n, ast.Import) or all(a.name == 're' for a in n.names) for n in imports)
    assert not hasattr(p, 'DATA')
