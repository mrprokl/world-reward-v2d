"""Manufactured boxes only, no local challenge rendering."""
import pytest
from full4d_focus import viewport, save


def test_crop_preserves_both_disagreeing_automatic_observations():
    boxes = ([200,220,240,240], [420,300,440,330])
    x0,y0,x1,y1 = viewport(boxes,640,480)
    assert x0 <= 200 and y0 <= 220 and x1 >= 440 and y1 >= 330


def test_missing_views_remain_full_context_not_invented_boxes():
    assert viewport((None,None),640,480) == (0,0,640,480)


def test_full_extent_not_cropped_away():
    assert viewport(([0,0,640,480],),640,480) == (0,0,640,480)


def test_box_on_border_is_not_dropped():
    x0,y0,x1,y1 = viewport(([620,470,640,480],),640,480)
    assert x0 <= 620 and y0 <= 470 and x1 == 640 and y1 == 480


@pytest.mark.parametrize('box', ([2,3,1,4], [-1,0,3,4], [1,2,float('nan'),4]))
def test_reject_invalid_automatic_boxes(box):
    with pytest.raises(ValueError):
        viewport((box,),640,480)


def test_host_metadata_is_stdlib_only(tmp_path, monkeypatch):
    import builtins
    real_import = builtins.__import__
    def restricted(name, *args, **kwargs):
        if name == 'numpy':
            raise AssertionError('Host must not require scientific runtime')
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', restricted)
    path = tmp_path/'receipt.json'
    save(path, {'predictions_modified': False, 'bytes': 10})
    assert path.read_text() == '{"bytes": 10, "predictions_modified": false}\n'
    assert not path.stat().st_mode & 0o222
