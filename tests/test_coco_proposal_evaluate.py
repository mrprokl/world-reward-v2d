"""Private-loader spies prove capacity/DEV gates precede reserved labels."""
import copy
from pathlib import Path

import numpy as np
import pytest

import coco_proposal_evaluate as d


def fixture():
    records = []; maps = []; statuses = []; banks = {}; rows = {}; labels = {}; calls = []
    boxes = [[0, 0, 2, 2], [4, 0, 2, 2], [0, 4, 2, 2], [4, 4, 2, 2]]
    for i in range(32):
        iid = f'{i:032x}'
        record = dict(slot=i, split='DEV' if i < 16 else 'RESERVED', image_id=i,
                      image=dict(height=8, width=8))
        records.append(record); maps.append(dict(slot=i, public_image_id=iid))
        statuses.append(dict(slot=i, status='acquired'))
        banks[iid] = dict(native_bbox_xywh=np.array(boxes, np.float64), image_size=np.array([8, 8], np.int64),
                          native_mask_indices=np.arange(4, dtype=np.int64))
        rows[iid] = dict(native_masks=4, native_generate_seconds=.01)
        labels[i] = dict(annotations=[dict(id=10*i+j, image_id=i, iscrowd=0, area=4,
                            category_id=1 if j < 2 else 2, bbox=b) for j, b in enumerate(boxes)])
    cohort = dict(records=records, categories=[dict(id=1, name='person'), dict(id=2, name='cup')])
    acq = dict(public_mappings=maps, records=statuses)
    def load(record): calls.append(record['slot']); return labels[record['slot']]
    return [cohort, acq, banks, rows, load, d.configuration(Path(__file__).resolve().parents[1])], labels, calls


def missing(args, slots):
    for s in slots:
        iid = f'{s:032x}'; args[1]['records'][s]['status'] = 'missing'
        args[2].pop(iid); args[3].pop(iid)
    args[1]['public_mappings'] = [m for m in args[1]['public_mappings'] if m['slot'] not in slots]


def test_all_native_full_slots_qualified_but_not_ownership():
    args, _, calls = fixture(); r = d.evaluate(*args)
    assert calls == list(range(32)) and r['reserved_references_opened']
    assert r['decision'] == 'REAL_BBOX_PROPOSAL_CAPACITY_QUALIFIED_NOT_OWNERSHIP'
    assert all(s['recalls']['0.5']['all']['macro_fixed_slot_recall'] == 1 for s in r['splits'].values())
    assert not r['quality_verified'] and not r['adoption']


def test_missing_images_stay_in_denominator():
    args, _, calls = fixture(); missing(args, {0, 1, 16, 17})
    r = d.evaluate(*args)
    assert all(s['recalls']['0.5']['all']['macro_fixed_slot_recall'] == 14/16 for s in r['splits'].values())
    assert len(calls) == 28


def test_capacity_closed_without_any_reference_access():
    args, _, calls = fixture(); missing(args, set(range(5)))
    r = d.evaluate(*args)
    assert r['decision'] == 'INCONCLUSIVE_CLOSED_CAPACITY_NO_REFERENCE_ACCESS'
    assert not calls and not r['reserved_references_opened'] and not r['splits']


def test_dev_reject_does_not_open_reserved():
    args, _, calls = fixture()
    for i in range(16): args[2][f'{i:032x}']['native_bbox_xywh'][:, :2] += 100
    r = d.evaluate(*args)
    assert calls == list(range(16)) and not r['reserved_references_opened']
    assert r['decision'] == 'CLOSED_DEVELOPMENT_RECALL_FAILURE_RESERVED_UNOPENED'


def test_reserved_reject_no_rescue_or_fabricated_pass():
    args, _, calls = fixture()
    for i in range(16, 32): args[2][f'{i:032x}']['native_bbox_xywh'][:, :2] += 100
    r = d.evaluate(*args)
    assert calls == list(range(32)) and r['decision'] == 'CLOSED_RESERVED_RECALL_FAILURE'


@pytest.mark.parametrize('bad', ['id', 'crowd', 'area', 'bbox', 'category', 'duplicate', 'insufficient'])
def test_reference_parser_keeps_native_ids_and_rejects_invalid_truth(bad):
    args, labels, _ = fixture(); rows = labels[0]['annotations']
    if bad == 'id': rows[0]['image_id'] = 999
    elif bad == 'crowd': rows[0]['iscrowd'] = 1
    elif bad == 'area': rows[0]['area'] = -1
    elif bad == 'bbox': rows[0]['bbox'] = [1, 2]
    elif bad == 'category': rows[0]['category_id'] = 999
    elif bad == 'duplicate': rows.append(copy.deepcopy(rows[0]))
    else: rows.pop()
    with pytest.raises(ValueError): d.evaluate(*args)


@pytest.mark.parametrize('bad', ['order', 'image', 'missing_bank', 'slot', 'split', 'fake_rgb', 'catalog'])
def test_saved_bank_or_slot_mutations_rejected(bad):
    args, _, _ = fixture(); iid = '0'*32
    if bad == 'order': args[2][iid]['native_mask_indices'] = np.array([1, 0, 2, 3], np.int64)
    elif bad == 'image': args[2][iid]['image_size'][0] = 99
    elif bad == 'missing_bank': args[2].pop(iid)
    elif bad == 'slot': args[1]['records'][0]['slot'] = 99
    elif bad == 'split': args[0]['records'][0]['split'] = 'TEST'
    elif bad == 'fake_rgb': args[1]['records'][0]['status'] = 'missing'
    else: args[0]['categories'].append(dict(id=1, name='person'))
    with pytest.raises(ValueError): d.evaluate(*args)
