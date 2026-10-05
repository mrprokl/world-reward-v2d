"""Once-only saved native box recall; reserved references are opened lazily."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1]/'src')]
import mediapipe_cpu_runtime_verify as rt
from world_reward.proposal_bbox_recall import evaluate_bbox_proposals, aggregate_fixed_slot_bbox_recall

ROOT = rt.ROOT
DATA = Path('/srv/world-reward-data/coco_proposal_v1')
ENTRY = 'run_coco_proposal_evaluate'
CONFIG = 'configs/coco_proposal_evaluation_v1.json'
HELPERS = ('infra/coco_proposal_evaluate.py', 'infra/run_coco_proposal_evaluate.sh', CONFIG,
           'infra/mediapipe_cpu_runtime_verify.py', 'src/world_reward/proposal_bbox_recall.py',
           'src/world_reward/proposal_recall.py')
REGIONS = ROOT/'results/coco-proposal-regions-v1'
OUT = ROOT/'results/coco-proposal-evaluation-v1/report.json'


def configuration(code):
    p = rt.strict((code/CONFIG).read_bytes())
    rt.require(p['schema'] == 'world_reward.coco_proposal_evaluation.v1' and p['slots'] == 32
        and p['development_slots'] == p['reserved_slots'] == 16
        and p['minimum_acquired_development_images_before_reference_access'] == 12
        and p['primary_bbox_iou_threshold'] == .5 and p['diagnostic_bbox_iou_thresholds'] == [.25, .75]
        and p['development_gate_macro_all_fixed_slot_bbox_recall_at_half'] == .6
        and p['reserved_gate_macro_all_fixed_slot_bbox_recall_at_half'] == .6
        and p['missing_slot_recall'] == 0. and p['adoption'] is False,
        'Prospectively frozen unchanged bbox protocol required')
    return p


def reference_boxes(value, record, categories):
    import numpy as np
    rows = value['annotations']; ids = []; kinds = []; boxes = []
    rt.require(type(rows) is list and len(rows) > 0, 'All original noncrowd reference instances required')
    for r in rows:
        rt.require(type(r['id']) is int and r['id'] not in ids and r['image_id'] == record['image_id']
            and type(r['iscrowd']) is int and r['iscrowd'] == 0
            and type(r['area']) in (int, float) and r['area'] > 0
            and type(r['category_id']) is int and r['category_id'] in categories
            and type(r['bbox']) is list and len(r['bbox']) == 4
            and all(type(x) in (int, float) for x in r['bbox']), 'Exact native instance IDs/grid/boxes required')
        ids.append(r['id']); boxes.append(r['bbox'])
        kinds.append('human' if categories[r['category_id']] == 'person' else 'object')
    rt.require(kinds.count('human') >= 2 and kinds.count('object') >= 2, 'Original frozen crowd eligibility required')
    return np.asarray(boxes, np.float64), tuple(kinds), tuple(str(x) for x in ids)


def evaluate(cohort, acquisition, banks, native_rows, load_reference, p):
    """All bank bytes are already frozen; loader alone opens private labels."""
    import numpy as np
    records = cohort['records']; maps = acquisition['public_mappings']
    rt.require(len(records) == 32 and [r['slot'] for r in records] == list(range(32))
        and [r['split'] for r in records] == ['DEV']*16+['RESERVED']*16, 'All original ordered frozen slots required')
    mapping = {r['slot']:r for r in maps}
    rt.require(len(mapping) == len(maps) and set(mapping) <= set(range(32))
        and len({r['public_image_id'] for r in maps}) == len(maps)
        and set(banks) == set(native_rows) == {r['public_image_id'] for r in maps}, 'Every acquired native bank required')
    statuses = acquisition['records']
    rt.require(len(statuses) == 32 and [r['slot'] for r in statuses] == list(range(32))
        and all((r['status'] == 'acquired') == (r['slot'] in mapping) for r in statuses), 'Missing slots cannot disappear')
    categories = {c['id']:c['name'] for c in cohort['categories']}
    rt.require(len(categories) == len(cohort['categories']) and list(categories.values()).count('person') == 1,
        'Original native category catalog required')
    report = dict(schema='world_reward.coco_proposal_recall.v1', status='pass', reserved_references_opened=False,
        all_native_banks_frozen_before_truth=True, ownership_verified=False, quality_verified=False,
        training_overlap_verified=False, author_disjointness_verified=False, adoption=False, splits={})
    if sum(s < 16 for s in mapping) < p['minimum_acquired_development_images_before_reference_access']:
        report['decision'] = 'INCONCLUSIVE_CLOSED_CAPACITY_NO_REFERENCE_ACCESS'; return report
    for split, chosen in (('DEV', records[:16]), ('RESERVED', records[16:])):
        results = []
        for record in chosen:
            if record['slot'] not in mapping: results.append(None); continue
            m = mapping[record['slot']]; iid = m['public_image_id']; z = banks[iid]; row = native_rows[iid]
            n = row['native_masks']; image = record['image']
            rt.require(z['image_size'].dtype == np.int64 and np.array_equal(z['image_size'], [image['height'], image['width']])
                and z['native_mask_indices'].dtype == np.int64
                and np.array_equal(z['native_mask_indices'], np.arange(n, dtype=np.int64))
                and z['native_bbox_xywh'].shape == (n, 4), 'Full native original-grid XYWH bank in original slot order required')
            boxes, kinds, ids = reference_boxes(load_reference(record), record, categories)
            results.append(evaluate_bbox_proposals(z['native_bbox_xywh'], boxes, kinds, entity_ids=ids,
                generate_seconds=row['native_generate_seconds']))
        report['splits'][split] = aggregate_fixed_slot_bbox_recall(results)
        if split == 'DEV':
            if report['splits'][split]['recalls']['0.5']['all']['macro_fixed_slot_recall'] < p['development_gate_macro_all_fixed_slot_bbox_recall_at_half']:
                report['decision'] = 'CLOSED_DEVELOPMENT_RECALL_FAILURE_RESERVED_UNOPENED'; return report
            report['reserved_references_opened'] = True
    accepted = report['splits']['RESERVED']['recalls']['0.5']['all']['macro_fixed_slot_recall'] >= p['reserved_gate_macro_all_fixed_slot_bbox_recall_at_half']
    report['decision'] = 'REAL_BBOX_PROPOSAL_CAPACITY_QUALIFIED_NOT_OWNERSHIP' if accepted else 'CLOSED_RESERVED_RECALL_FAILURE'
    return report


def main():
    import numpy as np
    parser = argparse.ArgumentParser(allow_abbrev=False)
    for name in ('census', 'cohort', 'acquisition', 'native', 'host'):
        parser.add_argument('--'+name+'-bytes', type=int, required=True)
        parser.add_argument('--'+name+'-sha256', required=True)
    a = parser.parse_args()
    rt.require(sys.platform == 'linux' and os.geteuid() == 0
        and {x.name for x in Path('/sys/class/net').iterdir()} == {'lo'}, 'Offline Azure saved-only CPU evaluation required')
    code = Path(os.environ['WR_CODE']); revision = os.environ['WR_CODE_REVISION']
    source = rt.source(ROOT, code, revision, ENTRY, HELPERS); p = configuration(code)
    rt.require(not OUT.exists(), 'Fresh once-only evaluation required')
    files = {'census':DATA/'metadata/report.json', 'cohort':DATA/'metadata/cohort.json',
             'acquisition':DATA/'eval_private/manifest.json', 'native':REGIONS/'native.json', 'host':REGIONS/'host.json'}
    frozen = {path:dict(bytes=getattr(a, name+'_bytes'), sha256=getattr(a, name+'_sha256')) for name, path in files.items()}
    values = {name:rt.pinned(path, frozen[path], 8 << 20) for name, path in files.items()}
    census, cohort, acquisition, native, host = (values[n] for n in files)
    rt.require(census['status'] == acquisition['status'] == native['status'] == host['status'] == 'pass'
        and census['source_and_inputs_rehashed_after'] is acquisition['source_and_inputs_rehashed_after'] is True
        and native['native_model_loads'] == 1 and native['source_rgb_model_rehashed_after'] is True
        and host['owned_cleanup_verified'] is True
        and host['native_report_identity'] == frozen[files['native']]
        and census['cohort_identity'] == frozen[files['cohort']], 'Qualified complete frozen producer receipts required')
    publicpath = DATA/'inputs/manifest.json'; frozen[publicpath] = acquisition['public_inputs_identity']
    public = rt.pinned(publicpath, frozen[publicpath], 1 << 20)
    rt.require(public['schema'] == 'world_reward.rgb_proposal_inputs.v1'
        and [r['image_id'] for r in native['banks']] == [r['image_id'] for r in public['images']]
        and native['amg_calls'] == native['amg_attempts'] == native['rgb_decodes'] == len(public['images']),
        'Every acquired RGB has its complete native bank')
    banks = {}; native_rows = {}
    for row in native['banks']:
        iid = row['image_id']; path = REGIONS/row['file']
        rt.require(path.name == iid+'.npz' and iid not in banks, 'Distinct original native bank path required')
        frozen[path] = {k:row[k] for k in ('bytes', 'sha256')}
        rt.require(rt.identity(path, 1 << 30) == frozen[path], 'Every prediction byte must freeze before private truth')
        with np.load(path, allow_pickle=False) as z:
            banks[iid] = {k:z[k] for k in ('native_bbox_xywh', 'native_mask_indices', 'image_size')}
        native_rows[iid] = row
    # Hash all references, including reserved, but do not parse their values.
    for record in cohort['records']:
        name = record['reference_file']; path = DATA/'metadata'/name
        rt.require(name == f"reference_{record['slot']:06d}.json", 'Original private per-slot reference filename required')
        frozen[path] = record['reference_identity']
        rt.require(rt.identity(path, 8 << 20) == frozen[path], 'Original private reference bytes changed')
    def load(record):
        path = DATA/'metadata'/record['reference_file']
        return rt.pinned(path, frozen[path], 8 << 20)
    report = evaluate(cohort, acquisition, banks, native_rows, load, p)
    for path, pin in frozen.items(): rt.require(rt.identity(path, 1 << 30) == pin, 'Frozen bytes changed after CPU scoring')
    rt.require(rt.source(ROOT, code, revision, ENTRY, HELPERS) == source, 'Evaluator source changed')
    report.update(producer_revision=revision, source_binding=source,
        inputs={name:frozen[path] for name, path in files.items()}, source_predictions_references_rehashed_after=True)
    raw = (json.dumps(report, sort_keys=True, allow_nan=False)+'\n').encode(); rt.write(OUT, raw, 0o400)
    print(json.dumps(dict(status='pass', decision=report['decision'], report=dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))))


if __name__ == '__main__': main()
