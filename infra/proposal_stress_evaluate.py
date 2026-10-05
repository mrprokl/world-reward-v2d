"""Saved-only fixed visible-mask recall, after all native banks are frozen."""
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path[:0] = [str(Path(__file__).resolve().parent),str(Path(__file__).resolve().parents[1]/'src')]
import mediapipe_cpu_runtime_verify as rt
from world_reward.proposal_recall import evaluate_visible_proposals, aggregate_visible_recall, validate_raster_truth

ROOT = rt.ROOT
ENTRY = 'run_proposal_stress_evaluate'
HELPERS = ('infra/proposal_stress_evaluate.py','infra/run_proposal_stress_evaluate.sh',
    'infra/mediapipe_cpu_runtime_verify.py','src/world_reward/proposal_recall.py')


def evaluate(source, manifest, native, render, banks, truths, inventories):
    """Caller authenticates all banks before any private reference is decoded."""
    results = []; groups = {'DEV':[],'reserved':[]}
    for rgb,row,original in zip(manifest['images'],native['banks'],render['images']):
        rt.require(rgb['image_id'] == row['image_id'] == original['image_id'], 'Exact original ordered image identity required')
        z = banks[row['image_id']]; t = truths[row['image_id']]; inventory = inventories[row['image_id']]
        rt.require(inventory['image_id'] == row['image_id'] and inventory['split'] in groups, 'Original private split required')
        validate_raster_truth(t['visible_entity_masks'],t['visible_entity_ids'],t['visible_face_indices'],t['face_entity_ids'])
        kinds = tuple(e['kind'] for e in inventory['entities'])
        result = evaluate_visible_proposals(z['packed_masks'],z['image_size'],t['visible_entity_masks'],kinds,
            native_areas=z['mask_area'],generate_seconds=row['native_generate_seconds'])
        import numpy as np
        rt.require(len(z['native_mask_indices']) == row['native_masks']
            and np.array_equal(z['native_mask_indices'],np.arange(row['native_masks'],dtype=np.int64)), 'Every native mask in original order required')
        results.append(result); groups[inventory['split']].append(result)
    rt.require(len(results) == 32 and all(len(v) == 16 for v in groups.values()), 'All sixteen full two-frame scenes required')
    return dict(schema='world_reward.proposal_stress_recall.v1',status='pass',source_binding=source,
        fixed_method='native_SAM2.1_HieraL_AMG_author_defaults',all_native_banks_frozen_before_truth=True,
        all=aggregate_visible_recall(results),splits={k:aggregate_visible_recall(v) for k,v in groups.items()},
        reference_scope='private_authored_visible_first_surface_only',quality_verified=False,
        real_transfer_verified=False,ownership_verified=False,adoption=False)


def main():
    import numpy as np
    rt.require(sys.platform == 'linux' and os.geteuid() == 0
        and {x.name for x in Path('/sys/class/net').iterdir()} == {'lo'}, 'Offline Azure CPU-only saved evaluation required')
    code = Path(os.environ['WR_CODE']);revision = os.environ['WR_CODE_REVISION']
    source = rt.source(ROOT,code,revision,ENTRY,HELPERS)
    region = ROOT/'results/proposal-stress-regions-v2'; scene = ROOT/'validation/proposal_stress_v1'
    out = ROOT/'results/proposal-stress-evaluation-v1/report.json'
    rt.require(not out.exists(), 'Fresh once-only evaluation artifact required')
    native = rt.strict((region/'native.json').read_bytes());host = rt.strict((region/'host.json').read_bytes())
    rt.require(native['status'] == host['status'] == 'pass' and native['amg_calls'] == 32
        and native['native_model_loads'] == 1 and native['source_rgb_model_rehashed_after'] is True
        and host['owned_cleanup_verified'] is True, 'Complete qualified frozen native bank required before references')
    rt.require(rt.identity(region/'native.json') == host['native_report_identity'], 'Original native receipt differs')
    frozen = {region/'native.json':rt.identity(region/'native.json'),region/'host.json':rt.identity(region/'host.json')}
    banks = {}
    for row in native['banks']:
        path = region/row['file'];pin = {k:row[k] for k in ('bytes','sha256')}
        rt.require(rt.identity(path,1<<30) == pin, 'Complete prediction bank changed before private reference access')
        frozen[path] = pin
        with np.load(path,allow_pickle=False) as z: banks[row['image_id']] = {k:z[k] for k in z.files}
    rt.require(len(banks) == 32, 'Distinct complete 32 banks required')
    # All prediction bytes are authenticated above; only now read private references.
    manifest = rt.strict((scene/'inputs/manifest.json').read_bytes())
    rt.require(rt.identity(scene/'inputs/manifest.json') == dict(bytes=6494,
        sha256='c8faacc4487d2e3f4172c666334465321fd6bbc16a4660687629a604942834ce'), 'Original authored public manifest required')
    renderpath=scene/'eval_private/report.json';renderpin=dict(bytes=22773,
        sha256='aac632ee57d97266477042c7741e373e11740f98aba65d89daacba3e632b5549')
    render=rt.pinned(renderpath,renderpin);frozen[renderpath]=renderpin
    truths={};inventories={}
    for row in render['images']:
        iid=row['image_id'];path=scene/'eval_private'/(iid+'.npz');meta=path.with_suffix('.json')
        for p,pin in ((path,row['truth_identity']),(meta,row['inventory_identity'])):
            rt.require(rt.identity(p,100<<20)==pin,'Original authored reference changed');frozen[p]=pin
        with np.load(path,allow_pickle=False)as z: truths[iid]={k:z[k]for k in ('visible_entity_masks','visible_entity_ids','visible_face_indices','face_entity_ids')}
        inventories[iid]=rt.strict(meta.read_bytes())
    report=evaluate(source,manifest,native,render,banks,truths,inventories)
    for path,pin in frozen.items(): rt.require(rt.identity(path,1<<30)==pin,'Frozen predictions/references changed after CPU scoring')
    rt.require(rt.source(ROOT,code,revision,ENTRY,HELPERS)==source,'Original evaluator source changed')
    report.update(producer_revision=revision,native_report=frozen[region/'native.json'],
        authored_render_report=renderpin,source_predictions_references_rehashed_after=True)
    raw=(json.dumps(report,sort_keys=True,allow_nan=False)+'\n').encode();rt.write(out,raw,0o400)
    print(json.dumps(dict(status='pass',report=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()),
        object_macro_recall_at_half=report['all']['recalls']['0.5']['object']['macro_image_recall'],quality_verified=False)))


if __name__=='__main__': main()
