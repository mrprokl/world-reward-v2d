"""Explicit TRAIN/VAL metadata census; caller authenticates inputs and freshness.

No source authentication, publishing, selection, RGB, model or fitting occurs
here. The separate inventory is private identity metadata, never a predictor
input. Original TEST role values are not opened. All consumed streams reach EOF.
"""
from collections import Counter
import os
from pathlib import Path
import zipfile

import vcoco_role_census as c


def census_population(cfg, excluded, *, roles, minimum_photos, checkpoint=lambda: None):
    """Return (scalar census report, private split/image/photo identity inventory).

    `roles` explicitly maps TRAIN/VAL to original authenticated input keys;
    `minimum_photos` is caller-frozen (prospective FIT32/CAL16), not inferred
    from available records. The original cfg, cfg['roles'] and globals stay intact.
    Caller binds all bytes/modes before and after, including historical/pilot
    exclusions; this helper does not certify those exclusions or creator rights.
    """
    c.rt.require(type(roles) is dict and roles == {n:f'data__vcoco__vcoco_{n}.json' for n in ('train','val')},
                 'Explicit original TRAIN/VAL roles required; no TEST roles')
    c.rt.require(type(minimum_photos) is dict and set(minimum_photos) == {'train','val'}
                 and all(type(v) is int and v > 0 for v in minimum_photos.values()), 'Explicit positive photo minimums required')
    c.rt.require(all(cfg[k] == v and type(cfg[k]) is int for k,v in
                 (('minimum_noncrowd_people',2),('minimum_nonperson_objects',2),('minimum_localized_positive_pairs',1))),
                 'Original P2/O2/positive-pair1 predicates required')
    c.rt.require(type(excluded['photos']) in (set,frozenset)
                 and all(type(p) is str for p in excluded['photos']), 'Explicit excluded publisher photo IDs required')
    path = cfg['inputs']['coco_archive']['path']; specs = {r['member']:r for r in cfg['archive']['member_catalogue']}
    with Path(path).open('rb') as handle:
        tail = cfg['archive']['catalogue_tail_identity']; handle.seek(-tail['bytes'],os.SEEK_END)
        c.rt.require(c.pin(handle.read(tail['bytes'])) == tail,'Original catalogue tail changed')
    with zipfile.ZipFile(path) as archive:
        actual = [dict(member=i.filename,compressed_bytes=i.compress_size,expanded_bytes=i.file_size,
            crc32=f'{i.CRC:08x}',external_attr=i.external_attr,compression=i.compress_type,flags=i.flag_bits) for i in archive.infolist()]
        c.rt.require(actual == cfg['archive']['member_catalogue'],'Complete original member catalogue changed')
    split = {name:c.split_ids(Path(cfg['inputs'][key]['path']).read_bytes()) for name,key in cfg['splits'].items()}
    c.rt.require(not split['train']&split['val'] and not split['trainval']&split['test']
        and split['train']|split['val'] == split['trainval'] and split['trainval']|split['test'] == split['all'],'Official split partition differs')
    native = c.rt.pinned(cfg['inputs']['metadata_report']['path'],cfg['inputs']['metadata_report']['pin'])
    pins = {r['member']:dict(bytes=r['expanded_bytes'],sha256=r['expanded_sha256']) for r in native['archive_members']}
    banks, taxonomy, counts, expanded, all_ids, partitions = {},None,Counter(),{},set(),{}
    for partition,member in cfg['archive']['instances_members'].items():
        with c.member_stream(path,member,specs[member],checkpoint) as handle:
            bank,categories,count,ids = c.catalog(handle,partition,excluded['photos'],checkpoint,cfg['max_expanded_bytes'])
        c.rt.require(taxonomy is None or taxonomy == categories,'Original partition taxonomies differ'); taxonomy = categories
        c.rt.require(not all_ids&ids,'Original image ID partitions conflict'); all_ids.update(ids); banks.update(bank); counts.update(count)
        partitions[partition] = ids; expanded[partition] = dict(bytes=handle.size,sha256=handle.digest.hexdigest())
        c.rt.require(expanded[partition] == pins[member],'Original complete member SHA differs')
    c.rt.require(split['all'] <= all_ids,'Original split references absent catalog IDs')
    duplicates = Counter(r['photo_id'] for r in banks.values())
    banks = {iid:r for iid,r in banks.items() if duplicates[r['photo_id']] == 1}
    eligible = (split['train']|split['val'])&set(banks); instances = []; seen = set()
    for partition,member in cfg['archive']['instances_members'].items():
        with c.member_stream(path,member,specs[member],checkpoint) as handle:
            for iid,row in c.projection.iter_filtered_coco_annotations(handle,eligible,check=checkpoint,
                max_bytes=cfg['max_expanded_bytes'],row_bytes=cfg['max_row_bytes']):
                c.rt.require(type(row['id']) is int and row['id'] > 0 and row['id'] not in seen
                    and row['category_id'] in taxonomy and iid in partitions[partition],'Original unique annotation/category/partition ID required')
                seen.add(row['id']); instances.append(row)
        c.rt.require(dict(bytes=handle.size,sha256=handle.digest.hexdigest()) == expanded[partition],'Complete member changed between catalog/annotation passes')
    people, objects = Counter(),Counter()
    for row in instances:
        valid = c.coco.countable(row); counts['crowd_instances'] += row['iscrowd'] == 1
        counts['invalid_or_nonpositive_geometry_instances'] += row['area'] <= 0 or row['bbox'][2] <= 0 or row['bbox'][3] <= 0
        if valid: (people if row['category_id'] == 1 else objects)[row['image_id']] += 1
    outputs = {}; inventory = []; images = [{k:r[k] for k in ('id','width','height')} for iid,r in banks.items() if iid in eligible]
    for name in ('train','val'):
        key = roles[name]; selected = eligible&split[name]
        def open_role():
            row = cfg['inputs'][key]; c.rt.require(c.rt.identity(row['path'],cfg['max_role_bytes']) == row['pin'],'Role source differs before each open')
            return Path(row['path']).open('rb')
        actions,proof = c.projection.project_vcoco_actions(open_role,selected,
            check=checkpoint,max_bytes=cfg['max_role_bytes'],field_bytes=cfg['max_field_bytes'],expected_image_ids=split[name])
        c.rt.require(proof['source_identity'] == cfg['inputs'][key]['pin'],'Original role bytes changed during projection')
        reference = c.parse_vcoco_role_reference(actions,instances,images); checkpoint()
        pairs, agents = Counter(),{}
        for pair in reference.localized_positive_pairs:
            pairs[pair.image_id] += 1; agents.setdefault(pair.image_id,set()).add(pair.agent_annotation_id)
        local = Counter(rights_eligible_images=len(selected),projected_rows=len(reference.rows),localized_positive_pairs=len(reference.localized_positive_pairs))
        for row in reference.rows:
            local['positive_action_rows'] += row.label
            local['missing_positive_role_ids'] += sum(row.label == 1 and i > 0 and rid == 0 for i,rid in enumerate(row.role_object_ids))
            local['unscorable_positive_roles'] += sum(row.label == 1 and i > 0 and rid != 0 and not row.nonperson_pair_eligible[i] for i,rid in enumerate(row.role_object_ids))
        photos = set()
        for iid in sorted(selected):
            if people[iid] >= 2 and objects[iid] >= 2 and pairs[iid] >= 1:
                photos.add(banks[iid]['photo_id']); inventory.append(dict(split=name,image_id=iid,photo_id=banks[iid]['photo_id']))
                local['eligible_images'] += 1; local['eligible_images_two_positive_agents'] += len(agents.get(iid,set())) >= 2
        local['distinct_eligible_photos'] = len(photos); outputs[name] = dict(local,projection=proof)
    counts['duplicate_fresh_photo_images_rejected'] = sum(v for v in duplicates.values() if v > 1)
    report = dict(catalog_counts=dict(counts),splits=outputs,eligibility_inventory_identity=c.pin(c.encode(inventory)),
        eligibility_inventory_rows=len(inventory),expanded_instance_members=expanded,minimum_distinct_photos=dict(minimum_photos),
        capacity_gate_passed=all(outputs[n]['distinct_eligible_photos'] >= minimum_photos[n] for n in ('train','val')))
    return report, inventory
