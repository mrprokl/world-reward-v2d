"""Azure-only, new-record metadata feasibility; not predictions or evaluation.

No RGB/model/network use. Old cohorts are exclusion ID lists only. All original
annotation classes/boxes remain unchanged; geometry gates select records, never
inference candidates. A known holds pair is not an exclusive task target.
"""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mediapipe_cpu_runtime_verify as rt

ROOT = Path('/srv/scenesmith/world-reward')
ENTRY = 'run_openimages_joint_pair_census'
CONFIG = 'configs/openimages_joint_pair_census_v1.json'
HELPERS = ('infra/openimages_joint_pair_census.py', 'infra/run_openimages_joint_pair_census.sh',
           'infra/mediapipe_cpu_runtime_verify.py', CONFIG)


def box(row, suffix=''):
    values = tuple(float(row[k+suffix]) for k in ('XMin', 'YMin', 'XMax', 'YMax'))
    rt.require(all(math.isfinite(x) and 0 <= x <= 1 for x in values)
               and values[2] > values[0] and values[3] > values[1], 'Original finite normalized box required')
    return values


def intersection(a, b):
    return max(0., min(a[2], b[2])-max(a[0], b[0])) * max(0., min(a[3], b[3])-max(a[1], b[1]))


def iou(a, b):
    overlap = intersection(a, b)
    return overlap / ((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-overlap)


def countable(row):
    # Unknown flags do not become zero. Occlusion/truncation do not censor.
    for key in ('IsGroupOf', 'IsDepiction', 'IsInside'):
        rt.require(row[key] in ('-1', '0', '1'), 'Original annotation flag required')
    return all(row[key] == '0' for key in ('IsGroupOf', 'IsDepiction', 'IsInside'))


def separated_pair(boxes):
    return any(intersection(a, b) == 0 for i, a in enumerate(boxes) for b in boxes[i+1:])


def image_census(bboxes, relations, human, body_parts):
    """Return metadata counts; no label-dependent detector/hand assignment."""
    inventory = [(row['LabelName'], box(row), countable(row)) for row in bboxes]
    humans = [b for cls, b, ok in inventory if ok and cls in human]
    objects = [b for cls, b, ok in inventory if ok and cls not in human | body_parts]
    unique = set()
    for row in relations:
        if row['RelationshipLabel'] == 'holds' and row['LabelName1'] in human and row['LabelName2'] not in human | body_parts:
            unique.add((row['LabelName1'], box(row, '1'), row['LabelName2'], box(row, '2')))
    def bind(cls, b):
        matches = [i for i, (c, bb, _) in enumerate(inventory) if c == cls and iou(b, bb) >= .5]
        # Do not privilege an exact box over a second/group overlap.
        return matches[0] if len(matches) == 1 and inventory[matches[0]][2] else None
    resolved = set()
    losses = 0
    for a, ba, b, bb in unique:
        pi, oi = bind(a, ba), bind(b, bb)
        if pi is None or oi is None:
            losses += 1
        else:
            resolved.add((pi, oi))
    return dict(bbox_rows=len(inventory), countable_person_boxes=len(humans), countable_object_boxes=len(objects),
        two_separated_person_boxes=separated_pair(humans), two_separated_object_boxes=separated_pair(objects),
        positive_pairs=len(unique), resolved_positive_pairs=len(resolved), unscorable_positive_pairs=losses,
        metadata_eligible=bool(separated_pair(humans) and separated_pair(objects) and resolved))


def csv_rows(path):
    with path.open(newline='') as stream:
        yield from csv.DictReader(stream)


def run():
    started = time.monotonic()
    rt.require(os.geteuid() == 0 and os.uname().sysname == 'Linux'
        and os.uname().nodename == 'world-reward-ncc-h100-02', 'Exact Azure CPU host required')
    revision = os.environ['WR_CODE_REVISION']; code = Path(os.environ['WR_CODE'])
    source = rt.source(ROOT, code, revision, ENTRY, HELPERS)
    protocol = rt.strict((code/CONFIG).read_bytes())
    rt.require(protocol['schema'] == 'world_reward.openimages_joint_pair_census.v1'
        and protocol['selection'] == 'none_feasibility_only_before_RGB_or_predictions', 'Frozen metadata-only scope required')
    files = {Path(name): pin for name, pin in protocol['files'].items()}
    def rehash():
        return {str(p): rt.identity(p, 100 << 20, readonly=False) for p in files}
    rt.require(rehash() == protocol['files'], 'Original publisher CSV/census pins differ')
    output = ROOT/protocol['output']; rt.canonical(output)
    rt.require(not output.exists(), 'Fresh census output required'); output.mkdir(mode=0o700)
    report = dict(schema=protocol['schema'], status='fail', producer_revision=revision, source_binding=source,
        frozen_inputs=protocol['files'], selection_performed=False, challenge_inputs_used=False, rgb_read=False,
        models_loaded=False, gpu_used=False, network_used=False, local_heavy_transfer=False,
        individual_image_license_verified=False, training_overlap_verified=False, challenge_overlap_verified=False,
        ownership_verified=False, task_target_verified=False, accuracy_verified=False, adopted=False)
    try:
        excluded = set()
        for suffix, expected in (('openimages_holds_census_v1/census_v2.json', 16),
                                 ('openimages_holds_census_fresh128_v1/census.json', 128)):
            p = next(p for p in files if str(p).endswith(suffix))
            ids = rt.strict(p.read_bytes())['selected_ids']
            rt.require(len(ids) == len(set(ids)) == expected and not excluded.intersection(ids), 'Exact old disjoint cohort IDs required')
            excluded.update(ids)
        rt.require(len(excluded) == protocol['rules']['old_ids_excluded'] == 144, 'All old IDs excluded')
        human, parts = set(protocol['human_classes']), set(protocol['body_part_classes'])
        ref = ROOT/'results/openimages-holds-evaluate-v1'
        triplets = {(r['LabelName1'], r['LabelName2']) for r in csv_rows(ref/'triplets.csv')
            if r['RelationshipLabel'] == 'holds' and r['LabelName1'] in human and r['LabelName2'] not in human | parts}
        relations = {}
        for row in csv_rows(ref/'relations.csv'):
            if row['ImageID'] not in excluded and row['RelationshipLabel'] == 'holds' and (row['LabelName1'], row['LabelName2']) in triplets:
                relations.setdefault(row['ImageID'], []).append(row)
        bboxes = {iid: [] for iid in relations}
        for row in csv_rows(ref/'boxes.csv'):
            if row['ImageID'] in bboxes:
                bboxes[row['ImageID']].append(row)
        metadata_path = next(p for p in files if str(p).endswith('/metadata.csv'))
        metadata = {r['ImageID']: r for r in csv_rows(metadata_path) if r['ImageID'] in relations}
        rt.require(set(metadata) == set(relations), 'Publisher metadata missing an untouched relation image')
        records = []
        for iid in sorted(relations):
            rt.require(time.monotonic()-started < protocol['budget_seconds'], 'Bounded metadata census required')
            record = image_census(bboxes[iid], relations[iid], human, parts)
            record.update(image_id=iid, publisher_rotation_zero=metadata[iid]['Rotation'] == '0.0')
            record['eligible'] = record['metadata_eligible'] and record['publisher_rotation_zero']
            # Original public rights/source metadata, never RGB or predictions.
            record['publisher_metadata'] = metadata[iid]
            records.append(record)
        eligible = [r for r in records if r['eligible']]
        authors = {r['publisher_metadata']['AuthorProfileURL'] for r in eligible if r['publisher_metadata']['AuthorProfileURL']}
        counts = dict(new_relation_images=len(records), old_ids_excluded=len(excluded),
            two_separated_person_images=sum(r['two_separated_person_boxes'] for r in records),
            two_separated_object_images=sum(r['two_separated_object_boxes'] for r in records),
            metadata_eligible_images=sum(r['metadata_eligible'] for r in records), eligible_images=len(eligible),
            source_author_groups=len(authors), unscorable_positive_pairs=sum(r['unscorable_positive_pairs'] for r in records))
        enough = len(eligible) >= protocol['minimum_eligible_images'] and len(authors) >= protocol['minimum_source_authors']
        report.update(status='pass', stage='new_joint_pair_metadata_feasibility', counts=counts,
            decision='FEASIBLE_PENDING_INDIVIDUAL_RIGHTS_AND_FROZEN_NEW_EXPERIMENT' if enough else 'INCONCLUSIVE_CLOSED_NO_RGB',
            record_gate='conservative_nonintersecting_box_lower_bound_not_physical_identity_proof',
            body_part_class_count=len(parts), triplet_count=len(triplets), records=records)
    except BaseException as exc:
        report.update(status='fail', error_type=type(exc).__name__)
        raise
    finally:
        try:
            rt.require(rehash() == protocol['files'] and rt.source(ROOT, code, revision, ENTRY, HELPERS) == source,
                       'Source/input bytes changed after census')
            report['source_and_inputs_rehashed_after'] = True
        except BaseException:
            report.update(status='fail', source_and_inputs_rehashed_after=False)
            raise
        finally:
            report['elapsed_seconds'] = time.monotonic()-started
            raw = (json.dumps(report, sort_keys=True, allow_nan=False)+'\n').encode()
            rt.write(output/'report.json', raw, 0o444); output.chmod(0o555)
            print(json.dumps({k: report[k] for k in ('status', 'counts', 'decision', 'elapsed_seconds') if k in report}, sort_keys=True))
            print(json.dumps(dict(report=dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))))
    return report


if __name__ == '__main__':
    rt.require(len(sys.argv) == 1, 'No per-record parameters allowed')
    run()
