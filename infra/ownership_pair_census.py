"""New Open Images holds feasibility only: metadata, no cohort or RGB selection."""
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import stat
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mediapipe_cpu_runtime_verify as rt
import openimages_joint_pair_census as census
import openimages_joint_pair_acquire as identities
import coco_proposal_prepare as coco

ROOT = Path('/srv/scenesmith/world-reward')
ENTRY = 'run_ownership_pair_census'
CONFIG = 'configs/ownership_pair_census_v1.json'
HELPERS = ('infra/ownership_pair_census.py', 'infra/run_ownership_pair_census.sh', CONFIG,
           'infra/mediapipe_cpu_runtime_verify.py', 'infra/openimages_joint_pair_census.py',
           'infra/openimages_joint_pair_acquire.py', 'infra/coco_proposal_prepare.py',
           'configs/proposal_external_census_v1.json', 'configs/coco_endpoint_v2.json')


def encode(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False)+'\n').encode()


def pin(raw):
    return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def _metadata_identity(row):
    rt.require(type(row) is dict and re.fullmatch('[0-9a-f]{16}', row['ImageID']), 'Original metadata ID required')
    rt.require(type(row['OriginalURL']) is str and bool(row['OriginalURL']), 'Original source URL required')
    return (identities.profile_identity(row['AuthorProfileURL']),
            identities.excluded_photo_identity(row['OriginalLandingURL']),
            identities.md5_identity(row['OriginalMD5']) if row['OriginalMD5'] else None, row['OriginalURL'])


def _historical_exclusions(values):
    """Exactly six complete metadata ledgers; no reference paths are opened."""
    rt.require(type(values) is list and len(values) == 6, 'All six historical metadata cohorts required')
    out = {k: set() for k in ('ids', 'authors', 'photos', 'md5', 'urls', 'coco_ids')}
    missing_md5 = 0
    for value, count in zip(values[:4], (16, 128, 64, 32)):
        if count in (16, 128):
            rows, ids = value['public_metadata'], value['selected_ids']
            rt.require(len(ids) == len(set(ids)) == len(rows) == count
                       and {r['ImageID'] for r in rows} == set(ids), 'All historical OI slots required')
        else:
            schema = 'world_reward.openimages_joint_pair_cohort.v1' if count == 64 else 'world_reward.proposal_external_cohort.v1'
            slots = value['records']; tail = 'TEST' if count == 64 else 'RESERVED'
            rt.require(value['schema'] == schema and value['no_replacements'] is True
                       and len(slots) == count and [r['slot'] for r in slots] == list(range(count))
                       and [r['split'] for r in slots] == ['DEV']*(count//2)+[tail]*(count//2), 'Frozen historical OI slots required')
            rows = [r['publisher_metadata'] for r in slots]
        for row in rows:
            iid = row['ImageID']; rt.require(iid not in out['ids'], 'Disjoint historical OI IDs required')
            author, photo, digest, url = _metadata_identity(row)  # Unknown OI author fails.
            out['ids'].add(iid); out['authors'].add(author); out['photos'].add(photo); out['urls'].add(url)
            if digest is None: missing_md5 += 1
            else: out['md5'].add(digest)
    for value, count, schema in ((values[4], 32, 'world_reward.coco_proposal_cohort.v1'),
                                 (values[5], 64, 'world_reward.coco_endpoint_cohort.v2')):
        rows = value['records']
        rt.require(value['schema'] == schema and value['freeze_before_rgb'] is True
                   and value['no_replacements'] is True and len(rows) == count
                   and [r['slot'] for r in rows] == list(range(count))
                   and [r['split'] for r in rows] == ['DEV']*(count//2)+['RESERVED']*(count//2), 'Frozen historical COCO slots required')
        seen_photos = set()
        for row in rows:
            iid = row['image_id']; image = row['image']
            rt.require(set(row) == {'slot', 'split', 'image_id', 'image', 'photo_id', 'reference_file', 'reference_identity'}
                       and type(iid) is int and 0 <= iid < 10**12 and image['id'] == iid and image['license'] == 4
                       and iid not in out['coco_ids'] and row['reference_file'] == f"reference_{row['slot']:06d}.json",
                       'Original COCO metadata identity required')
            photo = coco.photo_identity(image['flickr_url'])
            rt.require(photo == row['photo_id'] and photo not in seen_photos, 'Unique native COCO photo identity required')
            seen_photos.add(photo); out['photos'].add(photo); out['coco_ids'].add(iid); out['urls'].add(image['flickr_url'])
            missing_md5 += 1  # Original COCO cohort has no image-byte MD5 or creator identity.
    rt.require(len(out['ids']) == 240 and len(out['coco_ids']) == 96, 'All historical336 slots required')
    counts = dict(inspected_historical_slots=336, excluded_oi_ids=240, excluded_coco_ids=96,
                  excluded_author_profiles=len(out['authors']), excluded_unique_photos=len(out['photos']),
                  excluded_known_md5=len(out['md5']), historical_missing_md5=missing_md5,
                  historical_unknown_author_slots=96)
    return out, counts


def _capacity(records, excluded):
    """Deterministic independent lower bound, not maximum matching or selection."""
    available, seen = [], set()
    for r in records:
        iid = r['image_id']; row = r['publisher_metadata']
        rt.require(iid == row['ImageID'] and iid not in seen and type(r['eligible']) is bool, 'Unique census record required')
        seen.add(iid)
        if not r['eligible'] or iid in excluded['ids'] or not row['AuthorProfileURL'] or not row['OriginalMD5']: continue
        rt.require(row['Rotation'] == '0.0', 'Original zero orientation required')
        identity = _metadata_identity(row)
        if any(v in excluded[k] for v, k in zip(identity, ('authors', 'photos', 'md5', 'urls'))): continue
        available.append((iid, *identity))
    available.sort()
    unique = [set() for _ in range(4)]; lower_bound = 0
    for _, *identity in available:
        if any(v in group for v, group in zip(identity, unique)): continue
        for v, group in zip(identity, unique): group.add(v)
        lower_bound += 1
    return dict(eligible_after_all_exclusions=len(available), source_authors=len({r[1] for r in available}),
                unique_photos=len({r[2] for r in available}), unique_md5=len({r[3] for r in available}),
                independent_slots_lower_bound=lower_bound, metadata_inventory_identity=pin(encode(available)))


def collect(paths, cfg, excluded, check):
    human, parts = set(cfg['human_classes']), set(cfg['body_part_classes'])
    triplets = set()
    for r in census.csv_rows(paths['triplets']):
        check()
        if r['RelationshipLabel'] == 'holds' and r['LabelName1'] in human and r['LabelName2'] not in human | parts:
            triplets.add((r['LabelName1'], r['LabelName2']))
    relations, metadata = {}, {}
    for r in census.csv_rows(paths['relations']):
        check()
        if r['ImageID'] not in excluded['ids'] and r['RelationshipLabel'] == 'holds' and (r['LabelName1'], r['LabelName2']) in triplets:
            relations.setdefault(r['ImageID'], []).append(r)
    for r in census.csv_rows(paths['metadata']):
        check()
        if r['ImageID'] in relations:
            rt.require(r['ImageID'] not in metadata, 'Duplicate publisher metadata identity')
            metadata[r['ImageID']] = r
    rt.require(set(metadata) == set(relations), 'Publisher metadata missing untouched relation ID')
    boxes = {}
    for iid, row in metadata.items():
        check()
        if row['Rotation'] != '0.0' or not row['AuthorProfileURL'] or not row['OriginalMD5']: continue
        identity = _metadata_identity(row)
        if not any(v in excluded[k] for v, k in zip(identity, ('authors', 'photos', 'md5', 'urls'))): boxes[iid] = []
    for r in census.csv_rows(paths['boxes']):
        check()
        if r['ImageID'] in boxes: boxes[r['ImageID']].append(r)
    records = []; unresolved = 0
    for iid in sorted(boxes):
        check(); facts = census.image_census(boxes[iid], relations[iid], human, parts)
        unresolved += facts['unscorable_positive_pairs']
        records.append(dict(image_id=iid, publisher_metadata=metadata[iid], eligible=facts['metadata_eligible']))
    counts = _capacity(records, excluded)
    counts.update(new_relation_images=len(relations), metadata_identity_eligible_images=len(boxes),
                  geometry_orientation_eligible_images=sum(r['eligible'] for r in records),
                  unscorable_positive_pairs=unresolved)
    return counts


def configuration(code, source):
    cfg = rt.pinned(code/CONFIG, source['helpers'][CONFIG], 16 << 10)
    fixed = dict(schema='world_reward.ownership_pair_census.v1', budget_seconds=180,
                 output='results/ownership-pair-census-v1', selection='none_feasibility_only', minimum_independent_slots=96,
                 historical_slots=336, historical_oi_slots=240, historical_coco_slots=96, network_allowed=False, retry_count=0)
    rt.require(all(type(cfg.get(k)) is type(v) and cfg[k] == v for k, v in fixed.items()), 'Frozen feasibility scope required')
    rt.require(set(cfg['files']) == {'boxes', 'relations', 'triplets', 'metadata'}
               and [r['name'] for r in cfg['historical']] == ['oi16', 'oi128', 'oi64', 'oi32', 'coco32', 'coco64']
               and set(cfg['reused_helper_pins']) == set(HELPERS[3:]), 'Complete metadata/source inventory required')
    for name, expected in cfg['reused_helper_pins'].items():
        rt.require(source['helpers'].get(name) == expected, 'Original helper bytes differ')
    original = rt.pinned(code/HELPERS[7], source['helpers'][HELPERS[7]], 16 << 10)
    rt.require(cfg['human_classes'] == original['human_classes'] and cfg['body_part_classes'] == original['body_part_classes'],
               'Unchanged original human/body-part catalogs required')
    for module, name in ((rt, HELPERS[3]), (census, HELPERS[4]), (identities, HELPERS[5]), (coco, HELPERS[6])):
        rt.require(Path(module.__file__).resolve() == code/name, 'Imported helper origin differs')
    return cfg


def _state(path):
    s = path.lstat()
    return tuple(getattr(s, k) for k in ('st_dev', 'st_ino', 'st_mode', 'st_size', 'st_mtime_ns', 'st_ctime_ns', 'st_nlink', 'st_uid', 'st_gid'))


def _publish(output, report, deadline, started):
    """Retain the owned FD through sealing; a late publication is never PASS."""
    path = output/'report.json'; candidate = report['status']
    report['status'] = 'fail'
    with path.open('xb') as stream:
        os.fchmod(stream.fileno(), 0o444); stream.write(encode(report)); stream.flush(); os.fsync(stream.fileno())
        output.chmod(0o555)
        report.update(outputs_sealed=True, elapsed_seconds=time.monotonic()-started, status=candidate)
        if time.monotonic() >= deadline: report.update(status='fail', error_type='TimeoutError')
        raw = encode(report); stream.seek(0); stream.write(raw); stream.truncate(); stream.flush(); os.fsync(stream.fileno())
        if time.monotonic() >= deadline:
            report.update(status='fail', error_type='TimeoutError', elapsed_seconds=time.monotonic()-started)
            raw = encode(report); stream.seek(0); stream.write(raw); stream.truncate(); stream.flush(); os.fsync(stream.fileno())
        rt.require(rt.identity(path, 2 << 20) == pin(raw), 'Final census receipt bytes differ')
        if time.monotonic() >= deadline and report['status'] == 'pass':
            report.update(status='fail', error_type='TimeoutError', elapsed_seconds=time.monotonic()-started)
            raw = encode(report); stream.seek(0); stream.write(raw); stream.truncate(); stream.flush(); os.fsync(stream.fileno())
    return pin(raw)


def run():
    started = time.monotonic(); deadline = started+180
    rt.require(sys.platform == 'linux' and os.geteuid() == 0 and os.uname().nodename == 'world-reward-ncc-h100-02', 'Exact Azure CPU metadata host required')
    revision, code = os.environ['WR_CODE_REVISION'], Path(os.environ['WR_CODE'])
    rt.require(Path(__file__).resolve() == code/HELPERS[0], 'Original immutable driver required')
    source = rt.source(ROOT, code, revision, ENTRY, HELPERS); cfg = configuration(code, source)
    files = [*cfg['files'].values(), *cfg['historical']]; paths = [rt.canonical(r['path']) for r in files]
    rt.require(len(paths) == len(set(paths)) == 10, 'Exactly ten distinct metadata inputs required')
    states = {str(p): _state(p) for p in paths}
    source_modes = {str(p.relative_to(code)): stat.S_IMODE(p.lstat().st_mode) for p in (code, *sorted(code.rglob('*')))}
    def check():
        if time.monotonic() >= deadline: raise TimeoutError('Census deadline reached')
    def rehash():
        check()
        for path, row in zip(paths, files):
            rt.require(rt.identity(path, 100 << 20, readonly=row['readonly']) == row['pin']
                       and _state(path) == states[str(path)], 'Metadata input identity or mode changed')
        check()
    rehash(); excluded, history_counts = _historical_exclusions([rt.strict(p.read_bytes()) for p in paths[4:]])
    output = rt.canonical(ROOT/cfg['output']); rt.require(not output.exists() and output.parent.is_dir(), 'Fresh census output required')
    output.mkdir(mode=0o700)
    report = dict(schema=cfg['schema'], producer_revision=revision, status='fail', source_binding=source,
                  configuration_identity=source['helpers'][CONFIG], frozen_inputs=files,
                  source_modes_identity=pin(encode(source_modes)), counts=history_counts, selection_performed=False,
                  historical_reference_values_read=False, historical_reference_files_opened=False,
                  network_used=False, rgb_read=False, predictions_read=False, models_loaded=False, gpu_used=False,
                  challenge_inputs_used=False, rights_verified=False, training_overlap_verified=False,
                  ownership_verified=False, quality_verified=False, adopted=False, old_studies_reopened=False,
                  coco_author_disjointness_verified=False, coco_byte_alias_exclusions_complete=False,
                  limitations=cfg['limitations'], source_and_inputs_rehashed_after=False, outputs_sealed=False)
    def interrupted(*_): raise TimeoutError('Census interrupted')
    handlers = {s: signal.signal(s, interrupted) for s in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM)}
    signal.setitimer(signal.ITIMER_REAL, max(.001, deadline-time.monotonic()))
    try:
        report['counts'].update(collect(dict(zip(cfg['files'], paths[:4])), cfg, excluded, check))
        enough = report['counts']['independent_slots_lower_bound'] >= cfg['minimum_independent_slots']
        report['capacity_gate_passed'] = enough
        report['decision'] = 'FEASIBLE_PENDING_FROZEN_NEW_STUDY' if enough else 'CLOSED_CAPACITY_INCONCLUSIVE'
        rehash(); rt.require(rt.source(ROOT, code, revision, ENTRY, HELPERS) == source
            and {str(p.relative_to(code)): stat.S_IMODE(p.lstat().st_mode) for p in (code, *sorted(code.rglob('*')))} == source_modes,
            'Source identity or modes changed')
        report['source_and_inputs_rehashed_after'] = True; check(); report['status'] = 'pass'
    except BaseException as exc:
        report.update(status='fail', error_type=type(exc).__name__ if type(exc) in (ValueError, TimeoutError, OSError, KeyError) else 'other')
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        try:
            if not report['source_and_inputs_rehashed_after']:
                try:
                    rehash(); rt.require(rt.source(ROOT, code, revision, ENTRY, HELPERS) == source
                        and {str(p.relative_to(code)): stat.S_IMODE(p.lstat().st_mode) for p in (code, *sorted(code.rglob('*')))} == source_modes,
                        'Source identity or modes changed')
                    report['source_and_inputs_rehashed_after'] = True
                except BaseException:
                    report['status'] = 'fail'
            receipt = _publish(output, report, deadline, started)
            print(json.dumps(dict(status=report['status'], decision=report.get('decision'), capacity_gate_passed=report.get('capacity_gate_passed'), counts=report['counts'], report=receipt), sort_keys=True))
        finally:
            for sig, handler in handlers.items(): signal.signal(sig, handler)
    return report


if __name__ == '__main__':
    rt.require(len(sys.argv) == 1, 'No per-record parameters allowed')
    sys.exit(0 if run()['status'] == 'pass' else 1)
