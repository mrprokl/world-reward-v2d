"""Fresh author/photo-disjoint Open Images proposal cohort; metadata only.

No acquisition, RGB, model, predictions or quality evaluation. References select
records privately, never inference candidates. Historical studies stay closed.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mediapipe_cpu_runtime_verify as rt
import openimages_joint_pair_census as census
import openimages_joint_pair_acquire as identities

ROOT = Path('/srv/scenesmith/world-reward')
ENTRY = 'run_proposal_external_census'
CONFIG = 'configs/proposal_external_census_v1.json'
HASH_NAMESPACE = 'world_reward.oi_proposal_external_v1/'
HELPERS = ('infra/proposal_external_census.py', 'infra/run_proposal_external_census.sh',
           CONFIG, 'infra/mediapipe_cpu_runtime_verify.py',
           'infra/openimages_joint_pair_census.py', 'infra/openimages_joint_pair_acquire.py')


def encode(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False)+'\n').encode()


def pin(raw):
    return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def exclusions(old16, old128, old64):
    """All selected slots, including missing RGB, are historical exclusions."""
    rt.require(old64.get('schema') == 'world_reward.openimages_joint_pair_cohort.v1'
               and old64.get('no_replacements') is True
               and old64.get('challenge_inputs_used') is False, 'Exact old64 cohort schema required')
    result = {k: set() for k in ('ids', 'authors', 'md5', 'photos', 'urls')}
    groups = []
    for value, count in ((old16, 16), (old128, 128)):
        ids, rows = value['selected_ids'], value['public_metadata']
        rt.require(len(ids) == len(set(ids)) == len(rows) == count
                   and {r['ImageID'] for r in rows} == set(ids), 'Exact historical metadata slots required')
        groups.append(rows)
    slots = old64['records']
    rt.require(len(slots) == 64 and [r['slot'] for r in slots] == list(range(64))
               and [r['split'] for r in slots] == ['DEV']*32+['TEST']*32, 'All frozen old64 slots required')
    groups.append([r['publisher_metadata'] for r in slots])
    for rows in groups:
        for row in rows:
            iid = row['ImageID']
            rt.require(re.fullmatch('[0-9a-f]{16}', iid) and iid not in result['ids'], 'Unique historical208 IDs required')
            result['ids'].add(iid)
            # Missing old author identity cannot certify author disjointness.
            result['authors'].add(identities.profile_identity(row['AuthorProfileURL']))
            result['photos'].add(identities.excluded_photo_identity(row['OriginalLandingURL']))
            rt.require(type(row['OriginalURL']) is str and bool(row['OriginalURL']), 'Historical original URL required')
            result['urls'].add(row['OriginalURL'])
            if row['OriginalMD5']:
                result['md5'].add(identities.md5_identity(row['OriginalMD5']))
    rt.require(len(result['ids']) == 208, 'All historical208 IDs excluded')
    return result


def candidate_identity(row):
    rt.require(type(row) is dict and re.fullmatch('[0-9a-f]{16}', row['ImageID']), 'Original metadata identity required')
    author = identities.profile_identity(row['AuthorProfileURL'])
    photo = identities.excluded_photo_identity(row['OriginalLandingURL'])
    digest = identities.md5_identity(row['OriginalMD5'])
    rt.require(type(row['OriginalURL']) is str and bool(row['OriginalURL']), 'Original URL required')
    return author, photo, digest, row['OriginalURL']


def choose(records, excluded):
    """Pure deterministic selection before any accessibility/rights request."""
    seen = set(); available = []
    for record in records:
        row = record['publisher_metadata']; iid = row['ImageID']
        rt.require(iid == record['image_id'] and iid not in seen
                   and type(record['eligible']) is bool, 'Unique census record required')
        seen.add(iid)
        if not record['eligible'] or iid in excluded['ids']:
            continue
        rt.require(row['Rotation'] == '0.0', 'Original zero orientation required')
        if not row['AuthorProfileURL'] or not row['OriginalMD5']:
            continue
        author, photo, digest, url = candidate_identity(row)
        if author in excluded['authors'] or photo in excluded['photos'] or digest in excluded['md5'] or url in excluded['urls']:
            continue
        available.append((row, author, photo, digest, url))
    available.sort(key=lambda v: (hashlib.sha256((HASH_NAMESPACE+v[0]['ImageID']).encode()).hexdigest(), v[0]['ImageID']))
    authors = set(); photos = set(); hashes = set(); urls = set(); selected = []
    for row, author, photo, digest, url in available:
        if author in authors or photo in photos or digest in hashes or url in urls:
            continue
        metadata = {k: row[k] for k in identities.METADATA_KEYS}
        rt.require(all(type(v) is str for v in metadata.values()), 'Original textual publisher metadata required')
        selected.append(dict(slot=len(selected), split='DEV' if len(selected) < 16 else 'RESERVED', publisher_metadata=metadata))
        authors.add(author); photos.add(photo); hashes.add(digest); urls.add(url)
    counts = dict(eligible_after_all_exclusions=len(available),
                  source_author_profiles=len({v[1] for v in available}),
                  unique_author_photo_md5_slots=len(selected))
    return (selected[:32] if len(selected) >= 32 else []), counts


def collect(paths, cfg, excluded, check):
    """Reuse original unchanged geometry semantics, never historical predictions."""
    human, parts = set(cfg['human_classes']), set(cfg['body_part_classes'])
    triplets = {(r['LabelName1'], r['LabelName2']) for r in census.csv_rows(paths['triplets'])
                if r['RelationshipLabel'] == 'holds' and r['LabelName1'] in human and r['LabelName2'] not in human | parts}
    relations = {}
    for row in census.csv_rows(paths['relations']):
        check()
        if row['ImageID'] not in excluded['ids'] and row['RelationshipLabel'] == 'holds' and (row['LabelName1'], row['LabelName2']) in triplets:
            relations.setdefault(row['ImageID'], []).append(row)
    boxes = {iid: [] for iid in relations}; metadata = {}
    for row in census.csv_rows(paths['boxes']):
        check()
        if row['ImageID'] in boxes:
            boxes[row['ImageID']].append(row)
    for row in census.csv_rows(paths['metadata']):
        check()
        if row['ImageID'] in relations:
            rt.require(row['ImageID'] not in metadata, 'Duplicate publisher metadata identity')
            metadata[row['ImageID']] = row
    rt.require(set(metadata) == set(relations), 'Publisher metadata missing a new relationship image')
    records = []
    for iid in sorted(relations):
        check(); facts = census.image_census(boxes[iid], relations[iid], human, parts)
        records.append(dict(image_id=iid, publisher_metadata=metadata[iid],
                            eligible=facts['metadata_eligible'] and metadata[iid]['Rotation'] == '0.0'))
    selected, counts = choose(records, excluded)
    counts.update(new_relation_images=len(records), geometry_orientation_eligible=sum(r['eligible'] for r in records),
                  excluded_ids=len(excluded['ids']), excluded_author_profiles=len(excluded['authors']))
    return selected, counts


def configuration(code, source):
    cfg = rt.pinned(code/CONFIG, source['helpers'][CONFIG], 16 << 10)
    rt.require(cfg['schema'] == 'world_reward.proposal_external_census.v1'
               and cfg['budget_seconds'] == 180 and cfg['slots'] == 32 and cfg['dev'] == cfg['reserved'] == 16
               and cfg['hash_namespace'] == HASH_NAMESPACE and cfg['output'] == 'results/proposal-external-census-v1'
               and cfg['old_ids_excluded'] == 208 and cfg['network_allowed'] is False
               and set(cfg['files']) == {'boxes', 'relations', 'triplets', 'metadata', 'old16', 'old128', 'old64'},
               'Frozen metadata-only32-slot scope required')
    for name, expected in cfg['reused_helper_pins'].items():
        rt.require(source['helpers'].get(name) == expected, 'Reused original helper bytes differ')
    rt.require(set(cfg['reused_helper_pins']) == set(HELPERS[3:]), 'All imported helper pins required')
    for module, name in ((rt, HELPERS[3]), (census, HELPERS[4]), (identities, HELPERS[5])):
        rt.require(Path(module.__file__).resolve() == code/name, 'Imported helper outside frozen source')
    return cfg


def run():
    started = time.monotonic(); deadline = started+180
    rt.require(sys.platform == 'linux' and os.geteuid() == 0 and os.uname().nodename == 'world-reward-ncc-h100-02',
               'Exact Azure CPU metadata host required')
    revision, code = os.environ['WR_CODE_REVISION'], Path(os.environ['WR_CODE'])
    rt.require(Path(__file__).resolve() == code/HELPERS[0], 'Original immutable driver required')
    source = rt.source(ROOT, code, revision, ENTRY, HELPERS); cfg = configuration(code, source)
    paths = {k: Path(v['path']) for k, v in cfg['files'].items()}
    def check():
        rt.require(time.monotonic() < deadline, 'Inclusive180s census deadline reached')
    def rehash():
        check()
        for name, row in cfg['files'].items():
            rt.require(rt.identity(paths[name], 100 << 20, readonly=name == 'old64') == row['pin'], 'Frozen metadata input differs')
        check()
    rehash()
    old = {k: rt.strict(paths[k].read_bytes()) for k in ('old16', 'old128', 'old64')}
    excluded = exclusions(old['old16'], old['old128'], old['old64'])
    output = rt.canonical(ROOT/cfg['output'])
    rt.require(not output.exists() and output.parent.is_dir(), 'Fresh census output namespace required')
    output.mkdir(mode=0o700)
    report = dict(schema=cfg['schema'], producer_revision=revision, source_binding=source,
                  configuration_identity=source['helpers'][CONFIG], frozen_inputs=cfg['files'], status='fail',
                  network_used=False, rgb_read=False, predictions_read=False, models_loaded=False, gpu_used=False,
                  challenge_inputs_used=False, quality_verified=False, training_overlap_verified=False,
                  challenge_overlap_verified=False, individual_image_license_verified=False, adopted=False,
                  old_studies_reopened=False, rights_requests_before_freeze=0, local_heavy_transfer=False,
                  source_and_inputs_rehashed_after=False)
    cohort = None
    def interrupted(*_):
        raise TimeoutError('Metadata census interrupted')
    handlers = {s: signal.signal(s, interrupted) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        selected, counts = collect(paths, cfg, excluded, check); report['counts'] = counts
        if selected:
            cohort = dict(schema='world_reward.proposal_external_cohort.v1', producer_revision=revision,
                          source_binding=source, configuration_identity=source['helpers'][CONFIG],
                          frozen_input_identities=cfg['files'], hash_namespace=HASH_NAMESPACE, records=selected,
                          no_replacements=True, rights_requests_before_freeze=0, reference_geometry_exposed=False,
                          exclusions_verified=['208_ids', 'author_profiles', 'original_md5', 'photo_ids', 'original_urls'],
                          challenge_inputs_used=False, quality_verified=False, training_overlap_verified=False,
                          challenge_overlap_verified=False, individual_image_license_verified=False, adopted=False)
        report['decision'] = 'FROZEN32_PENDING_INDIVIDUAL_RIGHTS_AND_SEPARATE_INFERENCE' if cohort else 'INCONCLUSIVE_CLOSED_NO_RGB'
        rehash(); rt.require(rt.source(ROOT, code, revision, ENTRY, HELPERS) == source, 'Source changed after census')
        report['source_and_inputs_rehashed_after'] = True; check()
        if cohort:
            raw = encode(cohort); rt.write(output/'cohort.json', raw, 0o444)
            report['cohort_identity'] = pin(raw)
            rt.require(rt.identity(output/'cohort.json', 1 << 20) == pin(raw), 'Frozen cohort changed')
        check(); report['status'] = 'pass'
    except BaseException as exc:
        report['error_type'] = type(exc).__name__ if type(exc) in (ValueError, TimeoutError, OSError, KeyError) else 'other'
        raise
    finally:
        try:
            report['elapsed_seconds'] = time.monotonic()-started
            if report['elapsed_seconds'] >= cfg['budget_seconds']:
                report['status'] = 'fail'; report['error_type'] = 'TimeoutError'
            # Keep this owned descriptor until the directory seal/deadline check;
            # a late publication remains FAIL rather than a historical PASS.
            with (output/'report.json').open('xb') as stream:
                os.fchmod(stream.fileno(), 0o444)
                raw = encode(report); stream.write(raw); stream.flush(); os.fsync(stream.fileno())
                output.chmod(0o555)
                report['elapsed_seconds'] = time.monotonic()-started
                if report['elapsed_seconds'] >= cfg['budget_seconds']:
                    report['status'] = 'fail'; report['error_type'] = 'TimeoutError'
                raw = encode(report); stream.seek(0); stream.write(raw); stream.truncate()
                stream.flush(); os.fsync(stream.fileno())
                if time.monotonic()-started >= cfg['budget_seconds'] and report['status'] == 'pass':
                    report.update(status='fail', error_type='TimeoutError', elapsed_seconds=time.monotonic()-started)
                    raw = encode(report); stream.seek(0); stream.write(raw); stream.truncate()
                    stream.flush(); os.fsync(stream.fileno())
            print(json.dumps(dict(status=report['status'], counts=report.get('counts'), decision=report.get('decision'),
                                  report=pin(raw), cohort=report.get('cohort_identity'), elapsed_seconds=report['elapsed_seconds']), sort_keys=True))
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
    return report


if __name__ == '__main__':
    rt.require(len(sys.argv) == 1, 'No per-record parameters allowed'); run()
