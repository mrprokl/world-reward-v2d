"""Private deterministic FIT32/CAL16 identities; no references or acquisition.

The caller authenticates the frozen census source/proof and derives all448
historical photo exclusions. This pure projection validates that supplied proof,
not the original filesystem or execution. Its output is NOT a predictor input.
"""
import hashlib
import json
import re

NAMESPACE = 'world_reward.vcoco_fit_cal_v1/'
SCHEMA = 'world_reward.vcoco_fit_cal_identity_cohort.v1'
CENSUS_REVISION = 'b9270d55a3ed10c28e080e019107cedb18e63fd1'
CENSUS_CLOSURE = 'eb0dfc966a2a8dba1b249358d5152a23e16b42ff58eab1d66112551b74aebf76'
REPORT_PIN = dict(bytes=57524, sha256='703e0e341d3ceb8c44d40f5a90823299719ade1559c36945bc6f8d92120ffee0')
INVENTORY_PIN = dict(bytes=24991, sha256='ea7503e6e2e09cc6b4536ce12bfed9ad73698488ebc50b22db0fcfe8276fbd70')
COUNTS = dict(train=195, val=191)
SLOTS = (('train', 'FIT', 32), ('val', 'CAL', 16))


def _require(ok, message):
    if not ok: raise ValueError(message)


def _encode(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False) + '\n').encode()


def _pin(raw):
    return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def _strict(raw):
    def pairs(rows):
        result = {}
        for key, value in rows:
            _require(key not in result, 'Duplicate metadata field')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite metadata')))


def _photo(value):
    return type(value) is str and re.fullmatch(r'[0-9]+', value) is not None


def freeze_fit_cal_cohort(report_raw, inventory_raw, *, excluded_photo_ids, expected_input_proof):
    """Validate exact saved bytes, then freeze identities without semantic labels.

    `expected_input_proof` must come from an independently authenticated original
    census caller, with `excluded_photo_ids` its reconstructed432+16 photo set.
    Neither argument is authenticated by this pure function. No availability,
    pose, object, role, action, score or quality filter is consulted here.
    """
    _require(type(report_raw) is bytes and type(inventory_raw) is bytes, 'Exact original byte inputs required')
    _require(_pin(report_raw) == REPORT_PIN and _pin(inventory_raw) == INVENTORY_PIN,
             'Independently frozen census report/inventory pins differ')
    report, rows = _strict(report_raw), _strict(inventory_raw)
    _require(type(expected_input_proof) is dict and set(expected_input_proof) ==
             {'current', 'original_prepare', 'original_census', 'prepare_proof', 'pilot'},
             'Full independently authenticated original input proof required')
    proof = _strict(_encode(expected_input_proof))
    _require(report['input_proof'] == proof, 'Original input proof differs after JSON round trip')
    binding = proof['current']['binding']
    _require(binding['producer_revision'] == CENSUS_REVISION and type(binding['entries']) is int
             and binding['entries'] == 313 and binding['closure_sha256'] == CENSUS_CLOSURE,
             'Frozen308-file census source required')
    _require(report['schema'] == 'world_reward.vcoco_fit_cal_census.v1'
             and report['producer_revision'] == CENSUS_REVISION and report['status'] == 'pass'
             and report['stage'] == 'complete' and report['decision'] == 'CAPACITY_METADATA_ONLY_NO_SELECTION'
             and report['capacity_gate_passed'] is True and report['outputs_sealed'] is True
             and report['source_and_inputs_rehashed_after'] is True and report['historical_photos'] == 448
             and report['minimum_distinct_photos'] == dict(train=32, val=16)
             and report['artifact_identities'] == {'inventory.json': INVENTORY_PIN}
             and report['eligibility_inventory_identity'] == INVENTORY_PIN
             and type(report['eligibility_inventory_rows']) is int and report['eligibility_inventory_rows'] == 386,
             'Complete original count-only receipt required')
    false_flags = ('historical_reference_values_read', 'pilot_reference_values_read', 'TEST_role_values_read',
                   'RGB_read', 'network_used', 'GPU_used', 'models_loaded', 'FIT_performed', 'selection_performed',
                   'author_disjointness_verified', 'challenge_overlap_verified', 'training_overlap_verified', 'adopted')
    _require(all(report[k] is False for k in false_flags)
             and report['fresh_reference_geometry_consulted'] is True, 'Original scope flags differ')
    _require(type(excluded_photo_ids) in (set, frozenset) and len(excluded_photo_ids) == 448
             and all(_photo(x) for x in excluded_photo_ids), 'All448 upstream-authenticated exclusions required')
    _require(type(rows) is list and len(rows) == 386, 'Complete386-row private census inventory required')
    populations = {name: [] for name in COUNTS}; ids, photos = set(), set()
    for row in rows:
        _require(type(row) is dict and set(row) == {'split', 'image_id', 'photo_id'}
                 and row['split'] in COUNTS and type(row['image_id']) is int and 0 < row['image_id'] < 10**12
                 and _photo(row['photo_id']), 'Exact native identity-only row required')
        _require(row['image_id'] not in ids and row['photo_id'] not in photos
                 and row['photo_id'] not in excluded_photo_ids, 'Duplicate/cross-population/historical identity rejected')
        ids.add(row['image_id']); photos.add(row['photo_id']); populations[row['split']].append(row)
    _require(set(report['splits']) == set(COUNTS) and all(len(populations[name]) == count
             and type(report['splits'][name]['distinct_eligible_photos']) is int
             and report['splits'][name]['distinct_eligible_photos'] == count
             and type(report['splits'][name]['eligible_images']) is int
             and report['splits'][name]['eligible_images'] == count for name, count in COUNTS.items()),
             'Exact195TRAIN/191VAL census required; no availability subset')
    selected = []
    for official, study, count in SLOTS:
        ranked = sorted(populations[official], key=lambda row: (hashlib.sha256(
            (NAMESPACE + f"{row['image_id']:012d}").encode()).hexdigest(), row['image_id']))
        start = len(selected)
        selected.extend(dict(slot=start+i, split=study, official_split=official,
                             image_id=row['image_id'], photo_id=row['photo_id'],
                             rank_sha256=hashlib.sha256((NAMESPACE + f"{row['image_id']:012d}").encode()).hexdigest())
                        for i, row in enumerate(ranked[:count]))
    _require(len(selected) == 48 and len({r['photo_id'] for r in selected}) == 48,
             'Fixed48 unique-photo denominator required')
    return dict(schema=SCHEMA, namespace=NAMESPACE, census_producer_revision=CENSUS_REVISION,
                census_report_identity=dict(REPORT_PIN), census_inventory_identity=dict(INVENTORY_PIN),
                input_proof_identity=_pin(_encode(proof)), excluded_photo_ids_identity=_pin(_encode(sorted(excluded_photo_ids))),
                population_counts=dict(COUNTS), selected_slots=48, split_counts=dict(FIT=32, CAL=16), records=selected,
                retry_count=0, replacement_count=0, all_48_acquired_required=True,
                reference_values_exposed=False, predictor_input=False, source_authenticated=False,
                author_disjointness_verified=False, training_overlap_verified=False, challenge_overlap_verified=False,
                acquisition_performed=False, models_loaded=False, FIT_performed=False, CAL_evaluated=False,
                task_scope='published_positive_person_object_role_retrieval_not_anatomical_ownership_or_contact')
