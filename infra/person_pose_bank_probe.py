"""CPU-only full-person crop diagnostic; no target selection or quality claim.

Reuses frozen automatic detector records, not manually supplied boxes. All
retained persons in the three recorded seed banks are processed in original
order. The other seed positions have counts only and are never interpolated.
Original RGB and DWPose native preprocessing/SimCC decoding are unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

import numpy as np

import dwpose_smoke as dw
import keypoint_rgb_dwpose as dw_evidence
import mediapipe_cpu_runtime_verify as rt
from world_reward.person_pose_observations import infer_person_pose_frame
from world_reward.prompt_selection import BoxDetection, non_maximum_suppression

ROOT = Path('/srv/scenesmith/world-reward')
ENTRY = 'run_person_pose_bank_probe'
CONFIG = 'configs/person_pose_bank_probe_v1.json'
IMAGE = 'sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7'
BUDGET = 300
HELPERS = ('infra/person_pose_bank_probe.py', 'infra/run_person_pose_bank_probe.sh', CONFIG,
           'src/world_reward/person_pose_observations.py', 'src/world_reward/prompt_selection.py',
           'infra/dwpose_smoke.py', 'infra/run_dwpose_smoke.sh', 'infra/dwpose_acquire.py',
           'infra/dwpose_wheel_audit.py', 'infra/keypoint_rgb_dwpose.py',
           'infra/mediapipe_cpu_runtime_verify.py')


def file_id(path, maximum=32 << 20):
    return rt.identity(path, maximum, readonly=False)


def write(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    path.chmod(0o444)


def saved_person_banks(diagnostic, width, height, total, episode):
    """Verify the unchanged raw→NMS census; never infer missing seed boxes."""
    rt.require(type(episode) is int and episode in (9, 26)
               and type(total) is int and total >= 3, 'Fixed diagnostic episodes required')
    expected = dict(episode=episode, frames=total, confidence=.3, nms_iou=.7,
                    detector_revision='12bdfa3120f3e7ec7b434d90674b3396eccf88eb',
                    input_track='track_1', ground_truth_used=False, hand_labeled_test=False)
    rt.require(all(type(diagnostic.get(k)) is type(v) and diagnostic[k] == v
                   for k, v in expected.items()), 'Actual automatic detector protocol required')
    rows = diagnostic.get('observations')
    indices = np.unique(np.linspace(0, total-1, min(total, 16), dtype=int)).tolist()
    rt.require(type(rows) is list and [x.get('frame') for x in rows] == indices
               and all(type(x.get('frame')) is int for x in rows), 'All original seed indices required')
    banks = []
    for position, row in enumerate(rows):
        if position >= 3:
            rt.require('detector_observations' not in row, 'Do not silently select extra recorded banks')
            continue
        records = row.get('detector_observations')
        rt.require(type(records) is list and len(records) == 2, 'Complete recorded two-query census required')
        people = [x for x in records if x.get('query') == 'person.']
        rt.require(len(people) == 1, 'One original person query required')
        person = people[0]
        raw, retained = person.get('boxes'), person.get('retained')
        rt.require(type(raw) is list and type(retained) is list
                   and type(row.get('person_candidates')) is int
                   and row['person_candidates'] == len(retained), 'No person census omission allowed')
        def detection(x):
            rt.require(type(x) is dict and set(x) == {'box', 'score'}
                       and type(x['box']) is list and len(x['box']) == 4
                       and all(type(v) in (float, int) for v in (*x['box'], x['score'])),
                       'Original raw numeric boxes/scores required')
            return BoxDetection(tuple(x['box']), x['score'])
        actual = non_maximum_suppression(tuple(map(detection, raw)), width, height, .3, .7)
        rt.require(tuple(map(detection, retained)) == actual, 'Original complete retained NMS rows differ')
        boxes = np.asarray([x.box for x in actual], np.float64).reshape(len(actual), 4)
        scores = np.asarray([x.score for x in actual], np.float64)
        frame = row['frame']
        ids = tuple(f'episode:{episode:06d}/frame:{frame:06d}/person/retained:{i:06d}'
                    for i in range(len(actual)))
        banks.append(dict(frame_index=frame, person_ids=ids, boxes=boxes, detector_scores=scores))
    rt.require(len(banks) == 3, 'All three recorded banks required')
    return banks


def configuration(code):
    p = rt.strict((code/CONFIG).read_bytes())
    rt.require(p['schema'] == 'world_reward.person_pose_bank_probe.v1'
               and p['image_id'] == IMAGE and p['budget_seconds'] == BUDGET
               and [x['episode'] for x in p['episodes']] == [9, 26], 'Frozen automatic diagnostic scope required')
    validate_paths(p)
    return p


def validate_paths(p):
    """Whitelist exact public inputs before a shell mount can be emitted."""
    metadata = {'results/input-manifest.json', 'data/track_1/meta/episodes.jsonl'}
    rt.require(set(p['metadata_files']) == metadata, 'Only Track1 RGB metadata required')
    for row in p['episodes']:
        episode = row['episode']
        rt.require(type(episode) is int and episode in (9, 26), 'Fixed two episode diagnostic required')
        base = f'outputs/episode_{episode:06d}/automatic_masks'
        expected = dict(video=f'data/track_1/videos/chunk-000/observation.images.exo_camera/episode_{episode:06d}.mp4',
                        report=base+'/report.json', diagnostic=base+'/seed-diagnostics.json')
        rt.require(all(row[k] == v for k, v in expected.items())
                   and set(row['files']) == set(expected.values())
                   and type(row['total_frames']) is int and row['total_frames'] >= 3
                   and re.fullmatch('[0-9a-f]{64}', row['automatic_source_sha256']) is not None,
                   'Exact automatic source/video/diagnostic paths required')
    for pin in list(p['metadata_files'].values()) + [x for row in p['episodes'] for x in row['files'].values()]:
        rt.require(type(pin) is dict and set(pin) == {'bytes', 'sha256'}
                   and type(pin['bytes']) is int and 0 < pin['bytes'] < (32 << 30)
                   and type(pin['sha256']) is str and re.fullmatch('[0-9a-f]{64}', pin['sha256']) is not None,
                   'Independent exact file pins required')


def authenticate(code, revision, p):
    source = rt.source(ROOT, code, revision, ENTRY, HELPERS)
    dw.source_identity()
    evidence = dw_evidence.validate_smoke(ROOT)
    assets = dw.validate_assets(ROOT)
    files = {}
    for name, pin in p['metadata_files'].items():
        rt.require(file_id(ROOT/name) == pin, 'Pinned Track1 metadata changed')
        files[name] = pin
    for row in p['episodes']:
        for name, pin in row['files'].items():
            path = ROOT/name
            rt.require(file_id(path, 32 << 30) == pin, 'Independent diagnostic/video pins differ')
            files[name] = pin
        report = rt.strict((ROOT/row['report']).read_bytes())
        facts = dict(stage='automatic_masks', status='pass', episode_index=row['episode'],
                     input_track='track_1', ground_truth_used=False, hand_labeled_test=False, oracle_modes=[])
        rt.require(all(type(report.get(k)) is type(v) and report[k] == v for k, v in facts.items())
                   and report['input_sha256'] == row['files'][row['video']]['sha256']
                   and report['script_sha256'] == row['automatic_source_sha256'], 'Original automatic producer required')
    manifest = rt.strict((ROOT/'results/input-manifest.json').read_bytes())
    rt.require((manifest.get('track'), manifest.get('repo_id'), manifest.get('revision')) ==
               ('track_1', 'nvidia/video_to_data_challenge', '5f68335f3acc802033d1e80728c1633197521de8'),
               'Official pinned Track1 manifest required')
    metadata = [rt.strict(line) for line in (ROOT/'data/track_1/meta/episodes.jsonl').read_bytes().splitlines()]
    for row in p['episodes']:
        records = [x for x in manifest['files'] if 'data/'+x['path'] == row['video']]
        episodes = [x for x in metadata if x['episode_index'] == row['episode']]
        rt.require(len(records) == len(episodes) == 1 and episodes[0]['length'] == row['total_frames']
                   and {k: records[0][k] for k in ('bytes', 'sha256')} == row['files'][row['video']],
                   'Original video and fullT metadata differ')
    return dict(source=source, assets=assets, capability=evidence, files=files)


def native(code, revision, p, out):
    rt.require(os.environ.get('WR_IMAGE_ID') == IMAGE and os.uname().sysname == 'Linux'
               and {x.name for x in Path('/sys/class/net').iterdir()} == {'lo'}
               and not (ROOT/'weights/cari4d').exists(), 'Offline CPU diagnostic mounts required')
    started = time.monotonic()
    report = dict(schema=p['schema'], stage='automatic_all_person_dwpose_diagnostic', status='fail',
                  producer_revision=revision, image_id=IMAGE, quality_verified=False,
                  anatomical_ownership_verified=False, adoption=False, private_truth_read=False,
                  ground_truth_used=False, hand_labeled_test=False, oracle_modes=[], gpu_used=False,
                  network='none', budget_seconds=BUDGET, banks=[])
    def expired(*_):
        raise TimeoutError('Inclusive CPU diagnostic budget exhausted')
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(BUDGET)
    try:
        before = authenticate(code, revision, p)
        report.update(input_pins=before['files'], capability_evidence=before['capability'])
        dw.dependency_identity()
        import cv2
        calls = []
        with dw.private_prefix(report, lambda: None) as prefix:
            subprocess.run(dw.pip_argv(prefix, ROOT), check=True, capture_output=True,
                           timeout=max(.001, BUDGET-(time.monotonic()-started)))
            ort, imports = dw.import_runtime(prefix)
            path = ROOT/dw.acquisition.BASE/'source/onnxpose.py'
            spec = importlib.util.spec_from_file_location('wr_original_dwpose_bank', path)
            original = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(original)
            options = ort.SessionOptions()
            options.intra_op_num_threads, options.inter_op_num_threads = 4, 1
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            session = ort.InferenceSession(str(ROOT/dw.acquisition.BASE/dw.acquisition.ASSETS[0][0]),
                                           sess_options=options, providers=['CPUExecutionProvider'])
            session.disable_fallback()
            dw.validate_session(session, graph=dw.session_metadata(session))
            dw.validate_options(session, ort)
            proxy = dw.SessionProxy(session, calls, lambda: None)
            for row in p['episodes']:
                cap = cv2.VideoCapture(str(ROOT/row['video']))
                try:
                    rt.require(cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == row['total_frames'],
                               'Original full video frame count required')
                    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    banks = saved_person_banks(rt.strict((ROOT/row['diagnostic']).read_bytes()),
                                               width, height, row['total_frames'], row['episode'])
                    for bank in banks:
                        index = bank['frame_index']
                        rt.require(cap.set(cv2.CAP_PROP_POS_FRAMES, index), 'Original frame seek required')
                        ok, bgr = cap.read()
                        rt.require(ok and int(round(cap.get(cv2.CAP_PROP_POS_FRAMES))) == index+1,
                                   'Exact original frame decode required')
                        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                        result = infer_person_pose_frame(original.inference_pose, proxy, rgb, index,
                                                         bank['person_ids'], bank['boxes'], bank['detector_scores'])
                        name = f'episode_{row["episode"]:06d}_frame_{index:06d}.npz'
                        arrays = {k: getattr(result, k) for k in ('boxes_original_xyxy', 'detector_scores',
                                  'keypoints_original_xy', 'raw_scores', 'native_valid', 'in_original_image')}
                        with (out/name).open('xb') as f:
                            np.savez(f, **arrays)
                        (out/name).chmod(0o444)
                        report['banks'].append(dict(episode=row['episode'], frame_index=index,
                            person_ids=result.person_ids, persons=len(result.person_ids), prediction_file=name,
                            prediction=file_id(out/name), decoded_RGB_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                            keypoints=dw.array_identity(result.keypoints_original_xy), scores=dw.array_identity(result.raw_scores)))
                finally:
                    cap.release()
            rt.require(len(report['banks']) == 6 and len(calls) == sum(x['persons'] for x in report['banks'])
                       and all(x['run_completed'] for x in calls), 'Complete native person census required')
            del proxy, session
            report.update(runtime_imports=imports, native_calls=calls)
        rt.require(before == authenticate(code, revision, p) and not prefix.exists(), 'Original inputs/runtime changed')
        rt.require(time.monotonic()-started < BUDGET, 'Inclusive CPU deadline includes publication')
        report.update(status='pass', source_inputs_assets_rehashed_after=True, source_binding=before['source'],
                      all_recorded_person_banks_retained=True, native_session_count=1)
    except Exception as exc:
        report['error_type'] = type(exc).__name__
        raise
    finally:
        signal.alarm(0)
        report['elapsed_seconds'] = time.monotonic()-started
        write(out/'report.json', report)


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.parse_args()
    code, revision = Path(os.environ['WR_CODE']), os.environ['WR_CODE_REVISION']
    rt.require(re.fullmatch('[0-9a-f]{40}', revision) is not None, 'Committed exact source required')
    p = configuration(code)
    out = ROOT/('results/person-pose-bank-probe-'+revision)
    native(code, revision, p, out)


if __name__ == '__main__':
    main()
