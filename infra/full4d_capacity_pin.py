"""Publish a new input-producer pin in the original full4D experiment.

One explicit continuation may run the metadata-capacity-only input consumer.
Existing surface/pose pins remain unchanged; newly prepared inputs truthfully
identify the continuation source. All later original CARI producers can consume
the new public input pin without model or numerical modifications.
"""
from __future__ import annotations

from dataclasses import asdict
import os
from pathlib import Path
import re

import cari_clip_inputs as inputs
import full4d_pins as publication
from mediapipe_cpu_runtime_verify import source
from surface_pose_report_capacity import verify_capacity_source
from world_reward.artifact_paths import episode_output, output_prefix

ROOT = Path('/srv/scenesmith/world-reward')
ORIGINAL = 'de62258a3f0ca1f12dd0a151c8fe96f0256ea3ba'


def input_pin(root, code, revision, episode, total, *, original_code=None, original_revision=ORIGINAL):
    root, code = publication.canonical(root), publication.canonical(code)
    publication.require(root == ROOT and type(revision) is str and re.fullmatch('[0-9a-f]{40}', revision)
                        and code == root/'jobs'/revision/'run_full4d_resume'/'code',
                        'Exact new immutable continuation input producer required')
    publication.require(output_prefix() == f'experiments/full4d-v1-{ORIGINAL}/outputs',
                        'Only the original interrupted full4D namespace may receive missing input pins')
    # Authenticate actual current and original code, not a receipt's self-claim.
    own = source(root, code, revision, 'run_full4d_resume',
                 ('infra/cari_prepare.py', 'infra/surface_pose_report_capacity.py', 'infra/full4d_capacity_pin.py'))
    original = root/'jobs'/ORIGINAL/'run_full4d_sample'/'code'
    publication.require(original_revision == ORIGINAL and
                        (original_code is None or Path(original_code) == original),
                        'Exact original interrupted experiment source required')
    prior = source(root, original, ORIGINAL, 'run_full4d_sample', ('infra/cari_prepare.py',))
    parity = verify_capacity_source((original/'infra/cari_prepare.py').read_bytes(),
                                   (code/'infra/cari_prepare.py').read_bytes())
    base = episode_output(root, episode)
    spec = inputs.PublicClipSpec(episode, total, 'front_stereo_camera_left', 1152, 1536)
    paths = inputs.relative_paths(spec)
    report, report_id = publication._json(root/paths['input_report'], readonly=False)
    publication._producer(report, code, revision, episode, 'world_reward_native_cari_inputs',
                          'cari_prepare.py', total=total)
    publication.require(report.get('object_source') == 'surface'
                        and report.get('original_frame_coverage_verified') is True,
                        'Complete original-timeline automatic surface input producer required')
    proof = report.get('surface_geometry_validation', {}).get('files', {})
    publication.require(proof.get(str(code/'infra/surface_pose_report_capacity.py')) ==
                        publication.identity(code/'infra/surface_pose_report_capacity.py'),
                        'Actual capacity helper must be source-bound by the new input receipt')
    names = inputs.source_paths(spec, object_source='surface')
    publication.require(len(names) == 15, 'Exact fifteen original public input sources required')
    rows = {name: publication.identity(root/name, readonly=False) for name in sorted(names)}
    publication.require(rows[paths['input_report']] == report_id, 'Input producer changed during pinning')
    pin = dict(schema='world-reward-cari-clip-input-pins-v3', object_source='surface', clip_spec=asdict(spec),
               input_report=report_id | dict(producer_revision=revision, script_sha256=report['script_sha256']),
               source_files=rows)
    inputs.validate_pins(spec, pin)
    inputs.validate_reports(root, spec, pin)
    for name, expected in rows.items():
        publication._seal(root/name, base, expected)
    result = publication._write(code, episode, 'input', pin)
    # Reauthenticate both immutable closures; original receipts/pins were never
    # edited or resealed, and neither reconstruction nor fitting was replayed.
    publication.require(source(root, code, revision, 'run_full4d_resume',
                               ('infra/cari_prepare.py', 'infra/surface_pose_report_capacity.py', 'infra/full4d_capacity_pin.py')) == own
                        and source(root, original, ORIGINAL, 'run_full4d_sample', ('infra/cari_prepare.py',)) == prior
                        and verify_capacity_source((original/'infra/cari_prepare.py').read_bytes(),
                            (code/'infra/cari_prepare.py').read_bytes()) == parity,
                        'Original/current source changed during explicit input publication')
    return result


capacity_input_pin = input_pin
