"""Azure CPU inventory/archive of exactly one initialized full-video pose input."""
from __future__ import annotations
import json
import os
from pathlib import Path
import platform
import sys
import time

import pose_peer_inputs as inputs

HELPERS = ('infra/pose_peer_inventory.py', 'infra/run_pose_peer_inventory.sh', 'infra/pose_peer_inputs.py')


def inventory(root, code, index, revision, producer_revision, entry, producer_sha):
    pinpath = code / f'configs/pose_peer_{index:06d}_source_pins.json'
    source_identity = inputs.identity(pinpath, True, inputs.MAX_MANIFEST)
    source = inputs.validate_source_pins(inputs.strict_json(pinpath.read_bytes()), index, producer_revision, entry, producer_sha)
    producer = root / 'jobs' / producer_revision / entry / 'code'
    inputs.require(producer.resolve() == producer, 'Exact original producer snapshot required')
    original_binding = inputs.code_binding(producer, producer_revision, source['producer_helpers'])
    binding = inputs.code_binding(code, revision, {**source['tracker_helpers'],
        **{name: inputs.identity(code / name, True, 2_000_000) for name in HELPERS}})
    spec = inputs.provenance(root, index, source)
    if source['mesh_source'] == 'volume':
        from volume_mesh_pin_inventory import verify_pinned_artifacts
        base = f'outputs/episode_{index:06d}'
        obj = inputs.strict_json((root / base / 'object_grounded/report.json').read_bytes())
        verify_pinned_artifacts(root, source['volume_pins'], index, spec['video_sha256'],
            source['reports'][base+'/object_grounded/report.json']['sha256'],
            source['reports'][base+'/scale_smoke/report.json']['sha256'], float(obj['transform']['scale'][0]))
    files = {name: inputs.identity(root / name) for name in sorted(inputs.paths(index, spec['total_frames'], source['mesh_source']))}
    manifest = dict(schema='world_reward.pose_peer.inputs.v1', episode_index=index, clip_spec=spec,
        mesh_source=source['mesh_source'], producer=dict(revision=producer_revision, entrypoint=entry, script_sha256=producer_sha),
        producer_source_binding=original_binding, tracker_helpers=source['tracker_helpers'], source_pins=source, source_pins_identity=source_identity,
        volume_pins=source['volume_pins'], files=files)
    inputs.validate_manifest(manifest, index)
    return manifest, source_identity, original_binding, binding


def main(argv=None):
    parser = inputs.parser()
    parser.add_argument('--producer-revision', required=True, action=inputs.Once)
    parser.add_argument('--producer-entrypoint', choices=inputs.PRODUCERS, required=True, action=inputs.Once)
    parser.add_argument('--producer-script-sha256', required=True, action=inputs.Once)
    args = parser.parse_args(argv)
    root, code, revision = inputs.ROOT, Path(os.environ['WR_CODE']), os.environ['WR_CODE_REVISION']
    inputs.require(platform.system() == 'Linux' and os.environ['WR_ROOT'] == str(root) and
        code == root / 'jobs' / revision / 'run_pose_peer_inventory/code' and Path(__file__) == code/HELPERS[0], 'Actual Azure immutable inventory namespace required')
    started = time.monotonic()
    out = inputs.canonical(root / 'results' / f'pose-peer-inventory-{args.episode:06d}-{revision}')
    inputs.require(out.parent.is_dir() and not out.exists(), 'Fresh inventory result required')
    out.mkdir(mode=0o700)
    report = dict(stage='pose_peer_input_inventory', status='fail', episode_index=args.episode,
        producer_revision=revision, script_sha256=inputs.identity(code/HELPERS[0], True)['sha256'],
        numeric_predictions_changed=False, model_or_label_read=False, quality_verified=False)
    try:
        manifest, source_identity, original_binding, binding = inventory(root, code, args.episode, revision,
            args.producer_revision, args.producer_entrypoint, args.producer_script_sha256)
        report['phase'] = 'archive'; archive_pin = inputs.archive(root, manifest, out/'archive.tar')
        pinpath = code / f'configs/pose_peer_{args.episode:06d}_source_pins.json'
        inputs.require(inputs.identity(pinpath, True) == source_identity and
            inputs.code_binding(code, revision, {**manifest['tracker_helpers'],
                **{name:inputs.identity(code/name,True)for name in HELPERS}}) == binding and
            inputs.code_binding(root/'jobs'/args.producer_revision/args.producer_entrypoint/'code', args.producer_revision,
                manifest['source_pins']['producer_helpers']) == original_binding, 'Source or original producer changed after archiving')
        inputs.write(out / inputs.MANIFEST_NAME, inputs.digest_json(manifest))
        report.update(status='pass', phase='complete', source_helpers={name:inputs.identity(code/name,True)for name in HELPERS},
            archive=archive_pin, manifest=inputs.identity(out/inputs.MANIFEST_NAME), files=len(manifest['files']),
            bytes=sum(row['bytes'] for row in manifest['files'].values()), source_pins=source_identity,
            clip_spec=manifest['clip_spec'], mesh_source=manifest['mesh_source'], tracker_helpers=manifest['tracker_helpers'],
            source_rehashed_after=True)
    except Exception:
        report.update(error='Original full pose input/source/archive gate failed; owned partial evidence retained')
    report['elapsed_seconds'] = time.monotonic()-started
    inputs.write(out/'report.json', inputs.digest_json(report))
    print(json.dumps({key:report.get(key)for key in('stage','status','episode_index','files','bytes','elapsed_seconds')}))
    return 0 if report['status']=='pass' else 1


if __name__ == '__main__':
    try: sys.exit(main())
    except Exception: print('Pose peer inventory preflight failed',file=sys.stderr);sys.exit(1)
