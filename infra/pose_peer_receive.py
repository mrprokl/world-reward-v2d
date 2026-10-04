"""Strict per-episode private Azure stream; no keys, model or network client."""
from __future__ import annotations
import json
import os
from pathlib import Path
import platform
import signal
import sys

# -I excludes the script directory; only this immutable literal sibling closure.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import frontend_peer_receive
import pose_peer_inputs as inputs

HELPERS = ('infra/pose_peer_receive.py', 'infra/run_pose_peer_receive.sh', 'infra/pose_peer_inputs.py', 'infra/frontend_peer_receive.py')


def receive(stream, root, code, revision, index):
    pinpath = code / f'configs/pose_peer_{index:06d}_pins.json'
    before = inputs.code_binding(code, revision, {name:inputs.identity(code/name,True)for name in HELPERS})
    pin_identity = inputs.identity(pinpath, True, inputs.MAX_PINS)
    pins = inputs.frozen_pins(inputs.strict_json(pinpath.read_bytes()), index)
    destination = inputs.canonical(inputs.DEST/f'pose-peer-{index:06d}-{revision}')
    inputs.require(destination.parent.is_dir() and not destination.exists(), 'Fresh private receiver namespace required')
    destination.mkdir(mode=0o700)
    transport = frontend_peer_receive.receive(stream, destination, pins['archive']['bytes'], pins['archive']['sha256'])
    inputs.require(transport['status']=='pass' and transport['receipt_written']is True, 'Exact pinned full stream required before extraction')
    output = destination/'inputs'; report = dict(stage='pose_peer_receive',status='fail',episode_index=index,
        producer_revision=revision, archive=pins['archive'], manifest=pins['manifest'], pins=pin_identity, extraction_performed=False,
        original_frame_coverage_verified=False, quality_verified=False, model_or_label_read=False)
    try:
        source = inputs.manifest_from_archive(destination/'archive.tar', pins, index)
        retained = inputs.extract(destination/'archive.tar', source, output)
        inputs.require(inputs.identity(destination/'archive.tar',True,inputs.MAX_ARCHIVE)==pins['archive'] and inputs.identity(pinpath,True)==pin_identity and
            inputs.code_binding(code,revision,{name:inputs.identity(code/name,True)for name in HELPERS})==before, 'Receiver source/archive changed after extraction')
        report.update(status='pass',retained=retained,extraction_performed=True, original_frame_coverage_verified=True,
            source_helpers={name:inputs.identity(code/name,True)for name in HELPERS}, source_rehashed_after=True)
    except Exception:report['error']='Private full pose archive extraction failed; owned partial evidence retained'
    inputs.write(destination/'pose-receipt.json',inputs.digest_json(report))
    return report


def validate_peer(connection, command, index):
    fields=connection.split() if type(connection)is str else []
    inputs.require(len(fields)==4 and fields[0]=='10.0.0.4' and fields[2:]==['10.0.0.9','2222'] and
        fields[1].isdigit() and 1<=int(fields[1])<=65535 and command==f'world-reward-pose-inputs-{index:06d}', 'Exact authorized private peer/forced episode command required')


def main(argv=None):
    args=inputs.parser(help=False).parse_args(argv)
    root,code,revision=inputs.ROOT,Path(os.environ['WR_CODE']),os.environ['WR_CODE_REVISION']
    inputs.require(platform.system()=='Linux'and os.geteuid()==0 and code==root/'jobs'/revision/'run_pose_peer_receive/code'and Path(__file__)==code/HELPERS[0],'Exact Linux root receiver snapshot required')
    validate_peer(os.environ.get('SSH_CONNECTION'),os.environ.get('SSH_ORIGINAL_COMMAND'),args.episode)
    def expired(*_):raise TimeoutError('Fixed private pose transport budget exceeded')
    signal.signal(signal.SIGALRM,expired);signal.alarm(1800)
    try:report=receive(sys.stdin.buffer,root,code,revision,args.episode)
    finally:signal.alarm(0)
    destination=inputs.DEST/f'pose-peer-{args.episode:06d}-{revision}'
    summary={k:report[k]for k in('stage','status','episode_index','extraction_performed','original_frame_coverage_verified')}
    summary['receipt']=inputs.identity(destination/'pose-receipt.json',True)
    print(json.dumps(summary))
    return 0 if report['status']=='pass'else 1


if __name__=='__main__':
    try:sys.exit(main())
    except Exception:print('Pose peer receive gate failed',file=sys.stderr);sys.exit(1)
