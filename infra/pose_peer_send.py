"""One exact full pose-input archive over authenticated private Azure SSH."""
from __future__ import annotations
import base64
import json
import os
from pathlib import Path
import platform
import re
import stat
import struct
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parent))
import pose_peer_inputs as inputs

HELPERS=('infra/pose_peer_send.py','infra/run_pose_peer_send.sh','infra/pose_peer_inputs.py')


def public_key(value):
    inputs.require(type(value)is str and re.fullmatch(r'ssh-ed25519 [A-Za-z0-9+/]+={0,2}(?: [^\x00-\x1f\x7f]+)?',value),'One independently verified Ed25519 public host key required')
    encoded=value.split(' ')[1];raw=base64.b64decode(encoded,validate=True)
    inputs.require(len(raw)==51 and raw[:19]==struct.pack('>I',11)+b'ssh-ed25519'+struct.pack('>I',32),'Exact Ed25519 public-key wire format required')
    return 'ssh-ed25519 '+encoded


def key_state(path):
    inputs.canonical(path);inputs.canonical(path.parent);s=path.lstat();d=path.parent.lstat()
    inputs.require(stat.S_ISREG(s.st_mode)and s.st_nlink==1 and s.st_uid==0 and s.st_mode&0o777==0o600 and 1<=s.st_size<=10000 and
        stat.S_ISDIR(d.st_mode)and d.st_uid==0 and d.st_mode&0o777==0o700,'Original private root-owned client key metadata required')
    return (s.st_dev,s.st_ino,s.st_mode,s.st_size,s.st_mtime_ns,s.st_ctime_ns,d.st_dev,d.st_ino)


def send(root,code,revision,index,hostkey):
    before=inputs.code_binding(code,revision,{name:inputs.identity(code/name,True)for name in HELPERS})
    pinpath=code/f'configs/pose_peer_{index:06d}_pins.json';pin_identity=inputs.identity(pinpath,True,inputs.MAX_PINS)
    pins=inputs.frozen_pins(inputs.strict_json(pinpath.read_bytes()),index)
    producer=pins['inventory_report'];original=root/'results'/f'pose-peer-inventory-{index:06d}-{producer["producer_revision"]}'
    inputs.require(inputs.identity(original/'report.json',True)=={k:producer[k]for k in('bytes','sha256')},'Independent original inventory receipt required before transport')
    receipt=inputs.strict_json((original/'report.json').read_bytes())
    inputs.require(receipt.get('stage')=='pose_peer_input_inventory'and receipt.get('status')=='pass'and receipt.get('episode_index')==index and
        receipt.get('producer_revision')==producer['producer_revision']and receipt.get('script_sha256')==producer['script_sha256']and
        receipt.get('archive')==pins['archive']and receipt.get('manifest')==pins['manifest'],'Actual completed inventory/archive producer required')
    manifestpath=original/inputs.MANIFEST_NAME
    inputs.require(inputs.identity(manifestpath,True,inputs.MAX_MANIFEST)==pins['manifest'],'Independent sealed Azure manifest required before transport')
    inputs.load_manifest(manifestpath.read_bytes(),pins,index)
    archive=original/'archive.tar';inputs.require(inputs.identity(archive,True,inputs.MAX_ARCHIVE)==pins['archive'],'Exact original full archive required')
    key=root/'transfer'/f'pose-peer-client-{index:06d}'/'client_ed25519';key_before=key_state(key)
    pub=key.with_suffix('.pub');pub_before=inputs.identity(pub,True,1000);public_key(pub.read_text().strip());hostkey=public_key(hostkey)
    out=inputs.canonical(root/'results'/f'pose-peer-send-{index:06d}-{revision}')
    inputs.require(out.parent.is_dir()and not out.exists(),'Fresh sender result required');out.mkdir(mode=0o700);started=time.monotonic()
    known=out/'known_hosts';inputs.write(known,('[10.0.0.9]:2222 '+hostkey+'\n').encode())
    report=dict(stage='pose_peer_send',status='fail',episode_index=index,producer_revision=revision,archive=pins['archive'],
        private_key_bytes_read_or_recorded=False,source_helpers={name:inputs.identity(code/name,True)for name in HELPERS},quality_verified=False)
    try:
        args=['ssh','-F','/dev/null','-T','-p','2222','-i',str(key),'-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes',
            '-o','UserKnownHostsFile='+str(known),'-o','GlobalKnownHostsFile=/dev/null','-o','ConnectTimeout=15','-o','ServerAliveInterval=30','-o','ServerAliveCountMax=3',
            'root@10.0.0.9',f'world-reward-pose-inputs-{index:06d}']
        with archive.open('rb')as stream,(out/'receiver-summary.json').open('xb')as output:
            os.fchmod(output.fileno(),0o400);result=subprocess.run(args,stdin=stream,stdout=output,stderr=subprocess.DEVNULL,
                env={'PATH':'/usr/bin:/bin','HOME':'/nonexistent','LANG':'C.UTF-8'},timeout=1700)
        inputs.require(result.returncode==0 and 1<=(out/'receiver-summary.json').stat().st_size<=4000,'Private full archive transfer failed')
        summary=inputs.strict_json((out/'receiver-summary.json').read_bytes())
        inputs.require(set(summary)=={'stage','status','episode_index','extraction_performed','original_frame_coverage_verified','receipt'}and
            summary['stage']=='pose_peer_receive'and summary['status']=='pass'and summary['episode_index']==index and
            summary['extraction_performed']is True and summary['original_frame_coverage_verified']is True,'Exact full received input verdict required')
        inputs.pin(summary['receipt']);report.update(status='pass',receiver_receipt=summary['receipt'])
    except Exception:report['error']='Private pose peer transport failed; owned evidence retained'
    finally:
        try:
            inputs.require(inputs.identity(archive,True,inputs.MAX_ARCHIVE)==pins['archive']and inputs.identity(manifestpath,True)==pins['manifest']and
                inputs.identity(original/'report.json',True)=={k:producer[k]for k in('bytes','sha256')}and inputs.identity(pinpath,True)==pin_identity and key_state(key)==key_before and
                inputs.identity(pub,True)==pub_before and inputs.code_binding(code,revision,{name:inputs.identity(code/name,True)for name in HELPERS})==before,
                'Immutable source/archive/private-key metadata changed');report['source_rehashed_after']=True
        except Exception:report.update(status='fail',source_rehashed_after=False,error='Post-transfer source/archive/key metadata gate failed')
        report['elapsed_seconds']=time.monotonic()-started;inputs.write(out/'report.json',inputs.digest_json(report))
    return report


def main(argv=None):
    parser=inputs.parser();parser.add_argument('--server-public-key',required=True,action=inputs.Once);args=parser.parse_args(argv)
    root,code,revision=inputs.ROOT,Path(os.environ['WR_CODE']),os.environ['WR_CODE_REVISION']
    inputs.require(platform.system()=='Linux'and code==root/'jobs'/revision/'run_pose_peer_send/code'and Path(__file__)==code/HELPERS[0],'Actual immutable Azure sender namespace required')
    report=send(root,code,revision,args.episode,args.server_public_key);print(json.dumps({k:report[k]for k in('stage','status','episode_index','private_key_bytes_read_or_recorded')}));return 0 if report['status']=='pass'else 1


if __name__=='__main__':
    try:sys.exit(main())
    except Exception:print('Pose peer sender preflight failed',file=sys.stderr);sys.exit(1)
