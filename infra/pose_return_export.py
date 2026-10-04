"""Readonly forced-command export of one independently frozen result archive."""
import os
from pathlib import Path
import re
import sys
import pose_return_inputs as payload

HELPERS=('infra/pose_return_export.py','infra/run_pose_return_export.sh','infra/pose_return_inputs.py','infra/pose_peer_inputs.py')


def authorize(connection,command,index):
    fields=connection.split()if type(connection)is str else[]
    payload.require(len(fields)==4 and fields[0]=='10.0.0.4'and fields[2:]==['10.0.0.9','2222']and
        re.fullmatch('[0-9]{1,5}',fields[1])and 1<=int(fields[1])<=65535 and command==f'world-reward-pose-results-{index:06d}','Only authorized private peer/exact forced pose export command allowed')


def archive_proof(root,code,index):
    pins,source,pinid=payload.load_transport(code,index);spec=pins['inventory_report']
    report=payload.checked(root/spec['path'],{k:spec[k]for k in('bytes','sha256')})
    expected=dict(stage='pose_return_inventory',status='pass',phase='complete',episode_index=index,producer_revision=spec['producer_revision'],
        script_sha256=spec['script_sha256'],archive=pins['archive'],manifest=pins['manifest'],source_pins=pins['source_pins'],
        source_rehashed_after=True,model_or_GPU_used=False,prediction_algorithm_changed=False,quality_verified=False,
        frames=source['frames'],output_files=source['outputs'])
    payload.require(all(type(report.get(k))is type(v)and report[k]==v for k,v in expected.items()),'Actual independently sealed return inventory required')
    old=root/'jobs'/spec['producer_revision']/'run_pose_return_inventory/code'
    helpers=report.get('source_helpers');payload.require(type(helpers)is dict and helpers.get('infra/pose_return_inventory.py',{}).get('sha256')==spec['script_sha256'],'Original inventory source binding required')
    payload.require(payload.original.code_binding(old,spec['producer_revision'],helpers)==report.get('source_binding'),'Original immutable inventory source changed')
    archive=(root/spec['path']).parent/'archive.tar';manifest=payload.inspect_archive(archive,pins,source)
    payload.require(manifest['original_source_binding']==report['original_source_binding'],'Original source ancestry differs from sealed archive')
    return archive,pins,source,pinid


def export(root,code,revision,index,stream):
    before,_=payload.binding(code,revision,HELPERS);archive,pins,source,pinid=archive_proof(root,code,index)
    old=payload.provenance(root,source)
    with archive.open('rb')as incoming:
        for block in iter(lambda:incoming.read(4*1024*1024),b''):stream.write(block)
    stream.flush()
    payload.require(payload.identity(archive,True,payload.MAX_ARCHIVE)==pins['archive']and payload.provenance(root,source)==old and
        payload.binding(code,revision,HELPERS)[0]==before and payload.load_transport(code,index)==(pins,source,pinid),'Source/archive/original results changed during forced export')


def main(argv=None):
    a=payload.original.parser(help=False).parse_args(argv);code=Path(os.environ['WR_CODE']);revision=os.environ['WR_CODE_REVISION']
    payload.require(sys.platform=='linux'and os.geteuid()==0 and os.uname().nodename=='world-reward-ncc-h100-02'and os.environ['WR_ROOT']==str(payload.ROOT)and
        code==payload.ROOT/'jobs'/revision/'run_pose_return_server/code'and Path(__file__)==code/HELPERS[0],'Actual immutable VM02 forced exporter required')
    authorize(os.environ.get('SSH_CONNECTION'),os.environ.get('SSH_ORIGINAL_COMMAND'),a.episode)
    export(payload.ROOT,code,revision,a.episode,sys.stdout.buffer)


if __name__=='__main__':
    try:main()
    except Exception:print('Pose result export failed closed',file=sys.stderr);sys.exit(1)
