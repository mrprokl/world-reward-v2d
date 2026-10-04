"""Azure CPU-only exact original pose archive; metadata summary only on stdout."""
import os
from pathlib import Path
import sys
import time
import pose_return_inputs as payload

HELPERS=('infra/pose_return_inventory.py','infra/run_pose_return_inventory.sh','infra/pose_return_inputs.py','infra/pose_peer_inputs.py')


def inventory(root,code,revision,index):
    before,helpers=payload.binding(code,revision,HELPERS);source,sourceid=payload.load_source(code,index)
    old=payload.provenance(root,source)
    out=payload.canonical(root/'results'/f'pose-return-inventory-{index:06d}-{revision}')
    payload.require(out.parent.is_dir()and not out.exists(),'Fresh owned archive namespace required')
    out.mkdir(mode=0o700);started=time.monotonic()
    report=dict(stage='pose_return_inventory',status='fail',episode_index=index,producer_revision=revision,script_sha256=helpers[HELPERS[0]]['sha256'],
        source_pins=sourceid,model_or_GPU_used=False,prediction_algorithm_changed=False,quality_verified=False)
    try:
        raw,archive=payload.archive(root,source,sourceid,old,out/'archive.tar')
        payload.original.write(out/payload.MANIFEST,raw)
        payload.require(payload.provenance(root,source)==old and payload.load_source(code,index)==(source,sourceid)and
            payload.binding(code,revision,HELPERS)[0]==before,'Original result/source changed after archive')
        report.update(status='pass',phase='complete',archive=archive,manifest=payload.identity(out/payload.MANIFEST,True,payload.MAX_METADATA),
            frames=source['frames'],output_files=source['outputs'],original_source_binding=old,source_helpers=helpers,
            source_binding=before,source_rehashed_after=True)
    except Exception:report['error']='Exact pose return inventory failed closed; owned evidence retained'
    report['elapsed_seconds']=time.monotonic()-started
    if report['elapsed_seconds']>300:report.update(status='fail',error='Frozen300s CPU inventory budget exceeded')
    payload.original.write(out/'report.json',payload.raw_json(report))
    return report


def main(argv=None):
    a=payload.original.parser().parse_args(argv);code=Path(os.environ['WR_CODE']);revision=os.environ['WR_CODE_REVISION']
    payload.require(sys.platform=='linux'and os.geteuid()==0 and os.uname().nodename=='world-reward-ncc-h100-02'and
        os.environ['WR_ROOT']==str(payload.ROOT)and code==payload.ROOT/'jobs'/revision/'run_pose_return_inventory/code'and Path(__file__)==code/HELPERS[0],'Actual immutable VM02 CPU inventory required')
    report=inventory(payload.ROOT,code,revision,a.episode)
    print(payload.raw_json({k:report.get(k)for k in('stage','status','episode_index','frames','elapsed_seconds','archive','manifest')}).decode(),end='')
    return 0 if report['status']=='pass'else 1


if __name__=='__main__':
    try:sys.exit(main())
    except Exception:print('Pose return inventory preflight failed closed',file=sys.stderr);sys.exit(1)
