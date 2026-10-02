"""One source-volume-constrained mesh proposal; no pose/scale fitting/adoption."""
import argparse
import json
import os
from pathlib import Path
import platform
import signal
import time

import object_budget_guarded as proposal
import volume_mesh_gate as volume
from mesh_link_gate import _write
from world_reward.data import sha256

STAGE='world_reward_cpu_volume_constrained_object_mesh'


def prerequisites(root,episode):
    inputs,sources,scale=proposal.endpoint.prerequisites(root,episode)
    build=volume.validate_build(root)
    p=proposal.endpoint.regular(root,root/'validation/volume_qem_v1/report.json');control=json.loads(p.read_text())
    if (control.get('stage')!=volume.STAGE or control.get('status')!='pass'
            or control.get('script_sha256')!=sha256(Path(volume.__file__))
            or control.get('shared_geometry_helper_sha256')!=sha256(Path(proposal.guarded.__file__))
            or control.get('build')!=build or control.get('target_faces')!=4096 or control.get('target_vertices')!=4096
            or control.get('challenge_inputs_used') is not False or control.get('adoption_performed') is not False
            or [x.get('fixture') for x in control.get('fixtures',[])]!=volume.FIXTURE_NAMES
            or any(x.get('independent_intersecting_faces')!=0 or x.get('source_arrays_unchanged') is not True for x in control['fixtures'])
            or any(x.get('containment',{}).get('all_cavities_contained') is not True for x in control['fixtures'])):
        raise ValueError('Require complete new source-bound volume controls before proposal')
    return inputs,sources,scale,{'volume_gate_sha256':sha256(p),**build}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--episode',type=int,choices=(0,),required=True);args=parser.parse_args(argv)
    if platform.system()!='Linux' or {p.name for p in Path('/sys/class/net').iterdir()}!={'lo'}:
        raise RuntimeError('Require remote CPU network-none')
    root=Path(os.environ['WR_ROOT']);out=root/f'outputs/episode_{args.episode:06d}/object_budget_volume';path=out/'report.json'
    if out.is_symlink() or not out.is_dir() or any(out.iterdir()):raise FileExistsError('Require reserved fresh volume-constrained proposal')
    report={'stage':STAGE,'status':'fail','episode_index':args.episode,'producer_revision':os.environ['WR_CODE_REVISION'],
            'image_id':os.environ['WR_IMAGE_ID'],'script_sha256':sha256(Path(__file__)),
            'input_track':'track_1','ground_truth_used':False,'hand_labeled_test':False,'oracle_modes':[],
            'adoption_performed':False,'challenge_performance_verified':False,'budget_seconds':900,
            'target_faces':4096,'target_vertices':4096,'source_shell_volume_relative_limit':.05,
            'components_deleted':False,'holes_filled':False,'normals_repaired':False,'frame_poses_changed':False,
            'native_cost_and_placement_unchanged':True,'source_embedding_exact_universal_proof':False}
    start=time.perf_counter();signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Frozen900s proposal deadline')));signal.alarm(900)
    try:
        _write(path,report)
        proposal.produce(root,args.episode,report,path,check_prerequisites=prerequisites,simplify=volume.simplify)
    except Exception as error:report.update(error_type=type(error).__name__,error=str(error));raise
    finally:signal.alarm(0);report['elapsed_seconds']=time.perf_counter()-start;_write(path,report)
    print(json.dumps({k:report[k] for k in ('stage','status','episode_index','elapsed_seconds')}))

if __name__=='__main__':main()
