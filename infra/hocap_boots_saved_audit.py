"""New saved-only CPU attestation; immutable historical HOSTFAIL stays FAIL.

No producer, model, decoder or evaluator is executed. Static arrays are rebuilt
only in memory to invoke the unchanged native prediction validator.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import time

import mediapipe_cpu_runtime_verify as rt
import hocap_boots_track as track

ROOT=rt.ROOT; ENTRY='run_hocap_boots_saved_audit'; IMAGE=track.boots.IMAGE
REV='9ab14d6e5dd662d8b50b7fee9b0656bd14afea59'
OUT='validation/hocap_boots_saved_audit_v1'; ORIGINAL='validation/hocap_boots_track_v1'
UNIT='world-reward-hocap-boots-track-v1.service'
CID='4b1d37b7965bcf71e690e3028475cb0ea9faaaa70fdb6b1f1c3ffec0bf808a9d'
BUDGET=120; GRACE=60
HELPERS=('infra/hocap_boots_saved_audit.py','infra/run_hocap_boots_saved_audit.sh',*track.HELPERS)
PINS={
 'report.json':dict(bytes=2006,sha256='e8bae07453e0629cf858eeb7fdebc2415554bee3848bddafa1e4ef109e15174d'),
 'host.json':dict(bytes=547,sha256='571adb0a21e0d2ca21a27c240a9623a286e82dac94032c25247aeb1b909abf49'),
 'proof.json':dict(bytes=11004,sha256='25dd2465595284e49446f67153865681937c0b8cfe2819feb3bea6887c21cb41'),
 '.container.cid':dict(bytes=64,sha256='42d90b11f2e82003ede9283b5b385f5234ad219a151cf63dd278c923f5975d90'),
 'native.log':dict(bytes=94,sha256='6ed498efc79c8673f9d5197a40c91bbc9c006f760f7d5645a6a9b518ed5fa00e'),
 '20231027_112303_tracks.npz':dict(bytes=29394484,sha256='f8d1b9f5fdb06e8820982a356e81d59c4ecabfa97317ac40d532fd340ba5cdb4'),
 '20231027_113202_tracks.npz':dict(bytes=30968919,sha256='89302408062f87bd62a52e951e9793a9cabce6d70aec85808442faf16f65a211')}
BANK_PIN=dict(bytes=7495,sha256='710adfbaa52125d489178fcdfc83d4d5bd62af11b45bbfab2e2424207518682d')
require=rt.require


def check(deadline):
    if time.monotonic()>=deadline: raise TimeoutError('Inclusive saved CPU audit deadline exhausted')


def source(code,revision):
    require(Path(__file__).resolve()==code/HELPERS[0],'Actual new audit source required')
    snapshot_parent(code.parent)
    return rt.source(ROOT,code,revision,ENTRY,HELPERS)


def snapshot_parent(path):
    rt.canonical(path)
    # azure_job publishes the source-only envelope as0755. The code tree and
    # both dispatch markers remain independently readonly/authenticated below.
    require(stat.S_IMODE(path.stat().st_mode) in (0o555,0o755)
            and {n.name for n in path.iterdir()}=={'code','revision','source-sha256'},
            'Exact published source-only snapshot parent required')


def saved_inputs(code,deadline):
    """Source/bytes first. No private path is read, not even the extraction receipt."""
    p=track.protocol(code); folder=rt.canonical(ROOT/ORIGINAL)
    require(stat.S_IMODE(folder.stat().st_mode)==0o555 and {v.name for v in folder.iterdir()}==set(PINS),
            'Exact historical failed namespace required')
    stat_fields=('st_dev','st_ino','st_mode','st_size','st_mtime_ns','st_ctime_ns','st_nlink','st_uid','st_gid')
    cid_stat={k:getattr((folder/'.container.cid').lstat(),k) for k in stat_fields}
    require(stat.S_IMODE(cid_stat['st_mode'])==0o644,'Original writable CID mode must remain0644')
    for name,pin in PINS.items():
        check(deadline); require(rt.identity(folder/name,1<<30,readonly=name!='.container.cid')==pin,'Historical original bytes differ')
        if name!='.container.cid':
            require(stat.S_IMODE((folder/name).stat().st_mode)==(0o400 if name=='native.log' else 0o444),'Original sealed artifact mode differs')
    require((folder/'.container.cid').read_bytes()==CID.encode(),'Original exact CID bytes differ')
    native=rt.pinned(folder/'report.json',PINS['report.json']); host=rt.pinned(folder/'host.json',PINS['host.json'])
    track.same_facts(native,dict(schema='world_reward.hocap_boots_tracks.v1',stage='hocap_native_all_seed_full_t_tracks',
        status='pass',phase='complete',producer_revision=REV,image_id=IMAGE,model_loads=1,rgb_decodes=1493,
        native_calls_attempted=2,native_calls_returned=2,native_calls_completed=2,private_labels_read=False,
        ground_truth_used=False,opaque_metadata_read=False,calibration_read=False,challenge_inputs_used=False,
        oracle_modes=[],quality_verified=False,adoption=False,all_original_jpegs_decoded_before_model=True,
        all_source_public_bank_assets_rehashed_after=True,proof_identity=PINS['proof.json'],bank_report_identity=BANK_PIN))
    track.same_facts(host,dict(stage='hocap_boots_host',status='fail',producer_revision=REV,image_id=IMAGE,
        native_exit_status=0,native_report_identity=PINS['report.json'],owned_cleanup_verified=False,
        source_public_bank_assets_rehashed_after=False,post_error_type='ValueError',private_labels_read=False,
        quality_verified=False,adoption=False))
    require(type(native['elapsed_seconds']) in (int,float) and 0<native['elapsed_seconds']<=1800,'Original native budget required')
    proof=rt.pinned(folder/'proof.json',PINS['proof.json']); old=ROOT/'jobs'/REV/track.ENTRY/'code'
    snapshot_parent(old.parent)
    original=rt.source(ROOT,old,REV,track.ENTRY,track.HELPERS)
    require(original==proof['source_binding'] and track.bank.json_digest(original)==native['source_binding_sha256'],
            'Whole historical tracker source/markers differ')
    for name,pin in original['helpers'].items():
        require(rt.identity(code/name,2_000_000)==pin,'Reused numerical/source helper differs from original producer')
    require(proof['protocol_identity']==track.PROTOCOL_PIN and proof['retention_protocol_identity']==p['retention_protocol']
        and proof['public_manifest_identity']==p['public_manifest']==native['public_manifest_identity']
        and native['native_parameters']==p['native'],'Original source/scientific numerical policy differs')
    bank=track.authenticate_bank(p,BANK_PIN,historical=True)
    snapshot_parent(ROOT/'jobs'/bank['producer_revision']/track.bank.ENTRY)
    _,runtime=track.noise.runtime_evidence(code)
    require(proof['bank']==bank and proof['runtime']==runtime,'Complete public bank/runtime/source asset proof differs')
    clips,_=track.bank.public_inputs(Path(p['inputs']),p['public_manifest'],p,deadline)
    require([c['num_frames'] for c in clips]==p['frames'],'Full original RGB hash-only timeline differs')
    rows=native['predictions']; require(type(rows)is list and len(rows)==2,'Both frozen predictions required')
    for row,clip,t,q,m in zip(rows,p['clips'],p['frames'],(1616,1517),(149,138)):
        track.same_facts(row,dict(clip=clip,frames=t,queries=q,seeds=m,zero_query_seeds=0,file=clip+'_tracks.npz',**PINS[clip+'_tracks.npz']))
    check(deadline)
    return p,dict(original_files=PINS,original_source_binding=original,bank=bank,runtime=runtime,
        original_cid_stat=cid_stat,protocol_identity=track.PROTOCOL_PIN,retention_protocol_identity=p['retention_protocol'],
        public_manifest_identity=p['public_manifest']),native


def validate_saved_predictions(p,evidence,native,deadline):
    import numpy as np
    prepared=track.prepare_queries(evidence['bank']['banks'],p,deadline); verified=[]
    for clip,row in zip(prepared,native['predictions']):
        check(deadline); path=ROOT/ORIGINAL/row['file']
        with np.load(path,allow_pickle=False)as z:
            expected={'tracks','tracks_256','occlusion','expected_dist','visible','query_points','point_indices','frame_index',*clip['metadata']}
            require(len(z.files)==len(expected) and set(z.files)==expected,'Exact all-query saved prediction fields required')
            a={n:z[n] for n in z.files}
        for name,value in clip['metadata'].items():
            require(a[name].dtype==value.dtype and a[name].shape==value.shape and np.array_equal(a[name],value),
                    'No original seed/query/birth metadata may change')
        raw={n:a[n] for n in expected-set(clip['metadata'])}
        # Exact unchanged validator expects its diagnostic A controls. Rebuild
        # these in RAM only; never emit static interaction predictions.
        raw['static_tracks']=np.broadcast_to(clip['query_points'][:,[2,1]][:,None],(clip['query_count'],clip['frames'],2)).copy()
        raw['static_visible']=np.ones((clip['query_count'],clip['frames']),bool)
        spec={k:clip[k] for k in ('query_count','frames','width','height')}; spec['point_indices']=clip['point_indices'].tolist()
        track.boots.validate_prediction(raw,spec,clip['query_points'],clip['point_indices']); check(deadline)
        verified.append(dict(clip=clip['clip'],frames=clip['frames'],queries=clip['query_count'],
            seeds=len(clip['metadata']['seed_ids']),file=row['file'],**PINS[row['file']]))
    return verified


def command(argv,seconds=5):
    result=subprocess.run(argv,capture_output=True,timeout=seconds,check=False)
    require(len(result.stdout)<=32768 and len(result.stderr)<=32768,'Bounded host control output required')
    return result


def absent_result(r,cid):
    errors=tuple(prefix+f'{word}: {message}: {cid}\n'.encode()
        for prefix in (b'',b'\n')for word,message in (('Error','No such object'),('Error','No such container'),
                                                    ('error','no such object'),('error','no such container')))
    return r.returncode==1 and r.stdout in (b'',b'\n') and r.stderr in errors


def absence(cid,name):
    r=command(['docker','inspect',cid])
    require(absent_result(r,cid),'Exact absent CID required; daemon failure is not absence')
    for expression in ('id='+cid,'name=^/'+name+'$'):
        r=command(['docker','ps','-aq','--no-trunc','--filter',expression])
        require(r.returncode==0 and r.stdout==b'' and r.stderr==b'','Independent CID/name absence required')
    return True


def host_environment():
    r=command(['systemctl','show',UNIT,'--property=LoadState,ActiveState,SubState,Result,ExecMainStatus'])
    require(r.returncode==0 and r.stderr==b'','Original terminal unit status required')
    rows=r.stdout.decode().splitlines(); require(len(rows)==5 and len(set(rows))==5,'Exact unit projection required')
    state=dict(line.split('=',1)for line in rows)
    require(state==dict(LoadState='loaded',ActiveState='failed',SubState='failed',Result='exit-code',ExecMainStatus='1'),
            'Original unit must remain loaded FAILED, never reset or relabelled')
    absence(CID,'world-reward-hocap-boots-'+REV[:12])
    r=command(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'])
    require(r.returncode==0 and r.stdout.strip()==b'' and r.stderr==b'','Independent GPU-idle census required; no GPU call')
    return dict(original_unit=UNIT,original_unit_state=state,original_cid=CID,original_cid_absent=True,
                original_container_name_absent=True,gpu_idle=True)


def native(code,revision,out,deadline):
    started=time.monotonic(); result=dict(schema='world_reward.hocap_boots_saved_native_audit.v1',stage='hocap_boots_saved_native_audit',
        status='fail',phase='source_authentication',producer_revision=revision,image_id=IMAGE,
        historical_producer_revision=REV,historical_host_status='fail',historical_native_status='pass',
        original_files=PINS,budget_seconds=BUDGET,models_loaded=0,native_tracker_calls=0,rgb_decodes=0,
        private_labels_read=False,opaque_metadata_read=False,ground_truth_used=False,gpu_used=False,
        association_verified=False,quality_verified=False,adoption=False,originals_rehashed_after=False)
    failure=None;before=None;sb=None
    try:
        require(os.environ.get('WR_IMAGE_ID')==IMAGE and os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
            and {n.name for n in Path('/sys/class/net').iterdir()}=={'lo'} and {n.name for n in out.iterdir()}=={'container.cid'},
            'Fresh offline CPU-only audit container required')
        sb=source(code,revision);p,before,old=saved_inputs(code,deadline)
        result.update(phase='saved_arrays',source_binding=sb,original_proof=before)
        result['predictions']=validate_saved_predictions(p,before,old,deadline)
        require(not any(n in sys.modules for n in ('torch','tapnet','sam2','mediapipe')),'No model runtime imported')
        require(saved_inputs(code,deadline)[1]==before and source(code,revision)==sb,'Full original/current source/public/assets changed')
        result.update(status='pass',phase='complete',independent_saved_native_audit=True,originals_rehashed_after=True)
        check(deadline)
    except Exception as error:
        failure=error;result.update(status='fail',error_type=type(error).__name__,error=str(error)[:400])
    finally:
        signal.alarm(GRACE)
        try:
            if before is not None:
                require(saved_inputs(code,time.monotonic()+GRACE)[1]==before and source(code,revision)==sb,
                        'Original/current source/public/assets changed on failure path')
                result['originals_rehashed_after']=True
            if failure is None:check(deadline)
        except Exception as error:
            failure=failure or error;result.update(status='fail',post_error_type=type(error).__name__,post_error=str(error)[:400])
        result['elapsed_seconds']=time.monotonic()-started
        track.write_json(out/'native.json',result,deadline=None if failure else deadline);signal.alarm(0)
    require(failure is None,'Saved CPU array audit failed; original outputs remain unchanged')
    return result


def dispatch(code,revision,deadline):
    import fcntl
    started=time.monotonic();sb=source(code,revision);p,before,_=saved_inputs(code,deadline)
    lock=rt.canonical(ROOT/'jobs/.world-reward-h100.lock');st=lock.stat()
    require(stat.S_ISREG(st.st_mode) and st.st_nlink==1,'Existing read-only cooperative lock required')
    fd=os.open(lock,os.O_RDONLY|os.O_NOFOLLOW);owned=False;failure=None;owner=None;name='world-reward-hocap-boots-saved-'+revision[:12]
    out=rt.canonical(ROOT/OUT); require(not out.exists() and out.parent.is_dir(),'Fresh saved audit output required')
    report=dict(schema='world_reward.hocap_boots_saved_audit.v1',stage='hocap_boots_saved_host_audit',status='fail',phase='host_preflight',
        producer_revision=revision,image_id=IMAGE,historical_host_status='fail',historical_native_status='pass',
        historical_producer_revision=REV,original_files=PINS,source_binding=sb,budget_seconds=BUDGET,
        owned_cleanup_verified=False,originals_rehashed_after=False,private_labels_read=False,gpu_used=False,
        models_loaded=0,native_tracker_calls=0,rgb_decodes=0,quality_verified=False,association_verified=False,adoption=False)
    try:
        require((os.fstat(fd).st_dev,os.fstat(fd).st_ino)==(st.st_dev,st.st_ino),'Original lock inode changed')
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);report['terminal_census_before']=host_environment()
        check(deadline);img=command(['docker','image','inspect',IMAGE,'--format','{{.Id}}'])
        require(img.returncode==0 and img.stdout== (IMAGE+'\n').encode() and img.stderr==b'','Exact existing qualified CPU image required')
        original=ROOT/'jobs'/REV/track.ENTRY
        amg=ROOT/'jobs'/before['bank']['producer_revision']/track.bank.ENTRY
        rp=track.noise.native.strict_json((code/'configs/robotap_boots_inference_pins.json').read_bytes())['runtime']
        assets=ROOT/track.noise.ORIGINAL/'assets';protocol=track.noise.acquisition.read_protocol(code/'configs/robotap_boots_protocol.json')
        paths=[code.parent,original,amg,ROOT/ORIGINAL,Path(p['bank']),Path(p['inputs']),ROOT/rp['report_path']]
        paths += [assets/'tapnet_source'/n for n in before['runtime']['native_sources']]
        paths.append(assets/protocol['checkpoint']['file'])
        require(len(paths)==len(set(paths)) and all(','not in str(n) and '\n'not in str(n)
            and not any(v in n.parts for v in ('metadata_private','eval_private','data','vendor'))for n in paths),'Narrow public source/runtime-only mounts required')
        out.mkdir(mode=0o700);owned=True;owner=(out.stat().st_dev,out.stat().st_ino,out.stat().st_uid)
        cmd=['docker','run','--rm','--name',name,'--cidfile',str(out/'container.cid'),'--label','world_reward.saved_audit.owner='+revision,
            '--network','none','--read-only','--user','0:0','--cap-drop','ALL','--security-opt','no-new-privileges',
            '--memory','8g','--cpus','4','--tmpfs','/tmp:rw,noexec,nosuid,size=128m','--entrypoint','/usr/bin/env']
        for path in paths:cmd+=['--mount',f'type=bind,src={path},dst={path},readonly']
        cmd+=['--mount',f'type=bind,src={out},dst={out}',IMAGE,'-i','PATH=/opt/conda/bin:/usr/bin:/bin','HOME=/tmp',
            'PYTHONPATH='+str(code/'infra'),'PYTHONDONTWRITEBYTECODE=1','CUDA_VISIBLE_DEVICES=-1','WR_IMAGE_ID='+IMAGE,
            'WR_ROOT='+str(ROOT),'WR_CODE='+str(code),'WR_CODE_REVISION='+revision,'WR_SAVED_AUDIT_DEADLINE='+format(deadline,'.17g'),
            'OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1','/opt/conda/bin/python','-B',str(code/HELPERS[0]),'--native']
        process=subprocess.run(cmd,capture_output=True,timeout=max(.1,deadline-time.monotonic()),check=False)
        report['native_exit_status']=process.returncode
        if process.returncode:report['native_stderr_tail']=process.stderr[-1000:].decode('utf-8','replace')
        require(process.returncode==0 and len(process.stdout)<=8192 and len(process.stderr)<=32768,'Saved-only native CPU audit failed')
        native_pin=rt.identity(out/'native.json',2<<20);r=rt.pinned(out/'native.json',native_pin)
        track.same_facts(r,dict(schema='world_reward.hocap_boots_saved_native_audit.v1',stage='hocap_boots_saved_native_audit',
            status='pass',phase='complete',producer_revision=revision,image_id=IMAGE,source_binding=sb,original_proof=before,
            historical_producer_revision=REV,historical_host_status='fail',historical_native_status='pass',original_files=PINS,
            originals_rehashed_after=True,independent_saved_native_audit=True,models_loaded=0,native_tracker_calls=0,rgb_decodes=0,
            private_labels_read=False,opaque_metadata_read=False,ground_truth_used=False,gpu_used=False,
            quality_verified=False,association_verified=False,adoption=False))
        report['native_report_identity']=native_pin;check(deadline)
    except Exception as error:
        failure=error;report.update(error_type=type(error).__name__,error=str(error)[:400])
    finally:
        signal.alarm(GRACE)
        try:
            if owned:
                require(rt.canonical(out)==out and (out.stat().st_dev,out.stat().st_ino,out.stat().st_uid)==owner,
                        'Foreign/replaced namespace not cleaned')
                cidfile=rt.canonical(out/'container.cid');rt.identity(cidfile,65,readonly=False)
                require(cidfile.stat().st_uid==0,'Owned root audit CID required')
                raw=cidfile.read_bytes();require(re.fullmatch(b'[0-9a-f]{64}\n?',raw),'Owned audit CID required')
                cid=raw.decode().rstrip('\n');inspection=command(['docker','inspect',cid,'--format','{{.Image}}|{{.Name}}|{{index .Config.Labels "world_reward.saved_audit.owner"}}'])
                if inspection.returncode==0:
                    require(inspection.stderr==b'' and inspection.stdout==(IMAGE+'|/'+name+'|'+revision+'\n').encode(),'Foreign container not removed')
                    removed=command(['docker','rm','-f',cid],15);require(removed.returncode==0 and removed.stdout==(cid+'\n').encode(),'Owned removal failed')
                else:require(absent_result(inspection,cid),'Owned audit inspect failed, not absence')
                absence(cid,name);(out/'container.cid').chmod(0o444);report['owned_cleanup_verified']=True
            require(saved_inputs(code,time.monotonic()+GRACE)[1]==before and source(code,revision)==sb
                and (lock.stat().st_dev,lock.stat().st_ino)==(st.st_dev,st.st_ino),'Full saved source/public/assets/lock changed')
            report['terminal_census_after']=host_environment();report['originals_rehashed_after']=True
            if failure is None:
                require(rt.identity(out/'native.json',2<<20)==report['native_report_identity'],'Native audit changed')
                require({n.name for n in out.iterdir()}=={'container.cid','native.json'},'Exact independent audit output required')
                check(deadline)
        except Exception as error:
            failure=failure or error;report.update(post_error_type=type(error).__name__,post_error=str(error)[:400])
        finally:
            os.close(fd)
            if owned:
                require(rt.canonical(out)==out and (out.stat().st_dev,out.stat().st_ino,out.stat().st_uid)==owner,
                        'Foreign/replaced namespace not published')
                report.update(status='fail'if failure else 'pass',phase='failed'if failure else 'complete',elapsed_seconds=time.monotonic()-started)
                track.write_json(out/'report.json',report,deadline=None if failure else deadline,seal_parent=True)
            signal.alarm(0)
    require(failure is None,'New saved attestation failed; historical HOSTFAIL never modified');return report


def main(argv=None):
    ap=argparse.ArgumentParser(allow_abbrev=False);m=ap.add_mutually_exclusive_group(required=True)
    m.add_argument('--dispatch',action='store_true');m.add_argument('--native',action='store_true');args=ap.parse_args(argv)
    code=Path(os.environ['WR_CODE']);revision=os.environ['WR_CODE_REVISION']
    require(sys.platform=='linux'and os.geteuid()==0 and os.environ['WR_ROOT']==str(ROOT),'Owned immutable Linux CPU caller required')
    start=time.monotonic();deadline=float(os.environ['WR_SAVED_AUDIT_DEADLINE'])if args.native else start+BUDGET
    def expired(*_):raise TimeoutError('Inclusive saved CPU audit expired')
    signal.signal(signal.SIGALRM,expired);signal.signal(signal.SIGTERM,expired);signal.alarm(max(1,math.ceil(deadline-start)))
    if args.dispatch:require(os.uname().nodename=='world-reward-ncc-h100-02','Original VM02 saved audit host required')
    try:r=native(code,revision,rt.canonical(ROOT/OUT),deadline)if args.native else dispatch(code,revision,deadline)
    finally:signal.alarm(0)
    print(json.dumps(dict(stage=r['stage'],status=r['status'],historical_host_status='fail',quality_verified=False)),flush=True)


if __name__=='__main__':
    try:main()
    except Exception as e:print(json.dumps(dict(status='fail',error_type=type(e).__name__)),flush=True);raise SystemExit(1)from None
