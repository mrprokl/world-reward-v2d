"""Exact readonly pose-result bytes, not inference or a generic data transporter."""
from __future__ import annotations
import ctypes
import hashlib
import io
import os
from pathlib import Path
import re
import stat
import tarfile

import pose_peer_inputs as original

ROOT=original.ROOT
FILES=('geometry_and_poses.npz','object_fixed_canonical.glb','report.json')
MANIFEST='manifest.json';PARENT='producer-report.json';MAX_METADATA=16384;MAX_ARCHIVE=256_000_000
require=original.require;canonical=original.canonical;identity=original.identity;pin=original.pin;strict=original.strict_json;raw_json=original.digest_json


def source_pins(value,index):
    fields={'schema','episode_index','frames','full_original_indices','ground_truth_used','input_pins','outputs','prediction_algorithm_changed',
        'producer_entrypoint','producer_helpers','producer_report','producer_revision','producer_script_sha256','quality_verified',
        'received_full_input_files_independently_rehashed','received_receipt','source_independently_rehashed','source_vm','tracker_helpers'}
    require(type(value)is dict and set(value)==fields and value['schema']=='world_reward.pose_return.source_pins.v1'and
        type(value['episode_index'])is int and value['episode_index']==original.episode(index)and type(value['frames'])is int and 3<=value['frames']<=3000,'Exact frozen original pose return pins required')
    for key,expected in dict(full_original_indices=True,ground_truth_used=False,prediction_algorithm_changed=False,quality_verified=False,
        received_full_input_files_independently_rehashed=True,source_independently_rehashed=True,source_vm='world-reward-ncc-h100-02',producer_entrypoint='run_pose_peer_run').items():
        require(type(value[key])is type(expected)and value[key]==expected,'Only independently verified unmodified full original pose results allowed')
    revision=value['producer_revision'];require(type(revision)is str and re.fullmatch('[0-9a-f]{40}',revision)and
        type(value['producer_script_sha256'])is str and re.fullmatch('[0-9a-f]{64}',value['producer_script_sha256']),'Full original pose producer identities required')
    require(type(value['outputs'])is dict and set(value['outputs'])==set(FILES),'Exactly3 original native output byte pins required')
    for row in value['outputs'].values():pin(row,maximum=MAX_ARCHIVE)
    pin(value['input_pins'],maximum=MAX_METADATA)
    report=value['producer_report'];require(type(report)is dict and set(report)=={'path','bytes','sha256'}and
        report['path']==f'results/pose-peer-run-{index:06d}-{revision}/report.json','Exact original sealed host report namespace required')
    pin({k:report[k]for k in('bytes','sha256')},maximum=MAX_METADATA)
    expected={'infra/pose_peer_run.py','infra/run_pose_peer_run.sh','infra/pose_peer_inputs.py'}
    require(type(value['producer_helpers'])is dict and set(value['producer_helpers'])==expected and value['producer_helpers']['infra/pose_peer_run.py']['sha256']==value['producer_script_sha256'],'Original pose-only helper pins required')
    for group in('producer_helpers','tracker_helpers'):
        require(type(value[group])is dict and value[group],'Actual numeric/producer helper identities required')
        for name,row in value[group].items():original.safe_name(name);pin(row,True,2_000_000)
    original.tracker_pins(value['tracker_helpers'],'volume'if len(value['tracker_helpers'])>len(original.TRACKER_HELPERS)else'default')
    received=value['received_receipt'];require(type(received)is dict and set(received)=={'schema','episode_index','producer_revision','receipt'}and
        received['schema']=='world_reward.pose_peer.received_pins.v1'and type(received['episode_index'])is int and received['episode_index']==index and
        type(received['producer_revision'])is str and re.fullmatch('[0-9a-f]{40}',received['producer_revision']),'Original full-input receiver receipt pin required')
    pin(received['receipt'],maximum=MAX_METADATA)
    return value


def checked(path,wanted,maximum=MAX_METADATA):
    pin(wanted,maximum=maximum);require(identity(path,True,maximum)==wanted,'Independently frozen readonly bytes required before JSON')
    raw=path.read_bytes();require(len(raw)==wanted['bytes']and hashlib.sha256(raw).hexdigest()==wanted['sha256'],'Pinned original JSON changed before interpretation')
    return strict(raw)


def load_source(code,index):
    path=code/f'configs/pose_return_{index:06d}_source_pins.json';wanted=identity(path,True,MAX_METADATA)
    return source_pins(checked(path,wanted),index),wanted


def binding(code,revision,helpers):
    require(code==ROOT/'jobs'/revision/code.parent.name/'code'and re.fullmatch('[0-9a-f]{40}',revision),'Actual immutable return source namespace required')
    own={name:identity(code/name,True,2_000_000)for name in helpers}
    return original.code_binding(code,revision,own),own


def records(source,parent,native):
    """Only aggregate provenance and original frame indices; never decode NPZ/GLB."""
    index=source['episode_index'];expected=dict(stage='pose_peer_run',status='pass',episode_index=index,producer_revision=source['producer_revision'],
        script_sha256=source['producer_script_sha256'],frames=source['frames'],full_original_indices=True,prediction_algorithm_changed=False,
        model_weights_read=False,ground_truth_used=False,quality_verified=False,source_rehashed_after=True,
        source_helpers=source['producer_helpers'],tracker_helpers=source['tracker_helpers'],pins=source['input_pins'],
        received_receipt=source['received_receipt']['receipt'],output_files=source['outputs'])
    require(type(parent)is dict and all(type(parent.get(k))is type(v)and parent[k]==v for k,v in expected.items()),'Original complete post-sealed full native pose host report required')
    fields=dict(stage='fixed_scale_full_object_pose_initializer',status='pass',episode_index=index,input_track='track_1',ground_truth_used=False,
        hand_labeled_test=False,oracle_modes=[],fixed_shape=True,original_frame_coverage_verified=True,
        script_sha256=source['tracker_helpers']['infra/object_pose_smoke.py']['sha256'],
        geometry_and_poses_sha256=source['outputs']['geometry_and_poses.npz']['sha256'],fixed_canonical_mesh_sha256=source['outputs']['object_fixed_canonical.glb']['sha256'])
    require(type(native)is dict and all(type(native.get(k))is type(v)and native[k]==v for k,v in fields.items()),'Original full video-only fixed geometry pose report required')
    frames=native.get('frames');require(type(frames)is list and len(frames)==source['frames']and
        all(type(row)is dict and type(row.get('frame_index'))is int and row['frame_index']==i for i,row in enumerate(frames)),'Every original native frame must remain in order')


def provenance(root,source):
    index=source['episode_index'];revision=source['producer_revision'];old=canonical(root/'jobs'/revision/'run_pose_peer_run/code')
    old_binding=original.code_binding(old,revision,{**source['producer_helpers'],**source['tracker_helpers']})
    inputpath=old/f'configs/pose_peer_{index:06d}_pins.json';incoming=original.frozen_pins(checked(inputpath,source['input_pins']),index)
    received=source['received_receipt'];receiptpath=original.DEST/f'pose-peer-{index:06d}-{received["producer_revision"]}'/'pose-receipt.json'
    receipt=checked(receiptpath,received['receipt'])
    require(receipt.get('stage')=='pose_peer_receive'and receipt.get('status')=='pass'and receipt.get('episode_index')==index and
        receipt.get('producer_revision')==received['producer_revision']and receipt.get('archive')==incoming['archive']and receipt.get('manifest')==incoming['manifest']and
        receipt.get('original_frame_coverage_verified')is True and receipt.get('extraction_performed')is True,'Actual original full input receiver required')
    folder=canonical(root/f'outputs/episode_{index:06d}/object_pose_full')
    require(folder.is_dir()and folder.stat().st_uid==1000 and folder.stat().st_gid==1000 and {p.name for p in folder.iterdir()}==set(FILES),'Exact original UID1000 native output folder required')
    for name,row in source['outputs'].items():
        path=folder/name;s=path.lstat();require(s.st_uid==1000 and s.st_gid==1000 and s.st_mode&0o777==0o444 and identity(path,True,MAX_ARCHIVE)==row,'Original readonly UID1000 output bytes changed')
    parent=checked(root/source['producer_report']['path'],{k:source['producer_report'][k]for k in('bytes','sha256')})
    native=checked(folder/'report.json',source['outputs']['report.json'],MAX_ARCHIVE);records(source,parent,native)
    require(native.get('input_sha256')==incoming['clip_spec']['video_sha256'],'Original input video/pose lineage differs')
    return old_binding


def transport_pins(value,index):
    require(type(value)is dict and set(value)=={'schema','episode_index','source_pins','manifest','archive','inventory_report'}and
        value['schema']=='world_reward.pose_return.pins.v1'and type(value['episode_index'])is int and value['episode_index']==original.episode(index),'Exact independently frozen return transport pins required')
    for name in('source_pins','manifest'):pin(value[name],maximum=MAX_METADATA)
    pin(value['archive'],maximum=MAX_ARCHIVE)
    row=value['inventory_report'];require(type(row)is dict and set(row)=={'path','bytes','sha256','producer_revision','script_sha256'}and
        type(row['producer_revision'])is str and re.fullmatch('[0-9a-f]{40}',row['producer_revision'])and
        type(row['script_sha256'])is str and re.fullmatch('[0-9a-f]{64}',row['script_sha256'])and
        row['path']==f'results/pose-return-inventory-{index:06d}-{row["producer_revision"]}/report.json','Exact original return inventory producer required')
    pin({k:row[k]for k in('bytes','sha256')},maximum=MAX_METADATA)
    return value


def load_transport(code,index):
    source,wanted=load_source(code,index);path=code/f'configs/pose_return_{index:06d}_pins.json';pinid=identity(path,True,MAX_METADATA)
    pins=transport_pins(checked(path,pinid),index);require(pins['source_pins']==wanted,'Same independently frozen original result source pins required')
    return pins,source,pinid


def archive(root,source,source_id,old_binding,path):
    manifest=dict(schema='world_reward.pose_return.manifest.v1',episode_index=source['episode_index'],source_pins=source,source_pins_identity=source_id,
        original_source_binding=old_binding,files={**source['outputs'],PARENT:{k:source['producer_report'][k]for k in('bytes','sha256')}})
    raw=raw_json(manifest);require(len(raw)<=MAX_METADATA,'Bounded manifest only; dense results stay Azure')
    folder=root/f'outputs/episode_{source["episode_index"]:06d}/object_pose_full'
    with path.open('xb')as stream:
        os.fchmod(stream.fileno(),0o400)
        with tarfile.open(fileobj=stream,mode='w|',format=tarfile.PAX_FORMAT)as tar:
            member=tarfile.TarInfo(MANIFEST);member.size=len(raw);member.mode=0o400;tar.addfile(member,io.BytesIO(raw))
            for name,row in sorted(manifest['files'].items()):
                inputpath=root/source['producer_report']['path']if name==PARENT else folder/name
                require(identity(inputpath,True,MAX_ARCHIVE)==row,'Original bytes changed before archive stream')
                member=tarfile.TarInfo(name);member.size=row['bytes'];member.mode=0o444;member.uid=member.gid=1000
                with inputpath.open('rb')as payload:tar.addfile(member,payload)
        stream.flush();os.fsync(stream.fileno())
    return raw,identity(path,True,MAX_ARCHIVE)


def inspect_archive(path,pins,source):
    require(identity(path,True,MAX_ARCHIVE)==pins['archive'],'Frozen complete archive SHA required before TAR interpretation')
    with tarfile.open(path,'r|')as tar:
        first=next(iter(tar),None);require(first is not None and first.name==MANIFEST and first.isfile()and first.size==pins['manifest']['bytes']and first.size<=MAX_METADATA,'Independently pinned bounded manifest must be first')
        raw=tar.extractfile(first).read();require(len(raw)==pins['manifest']['bytes']and hashlib.sha256(raw).hexdigest()==pins['manifest']['sha256'],'Inner manifest SHA required before JSON')
        value=strict(raw);require(type(value)is dict and set(value)=={'schema','episode_index','source_pins','source_pins_identity','original_source_binding','files'}and
            value['schema']=='world_reward.pose_return.manifest.v1'and type(value['episode_index'])is int and value['episode_index']==source['episode_index']and
            value['source_pins']==source and value['source_pins_identity']==pins['source_pins']and
            value['files']=={**source['outputs'],PARENT:{k:source['producer_report'][k]for k in('bytes','sha256')}},'Exact independently bound return manifest required')
        require(type(value['original_source_binding'])is dict and set(value['original_source_binding'])=={'closure_sha256','markers'},'Original producer source closure proof required')
    return value


def extract(path,manifest,destination):
    require(not canonical(destination).exists(),'Fresh private extraction only; no merge')
    destination.mkdir(mode=0o700);seen=set()
    with tarfile.open(path,'r|')as tar:
        iterator=iter(tar);first=next(iterator,None)
        require(first is not None and first.name==MANIFEST and first.isfile()and tar.extractfile(first).read()==raw_json(manifest),'Exact frozen manifest before output writes')
        for member in iterator:
            require(member.name in manifest['files']and member.name not in seen and member.isfile()and not member.linkname and
                member.size==manifest['files'][member.name]['bytes'],'Only4 exact unique original regular files allowed')
            target=destination/member.name;digest=hashlib.sha256();count=0
            with target.open('xb')as out:
                os.fchmod(out.fileno(),0o400)
                source=tar.extractfile(member)
                for block in iter(lambda:source.read(4*1024*1024),b''):out.write(block);digest.update(block);count+=len(block)
                out.flush();os.fsync(out.fileno())
            require(dict(bytes=count,sha256=digest.hexdigest())==manifest['files'][member.name],'Extracted exact original bytes differ');seen.add(member.name)
    require(seen==set(manifest['files'])and {p.name for p in destination.iterdir()}==seen,'No missing/extra return result files permitted')


def promote(source,target):
    canonical(source);canonical(target);require(not target.exists(),'Original native target must remain absent')
    def sync(path):
        fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
    sync(source)
    rename=ctypes.CDLL(None,use_errno=True).renameat2
    rename.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint];rename.restype=ctypes.c_int
    require(rename(-100,os.fsencode(source),-100,os.fsencode(target),1)==0,'Atomic NOREPLACE pose return promotion failed')
    sync(target.parent)
