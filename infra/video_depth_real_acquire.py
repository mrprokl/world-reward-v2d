"""Azure-only contiguous original TUM RGB-D subset; values never decoded here."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import time
from urllib.request import Request, urlopen

ROOT=Path('/srv/scenesmith/world-reward')
CONFIG='configs/video_depth_real_v1.json'


def pin(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return dict(bytes=path.stat().st_size,sha256=h.hexdigest())


def safe_member(member,sequence):
    name=member.name.removeprefix('./').removesuffix('/')
    p=PurePosixPath(name)
    if (p.is_absolute() or str(p)!=name or '..' in p.parts or not p.parts
            or p.parts[0]!=sequence or not(member.isfile() or member.isdir())
            or not 0<=member.size<=4<<20):raise ValueError('Unsafe original archive member')
    return name


def run():
    start=time.monotonic();code=Path(os.environ['WR_CODE']);rev=os.environ['WR_CODE_REVISION']
    if code!=ROOT/'jobs'/rev/'run_video_depth_real_acquire/code':raise ValueError('Immutable Azure namespace required')
    cfg=json.loads((code/CONFIG).read_text());base=ROOT/cfg['namespace']
    import mediapipe_cpu_runtime_verify as provenance
    binding=provenance.source(ROOT,code,rev,'run_video_depth_real_acquire',
        ('infra/video_depth_real_acquire.py','infra/run_video_depth_real_acquire.sh',CONFIG))
    if cfg['challenge_inputs_allowed'] or base.exists():raise ValueError('Fresh non-challenge acquisition only')
    base.mkdir();public=base/'inputs';private=base/'eval_private';tmp=base/'staging'
    for p in [public,private,tmp]:p.mkdir(mode=0o700)
    report=dict(status='fail',producer_revision=rev,config_pin=pin(code/CONFIG),depth_values_decoded=False,
        challenge_inputs_used=False,private_values_select_settings=False,archives=[],public_files={},private_files={})
    try:
        # Verify primary terms, not a third-party mirror's licence inference.
        import tum_rgbd_depth_acquire as prior
        raw=prior.text_request(cfg['primary'],300000)
        section=prior.license_section(raw,cfg['license_section'])
        (base/'attribution.txt').write_bytes(section+b'\nSturm et al., TUM RGB-D, IROS 2012; CC-BY-4.0; original PNG bytes unchanged.\n')
        descriptions=prior.text_request('https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download',750000)
        for sid,seq in enumerate(cfg['sequences']):
            prior.sequence_description(descriptions,seq)
            archive=tmp/(seq['name']+'.tgz')
            record=prior.download_archive(seq['archive'],archive,publisher_redirect_v2=True)
            wanted={}
            for row in seq['frames']:
                for role in ['rgb','depth']:
                    if row[role] is not None:wanted[row[role]['path']]=(role,row[role])
            names={'rgb':[],'depth':[]};seen=set();retained={}
            with tarfile.open(archive,mode='r|gz') as stream:
                for m in stream:
                    name=safe_member(m,seq['name'])
                    if name in seen:raise ValueError('Duplicated archive member')
                    seen.add(name)
                    for role in names:
                        if name.startswith(seq['name']+'/'+role+'/') and name.endswith('.png'):names[role].append(name)
                    if m.isfile() and PurePosixPath(name).name.lower() in ('license','license.txt','copying','copying.txt'):
                        prior.archive_terms(stream.extractfile(m).read())
                    if name not in wanted:continue
                    role,expected=wanted[name];raw=stream.extractfile(m).read()
                    if {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}!={k:expected[k] for k in ['bytes','sha256']}:
                        raise ValueError('Original PNG differs from independent HF LFS pin')
                    prior.png_header(raw,role)
                    dest=(public if role=='rgb' else private)/f'{sid:02d}_{PurePosixPath(name).name}'
                    with dest.open('xb') as f:f.write(raw)
                    dest.chmod(0o444 if role=='rgb' else 0o400);retained[name]=dest.name
            if any(len(names[k])!=seq['count_'+k] for k in names):raise ValueError('Original filename count differs')
            if sorted(names['rgb'])[200:296]!=[r['rgb']['path'] for r in seq['frames']]:raise ValueError('Original contiguous ranks differ')
            if set(retained)!=set(wanted):raise ValueError('Incomplete original subset')
            report['archives'].append(dict(sequence=seq['name'],observed_original_archive=record,
                original_archive_hash_independently_known=False))
            archive.unlink()
        for folder,key in [(public,'public_files'),(private,'private_files')]:
            report[key]={p.name:pin(p) for p in sorted(folder.iterdir())}
        manifest=dict(schema='world_reward.video_depth_public.v1',config_pin=report['config_pin'],sequences=[])
        for sid,s in enumerate(cfg['sequences']):
            manifest['sequences'].append(dict(name=s['name'],frames=[dict(index=i,
                file=f'{sid:02d}_{PurePosixPath(r["rgb"]["path"]).name}',timestamp=PurePosixPath(r['rgb']['path']).stem,
                **{k:r['rgb'][k] for k in ['bytes','sha256']}) for i,r in enumerate(s['frames'])]))
        (public/'manifest.json').write_text(json.dumps(manifest,sort_keys=True)+'\n');(public/'manifest.json').chmod(0o444)
        report['public_files']['manifest.json']=pin(public/'manifest.json')
        if provenance.source(ROOT,code,rev,'run_video_depth_real_acquire',
                ('infra/video_depth_real_acquire.py','infra/run_video_depth_real_acquire.sh',CONFIG))!=binding:
            raise ValueError('Immutable producer changed')
        report['source_binding']=binding;report['status']='pass'
    except Exception as exc:
        report.update(error_type=type(exc).__name__,error=str(exc)[:300])
        for p in (public,private):
            for child in p.iterdir():child.unlink()
    finally:
        shutil.rmtree(tmp);report['elapsed_seconds']=time.monotonic()-start
        (base/'acquisition-report.json').write_text(json.dumps(report,sort_keys=True)+'\n')
        (base/'acquisition-report.json').chmod(0o444)
    print(json.dumps({k:report.get(k) for k in ['status','error_type','error','elapsed_seconds']},sort_keys=True))
    if report['status']!='pass':raise SystemExit(1)


if __name__=='__main__':run()
