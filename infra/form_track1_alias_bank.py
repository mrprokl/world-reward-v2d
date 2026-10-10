"""Azure CPU-only exact RGB exclusion bank; no model or private reference.

The bank rejects exact content duplicates, not every possible cropped/reencoded
or shifted recording alias. The original metadata exclusion remains mandatory.
Only hashes are published between Azure hosts; RGB never leaves Azure.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
import urllib.request

from mediapipe_cpu_runtime_verify import canonical, identity, source, strict, require
from form_hoi_external_dev import decode_frame_hashes, probe
from form_hoi_external_acquire import seal
from full4d_publish import PrivatePreviews, ENDPOINT

ROOT = Path('/srv/scenesmith/world-reward')
BASE = Path('/srv/world-reward-data/track1_rgb_alias_bank_v1')
ENTRY = 'run_form_track1_alias_bank'
MANIFEST = dict(bytes=13081, sha256='3df960ce0f594b8f51675b21bb070925de7aa87a583332674eb89b0e90fc6263')
DATASET = '5f68335f3acc802033d1e80728c1633197521de8'
HELPERS = ('infra/form_track1_alias_bank.py', 'infra/run_form_track1_alias_bank.sh',
           'infra/form_hoi_external_dev.py', 'infra/full4d_publish.py')


def build():
    started = time.monotonic(); deadline = started+600
    code = canonical(os.environ['WR_CODE']); rev = os.environ['WR_CODE_REVISION']
    binding = source(ROOT,code,rev,ENTRY,HELPERS)
    require(BASE.is_dir() and not (BASE/'bank.json').exists(), 'Fresh owned Azure bank required')
    path = ROOT/'results/input-manifest.json'
    require(identity(path,32768,readonly=False)==MANIFEST,'Original Track1 manifest required')
    manifest = strict(path.read_bytes())
    require((manifest['track'],manifest['repo_id'],manifest['revision']) ==
        ('track_1','nvidia/video_to_data_challenge',DATASET),'Track1-only bank required')
    cfg = dict(width=1536,height=1152,fps=30,contiguous_prefix_frames=96)
    rows=[]
    for ep in range(30):
        relative=f'track_1/videos/chunk-000/observation.images.exo_camera/episode_{ep:06d}.mp4'
        found=[r for r in manifest['files'] if r['path']==relative]
        require(len(found)==1,'One independently pinned original video required')
        pin={k:found[0][k]for k in ('bytes','sha256')}; p=ROOT/'data'/relative
        require(identity(p,2<<30,readonly=False)==pin,'Original permitted RGB bytes changed')
        metadata=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
            '-show_entries','stream=nb_frames','-of','json',str(p)],timeout=30))
        total=int(metadata['streams'][0]['nb_frames']);require(total>=96,'Original clip shorter than bank')
        probe(p,cfg,total,deadline)
        hashes=decode_frame_hashes(p,cfg,deadline)
        require(identity(p,2<<30,readonly=False)==pin,'Original video changed during decode')
        rows.append(dict(episode=ep,**pin,frame_rgb_sha256=hashes))
    require(identity(path,32768,readonly=False)==MANIFEST and source(ROOT,code,rev,ENTRY,HELPERS)==binding,
            'Sources changed during RGB bank generation')
    bank=dict(schema='world_reward.track1_rgb_alias_bank.v1',dataset_revision=DATASET,
        input_manifest=MANIFEST,frames_per_video=96,decoded_format='native_rgb24_sha256_no_resize',
        videos=rows,producer_revision=rev,near_alias_absence_verified=False)
    pin=seal(BASE/'bank.json',bank);require(pin['bytes']<=512<<10,'Bounded metadata bank required')
    seal(BASE/'report.json',dict(status='pass',producer_revision=rev,bank=pin,source_binding=binding,
        challenge_videos=30,reference_read=False,models_loaded=False,gpu_used=False,
        elapsed_seconds=time.monotonic()-started,heavy_data_transferred=False))
    print(json.dumps(dict(status='bank_sealed',bank=pin)),flush=True)


class PrivateBank(PrivatePreviews):
    """Separate fixed metadata route; the media publisher's allowlist is unchanged."""
    def request(self,method,name=None,data=None,headers=None,*,container_acl=False):
        if container_acl:return super().request(method,container_acl=True)
        require(method in {'PUT','HEAD','GET'} and re.fullmatch(
            r'research-audit-[0-9a-f]{40}/track1-rgb-alias-bank.json',name or ''),'Only exact bank metadata route')
        try:
            request=urllib.request.Request(ENDPOINT+'/'+name,data=data,method=method,
                headers={'x-ms-version':'2023-11-03',**self.authorization(),**(headers or {})})
            return self.opener.open(request,timeout=45)
        except Exception:raise RuntimeError('Private metadata bank transport failed') from None


def publish():
    rev=os.environ['WR_CODE_REVISION'];code=canonical(os.environ['WR_CODE'])
    binding=source(ROOT,code,rev,ENTRY,HELPERS)
    bank=BASE/'bank.json';pin=identity(bank,512<<10)
    require(strict(bank.read_bytes())['producer_revision']==rev,'Original bank producer required')
    client=PrivateBank();client.require_private()
    row=client.upload(f'research-audit-{rev}/track1-rgb-alias-bank.json',bank.read_bytes(),'application/json',rev)
    client.head(row['name'],pin,row['etag'])
    receipt=dict(status='pass',producer_revision=rev,bank=pin,blob=row,endpoint=ENDPOINT,
        source_binding=binding,private_container_verified=True,account_keys_used=False,public_access_changed=False)
    rp=seal(BASE/'publication.json',receipt)
    print(json.dumps(dict(status='bank_published',bank=pin,publication=rp,blob=row)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('build','publish'))
    (build if parser.parse_args().mode=='build' else publish)()
