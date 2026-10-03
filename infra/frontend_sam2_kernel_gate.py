"""Tiny new SAM2 CUDA operator gate; authenticate CPU build before GPU imports.

No images, weights, learning, challenge records or historical runtime parity.
CPU reference is independent 8-neighbour BFS; labels are compared by partition,
not by numeric component IDs. All six procedural inputs remain 16x16.
"""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time
import warnings

ROOT=Path('/srv/scenesmith/world-reward')
TARGET='world-reward/frontend-grounding-v6:0.1'
BASE='sha256:7ebfff18ba3b76dd919485c19115597d7531dfd3233f69461f1dce3f28a6c6d3'
REPORT_PATH=ROOT/'results/frontend-grounding-build-v6/report.json'
PROBE_PATH=ROOT/'results/frontend-grounding-build-v6/child-CPU-probe.log'
OUTPUT=ROOT/'results/frontend-sam2-kernel-gate-v1'
CONFIG='configs/frontend_grounding_source_pins.json'
BUILD_HELPERS=('infra/frontend_grounding_build.py','infra/run_frontend_grounding_build.sh',CONFIG)

def require(value,message):
 if not value:raise ValueError(message)

def canonical(path):
 path=Path(path);require(path.is_absolute()and path.resolve()==path and not any(p.is_symlink()for p in(path,*path.parents)),'Canonical gate source/control path required');return path

def read(path,maximum=200000,readonly=True):
 path=canonical(path);s=path.lstat()
 require(stat.S_ISREG(s.st_mode)and s.st_nlink==1 and 0<s.st_size<=maximum and(not readonly or not s.st_mode&0o222),'Bounded readonly regular gate source/control required')
 raw=path.read_bytes();a=path.lstat()
 require((s.st_dev,s.st_ino,s.st_mode,s.st_size,s.st_mtime_ns,s.st_ctime_ns)==(a.st_dev,a.st_ino,a.st_mode,a.st_size,a.st_mtime_ns,a.st_ctime_ns),'Gate source/control changed during read')
 return raw,{'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}

def strict_json(raw):
 def pairs(rows):
  result={}
  for key,value in rows:require(key not in result,'Duplicate gate JSON field');result[key]=value
  return result
 def invalid(_):raise ValueError('Nonfinite gate JSON')
 return json.loads(raw,object_pairs_hook=pairs,parse_constant=invalid)

def closure(code,revision,entry,helpers):
 code=canonical(code);require(code==ROOT/'jobs'/revision/entry/'code','Exact original dispatch source namespace required')
 require(re.fullmatch('[0-9a-f]{40}',revision),'Exact dispatch revision required');markers={}
 for name in('revision','source-sha256'):
  raw,_=read(code.parent/name,100,False);require(raw==(revision+'\n').encode()if name=='revision'else bool(re.fullmatch(b'[0-9a-f]{64}\n',raw)),'Exact original source dispatch markers required');markers[name]=raw.decode()
 digest=hashlib.sha256()
 for path in(code,*sorted(code.rglob('*'))):
  canonical(path);s=path.lstat();require(not s.st_mode&0o222 and(stat.S_ISREG(s.st_mode)or stat.S_ISDIR(s.st_mode)),'Complete frozen gate/build source closure required')
  if stat.S_ISREG(s.st_mode):_,pin=read(path,2_000_000);digest.update(str(path.relative_to(code)).encode()+b'\0'+bytes.fromhex(pin['sha256']))
 return {'markers':markers,'closure_sha256':digest.hexdigest(),'helpers':{n:read(code/n,2_000_000)[1]for n in helpers}}

def authenticate_build(args):
 """Caller independent report/script pins precede trusting any report fields."""
 code=Path(os.environ['WR_CODE']);revision=os.environ['WR_CODE_REVISION']
 require(canonical(Path(__file__))==code/'infra/frontend_sam2_kernel_gate.py','Actual frozen kernel gate source required')
 current=closure(code,revision,'run_frontend_sam2_kernel_gate',('infra/frontend_sam2_kernel_gate.py','infra/run_frontend_sam2_kernel_gate.sh',CONFIG))
 require(re.fullmatch('[0-9a-f]{40}',args.build_revision)and re.fullmatch('[0-9a-f]{64}',args.build_report_sha256)and re.fullmatch('[0-9a-f]{64}',args.build_script_sha256)and 0<args.build_report_bytes<=200000,'Independent bounded build pins required')
 raw,pin=read(REPORT_PATH);require(pin=={'bytes':args.build_report_bytes,'sha256':args.build_report_sha256},'Actual build receipt differs from independently supplied pins')
 report=strict_json(raw);old=ROOT/'jobs'/args.build_revision/'run_frontend_grounding_build/code'
 original=closure(old,args.build_revision,'run_frontend_grounding_build',BUILD_HELPERS)
 require(original['helpers']['infra/frontend_grounding_build.py']['sha256']==args.build_script_sha256 and report.get('source_binding')==original,'Actual build source/script/markers differ from independently pinned producer')
 require(current['helpers'][CONFIG]==original['helpers'][CONFIG],'Current gate config differs from original v6 build config')
 expected={'schema':'world_reward.frontend_grounding_build.v6','stage':'frontend_grounding_build','status':'pass','phase':'complete','producer_revision':args.build_revision,
  'offline_build_exit_code':0,'child_probe_exit_code':0,'parent_unchanged_verified':True,'source_rechecked_before_and_after':True,'extension_import_verified':True,
  'original_grounding_image_parity_claimed':False,'CUDA_execution_verified':False,'replica_ready':False,'license_eligibility_verified':False,'training_overlap_verified':False}
 require(all(report.get(k)==v for k,v in expected.items()),'Genuine complete CPU build required before GPU operator gate')
 cfg=strict_json(read(code/CONFIG)[0]);require(cfg.get('target_image')==TARGET and cfg.get('base_image_id')==BASE and cfg.get('schema')=='world_reward.frontend_grounding_source_pins.v6','Exact v6 target/source config required')
 probe_raw,probe_pin=read(PROBE_PATH,128*1024);declared=report.get('private_child_probe_log',{})
 require(declared.get('relative_path')==str(PROBE_PATH.relative_to(ROOT))and {k:declared.get(k)for k in('bytes','sha256')}==probe_pin,'Original actual CPU import probe log SHA/bytes differs')
 records=[strict_json(line)for line in probe_raw.splitlines()if line.startswith(b'{')and line.endswith(b'}')]
 require(records==report.get('child_probe')and len(records)==2 and records[1].get('extension_import_verified')is True and records[1].get('model_loaded')is False,'Actual CPU import report/probe binding required')
 owner=hashlib.sha256((args.build_revision+original['closure_sha256']).encode()).hexdigest();require(report.get('owner')==owner,'Original source-bound image owner differs')
 parent=report.get('parent_image',{});child=report.get('child_image',{})
 require(parent.get('Id')==BASE and len(parent.get('RootFS',{}).get('Layers',[]))==44,'Exact original 44-layer Body parent required')
 require(re.fullmatch('sha256:[0-9a-f]{64}',str(child.get('Id','')))and child['Id']!=BASE and child.get('Architecture')=='amd64'and child.get('Os')=='linux'and child.get('RootFS',{}).get('Layers',[])[:44]==parent['RootFS']['Layers'],'Genuine new child with preserved parent rootfs required')
 return {'build_report_identity':pin,'source_binding':original,'current_source':current,'owner':owner,'child_image':child,'parent_image':parent,'source_files':report.get('source_files',{})}

def command(arguments):
 env={'PATH':'/usr/bin:/bin:/usr/sbin:/sbin','HOME':'/nonexistent','LANG':'C.UTF-8','DOCKER_HOST':'unix://'+str(ROOT/'docker.sock')}
 try:r=subprocess.run(['/usr/bin/timeout','--signal=TERM','--kill-after=2s','8s',*arguments],env=env,capture_output=True,text=True,timeout=12,check=False)
 except(OSError,subprocess.TimeoutExpired):raise ValueError('Bounded image control unavailable')from None
 require(r.returncode==0,'Bounded image control failed');return r.stdout

def validate_live_image(binding):
 fmt='{"Id":{{json .Id}},"Architecture":{{json .Architecture}},"Os":{{json .Os}},"RootFS":{{json .RootFS}}}'
 for image,wanted in((TARGET,binding['child_image']),(BASE,binding['parent_image'])):
  require(strict_json(command(['docker','image','inspect',image,'--format',fmt]))==wanted,'Actual Docker image/platform/rootfs differs from authenticated CPU build')
 label=command(['docker','image','inspect',binding['child_image']['Id'],'--format','{{index .Config.Labels "world_reward_frontend_grounding_owner"}}']).strip()
 require(label==binding['owner'],'Actual new image owner differs')

def cases():
 zero=[[0]*16 for _ in range(16)];one=[[1]*16 for _ in range(16)]
 blocks=[[int(1<=y<4 and 1<=x<4 or 10<=y<14 and 10<=x<14)for x in range(16)]for y in range(16)]
 diagonal=[[int(y==x)for x in range(16)]for y in range(16)]
 checker=[[int((x+y)%2==0)for x in range(16)]for y in range(16)]
 sparse=[[int(y in(1,5,9,13)and x in(1,5,9,13))for x in range(16)]for y in range(16)]
 return [('zero',zero),('dense',one),('split',blocks),('diagonal8',diagonal),('checker8',checker),('isolated',sparse)]

def reference(mask):
 require(type(mask)is list and len(mask)==16 and all(type(row)is list and len(row)==16 and all(type(v)is int and v in(0,1)for v in row)for row in mask),'Procedural 16x16 binary grid required')
 labels=[[0]*16 for _ in range(16)];counts=[[0]*16 for _ in range(16)];number=0
 for y in range(16):
  for x in range(16):
   if not mask[y][x]or labels[y][x]:continue
   number+=1;queue=deque([(y,x)]);labels[y][x]=number;component=[]
   while queue:
    yy,xx=queue.popleft();component.append((yy,xx))
    for dy in(-1,0,1):
     for dx in(-1,0,1):
      ny,nx=yy+dy,xx+dx
      if (dy or dx)and 0<=ny<16 and 0<=nx<16 and mask[ny][nx]and not labels[ny][nx]:labels[ny][nx]=number;queue.append((ny,nx))
   for yy,xx in component:counts[yy][xx]=len(component)
 return labels,counts,number

def compare(mask,labels,counts):
 expected,areas,components=reference(mask);require(len(labels)==16 and len(counts)==16 and all(len(row)==16 for row in labels+counts),'Operator output shape differs')
 forward={};reverse={}
 for y in range(16):
  for x in range(16):
   actual=labels[y][x];wanted=expected[y][x]
   require(type(actual)is int and actual>=0 and counts[y][x]==areas[y][x],'Operator labels/counts differ from independent BFS')
   if not wanted:require(actual==0,'Background labels must be zero');continue
   require(actual>0,'Foreground component label missing')
   require(forward.setdefault(wanted,actual)==actual and reverse.setdefault(actual,wanted)==wanted,'Operator partition merges or splits independent components')
 return {'components':components,'foreground_pixels':sum(map(sum,mask))}

def run_kernel(binding):
 # This receipt/source authentication is already completed, before imports.
 import torch
 import sam2
 from sam2 import _C
 from sam2.utils.misc import fill_holes_in_mask_scores
 require(torch.cuda.is_available()and 'H100'in torch.cuda.get_device_name(),'Actual H100 CUDA required')
 module=canonical(Path(sam2.__file__).parent);pins={}
 for relative in('utils/misc.py',):
  _,actual=read(module/relative,200000,False);wanted=binding['source_files'].get('sam2/sam2/'+relative,{})
  require(all(actual[k]==wanted.get(k)for k in('bytes','sha256')),'Actual installed SAM2 helper differs from authenticated CPU source');pins[relative]=actual
 extension=canonical(Path(_C.__file__));require(extension.parent==module,'Original installed SAM2 extension path required');_,pin=read(extension,20_000_000,False);pins['extension']=pin
 rows=cases();tensor=torch.tensor([row for name,row in rows],dtype=torch.uint8,device='cuda')[:,None].contiguous()
 labels,counts=_C.get_connected_componnets(tensor);torch.cuda.synchronize()
 require(labels.dtype==torch.int32 and counts.dtype==torch.int32 and labels.shape==tensor.shape and counts.shape==tensor.shape and labels.is_cuda and counts.is_cuda,'Actual CUDA uint8→int32 operator ABI differs')
 labels_cpu=labels.cpu().tolist();counts_cpu=counts.cpu().tolist();results=[]
 for i,(name,mask)in enumerate(rows):results.append({'case':name,**compare(mask,labels_cpu[i][0],counts_cpu[i][0])})
 scores=[[[1.0]*16 for _ in range(16)]for _ in range(4)]
 scores[1][2][2]=-1.0
 for y in range(8,11):
  for x in range(8,11):scores[1][y][x]=-1.0
 for y in range(16):
  for x in range(16):scores[2][y][x]=-1.0
 for y in range(4,6):
  for x in range(4,8):scores[3][y][x]=-1.0
 source=torch.tensor(scores,dtype=torch.float32,device='cuda')[:,None]
 with warnings.catch_warnings():
  warnings.simplefilter('error');filled=fill_holes_in_mask_scores(source,8);torch.cuda.synchronize()
 actual=filled.cpu().tolist();wanted=source.cpu().tolist();wanted[1][0][2][2]=float(torch.tensor(0.1,dtype=torch.float32))
 for y in range(4,6):
  for x in range(4,8):wanted[3][0][y][x]=float(torch.tensor(0.1,dtype=torch.float32))
 require(actual==wanted,'Small-hole fill differs from fixed <=8 operator contract')
 require(torch.equal(source,torch.tensor(scores,dtype=torch.float32,device='cuda')[:,None]),'Hole postprocessor mutated original inputs')
 return {'cases':results,'small_hole_threshold':8,'small_hole_value':0.1,'operator_source_identities':pins,'gpu':torch.cuda.get_device_name(),'CUDA_operator_execution_verified':True}

def parser():
 p=argparse.ArgumentParser(allow_abbrev=False);mode=p.add_mutually_exclusive_group(required=True)
 mode.add_argument('--preflight',action='store_true');mode.add_argument('--verify',action='store_true');mode.add_argument('--run',action='store_true')
 p.add_argument('--build-revision',required=True);p.add_argument('--build-report-sha256',required=True);p.add_argument('--build-report-bytes',type=int,required=True);p.add_argument('--build-script-sha256',required=True);return p

def main(argv=None):
 args=parser().parse_args(argv);require(sys.platform=='linux'and os.geteuid()==0,'Linux root owned kernel gate required')
 binding=authenticate_build(args)
 if not args.run:
  require(os.uname().nodename=='world-reward-ncc-h100-02','Exact VM02 preflight host required');validate_live_image(binding)
  print(binding['child_image']['Id']if args.preflight else'kernel_gate_bindings_verified');return 0
 require({p.name for p in Path('/sys/class/net').iterdir()}=={'lo'},'Isolated network-none CUDA container required')
 require(os.environ.get('WR_KERNEL_IMAGE_ID')==binding['child_image']['Id'],'Root preflight selected actual image required')
 canonical(OUTPUT);require(OUTPUT.is_dir()and not list(OUTPUT.iterdir()),'Fresh reserved kernel output required')
 report={'schema':'world_reward.frontend_sam2_kernel_gate.v1','stage':'frontend_sam2_kernel_gate','status':'fail','build_report_identity':binding['build_report_identity'],
  'image_id':binding['child_image']['Id'],'models_loaded':False,'challenge_data_read':False,'historical_image_parity_claimed':False,'replica_ready':False,'license_eligibility_verified':False,'training_overlap_verified':False}
 report.update(producer_revision=os.environ["WR_CODE_REVISION"],script_sha256=binding["current_source"]["helpers"]["infra/frontend_sam2_kernel_gate.py"]["sha256"],source_binding=binding["current_source"],budget_seconds=120)
 started=time.monotonic();status=1
 try:
  report.update(run_kernel(binding));require(authenticate_build(args)==binding,'Frozen CPU build/source changed during operator gate');report['status']='pass';status=0
 except Exception as error:report['error']=str(error)[:180]if isinstance(error,ValueError)else type(error).__name__
 report["elapsed_seconds"]=time.monotonic()-started
 raw=(json.dumps(report,sort_keys=True,separators=(',',':'))+'\n').encode()
 with os.fdopen(os.open(OUTPUT/'report.json',os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o400),'wb')as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
 print(json.dumps({k:report[k]for k in('stage','status','replica_ready','historical_image_parity_claimed')}));return status

if __name__=='__main__':sys.exit(main())
