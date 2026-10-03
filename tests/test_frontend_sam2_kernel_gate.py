"""Procedural CPU-only tests; never import Torch, models, CUDA or Docker."""
import copy
import hashlib
import importlib.util
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('frontend_sam2_kernel_gate',ROOT/'infra/frontend_sam2_kernel_gate.py')
gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)

@pytest.mark.parametrize('name,components,pixels',[('zero',0,0),('dense',1,256),('split',2,25),('diagonal8',1,16),('checker8',1,128),('isolated',16,16)])
def test_six_independent8connected_reference_cases(name,components,pixels):
 mask=dict(gate.cases())[name];labels,counts,number=gate.reference(mask)
 assert number==components
 assert gate.compare(mask,labels,counts)=={'components':components,'foreground_pixels':pixels}

def test_partition_labels_need_not_match_numeric_ids():
 mask=dict(gate.cases())['split'];labels,counts,_=gate.reference(mask)
 labels=[[value*71 if value else 0 for value in row]for row in labels]
 assert gate.compare(mask,labels,counts)['components']==2

@pytest.mark.parametrize('kind',['counts','missing','background','merge','split'])
def test_wrong_kernel_partition_or_counts_fails(kind):
 mask=dict(gate.cases())['split'];labels,counts,_=gate.reference(mask)
 if kind=='counts':counts[1][1]+=1
 elif kind=='missing':labels[1][1]=0
 elif kind=='background':labels[0][0]=9
 elif kind=='merge':labels=[[1 if value else 0 for value in row]for row in labels]
 elif kind=='split':labels[1][1]=9
 with pytest.raises(ValueError):gate.compare(mask,labels,counts)

@pytest.mark.parametrize('mask',[[[0]],[[False]*16 for _ in range(16)],[[2]*16 for _ in range(16)]])
def test_reference_requires_exact_binary_original_fixture_grid(mask):
 with pytest.raises(ValueError):gate.reference(mask)

def test_parser_independent_build_pins_required():
 parser=gate.parser()
 with pytest.raises(SystemExit):parser.parse_args(['--run'])
 args=parser.parse_args(['--preflight','--build-revision','a'*40,'--build-report-sha256','b'*64,'--build-report-bytes','1234','--build-script-sha256','c'*64])
 assert args.preflight and args.build_report_bytes==1234
 with pytest.raises(SystemExit):parser.parse_args(['--run','--verify'])

def test_duplicate_or_nonfinite_report_rejected():
 for value in ('{"x":1,"x":2}','{"x":NaN}'):
  with pytest.raises(ValueError):gate.strict_json(value)

def test_control_errors_sanitized_and_bounded(monkeypatch):
 calls=[]
 def run(args,**kwargs):calls.append((args,kwargs));return subprocess.CompletedProcess(args,1,'hf_PRIVATE','hf_PRIVATE')
 monkeypatch.setattr(gate.subprocess,'run',run)
 with pytest.raises(ValueError)as caught:gate.command(['docker','image','inspect'])
 assert 'hf_PRIVATE'not in str(caught.value)
 args,kwargs=calls[0];assert args[:4]==['/usr/bin/timeout','--signal=TERM','--kill-after=2s','8s']
 assert kwargs['timeout']==12 and kwargs['env']['HOME']=='/nonexistent'and 'HF_TOKEN'not in kwargs['env']

def test_actual_live_image_projection_and_owner_only(monkeypatch):
 parent={'Id':gate.BASE,'Architecture':'amd64','Os':'linux','RootFS':{'Type':'layers','Layers':['sha256:'+'a'*64]*44}}
 child=copy.deepcopy(parent);child['Id']='sha256:'+'b'*64;binding={'parent_image':parent,'child_image':child,'owner':'c'*64}
 calls=[]
 def command(args):
  calls.append(args)
  if 'Labels'in args[-1]:return binding['owner']+'\n'
  return gate.json.dumps(parent if args[3]==gate.BASE else child)
 monkeypatch.setattr(gate,'command',command);gate.validate_live_image(binding)
 assert len(calls)==3 and all('.Config.Env'not in call[-1]for call in calls)
 monkeypatch.setattr(gate,'command',lambda args:'{}')
 with pytest.raises(ValueError):gate.validate_live_image(binding)

def test_static_no_model_or_data_mounts_and_owned_cleanup():
 wrapper=(ROOT/'infra/run_frontend_sam2_kernel_gate.sh').read_text();source=(ROOT/'infra/frontend_sam2_kernel_gate.py').read_text()
 assert '--gpus all --network none'in wrapper and 'flock --nonblock'in wrapper
 assert '--cap-drop ALL --security-opt no-new-privileges'in wrapper and '--read-only'in wrapper
 assert '/weights'not in wrapper and '/data'not in wrapper and '/validation'not in wrapper and '/frontend-assets-extracted'not in wrapper
 assert '--build-report-sha256'in wrapper and '--build-script-sha256'in wrapper
 assert 'docker stop --time 3 "$NAME"'in wrapper and 'docker kill "$NAME"'in wrapper and 'docker rm --force "$NAME"'in wrapper
 assert 'warnings.simplefilter(\'error\')'in source
 assert 'fill_holes_in_mask_scores(source,8)'in source
 assert 'from_pretrained'not in source and 'SAM2ImagePredictor('not in source

def test_independent_build_report_authentication_before_trust(monkeypatch):
 revision='a'*40;producer='b'*40;code=gate.ROOT/'jobs'/revision/'run_frontend_sam2_kernel_gate/code';old=gate.ROOT/'jobs'/producer/'run_frontend_grounding_build/code'
 configpin={'bytes':100,'sha256':'c'*64};scriptpin={'bytes':123,'sha256':'d'*64};source={'markers':{'revision':producer+'\n','source-sha256':'e'*64+'\n'},'closure_sha256':'f'*64,'helpers':{gate.CONFIG:configpin,'infra/frontend_grounding_build.py':scriptpin}}
 current={'helpers':{gate.CONFIG:configpin}};owner=hashlib.sha256((producer+source['closure_sha256']).encode()).hexdigest()
 parent={'Id':gate.BASE,'Architecture':'amd64','Os':'linux','RootFS':{'Type':'layers','Layers':['sha256:'+'1'*64]*44}}
 child=copy.deepcopy(parent);child['Id']='sha256:'+'2'*64
 records=[{'versions':{}},{'extension_import_verified':True,'model_loaded':False}]
 probe_raw=('\n'.join(gate.json.dumps(r)for r in records)+'\n').encode();probepin={'bytes':len(probe_raw),'sha256':hashlib.sha256(probe_raw).hexdigest()}
 report={'schema':'world_reward.frontend_grounding_build.v6','stage':'frontend_grounding_build','status':'pass','phase':'complete','producer_revision':producer,
  'offline_build_exit_code':0,'child_probe_exit_code':0,'parent_unchanged_verified':True,'source_rechecked_before_and_after':True,'extension_import_verified':True,
  'original_grounding_image_parity_claimed':False,'CUDA_execution_verified':False,'replica_ready':False,'license_eligibility_verified':False,'training_overlap_verified':False,
  'source_binding':source,'owner':owner,'parent_image':parent,'child_image':child,'private_child_probe_log':{'relative_path':str(gate.PROBE_PATH.relative_to(gate.ROOT)),**probepin},'child_probe':records}
 cfg={'schema':'world_reward.frontend_grounding_source_pins.v6','target_image':gate.TARGET,'base_image_id':gate.BASE}
 def read(path,*args):
  if path==gate.REPORT_PATH:
   raw=gate.json.dumps(report).encode();return raw,{'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
  if path==gate.PROBE_PATH:return probe_raw,probepin
  if path==code/gate.CONFIG:return gate.json.dumps(cfg).encode(),configpin
  raise AssertionError('Unexpected read '+str(path))
 monkeypatch.setenv('WR_CODE',str(code));monkeypatch.setenv('WR_CODE_REVISION',revision)
 monkeypatch.setattr(gate,'__file__',str(code/'infra/frontend_sam2_kernel_gate.py'))
 monkeypatch.setattr(gate,'read',read);monkeypatch.setattr(gate,'closure',lambda path,*args:source if path==old else current)
 raw,pin=read(gate.REPORT_PATH);args=SimpleNamespace(build_revision=producer,build_report_sha256=pin['sha256'],build_report_bytes=pin['bytes'],build_script_sha256=scriptpin['sha256'])
 assert gate.authenticate_build(args)['owner']==owner
 args.build_report_sha256='0'*64
 with pytest.raises(ValueError,match='independently supplied'):gate.authenticate_build(args)
 args.build_report_sha256=pin['sha256'];args.build_script_sha256='0'*64
 with pytest.raises(ValueError,match='source/script/markers'):gate.authenticate_build(args)
