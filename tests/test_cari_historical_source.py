"""Tiny original source binding fixtures; no Azure or model calls."""
import hashlib,importlib.util,json
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('historical',ROOT/'infra/cari_historical_source.py');gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)

def pin(raw):return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def write(path,raw,mode=0o444):
 path.parent.mkdir(parents=True,exist_ok=True)
 if path.exists():path.chmod(0o644)
 path.write_bytes(raw);path.chmod(mode)

@pytest.fixture
def fixture(tmp_path):
 rev='a'*40;code=tmp_path/'jobs'/rev/'run_cari_full_refine_queued/code';files={'infra/source.py':b'original code\n','pyproject.toml':b'original metadata\n'}
 for name,raw in files.items():write(code/name,raw)
 write(code.parent/'revision',(rev+'\n').encode());write(code.parent/'source-sha256',('b'*64+'\n').encode())
 for p in(code,*code.rglob('*')):
  if p.is_dir():p.chmod(0o555)
 audit=tmp_path/'results/episode3-queued-source-cache-audit.json';raw=b'{"status":"pass","queue_status":"fail"}\n';write(audit,raw)
 pins={'schema':'world_reward.historical_native_source_pins.v1','producer_revision':rev,'job':'run_cari_full_refine_queued','source_archive_sha256':'b'*64,'files':{k:pin(v)for k,v in files.items()},'separate_queue_failure_receipt':{'relative_path':str(audit.relative_to(tmp_path)),**pin(raw)},'queue_status_reclassified':False}
 path=tmp_path/'pins.json';write(path,(json.dumps(pins)+'\n').encode())
 return tmp_path,code,path,pins

def test_original_fullbytes_not_executed_current_pin_consumer(fixture):
 root,code,path,pins=fixture;selected,proof=gate.verify_historical_source(root,path)
 assert selected==code and proof['files_verified']==2 and proof['queue_status_reclassified']is False
 assert not (code/'__pycache__').exists()

@pytest.mark.parametrize('fault',['source','missing','extra','writable','marker','archive','audit','alias','queue'])
def test_exact_oldsource_or_marker_required(fixture,fault,tmp_path):
 root,code,path,pins=fixture;p=code/'infra/source.py'
 if fault=='source':p.chmod(0o644);p.write_bytes(b'new code');p.chmod(0o444)
 elif fault=='missing':p.parent.chmod(0o755);p.unlink()
 elif fault=='extra':code.chmod(0o755);write(code/'extra.py',b'extra')
 elif fault=='writable':p.chmod(0o644)
 elif fault in('marker','archive'):
  target=code.parent/('revision'if fault=='marker'else'source-sha256');target.chmod(0o644);target.write_bytes(b'changed');target.chmod(0o444)
 elif fault=='audit':write(root/'results/episode3-queued-source-cache-audit.json',b'{"status":"fail"}\n')
 elif fault=='alias':p.parent.chmod(0o755);p.unlink();p.symlink_to(path)
 else:
  pins['queue_status_reclassified']=True;path.chmod(0o644);path.write_text(json.dumps(pins));path.chmod(0o444)
 with pytest.raises(ValueError):gate.verify_historical_source(root,path)
