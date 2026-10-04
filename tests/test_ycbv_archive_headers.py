"""Offline central-directory fixtures only: no archive/network/Azure payloads."""
import ast,copy,hashlib,importlib.util,io,json,stat,struct,sys,zipfile
from pathlib import Path
from types import SimpleNamespace
import pytest
REPO=Path(__file__).resolve().parents[1]
@pytest.fixture
def gate():
 s=importlib.util.spec_from_file_location('header_test',REPO/'infra/ycbv_archive_headers.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def cd_record(name='test/000048/rgb/000000.png',size=3,flags=0,compression=8,attrs=stat.S_IFREG<<16,offset=0,extra=b'',zip64=False):
 raw=name.encode('utf-8');usize=csize=size;where=offset
 if zip64:usize=csize=where=0xffffffff;extra=struct.pack('<HHQQQ',1,24,size,size,offset)
 return struct.pack('<4s6H3I5H2I',b'PK\x01\x02',45,45,flags,compression,0,0,123,csize,usize,len(raw),len(extra),0,0,0,attrs,where)+raw+extra

def zip_headers(gate,monkeypatch,zip64=False):
 cd=cd_record(zip64=zip64);offset=100;parts={(offset,offset+len(cd)-1):cd}
 if zip64:
  zpos=offset+len(cd);z=struct.pack('<4sQ2H2I4Q',b'PK\x06\x06',44,45,45,0,0,1,1,len(cd),offset)
  loc=struct.pack('<4sIQI',b'PK\x06\x07',0,zpos,1);end=zpos+76
  parts[zpos,zpos+55]=z;parts[end-20,end-1]=loc
  e=struct.pack('<4s4H2IH',b'PK\x05\x06',0,0,65535,65535,0xffffffff,0xffffffff,0)
 else:
  end=offset+len(cd);e=struct.pack('<4s4H2IH',b'PK\x05\x06',0,0,1,1,len(cd),offset,0)
 parts[end,end+21]=e;monkeypatch.setitem(gate.ARCHIVES,'ycbv_test_all.zip',(end+22,'a'*64));requested=[]
 def get(name,a,b):requested.append((a,b));return parts[a,b]
 return get,requested,cd,offset

@pytest.mark.parametrize('zip64',[False,True])
def test_precise_metadata_only_ranges(gate,monkeypatch,zip64):
 get,ranges,data,offset=zip_headers(gate,monkeypatch,zip64);got,count,start=gate.directory(get,'ycbv_test_all.zip')
 assert got==data and count==1 and start==offset
 rows=gate.central(got,count,start);assert rows[0]['file_size']==3 and rows[0]['filename'].endswith('.png')
 assert all(a>=offset for a,b in ranges) and ranges[0][1]-ranges[0][0]+1==22
 assert rows[0]['raw_filename_sha256']==hashlib.sha256(rows[0]['filename'].encode()).hexdigest()

@pytest.mark.parametrize('fault',['comment','multidisk','offset','length','zip64_size','signature','extra','truncated','trailing','encoding'])
def test_malformed_protocol_no_payload_fallback(gate,monkeypatch,fault):
 get,ranges,cd,offset=zip_headers(gate,monkeypatch,True)
 if fault in ('comment','multidisk','offset','length','zip64_size'):
  def bad(name,a,b):
   data=get(name,a,b)
   if data[:4]==b'PK\x05\x06':
    value=list(struct.unpack('<4s4H2IH',data))
    if fault=='comment':value[-1]=1
    elif fault=='multidisk':value[1]=1
    return struct.pack('<4s4H2IH',*value)
   if data[:4]==b'PK\x06\x06':
    value=list(struct.unpack('<4sQ2H2I4Q',data))
    if fault=='offset':value[-1]+=1
    elif fault=='length':value[-2]=gate.MAX+1
    elif fault=='zip64_size':value[1]=45
    return struct.pack('<4sQ2H2I4Q',*value)
   return data
  with pytest.raises(gate.ContractError):gate.directory(bad,'ycbv_test_all.zip')
 else:
  if fault=='signature':cd=b'BAD!'+cd[4:]
  elif fault=='extra':cd=cd_record(extra=b'\x01')
  elif fault=='truncated':cd=cd[:-1]
  elif fault=='trailing':cd+=b'anything'
  else:cd=cd_record(flags=2048).replace(b'test',b'\xffest',1)
  with pytest.raises(gate.ContractError):gate.central(cd,1,offset)

def row(gate,name='test/000048/rgb/000000.png',size=3,**kwargs):return gate.central(cd_record(name,size,**kwargs),1,100)[0]
@pytest.mark.parametrize('fault',['path','duplicate','symlink','compression','encrypted','member','total','count','ancestor','empty','coco','prefix'])
def test_earliest_original_guard_and_optional_coco(gate,fault):
 r=row(gate);rows=[r];budget=[0,0];expected='Unsafe or duplicate ZIP member'
 if fault=='path':r['filename']='test/../escape'
 elif fault=='duplicate':rows.append(r)
 elif fault=='symlink':r['external_attr']=stat.S_IFLNK<<16
 elif fault=='compression':r['compress_type']=12
 elif fault=='encrypted':r['flag_bits']=1
 elif fault=='member':r['file_size']=2000000001;expected='ZIP expansion bound exceeded'
 elif fault=='total':budget[0]=60000000000;expected='ZIP expansion bound exceeded'
 elif fault=='count':budget[1]=1000000;expected='ZIP expansion bound exceeded'
 elif fault=='ancestor':rows.insert(0,row(gate,'test'));expected='Empty ZIP or file ancestor collision'
 elif fault=='empty':rows=[];expected='Empty ZIP or file ancestor collision'
 elif fault=='coco':r['filename']='test/000048/scene_gt_coco.json';expected='Full test root/layout differs'
 else:r['filename']='ycbv/'+r['filename'];expected='Full test root/layout differs'
 result=gate.audit(rows,'ycbv_test_all.zip',budget);assert result['guard']==expected
 if fault in ('path','symlink','compression','encrypted'):assert result['member_count']==0


def test_inventory_layout_equivalent_original_no_import_in_runtime(gate,monkeypatch):
 monkeypatch.syspath_prepend(str(REPO/'infra'));s=importlib.util.spec_from_file_location('original_headers_fixture',REPO/'infra/ycbv_point_acquire.py');old=importlib.util.module_from_spec(s);s.loader.exec_module(old)
 rows=[]
 for scene in range(48,60):
  rows.extend(row(gate,f'test/{scene:06d}/scene_{kind}.json')for kind in ('camera','gt','gt_info'))
  rows.append(row(gate,f'test/{scene:06d}/rgb/000000.png'))
 for change in (None,'coco','unsafe','total','mask'):
  values=copy.deepcopy(rows);budget=[100,4];oldbudget=budget.copy()
  if change=='coco':values.append(row(gate,'test/000048/scene_gt_coco.json'))
  elif change=='unsafe':values[-1]['filename']='../escape'
  elif change=='total':values[10]['file_size']=60000000000
  elif change=='mask':values.append(row(gate,'test/000048/mask/000000_000000.png'))
  fake=[SimpleNamespace(**v,is_dir=lambda n=v['filename']:n.endswith('/'))for v in values]
  error=None
  try:members=old.zip_inventory(SimpleNamespace(infolist=lambda:fake),oldbudget);old.inspect_layout('ycbv_test_all.zip',members)
  except ValueError as e:error=str(e)
  result=gate.audit(values,'ycbv_test_all.zip',budget)
  assert (None if result is None else result['guard'])==error and budget==oldbudget
 assert not any(isinstance(x,(ast.Import,ast.ImportFrom))and 'acquire'in ast.unparse(x)for x in ast.walk(ast.parse(Path(gate.__file__).read_text())))

@pytest.mark.parametrize('fault',['status','range','encoding','length','etag','overflow','short','budget','redirect'])
def test_http206_etag_and_bounded_reads_only(gate,monkeypatch,fault):
 payload=b'x'*22;sha='a'*64;monkeypatch.setitem(gate.ARCHIVES,'ycbv_test_all.zip',(100,sha));reads=[]
 h={'Content-Range':'bytes 78-99/100','Content-Length':'22','ETag':'"'+sha+'"','Content-Encoding':'identity'};status=206
 if fault=='status':status=200
 elif fault=='range':h['Content-Range']='bytes 78-99/101'
 elif fault=='encoding':h['Content-Encoding']='gzip'
 elif fault=='length':h.pop('Content-Length')
 elif fault=='etag':h['ETag']='"other"'
 elif fault=='overflow':payload+=b'x'
 elif fault=='short':payload=payload[:-1]
 class Response(io.BytesIO):
  def __init__(self):super().__init__(payload);self.status=status;self.headers=h
  def geturl(self):return 'https://evil.invalid/archive'if fault=='redirect'else'https://cas-bridge.xethub.hf.co/archive?temporary=redacted'
  def read(self,n):reads.append(n);return super().read(n)
 monkeypatch.setattr(gate.urllib.request,'build_opener',lambda *a:SimpleNamespace(open=lambda *a,**k:Response()))
 client=gate.Ranges()
 if fault=='budget':client.bytes=gate.MAX
 with pytest.raises(gate.ContractError):client.get('ycbv_test_all.zip',78,99)
 assert all(n==23 for n in reads) and not reads if fault in ('status','range','encoding','length','etag','budget','redirect')else reads==[23]


def test_http_success_evidence_and_redirect_safe(gate,monkeypatch):
 monkeypatch.setitem(gate.ARCHIVES,'ycbv_test_all.zip',(22,'a'*64))
 class Response(io.BytesIO):
  status=206;headers={'Content-Range':'bytes 0-21/22','Content-Length':'22','ETag':'a'*64}
  def geturl(self):return'https://huggingface.co/exact'
 monkeypatch.setattr(gate.urllib.request,'build_opener',lambda *a:SimpleNamespace(open=lambda *a,**k:Response(b'x'*22)))
 c=gate.Ranges();assert c.get('ycbv_test_all.zip',0,21)==b'x'*22 and c.bytes==22 and c.requests==1
 for url in ('http://huggingface.co/a','https://u:p@huggingface.co/a','https://huggingface.co/a?token=secret','https://evil.invalid/a'):
  with pytest.raises(gate.ContractError):gate.public_url(url)
 with pytest.raises(gate.ContractError):gate.Redirect('a'*64).redirect_request(None,None,302,'',{'X-Linked-ETag':'wrong'},'https://cas-bridge.xethub.hf.co/a')


def test_wrapper_closure_host_cpu_no_payload_claim(gate):
 wrapper=(REPO/'infra/run_ycbv_archive_headers.sh').read_text();tree=ast.parse(Path(gate.__file__).read_text())
 assert '/infra/ycbv_archive_headers.py'in wrapper and '/configs/ycbv_point_inventory_failed_pins.json'in wrapper and '310s'in wrapper and 'env -i'in wrapper
 assert 'docker'not in wrapper.lower()
 assert not any(isinstance(x,ast.Call)and isinstance(x.func,ast.Attribute)and x.func.attr in ('extract','extractall','open')and isinstance(x.func.value,ast.Name)and x.func.value.id=='archive'for x in ast.walk(tree))
 source=Path(gate.__file__).read_text()
 assert 'whole_archive_SHA_verified=False'in source and 'CRC_verified=False'in source and 'member_payload_read=False'in source


def sealed(path,data,mode=0o444):
 path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data);path.chmod(mode)
 return dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())

def binding_fixture(gate,tmp_path,monkeypatch):
 root=tmp_path/'root';rev='b'*40;code=root/'jobs'/rev/gate.JOB/'code';old=root/'jobs'/gate.FAILED_REV/'run_ycbv_point_acquire/code'
 for n in ('revision','source-sha256'):
  sealed(code.parent/n,((rev if n=='revision'else'c'*64)+'\n').encode(),0o644)
  sealed(old.parent/n,((gate.FAILED_REV if n=='revision'else'd'*64)+'\n').encode(),0o644)
 hist={}
 for n in ('infra/ycbv_point_acquire.py','infra/run_ycbv_point_acquire.sh','infra/tudl_acquire.py','configs/ycbv_point_protocol.json'):
  data=(REPO/n).read_bytes()if n.endswith('.json')else b'original-source-exact\n'
  hist[n]=sealed(old/n,data)
 monkeypatch.setattr(gate,'FAILED_SCRIPT',hist['infra/ycbv_point_acquire.py']['sha256'])
 markers={n:dict(bytes=(old.parent/n).stat().st_size,sha256=hashlib.sha256((old.parent/n).read_bytes()).hexdigest())for n in ('revision','source-sha256')}
 report=dict(status='fail',phase='inventory',error_type='ValueError',producer_revision=gate.FAILED_REV,script_sha256=gate.FAILED_SCRIPT,source_rehashed_after=True,disposable_archives_removed=True,cleanup_completed=True,source_helpers=dict(files=hist,markers=markers))
 reportpin=sealed(root/'validation/ycbv_point_pose_v1/report.json',json.dumps(report).encode(),0o400)
 monkeypatch.setattr(gate,'FAILURE_REPORT',dict(path='validation/ycbv_point_pose_v1/report.json',**reportpin))
 sealed(root/'validation/ycbv_point_pose_v1/.container.cid',b'e'*64,0o400)
 pins=json.loads((REPO/gate.PINS).read_bytes());pins['script_sha256']=gate.FAILED_SCRIPT;pins['report']=copy.deepcopy(gate.FAILURE_REPORT)
 sealed(code/gate.PINS,json.dumps(pins).encode());sealed(code/'infra/ycbv_archive_headers.py',Path(gate.__file__).read_bytes());sealed(code/'infra/run_ycbv_archive_headers.sh',(REPO/'infra/run_ycbv_archive_headers.sh').read_bytes())
 for base in (code,old):
  for p in sorted(base.rglob('*'),reverse=True):
   if p.is_dir():p.chmod(0o555)
  base.chmod(0o555)
 (root/'results').mkdir();monkeypatch.setattr(gate,'ROOT',root);monkeypatch.setattr(gate,'__file__',str(code/'infra/ycbv_archive_headers.py'));monkeypatch.setattr(gate,'inactive',lambda cid:dict(MainPID='0'))
 return root,code,rev,pins


def test_bindings_full_failed_producer_chain_and_prepost(gate,tmp_path,monkeypatch):
 root,code,rev,pins=binding_fixture(gate,tmp_path,monkeypatch);proof=gate.bindings(code,rev)
 assert proof['original_failure']=={k:pins['report'][k]for k in ('bytes','sha256')} and len(proof['historical_sources'])==4 and len(proof['historical_markers'])==2
 path=root/'jobs'/gate.FAILED_REV/'run_ycbv_point_acquire/source-sha256';path.write_bytes(b'f'*64+b'\n')
 with pytest.raises(gate.ContractError):gate.bindings(code,rev)

@pytest.mark.parametrize('fault',['revision','reportpath','extra','boolean','value','sha','additional_file'])
def test_failclosed_pin_schema_before_network(gate,tmp_path,monkeypatch,fault):
 root,code,rev,pins=binding_fixture(gate,tmp_path,monkeypatch)
 if fault=='revision':pins['producer_revision']='f'*40
 elif fault=='reportpath':pins['report']['path']='eval_private/hidden'
 elif fault=='extra':pins['extra']=0
 elif fault=='boolean':pins['RGB_inputs_retained']=False
 elif fault=='value':pins['further_acquisition_authorized']=True
 elif fault=='sha':pins['script_sha256']='0'*64
 else:sealed(root/'validation/ycbv_point_pose_v1/unknown',b'NO')
 if fault!='additional_file':
  (code/gate.PINS).chmod(0o644);sealed(code/gate.PINS,json.dumps(pins).encode())
 with pytest.raises(gate.ContractError):gate.bindings(code,rev)


def test_inactive_exact_unit_and_container_no_kill(gate,monkeypatch):
 calls=[]
 def run(args,**kw):
  calls.append(args);assert kw['env']['DOCKER_HOST'].endswith('/docker.sock')
  return SimpleNamespace(returncode=0,stdout='LoadState=loaded\nActiveState=failed\nSubState=failed\nMainPID=0\nExecMainStatus=1\n'if args[0]=='systemctl'else'')
 monkeypatch.setattr(gate.subprocess,'run',run);assert gate.inactive('a'*64)['MainPID']=='0'
 assert calls[-1][-1]=='id='+'a'*64 and all('kill'not in x and 'rm'not in x for args in calls for x in args)
 monkeypatch.setattr(gate.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=0,stdout='invalid'))
 with pytest.raises(gate.ContractError):gate.inactive('a'*64)


def test_actual_flow_both_central_directories_shared_budget_and_failure_receipt(gate,tmp_path,monkeypatch):
 root,code,rev,pins=binding_fixture(gate,tmp_path,monkeypatch);parts={}
 for name,names in [('ycbv_base.zip',sorted(gate.BASE)),('ycbv_test_all.zip',['test/000048/scene_gt_coco.json'])]:
  cd=b''.join(cd_record(n,size=3)for n in names);offset=100;end=offset+len(cd);e=struct.pack('<4s4H2IH',b'PK\x05\x06',0,0,len(names),len(names),len(cd),offset,0)
  monkeypatch.setitem(gate.ARCHIVES,name,(end+22,gate.ARCHIVES[name][1]));parts[name,offset,end-1]=cd;parts[name,end,end+21]=e
 # Only URL archive sizes are fixture scaled; immutable protocol bindings already tested above.
 before=dict(source='separately-tested')
 monkeypatch.setattr(gate,'bindings',lambda *args:before)
 class Client:
  bytes=0;requests=0;proofs=[]
  def get(self,name,a,b):
   data=parts[name,a,b];self.bytes+=len(data);self.requests+=1;return data
 monkeypatch.setattr(gate,'Ranges',Client);monkeypatch.setattr(gate.os,'getuid',lambda:0);monkeypatch.setattr(gate.os,'uname',lambda:SimpleNamespace(sysname='Linux',nodename='world-reward-ncc-h100-02'))
 monkeypatch.setenv('WR_ROOT',str(root));monkeypatch.setenv('WR_CODE',str(code));monkeypatch.setenv('WR_CODE_REVISION',rev)
 gate.main();out=root/'results'/('ycbv-archive-header-diagnostic-'+rev)/'report.json';report=json.loads(out.read_bytes())
 assert out.stat().st_mode&0o777==0o400 and report['status']=='pass'and report['first_rejection']['guard']=='Full test root/layout differs'and report['expanded_bytes']==15 and report['member_count']==5
 assert report['whole_archive_SHA_verified']is False and report['CRC_verified']is False and report['member_payload_read']is False and report['original_failure_unchanged']is True and len(report['archives'])==2


def test_metadata_atime_not_integrity_false_positive(gate,tmp_path):
 p=tmp_path/'metadata';sealed(p,b'public metadata');before=gate.metadata(p)[1];assert gate.metadata(p)[1]==before


def test_primary_pattern_literal_matches_pinned_original_ast(gate):
 tree=ast.parse((REPO/'infra/ycbv_point_acquire.py').read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='inspect_layout')
 original=next(n.value.value for n in ast.walk(fn)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='pattern'for t in n.targets))
 assert gate.PATTERN==original and gate.MAX==32*1024**2 and gate.SECONDS==300


def test_overbound_centralfacts_before_body(gate,monkeypatch):
 get,ranges,cd,offset=zip_headers(gate,monkeypatch,True)
 def bad(name,a,b):
  data=get(name,a,b)
  if data[:4]==b'PK\x06\x06':
   v=list(struct.unpack('<4sQ2H2I4Q',data));v[-2]=gate.MAX+1;return struct.pack('<4sQ2H2I4Q',*v)
  return data
 with pytest.raises(gate.ContractError)as exc:gate.directory(bad,'ycbv_test_all.zip')
 assert exc.value.facts['central_bytes']==gate.MAX+1 and len(ranges)==3


def test_redirect_never_reads_intermediate_body(gate):
 class Forbidden:
  closed=False
  def read(self,*a):raise AssertionError('Redirect payload read forbidden')
  def close(self):self.closed=True
 response=Forbidden();redirect=gate.Redirect('a'*64)
 redirect.parent=SimpleNamespace(open=lambda req,timeout:b'next-response')
 req=gate.urllib.request.Request('https://huggingface.co/immutable',headers={'Range':'bytes=1-2'});req.timeout=30
 result=redirect.http_error_302(req,response,302,'',{'Location':'https://cas-bridge.xethub.hf.co/public?temporary=secret','X-Linked-ETag':'a'*64})
 assert result==b'next-response'and response.closed


def test_empty_source_initializer_only_not_metadata_reports(gate,tmp_path,monkeypatch):
 root,code,rev,pins=binding_fixture(gate,tmp_path,monkeypatch);code.chmod(0o755);sealed(code/'empty.py',b'');code.chmod(0o555)
 assert gate.bindings(code,rev)['files']['empty.py']['bytes']==0
 with pytest.raises(gate.ContractError):gate.metadata(code/'empty.py')
