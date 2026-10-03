"""Procedural CPU-only contract tests; no sshd, keys, network or Azure execution."""
import ast
import base64
from pathlib import Path
import struct
import subprocess

import pytest

WRAPPER=Path(__file__).resolve().parents[1]/'infra/run_frontend_peer_server.sh'
TEXT=WRAPPER.read_text()
SOURCE=TEXT.split("<<'PY'\n",1)[1].rsplit('\nPY',1)[0]

@pytest.fixture
def api():
 namespace={'__name__':'test_frontend_peer_server'}
 exec(compile(SOURCE,str(WRAPPER),'exec'),namespace)
 return namespace

def key(payload=b'x'*32):
 return 'ssh-ed25519 '+base64.b64encode(struct.pack('>I',11)+b'ssh-ed25519'+struct.pack('>I',32)+payload).decode()

def test_canonical_public_key(api):
 assert api['public_key'](key())==key()

@pytest.mark.parametrize('value',['', 'ssh-rsa AAAA','ssh-ed25519 AAAA',key()+' comment',key()+'\n',key().replace(' ','  '),key()+'\r','ssh-ed25519 '+base64.b64encode(b'\0'*51).decode(),key(b'x'*31),key()[:-1]])
def test_invalid_client_public_keys(api,value):
 with pytest.raises(ValueError):api['public_key'](value)

def test_generated_host_public_comment_is_not_forwarded(api):
 assert api['public_key'](key()+' root@host',True)==key()

def test_config_only_owned_receiver_and_explicit_isolation(api):
 receiver=api['ROOT']/'jobs'/('a'*40)/'run_frontend_peer_server/code/infra/frontend_peer_receive.py'
 config=api['configuration'](receiver)
 values=dict(line.split(' ',1) for line in config.splitlines())
 expected={'ListenAddress':'10.0.0.9','Port':'2222','Protocol':'2','PermitRootLogin':'forced-commands-only','AllowUsers':'root','StrictModes':'yes','PasswordAuthentication':'no','KbdInteractiveAuthentication':'no','UsePAM':'no','PermitUserEnvironment':'no','AllowTcpForwarding':'no','AllowAgentForwarding':'no','X11Forwarding':'no','PermitTunnel':'no','PermitTTY':'no','PermitUserRC':'no','MaxSessions':'1','MaxAuthTries':'2','LogLevel':'ERROR','AuthenticationMethods':'publickey'}
 assert all(values[k]==v for k,v in expected.items())
 assert values['ForceCommand']==f'/usr/bin/python3 -I -B {receiver} --receive'
 assert values['HostKey']==str(api['CONTROL']/'host_ed25519')
 assert values['AuthorizedKeysFile']==str(api['CONTROL']/'authorized_keys')
 assert 'Include' not in values and 'AcceptEnv' not in values

@pytest.mark.parametrize('path',['/tmp/receiver.py','/srv/scenesmith/world-reward/jobs/has space/receiver.py','/srv/scenesmith/world-reward/jobs/../receiver.py'])
def test_foreign_or_ambiguous_receiver_rejected(api,path):
 with pytest.raises(ValueError):api['configuration'](Path(path))

def test_json_duplicate_pins_rejected(api):
 with pytest.raises(ValueError):api['unique_pairs']([('archive_identity',1),('archive_identity',2)])

def test_bounded_command_projection(api,monkeypatch):
 calls=[]
 def fake(arguments,**kwargs):
  calls.append((arguments,kwargs));return subprocess.CompletedProcess(arguments,0,'active\n','not surfaced')
 monkeypatch.setattr(api['subprocess'],'run',fake)
 assert api['run'](['/usr/bin/systemctl','show','owned'])=='active\n'
 args,kwargs=calls[0]
 assert args[:4]==['/usr/bin/timeout','--signal=TERM','--kill-after=2s','8s']
 assert kwargs['timeout']==12 and kwargs['capture_output'] and kwargs['check'] is False

def test_control_errors_never_surface_subprocess_output(api,monkeypatch):
 monkeypatch.setattr(api['subprocess'],'run',lambda args,**kw:subprocess.CompletedProcess(args,1,'DO_NOT_LOG','DO_NOT_LOG'))
 with pytest.raises(ValueError,match='Control command failed: systemctl') as caught:api['run'](['/usr/bin/systemctl','show','owned'])
 assert 'DO_NOT_LOG' not in str(caught.value)

def test_control_timeout_sanitized(api,monkeypatch):
 def fail(*args,**kwargs):raise subprocess.TimeoutExpired('DO_NOT_LOG',10,output='DO_NOT_LOG')
 monkeypatch.setattr(api['subprocess'],'run',fail)
 with pytest.raises(ValueError,match='timed out') as caught:api['run'](['/usr/bin/systemctl'])
 assert 'DO_NOT_LOG' not in str(caught.value)

@pytest.mark.parametrize('code',[0,1,4])
def test_exact_not_found_control_projection_accepted(api,monkeypatch,code):
 monkeypatch.setattr(api['subprocess'],'run',lambda args,**kw:subprocess.CompletedProcess(args,code,'not-found\n','ignored'))
 assert api['run'](['/usr/bin/systemctl','show'],missing_unit=True)=='not-found\n'

@pytest.mark.parametrize('output',['','loaded\n','not-found\nother\n'])
def test_missing_unit_noncanonical_projection_rejected(api,monkeypatch,output):
 monkeypatch.setattr(api['subprocess'],'run',lambda args,**kw:subprocess.CompletedProcess(args,4,output,'ignored'))
 with pytest.raises(ValueError):api['run'](['/usr/bin/systemctl','show'],missing_unit=True)

def test_exclusive_receipt_no_overwrite(api,tmp_path):
 target=tmp_path/'receipt.json';api['exclusive'](target,b'first')
 assert target.stat().st_mode&0o777==0o400
 with pytest.raises(FileExistsError):api['exclusive'](target,b'second')
 assert target.read_bytes()==b'first'

def test_symlink_namespace_rejected(api,tmp_path):
 actual=tmp_path/'actual';actual.mkdir();link=tmp_path/'link';link.symlink_to(actual,target_is_directory=True)
 with pytest.raises(ValueError):api['canonical'](link/'receipt')

def test_static_exact_source_archive_unit_and_no_private_key_reads(api):
 receiver=WRAPPER.parent/'frontend_peer_receive.py'
 literals={}
 for node in ast.parse(receiver.read_bytes()).body:
  if isinstance(node,ast.Assign):
   for target in node.targets:
    if isinstance(target,ast.Name) and target.id in('ARCHIVE_BYTES','ARCHIVE_SHA256'):literals[target.id]=ast.literal_eval(node.value)
 assert literals=={'ARCHIVE_BYTES':api['ARCHIVE_BYTES'],'ARCHIVE_SHA256':api['ARCHIVE_SHA']}
 assert api['UNIT']=='world-reward-frontend-peer-sshd-v2.service'
 assert api['CONTROL']==Path('/run/world-reward-frontend-peer-v2')
 assert "directory(ROOT,owner=1000)" in SOURCE and "directory(DISK,owner=1000)" in SOURCE
 assert api['UUID']=='24df126a-5f5f-41d8-801c-9ddaa7a582d8'
 assert "'--property=Type=exec','--property=Restart=no','--property=RuntimeMaxSec=2400'" in SOURCE
 assert "'/usr/sbin/sshd','-D','-e','-f'" in SOURCE
 assert 'from="10.0.0.4",restrict,command="{forced}"' in SOURCE
 assert "45*1024**3" in SOURCE
 assert "'host_ed25519').read" not in SOURCE
 assert 'docker' not in SOURCE and 'nvidia-smi' not in SOURCE and 'iptables' not in SOURCE

@pytest.mark.parametrize('actual,expected,accepted',[(1000,1000,True),(1000,0,False),(0,0,True),(99,1000,False)])
def test_expected_existing_owner_does_not_weaken_new_root_control(api,monkeypatch,tmp_path,actual,expected,accepted):
 path=tmp_path/'safe';path.mkdir(mode=0o700);s=path.stat()
 monkeypatch.setattr(Path,'lstat',lambda self:type('S',(),{'st_mode':s.st_mode,'st_uid':actual,'st_dev':s.st_dev,'st_ino':s.st_ino})())
 monkeypatch.setitem(api,'canonical',lambda _:None)
 if accepted:assert api['directory'](path,True,owner=expected)[3]==actual
 else:
  with pytest.raises(ValueError):api['directory'](path,True,owner=expected)

def test_preflight_failure_no_mutation_or_keys(api,monkeypatch,capsys):
 monkeypatch.setattr(api['os'],'geteuid',lambda:99)
 assert api['main']([str(api['ROOT']),'/foreign','a'*40,'/foreign/entry',key()])==1
 output=capsys.readouterr().out
 assert '"status":"fail"' in output and '"owned_control_created":false' in output
 assert key() not in output

@pytest.mark.parametrize('projection',['ExecStart=foreign\nFragmentPath=/run/systemd/transient/world-reward-frontend-peer-sshd-v1.service\n', 'ExecStart=path=/usr/sbin/sshd\nFragmentPath=/etc/systemd/system/foreign\n'])
def test_failed_dispatch_never_stops_foreign_unit(api,monkeypatch,projection):
 calls=[]
 def fake(args,*a,**kw):calls.append(args);return projection
 monkeypatch.setitem(api,'run',fake)
 with pytest.raises(ValueError):api['owned_stop']()
 assert len(calls)==1 and 'stop' not in calls[0]

def test_cleanup_stops_only_bound_owned_transient(api,monkeypatch):
 calls=[]
 def fake(args,*a,**kw):
  calls.append(args)
  return f'ExecStart={{ path=/usr/sbin/sshd ; argv[]=/usr/sbin/sshd -D -e -f {api["CONTROL"]}/sshd_config ; }}\nFragmentPath=/run/systemd/transient/{api["UNIT"]}\n'
 monkeypatch.setitem(api,'run',fake);api['owned_stop']()
 assert calls[-1]==['/usr/bin/systemctl','stop',api['UNIT']]

def test_mocked_owned_setup_actual_control_command_sequence(api,monkeypatch,tmp_path,capsys):
 root=tmp_path/'root';disk=tmp_path/'disk';(root/'runtime').mkdir(parents=True);disk.mkdir()
 control=root/'runtime/frontend_peer_v1';destination=disk/'frontend-assets-v1'
 monkeypatch.setitem(api,'ROOT',root);monkeypatch.setitem(api,'DISK',disk);monkeypatch.setitem(api,'CONTROL',control);monkeypatch.setitem(api,'DEST',destination)
 monkeypatch.setattr(api['sys'],'platform','linux');monkeypatch.setattr(api['os'],'geteuid',lambda:0)
 monkeypatch.setattr(api['os'],'uname',lambda:type('U',(),{'nodename':'world-reward-ncc-h100-02'})())
 monkeypatch.setattr(api['os'],'statvfs',lambda path:type('S',(),{'f_bavail':100*1024**3,'f_frsize':1})())
 directory=api['directory'];regular=api['regular']
 def test_directory(path,private=False,owner=0):
  if path==Path('/run/sshd'):return (1,1,0o40755,0)
  if private:assert path.stat().st_mode&0o777==0o700
  return (path.stat().st_dev,path.stat().st_ino,path.stat().st_mode,0)
 def test_regular(path,readonly=False):
  raw=path.read_bytes();return raw,{'bytes':len(raw),'sha256':api['hashlib'].sha256(raw).hexdigest()}
 monkeypatch.setitem(api,'directory',test_directory);monkeypatch.setitem(api,'regular',test_regular)
 monkeypatch.setitem(api,'source_identity',lambda *args:{'closure_sha256':'a'*64,'files':{}})
 # Keep actual private-key permission validation but project root ownership.
 original_lstat=Path.lstat
 def root_stat(path):
  s=original_lstat(path)
  if path.name=='host_ed25519':return type('S',(),{'st_mode':s.st_mode,'st_nlink':s.st_nlink,'st_uid':0})()
  return s
 monkeypatch.setattr(Path,'lstat',root_stat)
 calls=[]
 def fake(args,*a,**kw):
  calls.append(args)
  if args[0]=='/usr/bin/ssh-keygen':
   (control/'host_ed25519').write_bytes(b'procedural placeholder');(control/'host_ed25519').chmod(0o600)
   (control/'host_ed25519.pub').write_text(key()+' fixture\n');return ''
  if args[0]=='/usr/bin/findmnt':return api['json'].dumps({'filesystems':[{'target':str(disk),'uuid':api['UUID'],'options':'rw,noatime'}]})
  if args[0]=='/usr/bin/systemctl':return 'not-found\n' if '--property=LoadState' in args else 'ActiveState=active\nMainPID=1234\n'
  return ''
 monkeypatch.setitem(api,'run',fake)
 rev='b'*40;code=root/'jobs'/rev/'run_frontend_peer_server/code'
 old_umask=api['os'].umask(0o077)
 try:assert api['main']([str(root),str(code),rev,str(code/'infra/run_frontend_peer_server.sh'),key()])==0
 finally:api['os'].umask(old_umask)
 output=api['json'].loads(capsys.readouterr().out)
 assert output['status']=='pass' and output['replica_ready'] is False and output['private_key_read'] is False
 assert output['host_public_key']==key() and 'procedural placeholder' not in str(output)
 assert (control/'authorized_keys').read_text()==f'from="10.0.0.4",restrict,command="/usr/bin/python3 -I -B {code}/infra/frontend_peer_receive.py --receive" {key()}\n'
 assert (control/'server-receipt.json').stat().st_mode&0o777==0o400
 assert list(destination.iterdir())==[]
 command=next(args for args in calls if args[0]=='/usr/bin/systemd-run')
 assert command[-5:]==['/usr/sbin/sshd','-D','-e','-f',str(control/'sshd_config')]
