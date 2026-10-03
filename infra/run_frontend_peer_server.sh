#!/usr/bin/env bash
# Azure VM02 only. Public-key-restricted, owned, expiring asset receiver; NO extract.
# Source closure: /infra/frontend_peer_receive.py /configs/frontend_asset_archive_pins.json
set +x
set -euo pipefail
[[ $# == 2 && "$1" == --client-public-key ]] || exit 2
exec env -i PATH=/usr/bin:/bin /usr/bin/python3 -I -B - "${WR_ROOT:?}" "${WR_CODE:?}" "${WR_CODE_REVISION:?}" "${BASH_SOURCE[0]}" "$2" <<'PY'
import ast,base64,hashlib,json,os,re,stat,struct,subprocess,sys
from pathlib import Path

ROOT=Path('/srv/scenesmith/world-reward')
DISK=Path('/srv/world-reward-data')
DEST=DISK/'frontend-assets-v1'
CONTROL=Path('/run/world-reward-frontend-peer-v1')
UNIT='world-reward-frontend-peer-sshd-v1.service'
UUID='24df126a-5f5f-41d8-801c-9ddaa7a582d8'
ARCHIVE_BYTES=19_911_464_960
ARCHIVE_SHA='5b817ea15e98f1f18165529fcac3ca0b7fc9f88b96d22f2195396db6ffbf8342'

def require(condition,message):
 if not condition:raise ValueError(message)

def canonical(path):
 require(path.is_absolute() and path.resolve()==path and not any(p.is_symlink() for p in(path,*path.parents)), 'Noncanonical owned path')

def directory(path,private=False,owner=0):
 canonical(path);s=path.lstat()
 require(stat.S_ISDIR(s.st_mode) and s.st_uid==owner and not s.st_mode&0o022 and (not private or stat.S_IMODE(s.st_mode)==0o700),'Expected owned directory required')
 return(s.st_dev,s.st_ino,s.st_mode,s.st_uid)

def regular(path,readonly=False):
 canonical(path);s=path.lstat()
 require(stat.S_ISREG(s.st_mode) and s.st_nlink==1 and s.st_uid==0 and not s.st_mode&0o022 and (not readonly or not s.st_mode&0o222),'Root-owned regular source required')
 require(0<s.st_size<=1_000_000,'Bounded source required')
 raw=path.read_bytes();after=path.lstat()
 require((s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_mode)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_mode),'Source changed while reading')
 return raw,{'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}

def public_key(value,allow_comment=False):
 require(isinstance(value,str) and '\n' not in value and '\r' not in value and len(value)<512,'Single bounded public key required')
 parts=value.split(' ')
 require(len(parts)>=2 and parts[0]=='ssh-ed25519' and (allow_comment or len(parts)==2),'Canonical Ed25519 public key required')
 try:raw=base64.b64decode(parts[1],validate=True)
 except Exception:raise ValueError('Invalid Ed25519 public encoding') from None
 require(len(raw)==51 and raw[:4]==struct.pack('>I',11) and raw[4:15]==b'ssh-ed25519' and raw[15:19]==struct.pack('>I',32) and base64.b64encode(raw).decode()==parts[1],'Canonical Ed25519 public blob required')
 return 'ssh-ed25519 '+parts[1]

def unique_pairs(pairs):
 result={}
 for key,value in pairs:
  require(key not in result,'Duplicate config field');result[key]=value
 return result

def source_identity(code,rev,entry):
 require(re.fullmatch('[0-9a-f]{40}',rev) and code==ROOT/'jobs'/rev/'run_frontend_peer_server/code' and entry==code/'infra/run_frontend_peer_server.sh','Exact immutable server source required')
 canonical(code);digest=hashlib.sha256();identities={}
 for name in('revision','source-sha256'):
  raw,_=regular(code.parent/name)
  require(raw==(rev+'\n').encode() if name=='revision' else bool(re.fullmatch(b'[0-9a-f]{64}\n',raw)),'Original dispatch marker required')
  digest.update(name.encode()+b'\0'+raw)
 for path in(code,*sorted(code.rglob('*'))):
  canonical(path);s=path.lstat()
  require(s.st_uid==0 and not s.st_mode&0o222 and (stat.S_ISREG(s.st_mode) or stat.S_ISDIR(s.st_mode)),'Complete frozen closure required')
  if stat.S_ISREG(s.st_mode):
   raw,pin=regular(path,True);name=str(path.relative_to(code));digest.update(name.encode()+b'\0'+bytes.fromhex(pin['sha256']));identities[name]=pin
 names=('infra/run_frontend_peer_server.sh','infra/frontend_peer_receive.py','configs/frontend_asset_archive_pins.json')
 require(all(n in identities for n in names),'Complete server source closure required')
 cfg=json.loads((code/names[2]).read_bytes(),object_pairs_hook=unique_pairs)
 require(cfg.get('schema')=='world_reward.frontend_asset_archive.pins.v1' and cfg.get('archive_identity')=={'bytes':ARCHIVE_BYTES,'sha256':ARCHIVE_SHA} and cfg.get('replica_ready') is False,'Exact independently frozen archive pins required')
 literals={}
 for node in ast.parse((code/names[1]).read_bytes()).body:
  if isinstance(node,ast.Assign):
   for target in node.targets:
    if isinstance(target,ast.Name) and target.id in('ARCHIVE_BYTES','ARCHIVE_SHA256','COMMAND'):
     literals[target.id]=ast.literal_eval(node.value)
 require(literals=={'ARCHIVE_BYTES':ARCHIVE_BYTES,'ARCHIVE_SHA256':ARCHIVE_SHA,'COMMAND':'world-reward-frontend-assets-v1'},'Receiver archive contract mismatch')
 return {'closure_sha256':digest.hexdigest(),'files':{n:identities[n] for n in names}}

def run(arguments,seconds=8,missing_unit=False):
 try:
  result=subprocess.run(['/usr/bin/timeout','--signal=TERM','--kill-after=2s',str(seconds)+'s',*arguments],capture_output=True,text=True,timeout=seconds+4,check=False)
 except (OSError,subprocess.TimeoutExpired):raise ValueError('Bounded control command unavailable or timed out') from None
 require(result.returncode==0 or (missing_unit and result.returncode in(1,4) and result.stdout.strip()=='not-found'),'Control command failed: '+Path(arguments[0]).name)
 return result.stdout

def configuration(receiver):
 canonical(receiver);require(str(receiver).startswith(str(ROOT)+'/jobs/') and not re.search(r'\s',str(receiver)),'Frozen receiver path required')
 return '\n'.join((
  'ListenAddress 10.0.0.9','Port 2222','Protocol 2',f'HostKey {CONTROL}/host_ed25519',
  f'AuthorizedKeysFile {CONTROL}/authorized_keys',f'ForceCommand /usr/bin/python3 -I -B {receiver} --receive',
  'PermitRootLogin forced-commands-only','AllowUsers root','StrictModes yes','AuthenticationMethods publickey',
  'PubkeyAuthentication yes','PasswordAuthentication no','KbdInteractiveAuthentication no','UsePAM no',
  'PermitUserEnvironment no','PermitUserRC no','PermitTTY no','AllowTcpForwarding no','AllowAgentForwarding no',
  'X11Forwarding no','PermitTunnel no','GatewayPorts no','HostbasedAuthentication no','GSSAPIAuthentication no','AuthorizedKeysCommand none',
  'MaxSessions 1','MaxAuthTries 2','LogLevel ERROR',f'PidFile {CONTROL}/sshd.pid',''))

def exclusive(path,raw):
 canonical(path)
 with os.fdopen(os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,'O_NOFOLLOW',0),0o400),'wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())

def owned_stop():
 # A failed dispatch might have collided with a foreign unit. Never stop one
 # merely because its name matches: require the unique owned config binding.
 projection=run(['/usr/bin/systemctl','show',UNIT,'--property=ExecStart','--property=FragmentPath'])
 fields=dict(line.split('=',1) for line in projection.splitlines() if '=' in line)
 require(fields.get('FragmentPath')=='/run/systemd/transient/'+UNIT and 'path=/usr/sbin/sshd' in fields.get('ExecStart','') and str(CONTROL/'sshd_config') in fields.get('ExecStart',''),'Owned transient command binding required for stop')
 run(['/usr/bin/systemctl','stop',UNIT],8)

def main(arguments):
 owned=False;started=False;start_attempted=False;phase='preflight'
 try:
  require(len(arguments)==5,'Exact server arguments required')
  root,code,rev,entry,key=arguments;code,entry=Path(code),Path(entry)
  require(root==str(ROOT) and sys.platform=='linux' and os.geteuid()==0 and os.uname().nodename=='world-reward-ncc-h100-02','Exact Azure root host required')
  key=public_key(key);before=source_identity(code,rev,entry)
  # Existing bootstrap workspace/data mounts belong to scenesmith (UID1000).
  # Do not chown those shared parents; all new private children remain root-owned.
  directory(ROOT,owner=1000);directory(CONTROL.parent);directory(DISK,owner=1000)
  directory(Path('/run/sshd'))
  for path in(CONTROL,DEST):canonical(path);require(not path.exists(),'Fresh owned namespace required; no overwrite or retry')
  require(run(['/usr/bin/systemctl','show',UNIT,'--property=LoadState','--value'],missing_unit=True).strip()=='not-found','Absent owned server unit required')
  mount=json.loads(run(['/usr/bin/findmnt','--json','--mountpoint',str(DISK),'--output','TARGET,UUID,OPTIONS']))
  require(len(mount.get('filesystems',[]))==1,'Exact data mount required');fs=mount['filesystems'][0]
  require(fs.get('target')==str(DISK) and fs.get('uuid')==UUID and 'rw' in fs.get('options','').split(','),'Exact writable data disk UUID required')
  space=os.statvfs(DISK);require(space.f_bavail*space.f_frsize>45*1024**3,'More than 45GiB free required')
  os.umask(0o077);CONTROL.mkdir(mode=0o700);owned=True;control_state=directory(CONTROL,True)
  DEST.mkdir(mode=0o700);destination_state=directory(DEST,True);phase='host_key'
  run(['/usr/bin/ssh-keygen','-q','-t','ed25519','-N','','-f',str(CONTROL/'host_ed25519')],15)
  secret=(CONTROL/'host_ed25519').lstat()
  require(stat.S_ISREG(secret.st_mode) and secret.st_nlink==1 and secret.st_uid==0 and stat.S_IMODE(secret.st_mode)==0o600,'Owned private host key mode required')
  # The private key is NEVER read, hashed, copied, printed or included in receipts.
  os.chmod(CONTROL/'host_ed25519.pub',0o400);raw,pub_pin=regular(CONTROL/'host_ed25519.pub',True)
  host_public=public_key(raw.decode().rstrip('\n'),True)
  exclusive(CONTROL/'authorized_keys',(f'from="10.0.0.4",restrict {key}\n').encode())
  exclusive(CONTROL/'sshd_config',configuration(code/'infra/frontend_peer_receive.py').encode())
  public_control={name:regular(CONTROL/name,True)[1] for name in('authorized_keys','sshd_config','host_ed25519.pub')}
  phase='sshd_validation';run(['/usr/sbin/sshd','-t','-f',str(CONTROL/'sshd_config')])
  require(source_identity(code,rev,entry)==before,'Frozen source changed before server start')
  require(run(['/usr/bin/systemctl','show',UNIT,'--property=LoadState','--value'],missing_unit=True).strip()=='not-found','Owned server unit appeared')
  phase='server_start'
  start_attempted=True
  run(['/usr/bin/systemd-run','--quiet','--unit='+UNIT,'--property=Type=exec','--property=Restart=no','--property=RuntimeMaxSec=2400','/usr/sbin/sshd','-D','-e','-f',str(CONTROL/'sshd_config')]);started=True
  state=run(['/usr/bin/systemctl','show',UNIT,'--property=ActiveState','--property=MainPID'])
  values=dict(line.split('=',1) for line in state.splitlines() if '=' in line)
  require(values.get('ActiveState')=='active' and re.fullmatch('[1-9][0-9]*',values.get('MainPID','')),'Owned server not active')
  phase='postcheck';require(source_identity(code,rev,entry)==before,'Frozen source changed after server start')
  require(directory(CONTROL,True)==control_state and directory(DEST,True)==destination_state and not list(DEST.iterdir()),'Owned control or fresh incoming directory changed')
  require({name:regular(CONTROL/name,True)[1] for name in public_control}==public_control,'Public server control changed')
  receipt={'schema':'world_reward.frontend_peer_server.v1','status':'pass','stage':'frontend_peer_server','producer_revision':rev,'source_identity':before,
   'listen_address':'10.0.0.9','port':2222,'allowed_client_address':'10.0.0.4','unit':UNIT,'active_state':'active','main_pid':int(values['MainPID']),
   'runtime_max_seconds':2400,'archive_identity':{'bytes':ARCHIVE_BYTES,'sha256':ARCHIVE_SHA},'incoming_path':str(DEST),'data_disk_uuid':UUID,
   'host_public_key':host_public,'host_public_file_identity':pub_pin,'public_control_identities':public_control,'source_rechecked_before_and_after':True,'private_key_read':False,
   'nsg_changed':False,'extracted':False,'replica_ready':False,'license_eligibility_verified':False,'training_overlap_verified':False}
  raw=(json.dumps(receipt,sort_keys=True,separators=(',',':'))+'\n').encode();require(len(raw)<=4096,'Bounded public receipt required')
  exclusive(CONTROL/'server-receipt.json',raw);print(raw.decode(),end='');return 0
 except Exception as error:
  cleanup=None
  if start_attempted:
   try:owned_stop();cleanup=True
   except Exception:cleanup=False
  failure={'schema':'world_reward.frontend_peer_server.v1','status':'fail','stage':'frontend_peer_server','failed_phase':phase,
   'error':str(error)[:180] if isinstance(error,ValueError) else type(error).__name__,'owned_control_created':owned,'server_start_attempted':start_attempted,'server_started':started,'owned_stop_command_succeeded':cleanup,'replica_ready':False}
  raw=(json.dumps(failure,sort_keys=True,separators=(',',':'))+'\n').encode()
  if owned:
   try:exclusive(CONTROL/'server-failure.json',raw)
   except Exception:pass
  print(raw.decode(),end='');return 1

if __name__=='__main__':sys.exit(main(sys.argv[1:]))
PY
