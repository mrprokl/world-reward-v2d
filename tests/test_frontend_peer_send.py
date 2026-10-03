"""Tiny isolated sender controls, never SSH/network/model/archive transfer."""
import base64
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
WRAPPER=ROOT/'infra/run_frontend_peer_send.sh'
KEY='ssh-ed25519 '+base64.b64encode(struct.pack('>I',11)+b'ssh-ed25519'+struct.pack('>I',32)+b'k'*32).decode()


def block(name):return WRAPPER.read_text().split("<<'"+name+"'\n")[1].split('\n'+name)[0]


@pytest.fixture
def runtime(tmp_path):
    root=tmp_path/'runtime';code=root/'jobs'/('a'*40)/'run_frontend_peer_send/code';(code/'infra').mkdir(parents=True)
    wrapper=code/'infra/run_frontend_peer_send.sh';wrapper.write_text(WRAPPER.read_text())
    (code.parent/'revision').write_text('a'*40+'\n');(code.parent/'source-sha256').write_text('b'*64+'\n')
    wrapper.chmod(0o444);code.chmod(0o555);wrapper.parent.chmod(0o555)
    fingerprint=hashlib.sha256()
    for name in('revision','source-sha256'):fingerprint.update(name.encode()+b'\0'+(code.parent/name).read_bytes())
    fingerprint.update(b'infra/run_frontend_peer_send.sh\0'+hashlib.sha256(wrapper.read_bytes()).digest());before=fingerprint.hexdigest()
    archive=root/'results/frontend-asset-archive-c84558c673ef8af299dfa187eeb8fde8ce9fc8e1/archive.tar';archive.parent.mkdir(parents=True);archive.write_bytes(b'tiny procedural archive');archive.chmod(0o400)
    archivepin=dict(bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    producer=dict(stage='frontend_asset_archive',status='pass',producer_revision='c84558c673ef8af299dfa187eeb8fde8ce9fc8e1',archive=dict(archive_identity=archivepin))
    receipt=archive.with_name('report.json');receipt.write_text(json.dumps(producer));receipt.chmod(0o400)
    key=root/'transfer/frontend-peer-client-v1/client_ed25519';key.parent.mkdir(parents=True);key.parent.chmod(0o700);key.write_bytes(b'NEVER_READ_PRIVATE_KEY');key.chmod(0o600);key.with_suffix('.pub').write_text(KEY+'\n');key.with_suffix('.pub').chmod(0o400)
    out=root/'results'/('frontend-peer-send-'+'a'*40)
    source=block('PYCONTROL').replace('19911464960',str(archivepin['bytes'])).replace('5b817ea15e98f1f18165529fcac3ca0b7fc9f88b96d22f2195396db6ffbf8342',archivepin['sha256'])
    source=source.replace('3191',str(receipt.stat().st_size)).replace('4c552c18ccb35e76efe40754a5c45d92703506bebc1f8bef2c355075543af0bf',hashlib.sha256(receipt.read_bytes()).hexdigest())
    # Procedural uid metadata only, production remains Linux root owned.
    source=source.replace("s.st_uid)","0)")
    calls=tmp_path/'calls.json'
    prefix=f'''from pathlib import Path
import json,subprocess,types
_read=Path.open
def guarded(self,*args,**kwargs):
 if self.name=='client_ed25519':raise AssertionError('Private key bytes opened by sender Python')
 return _read(self,*args,**kwargs)
Path.open=guarded
def fake(args,**kwargs):
 Path({str(calls)!r}).write_text(json.dumps(args))
 assert kwargs['stdin'].read()==b'tiny procedural archive'
 summary=dict(stage='frontend_peer_receive',status='pass',bytes_received={archivepin['bytes']},receipt_written=True,replica_ready=False)
 kwargs['stdout'].write(json.dumps(summary).encode())
 return types.SimpleNamespace(returncode=0)
subprocess.run=fake
'''
    def run(key_value=KEY,extra=''):
        return subprocess.run(['rtk','proxy',sys.executable,'-I','-B','-',str(root),str(code),'a'*40,str(archive),str(key),str(out),key_value,before],input=prefix+extra+source,capture_output=True,text=True,timeout=10,env=dict(PATH=os.environ['PATH'],HOME=str(tmp_path)))
    return dict(root=root,code=code,key=key,out=out,archive=archive,receipt=receipt,source=source,prefix=prefix,run=run,calls=calls)


def test_sender_exact_ssh_command_no_private_bytes_and_tiny_receipt(runtime):
    result=runtime['run']();assert result.returncode==0,result.stderr
    args=json.loads(runtime['calls'].read_text());assert args[:7]==['ssh','-F','/dev/null','-T','-p','2222','-i']
    assert args[-2:]==['root@10.0.0.9','world-reward-frontend-assets-v1']
    assert 'StrictHostKeyChecking=yes'in args and 'IdentitiesOnly=yes'in args and 'BatchMode=yes'in args
    assert 'GlobalKnownHostsFile=/dev/null'in args
    out=runtime['out'];assert out.stat().st_mode&0o777==0o700
    assert all(p.stat().st_mode&0o777==0o400 for p in out.iterdir())
    report=json.loads((out/'report.json').read_text());assert report['status']=='pass'and report['private_key_bytes_recorded']is False
    assert report['replica_ready']is report['license_eligibility_verified']is False
    assert (out/'known_hosts').read_text()=='[10.0.0.9]:2222 '+KEY+'\n'
    assert 'NEVER_READ_PRIVATE_KEY'not in result.stdout+result.stderr


@pytest.mark.parametrize('value',['ssh-rsa xxx','ssh-ed25519 xxx',KEY+'\nmalicious',KEY+'\x00bad','',KEY.replace('ssh-ed25519','other')])
def test_host_key_rejected_before_output_or_transport(runtime,value):
    if '\x00'in value:
        with pytest.raises(ValueError):runtime['run'](value)
        assert not runtime['calls'].exists()and not runtime['out'].exists();return
    assert runtime['run'](value).returncode!=0 and not runtime['calls'].exists()and not runtime['out'].exists()


@pytest.mark.parametrize('fault',['archive','receipt','keymode','destination'])
def test_original_pins_permissions_no_overwrite_before_ssh(runtime,fault):
    if fault=='archive':runtime['archive'].chmod(0o600);runtime['archive'].write_bytes(b'tampered');runtime['archive'].chmod(0o400)
    elif fault=='receipt':runtime['receipt'].chmod(0o600);runtime['receipt'].write_bytes(b'{}');runtime['receipt'].chmod(0o400)
    elif fault=='keymode':runtime['key'].chmod(0o644)
    else:runtime['out'].mkdir();(runtime['out']/'retained').write_bytes(b'user work')
    assert runtime['run']().returncode!=0 and not runtime['calls'].exists()
    if fault=='destination':assert(runtime['out']/'retained').read_bytes()==b'user work'


@pytest.mark.parametrize('fault',['status','summary','source'])
def test_transport_or_poststream_mutation_fails_without_false_pass(runtime,fault):
    if fault=='status':extra="subprocess.run=lambda *a,**k:types.SimpleNamespace(returncode=1)\n"
    elif fault=='summary':extra="subprocess.run=lambda *a,**k:(k['stdout'].write(b'{\\\"status\\\":\\\"pass\\\"}'),types.SimpleNamespace(returncode=0))[1]\n"
    else:
        extra="_fake=subprocess.run\ndef changed(*a,**k):\n result=_fake(*a,**k);p=Path("+repr(str(runtime['code']/'infra/run_frontend_peer_send.sh'))+");p.chmod(0o600);p.write_text('changed');p.chmod(0o444);return result\nsubprocess.run=changed\n"
    result=runtime['run'](extra=extra);assert result.returncode!=0
    report=json.loads((runtime['out']/'report.json').read_text());assert report['status']=='fail'and not report['archive_sent']


def test_production_bash_no_arbitrary_paths_or_server_changes():
    subprocess.run(['rtk','proxy','bash','-n',str(WRAPPER)],check=True)
    text=WRAPPER.read_text()
    assert '1800s nice -n 15 ionice -c 3'in text and 'ulimit -v 2097152'in text
    assert text.count('env -i PATH=/usr/bin:/bin')==2 and 'scenesmith-ncc-h100-01'in text
    assert not any(x in text for x in('scp ','ssh-keyscan','StrictHostKeyChecking=no','--gpus','extractall','rmtree','az network','ssh-keygen -f'))
