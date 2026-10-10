"""Prepare four one-shot FORM OAuth envelopes; credentials never leave RAM here.

Only public RSA keys use local temporary files. Azure holds fresh private keys
and encrypted envelopes in the predictor's exact per-sequence namespace. A
failed preparation is not retried or cleaned blindly; inspect its owned state.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile

ROOT = '/srv/scenesmith/world-reward'
GROUP, VM = 'SCENESMITH-H100', 'scenesmith-ncc-h100-01'
COHORT = Path(__file__).resolve().parents[1] / 'configs/form_hoi_insight_v1.json'
COHORT_PIN = (10048, '2de3774a16bb9cdf2ad79555d7f3371dc3bab981230a429944ed7b41481a65b6')
MARKER = 'WORLD_REWARD_FORM_AUTH_V1:'


def require(condition):
    if not condition: raise ValueError('Invalid FORM authentication contract')


def strict(raw):
    def unique(rows):
        value = {}
        for key, item in rows: require(key not in value); value[key] = item
        return value
    return json.loads(raw, object_pairs_hook=unique,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite metadata')))


def sequences(revision):
    require(type(revision) is str and re.fullmatch('[0-9a-f]{40}', revision))
    raw = COHORT.read_bytes()
    require((len(raw), hashlib.sha256(raw).hexdigest()) == COHORT_PIN)
    rows = strict(raw)['cohort']; result = [r['sequence_id'] for r in rows if r['split'] == 'development']
    require(len(result) == 4 and len(set(result)) == 4 and all(
        type(s) is str and re.fullmatch('[A-Za-z0-9_-]{1,128}', s) for s in result))
    return result


def command(argv, *, data=None, timeout=45, cap=65536):
    """All local subprocesses are RTK-prefixed and captured, never logged."""
    try:
        result = subprocess.run(['rtk', 'proxy', *argv], input=data, capture_output=True,
                                timeout=timeout, check=False)
        require(result.returncode == 0 and type(result.stdout) is bytes and len(result.stdout) <= cap)
        return result.stdout
    except Exception:
        raise RuntimeError('FORM authentication subprocess unavailable') from None


def remote_script(action, revision, seqs, envelopes=None):
    require(action in {'keys', 'envelopes'})
    payload = dict(action=action, revision=revision, sequences=seqs, envelopes=envelopes)
    # No credentials appear in this script: public metadata and RSA ciphertext only.
    return "set +x\n/usr/bin/python3 -I -B - <<'PY_FORM_AUTH'\n" + f"request={payload!r}\n" + r'''
import base64,json,os,pathlib,re,stat,subprocess,sys
def need(x):
 if not x:raise ValueError('contract')
try:
 need(os.geteuid()==0 and os.uname().nodename=='scenesmith-ncc-h100-01')
 rev=request['revision'];seqs=request['sequences']
 need(re.fullmatch('[0-9a-f]{40}',rev) and len(seqs)==4 and len(set(seqs))==4)
 directory=pathlib.Path('/srv/scenesmith/world-reward/.secrets')
 need(directory.resolve()==directory and not any(p.is_symlink() for p in(directory,*directory.parents)))
 info=directory.stat();need(stat.S_ISDIR(info.st_mode) and not info.st_mode&0o077)
 pairs=[]
 for seq in seqs:
  need(type(seq)is str and re.fullmatch('[A-Za-z0-9_-]{1,128}',seq))
  prefix='form-hoi-external-'+rev+'-'+seq
  pairs.append((seq,directory/(prefix+'-key.pem'),directory/(prefix+'-envelope.enc')))
 if request['action']=='keys':
  # Check all four destinations before creating any; failures preserve owned keys.
  need(all(not os.path.lexists(k) and not os.path.lexists(e) for _,k,e in pairs))
  public=[]
  for seq,key,_ in pairs:
   result=subprocess.run(['/usr/bin/openssl','genpkey','-algorithm','RSA','-pkeyopt','rsa_keygen_bits:4096'],capture_output=True,timeout=60)
   need(result.returncode==0 and 2000<len(result.stdout)<4096)
   fd=os.open(key,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o400)
   with os.fdopen(fd,'wb')as f:f.write(result.stdout);f.flush();os.fsync(f.fileno())
   result=None
   value=subprocess.run(['/usr/bin/openssl','pkey','-in',str(key),'-pubout','-outform','DER'],capture_output=True,timeout=15)
   need(value.returncode==0 and len(value.stdout)==550)
   public.append(dict(sequence_id=seq,public_der=base64.b64encode(value.stdout).decode()))
  reply=dict(status='keys_ready',revision=rev,keys=public)
 else:
  rows=request['envelopes'];need(type(rows)is list and len(rows)==4)
  decoded=[]
  for (seq,key,envelope),row in zip(pairs,rows):
   need(set(row)=={'sequence_id','ciphertext'} and row['sequence_id']==seq and not os.path.lexists(envelope))
   s=key.lstat();need(stat.S_ISREG(s.st_mode)and s.st_uid==0 and s.st_nlink==1 and stat.S_IMODE(s.st_mode)==0o400)
   value=base64.b64decode(row['ciphertext'],validate=True);need(len(value)==512);decoded.append(value)
  for (_,_,envelope),value in zip(pairs,decoded):
   fd=os.open(envelope,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o400)
   with os.fdopen(fd,'wb')as f:f.write(value);f.flush();os.fsync(f.fileno())
  reply=dict(status='prepared',revision=rev,sequences=4)
 print('WORLD_REWARD_FORM_AUTH_V1:'+json.dumps(reply,separators=(',',':')),flush=True)
except Exception as error:
 print(json.dumps(dict(status='fail',error_type=type(error).__name__)),flush=True);sys.exit(1)
''' + '\nPY_FORM_AUTH\n'


def invoke(script):
    raw = command(['az', 'vm', 'run-command', 'invoke', '--resource-group', GROUP,
        '--name', VM, '--command-id', 'RunShellScript', '--scripts', script,
        '-o', 'json', '--only-show-errors'], timeout=300)
    try:
        value = strict(raw); rows = value['value']
        require(type(rows) is list and 1 <= len(rows) <= 8)
        found = [line[len(MARKER):] for row in rows for line in row.get('message', '').splitlines()
                 if line.startswith(MARKER)]
        require(len(found) == 1 and len(found[0]) <= 4000)
        return strict(found[0])
    except Exception:
        raise RuntimeError('FORM Azure preparation unacknowledged; inspect owned state before retry') from None


def encrypt_public(public_der, token):
    """RSA4096 OAEP/SHA256; the plaintext is passed exclusively through stdin."""
    der = base64.b64decode(public_der, validate=True)
    require(len(der) == 550 and type(token) is bytes and 0 < len(token) <= 446)
    with tempfile.TemporaryDirectory(prefix='world-reward-public-rsa-') as folder:
        path = Path(folder) / 'public.pem'
        public_pem = command(['openssl', 'pkey', '-pubin', '-inform', 'DER', '-outform', 'PEM'], data=der)
        require(public_pem.startswith(b'-----BEGIN PUBLIC KEY-----\n') and b'PRIVATE KEY' not in public_pem)
        with path.open('xb') as f: f.write(public_pem)
        ciphertext = command(['openssl', 'pkeyutl', '-encrypt', '-pubin', '-inkey', str(path),
            '-pkeyopt', 'rsa_padding_mode:oaep', '-pkeyopt', 'rsa_oaep_md:sha256'], data=token)
    require(len(ciphertext) == 512)
    return base64.b64encode(ciphertext).decode()


def prepare(revision):
    seqs = sequences(revision)  # All local cohort/revision gates precede service calls.
    keys = invoke(remote_script('keys', revision, seqs))
    require(type(keys) is dict and set(keys) == {'status', 'revision', 'keys'}
            and keys['status'] == 'keys_ready' and keys['revision'] == revision
            and type(keys['keys']) is list and len(keys['keys']) == 4)
    for seq, row in zip(seqs, keys['keys']):
        require(type(row) is dict and set(row) == {'sequence_id', 'public_der'} and row['sequence_id'] == seq
                and type(row['public_der']) is str and len(row['public_der']) == 736)
        require(len(base64.b64decode(row['public_der'], validate=True)) == 550)
    token = None
    try:
        token = command(['gcloud', 'auth', 'print-access-token', '--quiet'], cap=1024).strip()
        require(0 < len(token) <= 446 and not any(chr(ch).isspace() for ch in token))
        envelopes = [dict(sequence_id=row['sequence_id'], ciphertext=encrypt_public(row['public_der'], token))
                     for row in keys['keys']]
        token = None
        result = invoke(remote_script('envelopes', revision, seqs, envelopes))
        require(result == dict(status='prepared', revision=revision, sequences=4))
    finally:
        token = None
    return dict(prepared=True, sequences=4)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--revision', required=True)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(prepare(args.revision), separators=(',', ':')), flush=True)
        return 0
    except Exception as error:
        print(json.dumps(dict(prepared=False, error_type=type(error).__name__)), flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
