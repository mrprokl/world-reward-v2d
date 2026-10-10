"""Fake service replies and ephemeral public-key encryption; no Azure/GCP calls."""
import base64
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'infra'))
import form_localization_auth as auth


def public_reply(revision):
    return dict(status='keys_ready', revision=revision, keys=[dict(sequence_id=s,
        public_der=base64.b64encode(b'P' * 550).decode()) for s in auth.sequences(revision)])


@pytest.mark.parametrize('revision', ['a' * 39, 'a' * 41, 'A' * 40, 'a' * 39 + ';', ''])
def test_revision_gate_before_any_service_call(monkeypatch, revision):
    monkeypatch.setattr(auth, 'command', lambda *a, **kw: pytest.fail('No service call permitted'))
    with pytest.raises(ValueError): auth.prepare(revision)


def test_cohort_is_exact_four_development_not_reserved():
    seqs = auth.sequences('a' * 40)
    assert len(seqs) == 4 and all('guitar' not in s and 'snack' not in s for s in seqs)
    assert seqs[0] == '2026-06-03_17-02-20_beige_bin_ground_desk_03'


def test_changed_cohort_rejected_before_calls(tmp_path, monkeypatch):
    path = tmp_path / 'cohort.json'; path.write_text('{}')
    monkeypatch.setattr(auth, 'COHORT', path)
    monkeypatch.setattr(auth, 'command', lambda *a, **kw: pytest.fail('No service call permitted'))
    with pytest.raises(ValueError): auth.prepare('a' * 40)


def test_two_serialized_remote_phases_token_only_in_encryption_stdin(monkeypatch, capsys):
    rev = 'a' * 40; calls = []; fake = b'FAKE_TOKEN_FOR_TEST_ONLY'
    def invoke(script):
        calls.append(('remote', script))
        assert fake.decode() not in script
        if len(calls) == 1: return public_reply(rev)
        return dict(status='prepared', revision=rev, sequences=4)
    def command(argv, **kwargs):
        calls.append(('local', argv))
        assert argv == ['gcloud', 'auth', 'print-access-token', '--quiet']
        return fake + b'\n'
    def encrypt(public, token):
        assert token == fake and len(base64.b64decode(public)) == 550
        calls.append(('encrypt', None)); return base64.b64encode(b'C' * 512).decode()
    monkeypatch.setattr(auth, 'invoke', invoke)
    monkeypatch.setattr(auth, 'command', command)
    monkeypatch.setattr(auth, 'encrypt_public', encrypt)
    assert auth.prepare(rev) == dict(prepared=True, sequences=4)
    assert [row[0] for row in calls] == ['remote', 'local', 'encrypt', 'encrypt', 'encrypt', 'encrypt', 'remote']
    assert capsys.readouterr().out == ''


def test_subprocess_is_rtk_captured_and_errors_are_redacted(monkeypatch, capsys):
    calls = []
    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(returncode=1, stdout=b'PRIVATE_TOKEN', stderr=b'PRIVATE_ERROR')
    monkeypatch.setattr(auth.subprocess, 'run', run)
    with pytest.raises(RuntimeError) as error: auth.command(['gcloud', 'auth', 'print-access-token'])
    assert str(error.value) == 'FORM authentication subprocess unavailable'
    assert calls[0][0][:2] == ['rtk', 'proxy'] and calls[0][1]['capture_output']
    assert capsys.readouterr().out == ''


def test_exact_azure_target_and_marked_json_reply(monkeypatch):
    commands = []
    def command(argv, **kwargs):
        commands.append(argv)
        return json.dumps(dict(value=[dict(message='[stdout]\n' + auth.MARKER + '{"status":"prepared"}\n[stderr]\n')])).encode()
    monkeypatch.setattr(auth, 'command', command)
    assert auth.invoke('PUBLIC_SCRIPT_ONLY') == {'status': 'prepared'}
    assert commands[0][:4] == ['az', 'vm', 'run-command', 'invoke']
    assert commands[0][5] == 'SCENESMITH-H100' and commands[0][7] == 'scenesmith-ncc-h100-01'
    assert '-o' in commands[0] and 'json' in commands[0]


@pytest.mark.parametrize('reply', [dict(value=[]), dict(value=[dict(message='unacknowledged')]),
    dict(value=[dict(message=auth.MARKER + '{}\n' + auth.MARKER + '{}')])])
def test_unknown_remote_state_never_retries_or_prints_reply(monkeypatch, reply):
    monkeypatch.setattr(auth, 'command', lambda *a, **kw: json.dumps(reply).encode())
    with pytest.raises(RuntimeError) as error: auth.invoke('public')
    assert 'inspect owned state before retry' in str(error.value)


def test_remote_script_checks_all_fresh_before_creation_and_exact_predictor_paths():
    revision = 'a' * 40; seqs = auth.sequences(revision)
    script = auth.remote_script('keys', revision, seqs)
    assert script.index('need(all(not os.path.lexists(k)') < script.index("'/usr/bin/openssl','genpkey'")
    assert 'rsa_keygen_bits:4096' in script and 'os.O_EXCL|os.O_NOFOLLOW,0o400' in script
    assert "prefix='form-hoi-external-'+rev+'-'+seq" in script
    assert "directory/(prefix+'-key.pem')" in script and "directory/(prefix+'-envelope.enc')" in script
    assert 'unlink' not in script and 'print(result.stdout)' not in script and 'PRIVATE KEY' not in script


def test_real_ephemeral_public_RSA_encrypts_fake_token_without_plaintext_file(tmp_path, monkeypatch):
    # Test-only private RSA exists in RAM and is never passed to the helper/file.
    generated = subprocess.run(['rtk', 'proxy', 'openssl', 'genpkey', '-algorithm', 'RSA',
        '-pkeyopt', 'rsa_keygen_bits:4096'], capture_output=True, check=True, timeout=30).stdout
    der = subprocess.run(['rtk', 'proxy', 'openssl', 'pkey', '-pubout', '-outform', 'DER'],
        input=generated, capture_output=True, check=True, timeout=15).stdout
    assert len(der) == 550
    fake = b'FAKE_TOKEN_NO_SERVICE'
    seen = []; original = auth.command
    def observe(argv, **kwargs):
        if '-inkey' in argv:
            files = list(Path(argv[argv.index('-inkey') + 1]).parent.iterdir())
            assert len(files) == 1 and files[0].name == 'public.pem'
            raw = files[0].read_bytes(); assert b'PUBLIC KEY' in raw and fake not in raw and b'PRIVATE KEY' not in raw
            seen.append(files[0])
            assert kwargs['data'] == fake
        return original(argv, **kwargs)
    monkeypatch.setattr(auth, 'command', observe)
    value = auth.encrypt_public(base64.b64encode(der).decode(), fake)
    assert len(base64.b64decode(value)) == 512
    assert seen and not seen[0].exists() and list(tmp_path.iterdir()) == []


def test_cli_only_summary_and_typed_failure_no_credential_exception(monkeypatch, capsys):
    def fail(_): raise RuntimeError('FAKE_TOKEN_PRIVATE_REPLY')
    monkeypatch.setattr(auth, 'prepare', fail)
    assert auth.main(['--revision', 'a' * 40]) == 1
    assert json.loads(capsys.readouterr().out) == dict(prepared=False, error_type='RuntimeError')
