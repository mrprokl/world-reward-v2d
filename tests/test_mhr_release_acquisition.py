"""Tiny injected release fixtures only: no real downloads, JIT loads or HF."""
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import stat
import sys
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value); return value

acq = module('release_acquisition_test', ROOT/'infra/acquire_weights.py')
rt = module('release_identity_test', ROOT/'infra/mediapipe_cpu_runtime_verify.py')
def identity(raw): return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def archive_bytes(entries):
    raw = io.BytesIO()
    with zipfile.ZipFile(raw, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, data, mode in entries:
            info = zipfile.ZipInfo(name); info.external_attr = mode << 16
            info.compress_type = zipfile.ZIP_DEFLATED; archive.writestr(info, data)
    return raw.getvalue()


@pytest.fixture
def release(tmp_path, monkeypatch):
    root=tmp_path/'remote';root.mkdir();(root/'results').mkdir();rev='a'*40
    code=root/'jobs'/rev/'acquire_weights'/'code';code.mkdir(parents=True)
    protocol=json.loads((ROOT/acq.MHR_PROTOCOL).read_text());model=b'opaque tiny JIT-not-loaded fixture'
    license=b'Apache License Version 2.0 fixture';readme=b'MHR is licensed under Apache-2.0 fixture'
    protocol['model'].update(identity(model),accepted_uids=[__import__('os').getuid()])
    protocol['primary_texts']['LICENSE'].update(identity(license));protocol['primary_texts']['README.md'].update(identity(readme))
    context=dict(root=root,code=code,revision=rev,protocol=protocol,model=model,license=license,readme=readme,calls=[])
    context['entries']=[('assets/LICENSE',license,stat.S_IFREG|0o644),('assets/mhr_model.pt',model,stat.S_IFREG|0o644)]
    context['archive']=archive_bytes(context['entries'])
    path=root/protocol['model']['existing_path'];path.parent.mkdir(parents=True);path.write_bytes(model);path.chmod(0o644)
    context['model_path']=path
    def freeze():
        protocol['archive'].update(identity(context['archive']))
        raw=(json.dumps(protocol,sort_keys=True)+'\n').encode();monkeypatch.setattr(acq,'MHR_PROTOCOL_PIN',identity(raw))
        for name in acq.MHR_HELPERS:
            p=code/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw if name==acq.MHR_PROTOCOL else b'# frozen fixture source\n');p.chmod(0o444)
        for name,text in [('revision',rev),('source-sha256','b'*64)]:
            p=code.parent/name;p.write_text(text+'\n');p.chmod(0o444)
        for p in sorted(code.rglob('*'),reverse=True):
            if p.is_dir():p.chmod(0o555)
        code.chmod(0o555)
    def fetch(url,cap):
        context['calls'].append(('fetch',url,cap))
        if url==protocol['release']['url']:
            value=dict(tag_name='v1.0.1',published_at=protocol['release']['published_at'],body=protocol['release']['body'],draft=False,prerelease=False,
                assets=[dict(name='assets.zip',size=protocol['archive']['bytes'],digest='sha256:'+protocol['archive']['sha256'],browser_download_url=protocol['archive']['url'])])
            value.update(context.get('metadata_override',{}));return json.dumps(value).encode()
        return license if url==protocol['primary_texts']['LICENSE']['url'] else readme
    def download(url,path,cap):
        context['calls'].append(('download',url,cap));assert path.stat().st_size==0
        path.write_bytes(context['archive'])
        if context.get('mutation'):context['mutation']()
    def run():
        freeze();return acq.mhr_release(root,code,rev,rt,fetch=fetch,download=download)
    context.update(run=run,freeze=freeze,fetch=fetch,download=download)
    yield context
    # Only this manufactured fixture's source/receipt directories are made deletable.
    for p in root.rglob('*'):
        if p.is_dir():p.chmod(0o700)


def report(release):return json.loads((release['root']/release['protocol']['namespace']/'report.json').read_text())


def test_complete_independent_proof_no_duplicate_model(release):
    old=release['model_path'].stat();r=release['run']();out=release['root']/release['protocol']['namespace']
    assert r['status']=='pass' and r['model_byte_identical'] and r['asset_license_matches_primary_exactly']
    assert r['source_rehashed_after'] and r['existing_model_rehashed_after'] and r['owned_archive_removed']
    assert r['selected_members_crc_verified'] and not r['all_members_crc_verified']
    assert len(r['archive_inventory'])==2 and r['expanded_bytes']==len(release['model'])+len(release['license'])
    assert {p.name for p in out.iterdir()}=={'primary-LICENSE','primary-README.md','release-metadata.json','asset-notice-0.txt','report.json'}
    assert (out/'asset-notice-0.txt').read_bytes()==release['license'] and out.stat().st_mode&0o777==0o555
    assert all(p.stat().st_mode&0o777==0o444 for p in out.iterdir())
    assert release['model_path'].stat().st_ino==old.st_ino and release['model_path'].stat().st_mode==old.st_mode
    assert release['model_path'].read_bytes()==release['model'] and not r['model_copy_written']
    assert all(r[k] is False for k in ('models_loaded','packages_installed','gpu_used','dataset_read','sam_provenance_relabelled','competition_eligibility_verified'))


@pytest.mark.parametrize('kind',['missing_license','bad_license','bad_member_sha','traversal','absolute','backslash','duplicate','symlink','members_cap','expanded_cap','notice_cap'])
def test_archive_fail_closed_and_owned_partial_removed(release,kind):
    rows=list(release['entries']);limits=release['protocol']['archive_limits']
    if kind=='missing_license':rows=rows[1:]
    elif kind=='bad_license':rows[0]=(rows[0][0],b'Apache token NOT full grant',rows[0][2])
    elif kind=='bad_member_sha':rows[1]=(rows[1][0],b'x'*len(rows[1][1]),rows[1][2])
    elif kind in ('traversal','absolute','backslash'):rows.append(({'traversal':'../evil','absolute':'/evil','backslash':'assets\\evil'}[kind],b'x',stat.S_IFREG|0o644))
    elif kind=='duplicate':rows.append(rows[0])
    elif kind=='symlink':rows.append(('assets/link',b'mhr_model.pt',stat.S_IFLNK|0o777))
    elif kind=='members_cap':limits['members']=1
    elif kind=='expanded_cap':limits['expanded_bytes']=1
    elif kind=='notice_cap':limits['notice_bytes']=1
    release['archive']=archive_bytes(rows)
    with pytest.raises(RuntimeError):release['run']()
    r=report(release);assert r['status']=='fail' and r['owned_archive_removed']
    assert not (release['root']/release['protocol']['namespace']/'assets.zip.part').exists()
    assert release['model_path'].read_bytes()==release['model']
    if kind=='bad_license':assert r['asset_notices'][0]['sha256']==identity(b'Apache token NOT full grant')['sha256']


def test_whole_sha_precedes_zip_interpretation(release,monkeypatch):
    release['freeze']();release['archive']=b'not a ZIP with wrong archive bytes'
    monkeypatch.setattr(acq.zipfile,'ZipFile',lambda *_:pytest.fail('ZIP interpreted before published SHA'))
    with pytest.raises(RuntimeError):acq.mhr_release(release['root'],release['code'],release['revision'],rt,fetch=release['fetch'],download=release['download'])
    assert report(release)['phase']=='download'


@pytest.mark.parametrize('metadata',[{'draft':True},{'prerelease':True},{'tag_name':'v1.0.0'},{'body':'different'},{'assets':[]}])
def test_primary_metadata_before_download(release,metadata):
    release['metadata_override']=metadata
    with pytest.raises(RuntimeError):release['run']()
    assert not any(row[0]=='download' for row in release['calls'])


@pytest.mark.parametrize('target',['model','source'])
def test_tampered_original_fails_posthash(release,target):
    p=release['model_path'] if target=='model' else release['code']/'infra/acquire_weights.py'
    def mutate():p.chmod(0o644);p.write_bytes(b'mutated')
    release['mutation']=mutate
    with pytest.raises(RuntimeError):release['run']()
    assert report(release)['status']=='fail' and report(release)['postcheck_error_type']=='ValueError'


def test_existing_output_never_modified(release):
    out=release['root']/release['protocol']['namespace'];out.mkdir();(out/'report.json').write_bytes(b'historical failure')
    with pytest.raises(RuntimeError):release['run']()
    assert (out/'report.json').read_bytes()==b'historical failure' and not release['calls']


def test_interrupted_download_only_owned_partial_cleanup(release):
    release['freeze']()
    def interrupted(_url,path,_cap):path.write_bytes(b'owned partial');raise TimeoutError('fixture')
    with pytest.raises(RuntimeError):acq.mhr_release(release['root'],release['code'],release['revision'],rt,fetch=release['fetch'],download=interrupted)
    r=report(release);assert r['error_type']=='TimeoutError' and r['owned_archive_removed']


@pytest.mark.parametrize('flag',['--mhr-release-only','--mhr-release-inventory-only'])
def test_strict_cli_returns_before_secret_hf_torch(release,monkeypatch,flag):
    release['freeze']();monkeypatch.setattr(acq.platform,'system',lambda:'Linux')
    monkeypatch.setenv('WR_ROOT',str(release['root']));monkeypatch.setenv('WR_CODE',str(release['code']));monkeypatch.setenv('WR_CODE_REVISION',release['revision'])
    monkeypatch.setattr(acq,'__file__',str(release['code']/'infra/acquire_weights.py'))
    monkeypatch.setitem(sys.modules,'mediapipe_cpu_runtime_verify',rt);monkeypatch.setattr(rt,'__file__',str(release['code']/'infra/mediapipe_cpu_runtime_verify.py'))
    calls=[];monkeypatch.setattr(acq,'mhr_release',lambda *a,**kw: calls.append((a,kw)) or dict(stage='fixture',status='pass',elapsed_seconds=1,model_byte_identical=True))
    before=set(sys.modules);acq.main([flag]);assert len(calls)==1
    assert calls[0][1]==({'inventory_only':True} if flag=='--mhr-release-inventory-only' else {})
    assert not ({'torch','huggingface_hub'} & (set(sys.modules)-before)) and not (release['root']/'.secrets').exists()
    for args in (['--mhr-release'],['--mhr-release-only','unknown'],['--mhr-release-only','--mhr-release-only']):
        with pytest.raises(SystemExit):acq.main(args)


def test_original_all_asset_body_retained():
    import ast
    tree=ast.parse((ROOT/'infra/acquire_weights.py').read_text());main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    statements=main.body;optin=next(i for i,n in enumerate(statements) if isinstance(n,ast.If) and ast.unparse(n.test)=='args.mhr_release_only or args.mhr_release_inventory_only')
    assert any(isinstance(n,ast.Return) for n in statements[optin].body)
    hf=next(i for i,n in enumerate(statements) if isinstance(n,ast.ImportFrom) and n.module=='huggingface_hub')
    assert optin<hf and any('.secrets/hf_token' in ast.unparse(n) for n in statements[optin+1:hf])
    protocol=json.loads((ROOT/acq.MHR_PROTOCOL).read_text());raw=(ROOT/acq.MHR_PROTOCOL).read_bytes()
    assert acq.MHR_PROTOCOL_PIN==identity(raw) and protocol['archive']['bytes']==198943157
    assert protocol['model']['bytes']==696110248 and protocol['budget_seconds']==300
    assert protocol['purpose'].endswith('standalone_authored_truth_only')


def test_download_exact_cap_no_unbounded_read(tmp_path,monkeypatch):
    class Response(io.BytesIO):
        headers={'Content-Encoding':'identity'}
        def read(self,size=-1):
            assert 0<size<=1<<20;return super().read(size)
    class Opener:
        def open(self,request,timeout):
            assert timeout==30 and request.get_header('Accept-encoding')=='identity'
            return Response(b'oversized')
    monkeypatch.setattr(acq,'_mhr_opener',lambda:Opener())
    path=tmp_path/'owned.part';path.write_bytes(b'')
    with pytest.raises(ValueError,match='cap'):acq._mhr_download('https://github.com/fixture',path,3)
    assert path.read_bytes()==b''


def test_wrong_preexisting_model_stops_before_metadata(release):
    release['model_path'].write_bytes(b'wrong')
    with pytest.raises(RuntimeError):release['run']()
    assert not release['calls'] and not (release['root']/release['protocol']['namespace']).exists()


@pytest.fixture
def inventory_release(release,monkeypatch):
    c=release;c['freeze']();oldrev='c'*40;monkeypatch.setattr(acq,'MHR_FAILED_REVISION',oldrev)
    monkeypatch.setattr(acq,'MHR_FAILED_ARCHIVE','b'*64)
    oldcode=c['root']/'jobs'/oldrev/'acquire_weights/code';oldcode.mkdir(parents=True)
    for name in acq.MHR_HELPERS:
        p=oldcode/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((c['code']/name).read_bytes());p.chmod(0o444)
    for name,value in [('revision',oldrev),('source-sha256','b'*64)]:
        p=oldcode.parent/name;p.write_text(value+'\n');p.chmod(0o444)
    for p in oldcode.rglob('*'):
        if p.is_dir():p.chmod(0o555)
    oldcode.chmod(0o555)
    out=c['root']/'results/mhr-official-release-license-v1';out.mkdir()
    metadata=b'{"opaque":"original metadata already frozen"}'
    r=dict(stage='mhr_official_release_license_v1',status='fail',phase='inventory',producer_revision=oldrev,
        protocol_identity=acq.MHR_PROTOCOL_PIN,whole_archive_sha_verified=True,owned_archive_removed=True,
        source_rehashed_after=True,existing_model_rehashed_after=True,existing_model=identity(c['model']),
        archive=copy.deepcopy(c['protocol']['archive']),primary_metadata=identity(metadata),source_before=rt.source(c['root'],oldcode,oldrev,'acquire_weights',acq.MHR_HELPERS))
    r['archive']={k:r['archive'][k] for k in ('bytes','sha256')}
    for name,raw in [('primary-LICENSE',c['license']),('primary-README.md',c['readme']),('release-metadata.json',metadata),('report.json',json.dumps(r,sort_keys=True).encode())]:
        p=out/name;p.write_bytes(raw);p.chmod(0o444)
    out.chmod(0o555);monkeypatch.setattr(acq,'MHR_FAILED_REPORT',identity((out/'report.json').read_bytes()))
    c['old_files']={p.name:p.read_bytes() for p in out.iterdir()};c['old_out']=out
    c['inventory_out']=c['root']/'results/mhr-official-release-inventory-v1'
    def run():
        return acq.mhr_release(c['root'],c['code'],c['revision'],rt,fetch=c['fetch'],download=c['download'],inventory_only=True)
    c['inventory_run']=run;return c


@pytest.mark.parametrize('case',['root_license','unsafe_path','unsafe_type','expanded_cap'])
def test_inventory_diagnostic_no_model_payload_and_first_guard(inventory_release,monkeypatch,case):
    c=inventory_release
    rows=list(c['entries'])
    if case=='root_license':rows[0]=('LICENSE',rows[0][1],rows[0][2])
    elif case=='unsafe_path':rows.append(('../unsafe',b'x',stat.S_IFREG|0o644))
    elif case=='unsafe_type':rows.append(('assets/link',b'x',stat.S_IFLNK|0o777))
    elif case=='expanded_cap':
        # Actual bound alteration is a tiny mock protocol pin, not a runtime relaxation.
        c['protocol']['archive_limits']['expanded_bytes']=1
    c['archive']=archive_bytes(rows)
    # Keep the old failure's protocol/wholeSHA in sync with manufactured mock-only input.
    if case!='expanded_cap':
        c['protocol']['archive'].update(identity(c['archive']))
    raw=(json.dumps(c['protocol'],sort_keys=True)+'\n').encode();monkeypatch.setattr(acq,'MHR_PROTOCOL_PIN',identity(raw))
    (c['code']/acq.MHR_PROTOCOL).chmod(0o644);(c['code']/acq.MHR_PROTOCOL).write_bytes(raw);(c['code']/acq.MHR_PROTOCOL).chmod(0o444)
    # Test inventory behavior independently of historical binding; separate tests below prove real binding.
    monkeypatch.setattr(acq,'_mhr_failure',lambda *_:dict(report=acq.MHR_FAILED_REPORT,original_status='fail'))
    opened=[];original=zipfile.ZipFile.open
    def spy(z,name,*args,**kwargs):
        member=name.filename if hasattr(name,'filename') else name;opened.append(member)
        assert member!='assets/mhr_model.pt';return original(z,name,*args,**kwargs)
    monkeypatch.setattr(zipfile.ZipFile,'open',spy)
    r=c['inventory_run']();d=r['inventory_diagnostic']
    assert r['status']=='pass' and r['diagnostic_only'] and r['original_failure_unchanged']
    assert r['asset_LICENSE_verified'] is False and r['model_member_streamed'] is False
    assert d['first_v1_rejection']['guard']=={'root_license':'required_members','unsafe_path':'traversal','unsafe_type':'type','expanded_cap':'expanded_bytes'}[case]
    if case=='root_license':
        assert opened==['LICENSE'];assert r['asset_notices'][0]['matches_primary_LICENSE_exactly']
    else:assert not opened and r['notice_payloads_read']==0
    assert {p.name:p.read_bytes() for p in c['old_out'].iterdir()}==c['old_files']
    assert not (c['inventory_out']/'assets.zip.part').exists()


def test_inventory_real_historical_proof_rehashed(inventory_release):
    c=inventory_release;r=c['inventory_run']()
    assert r['previous_failure']['source']['producer_revision']==acq.MHR_FAILED_REVISION
    assert r['original_failure_unchanged'] and r['inventory_diagnostic']['v1_inventory_accepted']
    assert r['asset_LICENSE_verified'] is False and 'model_byte_identical' not in r
    assert {p.name:p.read_bytes() for p in c['old_out'].iterdir()}==c['old_files']


@pytest.mark.parametrize('when',['before','after'])
def test_inventory_historical_mutation_never_hidden(inventory_release,when):
    c=inventory_release;p=c['old_out']/'primary-README.md'
    def mutate():p.chmod(0o644);p.write_bytes(b'changed original')
    if when=='before':mutate()
    else:c['mutation']=mutate
    with pytest.raises(RuntimeError):c['inventory_run']()
    if when=='before':assert not c['calls'] and not c['inventory_out'].exists()
    else:
        r=json.loads((c['inventory_out']/'report.json').read_text())
        assert r['status']=='fail' and r['owned_archive_removed'] and not r.get('original_failure_unchanged',False)


def test_inventory_rejects_duplicate_or_mixed_cli_before_any_io(monkeypatch):
    monkeypatch.setattr(acq.platform,'system',lambda:pytest.fail('CLI read platform before rejecting malformed mode'))
    for argv in (['--mhr-release-inventory'],['--mhr-release-inventory-only','--mhr-release-only'],
                 ['--mhr-release-inventory-only','--mhr-release-inventory-only']):
        with pytest.raises(SystemExit):acq.main(argv)


def test_ordered_inventory_type_before_member_and_cumulative_limits():
    protocol=json.loads((ROOT/acq.MHR_PROTOCOL).read_text());protocol['archive_limits'].update(member_bytes=0,members=0,expanded_bytes=0)
    raw=archive_bytes([('assets/link',b'x',stat.S_IFLNK|0o777)])
    with zipfile.ZipFile(io.BytesIO(raw)) as z:d=acq._mhr_inventory(z,protocol)
    assert d['first_v1_rejection']['guard']=='type' and not d['structurally_safe']


def test_inventory_central_scalars_without_payload_reads(monkeypatch):
    protocol=json.loads((ROOT/acq.MHR_PROTOCOL).read_text())
    raw=archive_bytes([('LICENSE',b'text',stat.S_IFREG|0o644),('assets/model',b'opaque',stat.S_IFREG|0o444)])
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        monkeypatch.setattr(z,'open',lambda *_:pytest.fail('Central inventory read payload'))
        d=acq._mhr_inventory(z,protocol)
    assert d['maximum_member_bytes']==6 and d['expanded_bytes']==10
    assert d['maximum_compressed_member_bytes']>0 and d['count']==2
    assert d['rows'][0]['orig_filename']=='LICENSE' and d['rows'][0]['mode_octal']==oct(stat.S_IFREG|0o644)
    assert d['rows'][0]['compression']==zipfile.ZIP_DEFLATED and d['rows'][0]['flags']==0
    assert d['first_v1_rejection']['guard']=='required_members' and d['structurally_safe']
