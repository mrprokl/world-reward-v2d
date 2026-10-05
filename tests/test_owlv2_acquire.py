"""Tiny mocked public transport tests; no HTTP, actual weights or GPU."""
from email.message import Message
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace
import urllib.request

import pytest

import owlv2_acquire as p


def config():return json.loads((Path(__file__).resolve().parents[1]/p.CONFIG).read_bytes())


class Response:
    def __init__(self,raw,url,*,length=None,encoding='identity',status=200):
        self.raw=raw;self.offset=0;self.url=url;self.status=status;self.reads=[];self.headers=Message()
        self.headers['Content-Type']='application/octet-stream';self.headers['Content-Encoding']=encoding
        if length is not None:self.headers['Content-Length']=str(length)
    def __enter__(self):return self
    def __exit__(self,*_):pass
    def geturl(self):return self.url
    def read(self,size):
        self.reads.append(size);part=self.raw[self.offset:self.offset+size];self.offset+=len(part);return part


def row(name,raw,kind='primary_configuration'):
    return dict(file=name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),kind=kind,
        url=f'https://huggingface.co/{p.REPO}/raw/{p.REVISION}/{name}')


def simple_assets():
    rows=[]
    values=[('README.md',b'---\nlicense: apache-2.0\n---\nprimary', 'primary_model_card'),
        ('config.json',b'{"architectures":["Owlv2ForObjectDetection"]}', 'primary_configuration'),
        ('preprocessor_config.json',b'{"image_processor_type":"Owlv2ImageProcessor"}', 'primary_configuration'),
        ('model.safetensors',b'opaque-not-a-decoded-model'*20,'opaque_checkpoint')]
    for name,raw,kind in values:rows.append(row(name,raw,kind))
    return dict(assets=rows),dict((name,raw) for name,raw,_ in values)


def publisher():
    def publish(part,target):
        assert not target.exists();part.rename(target)
    return SimpleNamespace(publish=publish)


def test_original_streamer_and_publication_are_reused_and_configs_pinned():
    assert p.acq.fetch.__module__=='masa_acquire' and p.mp.publish.__module__=='mediapipe_hands_acquire'
    cfg=config();root=Path(__file__).resolve().parents[1]
    for name,pin in cfg['reused_helper_pins'].items():assert p.pin((root/name).read_bytes())==pin
    assert len(cfg['assets'])==4 and cfg['assets'][-1]['bytes']==619918824
    assert cfg['assets'][-1]['sha256']=='e1e130b9e404cf91a75ad45644c1da9d7fa5284085eecc864266a6923efb99e7'
    assert cfg['budget_seconds']==600 and cfg['retry_count']==0
    assert cfg['provenance_limits']['standalone_model_license_file_verified'] is False


@pytest.mark.parametrize('url',['https://huggingface.co/a','https://cdn-lfs-us-1.hf.co/a?Expires=123&Signature=opaque',
    'https://cas-bridge.xethub.hf.co/a?X-Amz-Credential=publisher_delivery&X-Amz-Signature=opaque',
    'https://us.aws.cdn.hf.co/a?Expires=123&Signature=opaque'])
def test_exact_approved_https_cdn_signed_delivery_only(url):assert p.endpoint(url)==url


def test_observed_publisher_cdn_intersects_unchanged_upstream_endpoint():
    url='https://us.aws.cdn.hf.co/a?Expires=123&Signature=opaque'
    assert p.endpoint(url)==p.acq.endpoint(url)==url
    assert len(p.HOSTS)==7


@pytest.mark.parametrize('url',['http://huggingface.co/a','https://evil.hf.co/a','https://hf.co.evil/a',
    'https://huggingface.co@evil/a','https://user:SECRET@huggingface.co/a','https://huggingface.co/a?token=SECRET',
    'https://huggingface.co/a#SECRET','https://huggingface.co:8443/a','https://huggingface.co/a\nheader:SECRET'])
def test_unsafe_or_user_credentials_rejected_no_secret_error(url):
    with pytest.raises(ValueError) as err:p.endpoint(url)
    assert 'SECRET' not in str(err.value)


def test_redirect_count_and_authorization_rejected():
    request=urllib.request.Request('https://huggingface.co/a');handler=p.PublicRedirect();headers=Message()
    for _ in range(3):assert handler.redirect_request(request,None,302,'',headers,'https://cdn-lfs-us-1.hf.co/a') is not None
    with pytest.raises(ValueError):handler.redirect_request(request,None,302,'',headers,'https://cdn-lfs-us-1.hf.co/a')
    request.add_unredirected_header('Authorization','SECRET')
    with pytest.raises(ValueError) as err:p.PublicRedirect().redirect_request(request,None,302,'',headers,'https://cdn-lfs-us-1.hf.co/a')
    assert 'SECRET' not in str(err.value)


def test_opener_only_frozen_initial_url_and_no_token(monkeypatch):
    cfg,raw=simple_assets();calls=[]
    class Native:
        def open(self,request,timeout):calls.append((request.full_url,timeout));return Response(b'x',request.full_url)
    monkeypatch.setattr(p.urllib.request,'build_opener',lambda *handlers:(calls.append(tuple(type(x).__name__ for x in handlers)) or Native()))
    opener=p.PublicOpener(cfg['assets']);req=urllib.request.Request(cfg['assets'][0]['url'])
    opener.open(req,timeout=20)
    assert calls[0]==('ProxyHandler','PublicRedirect') and calls[1][1]==20
    req.add_header('Cookie','SECRET')
    with pytest.raises(ValueError):opener.open(req,timeout=20)


def test_opener_final_nonallowlisted_cdn_closes_response(monkeypatch):
    cfg,_=simple_assets();closed=[]
    response=Response(b'x','https://unknown.example/SECRET_SIGNED_URL');response.close=lambda:closed.append(True)
    native=SimpleNamespace(open=lambda *a,**kw:response)
    monkeypatch.setattr(p.urllib.request,'build_opener',lambda *a:native)
    with pytest.raises(ValueError) as err:p.PublicOpener(cfg['assets']).open(urllib.request.Request(cfg['assets'][0]['url']),20)
    assert closed==[True] and 'SECRET' not in str(err.value)


def test_complete_tiny_streaming_assets_exact_hash_atomic_and_not_decoded(tmp_path):
    cfg,raw=simple_assets();report={'assets':[]};responses=[]
    class Opener:
        def open(self,request,timeout):
            name=request.full_url.rsplit('/',1)[1];response=Response(raw[name],request.full_url,length=len(raw[name]));responses.append(response);return response
    p.acquire_assets(tmp_path,cfg,report,time.monotonic()+10,opener=Opener(),publisher=publisher())
    p.posthash(tmp_path,cfg,report)
    assert report['primary_model_license_recorded'] and report['owned_partials_removed'] and report['artifacts_rehashed_after']
    assert len(report['assets'])==4 and not any(path.suffix=='.part' for path in tmp_path.iterdir())
    assert all(path.stat().st_mode&0o777==0o400 for path in tmp_path.iterdir())
    assert all(max(r.reads)<=1<<20 for r in responses)
    assert all('url' not in r for r in report['assets'])


@pytest.mark.parametrize('failure',['empty','sha','length','overflow','html','lfs','encoding','host','timeout','missing_license'])
def test_failed_exact_stream_cleans_only_owned_partials_and_no_retries(tmp_path,failure):
    cfg,raw=simple_assets();calls=[];report={'assets':[]}
    if failure=='missing_license':raw['README.md']=b'---\nlicense: other-license\n---';cfg['assets'][0]=row('README.md',raw['README.md'],'primary_model_card')
    class Opener:
        def open(self,request,timeout):
            calls.append(request.full_url);name=request.full_url.rsplit('/',1)[1];content=raw[name];url=request.full_url;length=None;encoding='identity'
            if failure=='empty':content=b''
            elif failure=='sha':content=b'x'*len(content)
            elif failure=='length':length=len(content)+1
            elif failure=='overflow':content+=b'x'
            elif failure=='html':content=b'<!doctype html>fake'
            elif failure=='lfs':content=b'version https://git-lfs.github.com/spec/fake'
            elif failure=='encoding':encoding='gzip'
            elif failure=='host':url='https://unknown.example/a'
            elif failure=='timeout':raise TimeoutError('SECRET_DELIVERY_FAILURE')
            return Response(content,url,length=length,encoding=encoding)
    with pytest.raises((ValueError,TimeoutError)):
        p.acquire_assets(tmp_path,cfg,report,time.monotonic()+10,opener=Opener(),publisher=publisher())
    assert len(calls)==1 and report['owned_partials_removed'] is True
    assert not any(path.name.endswith('.part') for path in tmp_path.iterdir())
    assert 'SECRET' not in json.dumps(report)


def test_original_stream_is_incremental_not_whole_checkpoint_ram(tmp_path):
    raw=b'opaque'*(400000);record=row('model.safetensors',raw,'opaque_checkpoint');responses=[]
    class Opener:
        def open(self,request,timeout):
            response=Response(raw,request.full_url);responses.append(response);return response
    partials=[];value=p.acq.fetch(p.rt,publisher(),tmp_path,record,[record],Opener(),time.monotonic()+10,partials)
    assert len(responses[0].reads)>=3 and all(size<=1<<20 for size in responses[0].reads)
    assert value['sha256']==record['sha256'] and (tmp_path/'model.safetensors').read_bytes()==raw
    p.cleanup(partials)


def test_partial_ownership_prevents_foreign_deletion(tmp_path):
    path=tmp_path/'foreign.part';path.write_bytes(b'keep');s=path.stat()
    with pytest.raises(ValueError):p.cleanup([(path,(s.st_dev,s.st_ino+1,s.st_uid))])
    assert path.read_bytes()==b'keep'


def test_changed_final_asset_or_foreign_leaf_rejected(tmp_path):
    cfg,raw=simple_assets();report={'assets':[]}
    class Opener:
        def open(self,request,timeout):return Response(raw[request.full_url.rsplit('/',1)[1]],request.full_url)
    p.acquire_assets(tmp_path,cfg,report,time.monotonic()+10,opener=Opener(),publisher=publisher())
    (tmp_path/'foreign').write_bytes(b'x')
    with pytest.raises(ValueError):p.posthash(tmp_path,cfg,report)
    (tmp_path/'foreign').unlink();leaf=tmp_path/'config.json';leaf.chmod(0o600);leaf.write_bytes(b'changed');leaf.chmod(0o400)
    with pytest.raises(ValueError):p.posthash(tmp_path,cfg,report)


def test_entry_clean_environment_no_model_decode_or_raw_error():
    root=Path(__file__).resolve().parents[1];src=(root/p.HELPERS[0]).read_text();wrapper=(root/p.HELPERS[1]).read_text()
    compile(src,p.HELPERS[0],'exec')
    assert 'acq.fetch(' not in src or 'fetch=acq.fetch' in src
    assert 'torch' not in src and 'safetensors.load' not in src and 'str(exc)' not in src and 'traceback' not in src
    assert 'env -i' in wrapper and '-I -B' in wrapper and '610s' in wrapper and 'set +x' in wrapper


def test_lifecycle_fail_receipt_sealed_sanitized_with_no_output_reuse(tmp_path,monkeypatch,capsys):
    root=tmp_path/'root';(root/'weights').mkdir(parents=True);(root/'results').mkdir()
    monkeypatch.setattr(p,'ROOT',root);monkeypatch.setattr(p.os,'geteuid',lambda:0);monkeypatch.setattr(p.sys,'platform','linux')
    monkeypatch.setattr(p.os,'uname',lambda:SimpleNamespace(nodename='world-reward-ncc-h100-02'))
    monkeypatch.setenv('WR_CODE',str(Path(p.__file__).resolve().parents[1]));monkeypatch.setenv('WR_CODE_REVISION','a'*40)
    source={'helpers':{p.CONFIG:dict(bytes=1,sha256='b'*64)}}
    monkeypatch.setattr(p.rt,'source',lambda *a:source);monkeypatch.setattr(p,'configuration',lambda *a:config())
    def fail(*a):raise TimeoutError('SECRET_RUNTIME_URL_TOKEN')
    monkeypatch.setattr(p,'acquire_assets',fail)
    report=p.run();text=(root/p.REPORT).read_text()+capsys.readouterr().out
    assert report['status']=='fail' and report['error_type']=='TimeoutError' and 'SECRET' not in text
    assert (root/p.BASE).stat().st_mode&0o777==0o500 and (root/p.REPORT).stat().st_mode&0o777==0o400
    with pytest.raises(ValueError):p.run()
