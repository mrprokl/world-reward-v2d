"""Tiny own OCI graphs only, never actual image/model/GPU assets."""
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile

import pytest


@pytest.fixture
def gate():
    p=Path(__file__).resolve().parents[1]/'infra/research_image_identity.py'
    s=importlib.util.spec_from_file_location('image_identity_test',p)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def fixture(gate,tmp_path,monkeypatch):
    content={};diff='sha256:'+'a'*64
    def add(value):
        b=json.dumps(value).encode();d='sha256:'+hashlib.sha256(b).hexdigest()
        content['blobs/sha256/'+d[7:]]=b
        return {'digest':d,'size':len(b)}
    config=add({'architecture':'amd64','os':'linux','rootfs':{'type':'layers','diff_ids':[diff]}})
    layer={'digest':'sha256:'+'b'*64,'size':2}
    platform=add({'config':config,'layers':[layer]})
    platform['platform']={'architecture':'amd64','os':'linux'}
    index=add({'manifests':[platform]})
    content['oci-layout']=json.dumps({'imageLayoutVersion':'1.0.0'}).encode()
    content['index.json']=json.dumps({'manifests':[index]}).encode()
    content['manifest.json']=json.dumps([{'Config':'blobs/sha256/'+config['digest'][7:],
        'RepoTags':[gate.TAG],'Layers':['blobs/sha256/'+layer['digest'][7:]]}]).encode()
    path=tmp_path/'image.tar'
    with tarfile.open(path,'w') as t:
        for name,b in content.items():
            member=tarfile.TarInfo(name);member.size=len(b);t.addfile(member,io.BytesIO(b))
    monkeypatch.setattr(gate,'EXPORT_SHA',hashlib.sha256(path.read_bytes()).hexdigest())
    monkeypatch.setattr(gate,'INDEX_ID',index['digest']);monkeypatch.setattr(gate,'CONFIG_ID',config['digest'])
    inspected=[{'Id':config['digest'],'Architecture':'amd64','Os':'linux',
                'RootFS':{'Type':'layers','Layers':[diff]},'RepoTags':[gate.TAG]}]
    return path,inspected


def test_sealed_OCI_to_classic_identity_and_order(gate,tmp_path,monkeypatch):
    path,data=fixture(gate,tmp_path,monkeypatch);r=gate.image_identity(path,data)
    assert r['image_id']==gate.CONFIG_ID and r['source_OCI_index_id']==gate.INDEX_ID
    assert r['image_content_changed'] is False and r['image_rebuilt'] is False and r['rootfs_layers']==1


@pytest.mark.parametrize('fault',['sha','id','arch','os','layers','type','tag','extra','symlink'])
def test_no_changed_image_or_unbound_alias(gate,tmp_path,monkeypatch,fault):
    path,data=fixture(gate,tmp_path,monkeypatch);data=copy.deepcopy(data)
    if fault=='sha':monkeypatch.setattr(gate,'EXPORT_SHA','0'*64)
    elif fault=='id':data[0]['Id']=gate.INDEX_ID
    elif fault=='arch':data[0]['Architecture']='arm64'
    elif fault=='os':data[0]['Os']='windows'
    elif fault=='layers':data[0]['RootFS']['Layers']=['sha256:'+'c'*64]
    elif fault=='type':data[0]['RootFS']['Type']='other'
    elif fault=='tag':data[0]['RepoTags']=[]
    elif fault=='extra':data.append(data[0])
    elif fault=='symlink':
        link=tmp_path/'link.tar';link.symlink_to(path);path=link
    with pytest.raises(ValueError):gate.image_identity(path,data)


def test_explicit_continue_no_reload_reextract_or_restart():
    p=Path(__file__).resolve().parents[1]/'infra/run_research_import_continue.sh';s=p.read_text()
    import subprocess
    subprocess.run(['bash','-n',str(p)],check=True)
    assert 'docker image load' not in s and 'systemctl restart' not in s and 'extract --' not in s
    assert s.index('research_image_identity.py')<s.index('docker run')
    assert '--network none' in s and 'model_inference_verified=False' in s
