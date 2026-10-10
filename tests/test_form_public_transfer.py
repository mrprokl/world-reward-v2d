import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'infra'))
import form_public_transfer as t


def manifest():
    rev='a'*40;seqs=['one','two','three','four']
    return rev,seqs,dict(schema='world_reward.form_public_transfer.v1',dev_revision=rev,sequences=seqs,
        files=[dict(sequence_id=s,file=n,name=f'form-public-{rev}/{s}/{n}',bytes=100,
                    sha256='b'*64,etag='"etag"')for s in seqs for n in ('rgb.mp4','input.json')],
        private_references_included=False,models_included=False,private_container_verified=True)


def test_exact_public_package_only():
    rev,seqs,m=manifest();t.contract(m,rev,seqs)
    for n in ['eval_private/edex','poses.npy','input.json/../rgb.mp4','rgb.mp4?sig=secret']:
        assert not t.allowed_blob(f'form-public-{rev}/one/{n}')
    assert t.allowed_blob(f'form-public-{rev}/one/rgb.mp4')


@pytest.mark.parametrize('fault',['private','missing','duplicate','huge','escape','etag','rev'])
def test_transfer_fail_closed(fault):
    rev,seqs,m=manifest()
    if fault=='private':m['private_references_included']=True
    if fault=='missing':m['files'].pop()
    if fault=='duplicate':m['files'][1]=m['files'][0]
    if fault=='huge':m['files'][0]['bytes']=t.MAX_VIDEO+1
    if fault=='escape':m['files'][0]['name']='../../private'
    if fault=='etag':m['files'][0]['etag']='secret'
    if fault=='rev':m['dev_revision']='c'*40
    with pytest.raises(ValueError):t.contract(m,rev,seqs)
