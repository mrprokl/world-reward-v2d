"""Tiny manufactured SAM transport/bank contracts; no models/data/network."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'infra'))
import sam31_runtime as runtime


def test_frozen_runtime_scope():
    c,h=runtime.settings(Path(__file__).resolve().parents[1])
    assert c['model_repo']=='facebook/sam3.1'
    assert c['weight_bytes']==3502755717
    assert h['episodes']==runtime.EXPECTED_EPISODES
    assert c['use_fa3'] is False


def test_signature_adapter_drops_only_unsupported_state_offload():
    class Model:
        def init_state(self, resource_path, offload_video_to_cpu=False, async_loading_frames=False):
            return dict(resource=resource_path,offload=offload_video_to_cpu,async_=async_loading_frames)
    original=[object()]
    out,dropped=runtime.compatible_init(Model(),original)
    assert out['resource'] is original and out['offload'] is False
    assert dropped==['offload_state_to_cpu']


def test_signature_adapter_preserves_supported_parameters():
    class Model:
        def init_state(self, resource_path, offload_video_to_cpu=False, offload_state_to_cpu=True, async_loading_frames=False):
            return offload_state_to_cpu
    out,dropped=runtime.compatible_init(Model(),[])
    assert out is False and dropped==[]


def output(mask):
    masks=np.asarray(mask,dtype=bool)
    return dict(out_obj_ids=np.array([17],dtype=np.int64),out_probs=np.array([.8],dtype=np.float32),
        out_boxes_xywh=np.array([[0,0,1,1]],dtype=np.float32),out_binary_masks=masks[None])


def test_exact_masks_and_exclusive_box():
    m=np.zeros((7,9),dtype=bool);m[2:5,3:8]=True
    arrays,records=runtime.mask_records(output(m),7,9)
    assert records==[dict(native_id=17,area=15,box=[3,2,8,5])]
    restored=np.unpackbits(arrays['packed_masks'],axis=1,count=63).reshape(1,7,9)
    assert np.array_equal(restored,m[None])
    assert arrays['boxes_xywh'].tolist()==[[0.,0.,1.,1.]]


def test_empty_original_frame_is_saved(tmp_path):
    o=dict(out_obj_ids=np.zeros(0,dtype=np.int64),out_probs=np.zeros(0,dtype=np.float32),
        out_boxes_xywh=np.zeros((0,4),dtype=np.float32),out_binary_masks=np.zeros((0,7,9),dtype=bool))
    pin,records=runtime.persist_frame(tmp_path,o,3,'a'*64,7,9)
    assert records==[] and pin['frame_index']==3 and pin['decoded_rgb_sha256']=='a'*64
    p=tmp_path/pin['file']
    assert hashlib.sha256(p.read_bytes()).hexdigest()==pin['sha256']
    with np.load(p,allow_pickle=False) as z:
        assert z['packed_masks'].shape==(0,8)
        assert z['mask_shape'].tolist()==[7,9]


@pytest.mark.parametrize('change',[lambda o:o.update(out_obj_ids=np.array([17],dtype=np.int32)),
    lambda o:o.update(out_probs=np.array([float('nan')])),
    lambda o:o.update(out_binary_masks=np.zeros((1,7,9),dtype=bool))])
def test_reject_invalid_native_visible_output(change):
    o=output(np.ones((7,9),dtype=bool));change(o)
    with pytest.raises(ValueError):runtime.mask_records(o,7,9)


def make_tree(rows):
    raw=b''.join(r['mode'].lstrip('0').encode()+b' '+Path(r['path']).name.encode()+b'\0'+bytes.fromhex(r['sha'])
                 for r in sorted(rows,key=lambda r:(Path(r['path']).name+('/'if r['type']=='tree'else'')).encode()))
    return hashlib.sha1(b'tree '+str(len(raw)).encode()+b'\0'+raw).hexdigest()


def test_git_source_merkle_tree_including_dot_directory():
    child=dict(path='.github/a.py',type='blob',mode='100644',sha=runtime.git_blob(b'hello'))
    directory=dict(path='.github',type='tree',mode='040000',sha=make_tree([child]))
    top=dict(path='README.md',type='blob',mode='100644',sha=runtime.git_blob(b'readme'))
    rows=[child,directory,top];root=make_tree([directory,top])
    runtime.verify_git_tree(rows,root)
    with pytest.raises(ValueError):runtime.verify_git_tree(rows,'0'*40)


def test_upstream_whitelist_excludes_videos_annotations_and_notebooks():
    assert runtime.allowed_source(dict(path='sam3/model/model.py',type='blob'))
    assert runtime.allowed_source(dict(path='sam3/assets/bpe_simple_vocab_16e6.txt.gz',type='blob'))
    assert not runtime.allowed_source(dict(path='assets/videos/bedroom.mp4',type='blob'))
    assert not runtime.allowed_source(dict(path='scripts/eval/silver/inaturalist_image_subset.json',type='blob'))
    assert not runtime.allowed_source(dict(path='examples/example.ipynb',type='blob'))


def test_shell_and_docker_do_not_copy_credentials_or_qwen():
    root=Path(__file__).resolve().parents[1]
    docker=(root/'infra/Dockerfile.sam31').read_text()
    assert 'COPY upstream /opt/sam3' in docker
    assert 'hf_token' not in docker and 'Qwen' not in docker
    script=(root/'infra/run_sam31_runtime.sh').read_text()
    assert 'set +x' in script and 'run_sam31_runtime/code' in script
