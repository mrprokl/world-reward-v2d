import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import time

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'infra'))
import form_hoi_external_dev as d


def config():
    return json.loads((Path(__file__).resolve().parents[1]/d.CONFIG).read_bytes())


def native():
    return (b'action_desc: Lift the container.\nframe_count: 300\nobject:\n  id: beige_bin\n'
            b'  prompt: beige color square container.\n  bbox: [1,2,3,4]\nperson:\n  id: "021"\n'
            b'interaction_trim:\n  start: 12\n  end: 80\n')


def test_native_text_ids_frames_only_no_trim_bbox():
    result=d.native_semantics(native(),config())
    assert result==dict(action='Lift the container.',full_frames=300,object_id='beige_bin',
                        object_prompt='beige color square container.',person_id='021')
    assert 'bbox' not in result and 'interaction_trim' not in result


@pytest.mark.parametrize('change', [b'action_desc: !unsafe x',b'action_desc: |',b'frame_count: 95',
                                  b'frame_count: 0',b'frame_count: 3.0',b'person:\n  id: [021]'])
def test_native_missing_unsafe_short_fail_not_reroll(change):
    raw=native()
    if change.startswith(b'action_desc'):raw=raw.replace(b'action_desc: Lift the container.',change)
    elif change.startswith(b'frame_count'):raw=raw.replace(b'frame_count: 300',change)
    else:raw=raw.replace(b'person:\n  id: "021"',change)
    with pytest.raises(ValueError):d.native_semantics(raw,config())


def test_native_duplicate_semantics_fail():
    with pytest.raises(ValueError,match='Unique'):d.native_semantics(native()+b'action_desc: Other\n',config())


def members(sequence):
    return [dict(name=sequence+'/'+n,bytes=20,directory=False) for n in
            ('videos/front_stereo_camera_left.mp4','videos/rear_stereo_camera_left.mp4','hoi_metadata.yaml',
             'object_mesh/output_aligned.glb','poses.npy','mhr_params_mv.pt','edex','failure_segments.json',
             'object_masks/front_stereo_camera_left.h5','depth/front_stereo_camera_left.h5','ground_plane.json')]


def test_selected_single_rgb_and_private_ref_only():
    selected=d.selected_members(members('seq'),'seq',config())
    assert sum(role=='rgb' for _,role in selected)==1
    names={r['name'] for r,_ in selected}
    assert 'seq/videos/rear_stereo_camera_left.mp4' not in names
    assert 'seq/depth/front_stereo_camera_left.h5' not in names
    assert 'seq/ground_plane.json' not in names
    assert 'seq/object_masks/front_stereo_camera_left.h5' in names
    with pytest.raises(ValueError,match='Complete'):
        d.selected_members([r for r in members('seq') if not r['name'].endswith('/mhr_params_mv.pt')],'seq',config())


def test_pinned_dev_protocol_no_reserved_same_contiguous_grid():
    cfg=config();root=Path(__file__).resolve().parents[1]
    cohort=json.loads((root/cfg['cohort_protocol']).read_bytes())
    assert len(d.acquisition.cohort(cohort,'acquire_dev'))==4
    assert cfg['contiguous_prefix_frames']==cfg['minimum_length']==96
    assert cfg['alias_guard']['actual_bank_identity'] is None
    assert cfg['reserved_acquired']==0 and cfg['training_overlap_verified'] is False


class Runtime:
    @staticmethod
    def pinned(path,pin,maximum):return json.loads(Path(path).read_bytes())


def bank(cfg):
    return dict(schema=cfg['alias_guard']['schema'],dataset_revision=cfg['challenge_revision'],
                input_manifest=cfg['challenge_manifest'],frames_per_video=96,
                decoded_format=cfg['alias_guard']['decoded_format'],
                videos=[dict(episode=i,bytes=100,sha256=hashlib.sha256(f'video{i}'.encode()).hexdigest(),
                             frame_rgb_sha256=[hashlib.sha256(f'{i}/{t}'.encode()).hexdigest() for t in range(96)])
                        for i in range(30)])


def test_missing_alias_bank_never_ready(tmp_path):
    cfg=config();cfg['alias_guard']['path']=str(tmp_path/'bank.json')
    result=d.alias_guard(Runtime(),cfg,decoded_hashes=['f'*64]*96,video_pin={'sha256':'e'*64})
    assert result['qualified'] is False and result['exact_content_duplicate_check_passed'] is False


def test_actual_alias_bank_all30_exact_byte_or_frame_guard(tmp_path):
    cfg=config();p=tmp_path/'bank.json';cfg['alias_guard'].update(path=str(p),actual_bank_identity=dict(bytes=1,sha256='a'*64))
    value=bank(cfg);p.write_text(json.dumps(value))
    result=d.alias_guard(Runtime(),cfg,decoded_hashes=['f'*64]*96,video_pin={'sha256':'e'*64})
    assert result['qualified'] is True and result['near_alias_absence_verified'] is False
    for sha,pixels in [(value['videos'][0]['sha256'],['f'*64]*96),
                       ('e'*64,[value['videos'][0]['frame_rgb_sha256'][0]]+['f'*64]*95)]:
        with pytest.raises(ValueError,match='duplicate'):
            d.alias_guard(Runtime(),cfg,decoded_hashes=pixels,video_pin={'sha256':sha})
    value['videos']=value['videos'][:-1];p.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='all-thirty'):
        d.alias_guard(Runtime(),cfg,decoded_hashes=['f'*64]*96,video_pin={'sha256':'e'*64})


def test_extract_public_package_no_reference_paths_values(tmp_path,monkeypatch):
    archive=tmp_path/'source.tar';sequence='sequence';cfg=config();folder=tmp_path/'out';folder.mkdir()
    payloads={'videos/front_stereo_camera_left.mp4':b'originalRGB', 'hoi_metadata.yaml':native(),
              'poses.npy':b'opaque-not-numpy','mhr_params_mv.pt':b'opaque-not-pickle',
              'object_mesh/output_aligned.glb':b'opaque-not-GLB','edex':b'opaque-not-JSON',
              'object_masks/front_stereo_camera_left.h5':b'opaque-not-HDF5'}
    with tarfile.open(archive,'w') as saved:
        for n,payload in payloads.items():
            m=tarfile.TarInfo(sequence+'/'+n);m.size=len(payload);saved.addfile(m,io.BytesIO(payload))
    row=dict(sequence_id=sequence,archive_size=archive.stat().st_size,member_count=len(payloads))
    monkeypatch.setattr(d,'probe',lambda *args:dict(width=1536,height=1152,fps=30,full_frames=300))
    monkeypatch.setattr(d,'decode_frame_hashes',lambda *args:['a'*64]*96)
    cfg['alias_guard']['path']=str(tmp_path/'bank.json')
    receipt=d.extract_one(archive,row,folder,cfg,Runtime(),time.monotonic()+10)
    package=json.loads((folder/'inputs/input.json').read_bytes())
    assert set(p.name for p in (folder/'inputs').iterdir())=={'rgb.mp4','input.json'}
    assert package['original_frame_indices']==list(range(96)) and package['total']==96
    assert package['full_source_frames']==300 and package['inference_ready'] is False
    assert package['reference_inputs_present'] is False and package['action']=='Lift the container.'
    assert not any(k in package for k in ('person_id','object_id','edex','bbox','mhr','poses','interaction_trim'))
    assert receipt['reference_arrays_decoded'] is False and receipt['source_calibration_decoded'] is False
    assert (folder/'eval_private/poses.npy').read_bytes()==b'opaque-not-numpy'
    d.validate_public_package(package,cfg)
    with pytest.raises(ValueError,match='blocked'):d.validate_public_package(package,cfg,require_ready=True)
    for key in ('masks','edex','ground_truth','person_id','source_intrinsics'):
        with pytest.raises(ValueError,match='forbidden'):d.validate_public_package({**package,key:'illegal'},cfg)


def test_runtime_code_only_closure_all_helpers():
    import azure_job
    root=Path(__file__).resolve().parents[1]
    files={str(p.relative_to(root)):p.read_bytes() for parent in ('infra','src','configs')
           for p in (root/parent).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    files['pyproject.toml']=(root/'pyproject.toml').read_bytes()
    selected=azure_job.runtime_bundle_paths(files,'infra/run_form_hoi_external_dev.sh')
    assert set(d.HELPERS)<=set(selected)
    assert not any(n.startswith('results/') for n in selected)
