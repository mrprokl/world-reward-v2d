"""Tiny original-byte tar fixtures only; never real datasets or labels."""
import copy
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import struct
import tarfile

import pytest

SPEC = importlib.util.spec_from_file_location('dexycb_acquire_test', Path(__file__).parents[1]/'infra/dexycb_acquire.py')
dex = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(dex)


def jpeg(width=640, height=480):
    return b'\xff\xd8\xff\xc0' + struct.pack('>H', 17) + struct.pack('>BHHB', 8, height, width, 3) + bytes(9) + b'\xff\xd9'


def tar(path, records):
    with tarfile.open(path, 'w:gz') as archive:
        for name, data, kind in records:
            info = tarfile.TarInfo(name); info.type = kind
            info.size = len(data) if kind == tarfile.REGTYPE else 0
            if kind in (tarfile.SYMTYPE, tarfile.LNKTYPE): info.linkname = '/outside'
            archive.addfile(info, io.BytesIO(data) if kind == tarfile.REGTYPE else None)


def records(subject, frames=2):
    rows = []
    for index in range(100):
        sequence = f'20200709_{index:06d}'; base=f'{subject}/{sequence}'
        rows.append((base+'/meta.yml', b'NOT_PARSED_TARGET_METADATA', tarfile.REGTYPE))
        for frame in range(frames):
            rows += [(base+f'/{dex.CAMERA}/color_{frame:06d}.jpg', jpeg(), tarfile.REGTYPE),
                     (base+f'/{dex.CAMERA}/labels_{frame:06d}.npz', b'OPAQUE_PRIVATE_NPZ_BYTES', tarfile.REGTYPE)]
        rows += [(base+'/other-camera/color_000000.jpg', b'OTHER_CAMERA_NOT_EXTRACTED', tarfile.REGTYPE),
                 (base+f'/{dex.CAMERA}/aligned_depth_to_color_000000.png', b'NO_SENSOR_DEPTH', tarfile.REGTYPE)]
    return rows


def setup(tmp_path, monkeypatch):
    root=tmp_path/'root'; root.mkdir(); incoming=tmp_path/'incoming'; incoming.mkdir()
    monkeypatch.setattr(dex, 'INCOMING', incoming)
    protocol=copy.deepcopy(dex.EXPECTED_PROTOCOL)
    proofs={}
    for subject in dex.SUBJECTS:
        path=incoming/(subject+'.tar.gz'); tar(path, records(subject))
        protocol['archives'][subject]['bytes']=path.stat().st_size
        proofs[subject]=hashlib.sha256(path.read_bytes()).hexdigest()
    evidence=root/dex.EVIDENCE; evidence.mkdir(parents=True)
    for name, data in [('publisher.html', b'DexYCB is licensed under by-nc/4.0'),
                       ('dex_ycb.py', b"ycb_grasp_ind color_{:06d}.jpg np.arange(meta['num_frames'])")]:
        path=evidence/name; path.write_bytes(data); path.chmod(0o444)
        protocol['primary_sources'][name].update(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
    monkeypatch.setattr(dex, 'EXPECTED_PROTOCOL', protocol)
    # __file__ is source text identity only; no injected producer revisions in tiny tests.
    script=tmp_path/'dexycb_acquire.py'; script.write_bytes(Path(dex.__file__).read_bytes());script.chmod(0o444)
    monkeypatch.setattr(dex, '__file__', str(script)); monkeypatch.delenv('WR_CODE_REVISION', raising=False)
    config=root/'configs/protocol.json'; config.parent.mkdir();config.write_text(json.dumps(protocol));config.chmod(0o444)
    return root,incoming,proofs,config


def test_frozen_real_protocol_exact_types_and_selection():
    original=json.loads((Path(__file__).parents[1]/'configs/dexycb_identity_protocol.json').read_text())
    dex.exact(original,dex.EXPECTED_PROTOCOL)
    assert original['scope']=='acquisition_plan_only_not_frozen_model_or_quality_protocol'
    assert original['sequence_lex_indices']==[0,16,32,48,64,80]
    assert original['archive_root']=='direct_subject_directory_only'
    assert dex.INCOMING==Path('/srv/world-reward-data/dexycb_identity_download_v1')
    for value in (dict(original, all_original_frames=1),dict(original,extra='x')):
        with pytest.raises(ValueError):dex.exact(value,original)


def test_real_tar_all_headers_before_selection_no_private_value_decode(tmp_path):
    subject=dex.SUBJECTS[0]; path=tmp_path/'tiny.gz';tar(path,records(subject))
    selected,wanted,audit=dex.inspect_archive(path,subject)
    assert [s['sequence_lex_index'] for s in selected]==dex.INDICES
    assert all(s['frames']==2 for s in selected)
    assert len(wanted)==6*5 and audit['gzip_crc_verified'] is True
    assert not any('other-camera' in n or 'depth' in n for n in wanted)


@pytest.mark.parametrize('name,kind',[
    ('../outside',tarfile.REGTYPE),('/outside',tarfile.REGTYPE),
    ('wrapper/20200709-subject-01/20200709_000000/meta.yml',tarfile.REGTYPE),
    ('20200709-subject-01/x/../meta.yml',tarfile.REGTYPE),
    ('20200709-subject-01/link',tarfile.SYMTYPE),('20200709-subject-01/link',tarfile.LNKTYPE),
    ('20200709-subject-01/fifo',tarfile.FIFOTYPE),('20200709-subject-01\\x',tarfile.REGTYPE),
])
def test_archive_rejects_path_links_wrappers_special(tmp_path,name,kind):
    path=tmp_path/'unsafe.gz';tar(path,[(name,b'x',kind)])
    with pytest.raises(ValueError):dex.inspect_archive(path,dex.SUBJECTS[0])


@pytest.mark.parametrize('case',['duplicate','missing_label','noncontiguous','missing_meta','extra_sequence','ancestor'])
def test_inventory_exact_allframe_scope_and_collision(tmp_path,case):
    subject=dex.SUBJECTS[0];rows=records(subject)
    if case=='duplicate':rows.append(rows[0])
    if case=='missing_label':rows=[x for x in rows if not x[0].endswith('/labels_000001.npz')]
    if case=='noncontiguous':rows=[(n.replace('color_000001','color_000002'),b,k) for n,b,k in rows]
    if case=='missing_meta':rows=[x for x in rows if not x[0].endswith('/meta.yml')]
    if case=='extra_sequence':rows.append((subject+'/20200709_000100/meta.yml',b'x',tarfile.REGTYPE))
    if case=='ancestor':rows.append((subject+'/20200709_000000',b'x',tarfile.REGTYPE))
    path=tmp_path/'bad.gz';tar(path,rows)
    with pytest.raises(ValueError):dex.inspect_archive(path,subject)


def test_gzip_crc_and_expanded_size_guards(tmp_path,monkeypatch):
    path=tmp_path/'crc.gz';tar(path,records(dex.SUBJECTS[0]))
    b=bytearray(path.read_bytes());b[-8]^=1;path.write_bytes(b)
    with pytest.raises((gzip.BadGzipFile,OSError)):dex.inspect_archive(path,dex.SUBJECTS[0])
    tar(path,records(dex.SUBJECTS[0]));monkeypatch.setattr(dex,'EXPANDED_CAP',512)
    with pytest.raises(ValueError,match='cap'):dex.inspect_archive(path,dex.SUBJECTS[0])


def test_nonzero_trailing_tar_payload_rejected(tmp_path):
    path=tmp_path/'extra.gz';tar(path,records(dex.SUBJECTS[0]))
    raw=gzip.decompress(path.read_bytes());path.write_bytes(gzip.compress(raw+b'NOT_A_MEMBER'))
    with pytest.raises(ValueError,match='after tar'):dex.inspect_archive(path,dex.SUBJECTS[0])


def test_jpeg_header_grid_and_segment_type_failclosed(tmp_path):
    path=tmp_path/'x.jpg';path.write_bytes(jpeg());assert dex.jpeg_size(path)==(640,480)
    bad=bytearray(jpeg());bad[4:6]=struct.pack('>H',8);path.write_bytes(bad)
    with pytest.raises(ValueError,match='SOF'):dex.jpeg_size(path)


def test_all12_sequences_exact_public_private_and_reports(tmp_path,monkeypatch):
    root,incoming,hashes,config=setup(tmp_path,monkeypatch)
    report=dex.acquire(root,hashes,config)
    base=root/dex.BASE;manifest=json.loads((base/'inputs/manifest.json').read_bytes())
    assert report['status']=='pass' and report['frames']==24 and report['sequences']==12
    assert report['source_rehashed_after'] and not report['annotation_values_parsed']
    assert len(list((base/'inputs').iterdir()))==25
    assert len(list((base/'eval_private').rglob('labels_*.npz')))==24
    assert len(list((base/'eval_private').rglob('meta.yml')))==12
    assert all(p.read_bytes()==b'OPAQUE_PRIVATE_NPZ_BYTES' for p in (base/'eval_private').rglob('*.npz'))
    assert (base/'eval_private').stat().st_mode&0o777==0o700
    assert all(p.stat().st_mode&0o777==0o444 for p in (base/'inputs').iterdir())
    assert (base/'report.json').stat().st_mode&0o777==0o444
    public=json.dumps(manifest)
    assert all(k not in public for k in ('ycb_ids','ycb_grasp_ind','intrinsics','fps','pose_y','joint_3d','contact'))
    assert manifest['timestamps_available'] is False
    assert [s['partition'] for s in manifest['sequences'][6:]]==['fit','decision','fit','decision','fit','decision']
    assert all(r['frame_position']==r['source_frame_id'] for r in manifest['images'])
    assert all((incoming/(s+'.tar.gz')).exists() for s in dex.SUBJECTS)
    with pytest.raises(FileExistsError):dex.acquire(root,hashes,config)


def test_remove_only_explicit_owned_verified_archive_after_pass(tmp_path,monkeypatch):
    root,incoming,hashes,config=setup(tmp_path,monkeypatch)
    report=dex.acquire(root,hashes,config,remove_owned_archives=True)
    assert report['disposable_archives_removed'] is True and not list(incoming.iterdir())


def test_wrong_archive_pin_fail_preserves_originals_seals_no_private(tmp_path,monkeypatch):
    root,incoming,hashes,config=setup(tmp_path,monkeypatch);hashes[dex.SUBJECTS[0]]='0'*64
    with pytest.raises(ValueError,match='size/SHA'):dex.acquire(root,hashes,config)
    report=json.loads((root/dex.BASE/'report.json').read_text())
    assert report['status']=='fail' and report['source_rehashed_after']
    assert len(list(incoming.iterdir()))==2 and not list((root/dex.BASE/'inputs').iterdir())


def test_jpeg_grid_failure_prunes_only_own_bytes_and_keeps_failed_receipt(tmp_path,monkeypatch):
    root,incoming,hashes,config=setup(tmp_path,monkeypatch)
    original=dex.jpeg_size;calls=[]
    def bad(path):
        calls.append(path)
        return (1,2) if len(calls)==3 else original(path)
    monkeypatch.setattr(dex,'jpeg_size',bad)
    with pytest.raises(ValueError,match='grid'):dex.acquire(root,hashes,config)
    report=json.loads((root/dex.BASE/'report.json').read_text())
    assert report['owned_partial_outputs_removed'] and report['status']=='fail'
    assert list((root/dex.BASE/'inputs').iterdir())==[] and list((root/dex.BASE/'eval_private').iterdir())==[]
    assert len(list(incoming.iterdir()))==2


def test_source_mutation_postphase_is_failed_not_pass(tmp_path,monkeypatch):
    root,incoming,hashes,config=setup(tmp_path,monkeypatch)
    original=dex.extract_archive
    def mutate(*args,**kwargs):
        result=original(*args,**kwargs);config.chmod(0o644);config.write_text('{}');config.chmod(0o444);return result
    monkeypatch.setattr(dex,'extract_archive',mutate)
    with pytest.raises(ValueError):dex.acquire(root,hashes,config)
    report=json.loads((root/dex.BASE/'report.json').read_text())
    assert report['status']=='fail' and report['source_rehashed_after'] is False


def test_samebyte_archive_replacement_rejected_by_original_stat(tmp_path,monkeypatch):
    root,incoming,hashes,config=setup(tmp_path,monkeypatch);original=dex.extract_archive
    def replace(*args,**kwargs):
        result=original(*args,**kwargs)
        path=args[0];raw=path.read_bytes();path.unlink();path.write_bytes(raw)
        return result
    monkeypatch.setattr(dex,'extract_archive',replace)
    with pytest.raises(ValueError,match='archives changed'):dex.acquire(root,hashes,config)
    report=json.loads((root/dex.BASE/'report.json').read_text())
    assert report['status']=='fail' and report['owned_partial_outputs_removed']


def test_combined_retained_cap_rejects_second_archive_not16gb(tmp_path,monkeypatch):
    root,incoming,hashes,config=setup(tmp_path,monkeypatch)
    size_one=6*(len(b'NOT_PARSED_TARGET_METADATA')+2*(len(jpeg())+len(b'OPAQUE_PRIVATE_NPZ_BYTES')))
    monkeypatch.setattr(dex,'RETAINED_CAP',size_one+1)
    with pytest.raises(ValueError,match='Retained-byte'):dex.acquire(root,hashes,config)
    report=json.loads((root/dex.BASE/'report.json').read_text())
    assert report['status']=='fail' and report['owned_partial_outputs_removed']
    assert not list((root/dex.BASE/'inputs').iterdir())


def test_identity_excludes_atime_and_rejects_alias(tmp_path):
    path=tmp_path/'x';path.write_bytes(b'tiny');assert dex.identity(path)['bytes']==4
    alias=tmp_path/'link';alias.symlink_to(path)
    with pytest.raises(ValueError):dex.identity(alias)


def test_no_models_network_pickle_private_decode_or_frame_replacement():
    source=Path(SPEC.origin).read_text()
    assert all(x not in source for x in ('import torch','import numpy','np.load','yaml.load','pickle.load','urlopen','requests.','gdown'))
    assert 'signal.alarm(BUDGET)' in source and 'signal.alarm(CLEANUP_GRACE)' in source
