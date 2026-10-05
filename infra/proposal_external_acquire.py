"""Frozen32 external RGB acquisition: private rights/slots, public RGB only.

Reuses unchanged original creator/HTTPS/JPEG acquisition mechanics. No labels,
predictions, retry, replacement, image transformation or model execution.
"""
import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mediapipe_cpu_runtime_verify as rt
import openimages_joint_pair_acquire as acq
import proposal_external_census as prior

ROOT = Path('/srv/scenesmith/world-reward')
DATA = Path('/srv/world-reward-data/proposal_external_rgb_v1')
ENTRY = 'run_proposal_external_acquire'
CONFIG = 'configs/proposal_external_acquire_v1.json'
HELPERS = ('infra/proposal_external_acquire.py', 'infra/run_proposal_external_acquire.sh', CONFIG,
           'infra/mediapipe_cpu_runtime_verify.py', 'infra/openimages_joint_pair_acquire.py',
           'infra/proposal_external_census.py', 'infra/openimages_joint_pair_census.py')
PUBLIC_KEYS = {'image_id', 'file', 'bytes', 'sha256', 'width', 'height'}


def encode(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False)+'\n').encode()


def pin(raw):
    return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def write(path, value):
    raw = encode(value); rt.write(path, raw, 0o400); return pin(raw)


def check(deadline):
    rt.require(time.monotonic() < deadline, 'Inclusive300s acquisition deadline reached')


def configuration(code, source):
    cfg = rt.pinned(code/CONFIG, source['helpers'][CONFIG], 16 << 10)
    expected = dict(schema='world_reward.proposal_external_acquire.v1', output=str(DATA), slots=32,
        dev=16, reserved=16, workers=6, overall=300, request_timeout=15, max_image_bytes=16 << 20,
        max_creator_page_bytes=2 << 20, max_decoded_pixels=16 << 20, max_total_image_bytes=512 << 20,
        public_id_namespace='world_reward.proposal_external_rgb_v1/', no_replacements=True, retry_count=0)
    rt.require(all(type(cfg.get(k)) is type(v) and cfg[k] == v for k, v in expected.items()), 'Frozen32 acquisition scope differs')
    for name, wanted in cfg['reused_helper_pins'].items():
        rt.require(source['helpers'].get(name) == wanted, 'Original imported helper differs')
    rt.require(set(cfg['reused_helper_pins']) == set(HELPERS[3:]), 'All imported helper pins required')
    for module, name in ((rt, HELPERS[3]), (acq, HELPERS[4]), (prior, HELPERS[5]), (prior.census, HELPERS[6])):
        rt.require(Path(module.__file__).resolve() == code/name, 'Foreign imported helper origin')
    return cfg


def authenticate(code, revision):
    source = rt.source(ROOT, code, revision, ENTRY, HELPERS); cfg = configuration(code, source)
    old = cfg['census']; old_code = ROOT/'jobs'/old['producer_revision']/prior.ENTRY/'code'
    old_source = rt.source(ROOT, old_code, old['producer_revision'], prior.ENTRY, prior.HELPERS)
    rt.require(old_source['helpers'] == cfg['census_helper_pins'], 'Actual census helper lineage differs')
    report_path, cohort_path = ROOT/old['report']['path'], ROOT/old['cohort']['path']
    report = rt.pinned(report_path, old['report']['pin'], 1 << 20)
    cohort = rt.pinned(cohort_path, old['cohort']['pin'], 1 << 20)
    expected = dict(schema='world_reward.proposal_external_census.v1', status='pass',
        producer_revision=old['producer_revision'], source_binding=old_source,
        configuration_identity=cfg['census_helper_pins'][prior.CONFIG], source_and_inputs_rehashed_after=True,
        cohort_identity=old['cohort']['pin'], decision='FROZEN32_PENDING_INDIVIDUAL_RIGHTS_AND_SEPARATE_INFERENCE',
        network_used=False, rgb_read=False, predictions_read=False, models_loaded=False, gpu_used=False,
        old_studies_reopened=False, rights_requests_before_freeze=0)
    rt.require(all(type(report.get(k)) is type(v) and report[k] == v for k,v in expected.items()), 'Actual frozen census PASS required')
    old_cfg = rt.pinned(old_code/prior.CONFIG, cfg['census_helper_pins'][prior.CONFIG], 16 << 10)
    rt.require(report['frozen_inputs'] == old_cfg['files'] == cohort['frozen_input_identities']
        and report['counts']['excluded_ids'] == 208
        and report['counts']['unique_author_photo_md5_slots'] >= 32, 'Actual census inputs/slot availability differs')
    rt.require(cohort['schema'] == 'world_reward.proposal_external_cohort.v1'
        and cohort['producer_revision'] == old['producer_revision'] and cohort['source_binding'] == old_source
        and cohort['configuration_identity'] == expected['configuration_identity']
        and cohort['no_replacements'] is True and cohort['rights_requests_before_freeze'] == 0
        and cohort['reference_geometry_exposed'] is False and cohort['hash_namespace'] == prior.HASH_NAMESPACE,
        'Original metadata-only cohort required')
    rows = cohort['records']
    rt.require(len(rows) == 32 and [r['slot'] for r in rows] == list(range(32))
        and [r['split'] for r in rows] == ['DEV']*16+['RESERVED']*16
        and all(set(r) == {'slot','split','publisher_metadata'} and set(r['publisher_metadata']) == set(acq.METADATA_KEYS) for r in rows),
        'All original ordered32 frozen slots required')
    frozen = {report_path:old['report']['pin'], cohort_path:old['cohort']['pin']}
    frozen.update({Path(v['path']):v['pin'] for v in old_cfg['files'].values()})
    for path, wanted in frozen.items():
        rt.require(rt.identity(path, 100 << 20, readonly=path in (report_path,cohort_path) or path == Path(old_cfg['files']['old64']['path'])) == wanted,
                   'Original census metadata input changed')
    return cfg, rows, source, old_source, frozen, cohort_path


def make_public(records, private, public, cfg, deadline):
    """Move authenticated original JPEG inodes; no copy/link/re-encoding."""
    images = []; mappings = []
    for r in records:
        check(deadline); folder = private/r['image_id']
        stored = rt.pinned(folder/'record.json', r['record_pin'], 1 << 20)
        rt.require(stored == {k:v for k,v in r.items() if k != 'record_pin'}, 'Original per-slot acquisition record differs')
        files = {'record.json'} | ({'rights.json'} if 'rights_pin' in r else set()) | ({'rgb.jpg'} if 'image_pin' in r else set())
        rt.require({x.name for x in folder.iterdir()} == files, 'Foreign/partial per-slot artifact')
        for field, name in (('rights_pin','rights.json'), ('image_pin','rgb.jpg')):
            if field in r:
                rt.require(rt.identity(folder/name, cfg['max_image_bytes']) == r[field], 'Acquired original leaf differs')
        folder.chmod(0o700)
        for leaf in folder.iterdir():
            leaf.chmod(0o400)
        if r['status'] == 'acquired':
            rt.require(r['creator_grant_verified'] is True and r['publisher_md5_matched'] is True
                and r['rotation'] == '0.0' and r['image_transformation'] == 'none_original_file', 'Original acquired JPEG/rights required')
            name = f"image_{r['slot']:06d}.jpg"; source_file = folder/'rgb.jpg'; target = public/name
            before = source_file.lstat(); source_file.rename(target); after = target.lstat()
            rt.require((before.st_dev,before.st_ino,before.st_size) == (after.st_dev,after.st_ino,after.st_size)
                and rt.identity(target, cfg['max_image_bytes']) == r['image_pin'], 'Original JPEG move identity differs')
            header = r['jpeg_header']; opaque = hashlib.sha256((cfg['public_id_namespace']+r['image_id']).encode()).hexdigest()[:32]
            images.append(dict(image_id=opaque,file=name,**r['image_pin'],width=header['width'],height=header['height']))
            mappings.append(dict(slot=r['slot'], public_image_id=opaque, public_file=name, image_pin=r['image_pin']))
    rt.require(len(images) == len({r['image_id'] for r in images}) and all(set(r) == PUBLIC_KEYS for r in images),
               'Opaque public six-key RGB inventory required')
    manifest = dict(schema='world_reward.rgb_proposal_inputs.v1', images=images)
    public_pin = write(public/'manifest.json', manifest); public.chmod(0o500)
    return public_pin, mappings


def verify_output(records, mappings, private, public, public_pin, cfg):
    manifest = rt.pinned(public/'manifest.json', public_pin, 1 << 20)
    rt.require(set(manifest) == {'schema','images'} and manifest['schema'] == 'world_reward.rgb_proposal_inputs.v1', 'Private data in public manifest')
    rt.require({x.name for x in public.iterdir()} == {'manifest.json'} | {m['public_file'] for m in mappings}, 'Exclusive public RGB inventory required')
    mapped = {m['slot']:m for m in mappings}
    expected_images=[]
    for r in records:
        folder = private/r['image_id']; expected = {'record.json'} | ({'rights.json'} if 'rights_pin' in r else set())
        rt.require({x.name for x in folder.iterdir()} == expected, 'Exclusive private slot/rights inventory required')
        for field,name in (('record_pin','record.json'),('rights_pin','rights.json')):
            if field in r:
                rt.require(rt.identity(folder/name, cfg['max_image_bytes']) == r[field], 'Private leaf changed after publication')
        rt.require((r['status'] == 'acquired') == (r['slot'] in mapped), 'Missing slots cannot have fake RGB')
        if r['slot'] in mapped:
            m = mapped[r['slot']]; rt.require(rt.identity(public/m['public_file'], cfg['max_image_bytes']) == r['image_pin'], 'Public original JPEG changed')
            expected_images.append(dict(image_id=m['public_image_id'],file=m['public_file'],**r['image_pin'],
                                        width=r['jpeg_header']['width'],height=r['jpeg_header']['height']))
    rt.require(manifest['images']==expected_images and all(set(r)==PUBLIC_KEYS for r in manifest['images'])
               and len(mapped)==len(mappings), 'Exact complete public six-key manifest differs')
    for path in (private,public,*private.rglob('*'),*public.iterdir()):
        s = path.lstat(); rt.require(s.st_uid == os.geteuid() and not s.st_mode & 0o077
            and (stat.S_ISDIR(s.st_mode) or stat.S_ISREG(s.st_mode)) and not path.is_symlink()
            and (stat.S_ISDIR(s.st_mode) or stat.S_IMODE(s.st_mode) == 0o400), 'Root-only private/public output modes required')


def run():
    started = time.monotonic(); deadline = started+300
    rt.require(sys.platform == 'linux' and os.geteuid() == 0 and os.uname().nodename == 'world-reward-ncc-h100-02', 'Exact Azure root CPU acquisition host required')
    revision,code = os.environ['WR_CODE_REVISION'],Path(os.environ['WR_CODE'])
    rt.require(Path(__file__).resolve() == code/HELPERS[0], 'Original immutable acquisition driver required')
    cfg,selected,source,census_source,frozen,cohort_path = authenticate(code,revision); check(deadline)
    out = rt.canonical(DATA); rt.require(not out.exists() and out.parent.is_dir(), 'Fresh acquisition namespace required')
    old_umask = os.umask(0o077); out.mkdir(mode=0o700); private=out/'eval_private'; public=out/'inputs'
    private.mkdir(mode=0o700); public.mkdir(mode=0o700)
    report = dict(schema=cfg['schema'],producer_revision=revision,source_binding=source,census_source_binding=census_source,
        configuration_identity=source['helpers'][CONFIG], census=cfg['census'], status='fail',records=[],
        selected_slots=selected,public_mappings=[],
        network_used=True,models_loaded=False,gpu_used=False,reference_geometry_read=False,predictions_read=False,
        retry_count=0,replacement_count=0,all32_slots_retained=True,quality_verified=False,training_overlap_verified=False,
        challenge_overlap_verified=False,adopted=False,local_heavy_transfer=False,source_and_inputs_rehashed_after=False,
        creator_name_license_landing_verified_only=True,creator_account_identity_independently_verified=False)
    def interrupted(*_):
        raise TimeoutError('Acquisition interrupted')
    handlers={s:signal.signal(s,interrupted) for s in (signal.SIGTERM,signal.SIGINT)}
    try:
        report['records']=acq.acquire_cohort(selected,private,cfg,deadline-15,cohort_path,cfg['census']['cohort']['pin'])
        public_pin,mappings=make_public(report['records'],private,public,cfg,deadline)
        report.update(public_inputs_identity=public_pin,public_mappings=mappings)
        verify_output(report['records'],mappings,private,public,public_pin,cfg)
        cfg2,rows2,source2,census2,frozen2,path2=authenticate(code,revision)
        rt.require((cfg2,rows2,source2,census2,frozen2,path2)==(cfg,selected,source,census_source,frozen,cohort_path),'Full original acquisition/census lineage changed')
        report['source_and_inputs_rehashed_after']=True; check(deadline)
        report['counts']=dict(slots=32,acquired=len(mappings),missing=32-len(mappings),
            acquired_DEV=sum(r['status']=='acquired' and r['split']=='DEV' for r in report['records']),
            acquired_RESERVED=sum(r['status']=='acquired' and r['split']=='RESERVED' for r in report['records']),
            original_rgb_bytes=sum(r.get('image_pin',{}).get('bytes',0) for r in report['records']))
        report['status']='pass'
    except BaseException as exc:
        report['error_type']=type(exc).__name__ if type(exc) in (ValueError,TimeoutError,OSError,KeyError) else 'other'; raise
    finally:
        try:
            report['elapsed_seconds']=time.monotonic()-started
            if report['elapsed_seconds']>=300:report.update(status='fail',error_type='TimeoutError')
            with (private/'manifest.json').open('xb') as stream:
                os.fchmod(stream.fileno(),0o400); raw=encode(report);stream.write(raw);stream.flush();os.fsync(stream.fileno())
                # Also seal safely on an incomplete transport/IO failure; the
                # frozen selected_slots ledger is retained independently.
                for leaf in (*private.rglob('*'),*public.iterdir()):
                    rt.canonical(leaf)
                    if leaf.is_file():leaf.chmod(0o400)
                for directory in sorted((p for p in private.rglob('*') if p.is_dir()),reverse=True):directory.chmod(0o500)
                public.chmod(0o500)
                out.chmod(0o500);private.chmod(0o500)
                report['elapsed_seconds']=time.monotonic()-started
                if report['elapsed_seconds']>=300:report.update(status='fail',error_type='TimeoutError')
                raw=encode(report);stream.seek(0);stream.write(raw);stream.truncate();stream.flush();os.fsync(stream.fileno())
                if time.monotonic()-started>=300 and report['status']=='pass':
                    report.update(status='fail',error_type='TimeoutError',elapsed_seconds=time.monotonic()-started)
                    raw=encode(report);stream.seek(0);stream.write(raw);stream.truncate();stream.flush();os.fsync(stream.fileno())
            print(json.dumps(dict(status=report['status'],counts=report.get('counts'),manifest=pin(raw),
                                  public_inputs=report.get('public_inputs_identity'),elapsed_seconds=report['elapsed_seconds']),sort_keys=True))
        finally:
            os.umask(old_umask)
            for sig,handler in handlers.items():signal.signal(sig,handler)
    return report


if __name__ == '__main__':
    rt.require(len(sys.argv)==1,'No sample/query/retry parameters allowed');run()
