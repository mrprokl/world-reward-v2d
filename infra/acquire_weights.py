"""Remote-only acquisition of pinned pretrained assets; never writes token values."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import signal
import stat
import sys
import time
import urllib.request
import zipfile


MHR_PROTOCOL = "configs/mhr_official_release_protocol_v1.json"
MHR_PROTOCOL_PIN = dict(bytes=1914, sha256="7b5128a0acee793501d685299510933ee4beb2f4605d222beebaae7fa7b84ce1")
MHR_HELPERS = ("infra/acquire_weights.py", "infra/acquire_weights.sh", MHR_PROTOCOL,
               "infra/mediapipe_cpu_runtime_verify.py")


def _mhr_opener():
    from urllib.parse import urlsplit
    class PublicRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, request, fp, code, msg, headers, newurl):
            u = urlsplit(newurl)
            if (u.scheme != 'https' or u.username or u.password or u.fragment or u.hostname not in
                    {'github.com', 'api.github.com', 'raw.githubusercontent.com',
                     'release-assets.githubusercontent.com', 'objects.githubusercontent.com'}):
                raise ValueError('Nonprimary release redirect refused')
            return super().redirect_request(request, fp, code, msg, headers, newurl)
    return urllib.request.build_opener(urllib.request.ProxyHandler({}), PublicRedirect())


def _mhr_fetch(url, maximum):
    with _mhr_opener().open(urllib.request.Request(url, headers={'User-Agent':'WorldReward-source-acquisition'}), timeout=30) as response:
        raw = response.read(maximum+1)
    if not 0 < len(raw) <= maximum: raise ValueError('Bounded primary metadata required')
    return raw


def _mhr_download(url, path, maximum):
    with _mhr_opener().open(urllib.request.Request(url, headers={'User-Agent':'WorldReward-source-acquisition',
            'Accept-Encoding':'identity'}), timeout=30) as response, path.open('r+b') as stream:
        if path.stat().st_size: raise ValueError('Fresh owned partial required')
        if response.headers.get('Content-Encoding', 'identity') != 'identity': raise ValueError('Identity archive encoding required')
        count = 0
        while block := response.read(min(1<<20, maximum-count+1)):
            count += len(block)
            if count > maximum: raise ValueError('Exact release download cap exceeded')
            stream.write(block)
        stream.flush(); os.fsync(stream.fileno())


def mhr_release(root, code, revision, rt, *, fetch=_mhr_fetch, download=_mhr_download):
    """Authenticate one independent official release; never load/replace a model."""
    start = time.monotonic()
    def expired(*_): raise TimeoutError('Official MHR release exceeded300s')
    old_alarm = signal.signal(signal.SIGALRM, expired); old_term = signal.signal(signal.SIGTERM, expired)
    signal.alarm(300); out = partial = None; lease = None; before = existing_before = None
    report = dict(stage='mhr_official_release_license_v1', status='fail', phase='source',
        models_loaded=False, packages_installed=False, gpu_used=False, dataset_read=False,
        sam_provenance_relabelled=False, competition_eligibility_verified=False,
        training_overlap_verified=False, adoption=False)
    try:
        before = rt.source(root, code, revision, 'acquire_weights', MHR_HELPERS)
        protocol = rt.pinned(code/MHR_PROTOCOL, MHR_PROTOCOL_PIN, 16<<10)
        rt.require(protocol['schema']=='world_reward.mhr_official_release_protocol.v1' and protocol['budget_seconds']==300,
                   'Exact release protocol required')
        model = rt.canonical(root/protocol['model']['existing_path']); wanted = {k:protocol['model'][k] for k in ('bytes','sha256')}
        rt.require(model.lstat().st_uid in protocol['model']['accepted_uids'], 'Unexpected existing model owner')
        existing_before = rt.identity(model, wanted['bytes'], readonly=False)
        rt.require(existing_before==wanted, 'Existing standalone MHR bytes differ')
        destination = rt.canonical(root/protocol['namespace']); destination.mkdir(mode=0o700); out = destination
        report.update(producer_revision=revision, source_before=before, protocol_identity=MHR_PROTOCOL_PIN,
                      existing_model=existing_before, phase='primary_metadata')
        primary = {}
        for name, pin in protocol['primary_texts'].items():
            raw = fetch(pin['url'], pin['bytes'])
            identity = dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
            rt.require(identity=={k:pin[k] for k in ('bytes','sha256')}, 'Primary source text differs')
            rt.require(b'Apache' in raw, 'Primary MHR Apache declaration missing')
            primary[name] = identity
            with (out/('primary-'+name)).open('xb') as stream: stream.write(raw); os.fchmod(stream.fileno(),0o444)
        release = protocol['release']; raw = fetch(release['url'], release['maximum_metadata_bytes'])
        metadata = rt.strict(raw); archive = protocol['archive']
        rt.require(all(type(metadata.get(k)) is str and metadata[k]==release[k] for k in ('published_at','body'))
            and metadata.get('tag_name')==release['tag'] and metadata.get('draft') is False
            and metadata.get('prerelease') is False, 'Original public release metadata differs')
        assets = metadata.get('assets'); rt.require(type(assets) is list, 'Primary release asset inventory required')
        rows = [r for r in assets if r.get('name')==archive['file']]
        rt.require(len(rows)==1 and type(rows[0].get('size')) is int and rows[0]['size']==archive['bytes']
            and rows[0].get('digest')=='sha256:'+archive['sha256']
            and rows[0].get('browser_download_url')==archive['url'], 'Published archive identity differs')
        with (out/'release-metadata.json').open('xb') as stream: stream.write(raw); os.fchmod(stream.fileno(),0o444)
        report.update(primary_texts=primary, primary_metadata=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()), phase='download')
        partial = out/'assets.zip.part'
        with partial.open('xb') as stream: os.fchmod(stream.fileno(),0o600)
        lease = partial.lstat(); download(archive['url'],partial,archive['bytes']); partial.chmod(0o400)
        current=partial.lstat();rt.require((current.st_dev,current.st_ino,current.st_uid)==(lease.st_dev,lease.st_ino,lease.st_uid), 'Downloaded partial ownership changed')
        rt.require(rt.identity(partial,archive['bytes'])=={k:archive[k] for k in ('bytes','sha256')}, 'Whole published archive SHA differs')
        report.update(archive={k:archive[k] for k in ('bytes','sha256')},whole_archive_sha_verified=True,phase='inventory')
        limits = protocol['archive_limits']; inventory = []; seen = set(); expanded = 0
        with zipfile.ZipFile(partial) as z:
            for member in z.infolist():
                name=member.filename; path=PurePosixPath(name); mode=member.external_attr>>16
                rt.require(name and member.orig_filename==name and not any(ord(c)<32 or ord(c)==127 for c in name) and '\\' not in name and not path.is_absolute() and '..' not in path.parts
                    and name==str(path)+('/' if member.is_dir() else '') and name not in seen and not member.flag_bits&1
                    and member.compress_type in (zipfile.ZIP_STORED,zipfile.ZIP_DEFLATED)
                    and (stat.S_IFMT(mode) in (0,stat.S_IFDIR) if member.is_dir() else stat.S_IFMT(mode) in (0,stat.S_IFREG))
                    and member.file_size<=limits['member_bytes'], 'Unsafe release ZIP member')
                seen.add(name); expanded+=member.file_size
                rt.require(len(seen)<=limits['members'] and expanded<=limits['expanded_bytes'], 'Release inventory bound exceeded')
                inventory.append(dict(name=name,bytes=member.file_size,CRC=member.CRC,directory=member.is_dir()))
            rt.require('assets/LICENSE' in seen and protocol['model']['member'] in seen, 'Official asset LICENSE/model missing')
            report.update(archive_inventory=inventory, expanded_bytes=expanded, phase='asset_proof')
            notices=[]; total=0
            for member in z.infolist():
                if member.is_dir() or PurePosixPath(member.filename).name.upper() not in {'LICENSE','LICENSE.TXT','LICENSE.MD','NOTICE','NOTICE.TXT','COPYING'}: continue
                rt.require(member.file_size<=limits['notice_bytes'], 'Asset notice bound exceeded')
                raw=z.read(member);total+=len(raw);rt.require(total<=limits['total_notice_bytes'], 'Total notice cap exceeded')
                pin=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
                output='asset-notice-'+str(len(notices))+'.txt'
                with (out/output).open('xb') as stream: stream.write(raw);os.fchmod(stream.fileno(),0o444)
                notices.append(dict(member=member.filename,file=output,**pin))
                report['asset_notices']=notices
                if member.filename=='assets/LICENSE': rt.require(pin==primary['LICENSE'], 'Embedded asset licence differs from primary Apache text')
            member=z.getinfo(protocol['model']['member']);rt.require(not member.is_dir() and member.file_size==wanted['bytes'], 'Official model member length differs')
            digest=hashlib.sha256()
            with z.open(member) as stream:
                for block in iter(lambda:stream.read(1<<20),b''):digest.update(block)
            rt.require(digest.hexdigest()==wanted['sha256'], 'Official model member differs from existing JIT')
            rt.require(rt.identity(partial,archive['bytes'])==report['archive'], 'Whole archive changed after member interpretation')
            report.update(asset_notices=notices,asset_license_matches_primary_exactly=True,archive_rehashed_after=True,
                member_model=wanted,model_byte_identical=True,model_copy_written=False,
                selected_members_crc_verified=True,all_members_crc_verified=False,
                purpose=protocol['purpose'],phase='posthash')
    except Exception as error:
        report.update(error_type=type(error).__name__)
    finally:
        # The bounded grace is solely hash/owned-file cleanup, never resumed acquisition.
        signal.alarm(20); post=True
        try:
            if before is not None: rt.require(rt.source(root,code,revision,'acquire_weights',MHR_HELPERS)==before,'Source changed')
            if existing_before is not None: rt.require(rt.identity(model,wanted['bytes'],readonly=False)==existing_before,'Original model changed')
            report.update(source_rehashed_after=before is not None,existing_model_rehashed_after=existing_before is not None)
        except Exception as error: post=False;report.update(postcheck_error_type=type(error).__name__)
        try:
            if partial is not None and partial.exists():
                current=partial.lstat();rt.require(lease is not None and stat.S_ISREG(current.st_mode)
                    and current.st_nlink==1 and (current.st_dev,current.st_ino,current.st_uid)==(lease.st_dev,lease.st_ino,lease.st_uid), 'Unknown partial cannot be removed')
                partial.unlink()
            report['owned_archive_removed']=True
        except Exception as error: post=False;report.update(cleanup_error_type=type(error).__name__)
        report['elapsed_seconds']=time.monotonic()-start
        if report['phase']=='posthash' and post and report['elapsed_seconds']<=300: report.update(status='pass',phase='complete')
        if out is not None:
            with (out/'report.json').open('xb') as stream:
                stream.write((json.dumps(report,sort_keys=True,allow_nan=False)+'\n').encode());stream.flush();os.fsync(stream.fileno());os.fchmod(stream.fileno(),0o444)
                out.chmod(0o555)
                report.update(elapsed_seconds=time.monotonic()-start,receipt_sealing_budget_checked=True)
                if report['status']=='pass' and report['elapsed_seconds']>300: report.update(status='fail',phase='receipt_deadline')
                stream.seek(0);stream.truncate();stream.write((json.dumps(report,sort_keys=True,allow_nan=False)+'\n').encode());stream.flush();os.fsync(stream.fileno())
                if report['status']=='pass' and time.monotonic()-start>300:
                    report.update(status='fail',phase='receipt_deadline',elapsed_seconds=time.monotonic()-start)
                    stream.seek(0);stream.truncate();stream.write((json.dumps(report,sort_keys=True,allow_nan=False)+'\n').encode());stream.flush();os.fsync(stream.fileno())
        signal.alarm(0);signal.signal(signal.SIGALRM,old_alarm);signal.signal(signal.SIGTERM,old_term)
    if report['status']!='pass': raise RuntimeError('Official MHR release proof failed: '+report.get('error_type',report.get('postcheck_error_type','deadline')))
    return report


def main(argv=None) -> None:
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--mhr-release-only',action='store_true');arguments=sys.argv[1:] if argv is None else argv
    args=parser.parse_args(arguments)
    if arguments.count('--mhr-release-only')>1: parser.error('Release mode must be specified exactly once')
    if platform.system() != "Linux":
        raise RuntimeError("Model downloads are restricted to Azure Linux")
    root = Path(os.environ.get("WR_ROOT", "/srv/scenesmith/world-reward"))
    if args.mhr_release_only:
        code=Path(os.environ['WR_CODE']);sys.path.insert(0,str(code/'infra'))
        import mediapipe_cpu_runtime_verify as rt
        if Path(rt.__file__).resolve()!=code/'infra/mediapipe_cpu_runtime_verify.py': raise ValueError('Actual immutable helper required')
        rt.require(Path(__file__).resolve()==code/'infra/acquire_weights.py','Actual immutable acquisition source required')
        report=mhr_release(root,code,os.environ['WR_CODE_REVISION'],rt)
        print(json.dumps({k:report[k] for k in ('stage','status','elapsed_seconds','model_byte_identical')}));return
    os.environ["HF_TOKEN"] = (root / ".secrets/hf_token").read_text().strip()
    os.environ["HF_HOME"] = str(root / "cache/huggingface")
    from huggingface_hub import snapshot_download

    weights = root / "weights"
    records = []
    specs = [
        ("nvidia/cari4d_commercial", "1f7287ac6fd5f72c30ce2222fb345a3e7d779fc9", "cari4d/cari4d", ["2026-08-25-09-35-57/*"]),
        ("facebook/sam-3d-body-dinov3", "11aaa346c7204874a1cbafe3d39a979080b2c55a", "cari4d/sam3d_body/checkpoints/sam-3d-body-dinov3", ["model.ckpt", "model_config.yaml", "assets/*", "LICENSE", "README.md"]),
        ("facebook/sam-3d-objects", "2e73555018d2741ccd486e56c24fac41155a1dc6", "sam3d/hf-download", ["checkpoints/*", "LICENSE"]),
        ("facebook/sam2.1-hiera-large", "665f8e2ad61cf5f53d65644ff27c8ee525124610", "sam2", ["sam2.1_hiera_large.pt", "sam2.1_hiera_l.yaml", "README.md"]),
        ("IDEA-Research/grounding-dino-base", "12bdfa3120f3e7ec7b434d90674b3396eccf88eb", "grounding_dino", ["*.json", "*.txt", "model.safetensors", "README.md"]),
    ]
    for repo, revision, folder, patterns in specs:
        print(json.dumps({"acquiring": repo, "revision": revision}), flush=True)
        snapshot_download(repo_id=repo, revision=revision, local_dir=weights/folder, allow_patterns=patterns)
        records.append({"repo_id":repo,"revision":revision,"path":str(weights/folder)})
    checkpoint = weights / "cari4d/cari4d/2026-08-25-09-35-57/step200000.pth"
    with checkpoint.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != "78ff5cb874dd012a272382e3f2d8bc11226d5b7d0ecc739a60fbb4a97a5a5ba3":
        raise RuntimeError("Pinned CARI4D checkpoint integrity failed")
    for repo, revision, cache in [
        ("Ruicheng/moge-2-vitl-normal", "b135031bae30b5ac2ae141a0e68717795ce38340", weights/"cari4d/hf_home/hub"),
        ("Ruicheng/moge-vitl", "ad326bfb61facd6c52b5a825bc1e34d7c97d9672", weights/"sam3d/hf_home/hub"),
    ]:
        snapshot_download(repo_id=repo, revision=revision, cache_dir=cache, allow_patterns=["model.pt", "README.md"])
        records.append({"repo_id":repo,"revision":revision,"cache_dir":str(cache)})
    model_dir = weights / "mhr"
    model_dir.mkdir(parents=True, exist_ok=True)
    model = model_dir / "mhr_model.pt"
    if not model.exists():
        archive = model_dir / "assets.zip"
        urllib.request.urlretrieve("https://github.com/facebookresearch/MHR/releases/download/v1.0.1/assets.zip", archive)
        temporary = model.with_suffix(".part")
        with zipfile.ZipFile(archive) as z:
            with z.open("assets/mhr_model.pt") as src, temporary.open("wb") as dst:
                while chunk := src.read(1024*1024):
                    dst.write(chunk)
        with temporary.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != "352e271a6c42729c68554ceaea0c955e866970160c31e35506d782dc0f7377bc":
                raise RuntimeError("Reference MHR integrity failed")
        temporary.replace(model)
        archive.unlink()
    with model.open("rb") as stream:
        mhr_digest = hashlib.file_digest(stream,"sha256").hexdigest()
    if mhr_digest != "352e271a6c42729c68554ceaea0c955e866970160c31e35506d782dc0f7377bc":
        raise RuntimeError("Reference MHR integrity failed")
    result = {"assets":records,"cari4d_sha256":digest,"mhr_license":"Apache-2.0",
              "scope":"principal_hf_and_mhr_assets_only",
              "auxiliary_assets_required":["FoundationPose", "DINOv2", "DINOv3_torch_hub"]}
    (root/"results/weights-acquisition.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"principal_assets":"complete","auxiliary_assets":"separate_stage",
                      "cari4d_integrity":"verified"}))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # Authentication failures must remain actionable without exposing secrets.
        token = os.environ.get("HF_TOKEN")
        message = str(exc).replace(token, "[REDACTED]") if token else str(exc)
        print(json.dumps({"error": type(exc).__name__, "message": message}), file=sys.stderr)
        sys.exit(1)
