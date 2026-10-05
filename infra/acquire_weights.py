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
MHR_FAILED_REVISION = '4ed17ad0846b3acbf020fe29e1f58cd272c4ebc4'
MHR_FAILED_REPORT = dict(bytes=2270,sha256='a6f68fac3a81a2cba27bed853e907aeb66ab596bf1c6deb46d194aec3f5c2e31')
MHR_FAILED_ARCHIVE = 'd74c1263edefa0e531f2d71394c4e8c62f19b7434fe3d8defc285fea582ed54b'
MHR_SELECTED_PROTOCOL = 'configs/mhr_official_release_selected_protocol_v1.json'
MHR_SELECTED_PIN = dict(bytes=4428,sha256='ad418f042446148f22ec4f4d20b3fd8ec1ff2541d4d9e577c1382e5339adb3cb')
MHR_SELECTED_HELPERS = (*MHR_HELPERS,MHR_SELECTED_PROTOCOL)


def _mhr_failure(rt, root, protocol):
    out=rt.canonical(root/'results/mhr-official-release-license-v1')
    rt.require(out.stat().st_mode&0o777==0o555 and {p.name for p in out.iterdir()}==
        {'primary-LICENSE','primary-README.md','release-metadata.json','report.json'}, 'Exact original failed inventory required')
    old=rt.pinned(out/'report.json',MHR_FAILED_REPORT,4<<20)
    rt.require(old.get('stage')=='mhr_official_release_license_v1' and old.get('status')=='fail'
        and old.get('phase')=='inventory' and old.get('producer_revision')==MHR_FAILED_REVISION
        and old.get('protocol_identity')==MHR_PROTOCOL_PIN and old.get('whole_archive_sha_verified') is True
        and old.get('owned_archive_removed') is True and old.get('source_rehashed_after') is True
        and old.get('existing_model_rehashed_after') is True and 'asset_notices' not in old
        and old.get('existing_model')=={k:protocol['model'][k] for k in ('bytes','sha256')}
        and old.get('archive')=={k:protocol['archive'][k] for k in ('bytes','sha256')}, 'Original failure scope differs')
    code=root/'jobs'/MHR_FAILED_REVISION/'acquire_weights/code'
    source=rt.source(root,code,MHR_FAILED_REVISION,'acquire_weights',MHR_HELPERS)
    rt.require(source==old['source_before'] and (code.parent/'source-sha256').read_bytes()==(MHR_FAILED_ARCHIVE+'\n').encode(), 'Original failed source changed')
    expected={'report.json':MHR_FAILED_REPORT,'release-metadata.json':old['primary_metadata']}
    expected.update({'primary-'+n:{k:p[k] for k in ('bytes','sha256')} for n,p in protocol['primary_texts'].items()})
    files={n:rt.identity(out/n,4<<20) for n in expected}
    rt.require(files==expected and all((out/n).stat().st_mode&0o777==0o444 for n in expected), 'Original retained text changed')
    return dict(report=MHR_FAILED_REPORT,source=source,files=files,original_status='fail')


def _mhr_inventory(z, protocol):
    """Same ordered v1 predicates; diagnostic never repairs or reads payload."""
    limits=protocol['archive_limits'];rows=[];seen=set();expanded=0;first=None;safe=True
    for index,member in enumerate(z.infolist()):
        name=member.filename;path=PurePosixPath(name);mode=member.external_attr>>16
        guards=(('name_nonempty',bool(name)),('original_name',member.orig_filename==name),
            ('name_controls',not any(ord(c)<32 or ord(c)==127 for c in name)),('backslash','\\' not in name),
            ('absolute',not path.is_absolute()),('traversal','..' not in path.parts),
            ('canonical',name==str(path)+('/' if member.is_dir() else '')),('duplicate',name not in seen),
            ('encrypted',not member.flag_bits&1),('compression',member.compress_type in (zipfile.ZIP_STORED,zipfile.ZIP_DEFLATED)),
            ('type',stat.S_IFMT(mode) in ((0,stat.S_IFDIR) if member.is_dir() else (0,stat.S_IFREG))),
            ('member_bytes',member.file_size<=limits['member_bytes']))
        rejected=next((key for key,passed in guards if not passed),None)
        seen.add(name);expanded+=member.file_size
        if rejected is None and len(seen)>limits['members']:rejected='member_count'
        if rejected is None and expanded>limits['expanded_bytes']:rejected='expanded_bytes'
        if rejected:
            safe=False
            if first is None:first=dict(index=index,guard=rejected,name=name[:1024],mode_octal=oct(mode),
                flags=member.flag_bits,compression=member.compress_type,bytes=member.file_size)
        if index<limits['members']:
            rows.append(dict(name=name[:1024],orig_filename=member.orig_filename[:1024],
                full_name_sha256=hashlib.sha256(name.encode('utf8')).hexdigest(),name_truncated=len(name)>1024,
                bytes=member.file_size,compressed_bytes=member.compress_size,CRC=member.CRC,directory=member.is_dir(),
                flags=member.flag_bits,mode_octal=oct(mode),compression=member.compress_type))
    if first is None and ('assets/LICENSE' not in seen or protocol['model']['member'] not in seen):
        first=dict(guard='required_members',assets_LICENSE_present='assets/LICENSE' in seen,
            model_present=protocol['model']['member'] in seen)
    return dict(rows=rows,count=len(z.infolist()),expanded_bytes=expanded,
        maximum_member_bytes=max((m.file_size for m in z.infolist()),default=0),
        maximum_compressed_member_bytes=max((m.compress_size for m in z.infolist()),default=0),structurally_safe=safe,
        first_v1_rejection=first,v1_inventory_accepted=first is None,inventory_rows_truncated=len(z.infolist())>limits['members'])


def _mhr_previous_inventory(rt, root, protocol, selected, failure):
    prior=selected['previous_inventory'];out=rt.canonical(root/prior['namespace'])
    rt.require(out.stat().st_mode&0o777==0o555 and {p.name for p in out.iterdir()}==
        {'primary-LICENSE','primary-README.md','release-metadata.json','report.json'},'Exact original diagnostic inventory required')
    report=rt.pinned(out/'report.json',prior['report'],4<<20)
    code=root/'jobs'/prior['revision']/'acquire_weights/code'
    source=rt.source(root,code,prior['revision'],'acquire_weights',MHR_HELPERS)
    rt.require(source==report['source_before'] and (code.parent/'source-sha256').read_bytes()==
        (prior['source_archive_sha256']+'\n').encode(),'Original diagnostic source changed')
    rt.require(report.get('stage')=='mhr_official_release_inventory_v1' and report.get('status')=='pass'
        and report.get('phase')=='complete' and report.get('producer_revision')==prior['revision']
        and report.get('protocol_identity')==MHR_PROTOCOL_PIN and report.get('previous_failure')==failure
        and report.get('archive')=={k:protocol['archive'][k] for k in ('bytes','sha256')}
        and report.get('existing_model')=={k:protocol['model'][k] for k in ('bytes','sha256')}
        and all(report.get(k) is True for k in ('source_rehashed_after','existing_model_rehashed_after',
            'whole_archive_sha_verified','archive_rehashed_after','original_failure_unchanged','owned_archive_removed'))
        and all(report.get(k) is False for k in ('asset_LICENSE_verified','model_member_streamed','model_member_extracted'))
        and report.get('notice_payloads_read')==0 and report.get('asset_notices')==[],'Original diagnostic scope differs')
    inventory=report['inventory_diagnostic'];columns=selected['inventory_columns']
    rows=[[r[k] for k in columns] for r in inventory['rows']]
    rt.require(json.dumps(rows)==json.dumps(selected['inventory_rows']) and inventory['count']==len(rows)
        and inventory['inventory_rows_truncated'] is False and inventory['expanded_bytes']==selected['inventory_expanded_bytes'],
        'Complete original diagnostic descriptors differ')
    expected={'report.json':prior['report'],'release-metadata.json':report['primary_metadata']}
    expected.update({'primary-'+n:{k:p[k] for k in ('bytes','sha256')} for n,p in protocol['primary_texts'].items()})
    files={n:rt.identity(out/n,4<<20) for n in expected}
    rt.require(files==expected and all((out/n).stat().st_mode&0o777==0o444 for n in expected),'Original diagnostic texts changed')
    return dict(report=prior['report'],source=source,files=files,original_status='pass')


def _mhr_selected_inventory(z, selected, rt):
    """Authenticate all central descriptors, without decoding inactive payloads."""
    rows=[];seen=set()
    for m in z.infolist():
        name=m.filename;path=PurePosixPath(name);mode=m.external_attr>>16
        rt.require(name and name==m.orig_filename and not any(ord(c)<32 or ord(c)==127 for c in name)
            and '\\' not in name and not path.is_absolute() and '..' not in path.parts
            and name==str(path)+('/' if m.is_dir() else '') and name not in seen and not m.flag_bits&1
            and m.compress_type in (zipfile.ZIP_STORED,zipfile.ZIP_DEFLATED)
            and stat.S_IFMT(mode) in ((0,stat.S_IFDIR) if m.is_dir() else (0,stat.S_IFREG)),
            'Unsafe selected release central member')
        seen.add(name)
        row=dict(name=name,bytes=m.file_size,compressed_bytes=m.compress_size,CRC=m.CRC,directory=m.is_dir(),
            flags=m.flag_bits,mode_octal=oct(mode),compression=m.compress_type)
        rows.append(row)
    rt.require(json.dumps([[r[k] for k in selected['inventory_columns']] for r in rows])==
        json.dumps(selected['inventory_rows']) and sum(r['bytes'] for r in rows)==selected['inventory_expanded_bytes'],
        'Exact complete selected metadata whitelist differs')
    return rows


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


def mhr_release(root, code, revision, rt, *, fetch=_mhr_fetch, download=_mhr_download, inventory_only=False, selected_only=False):
    """Authenticate one independent official release; never load/replace a model."""
    start = time.monotonic()
    def expired(*_): raise TimeoutError('Official MHR release exceeded300s')
    old_alarm = signal.signal(signal.SIGALRM, expired); old_term = signal.signal(signal.SIGTERM, expired)
    signal.alarm(300); out = partial = None; lease = None; before = existing_before = previous = previous_inventory = None
    helpers=MHR_SELECTED_HELPERS if selected_only else MHR_HELPERS;selected=None
    report = dict(stage='mhr_official_release_license_v2' if selected_only else 'mhr_official_release_inventory_v1' if inventory_only else 'mhr_official_release_license_v1', status='fail', phase='source',
        models_loaded=False, packages_installed=False, gpu_used=False, dataset_read=False,
        sam_provenance_relabelled=False, competition_eligibility_verified=False,
        training_overlap_verified=False, adoption=False)
    try:
        rt.require(type(inventory_only) is bool and type(selected_only) is bool and not (inventory_only and selected_only),'Exclusive release mode required')
        before = rt.source(root, code, revision, 'acquire_weights', helpers)
        protocol = rt.pinned(code/MHR_PROTOCOL, MHR_PROTOCOL_PIN, 16<<10)
        rt.require(protocol['schema']=='world_reward.mhr_official_release_protocol.v1' and protocol['budget_seconds']==300,
                   'Exact release protocol required')
        model = rt.canonical(root/protocol['model']['existing_path']); wanted = {k:protocol['model'][k] for k in ('bytes','sha256')}
        rt.require(model.lstat().st_uid in protocol['model']['accepted_uids'], 'Unexpected existing model owner')
        existing_before = rt.identity(model, wanted['bytes'], readonly=False)
        rt.require(existing_before==wanted, 'Existing standalone MHR bytes differ')
        if selected_only:
            selected=rt.pinned(code/MHR_SELECTED_PROTOCOL,MHR_SELECTED_PIN,16<<10)
            rt.require(selected['schema']=='world_reward.mhr_official_release_selected_protocol.v1'
                and selected['budget_seconds']==300 and selected['base_protocol']==dict(path=MHR_PROTOCOL,**MHR_PROTOCOL_PIN)
                and selected['selected_members']==['assets/LICENSE.txt',protocol['model']['member']]
                and selected['selected_license']==dict(member='assets/LICENSE.txt',**{k:protocol['primary_texts']['LICENSE'][k] for k in ('bytes','sha256')})
                and type(selected['selected_expanded_byte_cap']) is int
                and selected['selected_expanded_byte_cap']==selected['selected_license']['bytes']+wanted['bytes'],
                'Exact selective release protocol required')
        if inventory_only or selected_only:previous=_mhr_failure(rt,root,protocol)
        if selected_only:previous_inventory=_mhr_previous_inventory(rt,root,protocol,selected,previous)
        destination = rt.canonical(root/(selected['namespace'] if selected_only else 'results/mhr-official-release-inventory-v1' if inventory_only else protocol['namespace']))
        destination.mkdir(mode=0o700); out = destination
        if inventory_only:report.update(previous_failure=previous,asset_LICENSE_verified=False,model_member_streamed=False,
            model_member_extracted=False,whole_members_CRC_verified=False,diagnostic_only=True)
        if selected_only:report.update(previous_failure=previous,previous_inventory=previous_inventory,
            selected_protocol_identity=MHR_SELECTED_PIN,selected_payloads_opened=[],inactive_payloads_opened=0,
            selected_expanded_byte_cap=selected['selected_expanded_byte_cap'],old_generic_inventory_bounds_changed=False,
            embedded_license_scope='published_MHR_standalone_bytes_only_not_SAM_or_competition_clearance')
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
        limits = protocol['archive_limits']
        with zipfile.ZipFile(partial) as z:
            if selected_only:
                rows=_mhr_selected_inventory(z,selected,rt)
                report.update(archive_inventory=rows,expanded_bytes=selected['inventory_expanded_bytes'],complete_metadata_whitelist_verified=True,phase='asset_proof')
            else:
                inventory=_mhr_inventory(z,protocol)
                if not inventory_only:rt.require(inventory['v1_inventory_accepted'],'Original release inventory predicate rejected')
                rows=inventory['rows'] if inventory_only else [{k:r[k] for k in ('name','bytes','CRC','directory')} for r in inventory['rows']]
                report.update(archive_inventory=rows, expanded_bytes=inventory['expanded_bytes'], phase='asset_proof')
                if inventory_only:report['inventory_diagnostic']=inventory
            notices=[]; total=0
            notice_members=[z.getinfo(selected['selected_license']['member'])] if selected_only else z.infolist() if not inventory_only or inventory['structurally_safe'] else ()
            for member in notice_members:
                if member.is_dir() or PurePosixPath(member.filename).name.upper() not in {'LICENSE','LICENSE.TXT','LICENSE.MD','NOTICE','NOTICE.TXT','COPYING'}: continue
                rt.require(member.file_size<=limits['notice_bytes'], 'Asset notice bound exceeded')
                rt.require(total+member.file_size<=limits['total_notice_bytes'], 'Total notice cap exceeded')
                if selected_only:
                    rt.require(not member.is_dir() and member.file_size==selected['selected_license']['bytes'],'Exact selected notice length differs')
                    report['selected_payloads_opened'].append(member.filename)
                raw=z.read(member);total+=len(raw)
                pin=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
                output='asset-notice-'+str(len(notices))+'.txt'
                with (out/output).open('xb') as stream: stream.write(raw);os.fchmod(stream.fileno(),0o444)
                notice=dict(member=member.filename,file=output,**pin)
                if inventory_only:notice['matches_primary_LICENSE_exactly']=pin==primary['LICENSE']
                notices.append(notice)
                report['asset_notices']=notices
                if not inventory_only and member.filename=='assets/LICENSE': rt.require(pin==primary['LICENSE'], 'Embedded asset licence differs from primary Apache text')
                if selected_only:rt.require(pin==primary['LICENSE'],'Selected asset licence differs from complete primary Apache text')
            if inventory_only:
                rt.require(rt.identity(partial,archive['bytes'])==report['archive'],'Archive changed after central inventory')
                report.update(asset_notices=notices,notice_payloads_read=len(notices),archive_rehashed_after=True,phase='posthash')
            else:
                member=z.getinfo(protocol['model']['member']);rt.require(not member.is_dir() and member.file_size==wanted['bytes'], 'Official model member length differs')
                digest=hashlib.sha256();count=0
                if selected_only:report['selected_payloads_opened'].append(member.filename)
                with z.open(member) as stream:
                    for block in iter(lambda:stream.read(1<<20),b''):
                        count+=len(block)
                        if selected_only:rt.require(count<=wanted['bytes'] and count+total<=selected['selected_expanded_byte_cap'],'Selected decoded byte cap exceeded')
                        digest.update(block)
                if selected_only:rt.require(count==wanted['bytes'],'Selected model stream length differs')
                rt.require(digest.hexdigest()==wanted['sha256'], 'Official model member differs from existing JIT')
                rt.require(rt.identity(partial,archive['bytes'])==report['archive'], 'Whole archive changed after member interpretation')
                report.update(asset_notices=notices,asset_license_matches_primary_exactly=True,archive_rehashed_after=True,
                    member_model=wanted,model_byte_identical=True,model_copy_written=False,
                    selected_members_crc_verified=True,all_members_crc_verified=False,
                    purpose=protocol['purpose'],phase='posthash')
                if selected_only:report.update(selected_expanded_bytes=count+total,selected_members_decoded=2,model_member_extracted=False)
    except Exception as error:
        report.update(error_type=type(error).__name__)
    finally:
        # The bounded grace is solely hash/owned-file cleanup, never resumed acquisition.
        signal.alarm(20); post=True
        try:
            if before is not None: rt.require(rt.source(root,code,revision,'acquire_weights',helpers)==before,'Source changed')
            if existing_before is not None: rt.require(rt.identity(model,wanted['bytes'],readonly=False)==existing_before,'Original model changed')
            if previous is not None:
                rt.require(_mhr_failure(rt,root,protocol)==previous,'Original failure/source changed')
                report['original_failure_unchanged']=True
            if previous_inventory is not None:
                rt.require(_mhr_previous_inventory(rt,root,protocol,selected,previous)==previous_inventory,'Original diagnostic/source changed')
                report['original_inventory_unchanged']=True
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
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--mhr-release-only',action='store_true');group.add_argument('--mhr-release-inventory-only',action='store_true')
    group.add_argument('--mhr-release-selected-only',action='store_true')
    arguments=sys.argv[1:] if argv is None else argv
    args=parser.parse_args(arguments)
    if any(arguments.count(flag)>1 for flag in ('--mhr-release-only','--mhr-release-inventory-only','--mhr-release-selected-only')): parser.error('Release mode must be specified exactly once')
    if platform.system() != "Linux":
        raise RuntimeError("Model downloads are restricted to Azure Linux")
    root = Path(os.environ.get("WR_ROOT", "/srv/scenesmith/world-reward"))
    if args.mhr_release_only or args.mhr_release_inventory_only or args.mhr_release_selected_only:
        code=Path(os.environ['WR_CODE']);sys.path.insert(0,str(code/'infra'))
        import mediapipe_cpu_runtime_verify as rt
        if Path(rt.__file__).resolve()!=code/'infra/mediapipe_cpu_runtime_verify.py': raise ValueError('Actual immutable helper required')
        rt.require(Path(__file__).resolve()==code/'infra/acquire_weights.py','Actual immutable acquisition source required')
        options={'selected_only':True} if args.mhr_release_selected_only else {'inventory_only':True} if args.mhr_release_inventory_only else {}
        report=mhr_release(root,code,os.environ['WR_CODE_REVISION'],rt,**options)
        fields=('stage','status','elapsed_seconds') if args.mhr_release_inventory_only else ('stage','status','elapsed_seconds','model_byte_identical')
        print(json.dumps({k:report[k] for k in fields}));return
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
