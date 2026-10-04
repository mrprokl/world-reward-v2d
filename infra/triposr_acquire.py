"""Azure-only immutable source/checkpoint acquisition; no model execution.

Closed-solid certification is our optional backend policy, not a Track 1 rule.
MIT declarations below do not certify all dependencies or training non-overlap.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import pwd
import re
import signal
import stat
import sys
import time
import urllib.parse
import urllib.request

ROOT = Path('/srv/scenesmith/world-reward')
JOB = 'run_triposr_acquire'
SOURCE = 'vendor/research/triposr_raw_v1'
WEIGHTS = 'weights/triposr_raw_v1'
REPORT = 'results/triposr-acquisition-v1.json'
BUDGET, OUTER, MAXIMUM, BLOCK = 180, 183, 2_000_000_000, 1 << 20
TRIPOSR = '107cefdc244c39106fa830359024f6a2f1c78871'
MODEL = '5b521936b01fbe1890f6f9baed0254ab6351c04a'
DINO = 'f205d5d8e640a89a2b8ef0369670dfc37cc07fc2'
MCUBES = '879926d0ef58e6ce0ac2630fdecb5e53af7ed3ff'
HELPERS = ('infra/triposr_acquire.py', 'infra/run_triposr_acquire.sh',
           'infra/mediapipe_cpu_runtime_verify.py', 'infra/mediapipe_hands_acquire.py')

# All small-text hashes were measured independently at the immutable primary
# revisions. The checkpoint SHA/size are the publisher's Git LFS declaration.
TRIPOSR_FILES = (
 ('tsr/system.py',6608,'7aec93e421394ad7c746b6d63a0189532b57b26af9a199fc75ac3bb7c993c94f'),
 ('tsr/utils.py',15172,'d9d1a645fef7ea489615bff3471fd01a4941a2c0e138c5dff5daae619563279b'),
 ('tsr/models/isosurface.py',1926,'94c9a2ec27f8fe3c2fb57558ef28be253eb77074d2c171d49a41cf73d95d6f5e'),
 ('tsr/models/tokenizers/image.py',2116,'f6da23124dfabe6a1f8d0d27693d0cece6574de68be15487a77781b6a9d73195'),
 ('tsr/models/tokenizers/triplane.py',1237,'44811fbd53013d6f49e229ecfd0717f89dd8ebeb71608aeb1f8e5ed1d5e8e378'),
 ('tsr/models/transformer/transformer_1d.py',9325,'fd83a71d59d04ef0a45dbadcc63f9d56d354ebc011a40f5589eb1786a1dd15f4'),
 ('tsr/models/transformer/basic_transformer_block.py',13202,'ced8ce4c5c5bbc13368130bc15dee1717cc6bc7f35699f64d03aeec185b1198e'),
 ('tsr/models/transformer/attention.py',26341,'347c1a707257cb4eb664ebe9ecd16db7b6006131e74925c049245d2c0101ba7e'),
 ('tsr/models/network_utils.py',3473,'b3d48e86be9ca3f3abc1bfd774f640fd350e4365843d804e2a3145e7692e1419'),
 ('tsr/models/nerf_renderer.py',5673,'afe8c12c964078b2c6bb3922d2e30969aa59f7edbacb8cb73d2c3df839dc785f'),
 ('LICENSE',1080,'ade0a66629bdd7e01e46b3296b3851cff0fd27989bca53da470ad6e96ed620fb'),
 ('README.md',5419,'ed40f078c5e451c58eae716187e0128f0ce8575e467aab2440b4ed9c75e490b9'),
)
MCUBES_FILES = (
 ('CMakeLists.txt',2854,'9ed71e9cc1e29ad7d7cc84ed55c23f955731af9efdf262597d92727d0c04ac86'),
 ('cxx/CMakeLists.txt',782,'ded4c1aae611256c38e005bf8ac7c6314b47ddf15b107ba3935839a6df7fb597'),
 ('_torch_dep.py',1598,'7402d887f736d26fb0d0375b56315f0f0c1e110b478789e48e4d8193c0d01276'),
 ('pyproject.toml',1920,'b9ba2149289b5018f17f146fa4b8aa070f524f4da6261d0402721f2acf208641'),
 ('torchmcubes/__init__.py',1599,'176adbf6fdb9ea014504e50d76c2c5c0ce0745626e285ea24de6a185ff32dac1'),
 ('cxx/macros.h',1760,'0e76582a5f8a99ae940ee54434ad8ac6dd97068f38ee3279ee1490a462d8bb5d'),
 ('cxx/mcubes.cpp',1005,'4d6e922334e0c5064a00db6835449d25717ca0b2bdb9c9a6885355c51e7780cb'),
 ('cxx/mcubes_cpu.cpp',34423,'184b56832f4ba8a74106721907d255377f17453f80fd3471de7f885d68e5ce89'),
 ('cxx/grid_interp_cpu.cpp',2232,'882bb545d7ce2002b7a5638bd1fce511af5be0af17d4f1303d70d47fccc58648'),
 ('cxx/cuda_utils.h',855,'6b87f9ddf9cb08e398eff52fbd214dea16ad416e25e391b65d460bba65b6ab80'),
 ('cxx/helper_math.h',36774,'5bd3579f5be3201c385e537e75d297a322145f7f0e69a6702e48c7bc893ab187'),
 ('cxx/mcubes_cuda.cu',35786,'066c3079fe2b3444a0d7c5a5901b970c18e35ee6743b08badd6f9a03b034e0cc'),
 ('cxx/grid_interp_cuda.cu',3807,'2655ec2ac26a1604f84897d4655d8a752eec8c326809f234986bcfc2d3fcc2e2'),
 ('LICENSE',16726,'393eaf2b92b3d49bea139e39b6fd205b6e99c499999986aec5554586a830dfdd'),
 ('README.md',3084,'f87ec84bba8add5b1d5164fcfb50fcffb17ce12a6a1c548dadebf30c2313da08'),
)


def assets():
    rows = [dict(file=SOURCE+'/triposr/'+p, bytes=n, sha256=h,
                 url='https://raw.githubusercontent.com/VAST-AI-Research/TripoSR/'+TRIPOSR+'/'+p)
            for p,n,h in TRIPOSR_FILES]
    rows += [dict(file=SOURCE+'/torchmcubes/'+p, bytes=n, sha256=h,
                  url='https://raw.githubusercontent.com/tatsy/torchmcubes/'+MCUBES+'/'+p)
             for p,n,h in MCUBES_FILES]
    for file,repo,revision,name,n,h in (
      (SOURCE+'/publisher/README.md','stabilityai/TripoSR',MODEL,'README.md',2488,'5c12ef5b850c6ee2aa7454b1a264f6f8dfa95164384dbb1ab374acd917aa558b'),
      (WEIGHTS+'/config.yaml','stabilityai/TripoSR',MODEL,'config.yaml',987,'74ca708ce086bf68e97709ea6b3d91f14717921c04691e84043f0eb8fcc68e62'),
      (SOURCE+'/dino/config.json','facebook/dino-vitb16',DINO,'config.json',454,'b87c0270b97db085fd82cf114a761fd0f62ae7914fbd407c752a2260646b689c'),
      (WEIGHTS+'/model.ckpt','stabilityai/TripoSR',MODEL,'model.ckpt',1677246742,'429e2c6b22a0923967459de24d67f05962b235f79cde6b032aa7ed2ffcd970ee')):
        rows.append(dict(file=file, bytes=n, sha256=h,
                         url='https://huggingface.co/'+repo+'/resolve/'+revision+'/'+name))
    return tuple(rows)


ASSETS = assets()


def helpers(code):
    loaded = []
    for name in HELPERS[2:]:
        path = code/name
        s=path.lstat()
        if (not code.is_absolute() or code.resolve()!=code or path.resolve()!=path
            or any(p.is_symlink() for p in (path,*path.parents)) or not stat.S_ISREG(s.st_mode)
            or s.st_nlink!=1 or s.st_mode&0o222 or not 0<s.st_size<=2_000_000):
            raise ValueError('Original readonly source helper required')
        spec = importlib.util.spec_from_file_location('wr_triposr_'+path.stem, path)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        if Path(module.__file__)!=path:raise ValueError('Imported source helper origin differs')
        loaded.append(module)
    return tuple(loaded)


def binding(rt, root, code, revision):
    rt.require(Path(__file__).resolve() == code/HELPERS[0], 'Actual acquisition source required')
    return rt.source(root, code, revision, JOB, HELPERS)


def endpoint(url, *, initial=False):
    p = urllib.parse.urlsplit(url); host = p.hostname or ''
    valid = (p.scheme == 'https' and not p.username and not p.password and p.port in (None,443)
             and not p.fragment and (host in ('huggingface.co','raw.githubusercontent.com')
             or host.endswith(('.hf.co','.huggingface.co'))))
    if not valid or (initial and url not in {r['url'] for r in ASSETS}):
        raise ValueError('Unlisted public HTTPS endpoint')
    return url


class PublicRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        endpoint(newurl)
        if any(k.lower() in ('authorization','cookie') for k in req.headers):
            raise ValueError('Authenticated acquisition is forbidden')
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def fetch(rt, mp, root, row, opener, deadline, owned):
    rt.require(time.monotonic() < deadline, 'Inclusive acquisition deadline')
    target = rt.canonical(root/row['file']); part = target.with_name(target.name+'.part')
    request = urllib.request.Request(endpoint(row['url'], initial=True), headers={'Accept-Encoding':'identity'})
    with opener.open(request, timeout=min(20, max(.001, deadline-time.monotonic()))) as response:
        endpoint(response.geturl())
        rt.require(response.status == 200 and response.headers.get('Content-Encoding','identity').lower() == 'identity'
                   and response.headers.get_content_type().lower() != 'text/html', 'Publisher transport differs')
        if urllib.parse.urlsplit(row['url']).hostname == 'raw.githubusercontent.com':
            rt.require(response.geturl() == row['url'], 'Source redirect rejected')
        length = response.headers.get('Content-Length')
        rt.require(length is None or re.fullmatch('[0-9]+',length) and int(length) == row['bytes'], 'Publisher length differs')
        digest = hashlib.sha256(); count = 0
        with part.open('xb') as stream:
            os.fchmod(stream.fileno(),0o600); s = part.lstat(); owned.append((part,(s.st_dev,s.st_ino,s.st_uid)))
            while True:
                rt.require(time.monotonic() < deadline, 'Inclusive acquisition deadline')
                block = response.read(min(BLOCK,row['bytes']-count+1))
                rt.require(len(block) <= BLOCK and count+len(block) <= row['bytes'], 'Publisher body overflow')
                if not block: break
                if count == 0:
                    rt.require(not block.lstrip().lower().startswith((b'<html',b'<!doctype html'))
                               and not block.startswith(b'version https://git-lfs.github.com/spec/'), 'HTML/LFS pointer rejected')
                stream.write(block); digest.update(block); count += len(block)
            rt.require(count == row['bytes'] and digest.hexdigest() == row['sha256'], 'Publisher byte/SHA differs')
            stream.flush(); os.fsync(stream.fileno()); os.fchmod(stream.fileno(),0o444)
        rt.require(time.monotonic() < deadline, 'Inclusive acquisition deadline')
        mp.publish(part,target)
    return dict(file=row['file'],bytes=count,sha256=digest.hexdigest(),
                hash_basis='publisher_git_lfs_sha256' if target.name == 'model.ckpt' else 'independent_primary_text_sha256')


def declarations(rt, root):
    rt.require(b'MIT License' in (root/SOURCE/'triposr/LICENSE').read_bytes()
               and b'license: mit' in (root/SOURCE/'publisher/README.md').read_bytes()
               and b'Mozilla Public License Version 2.0' in (root/SOURCE/'torchmcubes/LICENSE').read_bytes(),
               'Pinned primary license declaration differs')
    config = rt.strict((root/SOURCE/'dino/config.json').read_bytes())
    rt.require(config.get('architectures') == ['ViTModel'] and config.get('hidden_size') == 768
               and config.get('patch_size') == 16, 'Pinned DINO configuration differs')


def acquire(root, code, revision, *, opener=None, namespace_lease=None):
    started = time.monotonic(); rt, mp = helpers(code); root = rt.canonical(root); code = rt.canonical(code)
    before = binding(rt,root,code,revision)
    rt.require(sum(r['bytes'] for r in ASSETS) <= MAXIMUM and len({r['file'] for r in ASSETS}) == len(ASSETS), 'Frozen total/inventory differs')
    for row in ASSETS:
        p = PurePosixPath(row['file'])
        rt.require(not p.is_absolute() and '..' not in p.parts and p.as_posix() == row['file']
                   and (p.is_relative_to(SOURCE) or p.is_relative_to(WEIGHTS)) and type(row['bytes']) is int
                   and 0 < row['bytes'] <= MAXIMUM and re.fullmatch('[0-9a-f]{64}',row['sha256']), 'Frozen asset contract differs')
    folders = [rt.canonical(root/n) for n in (SOURCE,WEIGHTS)]
    out = rt.canonical(root/REPORT)
    rt.require(out.parent.is_dir() and not out.exists() and (namespace_lease is not None or all(not p.exists() for p in folders)), 'Fresh acquisition only; no resume')
    owned, created, records, failure = [], [], [], None
    report = dict(schema='world_reward.triposr_acquisition.v1',stage='triposr_source_model_acquisition',status='fail',
      producer_revision=revision,source_binding=before,artifacts=records,budget_seconds=BUDGET,
      maximum_total_bytes=MAXIMUM,budget_scope='download_checks_public_sealing_source_artifact_posthash',
      receipt_publication_outer_seconds=OUTER,upstream_license_declarations={'TripoSR_source':'MIT','TripoSR_checkpoint':'MIT','torchmcubes':'MPL-2.0'},
      models_loaded=False,packages_installed=False,gpu_used=False,dataset_read=False,private_values_read=False,
      quality_verified=False,license_eligibility_verified=False,training_overlap_verified=False,challenge_overlap_verified=False,
      official_closed_solid_requirement=False,source_rehashed_after=False,artifacts_rehashed_after=False,owned_partials_removed=False)
    try:
        if namespace_lease is not None: created = mp.validate_namespace_lease(namespace_lease,folders,before['closure_sha256'])
        else:
            for p in folders:
                p.mkdir(mode=0o700); p.chmod(0o700); s=p.lstat(); created.append((p,(s.st_dev,s.st_ino)))
        for row in ASSETS: (root/row['file']).parent.mkdir(mode=0o700,parents=True,exist_ok=True)
        opener = opener or urllib.request.build_opener(urllib.request.ProxyHandler({}),PublicRedirect())
        for row in ASSETS:
            if row['file'] == WEIGHTS+'/model.ckpt': declarations(rt,root)
            records.append(fetch(rt,mp,root,row,opener,started+BUDGET,owned))
        report['status']='pass'
    except Exception as exc:
        failure=exc; report['error_type']=type(exc).__name__  # URLs/errors may contain signed CDN query values.
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        try:
            for p,inode in owned:
                if p.exists() or p.is_symlink():
                    s=p.lstat(); rt.require(not p.is_symlink() and (s.st_dev,s.st_ino,s.st_uid)==inode and s.st_nlink==1, 'Owned partial replaced')
                    p.unlink()
            report['owned_partials_removed']=all(not p.exists() and not p.is_symlink() for p,_ in owned)
            expected_files={root/r['file'] for r in records}
            expected_dirs={p for row in ASSETS for p in (root/row['file']).parents
                           if any(p==folder or p.is_relative_to(folder) for folder in folders)}
            for p,inode in created:
                s=p.lstat(); rt.require(rt.canonical(p)==p and (s.st_dev,s.st_ino)==inode and s.st_uid==os.getuid(), 'Owned namespace replaced')
                entries=sorted(p.rglob('*'),reverse=True)
                for q in entries:
                    rt.canonical(q); t=q.lstat()
                    rt.require(t.st_uid==os.getuid() and (stat.S_ISDIR(t.st_mode) or stat.S_ISREG(t.st_mode) and t.st_nlink==1), 'Foreign namespace entry')
                    rt.require(q in (expected_dirs if q.is_dir() else expected_files),'Unexpected namespace entry')
                for q in entries:
                    q.chmod(0o555 if q.is_dir() else 0o444)
                p.chmod(0o555)
            report['source_rehashed_after']=binding(rt,root,code,revision)==before
            report['artifacts_rehashed_after']=all(rt.identity(root/r['file'],MAXIMUM)=={k:r[k] for k in ('bytes','sha256')} for r in records)
        except Exception as exc:
            failure=failure or exc; report['post_error_type']=type(exc).__name__
        report['elapsed_seconds']=time.monotonic()-started
        if failure or not all(report[k] for k in ('source_rehashed_after','artifacts_rehashed_after','owned_partials_removed')) or report['elapsed_seconds']>BUDGET:
            report['status']='fail'
        rt.write(out,(json.dumps(report,sort_keys=True,allow_nan=False)+'\n').encode(),0o444)
    if report['status']!='pass': raise ValueError('Acquisition failed; immutable tiny receipt retained') from None
    return report


def cancelled(*_):
    raise TimeoutError('Acquisition deadline or cancellation')


def main():
    if (len(sys.argv)!=1 or sys.platform!='linux' or os.geteuid()!=1000
        or pwd.getpwnam('scenesmith').pw_uid!=1000 or os.environ.get('WR_ROOT')!=str(ROOT)):
        raise ValueError('Actual Azure unprivileged acquisition required')
    code=Path(os.environ['WR_CODE']); revision=os.environ['WR_CODE_REVISION']
    signal.signal(signal.SIGALRM,cancelled); signal.signal(signal.SIGTERM,cancelled); signal.signal(signal.SIGINT,cancelled)
    signal.setitimer(signal.ITIMER_REAL,BUDGET)
    try:
        rt,_=helpers(code)
        lease=rt.strict(os.environ['WR_NAMESPACE_LEASE']) if 'WR_NAMESPACE_LEASE' in os.environ else None
        result=acquire(ROOT,code,revision,namespace_lease=lease)
        print(json.dumps({k:result[k] for k in ('stage','status','elapsed_seconds')}))
    finally: signal.setitimer(signal.ITIMER_REAL,0)


if __name__=='__main__':
    try: main()
    except Exception as exc:
        print(json.dumps(dict(stage='triposr_source_model_acquisition',status='fail',error_type=type(exc).__name__)))
        raise SystemExit(1) from None
