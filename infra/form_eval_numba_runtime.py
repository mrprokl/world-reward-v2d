"""Build/qualify a NEW CPU-only evaluator child; never alter the base image.

Only two exact public CP311 wheels are acquired on Azure. Installation is
offline, no-deps and isolated in /opt/world-reward/form-eval-numba. Unchanged
published PEN is compared with its own unchanged NumPy fallback and analytic
manufactured controls. This is runtime qualification, not HOI validation.
No data, references, prediction, checkpoint, model or GPU is mounted/read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import zipfile

from mediapipe_cpu_runtime_verify import canonical, identity, require, source, strict

ROOT = Path('/srv/scenesmith/world-reward')
ENTRY = 'run_form_eval_numba_runtime'
BASE = 'sha256:7ebfff18ba3b76dd919485c19115597d7531dfd3233f69461f1dce3f28a6c6d3'
LABEL = 'world_reward.form_eval_numba.owner'
HELPERS = ('infra/form_eval_numba_runtime.py', 'infra/run_form_eval_numba_runtime.sh',
           'infra/mediapipe_cpu_runtime_verify.py')
SITE = '/opt/world-reward/form-eval-numba'
BUDGET = 900
PEN_ATOL_M = 1e-12
OFFICIAL = {
    'mhr_metrics.py': dict(bytes=25647, sha256='73077b65b5c3e0204307feef784317aab8c733d7462b6b8e6f4f397f7b91d2e0'),
    'mhr_submission.py': dict(bytes=24344, sha256='06fbd58d07bae1c583a92598f879771a4805ccea3e8346605975bcf5007b1fa3'),
    'mesh_common.py': dict(bytes=2175, sha256='aaeac13286c839a7d7e271888613875c934e46de094b2ac928ff2ff01c5a1f06'),
}
WHEELS = (
    dict(name='numba', version='0.60.0', bytes=3705018,
         filename='numba-0.60.0-cp311-cp311-manylinux2014_x86_64.manylinux_2_17_x86_64.whl',
         sha256='4142d7ac0210cc86432b818338a2bc368dc773a2f5cf1e32ff7c5b378bd63ee8',
         url='https://files.pythonhosted.org/packages/57/03/2b4245b05b71c0cee667e6a0b51606dfa7f4157c9093d71c6b208385a611/numba-0.60.0-cp311-cp311-manylinux2014_x86_64.manylinux_2_17_x86_64.whl'),
    dict(name='llvmlite', version='0.43.0', bytes=43871781,
         filename='llvmlite-0.43.0-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl',
         sha256='977525a1e5f4059316b183fb4fd34fa858c9eade31f165427a3977c95e3ee749',
         url='https://files.pythonhosted.org/packages/6b/99/5d00a7d671b1ba1751fc9f19d3b36f3300774c6eebe2bcdb5f6191763eb4/llvmlite-0.43.0-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl'),
)


def wheel_pin(row):
    return {key: row[key] for key in ('bytes', 'sha256')}


def target_name(revision):
    require(type(revision) is str and re.fullmatch('[0-9a-f]{40}', revision),
            'Exact immutable runtime producer revision required')
    return 'world-reward/form-eval-numba:'+revision


def seal(path, value):
    path = canonical(path); require(not path.exists(), 'Exclusive runtime output required')
    with path.open('xb') as f:
        f.write((json.dumps(value, sort_keys=True, allow_nan=False)+'\n').encode())
        f.flush(); os.fsync(f.fileno()); os.fchmod(f.fileno(), 0o444)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *_args, **_kwargs):
        raise ValueError('Public runtime acquisition redirect forbidden')


def acquire_public(url, maximum):
    u = urllib.parse.urlsplit(url)
    require(u.scheme == 'https' and u.netloc in {'pypi.org', 'files.pythonhosted.org'}
            and not u.query and not u.fragment and 0 < maximum <= 45 << 20,
            'Only bounded primary public runtime endpoints allowed')
    client = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    try:
        with client.open(url, timeout=120) as f:
            raw = f.read(maximum+1)
    except Exception as error:
        raise ValueError('Primary pinned runtime acquisition failed: '+type(error).__name__) from None
    require(len(raw) <= maximum, 'Public runtime endpoint exceeds bound')
    return raw


def validate_pypi(raw, row):
    metadata = strict(raw); info = metadata['info']
    require(info['name'].lower() == row['name'] and info['version'] == row['version']
            and info['license'] == 'BSD', 'Primary pinned package/version/BSD metadata differs')
    matches = [item for item in metadata['urls'] if item['filename'] == row['filename']]
    require(len(matches) == 1, 'One exact CP311 Linux public wheel required')
    item = matches[0]
    require(item['url'] == row['url'] and item['size'] == row['bytes']
            and item['digests']['sha256'] == row['sha256'] and item['yanked'] is False,
            'Primary PyPI wheel identity contradicts frozen bytes')
    return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(), license='BSD')


def wheel_notices(path):
    notices = {}
    with zipfile.ZipFile(path) as z:
        for item in z.infolist():
            if '.dist-info/' not in item.filename or not re.search(
                    r'(?i)(?:^|/)(?:license|copying|notice)(?:[._-][^/]*)?$', item.filename):
                continue
            require(0 < item.file_size < 1 << 20, 'Bounded packaged license required')
            raw = z.read(item)
            notices[item.filename] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    require(notices, 'Exact wheels must retain their packaged BSD/LLVM license notices')
    return notices


def docker(arguments, *, seconds, log=None):
    env = dict(PATH='/usr/bin:/bin', HOME='/nonexistent', LANG='C.UTF-8',
               DOCKER_HOST='unix://'+str(ROOT/'docker.sock'), DOCKER_BUILDKIT='0')
    command = ['/usr/bin/timeout', '--signal=TERM', '--kill-after=15s', str(seconds)+'s',
               'docker', *arguments]
    try:
        if log:
            with Path(log).open('xb') as f:
                result = subprocess.run(command, env=env, stdout=f, stderr=subprocess.STDOUT,
                                        timeout=seconds+20, check=False)
                f.flush(); os.fchmod(f.fileno(), 0o444)
            require(result.returncode == 0, 'Bounded runtime Docker command failed; Azure-only log retained')
            return ''
        result = subprocess.run(command, env=env, capture_output=True, text=True,
                                timeout=seconds+20, check=False)
    except (OSError, subprocess.TimeoutExpired):
        raise ValueError('Bounded runtime Docker control failed or timed out') from None
    require(result.returncode == 0, 'Runtime Docker command failed: '+arguments[0])
    return result.stdout


def child_dockerfile():
    names = ' '.join('/wheels/'+row['filename'] for row in WHEELS)
    return (f'FROM {BASE}\nCOPY wheels /wheels\n'
        'RUN /opt/conda/bin/python -c "import sys,numpy,torch; '
        "assert sys.version_info[:2]==(3,11); assert numpy.__version__=='1.26.3'; "
        "assert torch.__version__=='2.5.1+cu124'; assert not torch.cuda.is_initialized()"+'"\n'
        f'RUN /opt/conda/bin/python -m pip install --no-index --no-deps --disable-pip-version-check '
        f'--no-cache-dir --target {SITE} {names} && rm -rf /wheels\n'
        f'ENV PYTHONPATH={SITE}\n')


def procedural_cases():
    """Only analytic own geometry; edge/on-surface cases remain explicit."""
    import numpy as np
    cube = np.array([[-1.,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                     [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]])
    faces = np.array([[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],
                      [1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]])
    points = np.array([[0.,0,0],[.75,.1,-.2],[.1,.2,.9],[1.1,0,0],
                       [0,-1.2,0],[0,0,1.3],[1,0,0],[1,1,1]])
    expected = np.maximum(0., 1-np.abs(points).max(axis=1))
    tetra = np.array([[0.,0,0],[1,0,0],[0,1,0],[0,0,1]])
    tetra_faces = np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]])
    tetra_points = np.array([[.1,.1,.1],[.2,.2,.2],[.2,.2,.5],
                             [.5,.5,.5],[-.1,.1,.1],[0,0,0]])
    tetra_expected = np.array([.1,.2,.1/np.sqrt(3),0.,0.,0.])
    # Larger batches exercise the public parallel kernels and bbox denominator.
    rng = np.random.default_rng(20481010)
    batch = rng.uniform(-1.4,1.4,(2048,3)); batch_expected = np.maximum(0.,1-np.abs(batch).max(axis=1))
    return [('cube',points,cube,faces,expected),
            ('tetra',tetra_points,tetra,tetra_faces,tetra_expected),
            ('cube_parallel',batch,cube,faces,batch_expected),
            ('cube_reversed',points,cube,faces[:,::-1],expected),
            ('cube_degenerate_retained',points,cube,np.r_[faces,[[0,0,0]]],expected)]


def depth_parity(module):
    """Unchanged compiled PEN vs unchanged NumPy fallback and analytic truth."""
    import numpy as np
    compiled = module._PENETRATION_KERNELS
    require(compiled is not None, 'Actual unchanged public Numba kernels must be active')
    results = []
    try:
        for name, points, vertices, faces, expected in procedural_cases():
            module._PENETRATION_KERNELS = compiled
            start = time.monotonic(); accelerated = module.penetration_depth(points, vertices, faces)
            elapsed = time.monotonic()-start
            module._PENETRATION_KERNELS = None
            original = module.penetration_depth(points, vertices, faces)
            error = float(np.abs(accelerated-original).max())
            analytic = float(np.abs(accelerated-expected).max())
            require(error <= PEN_ATOL_M and analytic <= PEN_ATOL_M
                    and np.isfinite(accelerated).all() and (accelerated >= 0).all(),
                    'Unchanged public compiled/NumPy/analytic penetration parity failed')
            results.append(dict(case=name,points=len(points),faces=len(faces),
                maximum_numpy_difference_m=error,maximum_analytic_difference_m=analytic,
                byte_exact_numpy=np.array_equal(accelerated,original),elapsed_seconds=elapsed))
    finally:
        module._PENETRATION_KERNELS = compiled
    return results


def episode_parity(module):
    """Own moving Sim3 scene; unchanged public episode PEN and metric operators."""
    import numpy as np
    _, points, vertices, faces, _ = procedural_cases()[0]
    hands = np.resize(points,(96,512,3)).copy()
    angles = np.linspace(0,.4,96); rotation = np.zeros((96,3,3))
    rotation[:,0,0]=rotation[:,1,1]=np.cos(angles)
    rotation[:,1,0]=np.sin(angles);rotation[:,0,1]=-np.sin(angles);rotation[:,2,2]=1
    translation = np.c_[np.linspace(0,.4,96),np.zeros(96),np.linspace(2,2.3,96)]
    scale = .3; align_scale = 1.2
    posed = scale*np.einsum('tij,tpj->tpi',rotation,hands)+translation[:,None]
    body = np.tile(np.array([[-.3,0,2.],[.3,0,2.],[0,.6,2.],[0,0,2.6]])[None],(96,1,1))
    joints = np.resize(body,(96,127,3)).copy()
    pred = dict(mhr_vertices=body, mhr_joints=joints, human_surface_points=body,
        object_surface_points=body.copy(), object_translation=translation)
    target = {k:align_scale*v+np.array([1.,2.,3.]) for k,v in pred.items()}
    scene = dict(hands=posed,mesh_vertices=vertices,mesh_faces=faces,
                 rotation=rotation,translation=translation,scale=scale)
    compiled = module._PENETRATION_KERNELS
    try:
        module._PENETRATION_KERNELS = compiled
        first = module.episode_penetration(pred,scene,target)
        module._PENETRATION_KERNELS = None
        fallback = module.episode_penetration(pred,scene,target)
    finally:
        module._PENETRATION_KERNELS = compiled
    require(abs(first-fallback) <= 100*scale*align_scale*PEN_ATOL_M,
            'Full96 unchanged episode penetration parity failed')
    metrics = module.episode_metrics(pred,target,np.arange(96,dtype=np.int64),
                                    tuple((1,2,18,35,3,19,36,4,20,37,8,24,110,74,38,113,75,39,76,40,78,42)))
    require(set(metrics)=={'cd_h_cm','cd_o_cm','cd_c_cm','acc_h_cm','acc_o_cm'}
            and all(np.isfinite(x) and 0 <= x <= 1e-10 for x in metrics.values()),
            'Unchanged full96 CD/ACC operators failed exact shared Sim3 control')
    return dict(frames=96,hand_points=512,compiled_pen_cm=first,numpy_pen_cm=fallback,
                maximum_difference_cm=abs(first-fallback),CD_ACC_metrics=metrics,
                per_frame_alignment=False,real_data_read=False)


def qualify(code, revision, out):
    """Runs only inside the child, with pinned official code and no data mounts."""
    import importlib
    import platform
    import types
    import numpy as np
    import torch
    import numba
    import llvmlite
    require(sys.version_info[:2] == (3,11) and platform.machine()=='x86_64'
            and np.__version__=='1.26.3' and torch.__version__=='2.5.1+cu124'
            and numba.__version__=='0.60.0' and llvmlite.__version__=='0.43.0'
            and not torch.cuda.is_initialized(), 'Literal CPU child/runtime versions required')
    for package in (numba,llvmlite):
        require(Path(package.__file__).is_relative_to(SITE), 'Isolated child dependency origin required')
    numba.set_num_threads(4)
    kit = ROOT/'vendor/v2d_submission_kit/v2dlb'
    for name,pin in OFFICIAL.items(): require(identity(kit/name)==pin, 'Unchanged official operator source required')
    require('v2dlb' not in sys.modules, 'No foreign metric package initializer allowed')
    namespace=types.ModuleType('v2dlb');namespace.__path__=[str(kit)];sys.modules['v2dlb']=namespace
    module=importlib.import_module('v2dlb.mhr_submission')
    require(Path(module.__file__)==kit/'mhr_submission.py', 'Actual unchanged public operator module required')
    started=time.monotonic();cases=depth_parity(module);episode=episode_parity(module)
    require(not torch.cuda.is_initialized() and numba.get_num_threads()==4,
            'No CUDA execution and exactly four public compiled-kernel threads required')
    report=dict(schema='world_reward.form_eval_numba_qualification.v1',status='pass',
        producer_revision=revision,versions=dict(python='3.11',numpy=np.__version__,torch=torch.__version__,
            numba=numba.__version__,llvmlite=llvmlite.__version__),public_operator_pins=OFFICIAL,
        parity_tolerance_m=PEN_ATOL_M,parity_cases=cases,episode_control=episode,
        numba_threads=4,compiled_public_kernels_active=True,public_operator_source_modified=False,
        base_packages_modified=False,GPU_requested=False,CUDA_initialized=False,
        model_loaded=False,dataset_read=False,reference_values_read=False,predictions_read=False,
        elapsed_seconds=time.monotonic()-started)
    seal(out/'qualification.json',report)
    return report


def build(code, revision):
    started=time.monotonic();binding=source(ROOT,code,revision,ENTRY,HELPERS)
    out=ROOT/'results'/('form-eval-numba-runtime-'+revision)
    require(not out.exists(), 'Fresh immutable runtime result namespace required');out.mkdir(mode=0o755)
    context=out/'context';context.mkdir(mode=0o700);(context/'wheels').mkdir(mode=0o700)
    report=dict(schema='world_reward.form_eval_numba_runtime.v1',status='fail',producer_revision=revision,
        source_before=binding,base_image_id=BASE,target_image=target_name(revision),
        downloaded_wheels=[],GPU_requested=False,models_loaded=False,dataset_read=False,
        reference_values_read=False,base_image_modified=False,public_operator_source_modified=False)
    try:
        # An existing tag belongs to a previous producer attempt. Never replace
        # or silently reuse it before diagnosing that actual runtime receipt.
        require(not docker(['image','ls','-q','--no-trunc',target_name(revision)],seconds=20).strip(),
                'Never overwrite/reuse an existing runtime child tag')
        base_before=strict(docker(['image','inspect',BASE],seconds=20))[0]
        require(base_before['Id']==BASE, 'Existing exact base image required; no pull/retag')
        for name,pin in OFFICIAL.items():
            require(identity(ROOT/'vendor/v2d_submission_kit/v2dlb'/name)==pin,
                    'Pin tiny public operators on CPU02 before runtime build')
        report['base_rootfs_layers']=base_before['RootFS']['Layers']
        for row in WHEELS:
            metadata=acquire_public(f'https://pypi.org/pypi/{row["name"]}/{row["version"]}/json',2 << 20)
            publisher=validate_pypi(metadata,row)
            raw=acquire_public(row['url'],row['bytes'])
            require(len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256'],
                    'Acquired exact public wheel bytes differ')
            path=context/'wheels'/row['filename']
            with path.open('xb') as f:f.write(raw);os.fchmod(f.fileno(),0o400)
            notices=wheel_notices(path)
            report['downloaded_wheels'].append(dict(row,publisher_metadata=publisher,packaged_notices=notices))
        (context/'Dockerfile').write_text(child_dockerfile())
        docker(['build','--network=none','--pull=false','--label',LABEL+'='+revision,
                '-t',target_name(revision),str(context)],seconds=300,log=out/'build.log')
        child=strict(docker(['image','inspect',target_name(revision)],seconds=20))[0]
        require(child['Id']!=BASE and child['Config']['Labels'][LABEL]==revision
                and child['RootFS']['Layers'][:len(report['base_rootfs_layers'])]==report['base_rootfs_layers'],
                'New isolated child must retain original complete base layer prefix')
        require(len(docker(['image','ls','-q','--no-trunc',target_name(revision)],seconds=20).splitlines())==1,
                'Exactly one actual child image identity required')
        report['qualified_image_id']=child['Id']; report['child_rootfs_layers']=child['RootFS']['Layers']
        name='wr-form-eval-numba-qualify-'+revision;cid=out/'.container.cid'
        try:
            docker(['run','--rm','--name',name,'--cidfile',str(cid),'--label',LABEL+'='+revision,
                '--network=none','--read-only','--user','0:0','--cap-drop','ALL',
                '--security-opt','no-new-privileges','--cpus','4','--memory','4g',
                '--tmpfs','/tmp:rw,nosuid,size=256m',
                '--mount',f'type=bind,src={code.parent},dst={code.parent},readonly',
                '--mount',f'type=bind,src={ROOT}/vendor/v2d_submission_kit/v2dlb,dst={ROOT}/vendor/v2d_submission_kit/v2dlb,readonly',
                '--mount',f'type=bind,src={out},dst={out}',
                '--entrypoint','/usr/bin/env',child['Id'],'-i','PATH=/opt/conda/bin:/usr/bin:/bin',
                'HOME=/tmp','PYTHONDONTWRITEBYTECODE=1','CUDA_VISIBLE_DEVICES=-1',
                'OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1','NUMBA_NUM_THREADS=4',
                f'PYTHONPATH={SITE}:{code}/src:{code}/infra',f'WR_ROOT={ROOT}',
                f'WR_CODE={code}',f'WR_CODE_REVISION={revision}',
                '/opt/conda/bin/python','-B',str(code/'infra/form_eval_numba_runtime.py'),'qualify'],
                seconds=300,log=out/'qualification.log')
        finally:
            if cid.exists():
                container=cid.read_text().strip()
                require(re.fullmatch('[0-9a-f]{64}',container), 'Actual bounded qualification container ID required')
                active=docker(['ps','-aq','--no-trunc','--filter','id='+container],seconds=20).strip()
                if active:
                    info=strict(docker(['inspect',container],seconds=20))[0]
                    require(info['Image']==child['Id'] and info['Name']=='/'+name
                            and info['Config']['Labels'][LABEL]==revision,'Only own actual runtime container may be removed')
                    docker(['rm','-f',container],seconds=20)
                require(not docker(['ps','-aq','--no-trunc','--filter','id='+container],seconds=20).strip(),
                        'Qualification container absence required');os.chmod(cid,0o444)
            else:
                require(not docker(['ps','-aq','--no-trunc','--filter','name=^/'+name+'$'],seconds=20).strip(),
                        'Qualification failure without CID must not leave an owned container')
        qualification=strict((out/'qualification.json').read_bytes())
        require(qualification['status']=='pass' and qualification['producer_revision']==revision
                and qualification['compiled_public_kernels_active'] is True, 'Actual complete operator parity PASS required')
        report['qualification']=identity(out/'qualification.json')
        require(strict(docker(['image','inspect',BASE],seconds=20))[0]==base_before,
                'Original existing base image metadata must remain exactly unchanged')
        report['source_after']=source(ROOT,code,revision,ENTRY,HELPERS)
        require(report['source_after']==binding, 'Immutable runtime producer source changed')
        report['status']='pass';report['decision']='CPU_RUNTIME_QUALIFIED_NOT_HOI_METRIC_GAIN'
    except Exception as error:
        report['error_type']=type(error).__name__
        if isinstance(error,ValueError):report['error_context']=str(error)[:300]
    finally:
        shutil.rmtree(context)
        report['disposable_wheels_and_build_context_removed']=True
        report['elapsed_seconds']=time.monotonic()-started
        seal(out/'report.json',report)
    print(json.dumps({k:report[k] for k in ('status','elapsed_seconds')},sort_keys=True),flush=True)
    return 0 if report['status']=='pass' else 1


def main():
    parser=argparse.ArgumentParser(allow_abbrev=False);parser.add_argument('mode',choices=('build','qualify'))
    args=parser.parse_args();code=canonical(os.environ['WR_CODE']);revision=os.environ['WR_CODE_REVISION']
    require(sys.platform=='linux' and os.geteuid()==0 and os.environ.get('WR_ROOT')==str(ROOT)
            and code==ROOT/'jobs'/revision/ENTRY/'code', 'Azure immutable CPU runtime namespace required')
    if args.mode=='build':
        require(os.uname().nodename=='world-reward-ncc-h100-02','Only existing CPU02 image/runtime may be built')
    def interrupted(*_):raise TimeoutError('Inclusive runtime build budget or termination')
    signal.signal(signal.SIGALRM,interrupted);signal.signal(signal.SIGTERM,interrupted)
    signal.alarm(BUDGET)
    if args.mode=='qualify':
        qualify(code,revision,ROOT/'results'/('form-eval-numba-runtime-'+revision));return 0
    return build(code,revision)


if __name__=='__main__':raise SystemExit(main())
