#!/usr/bin/env bash
set -euo pipefail
(( $# == 0 )) || exit 2
# Acquisition and compilation share one hard deadline; no apt or dependency fallback.
timeout --signal=TERM --kill-after=5s 600s python3 - <<'PYBUILD'
import hashlib,json,os,pathlib,shutil,subprocess,tarfile,tempfile,time,urllib.request
start=time.monotonic(); root=pathlib.Path('/opt/world-reward/guarded-qem')
source=root/'source'; source.mkdir(exist_ok=False)
igl='40e7900ccbd767f1f360e0eb10f0f1a6432e0993'
eigen='3147391d946bb4b6c68edd901f2add6ac1f31f8c'
def remaining():
    value=600-(time.monotonic()-start)
    if value <= 0: raise TimeoutError('Whole source/compile deadline reached')
    return value
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
archive_hashes={}
def acquire(name,url,selected):
    destination=source/name; destination.mkdir()
    with tempfile.TemporaryDirectory() as temporary:
        archive=pathlib.Path(temporary)/'source.tar.gz'
        digest=hashlib.sha256(); size=0
        with urllib.request.urlopen(url,timeout=min(45,remaining())) as response, archive.open('xb') as handle:
            while True:
                remaining(); block=response.read(1024*1024)
                if not block: break
                size += len(block)
                if size > 64*1024*1024: raise ValueError('Unexpected source archive size')
                digest.update(block); handle.write(block)
        archive_hashes[name]={'url':url,'sha256':digest.hexdigest(),'bytes':size}
        with tarfile.open(archive,'r:gz') as tar:
            prefixes=set(); paths=set()
            for member in tar:
                remaining(); components=pathlib.PurePosixPath(member.name).parts
                if not components or any(p in ('..','') for p in components) or member.name.startswith('/'):
                    raise ValueError('Unsafe source archive path')
                prefixes.add(components[0]); relative=pathlib.PurePosixPath(*components[1:])
                if not selected(relative.as_posix()): continue
                if member.isdir(): continue
                if not member.isfile() or relative in paths: raise ValueError('Nonregular/duplicate source file')
                paths.add(relative); output=destination.joinpath(*relative.parts)
                output.parent.mkdir(parents=True,exist_ok=True)
                with tar.extractfile(member) as incoming, output.open('xb') as outgoing: shutil.copyfileobj(incoming,outgoing)
            if len(prefixes) != 1: raise ValueError('Unexpected source archive roots')
    return destination
igldir=acquire('libigl',f'https://codeload.github.com/libigl/libigl/tar.gz/{igl}',lambda p:(p.startswith('include/igl/') and not p.startswith('include/igl/copyleft/')) or p.startswith('LICENSE') or p.startswith('COPYING'))
eigendir=acquire('eigen',f'https://gitlab.com/libeigen/eigen/-/archive/{eigen}/eigen-{eigen}.tar.gz',lambda p:p.startswith('Eigen/') or p.startswith('COPYING'))
expected={
 'libigl/include/igl/qslim.h':'188941d1d1d608dd59dcbd1f5c0c69f9faaecdbbd1ff6c48658af3d4110b9d31',
 'libigl/include/igl/qslim.cpp':'96b5c7c009c9b9539b200d77158cd90093d8b7d0803dcd4b2f357b77bb5bfa40',
 'libigl/include/igl/intersection_blocking_collapse_edge_callbacks.cpp':'e598a18cd62db1f0995fc310cbebc6ca51d716c7e61d55d5a3435f20db547b47',
 'libigl/include/igl/collapse_edge_would_create_intersections.cpp':'b70fe52dfaf03c5544cd3660ee626f6fbfc91f1109d7d2137a2293b1a6466c3a',
 'libigl/LICENSE.MPL2':'fab3dd6bdab226f1c08630b1dd917e11fcb4ec5e1e020e2c16f83a0a13863e85',
 'eigen/Eigen/src/Core/util/Macros.h':'8d73259b4ba482e6dbd11b31d8887fd350dfb0836700d32793446bba6ab1ae7a',
 'eigen/COPYING.MPL2':'fab3dd6bdab226f1c08630b1dd917e11fcb4ec5e1e020e2c16f83a0a13863e85'}
for path,digest in expected.items():
    if sha(source/path) != digest: raise ValueError('Pinned primary source SHA mismatch: '+path)
compiler=shutil.which('c++')
if compiler is None: raise RuntimeError('Existing base has no C++ compiler; no automatic apt repair')
version=subprocess.check_output([compiler,'--version'],text=True,timeout=remaining())
cpp=root/'mesh_guarded_qem.cpp'; cppsha=sha(cpp); binary=root/'mesh_guarded_qem'
command=[compiler,'-std=c++17','-O2','-fno-fast-math','-ffp-contract=off','-DEIGEN_DONT_PARALLELIZE','-DEIGEN_MPL2_ONLY',f'-DWR_SOURCE_SHA256="{cppsha}"',f'-I{igldir}/include',f'-I{eigendir}',str(cpp),'-o',str(binary)]
subprocess.run(command,check=True,timeout=remaining())
info=json.loads(subprocess.check_output([str(binary),'--build-info'],text=True,timeout=remaining()))
if info != {'libigl_revision':igl,'eigen_revision':eigen,'source_sha256':cppsha,'target_faces':4096,'block_intersections':True}: raise ValueError('Binary ABI build binding mismatch')
inventory={p.relative_to(source).as_posix():sha(p) for p in sorted(source.rglob('*')) if p.is_file()}
report={**info,'stage':'world_reward_guarded_qem_binary_build','status':'pass','binary_sha256':sha(binary),'compiler':version,'compile_command':command,'archive_sources':archive_hashes,'source_inventory_sha256':hashlib.sha256(json.dumps(inventory,sort_keys=True,separators=(',',':')).encode()).hexdigest(),'pinned_primary_sha256':expected,'source_cpp_sha256':cppsha,'build_script_sha256':sha(root/'build_guarded_qem.sh'),'elapsed_seconds':time.monotonic()-start,'licenses':['libigl MPL-2.0','Eigen MPL-2.0/BSD notices retained; EIGEN_MPL2_ONLY'],'challenge_inputs_used':False,'dependency_installation_performed':False}
with (root/'build.json').open('x') as handle: json.dump(report,handle,indent=2);handle.write('\n')
PYBUILD
