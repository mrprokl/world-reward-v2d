#!/usr/bin/env bash
set -euo pipefail
(( $# == 0 )) || exit 2
# Existing frozen source only: no acquisition, package install or network fallback.
timeout --signal=TERM --kill-after=5s 600s python3 - <<'PYBUILD'
import hashlib,json,os,pathlib,re,shutil,stat,subprocess,time
started=time.monotonic(); base=pathlib.Path('/opt/world-reward/guarded-qem'); root=pathlib.Path('/opt/world-reward/volume-qem')
def remaining():
    value=600-(time.monotonic()-started)
    if value<=0: raise TimeoutError('Whole inherited-source/compile deadline exceeded')
    return value
def regular(path):
    if not stat.S_ISREG(path.lstat().st_mode): raise ValueError('Require regular inherited/build file: '+str(path))
    return path
def sha(path):
    remaining(); return hashlib.sha256(regular(path).read_bytes()).hexdigest()
expected_base=os.environ['WR_BASE_CPP_SHA256']
if not re.fullmatch('[0-9a-f]{64}',expected_base): raise ValueError('Require explicit frozen base CPP identity')
inherited=json.loads(regular(base/'build.json').read_text())
if (inherited.get('status')!='pass' or inherited.get('stage')!='world_reward_guarded_qem_binary_build'
        or inherited.get('libigl_revision')!='40e7900ccbd767f1f360e0eb10f0f1a6432e0993'
        or inherited.get('eigen_revision')!='3147391d946bb4b6c68edd901f2add6ac1f31f8c'
        or inherited.get('source_cpp_sha256')!=expected_base or sha(base/'mesh_guarded_qem.cpp')!=expected_base
        or sha(base/'mesh_guarded_qem')!=inherited.get('binary_sha256')):
    raise ValueError('Inherited guarded image/source/binary identity differs')
source=base/'source'
if source.is_symlink() or not source.is_dir(): raise ValueError('Require inherited regular source tree')
inventory={}
for path in sorted(source.rglob('*')):
    remaining()
    if path.is_symlink(): raise ValueError('No inherited source symlinks')
    if path.is_file(): inventory[path.relative_to(source).as_posix()]=sha(path)
inventory_sha=hashlib.sha256(json.dumps(inventory,sort_keys=True,separators=(',',':')).encode()).hexdigest()
if inventory_sha!=inherited.get('source_inventory_sha256'): raise ValueError('Inherited header inventory changed')
for path,digest in inherited['pinned_primary_sha256'].items():
    if inventory.get(path)!=digest: raise ValueError('Inherited primary source identity differs')
compiler=shutil.which('c++')
if compiler is None: raise RuntimeError('Existing base has no compiler; no apt fallback')
version=subprocess.check_output([compiler,'--version'],text=True,timeout=remaining())
cpp=root/'mesh_volume_qem.cpp'; cppsha=sha(cpp); binary=root/'mesh_volume_qem'
if binary.exists() or binary.is_symlink() or (root/'build.json').exists(): raise FileExistsError('Fresh volume build required')
command=[compiler,'-std=c++17','-O2','-fno-fast-math','-ffp-contract=off','-DEIGEN_DONT_PARALLELIZE','-DEIGEN_MPL2_ONLY',
    f'-DWR_SOURCE_SHA256="{expected_base}"',f'-DWR_BASE_SOURCE_SHA256="{expected_base}"',f'-DWR_VOLUME_SOURCE_SHA256="{cppsha}"',
    f'-I{source}/libigl/include',f'-I{source}/eigen',str(cpp),'-o',str(binary)]
subprocess.run(command,check=True,timeout=remaining())
info=json.loads(subprocess.check_output([str(binary),'--build-info'],text=True,timeout=remaining()))
expected={'libigl_revision':inherited['libigl_revision'],'eigen_revision':inherited['eigen_revision'],
    'source_sha256':cppsha,'base_source_sha256':expected_base,'target_faces':4096,'block_intersections':True,
    'volume_relative_limit':.05,'native_cost_and_placement_unchanged':True,'cost_normalization':False}
if info!=expected: raise ValueError('Volume binary/source ABI differs')
report={**info,'stage':'world_reward_volume_qem_binary_build','status':'pass','binary_sha256':sha(binary),
    'compiler':version,'compile_command':command,'source_cpp_sha256':cppsha,'base_source_cpp_sha256':expected_base,
    'inherited_build_report_sha256':sha(base/'build.json'),'source_inventory_sha256':inventory_sha,
    'build_script_sha256':sha(root/'build_volume_qem.sh'),'elapsed_seconds':time.monotonic()-started,
    'licenses':inherited['licenses'],'challenge_inputs_used':False,'dependency_installation_performed':False,
    'source_acquisition_performed':False,'build_network':'none','geometry_controls_executed':False}
with (root/'build.json').open('x') as stream: json.dump(report,stream,indent=2);stream.write('\n')
PYBUILD
