"""Static mandatory wheel closure only; no network, install, model or images."""
import copy
from datetime import datetime
import json
from pathlib import Path
import re
from urllib.parse import urlsplit

import pytest
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.tags import compatible_tags, cpython_tags
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version

REPO=Path(__file__).resolve().parents[1]
PATH=REPO/'configs/mediapipe_cpu_dependencies_v1.json'
BASE='sha256:7ebfff18ba3b76dd919485c19115597d7531dfd3233f69461f1dce3f28a6c6d3'
EXPECTED={
    'mediapipe':'0.10.21','numpy':'1.26.4','jax':'0.4.35','jaxlib':'0.4.35',
    'opencv-contrib-python':'4.11.0.86','protobuf':'4.25.8','matplotlib':'3.9.4',
    'scipy':'1.14.1','ml-dtypes':'0.5.1','opt-einsum':'3.4.0','absl-py':'2.1.0',
    'attrs':'25.3.0','flatbuffers':'25.2.10','sounddevice':'0.5.2','sentencepiece':'0.2.0',
    'cffi':'1.17.1','pycparser':'2.22','contourpy':'1.3.1','cycler':'0.12.1',
    'fonttools':'4.59.0','kiwisolver':'1.4.7','packaging':'25.0','pillow':'11.3.0',
    'pyparsing':'3.2.3','python-dateutil':'2.9.0.post0','six':'1.17.0',
}


def validate(manifest):
    assert manifest['schema']=='world_reward.mediapipe_cpu_dependencies.v1'
    assert manifest['status']=='preregistered_metadata_only_not_built_or_installed'
    assert manifest['base_image_id']==BASE
    assert manifest['platform']==dict(os='linux',architecture='amd64',python_implementation='CPython',
                                      python_version='3.11',minimum_glibc='2.28')
    environment=manifest['environment']
    assert environment['isolated_venv']is True and environment['system_site_packages']is False
    assert environment['resolver_at_build']is False and environment['GPU_extras']==[]
    assert environment['sdists_allowed']is False and environment['opencv_substitution_allowed']is False
    selected={row['name']:row for row in manifest['packages']}
    assert len(selected)==len(manifest['packages'])==manifest['package_count']==26
    assert {name:row['version']for name,row in selected.items()}==EXPECTED
    env=manifest['marker_environment']
    assert env['implementation_name']=='cpython'and env['platform_machine']=='x86_64'
    assert env['platform_system']=='Linux'and env['sys_platform']=='linux'
    assert env['python_version']=='3.11'and env['python_full_version']=='3.11.0'and env['extra']==''
    platforms=['manylinux_2_'+str(n)+'_x86_64'for n in range(28,4,-1)]
    platforms+=['manylinux2014_x86_64','manylinux2010_x86_64','manylinux1_x86_64']
    tags=set(cpython_tags((3,11),abis=['cp311'],platforms=platforms))|set(compatible_tags((3,11),interpreter='cp311',platforms=platforms))
    cutoff=datetime.fromisoformat(manifest['publication_cutoff'].replace('Z','+00:00'));graph={}
    for name,row in selected.items():
        assert row['yanked']is False and type(row['bytes'])is int and 0<row['bytes']<1_000_000_000
        assert datetime.fromisoformat(row['upload_time'].replace('Z','+00:00'))<=cutoff
        url=urlsplit(row['url']);assert url.scheme=='https'and url.hostname=='files.pythonhosted.org'
        assert not url.query and not url.fragment and not url.username and url.path.startswith('/packages/')
        assert url.path.endswith('/'+row['filename'])and row['filename'].endswith('.whl')
        parsed_name,version,_,wheel_tags=parse_wheel_filename(row['filename'])
        assert parsed_name==name and version==Version(row['version'])and tags&wheel_tags
        meta=row['metadata'];assert meta['url']==row['url']+'.metadata'
        assert meta['hash_basis']=='publisher_PEP658_sha256'and 0<meta['bytes']<=256<<10
        assert re.fullmatch('[0-9a-f]{64}',row['sha256'])and re.fullmatch('[0-9a-f]{64}',meta['sha256'])
        assert not meta['requires_python']or Version('3.11.0')in SpecifierSet(meta['requires_python'])
        license=meta['license'];assert license['embedded_notices_verified']is False
        assert license['expression']or license['declared_text_sha256']or license['classifiers']
        graph[name]=[]
        for raw in meta['requires_dist']:
            requirement=Requirement(raw)
            if requirement.marker and not requirement.marker.evaluate(env):continue
            dependency=canonicalize_name(requirement.name)
            assert dependency in selected and not requirement.url and not requirement.extras
            assert Version(selected[dependency]['version'])in requirement.specifier
            graph[name].append(dependency)
        graph[name]=sorted(set(graph[name]))
        release=row['pypi_release'];assert release['url']=='https://pypi.org/pypi/'+name+'/'+row['version']+'/json'
        assert release['hash_basis']=='audit_observed_metadata_not_immutable_release_json'
        assert 0<release['bytes']<=1_000_000 and re.fullmatch('[0-9a-f]{64}',release['sha256'])
    assert graph==manifest['mandatory_dependency_graph']
    reachable=set();pending=['mediapipe']
    while pending:
        name=pending.pop()
        if name in reachable:continue
        reachable.add(name);pending.extend(graph[name])
    assert reachable==set(selected)  # No missing requirement or unnecessary extra.
    total=sum(row['bytes']for row in selected.values())
    assert total==manifest['total_wheel_bytes']==282_981_523
    assert manifest['new_wheel_bytes']==total-selected['mediapipe']['bytes']==247_358_885
    assert total<=manifest['maximum_total_wheel_bytes']==1_000_000_000
    assert selected['mediapipe']['acquisition']=='reference_existing_mediapipe_hands_acquire_v1'
    assert all(r['acquisition']=='new_Azure_wheel_only'for n,r in selected.items()if n!='mediapipe')
    return selected


def test_exact_official_platform_cutoff_and_closed_mandatory_dependencies():
    validate(json.loads(PATH.read_text()))


def test_original_mediapipe_wheel_reference_matches_acquisition_without_duplicate_fetch():
    # Read source constants, without importing/running an acquisition routine.
    import ast
    source=ast.parse((REPO/'infra/mediapipe_hands_acquire.py').read_text())
    constants={n.targets[0].id:ast.literal_eval(n.value)for n in source.body if isinstance(n,ast.Assign)
               and isinstance(n.targets[0],ast.Name)and isinstance(n.value,ast.Constant)}
    row=validate(json.loads(PATH.read_text()))['mediapipe']
    assert row['filename']==constants['WHEEL']and row['metadata']['sha256']==constants['METADATA_SHA']
    assert row['bytes']==35_622_638 and row['sha256']=='05dc4a9e593655a79558d05d6227d31018c2537a4bd3362b51e230cf22aecfe3'


def test_no_built_runtime_accuracy_or_license_clearance_inferred_from_metadata():
    m=json.loads(PATH.read_text());q=m['qualification'];p=m['prerequisites']
    assert q['gpus']is False and q['dataset_read']is False and q['detect_calls']==0
    assert q['native_model_load_only']is True and q['network']=='none'
    assert q['build_budget_seconds']==600 and q['cpu_smoke_budget_seconds']==120
    assert q['hand_landmarker_mode']=='IMAGE'and q['delegate']=='CPU'and q['num_hands']==4
    assert q['success_means']=='dependency_import_and_native_graph_load_not_hand_accuracy'
    assert p['embedded_wheel_notice_audit_required']is True and p['mediapipe_acquisition_pass_required']is True
    assert all(p[name]is False for name in('task_constituent_license_verified','license_eligibility_verified',
                                         'training_overlap_verified','challenge_overlap_verified'))


@pytest.mark.parametrize('mutation',['missing','numpy2','latest_jax','platform','yanked','late','sha','url','metadata',
                                   'required_dependency','extra','base','system_sites','headless','budget'])
def test_static_mutations_break_the_frozen_install_contract(mutation):
    m=copy.deepcopy(json.loads(PATH.read_text()));rows={r['name']:r for r in m['packages']}
    if mutation=='missing':m['packages']=[r for r in m['packages']if r['name']!='pycparser']
    elif mutation=='numpy2':rows['numpy']['version']='2.0.0'
    elif mutation=='latest_jax':rows['jax']['version']='0.8.0'
    elif mutation=='platform':rows['numpy']['filename']=rows['numpy']['filename'].replace('x86_64','aarch64')
    elif mutation=='yanked':rows['cffi']['yanked']=True
    elif mutation=='late':rows['cffi']['upload_time']='2026-10-01T00:00:00Z'
    elif mutation=='sha':rows['cffi']['sha256']='unknown'
    elif mutation=='url':rows['cffi']['url']='https://untrusted.invalid/'+rows['cffi']['filename']
    elif mutation=='metadata':rows['cffi']['metadata']['hash_basis']='self_asserted'
    elif mutation=='required_dependency':rows['mediapipe']['metadata']['requires_dist'].append('torch>=2')
    elif mutation=='extra':rows['jax']['metadata']['requires_dist'].append('jax-cuda12-plugin; platform_system == "Linux"')
    elif mutation=='base':m['base_image_id']='sha256:'+'0'*64
    elif mutation=='system_sites':m['environment']['system_site_packages']=True
    elif mutation=='headless':m['environment']['opencv_substitution_allowed']=True
    else:m['maximum_total_wheel_bytes']=1
    with pytest.raises((AssertionError,ValueError)):validate(m)
