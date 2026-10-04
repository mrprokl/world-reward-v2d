"""Four fresh paired physical-geometry controls; no data, reroll or adoption.

Only numeric conditioning is new. Every candidate is evaluated in physical
coordinates through the existing F32/GLB/default-eight/official-pack/.375 gates.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import stat
import subprocess
import time

import numpy as np

import mesh_serialization_geometry as original
from world_reward.mesh_conditioning import prepare_conditioning, POLICY_SHA256
from world_reward.mesh_serialization import serialization_preflight

GeometryControlError = original.GeometryControlError
identity, array_hashes, math_mesh, stage = original.identity, original.array_hashes, original.math_mesh, original.stage
require, geometry, endpoint = original.require, original.geometry, original.endpoint
topology_and_embedding = original.topology_and_embedding
ORIGINAL_BINARY = original.ORIGINAL_BINARY
SHAPE_NAMES = ('fresh_symmetric_superellipsoid', 'fresh_thick_offset_superellipsoid_cavity')
AZIMUTH, POLAR, INNER_AZIMUTH, INNER_POLAR = 96, 48, 32, 16
AXES, POWER, INNER_SCALE, INNER_TRANSLATION = (1., 7/8, 3/4), .8, .4, (.05, 0., 0.)
SCALES = (2.**-14, 2.**2)
TRANSLATION = (4., -2., 1.)
ROTATIONS = (((1., 0., 0.), (0., 1., 0.), (0., 0., 1.)),
             ((0., -1., 0.), (1., 0., 0.), (0., 0., 1.)))
FIXTURE_NAMES = tuple(name+'_'+suffix for name in SHAPE_NAMES for suffix in ('small_translated', 'large_rotated_translated'))
NATIVE_SECONDS, METRIC_SCALE = 600, .375


def radial_mesh(azimuth=AZIMUTH, polar=POLAR):
    """Closed sign-power surface, antipodal pairs, unique poles and no seam copy."""
    require(type(azimuth)is int and azimuth>=4 and azimuth%4==0 and type(polar)is int and polar>=4 and polar%2==0,
            'Even fixed sphere-like sampling required')
    theta=2*np.pi*np.arange(azimuth//2)/azimuth
    angle=np.c_[np.cos(theta),np.sin(theta)]
    angle=np.r_[angle,-angle]  # Angular antipodes are constructed, not snapped.
    power=lambda x:np.sign(x)*np.abs(x)**POWER
    north=[]
    for k in range(1,polar//2):
        phi=np.pi*k/polar
        north.append(np.c_[power(angle*np.sin(phi)),np.full(azimuth,np.cos(phi)**POWER)])
    equator=np.c_[power(angle),np.zeros(azimuth)]  # Analytic equator z=0.
    shift=(np.arange(azimuth)+azimuth//2)%azimuth
    rings=[-ring[shift]for ring in north]+[equator]+list(reversed(north))
    vertices=np.vstack((*rings,[[0.,0.,-1.],[0.,0.,1.]]))*np.array(AXES)
    bottom,top=len(vertices)-2,len(vertices)-1;faces=[]
    for ring in range(polar-2):
        for j in range(azimuth):
            a=ring*azimuth+j;b=ring*azimuth+(j+1)%azimuth;c=a+azimuth;d=b+azimuth
            faces.extend(((a,b,c),(b,d,c)))
    for j in range(azimuth):
        faces.extend(((bottom,(j+1)%azimuth,j),(top,(polar-2)*azimuth+j,(polar-2)*azimuth+(j+1)%azimuth)))
    return vertices.astype(np.float64),np.asarray(faces,np.int64)


def fixtures():
    outer=radial_mesh();inner=radial_mesh(INNER_AZIMUTH,INNER_POLAR)
    hollow=(np.r_[outer[0],inner[0]*INNER_SCALE+np.array(INNER_TRANSLATION)],
            np.r_[outer[1],inner[1][:,::-1]+len(outer[0])])
    shapes=((outer,False),(hollow,True));records=[]
    for index,(mesh,cavity)in enumerate(shapes):
        for variant,(scale,rotation)in enumerate(zip(SCALES,ROTATIONS)):
            v,f=mesh
            physical=(v@np.array(rotation).T)*scale+np.array(TRANSLATION)*scale
            records.append((FIXTURE_NAMES[2*index+variant],(physical,f.copy()),cavity))
    return tuple(records)


def branch(source,cavity,binary,method,work,helper,remaining,record):
    a,b,m=(work/n for n in('input.obj','candidate.obj','mapping.json'))
    geometry.write_obj(a,*source);before=identity(a)
    record.update(phase='native',native_attempts=1,native_returned=False,native_input=before)
    started=time.monotonic()
    try:child=subprocess.run([str(binary),str(a),str(b),str(m)],capture_output=True,timeout=min(NATIVE_SECONDS,remaining()))
    finally:
        record['native_elapsed_seconds']=time.monotonic()-started
        require(identity(a)==before,'Native input OBJ changed')
    record.update(native_returned=True,native_exit_code=child.returncode)
    if child.returncode:record['native_stderr_tail']=child.stderr[-500:].decode(errors='replace')
    require(child.returncode==0,'Native compiler failed fixed fresh control')
    candidate=geometry.read_obj(b);mapping=json.loads(m.read_text())
    if method=='conditioned':
        conditioning,serialization=mapping['conditioning'],mapping['serialization']
        required=dict(chart_scale_positive=True,physical_geometry_rescaled=False,source_roundtrip_numerically_exact=True,
                      new_numeric_algorithm=True,native_qslim_implementation_reused=True)
        require(all(type(conditioning.get(k))is type(v)and conditioning[k]==v for k,v in required.items()),
                'Actual fixed conditioning implementation required')
        require(serialization['serialization_safe']is True and type(serialization['serialization_vetoes'])is int
                and serialization['serialization_vetoes']>=0 and serialization['committed_collapses']>0,
                'Real admissible conditioned collapses required')
        record.update(conditioning=conditioning,serialization=serialization,serialization_vetoes=serialization['serialization_vetoes'])
        mapping=mapping['native_volume']
        require(serialization['committed_collapses']==mapping['committed_collapses'],'Collapse accounting differs')
    require(mapping['native_cost_and_placement_unchanged']is(method=='original')
            and mapping['cost_normalization']is(method=='conditioned')and mapping['final_shell_volumes_verified']is True
            and mapping['volume_relative_limit']==.05 and mapping['committed_collapses']>0,'Native volume/cost contract differs')
    volume_scalars={k:v for k,v in mapping.items()if type(v)in(bool,int,float,str)}
    json.dumps(volume_scalars,allow_nan=False)
    record.update(native_artifacts={p.name:identity(p)for p in(a,b,m)},native_volume=volume_scalars,
                  committed_collapses=mapping['committed_collapses'],phase='candidate')
    record['candidate']=stage(source,candidate,mapping,cavity,remaining)
    require(serialization_preflight(*candidate)['position_weld_admissible'],'Candidate physical serialization unsafe')
    import trimesh
    glb=work/'candidate.glb';record['phase']='float32_export';trimesh.Trimesh(*candidate,process=False).export(glb)
    loaded=endpoint._load_mesh(glb)
    require(np.array_equal(loaded[0],loaded[0].astype(np.float32).astype(np.float64)),'GLB did not preserve binary32')
    exported=endpoint.exact_weld(*loaded)[:2];emap=original.export_birth_mapping(candidate,exported,mapping)
    record['exported']=stage(source,exported,emap,cavity,remaining);record['glb_identity']=identity(glb)
    record['phase']='official_pack';pv,pf=helper.budget_mesh(str(glb),faces=4096,vertices=4096)
    compact,fidelity=geometry.verify_pack_fidelity(exported,pv,pf);pmap=original.packed_mapping(exported,compact,emap)
    record['packed']=stage(source,compact,pmap,cavity,remaining);record['official_pack_fidelity']=fidelity
    record['phase']='metric_bake';metric=pv*METRIC_SCALE;expected=(compact[0].astype(pv.dtype)*METRIC_SCALE,compact[1])
    metric_compact,_=geometry.verify_pack_fidelity(expected,metric,pf)
    record['metric_baked']=stage((source[0]*METRIC_SCALE,source[1]),metric_compact,pmap,cavity,remaining)
    record.update(status='pass',phase='complete',metric_scale_baked_once=METRIC_SCALE)


def geometry_controls(binary,scratch,remaining,*,official_helper):
    """Four new sources, eight bounded calls; partial scalar report on failure.

    Caller authenticates binary/runtime and owns the inclusive 5400s deadline.
    Original measured failures remain comparisons. Technical failures or ANY
    conditioned failure stop immediately; source cohort/gates are never rerolled.
    """
    binary,scratch,official_helper=map(Path,(binary,scratch,official_helper))
    require(scratch.is_absolute()and scratch.resolve()==scratch and scratch.is_dir()
            and not any(p.is_symlink()for p in(scratch,*scratch.parents)),'Canonical caller scratch required')
    work=scratch/'conditioned-geometry-controls';require(not work.exists()and not work.is_symlink(),'Fresh geometry scratch required')
    remaining();before={str(p):identity(p)for p in(binary,ORIGINAL_BINARY,official_helper)}
    require(binary!=ORIGINAL_BINARY and before[str(official_helper)]==dict(bytes=2031,sha256=endpoint.BUDGET_HELPER_SHA),
            'Distinct compiler and independently pinned single official helper required')
    legacy=geometry.validate_legacy_sources()
    report=dict(stage='mesh_conditioned_geometry_controls_v1',status='fail',adoption=False,challenge_inputs_used=False,
        reconstruction_accuracy_verified=False,procedural_geometry_qualification_only=True,native_budget_seconds=NATIVE_SECONDS,
        maximum_native_calls=8,paired_fixtures=[],source_artifacts=before,legacy_sources=legacy,metric_scale=METRIC_SCALE,
        conditioning_policy_sha256=POLICY_SHA256,reroll_performed=False)
    work.mkdir(mode=0o700);owner=work.stat();failure=None;source_records=[]
    try:
        spec=importlib.util.spec_from_file_location('wr_conditioned_official_single_helper',official_helper)
        helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper);sources=fixtures()
        require(tuple(n for n,_,_ in sources)==FIXTURE_NAMES,'Exactly frozen new cohort required')
        for name,source,cavity in sources:
            remaining();hashes=array_hashes(source);source_records.append((source,hashes))
            chart=prepare_conditioning(*source)
            require(chart.diagnostics['roundtrip_numerically_exact']is True and chart.scale>0,'Whole referenced F64 chart roundtrip must be exact')
            require(len(source[1])>4096 and serialization_preflight(*source)['position_weld_admissible'],'Safe complete reduction source required')
            top=topology_and_embedding(math_mesh(source),cavity);stored=(source[0].astype(np.float32),source[1])
            stored_top=topology_and_embedding(math_mesh(stored),cavity)
            require(serialization_preflight(*stored)['position_weld_admissible'],'Source F32/default-eight serialization unsafe')
            require(array_hashes(source)==hashes,'Source qualification changed geometry')
            report['paired_fixtures'].append(dict(fixture=name,source_array_sha256=hashes,source_topology=top,
                source_float32_topology=stored_top,source_float32_array_sha256=array_hashes(stored),
                conditioning=dict(chart.diagnostics),comparisons=[]))
        for(name,source,cavity),record in zip(sources,report['paired_fixtures']):
            for method,compiler in(('original',ORIGINAL_BINARY),('conditioned',binary)):
                remaining();selected=dict(method=method,status='fail',phase='pre_native',native_attempts=0,native_returned=False)
                record['comparisons'].append(selected);directory=work/(name+'-'+method);directory.mkdir(mode=0o700)
                try:branch(source,cavity,compiler,method,directory,helper,remaining,selected)
                except Exception as exc:
                    scope=('native_timeout'if isinstance(exc,subprocess.TimeoutExpired)else
                        'measured_geometry_or_native_rejection'if selected.get('native_attempts')==1 and isinstance(exc,ValueError)
                        else'technical_unavailable')
                    selected.update(error_type=type(exc).__name__,error=str(exc)[-500:],failure_scope=scope)
                    if method=='conditioned'or scope in('technical_unavailable','native_timeout'):raise
                require(array_hashes(source)==record['source_array_sha256'],'Paired source changed')
        new=[r for fixture in report['paired_fixtures']for r in fixture['comparisons']if r['method']=='conditioned']
        require(len(new)==4 and all(r['status']=='pass'and r['committed_collapses']>0 for r in new),'Every fresh conditioned arm must collapse and pass')
        vetoes=sum(r['serialization_vetoes']for r in new)
        report.update(status='pass',serialization_vetoes=vetoes,serialization_veto_mechanism_exercised=bool(vetoes),
                      conditioning_mechanism_exercised=True)
    except Exception as exc:failure=exc;report.update(error_type=type(exc).__name__,error=str(exc)[-500:])
    finally:
        try:
            require(all(array_hashes(mesh)==hashes for mesh,hashes in source_records),'Source arrays changed')
            require({str(p):identity(p)for p in(binary,ORIGINAL_BINARY,official_helper)}==before
                    and geometry.validate_legacy_sources()==legacy,'Original binary/helper/source changed')
            remaining();report['sources_rehashed_after']=True
        except Exception as exc:failure=failure or exc;report['post_error_type']=type(exc).__name__
        try:
            now=work.lstat();require(stat.S_ISDIR(now.st_mode)and(now.st_dev,now.st_ino)==(owner.st_dev,owner.st_ino),'Owned scratch replaced')
            shutil.rmtree(work);report['owned_scratch_removed']=True;remaining()
        except Exception as exc:failure=failure or exc;report['cleanup_error_type']=type(exc).__name__
    if failure:
        report['status']='fail';raise GeometryControlError('Frozen conditioned geometry control failed',report)from failure
    return report
