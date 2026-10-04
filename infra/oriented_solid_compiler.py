"""Prepared whole-solid compiler: exact forest gates around unchanged cached QEM.

Caller authenticates both qualified binaries/image and the single official helper.
No production selection, host launcher, solver modification, repair or adoption.
"""
from __future__ import annotations
from fractions import Fraction
import hashlib
import importlib.util
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import certified_solid_source as certificate
import exact_mesh_geometry as geometry
import mesh_precision_diagnostic as precision
import object_budget_endpoint as endpoint
import mesh_serialization_geometry as serialized_geometry
import object_budget_guarded as export_helper
import precision_volume_gate as volume_helper
import world_reward.mesh_conditioning as chart_helper
import world_reward.mesh_conditioning_v2 as chart_v2_helper
import mesh_conditioned_chart_v2_source as chart_v2_source
import world_reward.mesh_serialization as serialization_helper
import world_reward.oriented_solid_forest as forest_helper
from object_budget_conditioned import validate_mapping
from object_budget_guarded import export_birth_mapping
from precision_volume_gate import packed_mapping
from mesh_serialization_geometry import array_hashes, math_mesh
from world_reward.mesh_conditioning import prepare_conditioning
from world_reward.mesh_conditioning_v2 import prepare_conditioning_v2
from world_reward.mesh_serialization import serialization_preflight
from world_reward.oriented_solid_forest import adjudicate_oriented_solid_forest
from world_reward.solid_forest_fidelity import compare_solid_forest_fidelity
from world_reward.mesh_serialization import _weld_keys

TOTAL_SECONDS, QEM_SECONDS, QUERY_SECONDS = 1800, 1200, 180
STAGES = ('physical_source','native_candidate','float32_glb','default8_exact_weld',
          'unmodified_official_pack','original_grounding_metric_bake')
FILES = frozenset(('input.obj','candidate.obj','mapping.json','candidate.glb'))
CHART_V2_HEADER = Path(__file__).with_name(chart_v2_source.HEADER_FILE)


class SolidCompilerError(RuntimeError):
    def __init__(self,report):
        super().__init__('Whole oriented-solid compiler failed');self.report=report


def require(ok,message):
    if not ok:raise ValueError(message)


def conditioning_function(version):
    require(type(version) is int and version in (1,2),'Explicit conditioning_version integer1 or2 required')
    return prepare_conditioning if version==1 else prepare_conditioning_v2


def validate_conditioned_mapping(source,candidate,document,chart,*,conditioning_version=1):
    """Same numeric mapping gates; v2 additionally binds the new source chart.

    Binary build/qualification authentication remains the immutable caller's
    responsibility. A v2 source/header identity is not a qualification claim.
    """
    conditioning_function(conditioning_version)
    if conditioning_version==2:
        require(isinstance(chart,chart_v2_helper.MeshConditioningChartV2),'Actual source v2 chart required')
        c=document['conditioning']
        keys={'chart_version','header_sha256','policy_sha256','origin','origin_modes','scale','scale_exponent',
              'source_roundtrip_vertices','chart_scale_positive','source_roundtrip_numerically_exact',
              'source_roundtrip_byte_exact','origin_search_performed','physical_geometry_rescaled',
              'new_numeric_algorithm','native_qslim_implementation_reused','chart_refitted',
              'native_backend_qualified','adopted'}
        require(type(c) is dict and set(c)==keys and type(c['chart_version']) is int and c['chart_version']==2
                and c['header_sha256']==certificate.identity(CHART_V2_HEADER)['sha256']
                and c['policy_sha256']==chart_v2_helper.POLICY_SHA256==chart.diagnostics['policy_sha256']
                and type(c['origin_modes']) is list and c['origin_modes']==list(chart.origin_modes)
                and c['origin_search_performed'] is False and c['native_backend_qualified'] is False
                and c['source_roundtrip_byte_exact'] is chart.diagnostics['roundtrip_byte_exact']
                and type(c['origin']) is list and len(c['origin'])==3
                and all(type(x) in (int,float) and np.isfinite(x) for x in c['origin'])
                and type(c['scale']) in (int,float),
                'Native v2 chart version/header/policy/origin evidence differs')
    return validate_mapping(source,candidate,document,chart)


def deadline(outer=None):
    started=time.monotonic()
    def remaining():
        value=TOTAL_SECONDS-(time.monotonic()-started)
        if outer is not None:value=min(value,outer())
        if value<=0:raise TimeoutError('Fixed inclusive1800s whole-solid deadline')
        return value
    return remaining


def load_source(mesh_path):
    """Unprocessed scene only; exact seams retained with every oriented face."""
    before=certificate.identity(Path(mesh_path));local,triangles,_=precision.raw_glb(Path(mesh_path))
    raw=endpoint._load_mesh(Path(mesh_path))
    require(all(np.array_equal(np.unique(f),np.arange(len(v))) for v,f in (*local,raw)),
            'Unreferenced raw vertices unsupported; no orphan deletion')
    digest=precision.triangle_hash(triangles)
    require(precision.triangle_hash(raw[0][raw[1]])==digest,'Raw/loader oriented triangle multiset differs')
    v,f,weld=endpoint.exact_weld(*raw)
    require(len(f)==len(raw[1]) and np.array_equal(v[f],raw[0][raw[1]]),'Exact weld changed loaded face order')
    require(certificate.identity(Path(mesh_path))==before,'Original source changed while loaded')
    return (v.astype(np.float64),f),dict(source_identity=before,raw_oriented_triangles_sha256=digest,
        exact_welding=weld,order_scope=certificate.ORDER_SCOPE)


def certify_arrays(mesh,binary,source_sha256,remaining,record):
    """One actual EPECK query on full represented arrays; never invent inside."""
    remaining();v,f=math_mesh(mesh);labels,keys,rows=certificate.components(v,f)
    data=certificate.input_ascii(v,f,labels,len(keys));record.update(query_attempted=True,
        query_input=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
    child=subprocess.run([str(binary)],input=data,capture_output=True,check=False,
                         timeout=min(QUERY_SECONDS,remaining()))
    record.update(query_returned=True,query_returncode=child.returncode)
    require(len(child.stdout)<=certificate.MAX_STDOUT and len(child.stderr)<=certificate.MAX_STDERR,'Exact query output capacity')
    if child.returncode:
        record['failure_scope']='native_execution_contract'
        prefix=b'certified_solid_query FAIL: '
        if child.returncode==1 and not child.stdout and child.stderr.startswith(prefix) and child.stderr.endswith(b'\n'):
            reason=child.stderr[len(prefix):-1].decode('ascii',errors='strict')
            record['native_rejection']=reason[:300]
            if reason in certificate.GEOMETRY_REJECTIONS:record['failure_scope']='native_geometric_rejection'
        raise ValueError('Exact query failed; no substituted containment')
    require(not child.stderr,'Successful exact query emitted stderr')
    native,signs,inside=certificate.native_certificate(child.stdout,source_sha256,v,f,rows)
    forest=adjudicate_oriented_solid_forest(keys,signs,inside)
    top=geometry.exact_mesh_topology(v,f)
    record.update(native_certificate=native,topology=top,stored_array_sha256=array_hashes(mesh),
        forest=dict(component_keys=keys,signs=signs.tolist(),inside=inside.tolist(),
                    parents=forest.parents.tolist(),depths=forest.depths.tolist()))
    remaining();return labels,forest


def float32_orientation(source):
    """Exact normal dot on original F64 versus represented F32, no epsilon."""
    v,f=source;stored=v.astype(np.float32).astype(np.float64)
    require(np.isfinite(stored).all(),'Source cannot be represented as finiteF32')
    def normal(triangle):
        p=[[Fraction.from_float(float(x)) for x in row] for row in triangle]
        a=[p[1][i]-p[0][i] for i in range(3)];b=[p[2][i]-p[0][i] for i in range(3)]
        return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])
    for face in f:
        a,b=normal(v[face]),normal(stored[face])
        require(any(a) and any(b) and sum(x*y for x,y in zip(a,b))>0,
                'Source F32 triangle collapsed or changed local orientation')
    return dict(triangles=len(f),exact_positive_normal_dot=True,area_tolerance_used=False)


def identity_mapping(mesh):
    v,f=mesh
    return dict(source_vertices=len(v),source_faces=len(f),output_vertices=len(v),output_faces=len(f),
                target_reached=True,mapping_complete=True,I=list(range(len(v))),J=list(range(len(f))))


def reordered_mapping(before,after,mapping):
    """Exact coordinate/cyclic-face permutation only; no nearest birth inference."""
    bv,bf=before;av,af=after
    require(precision.triangle_hash(bv[bf])==precision.triangle_hash(av[af]),'Processing changed oriented triangle multiset')
    require(len(bv)==len(av) and len(bf)==len(af),'Processing removed vertices or faces')
    lookup={tuple(p):i for i,p in enumerate(bv)}
    require(len(lookup)==len(bv),'Exact vertex birth permutation ambiguous')
    order=np.array([lookup[tuple(p)] for p in av],np.int64)
    require(len(np.unique(order))==len(bv),'Processing merged original vertex positions')
    converted=order[af]
    key=lambda row:min(tuple(row),tuple(np.roll(row,1)),tuple(np.roll(row,2)))
    face_lookup={key(row):i for i,row in enumerate(bf)}
    require(len(face_lookup)==len(bf),'Original face birth permutation ambiguous')
    face_order=[face_lookup[key(row)] for row in converted]
    require(len(set(face_order))==len(bf),'Processing duplicated or deleted oriented faces')
    return mapping|dict(I=np.asarray(mapping['I'])[order].tolist(),J=np.asarray(mapping['J'])[face_order].tolist())


def matched_stage(source,source_labels,source_forest,mesh,mapping,binary,sha,remaining,record,*,measure=True):
    labels,forest=certify_arrays(mesh,binary,sha,remaining,record)
    I,J=np.asarray(mapping['I']),np.asarray(mapping['J'])
    match=compare_solid_forest_fidelity(source_forest,forest,source[1],mesh[1],source_labels,labels,
                                      J,I,source_vertex_count=len(source[0]))
    record['candidate_to_source_components']=match.candidate_to_source.tolist()
    if measure:record['fidelity']=geometry.mapped_geometry(math_mesh(source),math_mesh(mesh),mapping)
    remaining();return labels,forest


def preflight_source(source,binary,sha,remaining,record,*,conditioning_version=1):
    """Full source F64/F32 gate before QEM; default8 collisions are diagnostic."""
    prepare=conditioning_function(conditioning_version)
    labels,forest=certify_arrays(source,binary,sha,remaining,record)
    chart=prepare(*source)
    require(chart.diagnostics['roundtrip_numerically_exact'],'Source fixed chart must roundtrip exactly')
    record['conditioning']=dict(chart.diagnostics);record['serialization']=dict(serialization_preflight(*source))
    if conditioning_version==2:
        record['conditioning_version']=2
        record['conditioning_header']=certificate.identity(CHART_V2_HEADER)
    record['float32_orientation']=float32_orientation(source);record['float32_certificate']={}
    matched_stage(source,labels,forest,(source[0].astype(np.float32),source[1]),identity_mapping(source),
                  binary,sha,remaining,record['float32_certificate'],measure=False)
    return labels,forest,chart


def compile_solid(source,qem_binary,query_binary,query_source_sha256,scratch,official_helper,*,metric_scale,
                  remaining=None,conditioning_version=1):
    """Return padded metric arrays and scalar proof; no automatic adoption.

    Source is full exact-welded F64/I64 geometry. `load_source` supplies raw GLB
    parity for production callers. Existing source rounded-key collisions are
    diagnostic, not healed: QEM must resolve them by legal actual collapses.
    Every stage uses exact embedding and a full birth-matched material forest.
    Caller supplies the ORIGINAL positive grounded scalar, never a fitted scale.
    """
    conditioning_function(conditioning_version)
    remaining=deadline(remaining);work=Path(scratch)
    report=dict(stage=f'oriented_solid_compiler_v{conditioning_version}',conditioning_version=conditioning_version,
        status='fail',phase='source',stages={},native_attempts=0,
        native_returned=False,geometry_repaired=False,components_deleted=False,orientation_changed=False,
        cost_backend_changed=False,adoption=False,reconstruction_accuracy_verified=False,ground_truth_used=False,
        budget_seconds=TOTAL_SECONDS,qem_seconds=QEM_SECONDS,query_seconds=QUERY_SECONDS)
    paths=tuple(Path(p) for p in (qem_binary,query_binary,official_helper,Path(__file__),certificate.__file__,
        geometry.__file__,endpoint.__file__,precision.__file__,compare_solid_forest_fidelity.__code__.co_filename,
        validate_mapping.__code__.co_filename,Path(certificate.__file__).with_name('certified_solid_query.cpp'),
        serialized_geometry.__file__,export_helper.__file__,volume_helper.__file__,chart_helper.__file__,
        serialization_helper.__file__,forest_helper.__file__))
    if conditioning_version==2:
        paths+=tuple(Path(p) for p in (chart_v2_helper.__file__,CHART_V2_HEADER,chart_v2_source.__file__))
    before=None;source_hashes=None
    try:
        require(type(metric_scale) is float and np.isfinite(metric_scale) and metric_scale>0,'Original positive grounded scalar required')
        require(work.is_absolute() and work.resolve()==work and work.is_dir() and not any(work.iterdir()),'Fresh caller scratch required')
        before={str(p):certificate.identity(p) for p in paths};report['artifacts_before']=before
        require(before[str(Path(official_helper).resolve())]['sha256']==endpoint.BUDGET_HELPER_SHA,
                'Original unmodified single official budget helper required')
        report['legacy_sources']=geometry.validate_legacy_sources()
        require(certificate.identity(Path(certificate.__file__).with_name('certified_solid_query.cpp'))['sha256']==query_source_sha256,
                'Current exact-query source differs from qualified source')
        source=tuple(np.asarray(a) for a in source)
        require(source[0].dtype==np.float64 and source[1].dtype==np.int64,'Represented F64/I64 source required')
        source_hashes=array_hashes(source);report['source_array_sha256']=source_hashes
        stage=report['stages'].setdefault(STAGES[0],{})
        labels,forest,chart=preflight_source(source,query_binary,query_source_sha256,remaining,stage,
                                          conditioning_version=conditioning_version)
        report['phase']='native';a,b,m=(work/n for n in ('input.obj','candidate.obj','mapping.json'))
        geometry.write_obj(a,*source);input_pin=certificate.identity(a);report['native_input']=input_pin
        report['native_attempts']=1
        native_start=time.monotonic()
        try:
            child=subprocess.run([str(qem_binary),str(a),str(b),str(m)],capture_output=True,check=False,
                                 timeout=min(QEM_SECONDS,remaining()))
        finally:
            report['native_elapsed_seconds']=time.monotonic()-native_start
            require(certificate.identity(a)==input_pin,'Native source OBJ modified')
        report.update(native_returned=True,native_returncode=child.returncode)
        require(certificate.identity(a)==input_pin,'Native source OBJ modified')
        if child.returncode:report['native_stderr_tail']=child.stderr[-300:].decode(errors='replace')
        require(child.returncode==0,'Cached native QEM failed; no retry')
        report['native_artifacts']={p.name:certificate.identity(p) for p in (a,b,m)}
        candidate=geometry.read_obj(b);document=certificate.strict_json(m.read_bytes())
        mapping,native=validate_conditioned_mapping(source,candidate,document,chart,
            conditioning_version=conditioning_version);report['native_mapping']=native
        report['phase']=STAGES[1];stage=report['stages'].setdefault(STAGES[1],{})
        matched_stage(source,labels,forest,candidate,mapping,query_binary,query_source_sha256,remaining,stage)
        require(serialization_preflight(*candidate)['position_weld_admissible'],'Actual candidate cannot surviveF32/default8')
        import trimesh  # Actual qualified Azure image only; no local installation.
        report['trimesh_version']=trimesh.__version__;glb=work/'candidate.glb'
        trimesh.Trimesh(*candidate,process=False).export(glb);raw=endpoint._load_mesh(glb)
        require(np.array_equal(raw[0],raw[0].astype(np.float32).astype(np.float64)),'GLB is not exactF32 promotion')
        exported=endpoint.exact_weld(*raw)[:2];emap=export_birth_mapping(candidate,exported,mapping)
        report['phase']=STAGES[2];stage=report['stages'].setdefault(STAGES[2],{})
        matched_stage(source,labels,forest,exported,emap,query_binary,query_source_sha256,remaining,stage)
        report['glb_identity']=certificate.identity(glb)
        processed=trimesh.Trimesh(vertices=exported[0],faces=exported[1],process=True)
        welded=endpoint.exact_weld(np.asarray(processed.vertices,np.float64),np.asarray(processed.faces,np.int64))[:2]
        require(precision.triangle_hash(exported[0][exported[1]])==precision.triangle_hash(welded[0][welded[1]]),
                'Actual default8 processing changed meaningful oriented triangles')
        # Candidate default8 keys must be injective; exact seams were already welded.
        keys=_weld_keys(exported[0]);require(len(np.unique(keys,axis=0))==len(exported[0]),'Nonexact default8 merge')
        wmap=reordered_mapping(exported,welded,emap)
        report['phase']=STAGES[3];stage=report['stages'].setdefault(STAGES[3],{})
        matched_stage(source,labels,forest,welded,wmap,query_binary,query_source_sha256,remaining,stage)
        spec=importlib.util.spec_from_file_location('wr_whole_solid_official_helper',official_helper)
        helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
        pv,pf=helper.budget_mesh(str(glb),faces=4096,vertices=4096)
        compact,pack=geometry.verify_pack_fidelity(exported,pv,pf);pmap=packed_mapping(exported,compact,emap)
        report['phase']=STAGES[4];stage=report['stages'].setdefault(STAGES[4],{})
        matched_stage(source,labels,forest,compact,pmap,query_binary,query_source_sha256,remaining,stage)
        metric=pv*metric_scale;metric_source=(source[0]*metric_scale,source[1])
        require(np.isfinite(metric).all(),'Original metric scalar overflow')
        expected=(compact[0].astype(pv.dtype)*metric_scale,compact[1])
        baked,_=geometry.verify_pack_fidelity(expected,metric,pf)
        # Establish the full physical source forest at the same scale, not infer it from a candidate.
        metric_record={}
        ml,mforest=matched_stage(source,labels,forest,metric_source,identity_mapping(source),query_binary,
                                query_source_sha256,remaining,metric_record,measure=False)
        report['metric_source_certificate']=metric_record
        report['phase']=STAGES[5];stage=report['stages'].setdefault(STAGES[5],{})
        matched_stage(metric_source,ml,mforest,baked,pmap,query_binary,query_source_sha256,remaining,stage)
        require(tuple(report['stages'])==STAGES,'Whole six-stage pipeline incomplete')
        report.update(status='pass',phase='complete',official_pack_fidelity=pack,metric_scale_baked_once=metric_scale,
                      output_vertices=4096,output_faces=4096,metric_scale_accuracy_verified=False,
                      frame_poses_changed=False)
    except Exception as error:
        report.update(failure_type=type(error).__name__,failure_reason=str(error)[-300:],
                      failure_scope='deadline' if isinstance(error,(TimeoutError,subprocess.TimeoutExpired)) else 'geometry_or_contract')
    finally:
        try:
            require(before is not None and {str(p):certificate.identity(p) for p in paths}==before,'Compiler artifact changed')
            require(source_hashes is not None and array_hashes(source)==source_hashes,'Source arrays changed')
            require(geometry.validate_legacy_sources()==report['legacy_sources'],'Inherited math helpers changed')
            for name,pin in report.get('native_artifacts',{}).items():require(certificate.identity(work/name)==pin,'Native artifact changed')
            if 'glb_identity' in report:require(certificate.identity(work/'candidate.glb')==report['glb_identity'],'Candidate GLB changed')
            report['artifacts_after']=before;report['source_arrays_unchanged']=True;remaining()
        except Exception as error:report.update(status='fail',posthash_failure_type=type(error).__name__)
    if report['status']!='pass':raise SolidCompilerError(report)
    return (metric,pf),report


def geometry_controls(qem_binary,query_binary,query_source_sha256,scratch,official_helper,*,controls,
                      conditioning_version=1):
    """Four NEW curved controls, one immutable total1800s budget, no reroll."""
    conditioning_function(conditioning_version)
    remaining=deadline();root=Path(scratch);work=root/'oriented-solid-controls'
    require(root.is_absolute() and root.resolve()==root and root.is_dir() and not work.exists(),'Fresh caller controls scratch')
    work.mkdir(mode=0o700);owner=work.stat();report=dict(stage=f'oriented_solid_compiler_controls_v{conditioning_version}',
        conditioning_version=conditioning_version,status='fail',
        controls=[],adoption=False,reconstruction_accuracy_verified=False,maximum_native_calls=4,budget_seconds=TOTAL_SECONDS)
    error=None;sources=()
    try:
        sources=tuple((name,mesh,metadata) for name,mesh,metadata in controls)
        require(len(sources)==4,'Exactly four frozen caller controls required')
        require(len({n for n,_,_ in sources})==4 and all(type(n) is str and n and
                all(c.isascii() and (c.isalnum() or c=='_') for c in n) for n,_,_ in sources),'Bounded control directory names required')
        for name,source,metadata in sources:
            require(len(source[1])>4096,'Control must require actual reduction')
            require(tuple(array_hashes(source))==metadata['source_array_sha256'],'Frozen procedural source changed')
            row=dict(control=name,status='fail',source_preflight={});report['controls'].append(row)
            sl,sforest,_=preflight_source(source,query_binary,query_source_sha256,remaining,row['source_preflight'],
                                        conditioning_version=conditioning_version)
            require(np.array_equal(sl,metadata['face_components']) and
                    np.array_equal(sforest.parents,metadata['expected_parents']) and
                    np.array_equal(sforest.signs,metadata['expected_signs']), 'Frozen procedural forest differs before any QEM')
        for (name,source,metadata),row in zip(sources,report['controls']):
            directory=work/name;directory.mkdir(mode=0o700)
            try:
                _,result=compile_solid(source,qem_binary,query_binary,query_source_sha256,directory,official_helper,
                                      metric_scale=metadata['metric_scale'],remaining=remaining,
                                      conditioning_version=conditioning_version)
                physical=result['stages'][STAGES[0]]['forest']
                require(physical['parents']==metadata['expected_parents'].tolist() and
                        physical['signs']==metadata['expected_signs'].tolist(),'Procedural expected full forest differs')
                require(result['native_mapping']['serialization']['committed_collapses']>0,'Control did not perform actualcollapse')
                row.update(result)
            except SolidCompilerError as exc:row.update(exc.report);raise
        report.update(status='pass',phase='complete')
    except Exception as exc:error=exc;report['failure_type']=type(exc).__name__
    finally:
        try:
            require(all(tuple(array_hashes(source))==metadata['source_array_sha256'] for _,source,metadata in sources),
                    'Frozen whole procedural source changed')
            require(work.stat().st_ino==owner.st_ino and work.stat().st_dev==owner.st_dev,'Owned controls scratch replaced')
            for p in work.rglob('*'):
                require(not p.is_symlink() and (p.is_dir() or p.name in FILES and p.stat().st_nlink==1),'Unknown scratch artifact; refuse deletion')
            shutil.rmtree(work);report['owned_scratch_removed']=True;remaining()
        except Exception as exc:error=error or exc;report.update(status='fail',cleanup_failure_type=type(exc).__name__)
    if error:raise SolidCompilerError(report) from error
    return report
