"""Public full48 saved endpoint context; no model, job or private input reader.

Host authenticates whole Git/old endpoint lineage; native sees only a fresh
pinned public projection and current native helper pins. No model or future PASS.
"""
from dataclasses import dataclass
from pathlib import Path
import stat

import vcoco_fit_cal_endpoint_observations as endpoint

rt=endpoint.rt
COUNT=48
BANK_FIELDS=frozenset(('image_id','original_frame_index','bank_index','image_size','input_file','input_identity',
    'person_native_queries','person_postprocessor_rows','person_retained_rows','person_ids','owl_patches',
    'original_slot','acquired_ordinal','file','identity','arrays'))
HELPERS=tuple(dict.fromkeys(('infra/vcoco_public_observation_context.py',
    'infra/vcoco_cardinality_observation_adapters.py','infra/vcoco_fit_cal_endpoint_observations.py',
    *endpoint.original.NATIVE_FILES,*endpoint.REUSE,
    'src/world_reward/person_pose_observations.py','src/world_reward/hoi_detr_observations.py')))


@dataclass(frozen=True)
class PublicObservationContext:
    inputs: Path
    endpoint_banks: Path
    output: Path

    def __post_init__(self):
        paths=[]
        for name in ('inputs','endpoint_banks','output'):
            path=rt.canonical(getattr(self,name));object.__setattr__(self,name,path);paths.append(path)
        rt.require(all(a!=b and a not in b.parents and b not in a.parents
            for i,a in enumerate(paths)for b in paths[i+1:]),'Injective separate public namespaces required')


def _state(path):
    s=rt.canonical(path).lstat()
    return tuple(getattr(s,k)for k in('st_dev','st_ino','st_mode','st_size','st_nlink','st_uid','st_gid','st_mtime_ns','st_ctime_ns'))


def source_identity(code,current_source):
    """Actual helper origins/pins, not independent whole-Git verification."""
    code=rt.canonical(code);endpoint.source_identity(code,current_source)
    rt.require(Path(__file__).resolve()==code/HELPERS[0]
        and all(current_source['helpers'][n]==rt.identity(code/n,2<<20)for n in HELPERS),
        'Explicit current source helper binding differs')


def authenticate_endpoint(context,*,code,current_source,endpoint_source,report_pin,deadline):
    """HOST ONLY: authenticate old full48 PASS and return a public projection.

    `endpoint_source` is the caller's independently authenticated ORIGINAL
    whole-source binding, distinct from `current_source`. Neither is derived
    from unpinned receipt claims. No old assets, acquisition or roles are read.
    """
    rt.require(type(context)is PublicObservationContext,'Explicit public context required')
    source_identity(code,current_source);endpoint.check(deadline)
    out=context.endpoint_banks;report=rt.pinned(out/'report.json',report_pin,2<<20)
    rt.require(report['schema']==endpoint.SCHEMA and report['stage']=='full48_endpoint_observations_host'
        and report['status']=='pass'and report['source_binding']==endpoint_source
        and report['producer_revision']==endpoint_source['producer_revision']and report['image_id']==endpoint.original.IMAGE
        and report['acquired_images']==48 and report['native_exit_status']==0
        and report['outputs_sealed']is report['owned_cleanup_verified']is report['source_inputs_assets_runtime_rehashed_after']is True
        and all(report[k]is False for k in('reference_metadata_read','split_metadata_read','FIT_performed',
            'ground_truth_used','challenge_inputs_used','actor_selection_performed','ownership_verified','quality_verified','adoption')),
        'Independently pinned complete48 endpoint host PASS required')
    native=rt.pinned(out/'native.json',report['native_report_identity'],2<<20)
    proof=rt.pinned(out/'proof.json',native['proof_identity'],2<<20)
    rt.require(proof['source']==endpoint_source and proof['inputs_identity']==report['public_inputs_identity']
        and proof['images']==48 and proof['profile']==endpoint.PROFILE and proof['image_id']==endpoint.original.IMAGE
        and native['schema']==endpoint.SCHEMA and native['status']=='pass'and native['phase']=='complete'
        and native['stage']=='native_full48_endpoint_observations'and native['producer_revision']==report['producer_revision']
        and native['image_id']==proof['image_id']and native['profile']==endpoint.PROFILE
        and native['runtime_identity']==proof['owl_runtime']
        and type(native['model_loads'])is int and native['model_loads']==2
        and native['models_released']is native['source_inputs_assets_runtime_rehashed_after']is True
        and all(type(native[k])is int and native[k]==48 for k in('person_forward_calls','image_embed_calls','objectness_calls','box_calls'))
        and all(native[k]is False for k in('reference_metadata_read','split_metadata_read','FIT_performed',
            'ground_truth_used','challenge_inputs_used','actor_selection_performed','ownership_verified','quality_verified','adoption'))
        and native['images']==report['native_images'],'Original source/native/public binding differs')
    names={'report.json','native.json','proof.json','.container.cid'}|{f'image_{i:06d}.npz'for i in range(COUNT)}
    s=out.lstat();rt.require(stat.S_ISDIR(s.st_mode)and stat.S_IMODE(s.st_mode)==0o500
        and {p.name for p in out.iterdir()}==names,'Exact sealed full48 endpoint namespace required')
    rt.require(all(stat.S_IMODE(p.lstat().st_mode)==0o400 and p.lstat().st_uid==s.st_uid for p in out.iterdir()),
        'Original endpoint400 leaves required')
    inputs=endpoint.public_inputs(endpoint.EndpointContext(context.inputs,out),report['public_inputs_identity'])
    endpoint.validate_records(endpoint.EndpointContext(context.inputs,out),inputs,native['images'],deadline)
    return dict(schema='world_reward.public_endpoint_bank_reference.v1',inputs=inputs,
        manifest_identity=report['public_inputs_identity'],banks=[{k:r[k]for k in BANK_FIELDS}for r in native['images']])


@dataclass(frozen=True,eq=False)
class PublicBankReference:
    context: PublicObservationContext
    _projection: bytes
    _states: bytes
    _projection_path: Path
    _projection_identity: bytes
    _code: Path
    _native_source: bytes
    _native_mounts: bool

    def __post_init__(self):
        rt.require(type(self.context)is PublicObservationContext and type(self._native_mounts)is bool
            and all(type(getattr(self,n))is bytes for n in('_projection','_states','_projection_identity','_native_source')),
            'Immutable explicitly pinned public reference required')
        value=rt.pinned(self._projection_path,rt.strict(self._projection_identity),2<<20)
        rt.require(endpoint.original.encode(value)==self._projection and set(value)=={'schema','inputs','manifest_identity','banks'}
            and value['schema']=='world_reward.public_endpoint_bank_reference.v1'
            and type(value['banks'])is list and len(value['banks'])==48
            and all(type(r)is dict and set(r)==BANK_FIELDS for r in value['banks']),
            'Only public endpoint projection required')
        rt.require(type(rt.strict(self._native_source))is dict and set(rt.strict(self._native_source))=={'helpers'}
            and type(rt.strict(self._native_source)['helpers'])is dict
            and set(rt.strict(self._native_source)['helpers'])==set(HELPERS),
            'Only exact current native helper pins allowed; no host/private proof')
        source_identity(self._code,rt.strict(self._native_source));self._verify_states()

    @property
    def images(self):return rt.strict(self._projection)['inputs']['images']
    @property
    def banks(self):return rt.strict(self._projection)['banks']

    def _verify_states(self):
        names={str(self.context.inputs),str(self.context.endpoint_banks),str(self._projection_path)}
        names|={str(self.context.inputs/r['file'])for r in self.images}|{str(self.context.inputs/'manifest.json')}
        names|={str(self.context.endpoint_banks/r['file'])for r in self.banks}
        rt.require(set(rt.strict(self._states))==names and endpoint.original.encode({n:_state(n)for n in names})==self._states,
            'Public native source/file inode/modes changed')

    def verify(self,deadline):
        endpoint.check(deadline)
        value=rt.pinned(self._projection_path,rt.strict(self._projection_identity),2<<20)
        rt.require(endpoint.original.encode(value)==self._projection,'Pinned public projection changed')
        source_identity(self._code,rt.strict(self._native_source))
        inputs=endpoint.public_inputs(endpoint.EndpointContext(self.context.inputs,self.context.endpoint_banks),
            value['manifest_identity'],native_mounts=self._native_mounts)
        rt.require(inputs==value['inputs'],'Only original public48 RGB required')
        expected={'image_%06d.npz'%i for i in range(48)}
        rt.require({p.name for p in self.context.endpoint_banks.iterdir()}==expected,
            'Native endpoint directory contains only48 NPZ leaves')
        endpoint.validate_records(endpoint.EndpointContext(self.context.inputs,self.context.endpoint_banks),inputs,self.banks,deadline)
        self._verify_states();endpoint.check(deadline)


def public_bank_reference(context,*,projection_path,projection_pin,code,native_source,deadline,native_mounts=True):
    """Native public proof only: no old reports/proofs/assets/private metadata.

    Caller binds projection_pin in its fresh source-authenticated native proof.
    endpoint_banks is an individual-leaf projection directory (only48 NPZs),
    not the original producer's report directory. Native local states are newly
    observed; no synthetic bind-parent inode/mode is invented from the host.
    """
    rt.require(type(context)is PublicObservationContext,'Explicit public native context required')
    endpoint.check(deadline);source_identity(code,native_source);value=rt.pinned(projection_path,projection_pin,2<<20)
    banks=value['banks'];images=value['inputs']['images']
    rt.require(len(banks)==len(images)==48,'Every public endpoint bank/image required')
    names={str(context.inputs),str(context.endpoint_banks),str(projection_path)}
    names|={str(context.inputs/r['file'])for r in images}|{str(context.inputs/'manifest.json')}
    names|={str(context.endpoint_banks/r['file'])for r in banks}
    encode=endpoint.original.encode
    reference=PublicBankReference(context,encode(value),encode({n:_state(n)for n in names}),
        projection_path,encode(projection_pin),code,encode(native_source),native_mounts)
    reference.verify(deadline);return reference


def load_bank(reference,row,deadline):
    """One lossless original17 bank; numerical imports are lazy."""
    import numpy as np
    rt.require(type(reference)is PublicBankReference and row in reference.banks,'Original endpoint row required')
    endpoint.check(deadline);path=reference.context.endpoint_banks/row['file']
    rt.require(rt.identity(path,16<<20)==row['identity'],'Saved endpoint identity differs')
    with np.load(path,allow_pickle=False)as saved:
        rt.require(set(saved.files)==endpoint.KEYS,'Every original17 array required');arrays={n:saved[n]for n in saved.files}
    rt.require(endpoint.array_identities(arrays)==row['arrays'],'Saved endpoint array identity differs')
    endpoint.validate_arrays(arrays,reference.images[row['original_slot']]);return arrays
