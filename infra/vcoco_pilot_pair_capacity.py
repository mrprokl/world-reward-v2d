"""DEV-only saved V-COCO pair proposal ceilings, no selector or model calls."""
import argparse
import hashlib
import math
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import time

sys.path[:0] = [str(Path(__file__).resolve().parent),str(Path(__file__).resolve().parents[1]/'src')]
import vcoco_pilot_endpoint_bank as bank
import vcoco_role_prepare as prep
import coco_endpoint_evaluate as saved

rt,c=bank.rt,prep.c
ROOT,DATA,IMAGE=bank.ROOT,prep.DATA,bank.IMAGE
ENTRY='run_vcoco_pilot_pair_capacity'
CONFIG='configs/vcoco_pilot_pair_capacity_v1.json'
OUTPUT=ROOT/'results/vcoco-pilot-pair-capacity-v1'
PIN_NAMES=('host','native','acquisition','selection','cohort','public')
NATIVE_FILES=tuple(dict.fromkeys(('infra/vcoco_pilot_pair_capacity.py',CONFIG,
    'infra/coco_endpoint_evaluate.py','src/world_reward/vcoco_pair_retrieval.py',
    'src/world_reward/vcoco_role_reference.py',*bank.NATIVE_FILES,*prep.HELPERS)))
HELPERS=tuple(dict.fromkeys((*NATIVE_FILES,'infra/run_vcoco_pilot_pair_capacity.sh',*bank.HELPERS,*prep.HELPERS)))
FIXED=dict(schema='world_reward.vcoco_pilot_pair_capacity.v1',output=str(OUTPUT),image_id=IMAGE,
    slots=16,development_slots=8,reserved_slots=8,primary_iou=.5,pair_gate=.7,missing_slot_recall=0.,
    budget_seconds=180,outer_seconds=195,cpu_threads=4,memory_bytes=6 << 30,
    maximum_reference_bytes=4 << 20,maximum_bank_bytes=16 << 20,maximum_receipt_bytes=2 << 20,
    models_loaded=0,RGB_decoded=False,GPU_used=False,FIT_performed=False,selector_evaluated=False,
    RESERVED_reference_values_read=False,ownership_verified=False,quality_verified=False,adoption=False)
PRODUCERS=dict(bank=dict(revision='e1a01f303f533344a10ef0e4eb216c2eb13c9182',files=316,entries=321,
    closure_sha256='ddefd7f4663b15fee9dd8f6bb14693ea1f0d343e4407100b1152aafbcdc939ec'),
    prepare=dict(revision='a8da2da838dc4ef513e7819be4d25ece99488207',files=303,entries=308,
    closure_sha256='098811fa1eac670cfa081b8d5d10dcdc3b405ed4b9cc67356345e8de42b4ae0a'))
encode,error=bank.encode,bank.error


def check(deadline):
    if not math.isfinite(deadline) or time.monotonic()>=deadline:raise TimeoutError('Inclusive saved capacity deadline')


def paths():
    return dict(host=bank.OUTPUT/'host.json',native=bank.OUTPUT/'native.json',
        acquisition=DATA/'eval_private/report.json',selection=DATA/'metadata/report.json',
        cohort=DATA/'metadata/cohort.json',public=DATA/'inputs/manifest.json')


def configuration(code,source):
    p=rt.pinned(code/CONFIG,source['helpers'][CONFIG],32 << 10)
    rt.require(set(p)==set(FIXED)|{'helper_pins','producers'} and all(type(p[k])is type(v)and p[k]==v for k,v in FIXED.items())
        and p['producers']==PRODUCERS,'Exact frozen DEV-only ceiling policy required')
    expected=set(HELPERS)-{NATIVE_FILES[0],CONFIG,'infra/run_vcoco_pilot_pair_capacity.sh'}
    rt.require(set(p['helper_pins'])==expected and all(source['helpers'][n]==v for n,v in p['helper_pins'].items()),
        'Every original helper/config byte required')
    for module,name in ((bank,'infra/vcoco_pilot_endpoint_bank.py'),(prep,'infra/vcoco_role_prepare.py'),
        (saved,'infra/coco_endpoint_evaluate.py'),(rt,'infra/mediapipe_cpu_runtime_verify.py')):
        rt.require(Path(module.__file__).resolve()==code/name,'Actual imported source origin differs')
    return p


def original_source(code,label,revisions,module):
    spec=PRODUCERS[label];revision=revisions[label]
    rt.require(revision==spec['revision'],'Only audited original producer permitted')
    old=ROOT/'jobs'/revision/module.ENTRY/'code';source=rt.source(ROOT,old,revision,module.ENTRY,module.HELPERS)
    rt.require(source['entries']==spec['entries'] and source['closure_sha256']==spec['closure_sha256']
        and sum(p.is_file()for p in old.rglob('*'))==spec['files'],'Whole original Git source differs')
    for name in module.HELPERS:rt.require(rt.identity(code/name,2 << 20,empty=True)==source['helpers'][name],
        'Reused source must equal original producer')
    return old,source


def structure(values,pins,proof):
    selection,cohort,acq,public=(values[n]for n in ('selection','cohort','acquisition','public'))
    rt.require(selection['status']=='pass'and selection['phase']=='select'and selection['input_proof']==proof
        and selection['source_binding']==proof['source_binding']and selection['cohort_identity']==pins['cohort']
        and selection['source_and_inputs_rehashed_after']is selection['outputs_sealed']is True,
        'Original sealed selection required')
    rt.require(acq['status']=='pass'and acq['phase']=='acquire'and acq['input_proof']==proof
        and acq['source_binding']==proof['source_binding']and acq['producer_revision']==PRODUCERS['prepare']['revision']
        and acq['selection_report_identity']==pins['selection']and acq['cohort_identity']==pins['cohort']
        and acq['public_inputs_identity']==pins['public']and acq['availability_gate_passed']is True
        and acq['source_and_inputs_rehashed_after']is acq['outputs_sealed']is acq['public_outputs_sealed']is True
        and acq['decision']=='READY_PENDING_SEPARATE_RGB_BANK'and acq['selected_slots']==16,
        'Original complete no-replacement acquisition required')
    rows=cohort['records'];status=acq['records'];mapping=acq['public_mappings']
    rt.require(len(rows)==len(status)==16 and [r['slot']for r in rows]==list(range(16))
        and [r['split']for r in rows]==['DEV']*8+['RESERVED']*8
        and cohort['source_binding']==proof['source_binding']and cohort['producer_revision']==PRODUCERS['prepare']['revision']
        and cohort['namespace']==prep.NAMESPACE and cohort['retry_count']==cohort['replacement_count']==0,
        'Every original frozen16 slot required')
    acquired=[r for r in status if r['status']=='acquired']
    rt.require(len(acquired)==len(mapping)==len(public['images'])and set(public)=={'schema','images'}
        and public['schema']=='world_reward.rgb_proposal_inputs.v1','Complete public-only projection required')
    for r,original in zip(status,rows):rt.require(r['slot']==original['slot']and r['split']==original['split']
        and r['image_id']==original['image_id']and r['status']in('acquired','unavailable'),'Missing slot ledger changed')
    for r,m,image in zip(acquired,mapping,public['images']):
        original=rows[r['slot']];opaque=hashlib.sha256((prep.NAMESPACE+f"{r['image_id']:012d}").encode()).hexdigest()[:32]
        rt.require(m==dict(slot=r['slot'],public_image_id=opaque,public_file=f"image_{r['slot']:06d}.jpg",image_pin=r['image_pin'])
            and image==dict(image_id=opaque,file=m['public_file'],**r['image_pin'],width=original['image']['width'],height=original['image']['height'])
            and len(image)==6,'Original opaque slot/grid/byte mapping differs')
    counts=dict(slots=16,acquired=len(acquired),missing=16-len(acquired),acquired_DEV=sum(r['split']=='DEV'for r in acquired),
        acquired_RESERVED=sum(r['split']=='RESERVED'for r in acquired))
    rt.require(acq['counts']==counts and counts['acquired_DEV']>=6 and counts['acquired_RESERVED']>=6,
        'Original availability gate required, never denominator repair')
    return {m['slot']:m for m in mapping}


def authenticate(code,source,pins,revisions):
    values={n:rt.pinned(path,pins[n],2 << 20)for n,path in paths().items()}
    old,prepare_source=original_source(code,'prepare',revisions,prep)
    prepare_cfg=rt.pinned(code/prep.CONFIG,prepare_source['helpers'][prep.CONFIG],65536)
    rt.require(prepare_cfg['schema']=='world_reward.vcoco_role_prepare_config.v1'and prepare_cfg['output']==str(DATA)
        and prepare_cfg['select']==prep.SELECT and prepare_cfg['acquire']==prep.ACQUIRE,'Original frozen pilot policy required')
    spec=prepare_cfg['census'];old_code=ROOT/'jobs'/spec['producer_revision']/c.ENTRY/'code'
    census_source=rt.source(ROOT,old_code,spec['producer_revision'],c.ENTRY,c.HELPERS)
    rt.require(census_source['entries']==spec['entries']and census_source['closure_sha256']==spec['closure_sha256']
        and sum(p.is_file()for p in old_code.rglob('*'))==spec['files']
        and (old_code.parent/'source-sha256').read_bytes()==(spec['archive_xz_sha256']+'\n').encode()
        and Path(spec['configuration']['path'])==old_code/c.CONFIG and spec['configuration']['pin']==census_source['helpers'][c.CONFIG],
        'Whole original census source/config required')
    census_cfg=c.configuration(rt.pinned(Path(spec['configuration']['path']),spec['configuration']['pin'],65536))
    census_binding=c.authenticate(census_cfg,old_code,spec['producer_revision'])
    rt.require(Path(spec['report']['path'])==c.DATA/'report.json'and spec['report']['pin']==
        dict(bytes=20320,sha256='79142e6a49d11e3570ff28a41cb09b86056769052823e30a6b25b29712a37975'),'Actual census report pin required')
    census_report=rt.pinned(Path(spec['report']['path']),spec['report']['pin'],300000)
    rt.require(census_report['status']=='pass'and census_report['source_binding']==census_binding
        and census_report['outputs_sealed']is census_report['source_and_inputs_rehashed_after']is True,
        'Original source-bound census receipt required')
    rt.require(stat.S_IMODE(c.DATA.lstat().st_mode)==0o500 and c.DATA.lstat().st_uid==0
        and {p.name for p in c.DATA.iterdir()}=={'report.json'},'Original census readonly output required')
    states=c.pin(c.encode({str(p.relative_to(old)):c.state(p)for p in(old,*sorted(old.rglob('*')))}))
    prepare_proof=dict(source_binding=prepare_source,census_binding=census_binding,source_stat_identity=states,
        census_report_identity=spec['report']['pin'],census_report_state=c.state(Path(spec['report']['path'])),census_directory_state=c.state(c.DATA))
    mapping=structure(values,pins,prepare_proof)
    acq=values['acquisition'];rt.require(stat.S_IMODE((DATA/'eval_private').lstat().st_mode)==0o500
        and (DATA/'eval_private').lstat().st_uid==0 and {p.name for p in(DATA/'eval_private').iterdir()}=={'report.json'}
        and all(acq[k]is False for k in ('RGB_decoded','GPU_used','models_loaded','FIT_performed','predictions_read','selection_performed','fresh_reference_values_consulted')),
        'Private original acquisition namespace and no-reference semantics required')
    _,reference_states=prep.frozen(DATA/'metadata',pins['selection'],pins['cohort'],prepare_proof,lambda:None)
    _,bank_source=original_source(code,'bank',revisions,bank)
    recipe,original_recipe=bank.configuration(code,source);prior=bank.original.qualifications(code,original_recipe,live=True)
    host,native=values['host'],values['native'];proof_pin=native['proof_identity']
    rt.require(host['status']=='pass'and host['stage']=='vcoco_pilot_endpoint_bank_host'and host['schema']==recipe['schema']
        and host['producer_revision']==revisions['bank']and host['source_binding']==bank_source
        and host['original_qualification']==prior and host['original_recipe_identity']==recipe['original_recipe']
        and host['public_inputs_identity']==pins['public']and host['native_report_identity']==pins['native']and host['native_exit_status']==0
        and host['native_images']==native['images']and host['owned_cleanup_verified']is host['outputs_sealed']is host['source_inputs_runtime_assets_rehashed_after']is True,
        'Actual complete source-bound bank host PASS required')
    bank.validate_report(native,recipe,revisions['bank'],proof_pin,values['public'])
    bank_proof=rt.pinned(bank.OUTPUT/'proof.json',proof_pin,2 << 20)
    rt.require(bank_proof['source']==bank_source and bank_proof['native_files']=={n:bank_source['helpers'][n]for n in bank.NATIVE_FILES}
        and bank_proof['inputs_identity']==pins['public']and bank_proof['images']==len(values['public']['images'])
        and bank_proof['owl_runtime']==prior['owl']['native_runtime']==native['runtime_identity']and bank_proof['image_id']==IMAGE,
        'Original native bank proof/runtime differs')
    owl_policy=rt.pinned(code/bank.owl.CONFIG,source['helpers'][bank.owl.CONFIG],32 << 10)
    rt.require(bank_proof['assets']==bank.original.asset_files(prior,owl_policy),'All original proof assets required')
    rt.require(bank.public_inputs(pins['public'],len(values['public']['images']))==values['public'],'Public RGB original hashes differ')
    frozen={str(path):pins[n]for n,path in paths().items()}
    for path in paths().values():rt.require(stat.S_IMODE(path.lstat().st_mode)==0o400 and path.lstat().st_uid==0,'Original readonly root-owned receipt required')
    frozen[str(bank.OUTPUT/'proof.json')]=proof_pin
    frozen[str(bank.OUTPUT/'.container.cid')]=rt.identity(bank.OUTPUT/'.container.cid',65)
    for row in native['images']:
        path=bank.OUTPUT/row['file'];rt.require(rt.identity(path,16 << 20)==row['identity'],'Whole frozen NPZ differs');frozen[str(path)]=row['identity']
    rt.require(stat.S_IMODE(bank.OUTPUT.lstat().st_mode)==0o500 and bank.OUTPUT.lstat().st_uid==0,'Original sealed bank directory required')
    rt.require({p.name for p in bank.OUTPUT.iterdir()}=={Path(n).name for n in frozen if Path(n).parent==bank.OUTPUT},'Exact original bank output inventory required')
    return dict(values=values,pins=pins,revisions=revisions,sources=dict(bank=bank_source,prepare=prepare_source),
        mapping=mapping,frozen=frozen,reference_states=reference_states,original_qualification=prior,
        input_states={str(path):c.state(path)for path in (DATA,DATA/'inputs',DATA/'eval_private',bank.OUTPUT,*[Path(n)for n in frozen])})


def evaluate(binding,banks,load_reference):
    import numpy as np
    from world_reward.vcoco_role_reference import parse_vcoco_role_reference
    from world_reward.vcoco_pair_retrieval import evaluate_vcoco_pair_retrieval
    results=[]
    for record in binding['values']['cohort']['records'][:8]:
        slot=record['slot'];rt.require(record['split']=='DEV'and slot<8,'Only DEV reference semantics allowed')
        if slot not in binding['mapping']:
            results.append(dict(slot=slot,missing=True,pair_proposal_recall=0.,person_proposal_recall=0.,object_proposal_recall=0.));continue
        row=next(r for r in binding['values']['native']['images']if r['original_slot']==slot)
        p,objects=banks[slot];value=load_reference(record)
        rt.require(type(value)is dict and set(value)=={'schema','image','instances','actions','localized_positive_pairs'}
            and value['schema']=='world_reward.vcoco_pilot_reference.v1'
            and value['image']=={k:record['image'][k]for k in('id','width','height')},'Exact original same-image reference projection required')
        reference=parse_vcoco_role_reference(value['actions'],value['instances'],[value['image']])
        actual=[dict(image_id=x.image_id,agent_annotation_id=x.agent_annotation_id,object_annotation_id=x.object_annotation_id,
            row_role_references=x.row_role_references)for x in reference.localized_positive_pairs]
        rt.require(encode(actual)==encode(value['localized_positive_pairs'])and len(actual)>=1,'Every original localized positive pair required')
        o=objects.boxes_original_xyxy;support=np.all(p[:,2:]>p[:,:2],axis=1)[:,None]&np.all(o[:,2:]>o[:,:2],axis=1)[None,:]
        scores=np.full(support.shape,np.nan,np.float64);scores[support]=0.
        result=evaluate_vcoco_pair_retrieval(p,o,scores,support,reference,value['instances'],record['image_id'],
            (record['image']['height'],record['image']['width']),tuple(row['person_ids']),tuple(f"owl:patch:{i:06d}"for i in objects.patch_ids))
        results.append(dict(slot=slot,missing=False,pair_proposal_recall=result.positive_pair_proposal_recall,
            person_proposal_recall=result.positive_person_proposal_recall,object_proposal_recall=result.positive_object_proposal_recall,
            supported_pair_proposal_recall=result.supported_positive_pair_recall,localized_positive_pairs=result.localized_positive_pair_count,
            proposal_persons=len(p),proposal_objects=len(o),role_diagnostics=dict(result.role_diagnostics)))
    metrics=dict(slots=8,iou=.5,pair_gate=.7,missing_slot_recall=0.,
        pair_macro_fixed_slots=math.fsum(r['pair_proposal_recall']for r in results)/8,
        person_macro_fixed_slots=math.fsum(r['person_proposal_recall']for r in results)/8,
        object_macro_fixed_slots=math.fsum(r['object_proposal_recall']for r in results)/8)
    metrics['pair_gate_passed']=metrics['pair_macro_fixed_slots']>=.7
    return dict(metrics=metrics,per_image=results,decision=decision(metrics))


def decision(metrics):
    rt.require(metrics['slots']==8 and metrics['iou']==.5 and metrics['pair_gate']==.7 and metrics['missing_slot_recall']==0.
        and all(type(metrics[k])is float and math.isfinite(metrics[k])and 0<=metrics[k]<=1 for k in
            ('pair_macro_fixed_slots','person_macro_fixed_slots','object_macro_fixed_slots'))
        and metrics['pair_gate_passed']is(metrics['pair_macro_fixed_slots']>=.7),'Frozen fixed8 capacity decision differs')
    return 'DEV_PAIR_PROPOSAL_CAPACITY_PASS_PENDING_OBSERVATIONS'if metrics['pair_gate_passed']else'CLOSED_DEV_PAIR_PROPOSAL_CAPACITY_NO_DWPOSE_HOI_OR_FIT'


def publish(path,report,deadline,owner,*,seal=False):
    current=rt.canonical(path.parent).lstat()
    rt.require((current.st_dev,current.st_ino,current.st_uid,current.st_gid)==tuple(owner) and stat.S_IMODE(current.st_mode)==0o700,'Original owned output namespace required')
    before={p.name:rt.identity(p,2 << 20,readonly=p.name!='.container.cid')for p in path.parent.iterdir()}
    rt.require(set(before)<={'proof.json','.container.cid','native.json'},'Only owned scalar output files required')
    with path.open('x+b')as stream:
        os.fchmod(stream.fileno(),0o400);inode=os.fstat(stream.fileno())
        def update():
            stream.seek(0);stream.write(encode(report));stream.truncate();stream.flush();os.fsync(stream.fileno())
        try:
            if seal:
                for leaf in path.parent.iterdir():
                    info=rt.canonical(leaf).lstat();rt.require(stat.S_ISREG(info.st_mode)and info.st_nlink==1 and info.st_uid==owner[2],'Only original owned scalar leaves');leaf.chmod(0o400)
                path.parent.chmod(0o500)
            prep.acq.sync_directory(path.parent);prep.acq.sync_directory(path.parent.parent)
            if report['status']!='pass':report['decision']='CLOSED_TECHNICAL_EVALUATION'
            report['outputs_sealed']=seal;update()
            now=path.lstat();current=path.parent.lstat()
            rt.require((now.st_dev,now.st_ino)==(inode.st_dev,inode.st_ino)
                and(current.st_dev,current.st_ino,current.st_uid,current.st_gid)==tuple(owner)
                and {p.name for p in path.parent.iterdir()}==set(before)|{path.name}
                and all(rt.identity(path.parent/n,2 << 20,readonly=n!='.container.cid' or seal)==v for n,v in before.items())
                and rt.identity(path,2 << 20)==dict(bytes=len(encode(report)),sha256=hashlib.sha256(encode(report)).hexdigest()),
                'Owned scalar publication changed');check(deadline)
        except BaseException as exc:report.update(status='fail',decision='CLOSED_TECHNICAL_PUBLICATION',publication_error_type=error(exc));update()


def cpu(code,revision,proof_pin,deadline):
    started=time.monotonic();proof=rt.pinned(OUTPUT/'proof.json',proof_pin,2 << 20);binding=proof['binding']
    report=dict(schema=FIXED['schema'],stage='saved_cpu_vcoco_pair_proposals',status='fail',phase='authentication',
        producer_revision=revision,proof_identity=proof_pin,inputs=binding['pins'],models_loaded=0,GPU_used=False,
        RGB_decoded=False,FIT_performed=False,selector_evaluated=False,RESERVED_reference_values_read=False,
        ownership_verified=False,quality_verified=False,adoption=False,all16_banks_validated_before_DEV=False,all_acquired_banks_validated_before_DEV=False,reference_files_decoded=0)
    try:
        rt.require(os.environ['WR_IMAGE_ID']==IMAGE and {p.name for p in Path('/sys/class/net').iterdir()}=={'lo'}
            and set(proof['native_files'])==set(NATIVE_FILES)and {str(p.relative_to(code))for p in code.rglob('*')if p.is_file()}==set(NATIVE_FILES),
            'Only offline approved CPU sources allowed')
        rt.require(proof['image_id']==IMAGE and proof['source']['producer_revision']==revision and proof['native_files']=={n:proof['source']['helpers'][n]for n in NATIVE_FILES},'Actual current source projection required')
        configuration(code,proof['source']);bank.configuration(code,proof['source'])
        owl_policy=rt.pinned(code/bank.owl.CONFIG,proof['native_files'][bank.owl.CONFIG],32 << 10)
        runtime=bank.owl.installed(owl_policy);rt.require(runtime==binding['original_qualification']['owl']['native_runtime'],
            'Original native CPU package/source RECORD required')
        grounding=bank.original.installed_grounding(runtime['wheel_RECORD_identities']['transformers'])
        report['runtime_identity']=runtime;report['grounding_source_identity']=grounding
        for name,pin in proof['native_files'].items():rt.require(rt.identity(code/name,2 << 20,empty=True)==pin,'Mounted source changed')
        for path,pin in proof['frozen'].items():rt.require(rt.identity(Path(path),16 << 20)==pin,'All frozen receipts/banks required')
        rt.require({p.name for p in(DATA/'metadata').iterdir()}=={'report.json','cohort.json'}|{f'reference_{i:06d}.json'for i in range(8)},
            'No RESERVED reference may be mounted')
        bank.validate_report(binding['values']['native'],rt.pinned(code/bank.CONFIG,proof['native_files'][bank.CONFIG],32 << 10),
            binding['revisions']['bank'],binding['values']['native']['proof_identity'],binding['values']['public'])
        banks={}
        for row in binding['values']['native']['images']:
            check(deadline);banks[row['original_slot']]=saved.validate_npz(bank.OUTPUT/row['file'],row)
        rt.require(set(banks)==set(int(k)for k in binding['mapping']),'Every acquired full bank required before any reference')
        report['all_acquired_banks_validated_before_DEV']=True;report['validated_bank_count']=len(banks)
        report['all16_banks_validated_before_DEV']=len(banks)==16;report['phase']='DEV_only'
        binding['mapping']={int(k):v for k,v in binding['mapping'].items()}
        def load(record):
            check(deadline);rt.require(record['slot']<8 and record['reference_file']==f"reference_{record['slot']:06d}.json",'DEV only')
            report['reference_files_decoded']+=1
            return rt.pinned(DATA/'metadata'/record['reference_file'],record['reference_identity'],4 << 20)
        report.update(evaluate(binding,banks,load));report.update(status='pass',phase='complete')
    except BaseException as exc:report.update(status='fail',decision='CLOSED_TECHNICAL_EVALUATION',error_type=error(exc))
    finally:
        try:
            for path,pin in proof['frozen'].items():rt.require(rt.identity(Path(path),16 << 20)==pin,'Frozen scalar/bank/reference changed')
            for name,pin in proof['native_files'].items():rt.require(rt.identity(code/name,2 << 20,empty=True)==pin,'CPU helper source changed')
            if 'runtime'in locals():rt.require(bank.owl.installed(owl_policy)==runtime and bank.original.installed_grounding(runtime['wheel_RECORD_identities']['transformers'])==grounding,'Native CPU installed source changed')
            report['source_inputs_predictions_references_rehashed_after']=True;check(deadline)
        except BaseException as exc:report.update(status='fail',decision='CLOSED_TECHNICAL_POSTHASH',post_error_type=error(exc))
    report['elapsed_seconds']=time.monotonic()-started;publish(OUTPUT/'native.json',report,deadline,proof['output_owner']);return report


def cleanup(path,name,revision,deadline):
    if not path.exists():
        rt.require(not bank.command(['docker','ps','-aq','--no-trunc','--filter','name=^/'+name+'$'],deadline),'CPU survives without CID');return
    rt.identity(path,65,readonly=False);raw=path.read_bytes();rt.require(re.fullmatch(b'[0-9a-f]{64}\n?',raw),'Owned actual CPU CID required');cid=raw.decode().strip()
    found=bank.command(['docker','ps','-aq','--no-trunc','--filter','id='+cid],deadline);rt.require(found in('',cid),'Ambiguous owned CPU CID')
    if found:
        actual=bank.command(['docker','inspect',cid,'--format','{{.Image}}|{{.Name}}|{{index .Config.Labels "world-reward.job"}}|{{index .Config.Labels "world-reward.revision"}}'],deadline)
        rt.require(actual==IMAGE+'|/'+name+'|'+ENTRY+'|'+revision,'Cannot delete foreign container');bank.command(['docker','rm','-f',cid],deadline)
    rt.require(not bank.command(['docker','ps','-aq','--no-trunc','--filter','id='+cid],deadline)
        and not bank.command(['docker','ps','-aq','--no-trunc','--filter','name=^/'+name+'$'],deadline),'Owned CID/name survives');path.chmod(0o400)


def image_state(deadline):
    value=rt.strict(bank.command(['docker','image','inspect',IMAGE,'--format',
        '{"Id":{{json .Id}},"Architecture":{{json .Architecture}},"Os":{{json .Os}},"RootFS":{{json .RootFS}}}'],deadline))
    rt.require(value['Id']==IMAGE and value['Architecture']=='amd64'and value['Os']=='linux'
        and value['RootFS']['Type']=='layers'and len(value['RootFS']['Layers'])==50,'Exact original LinuxCPU image required')
    return value


def dispatch(code,revision,pins,revisions):
    started=time.monotonic();deadline=started+180;source=rt.source(ROOT,code,revision,ENTRY,HELPERS)
    policy=configuration(code,source);source_stats=bank.source_state(code);binding=authenticate(code,source,pins,revisions)
    rt.require(not OUTPUT.exists()and OUTPUT.parent.is_dir(),'Fresh once-only DEV output required')
    OUTPUT.mkdir(mode=0o700);s=OUTPUT.lstat();owner=[s.st_dev,s.st_ino,s.st_uid,s.st_gid];name='world-reward-vcoco-pair-capacity-'+revision[:12]
    report=dict(schema=policy['schema'],stage='vcoco_pair_capacity_host',status='fail',producer_revision=revision,
        source_binding=source,inputs=pins,producer_revisions=revisions,owned_cleanup_verified=False,outputs_sealed=False,
        models_loaded=0,GPU_used=False,RGB_decoded=False,FIT_performed=False,selector_evaluated=False,
        RESERVED_reference_values_read=False,ownership_verified=False,quality_verified=False,adoption=False)
    try:
        image=image_state(deadline)
        rt.require(not bank.command(['docker','ps','-aq','--no-trunc','--filter','name=^/'+name+'$'],deadline),'Occupied CPU namespace')
        frozen=dict(binding['frozen']);frozen.update({str(code.parent/n):pin for n,pin in source['markers'].items()});refs=binding['values']['cohort']['records'][:8]
        for row in refs:frozen[str(DATA/'metadata'/row['reference_file'])]=row['reference_identity']
        files={n:source['helpers'][n]for n in NATIVE_FILES}
        proof=dict(binding=binding,source=source,native_files=files,frozen=frozen,image_id=IMAGE,output_owner=owner)
        proof_pin=bank.write(OUTPUT/'proof.json',proof)
        cmd=['docker','run','--rm','--name',name,'--cidfile',str(OUTPUT/'.container.cid'),
            '--label','world-reward.job='+ENTRY,'--label','world-reward.revision='+revision,'--network','none',
            '--user','0:0','--memory','6g','--cpus','4','--pids-limit','128','--read-only','--cap-drop','ALL',
            '--security-opt','no-new-privileges','--tmpfs','/tmp:rw,noexec,nosuid,size=64m','--entrypoint','/usr/bin/env']
        for path in sorted({code/n for n in files}|{Path(n)for n in frozen}):
            rt.canonical(path);rt.require(path.is_file()and ','not in str(path)and '\n'not in str(path),'Individual literal readonly mounts only')
            cmd+=['--mount',f'type=bind,src={path},dst={path},readonly']
        cmd+=['--mount',f'type=bind,src={OUTPUT},dst={OUTPUT}',IMAGE,'-i','PATH=/opt/conda/bin:/usr/bin:/bin',
            'HOME=/tmp','CUDA_VISIBLE_DEVICES=-1','OMP_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4','MKL_NUM_THREADS=4',
            'WR_CODE='+str(code),'WR_CODE_REVISION='+revision,'WR_IMAGE_ID='+IMAGE,'WR_VCOCO_CAPACITY_DEADLINE='+format(deadline-15,'.17g'),
            '/opt/conda/bin/python','-I','-B',str(code/NATIVE_FILES[0]),'--native','--proof-bytes',str(proof_pin['bytes']),'--proof-sha256',proof_pin['sha256']]
        result=subprocess.run(cmd,env=dict(PATH='/usr/bin:/bin',HOME='/nonexistent',DOCKER_HOST='unix://'+str(ROOT/'docker.sock')),
            stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=max(.001,deadline-time.monotonic()-15),check=False)
        report.update(native_exit_status=result.returncode,native_report_identity=rt.identity(OUTPUT/'native.json',2 << 20))
        native=rt.pinned(OUTPUT/'native.json',report['native_report_identity'],2 << 20)
        rt.require(result.returncode==0 and native['status']=='pass'and native['schema']==policy['schema']
            and native['producer_revision']==revision and native['proof_identity']==proof_pin and native['inputs']==pins
            and native['stage']=='saved_cpu_vcoco_pair_proposals'and native['phase']=='complete'
            and native['all_acquired_banks_validated_before_DEV']is native['source_inputs_predictions_references_rehashed_after']is True
            and native['validated_bank_count']==len(binding['values']['public']['images'])
            and native['reference_files_decoded']==sum(i<8 for i in binding['mapping'])
            and native['decision']==decision(native['metrics'])
            and native['runtime_identity']==binding['original_qualification']['owl']['native_runtime']
            and native['models_loaded']==0 and native['GPU_used']is native['RGB_decoded']is native['FIT_performed']is native['selector_evaluated']is native['RESERVED_reference_values_read']is False,
            'Complete saved-only CPU evaluation required')
        report.update(decision=decision(native['metrics']),metrics=native['metrics'])
    except BaseException as exc:report.update(status='fail',decision='CLOSED_TECHNICAL_EVALUATION',error_type=error(exc))
    finally:
        try:
            now=OUTPUT.lstat();rt.require((now.st_dev,now.st_ino,now.st_uid,now.st_gid)==tuple(owner),'Owned output replaced')
            cleanup(OUTPUT/'.container.cid',name,revision,min(deadline+15,time.monotonic()+10));report['owned_cleanup_verified']=True
        except BaseException as exc:report.update(status='fail',decision='CLOSED_TECHNICAL_CLEANUP',cleanup_error_type=error(exc))
        try:
            rt.require(authenticate(code,source,pins,revisions)==binding and rt.source(ROOT,code,revision,ENTRY,HELPERS)==source
                and bank.source_state(code)==source_stats,'Source/original inputs changed after CPU evaluation')
            if 'proof_pin'in locals():rt.require(rt.identity(OUTPUT/'proof.json',2 << 20)==proof_pin,'Host proof changed')
            if 'native_report_identity'in report:rt.require(rt.identity(OUTPUT/'native.json',2 << 20)==report['native_report_identity'],'Native saved report changed')
            for row in binding['values']['cohort']['records'][:8]:rt.require(rt.identity(DATA/'metadata'/row['reference_file'],4 << 20)==row['reference_identity'],'DEV reference changed')
            rt.require('image'in locals()and image_state(deadline)==image,'Original CPU image changed');check(deadline)
            report['source_inputs_predictions_references_rehashed_after']=True
            if 'error_type'not in report and report['owned_cleanup_verified']and report.get('native_exit_status')==0:report['status']='pass'
        except BaseException as exc:report.update(status='fail',decision='CLOSED_TECHNICAL_POSTHASH',post_error_type=error(exc))
    report['elapsed_seconds']=time.monotonic()-started;publish(OUTPUT/'host.json',report,deadline,owner,seal=True);return report


def arguments(argv):
    p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--native',action='store_true')
    for n in (*PIN_NAMES,'proof'):p.add_argument('--'+n+'-bytes',type=int);p.add_argument('--'+n+'-sha256')
    for n in PRODUCERS:p.add_argument('--'+n+'-revision')
    rt.require(len([x for x in argv if x.startswith('--')])==len(set(x.split('=')[0]for x in argv if x.startswith('--'))),'Duplicate CLI forbidden')
    a=p.parse_args(argv);present=set()
    for n in (*PIN_NAMES,'proof'):
        size,digest=getattr(a,n+'_bytes'),getattr(a,n+'_sha256')
        rt.require((size is None)==(digest is None),'Exact byte/SHA pairs required')
        if size is not None:rt.require(type(size)is int and 0<size<=2 << 20 and re.fullmatch('[0-9a-f]{64}',digest),'Bounded literal pin required');present.add(n)
    if a.native:rt.require(present=={'proof'}and all(getattr(a,n+'_revision')is None for n in PRODUCERS),'Native host-proof-only ABI')
    else:rt.require(present==set(PIN_NAMES)and all(getattr(a,n+'_revision')==v['revision']for n,v in PRODUCERS.items()),'Independent audited producer/receipt pins required')
    return a


def main():
    a=arguments(sys.argv[1:]);code=Path(os.environ['WR_CODE']);revision=os.environ['WR_CODE_REVISION']
    rt.require(sys.platform=='linux'and os.geteuid()==0 and code==ROOT/'jobs'/revision/ENTRY/'code'
        and Path(__file__).resolve()==code/NATIVE_FILES[0],'Immutable Azure source namespace required')
    def interrupted(*_):raise TimeoutError('Frozen saved-only capacity interruption')
    deadline=float(os.environ['WR_VCOCO_CAPACITY_DEADLINE'])if a.native else time.monotonic()+180
    handlers={s:signal.signal(s,interrupted)for s in(signal.SIGALRM,signal.SIGTERM)}
    signal.setitimer(signal.ITIMER_REAL,max(.001,deadline-time.monotonic()))
    try:
        pin=lambda n:dict(bytes=getattr(a,n+'_bytes'),sha256=getattr(a,n+'_sha256'))
        if a.native:result=cpu(code,revision,pin('proof'),deadline)
        else:
            rt.require(os.uname().nodename=='world-reward-ncc-h100-02','Exact VM02 saved-only host required')
            result=dispatch(code,revision,{n:pin(n)for n in PIN_NAMES},{n:getattr(a,n+'_revision')for n in PRODUCERS})
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        for s,handler in handlers.items():signal.signal(s,handler)
    print(encode(dict(status=result['status'],decision=result.get('decision'),quality_verified=False)).decode().strip())
    return 0 if result['status']=='pass'else 1


if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as exc:
        print(encode(dict(status='fail',error_type=error(exc))).decode().strip());raise SystemExit(1)from None
