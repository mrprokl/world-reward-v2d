"""Explicit-context live authentication; no old execution/global profile changes."""
from pathlib import Path
import hashlib
import re
import stat
import sys
import vcoco_fit_cal_freeze as freeze

f,c,rt,prep,selection=freeze.f,freeze.c,freeze.rt,freeze.prep,freeze.selection
ROOT=freeze.ROOT
OLD=freeze.OLD
FREEZE=dict(revision='8cf55c23755975325fe00efb8fa3229392b38475',files=312,entries=317,
    closure_sha256='a21b375f4c83e1fe9c49358d09804af8c963f767e409774623e0f3661dd28a3c',
    archive_xz_sha256='022bfbb9885e36b68ef8de46ff1b5fdf5a46dabce4f73b6dc6363e248329b593')
REPORT_PIN=dict(bytes=63579,sha256='22b2830a1973fe61a1c2ad626cb077fd58f6def5cd89557f027db59e10026ebc')
COHORT_PIN=dict(bytes=10151,sha256='5af18ab329bd6dfc113013787c340ff6aeb6f161e7e4b8d7b1d777061fc83f51')
FREEZE_HELPER=dict(bytes=12514,sha256='f5c0e0eb4df02c15c397105d7445317adabbd8bfe73dddfea3c7ee36a574d30e')


def normalized(value):return rt.strict(c.encode(value))


def authenticate(code,revision,entry,helpers,checkpoint):
    current=c.source(code,revision,entry,helpers)
    rt.require(current['binding']['helpers']['infra/vcoco_fit_cal_freeze.py']==FREEZE_HELPER
        and all(current['binding']['helpers'][n]==p for n,p in freeze.FROZEN.items()),'Exact qualified reuse bytes required')
    for module,name in ((freeze,'infra/vcoco_fit_cal_freeze.py'),(f,'infra/vcoco_fit_cal_census.py'),
        (selection,'infra/vcoco_fit_cal_selection.py'),(prep,'infra/vcoco_role_prepare.py'),(c,'infra/vcoco_role_census.py'),
        (rt,'infra/mediapipe_cpu_runtime_verify.py'),(c.js,'infra/metadata_json_stream.py'),
        (c.projection,'infra/vcoco_role_stream.py'),(c.identities,'infra/openimages_joint_pair_acquire.py'),
        (c.coco,'infra/coco_proposal_prepare.py'),(f.population,'infra/vcoco_population_census.py')):
        rt.require(Path(module.__file__).resolve()==code/name,'Current helper origin differs')
    rt.require(Path(__file__).resolve()==code/'infra/vcoco_fit_cal_context.py'
        and Path(sys.modules[c.parse_vcoco_role_reference.__module__].__file__).resolve()==
        code/'src/world_reward/vcoco_role_reference.py','Current context/parser origin differs')
    freeze_code,frozen_source=f.original(code,FREEZE['revision'],freeze.ENTRY,freeze.HELPERS,FREEZE)
    old_code,old = f.original(code, OLD['revision'], f.ENTRY, f.HELPERS, OLD)
    prepare_code,prepare = f.original(code, f.PREPARE['revision'], prep.ENTRY, prep.HELPERS, f.PREPARE)
    cfg = prep.configuration(code, prepare['binding']); spec = cfg['census']
    census_code,_ = f.original(code, spec['producer_revision'], c.ENTRY, c.HELPERS, spec)
    old_cfg = c.configuration(rt.pinned(Path(spec['configuration']['path']), spec['configuration']['pin'],65536))
    rt.require(Path(spec['configuration']['path']) == census_code/c.CONFIG, 'Original configuration path differs')
    original_census = c.authenticate(old_cfg,census_code,spec['producer_revision']); checkpoint()
    prior = rt.pinned(Path(spec['report']['path']),spec['report']['pin'],300000)
    rt.require(Path(spec['report']['path']) == c.DATA/'report.json' and prior['status'] == 'pass'
        and prior['stage'] == 'complete' and prior['capacity_gate_passed'] is True
        and normalized(prior['source_binding']) == normalized(original_census)
        and prior['configuration_identity'] == spec['configuration']['pin']
        and prior['outputs_sealed'] is prior['source_and_inputs_rehashed_after'] is True
        and prior['historical_slots'] == 432 and prior['eligibility_inventory_rows'] == 561,
        'Original sealed census lineage differs')
    for directory,names in ((c.DATA,{'report.json'}),(f.OUTPUT,{'report.json','inventory.json'})):
        st=rt.canonical(directory).lstat()
        rt.require(stat.S_IMODE(st.st_mode) == 0o500 and st.st_uid == st.st_gid == 0
                   and {p.name for p in directory.iterdir()} == names,'Original sealed namespace differs')
        for path in directory.iterdir():
            st=path.lstat();rt.require(stat.S_IMODE(st.st_mode) == 0o400 and st.st_uid == st.st_gid == 0,
                                     'Original root-owned400 leaf required')
    proof = dict(source_binding=prepare['binding'],census_binding=original_census,
        source_stat_identity=prepare['stat_identity'],census_report_identity=spec['report']['pin'],
        census_report_state=c.state(Path(spec['report']['path'])),census_directory_state=c.state(c.DATA))
    values = {n:rt.strict(Path(old_cfg['inputs'][n]['path']).read_bytes()) for n in
        (*old_cfg['history_names'],'ownership_cohort','ownership_acquired','coco32_acquired','coco64_acquired')}
    excluded = c.exclusions([values[n] for n in old_cfg['history_names']], values['ownership_cohort'],
        values['ownership_acquired'],[values['coco32_acquired'],values['coco64_acquired']])
    excluded,pilot = f.pilot(excluded,proof,checkpoint)
    original_proof = dict(current=old, original_prepare=prepare,original_census=original_census,
                         prepare_proof=proof,pilot=pilot)
    report = rt.pinned(f.OUTPUT/'report.json',selection.REPORT_PIN,1 << 20)
    rt.require(rt.identity(f.OUTPUT/'inventory.json',1 << 20) == selection.INVENTORY_PIN
               and normalized(original_proof) == report['input_proof'],'Original complete census proof differs')
    # Include marker/parent modes and inode states, not just their byte identity.
    source_codes = [freeze_code,old_code,prepare_code,census_code] + [ROOT/'jobs'/v['revision']/v['entry']/'code'
                                                            for v in old_cfg['sources'].values()]
    states = {str(p):c.state(p) for d in source_codes for p in (d.parent,d.parent/'revision',d.parent/'source-sha256')}
    states.update({str(f.OUTPUT):c.state(f.OUTPUT),str(f.OUTPUT/'report.json'):c.state(f.OUTPUT/'report.json'),
                   str(f.OUTPUT/'inventory.json'):c.state(f.OUTPUT/'inventory.json')})
    freeze_input_proof = dict(current_source=frozen_source,original_input_proof=original_proof,
        live_states=dict(states),census_report_identity=selection.REPORT_PIN,census_inventory_identity=selection.INVENTORY_PIN)
    saved=rt.pinned(freeze.DATA/'metadata/report.json',REPORT_PIN,1 << 20)
    rt.require(saved['input_proof']==normalized(freeze_input_proof) and saved['status']=='pass'
        and saved['producer_revision']==FREEZE['revision'] and saved['stage']=='complete'
        and saved['source_and_inputs_rehashed_after']is saved['outputs_sealed']is True
        and saved['decision']=='FROZEN48_IDENTITIES_PENDING_SEPARATE_ACQUISITION'
        and saved['cohort_identity']==COHORT_PIN,'Actual frozen48 lineage required')
    cohort=rt.pinned(freeze.DATA/'metadata/cohort.json',COHORT_PIN,1 << 20)
    rt.require(cohort['schema']=='world_reward.vcoco_fit_cal_identity_cohort.v1' and cohort['namespace']==selection.NAMESPACE
        and cohort['census_producer_revision']==OLD['revision'] and cohort['census_report_identity']==selection.REPORT_PIN
        and cohort['census_inventory_identity']==selection.INVENTORY_PIN and cohort['selected_slots']==48
        and cohort['split_counts']==dict(FIT=32,CAL=16) and cohort['all_48_acquired_required']is True
        and cohort['retry_count']==cohort['replacement_count']==0 and cohort['reference_values_exposed']is False
        and cohort['predictor_input']is cohort['source_authenticated']is False
        and cohort['input_proof_identity']==c.pin(c.encode(normalized(original_proof)))
        and cohort['excluded_photo_ids_identity']==c.pin(c.encode(sorted(excluded['photos']))),'Frozen cohort contract differs')
    rows=cohort['records'];rt.require(len(rows)==48 and [r['slot']for r in rows]==list(range(48))
        and [r['split']for r in rows]==['FIT']*32+['CAL']*16
        and [r['official_split']for r in rows]==['train']*32+['val']*16
        and len({r['image_id']for r in rows})==len({r['photo_id']for r in rows})==48
        and all(set(r)=={'slot','split','official_split','image_id','photo_id','rank_sha256'} and type(r['slot'])is int and type(r['image_id'])is int
        and 0<r['image_id']<10**12 and type(r['photo_id'])is str and re.fullmatch('[0-9]+',r['photo_id'])
        and r['photo_id']not in excluded['photos'] and r['rank_sha256']==hashlib.sha256(
        (selection.NAMESPACE+f"{r['image_id']:012d}").encode()).hexdigest()for r in rows),'All frozen48 identities required')
    root=prep.namespace_identity(freeze.DATA);metadata=freeze.DATA/'metadata';st=metadata.lstat()
    rt.require(stat.S_IMODE(st.st_mode)==0o500 and st.st_uid==st.st_gid==0
        and {p.name for p in metadata.iterdir()}=={'report.json','cohort.json'},'Original private metadata namespace required')
    for path in metadata.iterdir():
        st=path.lstat();rt.require(stat.S_IMODE(st.st_mode)==0o400 and st.st_uid==st.st_gid==0,'Original root400 metadata required')
        states[str(path)]=c.state(path)
    states[str(metadata)]=c.state(metadata)
    states.update({str(p):c.state(p) for p in (code.parent,code.parent/'revision',code.parent/'source-sha256')})
    return old_cfg,excluded,cohort,dict(current_source=current,freeze_source=frozen_source,
        original_freeze_input_proof=freeze_input_proof,live_states=states,private_parent=root,
        freeze_report_identity=REPORT_PIN,cohort_identity=COHORT_PIN)
