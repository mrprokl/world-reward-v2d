"""Authenticate original48 bytes; return ONLY opaque public RGB input metadata.

Caller owns source/runtime authentication before and after its operation. This
seam does not run an old producer, parse roles, select records or clear licenses.
"""
import base64
from copy import deepcopy
import hashlib
from pathlib import Path
import stat

import vcoco_fit_cal_acquire as acquisition

context=acquisition.context
c,rt,prep=context.c,context.rt,context.prep
DATA=acquisition.DATA
ORIGINAL=dict(revision='d938e7c40c4fd90353fbe92ebd2970303a639b9b',files=316,entries=321,
    closure_sha256='f8c01005b0c2ba5d055784753d9cb8d903d1b1a6f02a5a3c483665f0ddee3b6b',
    archive_xz_sha256='567424d0eef72e5e6edfab82e7054aaddf01864d942d9d1f8446a4a859113d29')
REPORT_PIN=dict(bytes=133045,sha256='900596ae02a7b6fc60a79353f03517689fc8f540c1842b9ae3421df1114bf2c0')
PUBLIC_PIN=dict(bytes=9754,sha256='e98657edc8e081ed5b101edf22cdbb490b77bf36384e8952eff49f82d0c7b93c')
HELPER_PINS={'infra/vcoco_fit_cal_acquire.py':dict(bytes=13669,
    sha256='cdebb41d280f78285c4b658869dd35ac8148a72fb492dd982b7fadd9f0b70c50'),
    'infra/vcoco_fit_cal_context.py':dict(bytes=9325,
    sha256='b500491d15ba4d84d2966ec14be32347285ab87fa327975ba52b350f436ea58f')}


def authenticate_actual_acquisition(code,revision,entry,helpers,checkpoint):
    """Return (48 public records with paths, public-only live provenance).

    `code/entry/helpers` belong to the NEW caller. The old d938 source is read
    through its own immutable namespace, never imported as the current caller.
    No native/private IDs, split, publisher row or role appears in the return.
    """
    rt.require(Path(__file__).resolve()==code/'infra/vcoco_fit_cal_acquisition_inputs.py'
        and Path(acquisition.__file__).resolve()==code/'infra/vcoco_fit_cal_acquire.py',
        'Current input/acquisition helper origin differs')
    cfg,excluded,cohort,current=context.authenticate(code,revision,entry,helpers,checkpoint)
    rt.require(all(current['current_source']['binding']['helpers'][n]==p for n,p in HELPER_PINS.items()),
               'Exact qualified acquisition/context bytes required')
    old_code,old=context.f.original(code,ORIGINAL['revision'],acquisition.ENTRY,acquisition.HELPERS,ORIGINAL)
    old_input=deepcopy(current);old_input['current_source']=old
    for path in(code.parent,code.parent/'revision',code.parent/'source-sha256'):
        rt.require(str(path)in old_input['live_states'],'Current marker state missing')
        del old_input['live_states'][str(path)]
    for path in(old_code.parent,old_code.parent/'revision',old_code.parent/'source-sha256'):
        old_input['live_states'][str(path)]=c.state(path)
    private,public=DATA/'acquisition',DATA/'inputs'
    report=rt.pinned(private/'report.json',REPORT_PIN,1<<20)
    rt.require(report['input_proof']==context.normalized(old_input)
        and report['producer_revision']==ORIGINAL['revision']and report['status']=='pass'and report['stage']=='complete'
        and report['decision']=='READY_FULL48_PENDING_BLIND_BANKS'
        and report['freeze_report_identity']==context.REPORT_PIN and report['cohort_identity']==context.COHORT_PIN,
        'Actual original acquisition proof differs after JSON roundtrip')
    rt.require(all(type(report[k])is type(v)and report[k]==v for k,v in acquisition.RECIPE.items())
        and report['counts']==dict(slots=48,acquired=48,missing=0,acquired_FIT=32,acquired_CAL=16)
        and all(report[k]is True for k in('source_and_inputs_rehashed_after','outputs_sealed','public_outputs_sealed',
            'availability_gate_passed','acquisition_records_complete'))
        and report['duplicate_new_bytes']is report['http_window_expired']is False,
        'All48 complete availability/source/seals required')
    rt.require(all(report[k]is False for k in('annotation_values_consulted','role_values_consulted',
        'pilot_reference_values_read','RGB_decoded','GPU_used','models_loaded','FIT_performed','CAL_evaluated',
        'selection_performed','author_disjointness_verified','training_overlap_verified','challenge_overlap_verified','adopted'))
        and report['network_used']is True and type(report['elapsed_seconds'])in(int,float)
        and 0<report['elapsed_seconds']<=1200 and type(report['peak_rss_bytes'])is int
        and 0<report['peak_rss_bytes']<=8<<30,'Original acquisition scope/budget differs')
    rt.require(prep.namespace_identity(DATA)==current['private_parent'],'Original root700 parent differs')
    states={}
    for directory,names in((private,{'report.json'}),(public,{'manifest.json'}|{f'image_{i:06d}.jpg'for i in range(48)})):
        s=rt.canonical(directory).lstat()
        rt.require(stat.S_ISDIR(s.st_mode)and stat.S_IMODE(s.st_mode)==0o500 and s.st_uid==s.st_gid==0
            and {p.name for p in directory.iterdir()}==names,'Original exact sealed namespace differs')
        states[str(directory)]=c.state(directory)
        for path in directory.iterdir():
            s=path.lstat();rt.require(stat.S_IMODE(s.st_mode)==0o400 and s.st_uid==s.st_gid==0,
                                    'Original root400 artifact required')
            states[str(path)]=dict(pin=rt.identity(path,16<<20),state=c.state(path))
    manifest=rt.pinned(public/'manifest.json',PUBLIC_PIN,1<<20)
    rt.require(set(manifest)=={'schema','images'}and manifest['schema']=='world_reward.rgb_proposal_inputs.v1'
        and len(manifest['images'])==48 and report['public_inputs_identity']==PUBLIC_PIN
        and report['artifact_identities']=={},'Opaque public manifest/private ledger differs')
    records,publisher=report['records'],report['publisher_metadata']
    rt.require(len(records)==len(publisher)==len(report['public_mappings'])==48,'All48 private records required')
    ledger={'manifest.json':PUBLIC_PIN};md5s=set();output=[]
    for row,record,pub,mapping,image in zip(cohort['records'],records,publisher,report['public_mappings'],manifest['images']):
        checkpoint();slot=row['slot'];iid=row['image_id'];filename=f'image_{slot:06d}.jpg'
        rt.require(set(record)=={'slot','split','image_id','status','image_pin','original_md5','jpeg_header'}
            and type(record['slot'])is int and record['slot']==slot and record['image_id']==iid
            and record['split']==row['split']and record['status']=='acquired','Original slot/status differs')
        rt.require(set(pub)==set(row)|{'width','height','partition','file_name','original_url','license_id','license_url','creator_identity'}
            and all(pub[k]==v for k,v in row.items())and pub['partition']in('train2014','val2014')
            and pub['file_name']==f"COCO_{pub['partition']}_{iid:012d}.jpg"
            and pub['original_url']==prep.BUCKET+pub['partition']+'/'+pub['file_name']
            and pub['license_id']==4 and pub['license_url']==c.GRANT and pub['creator_identity']=='UNKNOWN'
            and all(type(pub[k])is int and pub[k]>0 for k in('width','height'))
            and pub['width']*pub['height']<=acquisition.RECIPE['max_decoded_pixels'],'Original publisher grant/grid differs')
        opaque=hashlib.sha256((context.selection.NAMESPACE+f'{iid:012d}').encode()).hexdigest()[:32]
        expected=dict(image_id=opaque,file=filename,**record['image_pin'],width=pub['width'],height=pub['height'])
        rt.require(image==expected and set(image)==prep.PUBLIC_KEYS and mapping==dict(slot=slot,public_image_id=opaque,
            public_file=filename,image_pin=record['image_pin']),'Public original slot mapping differs')
        path=public/filename;raw=path.read_bytes();checkpoint()
        rt.require(c.pin(raw)==record['image_pin']==states[str(path)]['pin'],'Original JPEG SHA differs')
        md5=base64.b64encode(hashlib.md5(raw).digest()).decode()
        rt.require(md5==record['original_md5']and md5 not in excluded['md5']and md5 not in md5s,
                   'Historical/duplicate byte alias rejected');md5s.add(md5)
        header=prep.acq.jpeg_header(raw,acquisition.RECIPE['max_decoded_pixels'])
        rt.require(header==record['jpeg_header']and(header['width'],header['height'])==(pub['width'],pub['height']),
                   'Original header grid differs; no pixel decode')
        ledger[filename]=record['image_pin'];output.append(dict(image,path=str(path)))
    rt.require(report['public_artifact_identities']==ledger and sum(p['bytes']for p in ledger.values())<=
        acquisition.RECIPE['max_total_image_bytes']+(1<<20),'Complete public49 ledger differs')
    checkpoint()
    return output,dict(schema='world_reward.vcoco_fit_cal_public_input_proof.v1',current_source=current['current_source'],
        original_acquisition_source=old,acquisition_report_identity=REPORT_PIN,public_manifest_identity=PUBLIC_PIN,
        original_private_state_identity=c.pin(c.encode(context.normalized(old_input))),
        live_context_state_identity=c.pin(c.encode(context.normalized(current))),public_artifact_states=states,
        input_records=48,source_authenticated=True,RGB_decoded=False,reference_values_exposed=False,
        role_values_consulted=False,publisher_metadata_exposed=False,FIT_CAL_splits_exposed=False,
        author_disjointness_verified=False,training_overlap_verified=False,challenge_overlap_verified=False,adopted=False)
