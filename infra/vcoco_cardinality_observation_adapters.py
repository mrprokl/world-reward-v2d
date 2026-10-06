"""Full48 pose/HOI callback adapters, no model constructor or private imports."""
import os
from pathlib import Path

import vcoco_public_observation_context as public

rt,endpoint=public.rt,public.endpoint
POSE_FIELDS=('person_ids','boxes_original_xyxy','detector_scores','keypoints_original_xy','raw_scores',
    'native_valid','in_original_image','image_size','original_frame_index','original_slot','acquired_ordinal')
HOI_FIELDS=('query_logits','query_boxes_cxcywh','query_tokens','native_nms_detections','native_nms_keep',
    'retained_nms_positions','query_ids','class_ids','boxes_original_xyxy','raw_scores','decayed_scores',
    'hand_object_pairs','hand_object_logits','object_target_pairs','object_target_logits',
    'image_size','original_frame_index','original_slot','acquired_ordinal')
HELPERS=public.HELPERS


def _payload(kind,result,image,bank,ordinal):
    import numpy as np
    from world_reward.person_pose_observations import PersonPoseObservations
    from world_reward.hoi_detr_observations import HOIDetrObservations
    rt.require(kind in('pose','hoi')and type(ordinal)is int and 0<=ordinal<48
        and type(result.original_frame_index)is int and result.original_frame_index==0 and result.image_size==(image['height'],image['width']),
        'Original frame/grid/kind required')
    fields=POSE_FIELDS if kind=='pose'else HOI_FIELDS
    if kind=='pose':
        rt.require(type(result)is PersonPoseObservations and list(result.person_ids)==bank['person_ids']
            and result.keypoints_original_xy.dtype==np.float64 and result.raw_scores.dtype==np.float32,
            'Every original person/native dtype required')
        reconstructed=PersonPoseObservations(0,result.image_size,result.person_ids,result.boxes_original_xyxy,
            result.detector_scores,result.keypoints_original_xy,result.raw_scores)
        arrays={n:getattr(result,n)for n in fields[1:7]}
        rt.require(endpoint.array_identities(arrays)==endpoint.array_identities({n:getattr(reconstructed,n)for n in arrays}),
            'Native pose validity/grid fields differ')
        arrays['person_ids']=np.asarray(result.person_ids,dtype=np.dtype(bank['arrays']['person_retained_ids']['dtype']))
        for target,source in(('person_ids','person_retained_ids'),('boxes_original_xyxy','person_retained_boxes'),('detector_scores','person_retained_scores')):
            rt.require(endpoint.array_identities({target:arrays[target]})[target]==bank['arrays'][source],
                'Pose changed original person census')
    else:
        rt.require(type(result)is HOIDetrObservations,'Original native HOI observation required')
        arrays={n:getattr(result,n)for n in fields[:15]}
        reconstructed=HOIDetrObservations(0,result.image_size,**arrays)
        rt.require(endpoint.array_identities(arrays)==endpoint.array_identities({n:getattr(reconstructed,n)for n in arrays}),
            'Full1500/raw Cartesian HOI arrays differ')
    arrays.update(image_size=np.asarray(result.image_size,np.int64),original_frame_index=np.asarray(0,np.int64),
        original_slot=np.asarray(ordinal,np.int64),acquired_ordinal=np.asarray(ordinal,np.int64))
    return arrays


def save_observation(reference,kind,result,image,bank,ordinal,deadline):
    """Original11/19 arrays, exclusive fsync/400 and byte-exact roundtrip."""
    import numpy as np
    endpoint.check(deadline)
    rt.require(type(reference)is public.PublicBankReference and type(ordinal)is int and 0<=ordinal<48
        and image==reference.images[ordinal] and bank==reference.banks[ordinal], 'Original public save row required')
    arrays=_payload(kind,result,image,bank,ordinal);before=endpoint.array_identities(arrays)
    path=reference.context.output/f'image_{ordinal:06d}.npz'
    with path.open('xb')as stream:
        os.fchmod(stream.fileno(),0o400);np.savez(stream,**arrays);stream.flush();os.fsync(stream.fileno())
    pin=rt.identity(path,(2 if kind=='pose'else 8)<<20)
    with np.load(path,allow_pickle=False)as saved:
        rt.require(set(saved.files)==set(arrays)and endpoint.array_identities({n:saved[n]for n in saved.files})==before,
            'Full native save roundtrip differs')
    rt.require(endpoint.array_identities(arrays)==before,'Original observations mutated during save')
    row=dict(image_id=image['image_id'],original_slot=ordinal,acquired_ordinal=ordinal,original_frame_index=0,
        image_size=list(result.image_size),file=path.name,identity=pin,arrays=before,
        endpoint_bank_identity=bank['identity'],source_person_ids=bank['person_ids'],owl_patches=3600)
    if kind=='pose':row.update(person_ids=list(result.person_ids),persons=len(result.person_ids))
    else:row.update(native_detections=len(result.query_ids),hand_object_pairs=len(result.hand_object_pairs),object_target_pairs=len(result.object_target_pairs))
    return row


def validate_saved_records(reference,kind,records,deadline,*,complete=True):
    import numpy as np
    from world_reward.person_pose_observations import PersonPoseObservations
    from world_reward.hoi_detr_observations import HOIDetrObservations
    rt.require(kind in('pose','hoi')and type(records)is list and (len(records)==48 if complete else len(records)<=48),
        'Complete48 success or explicit partial integrity required')
    fields=POSE_FIELDS if kind=='pose'else HOI_FIELDS
    for ordinal,row in enumerate(records):
        endpoint.check(deadline);image,bank=reference.images[ordinal],reference.banks[ordinal]
        path=reference.context.output/f'image_{ordinal:06d}.npz'
        rt.require(row['file']==path.name and row['image_id']==image['image_id']
            and row['original_slot']==row['acquired_ordinal']==ordinal and row['original_frame_index']==0
            and row['image_size']==bank['image_size'] and row['source_person_ids']==bank['person_ids']
            and row['owl_patches']==3600 and row['endpoint_bank_identity']==bank['identity']
            and rt.identity(path,(2 if kind=='pose'else 8)<<20)==row['identity'],'Every original saved observation required')
        with np.load(path,allow_pickle=False)as saved:
            rt.require(set(saved.files)==set(fields),'Full11/19 members required');arrays={n:saved[n]for n in saved.files}
        rt.require(endpoint.array_identities(arrays)==row['arrays'],'Saved array identity differs')
        if kind=='pose':
            result=PersonPoseObservations(0,tuple(bank['image_size']),tuple(arrays['person_ids'].tolist()),
                *[arrays[n]for n in POSE_FIELDS[1:5]])
            rt.require(row['person_ids']==bank['person_ids']and row['persons']==len(result.person_ids),'All person rows required')
        else:
            result=HOIDetrObservations(0,tuple(bank['image_size']),**{n:arrays[n]for n in HOI_FIELDS[:15]})
            rt.require(row['source_person_ids']==bank['person_ids']and row['native_detections']==len(result.query_ids)
                and row['hand_object_pairs']==len(result.hand_object_pairs)and row['object_target_pairs']==len(result.object_target_pairs),'All native pair rows required')
        rt.require(endpoint.array_identities(_payload(kind,result,image,bank,ordinal))==row['arrays'],'Original scalar/grid/native flags differ')


def _start(reference,records,deadline):
    rt.require(type(reference)is public.PublicBankReference and type(records)is list and not records,'Fresh public full48 context/records required')
    for name in HELPERS:rt.require(rt.identity(reference._code/name,2<<20)==rt.strict(reference._native_source)['helpers'][name],
        'Explicit current observation helper bytes required')
    import world_reward.person_pose_observations as pose
    import world_reward.hoi_detr_observations as hoi
    rt.require(all(Path(module.__file__).resolve()==reference._code/name for module,name in
        ((pose,'src/world_reward/person_pose_observations.py'),(hoi,'src/world_reward/hoi_detr_observations.py'))),
        'Actual current numerical class origins required')
    rt.require(Path(__file__).resolve()==reference._code/'infra/vcoco_cardinality_observation_adapters.py',
        'Actual cardinality adapter origin required')
    rt.canonical(reference.context.output);rt.require(reference.context.output.is_dir()
        and not any(reference.context.output.glob('image_*.npz')),'Fresh imagewise output namespace required')
    reference.verify(deadline)


def observe_pose(reference,original_observe,infer,deadline,*,records=None):
    """Delegate unchanged cardinality-generic pose.observe through callbacks.

    Caller binds the callback to its qualified source/session. No old producer
    is imported here. infer(rgb, ids, boxes, scores) preserves P0/no fallback.
    """
    records=[]if records is None else records;_start(reference,records,deadline)
    images,banks=reference.images,reference.banks
    try:
        original_observe(images,banks,lambda r,validate:public.load_bank(reference,r,deadline),
            lambda r:endpoint.public.decode_rgb(reference.context.inputs,r,identity=rt.identity),infer,
            lambda result,image,bank,i:save_observation(reference,'pose',result,image,bank,i,deadline),
            lambda:endpoint.check(deadline),records)
        validate_saved_records(reference,'pose',records,deadline);return records
    finally:
        try:reference.verify(deadline)
        finally:validate_saved_records(reference,'pose',records,deadline,complete=False)


def observe_hoi(reference,infer,deadline,*,records=None):
    """One original infer_hoi_detr_frame callback per fixed image, no K cap."""
    records=[]if records is None else records;_start(reference,records,deadline)
    try:
        for i,(image,bank)in enumerate(zip(reference.images,reference.banks)):
            endpoint.check(deadline);rgb=endpoint.public.decode_rgb(reference.context.inputs,image,identity=rt.identity)
            before=endpoint.array_identities({'rgb':rgb});result=infer(rgb)
            rt.require(endpoint.array_identities({'rgb':rgb})==before,'Native inference changed original RGB')
            records.append(save_observation(reference,'hoi',result,image,bank,i,deadline))
        validate_saved_records(reference,'hoi',records,deadline);return records
    finally:
        try:reference.verify(deadline)
        finally:validate_saved_records(reference,'hoi',records,deadline,complete=False)
