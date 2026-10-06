"""Public cardinality-generic pose callback, no old producer/private readers.

`observe` is the exact original function AST from vcoco_person_pose_observations;
only its imports are narrowed to hashlib and the native-public runtime require.
Caller authenticates callbacks/session/source and owns complete inputs/outputs.
The loop invokes supplied inference/save callbacks; no model constructor,
private reader or producer/source authentication is imported here.
"""
import hashlib
import mediapipe_cpu_runtime_verify as rt

FIELDS = ('person_ids','boxes_original_xyxy','detector_scores','keypoints_original_xy','raw_scores',
    'native_valid','in_original_image','image_size','original_frame_index','original_slot','acquired_ordinal')

def observe(images,banks,load_bank,decode,infer,save,check,records=None):
    """Validate EVERY complete bank before the first pose, preserve all person rows."""
    rt.require(type(images)is list and type(banks)is list and len(images)==len(banks), 'Aligned complete original RGB/bank rows required')
    rows=[]if records is None else records
    for bank in banks:check();load_bank(bank,validate=True)
    expected=0
    for ordinal,(image,bank)in enumerate(zip(images,banks)):
        check();a=load_bank(bank,validate=False);rgb=decode(image)
        ids=tuple(a['person_retained_ids'].tolist());expected+=len(ids)
        rt.require(list(ids)==bank['person_ids']and len(ids)==bank['person_retained_rows'], 'Every recorded person ID required')
        fingerprints={n:(a[n].dtype.str,a[n].shape,hashlib.sha256(a[n].tobytes()).hexdigest())for n in a}
        result=infer(rgb,ids,a['person_retained_boxes'],a['person_retained_scores'])
        rt.require(result.person_ids==ids and result.original_frame_index==0 and result.image_size==tuple(bank['image_size']),
            'Every original person/grid in original order required')
        rt.require(fingerprints=={n:(a[n].dtype.str,a[n].shape,hashlib.sha256(a[n].tobytes()).hexdigest())for n in a},'Supplied original person bank mutated')
        rows.append(save(result,image,bank,ordinal));check()
    return rows,expected
