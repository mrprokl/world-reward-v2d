"""Lossless full OWLv2 bank -> generic interaction evidence, not selection.

Raw scores here remain uncalibrated image-only OBJECTNESS LOGITS. No sigmoid,
class probability, ranking, NMS, topK, positive-area or image-bound filtering.
The fingerprint authenticates transport bytes, not physical identity, original
model provenance, licensing, complete detection recall or manipulation.
"""
from dataclasses import fields

from .interaction_candidate_evidence import GenericObjectObservations
from .interaction_tuple_evidence import _reference
from .owlv2_object_observations import Owlv2ObjectObservations, SCHEMA


def bridge_owlv2_candidates(observation):
    """Retain every original row for later all-person/all-side hypotheses."""
    if type(observation) is not Owlv2ObjectObservations:
        raise ValueError('Exact full native OWLv2 observation required')
    # Revalidate a snapshot: frozen dataclasses/NumPy flags alone cannot prove
    # that a caller has not forcibly changed a native field since manufacture.
    snapshot = Owlv2ObjectObservations(**{
        f.name: getattr(observation, f.name) for f in fields(observation)
    })
    before = _reference(snapshot, SCHEMA)
    result = GenericObjectObservations(
        snapshot.original_frame_index, snapshot.image_size, snapshot.object_ids,
        snapshot.boxes_original_xyxy, snapshot.objectness_logits, (before,),
    )
    if _reference(observation, SCHEMA) != before:
        raise ValueError('Original native bank changed during lossless transport')
    return result
