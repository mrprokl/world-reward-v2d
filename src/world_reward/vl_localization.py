"""Independent image observations and lossless SAM handoff; no identity oracle.

No box interpolation, offset fitting, confidence heuristic, judge or prior-frame
answer is used. Valid JSON is not semantic accuracy; missing evidence is retained.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json

from .task_grounding import build_task_grounding_prompt, fixed_frame_indices, parse_response


def requested_frames(total: int) -> tuple[int, ...]:
    if type(total) is not int or total < 30:
        raise ValueError('At least30 original frames required')
    return tuple(sorted(set(range(30)) | set(fixed_frame_indices(total, 9))))


def question(description: str, action: str, frame_index: int) -> str:
    # Same task/box contract as the baseline, with exactly one requested image.
    return build_task_grounding_prompt(description, action, (frame_index,))


def observation(text: str, *, episode: int, frame_index: int, width: int, height: int,
                rgb_sha256: str) -> dict:
    if (type(episode) is not int or not 0 <= episode < 30 or type(frame_index) is not int
            or frame_index < 0 or type(rgb_sha256) is not str or len(rgb_sha256) != 64
            or any(c not in '0123456789abcdef' for c in rgb_sha256)):
        raise ValueError('Explicit episode/frame/image identity required')
    result = dict(episode_index=episode, frame_index=frame_index, width=width, height=height,
                  decoded_rgb_sha256=rgb_sha256, response_sha256=hashlib.sha256(text.encode()).hexdigest(),
                  person_bbox=None, object_bbox=None, status='invalid_response',
                  semantic_accuracy_verified=False)
    try:
        record, = parse_response(text, (frame_index,), width, height)
        result.update(asdict(record))
    except ValueError:
        pass  # Keep invalid evidence; do not repair, infer or retry that image.
    return result


def sam_handoff(records: list[dict], *, episode: int, expected_indices: tuple[int, ...]) -> dict:
    """Literal original-resolution boxes, not fitted/predicted masks or seeds.

    This is a proposed dense prompting input, not permission to adopt a target
    or evidence that SAM ran. Each prompt preserves the original frame/entity.
    """
    if ([r['frame_index'] for r in records] != list(expected_indices)
            or any(r['episode_index'] != episode for r in records)):
        raise ValueError('Exact ordered original observations required')
    prompts, missing = [], []
    for r in records:
        for obj, key in ((0, 'person_bbox'), (1, 'object_bbox')):
            box = r[key]
            if box is None:
                missing.append(dict(frame_index=r['frame_index'], object_id=obj, reason=r['status']))
                continue
            x0,y0,x1,y1 = box
            if not (0 <= x0 < x1 <= r['width'] and 0 <= y0 < y1 <= r['height']):
                raise ValueError('Original pixel box invalid')
            prompts.append(dict(frame_index=r['frame_index'], object_id=obj, points=None,
                                point_labels=None, mask_path=None,
                                box=dict(zip(('x0','y0','x1','y1'),box))))
    return dict(schema='world_reward.vl_sam_handoff.v1', episode_index=episode,
                original_frame_indices=list(expected_indices), prompts=prompts, missing=missing,
                sam_executed=False, manual_labels=False, quality_verified=False,
                interpolation=False, offset_fit=False)


def encoded_layout(input_ids, attention_mask, image_grid, image_patch_count: int,
                   image_token_id: int, merge_size: int = 2):
    """Audit explicit batch layout; arrays are tiny manufactured/local or Azure.

    Return each row's patch slice and real token list for singleton comparison.
    Attention must be left padding; no input/image reordering is allowed.
    """
    import numpy as np
    ids, mask, grid = map(np.asarray, (input_ids, attention_mask, image_grid))
    if (ids.ndim != 2 or ids.dtype.kind not in 'iu' or mask.shape != ids.shape
            or not np.isin(mask,[0,1]).all() or grid.shape != (len(ids),3)
            or grid.dtype.kind not in 'iu' or (grid <= 0).any() or (grid[:,0] != 1).any()
            or type(merge_size) is not int or merge_size <= 0
            or type(image_token_id) is not int or type(image_patch_count) is not int):
        raise ValueError('One still image per independent left-padded batch row required')
    patches = np.prod(grid, axis=1)
    if (patches % merge_size**2).any() or int(patches.sum()) != image_patch_count:
        raise ValueError('Image patch packing differs')
    result = []; offset = 0
    for row, m, g, n in zip(ids,mask,grid,patches):
        if not m.any() or (np.diff(m.astype('int64')) < 0).any():
            raise ValueError('Left padding only; no empty prompt')
        tokens = row[m.astype(bool)]
        if int((tokens == image_token_id).sum()) != int(n)//merge_size**2:
            raise ValueError('Image placeholder and processed pixels differ')
        result.append(dict(tokens=tokens.tolist(), grid=g.tolist(), patch_slice=(offset,offset+int(n))))
        offset += int(n)
    return result


def compact_json(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False)+'\n').encode()
