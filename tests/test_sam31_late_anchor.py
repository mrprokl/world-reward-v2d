"""Manufactured pure late-anchor verifier; no credentials, API or images."""
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import pytest

import sam31_late_anchor as module

SHA = 'a' * 64
PNG = module.PNG + b'manufactured-not-real-image'


def request():
    return module.build_request(seed_png=PNG, late_png=PNG, late_frame_index=99, width=160, height=120,
        seed_rgb_sha256=SHA, late_rgb_sha256='b' * 64, seed_person_bbox=[10, 10, 60, 110],
        seed_object_bbox=[70, 40, 120, 90], object_description='red box', action='actor lifts the box')


def output():
    return dict(frame_index=99,
        person=dict(same_physical_identity=True, visible=True, box_2d=[100, 100, 900, 400], label='actor'),
        object=dict(same_physical_identity=True, visible=True, box_2d=[300, 450, 700, 800], label='box'))


def vertex(value=None, finish='STOP'):
    return json.dumps(dict(candidates=[dict(finishReason=finish, content={'parts': [
        dict(thought=True, text='not parsed'), dict(text=json.dumps(output() if value is None else value))]})],
        modelVersion=module.MODEL)).encode()


def test_exact_existing_model_single_body_two_original_images_and_data():
    raw, binding = request(); body = json.loads(raw)
    assert module.MODEL == 'gemini-3.5-flash' and '/locations/global/' in module.ENDPOINT
    assert binding['request_sha256'] == hashlib.sha256(raw).hexdigest()
    parts = body['contents'][0]['parts']; images = [p['inlineData'] for p in parts if 'inlineData' in p]
    assert len(images) == 2 and all(base64.b64decode(p['data']) == PNG for p in images)
    task = json.loads(parts[-1]['text'].split('\n', 1)[1])
    assert task['reference_frame_index'] == 0 and task['later_frame_index'] == 99
    assert 'late_candidate_bbox' not in task and task['reference_automatic_object_bbox_xyxy'] == [70., 40., 120., 90.]
    assert body['generationConfig']['responseJsonSchema']['additionalProperties'] is False


def test_axis_order_original_pixel_conversion_and_thought_exclusion():
    parsed = module.parse_vertex_response(vertex(), late_frame_index=99, width=160, height=120)
    assert parsed['object_bbox'] == [72, 36, 128, 84]
    assert parsed['person_bbox'] == [16, 12, 64, 108]
    assert parsed['quality_verified'] is False and parsed['model_version'] == module.MODEL


def test_abstention_no_invisible_box_prediction():
    value = output(); value['object'].update(same_physical_identity=False, visible=False, box_2d=None)
    parsed = module.parse_response(json.dumps(value), late_frame_index=99, width=160, height=120)
    assert parsed['object_bbox'] is None


@pytest.mark.parametrize('change', [
    lambda o: o.update(frame_index=98),
    lambda o: o.update(manual_instruction='fix this video'),
    lambda o: o['object'].update(same_physical_identity=1),
    lambda o: o['object'].update(visible=False),
    lambda o: o['object'].update(box_2d=[0, 0, 1001, 1]),
    lambda o: o['object'].update(box_2d=[0, 0, 0, 0]),
    lambda o: o['object'].update(box_2d=None),
])
def test_wrong_frame_extra_keys_boolean_lie_and_boxes_fail_closed(change):
    value = output(); change(value)
    with pytest.raises(ValueError): module.parse_response(json.dumps(value), late_frame_index=99, width=160, height=120)


def test_reject_duplicates_and_fences():
    text = json.dumps(output())
    with pytest.raises(ValueError): module.parse_response('```json\n' + text + '\n```', late_frame_index=99, width=160, height=120)
    with pytest.raises(ValueError): module.parse_response(text.replace('"frame_index": 99', '"frame_index":99,"frame_index":99'),
        late_frame_index=99, width=160, height=120)


@pytest.mark.parametrize('frame', [0, 14, 29, True, -1])
def test_only_one_genuinely_late_frame(frame):
    with pytest.raises(ValueError): module.response_schema(frame)


def test_corroboration_preserves_native_box_never_returns_mask():
    raw, binding = request(); parsed = module.parse_vertex_response(vertex(), late_frame_index=99, width=160, height=120)
    native = [72, 36, 128, 84]
    result = module.corroborate(parsed, binding=binding, native_object_bbox=native,
        native_person_bbox=[16, 12, 64, 108], saved_forward_tracking_sha256='c' * 64,
        policy=module.AnchorPolicy(.5, .5, 'external:manufactured-contracts-not-accuracy'))
    assert result['accepted'] and result['box'] == native and result['object_id'] == 1
    assert result['native_mask_modified'] is False and result['mask_generated'] is False
    assert result['semantic_response_is_not_mask_accuracy']


def test_different_actor_or_object_is_not_seed_identity():
    _, binding = request(); parsed = module.parse_vertex_response(vertex(), late_frame_index=99, width=160, height=120)
    result = module.corroborate(parsed, binding=binding, native_object_bbox=[1, 1, 10, 10],
        native_person_bbox=[16, 12, 64, 108], saved_forward_tracking_sha256='c' * 64,
        policy=module.AnchorPolicy(.5, .5, 'external:manufactured-contracts-not-accuracy'))
    assert not result['accepted'] and result['box'] is None


def test_no_content_retry_after_first_invalid_vertex_answer():
    raw, binding = request(); calls = []
    def transport(payload, timeout): calls.append(payload); return vertex(finish='MAX_TOKENS')
    with pytest.raises(ValueError): module.verify_one_request(raw, transport, binding=binding, deadline=time.monotonic() + 30)
    assert calls == [raw]


def test_one_successful_request_ledger_has_no_auth_or_images():
    raw, binding = request(); calls = []
    def transport(payload, timeout): calls.append(payload); return vertex()
    parsed, receipt = module.verify_one_request(raw, transport, binding=binding, deadline=time.monotonic() + 30)
    assert calls == [raw] and parsed['object_bbox'] == [72, 36, 128, 84]
    assert receipt['semantic_requests'] == 1 and receipt['model_content_retries'] == 0
    assert receipt['credential_read'] is False and 'Authorization' not in json.dumps(receipt)


def test_original_request_hash_required_before_transport():
    raw, binding = request(); calls = []
    with pytest.raises(ValueError): module.verify_one_request(raw + b' ', lambda *a: calls.append(a),
        binding=binding, deadline=time.monotonic() + 30)
    assert calls == []


@pytest.mark.parametrize('body', [[], {'candidates': [None]}, {'candidates': [dict(finishReason='STOP', content=[])]},
    {'candidates': [dict(finishReason='STOP', content={'parts': [dict(text=1)]})]},
    {'candidates': [dict(finishReason='STOP', content={'parts': [dict(text='{}', thought=1)]})]}])
def test_malformed_vertex_envelope_is_not_an_unhandled_type_error(body):
    with pytest.raises(ValueError): module.parse_vertex_response(json.dumps(body).encode(),
        late_frame_index=99, width=160, height=120)


def test_pure_module_imports_in_host_without_site_packages():
    root = Path(__file__).resolve().parents[1]
    code = r'''import sys;from pathlib import Path
r=Path(sys.argv[1]);sys.path[:0]=[str(r/'infra'),str(r/'src')]
import sam31_late_anchor as m
assert 'numpy' not in sys.modules and 'torch' not in sys.modules
assert m.MODEL=='gemini-3.5-flash'
print('stdlib-only')'''
    result = subprocess.run(['rtk', 'proxy', sys.executable, '-I', '-S', '-B', '-c', code, str(root)],
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'stdlib-only'
