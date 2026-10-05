"""Synthetic receipts/layer IDs only; no Docker, Azure, models or image data."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'infra'))
import articulated_point_study as study
import articulated_point_tracks as tracks
import articulated_runtime_transfer as transfer
import mediapipe_cpu_runtime_verify as rt


def identity(raw):
    return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


@pytest.fixture
def replica():
    """These tiny made-up rootfs IDs are not an actual runtime qualification."""
    pins = json.loads((ROOT/'configs/articulated_runtime_replica_pins.json').read_text())
    layers = ['sha256:'+hashlib.sha256(f'synthetic-layer-{i}'.encode()).hexdigest() for i in range(47)]
    pins['ordered_rootfs_sha256'] = hashlib.sha256(json.dumps(layers, separators=(',', ':')).encode()).hexdigest()
    receipt = dict(schema='world_reward.articulated_runtime_replica.v1', stage='articulated_runtime_import',
        status='pass', phase='complete', replica_only=True, model_execution=False, gpu_used=False,
        dataset_transferred=False, secrets_recorded=False, image_rebuilt=False, image_retagged=False,
        original_runtime_receipt_replayed=False, scratch_removed=True, asset_files=9,
        producer_revision=pins['producer_revision'], source_proof=pins['source_proof'],
        original_export_source_proof=pins['source_proof'], archive=pins['archive'], assets=pins['assets'],
        elapsed_seconds=1.25, image=dict(image_id=study.BOOT_IMAGE, layers=47,
            ordered_rootfs_sha256=pins['ordered_rootfs_sha256'], **pins['image_archive']),
        imported_runtime=dict(image_id=pins['actual_image_id'], qualified_config_id=study.BOOT_IMAGE,
            layers=47, ordered_rootfs_sha256=pins['ordered_rootfs_sha256'], size=123),
        image_graph=dict(config_id=study.BOOT_IMAGE, platform_manifest_id=pins['platform_manifest_id'],
            index_ids=pins['index_ids'], layer_count=47, rootfs_diff_ids=layers,
            ordered_rootfs_sha256=pins['ordered_rootfs_sha256'], pinned_legacy_metadata_verified=True,
            legacy_ids_derived=False, legacy_metadata_manifest_sha256=pins['legacy_metadata_manifest_sha256']))
    pins['report'] = identity((json.dumps(receipt, sort_keys=True)+'\n').encode())
    binding = dict(path=str(study.replica_receipt_path(pins)), identity=pins['report'], receipt=receipt)
    images = {stage: dict(Id=pins['actual_image_id'] if stage == 'tracks' else study.IMAGES[stage],
        Architecture='amd64', Os='linux', RootFS=dict(Type='layers', Layers=copy.deepcopy(layers)))
        for stage in study.STAGES}
    return pins, binding, dict(runtime_replica=binding, images=images)


def test_real_replica_pins_are_whole_file_bound_to_original_byte_census():
    raw = (ROOT/'configs/articulated_runtime_replica_pins.json').read_bytes()
    assert identity(raw) == study.REPLICA_PIN
    pins = json.loads(raw)
    archive = transfer.archive_pins(ROOT)
    assert pins['assets'] == transfer.asset_pins(ROOT)
    assert pins['qualified_config_id'] == study.BOOT_IMAGE
    assert pins['actual_image_id'] == pins['platform_manifest_id'] != study.BOOT_IMAGE
    for key in ('platform_manifest_id', 'ordered_rootfs_sha256', 'legacy_metadata_manifest_sha256', 'image_archive'):
        assert pins[key] == archive[key]
    assert pins['report'] == dict(bytes=7487,
        sha256='864f7b2c0526afdacd7bb51de929c7549ec78e908efe1be18f67b88f4029b5d8')
    assert pins['producer_revision'] == '226c781ac4421c405d17af0e8839e2eba16fdca2'
    assert pins['source_proof']['files'] == 211
    assert pins['archive']['bytes'] == 26572001280


def test_pinned_configuration_rejects_mutated_source_bytes(tmp_path):
    path = tmp_path/'configs/articulated_runtime_replica_pins.json'
    path.parent.mkdir(); path.write_bytes((ROOT/'configs/articulated_runtime_replica_pins.json').read_bytes())
    path.chmod(0o444)
    assert study.replica_configuration(rt, tmp_path)['actual_image_id'] != study.BOOT_IMAGE
    path.chmod(0o600); path.write_bytes(path.read_bytes()+b' '); path.chmod(0o444)
    with pytest.raises(ValueError): study.replica_configuration(rt, tmp_path)


def test_valid_synthetic_import_selects_only_actual_docker_id(replica, monkeypatch):
    pins, binding, proof = replica
    assert study.validate_replica(rt, binding['receipt'], pins) == pins['actual_image_id']
    assert study.validate_replica_binding(rt, binding, pins) == pins['actual_image_id']
    monkeypatch.setattr(study, 'replica_configuration', lambda *_: pins)
    for stage in study.STAGES:
        expected = pins['actual_image_id'] if stage == 'tracks' else study.IMAGES[stage]
        assert study.runtime_image_id(rt, ROOT, proof, stage) == expected


@pytest.mark.parametrize('field,bad', [
    ('schema', 'foreign'), ('stage', 'articulated_runtime_export'), ('status', 'fail'),
    ('phase', 'image_load'), ('replica_only', 1), ('model_execution', True), ('gpu_used', True),
    ('dataset_transferred', True), ('secrets_recorded', True), ('image_rebuilt', True),
    ('image_retagged', True), ('original_runtime_receipt_replayed', True),
    ('scratch_removed', False), ('asset_files', True), ('producer_revision', 'f'*40),
])
def test_import_must_be_typed_complete_original_receipt(replica, field, bad):
    pins, binding, _ = replica
    changed = copy.deepcopy(binding['receipt']); changed[field] = bad
    with pytest.raises(ValueError): study.validate_replica(rt, changed, pins)


@pytest.mark.parametrize('fault', [
    'source', 'export_source', 'archive', 'asset', 'config', 'platform', 'index', 'layer_count',
    'missing_layer', 'reordered_layers', 'foreign_layer', 'layer_format', 'rootfs_sha',
    'legacy_unverified', 'legacy_derived', 'legacy_sha', 'actual_alias', 'actual_foreign',
    'actual_config', 'actual_count', 'actual_rootfs', 'image_bytes', 'image_sha',
])
def test_original_graph_assets_and_actual_id_are_not_replaceable(replica, fault):
    pins, binding, _ = replica
    changed = copy.deepcopy(binding['receipt']); graph = changed['image_graph']; actual = changed['imported_runtime']
    if fault == 'source': changed['source_proof'] = dict(files=1, sha256='a'*64)
    elif fault == 'export_source': changed['original_export_source_proof'] = {}
    elif fault == 'archive': changed['archive']['bytes'] += 1
    elif fault == 'asset': changed['assets'].pop(next(iter(changed['assets'])))
    elif fault == 'config': graph['config_id'] = pins['actual_image_id']
    elif fault == 'platform': graph['platform_manifest_id'] = study.BOOT_IMAGE
    elif fault == 'index': graph['index_ids'] = []
    elif fault == 'layer_count': graph['layer_count'] = True
    elif fault == 'missing_layer': graph['rootfs_diff_ids'].pop()
    elif fault == 'reordered_layers': graph['rootfs_diff_ids'].reverse()
    elif fault == 'foreign_layer': graph['rootfs_diff_ids'][0] = 'sha256:'+'a'*64
    elif fault == 'layer_format': graph['rootfs_diff_ids'][0] = 'a'*64
    elif fault == 'rootfs_sha': graph['ordered_rootfs_sha256'] = 'a'*64
    elif fault == 'legacy_unverified': graph['pinned_legacy_metadata_verified'] = 1
    elif fault == 'legacy_derived': graph['legacy_ids_derived'] = True
    elif fault == 'legacy_sha': graph['legacy_metadata_manifest_sha256'] = 'a'*64
    elif fault == 'actual_alias': actual['image_id'] = study.BOOT_IMAGE
    elif fault == 'actual_foreign': actual['image_id'] = 'sha256:'+'a'*64
    elif fault == 'actual_config': actual['qualified_config_id'] = pins['actual_image_id']
    elif fault == 'actual_count': actual['layers'] = True
    elif fault == 'actual_rootfs': actual['ordered_rootfs_sha256'] = 'a'*64
    elif fault == 'image_bytes': changed['image']['bytes'] += 1
    elif fault == 'image_sha': changed['image']['sha256'] = 'a'*64
    with pytest.raises(ValueError): study.validate_replica(rt, changed, pins)


@pytest.mark.parametrize('fault', ['path', 'identity', 'extra', 'ignored_receipt_field'])
def test_offline_binding_authenticates_original_path_identity_and_complete_bytes(replica, fault):
    pins, binding, _ = replica
    changed = copy.deepcopy(binding)
    if fault == 'path': changed['path'] = changed['path'].replace('/report.json', '/replay.json')
    elif fault == 'identity': changed['identity']['sha256'] = 'a'*64
    elif fault == 'extra': changed['replay'] = True
    else: changed['receipt']['elapsed_seconds'] += 1
    with pytest.raises(ValueError): study.validate_replica_binding(rt, changed, pins)


@pytest.mark.parametrize('fault', ['id', 'rootfs_order', 'rootfs_value', 'architecture', 'os', 'rootfs_type'])
def test_actual_docker_inspect_cannot_be_swapped_after_pinned_import(replica, monkeypatch, fault):
    pins, _, proof = replica
    monkeypatch.setattr(study, 'replica_configuration', lambda *_: pins)
    changed = copy.deepcopy(proof); value = changed['images']['tracks']
    if fault == 'id': value['Id'] = study.BOOT_IMAGE
    elif fault == 'rootfs_order': value['RootFS']['Layers'].reverse()
    elif fault == 'rootfs_value': value['RootFS']['Layers'][0] = 'sha256:'+'a'*64
    elif fault == 'architecture': value['Architecture'] = 'arm64'
    elif fault == 'os': value['Os'] = 'windows'
    else: value['RootFS']['Type'] = 'foreign'
    with pytest.raises(ValueError): study.runtime_image_id(rt, ROOT, changed, 'tracks')


def test_actual_identity_propagates_into_container_environment_receipt_and_owned_cleanup():
    tree = ast.parse((ROOT/'infra/articulated_point_study.py').read_text())
    run = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'run_stage')
    assert any(isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'actual_image' for t in node.targets)
        and isinstance(node.value, ast.Call) and node.value.func.id == 'runtime_image_id' for node in run.body)
    cleanup = next(node for node in ast.walk(run) if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute) and node.func.attr == 'cleanup')
    assert any(k.arg == 'image' and isinstance(k.value, ast.Name) and k.value.id == 'actual_image' for k in cleanup.keywords)
    source = ast.get_source_segment((ROOT/'infra/articulated_point_study.py').read_text(), run)
    assert "'/usr/bin/env', actual_image" in source
    assert "'WR_IMAGE_ID='+actual_image" in source
    assert "value['image_id'] == actual_image" in source
    tracker = ast.parse((ROOT/'infra/articulated_point_tracks.py').read_text())
    native = next(node for node in tracker.body if isinstance(node, ast.FunctionDef) and node.name == 'native')
    resolve = next(node for node in ast.walk(native) if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute) and node.func.attr == 'runtime_image_id')
    torch_import = next(node for node in native.body if isinstance(node, ast.Import)
        and any(alias.name == 'torch' for alias in node.names))
    assert resolve.lineno < torch_import.lineno
    assert 'infra/articulated_point_study.py' in tracks.SOURCE_HELPERS


def test_immutable_scientific_protocol_not_changed_by_runtime_compatibility_adapter():
    assert identity((ROOT/study.PROTOCOL).read_bytes()) == study.PROTOCOL_PIN
    assert study.PROTOCOL_PIN['sha256'] == 'd1a3fd38cfd0496fd22a2a0280ff95cbc684d866eb345beb525bbbece921db0c'
    assert study.BUDGETS == dict(manufacture=600, tracks=900, fit=10800)
