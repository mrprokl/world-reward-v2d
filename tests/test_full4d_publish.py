"""Synthetic tiny publication only, no Azure call or media execution."""
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'infra'))
import full4d_publish as publish


def fixtures(tmp_path, episodes=(1, 12, 27)):
    revision = 'a'*40
    experiment = tmp_path/'experiments'/f'full4d-v1-{revision}'
    for episode in episodes:
        directory = experiment/'videos'/f'episode_{episode:06d}'
        directory.mkdir(parents=True)
        report = dict(schema='world_reward.full4d_video.v1', status='pass', producer_revision=revision,
            episode_index=episode, input_track='track_1', ground_truth_used=False, hand_labeled_test=False,
            oracle_modes=[], model_execution=False, optimizer_execution=False, metric_evaluation=False,
            quality_verified=False, original_geometry_unchanged=True, per_frame_alignment=False,
            per_frame_camera=False, per_frame_centring=False, source_rehashed_after=True, fps=30,
            original_frames=3, frames_encoded=3, original_frame_indices=[0, 1, 2])
        for extension, key in (('mp4', 'video'), ('jpg', 'poster')):
            path = directory/f'episode_{episode:06d}.{extension}'
            path.write_bytes(b'manufactured tiny non-media bytes')
            path.chmod(0o444)
            report[key] = publish.identity(path, 100)
        receipt = directory/'report.json'
        receipt.write_text(json.dumps(report)); receipt.chmod(0o444)
    return revision, experiment


class Client:
    def __init__(self, fail_head=False):
        self.records = []
        self.private_checks = 0
        self.fail_head = fail_head

    def require_private(self): self.private_checks += 1

    def upload(self, name, raw, mime, revision):
        assert name.startswith(f'full4d-{revision}/')
        row = dict(name=name, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
            mime=mime, etag=f'"etag-{len(self.records)}"')
        self.records.append(row)
        return row

    def head(self, name, expected, etag):
        if self.fail_head: raise RuntimeError('Manufactured private verification failure')
        assert any(r['name'] == name and r['etag'] == etag for r in self.records)


def test_only_bounded_new_preview_artifacts_allowed(tmp_path):
    revision, experiment = fixtures(tmp_path)
    root, files, reports = publish.preview_records(tmp_path, revision, [1, 12, 27])
    assert root == experiment and len(files) == 6 and len(reports) == 3
    assert {r['mime'] for r in files} == {'video/mp4', 'image/jpeg'}
    assert not any(r['path'].endswith(('.npz', '.npy', '.glb')) for r in files)


@pytest.mark.parametrize('episodes', [[1, 12], [1, 12, 12], [1, 12, True], [1, 12, 30]])
def test_reject_wrong_population_before_any_upload(tmp_path, episodes):
    revision, _ = fixtures(tmp_path)
    with pytest.raises(ValueError): publish.preview_records(tmp_path, revision, episodes)


def test_publisher_never_overwrites_existing_receipt(tmp_path):
    revision, experiment = fixtures(tmp_path)
    client = Client()
    result = publish.publish(tmp_path, revision, [1, 12, 27], client=client)
    assert result['status'] == 'pass' and len(result['files']) == 7
    assert result['account_keys_used'] is False and result['sas_tokens_persisted'] is False
    assert not (experiment/'published.json').stat().st_mode & 0o222
    with pytest.raises(ValueError): publish.publish(tmp_path, revision, [1, 12, 27], client=Client())


def test_successful_put_is_recorded_before_failed_head_for_owned_cleanup(tmp_path):
    revision, experiment = fixtures(tmp_path)
    client = Client(fail_head=True)
    with pytest.raises(RuntimeError): publish.publish(tmp_path, revision, [1, 12, 27], client=client)
    receipt = json.loads((experiment/'published.json').read_text())
    assert receipt['status'] == 'fail' and len(receipt['files']) == 1
    assert receipt['files'][0]['etag'] == '"etag-0"'


def test_tampered_or_writable_rendered_outputs_rejected(tmp_path):
    revision, experiment = fixtures(tmp_path)
    path = experiment/'videos/episode_000001/episode_000001.mp4'
    path.chmod(0o644)
    with pytest.raises(ValueError): publish.preview_records(tmp_path, revision, [1, 12, 27])
    path.write_bytes(b'different'); path.chmod(0o444)
    with pytest.raises(ValueError): publish.preview_records(tmp_path, revision, [1, 12, 27])


def test_nonpreview_endpoint_and_credential_strings_not_accepted():
    client = publish.PrivatePreviews()
    for name in ('../other.mp4', 'runtime-transfers/image.tar', 'full4d-'+('a'*40)+'/model.npz',
            'full4d-'+('a'*40)+'/episode_000001.mp4?sig=secret'):
        with pytest.raises(ValueError): client.request('PUT', name)
