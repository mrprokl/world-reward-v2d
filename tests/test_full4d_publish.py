"""Synthetic tiny publication only, no Azure call or media execution."""
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'infra'))
import full4d_publish as publish
import mediapipe_cpu_runtime_verify as runtime


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

    def delete_owned(self,row):
        self.head(row['name'],row,row['etag']); self.deleted=getattr(self,'deleted',[])+[row]


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


def test_private_properties_need_no_acl_permission_and_metadata_name_is_valid(monkeypatch):
    from types import SimpleNamespace
    import re
    client = publish.PrivatePreviews()
    monkeypatch.setattr(client, 'authorization', lambda: {})
    requests = []
    class Response:
        status = 200
        headers = {}
        def __enter__(self): return self
        def __exit__(self, *_): pass
    monkeypatch.setattr(client.opener, 'open', lambda request, **_: requests.append(request) or Response())
    client.require_private()
    assert requests[0].method == 'HEAD'
    assert requests[0].full_url == publish.ENDPOINT+'?restype=container'
    sent = []
    response = Response(); response.status = 201; response.headers = {'ETag':'"tiny"'}
    monkeypatch.setattr(client, 'request', lambda *args, **kwargs: sent.append((args, kwargs)) or response)
    client.upload('full4d-'+('a'*40)+'/episode_000001.jpg', b'tiny', 'image/jpeg', 'a'*40)
    headers = sent[0][0][3]
    assert all(re.fullmatch('[A-Za-z_][A-Za-z0-9_]*', k.removeprefix('x-ms-meta-'))
               for k in headers if k.startswith('x-ms-meta-'))


def partial_fixture(monkeypatch,tmp_path,complete=(9,)):
    (tmp_path/'results').mkdir()
    numerical, experiment=fixtures(tmp_path,episodes=complete)
    publication='c'*40; code=tmp_path/'jobs'/publication/publish.PUBLISH_ENTRY/'code'
    numeric_code=tmp_path/'jobs'/numerical/publish.NUMERICAL_ENTRY/'code'
    def write(path,value):
        path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists(): path.chmod(0o644)
        path.write_bytes(json.dumps(value).encode() if type(value) is dict else value)
        path.chmod(0o444);return publish.identity(path,8<<20)
    write(code/'infra/full4d_publish.py',b'# actual tiny publisher fixture\n')
    write(numeric_code/'infra/cari_full_export.py',b'# actual tiny export fixture\n')
    write(numeric_code/'infra/full4d_video.py',b'# actual tiny renderer fixture\n')
    monkeypatch.setattr(publish,'__file__',str(code/'infra/full4d_publish.py'))
    def source(root,current,rev,entry,helpers):
        assert current==root/'jobs'/rev/entry/'code'
        return dict(producer_revision=rev,closure_sha256='d'*64,
            helpers={'infra/full4d_publish.py' if entry==publish.PUBLISH_ENTRY else 'infra/gemini_full4d.py':dict(bytes=1,sha256='e'*64)})
    monkeypatch.setattr(runtime,'source',source)
    states=[]
    for ep in publish.COHORT:
        state=dict(episode=ep,original_frames=3,status='pending')
        if ep==7:state['status']='fail'
        if ep in complete:
            state['status']='complete_full4d_visual_diagnostic_not_quality_pass'
            spec=dict(episode_index=ep,total_frames=3,camera_name='front_stereo_camera_left',height=1152,width=1536)
            export=experiment/'outputs'/f'episode_{ep:06d}/cari_shared_export_v1'
            output_files={name:write(export/name,b'tiny original native array fixture') for name in publish.EXPORT_FILES-{'report.json'}}
            native=dict(stage='world_reward_native_cari_shared_full_video_direct_export',status='pass',phase='complete',
                producer_revision=numerical,episode_index=ep,frames=3,input_track='track_1',ground_truth_used=False,
                ground_truth_read=False,private_truth_read=False,hand_labeled_test=False,oracle_modes=[],
                unchanged_refined_predictions_verified=True,full_original_native_export_verified=True,
                source_inputs_assets_rehashed=True,source_helpers_rehashed=True,original_frame_indices=[0,1,2],
                clip_spec=spec,script_sha256=publish.identity(numeric_code/'infra/cari_full_export.py',1<<20)['sha256'],output_files=output_files)
            native_pin=write(export/'report.json',native)
            pinpath=experiment/'pins'/f'cari_clip_{ep:06d}_shared_export_pins.json'
            pins=dict(schema='world-reward-cari-shared-export-pins-v1',clip_spec=spec,
                export=native_pin | dict(producer_revision=numerical,script_sha256=native['script_sha256']),
                export_files=output_files | {'report.json':native_pin})
            pin=write(pinpath,pins)
            renderer_path=experiment/'videos'/f'episode_{ep:06d}/report.json'
            renderer=json.loads(renderer_path.read_text())
            renderer['sources']={str(export/name):row for name,row in pins['export_files'].items()}
            renderer['sources'][str(pinpath)]=pin
            renderer['sources'][str(numeric_code/'infra/full4d_video.py')]=publish.identity(numeric_code/'infra/full4d_video.py',1<<20)
            state['video_report']=write(renderer_path,renderer)
        states.append(state)
    report=dict(schema='world_reward.gemini_full4d.v1',producer_revision=numerical,
        source_binding=source(tmp_path,numeric_code,numerical,publish.NUMERICAL_ENTRY,()),status='running',
        sampling=dict(population=30,seed=20261008,count=4,replacement=False),episodes=states,
        ground_truth_used=False,hand_labeled_test=False,quality_verified=False,challenge_performance_verified=False,
        jitter_fix_claimed=False,baseline_modified=False,oracle_modes=[])
    write(experiment/'report.json',report)
    (experiment/'report.json').chmod(0o644) # Actual numeric producer remains active/mutable.
    return numerical,publication,code,experiment,write


@pytest.mark.parametrize('complete',[(9,),(9,14)])
def test_partial_publishes_all_actual_complete_clips_and_preserves_numeric_job(monkeypatch,tmp_path,complete):
    numerical,publication,code,experiment,_=partial_fixture(monkeypatch,tmp_path,complete)
    original=(experiment/'report.json').read_bytes(); client=Client()
    result=publish.publish_partial(tmp_path,numerical,code,publication,client=client)
    assert result['episodes']==list(complete) and result['cohort']==[9,1,14,7] and result['cohort_denominator']==4
    assert result['producer_revision']==publication and result['numerical_producer_revision']==numerical
    assert result['status']=='pass' and result['quality_verified'] is False and result['live_status_claimed'] is False
    assert len(client.records)==2*len(complete)+1 and all(row['name'].startswith('full4d-'+publication+'/') for row in client.records)
    assert (experiment/'report.json').read_bytes()==original and (experiment/'report.json').stat().st_mode&0o777==0o644
    saved=tmp_path/'results'/f'full4d-partial-{publication}'
    assert (saved/'numerical-report.json').read_bytes()==original
    assert set(path.name for path in saved.iterdir())=={'report.json','numerical-report.json'}
    assert not any(path.stat().st_mode&0o222 for path in saved.iterdir())
    with pytest.raises(ValueError):publish.publish_partial(tmp_path,numerical,code,publication,client=Client())


@pytest.mark.parametrize('mutation',['cohort','cohort_order','producer','source_binding','render_missing','render_extra',
    'export_missing','export_extra','export_producer','export_gt','export_tail','render_tail','source_inventory',
    'export_pin','render_not_pass','wrong_total','no_complete','three_complete'])
def test_partial_requires_source_bound_full_t_success_before_any_publication(monkeypatch,tmp_path,mutation):
    numerical,publication,code,experiment,write=partial_fixture(monkeypatch,tmp_path)
    reportpath=experiment/'report.json'; producer=json.loads(reportpath.read_text())
    renderdir=experiment/'videos/episode_000009'; renderpath=renderdir/'report.json'
    exportdir=experiment/'outputs/episode_000009/cari_shared_export_v1'; exportpath=exportdir/'report.json'
    if mutation=='cohort':producer['episodes'][1]['episode']=2
    elif mutation=='cohort_order':producer['episodes'].reverse()
    elif mutation=='producer':producer['producer_revision']='f'*40
    elif mutation=='source_binding':producer['source_binding']['closure_sha256']='f'*64
    elif mutation=='wrong_total':producer['episodes'][0]['original_frames']=4
    elif mutation=='no_complete':producer['episodes'][0]['status']='pending'
    elif mutation=='three_complete':
        for row in producer['episodes'][:3]:row['status']='complete_full4d_visual_diagnostic_not_quality_pass'
    elif mutation=='render_missing':(renderdir/'episode_000009.jpg').unlink()
    elif mutation=='render_extra':write(renderdir/'unexpected.npz',b'not a preview')
    elif mutation=='export_missing':(exportdir/'trajectory.npz').unlink()
    elif mutation=='export_extra':write(exportdir/'unexpected.npz',b'not an export')
    elif mutation in {'export_producer','export_gt','export_tail'}:
        native=json.loads(exportpath.read_text())
        if mutation=='export_producer':native['producer_revision']='f'*40
        elif mutation=='export_gt':native['ground_truth_used']=True
        else:native['original_frame_indices']=[0,1]
        write(exportpath,native)
    elif mutation in {'render_tail','source_inventory','render_not_pass'}:
        renderer=json.loads(renderpath.read_text())
        if mutation=='render_tail':renderer['original_frame_indices']=[0,1]
        elif mutation=='source_inventory':renderer['sources'].pop(str(exportdir/'trajectory.npz'))
        else:renderer['status']='fail'
        producer['episodes'][0]['video_report']=write(renderpath,renderer)
    elif mutation=='export_pin':
        pinpath=experiment/'pins/cari_clip_000009_shared_export_pins.json'
        pins=json.loads(pinpath.read_text());pins['export']['producer_revision']='f'*40;write(pinpath,pins)
    write(reportpath,producer);client=Client()
    with pytest.raises(ValueError):publish.publish_partial(tmp_path,numerical,code,publication,client=client)
    assert client.records==[] and not (tmp_path/'results'/f'full4d-partial-{publication}').exists()


@pytest.mark.parametrize('mutation',['foreign_namespace','extra','duplicate','missing','etag','mime','episode','owner'])
def test_partial_cleanup_refuses_foreign_or_unbound_blob_before_cloud_actions(monkeypatch,tmp_path,mutation):
    numerical,publication,code,experiment,write=partial_fixture(monkeypatch,tmp_path)
    result=publish.publish_partial(tmp_path,numerical,code,publication,client=Client())
    if mutation=='foreign_namespace':result['files'][0]['name']=result['files'][0]['name'].replace(publication,numerical)
    elif mutation=='extra':result['files'].append(dict(name='foreign/model.npz'))
    elif mutation=='duplicate':result['files'].append(result['files'][0])
    elif mutation=='missing':result['files'].pop(0)
    elif mutation=='etag':result['files'][0]['etag']='not-an-etag'
    elif mutation=='mime':result['files'][0]['mime']='application/octet-stream'
    elif mutation=='episode':result['files'][0]['episode_index']=14
    elif mutation=='owner':result['publication_source_binding']['producer_revision']='f'*40
    with pytest.raises(ValueError):publish.partial_cleanup_records(result,publication)


def test_partial_conditional_cleanup_retains_exact_receipt_and_accepts_failed_put_ledger(monkeypatch,tmp_path):
    numerical,publication,code,experiment,write=partial_fixture(monkeypatch,tmp_path)
    client=Client();result=publish.publish_partial(tmp_path,numerical,code,publication,client=client)
    saved=tmp_path/'results'/f'full4d-partial-{publication}'/'report.json';original=saved.read_bytes()
    publish.delete_partial(tmp_path,publication,publication_code=code,client=client)
    assert len(client.deleted)==3 and saved.read_bytes()==original
    assert not (experiment/'published.json').exists()
    failed=result | dict(status='fail',files=result['files'][:1])
    assert publish.partial_cleanup_records(failed,publication)==failed['files']


@pytest.mark.parametrize('when',['before','during_upload'])
def test_partial_rehashes_all_render_source_payloads_not_only_export_files(monkeypatch,tmp_path,when):
    numerical,publication,code,experiment,write=partial_fixture(monkeypatch,tmp_path)
    auxiliary=tmp_path/'results/input-manifest.json';pin=write(auxiliary,{'source':'original public input metadata'})
    renderpath=experiment/'videos/episode_000009/report.json';render=json.loads(renderpath.read_text())
    render['sources'][str(auxiliary)]=pin;renderpin=write(renderpath,render)
    reportpath=experiment/'report.json';report=json.loads(reportpath.read_text());report['episodes'][0]['video_report']=renderpin
    write(reportpath,report)
    client=Client()
    if when=='before':write(auxiliary,{'source':'changed'})
    else:
        original=client.upload
        def changed_after_put(*args):
            row=original(*args);write(auxiliary,{'source':'changed'});return row
        client.upload=changed_after_put
    with pytest.raises(ValueError,match='bound source payload changed'):
        publish.publish_partial(tmp_path,numerical,code,publication,client=client)
    if when=='before':assert client.records==[]
    else:
        result=json.loads((tmp_path/'results'/f'full4d-partial-{publication}'/'report.json').read_text())
        assert result['status']=='fail' and len(result['files'])==3


def test_partial_snapshot_does_not_require_active_producer_status_to_remain_unchanged(monkeypatch,tmp_path):
    numerical,publication,code,experiment,write=partial_fixture(monkeypatch,tmp_path)
    reportpath=experiment/'report.json';original=reportpath.read_bytes();client=Client();upload=client.upload
    def advancing(*args):
        row=upload(*args);report=json.loads(reportpath.read_text());report['phase']='another_clip_progress'
        write(reportpath,report);reportpath.chmod(0o644);return row
    client.upload=advancing
    result=publish.publish_partial(tmp_path,numerical,code,publication,client=client)
    assert result['status']=='pass' and reportpath.read_bytes()!=original
    assert (tmp_path/'results'/f'full4d-partial-{publication}'/'numerical-report.json').read_bytes()==original
