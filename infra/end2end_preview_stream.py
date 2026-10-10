"""On-demand, memory-only playback of sealed three-column Azure diagnostics.

This protocol is separate from the older full4d receipts. QA failures remain
visible; neither a visual comparison nor a proxy QA pass verifies 4D accuracy.
Only the tiny receipt is local. Playback consumes the selected compressed bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import html
from pathlib import Path
import re
import stat
import threading
import time

import full4d_stream as base

SCHEMA = 'world_reward.end2end_preview_publication.v1'
BASELINE = '052ba1554e9a573d566713a99a61d89a5f27681c'
COHORT, COMPLETE = [9, 1, 14, 7], [9, 14]
MAX_RECEIPT = 64 << 10
HELPERS = {'infra/end2end_preview.py', 'infra/run_end2end_preview.sh',
           'infra/full4d_video.py', 'infra/full4d_publish.py',
           'infra/mediapipe_cpu_runtime_verify.py'}
require = base.require


def revision(value):
    require(type(value) is str and re.fullmatch('[0-9a-f]{40}', value), 'Exact revision required')
    return value


def pin(value, maximum):
    require(type(value) is dict and set(value) == {'bytes', 'sha256'}
            and type(value['bytes']) is int and 0 < value['bytes'] <= maximum
            and type(value['sha256']) is str and re.fullmatch('[0-9a-f]{64}', value['sha256']),
            'Bounded original byte/SHA identity required')


def records(receipt, producer_revision, *, render_revision=None, candidate_revision=None):
    """Accept exactly the frozen cohort and two actual, unfiltered candidates."""
    revision(producer_revision)
    fields = {'schema', 'status', 'producer_revision', 'render_revision', 'candidate_revision',
              'baseline_revision', 'render_report_pin', 'cohort', 'states', 'files', 'source_binding',
              'endpoint', 'private_container_verified', 'public_access_changed', 'account_keys_used',
              'quality_verified', 'heavy_data_uploaded'}
    require(type(receipt) is dict and set(receipt) == fields and receipt['schema'] == SCHEMA
            and receipt['status'] == 'pass' and receipt['producer_revision'] == producer_revision
            and receipt['baseline_revision'] == BASELINE and receipt['endpoint'] == base.ENDPOINT
            and receipt['private_container_verified'] is True and receipt['public_access_changed'] is False
            and receipt['account_keys_used'] is False and receipt['quality_verified'] is False
            and receipt['heavy_data_uploaded'] is False, 'Original private non-quality publication required')
    for key, expected in (('render_revision', render_revision), ('candidate_revision', candidate_revision)):
        revision(receipt[key])
        require(expected is None or receipt[key] == revision(expected), 'Pinned upstream revision differs')
    pin(receipt['render_report_pin'], 4 << 20)
    binding = receipt['source_binding']
    require(type(binding) is dict and set(binding) == {'producer_revision', 'markers', 'entries', 'helpers', 'closure_sha256'}
            and binding['producer_revision'] == producer_revision
            and type(binding['entries']) is int and 6 <= binding['entries'] <= 10000
            and type(binding['closure_sha256']) is str and re.fullmatch('[0-9a-f]{64}', binding['closure_sha256'])
            and type(binding['helpers']) is dict and set(binding['helpers']) == HELPERS
            and type(binding['markers']) is dict and set(binding['markers']) == {'revision', 'source-sha256'},
            'Original publication source closure required')
    for value in binding['helpers'].values(): pin(value, 2_000_000)
    for name, length in (('revision', 41), ('source-sha256', 65)):
        pin(binding['markers'][name], length)
        require(binding['markers'][name]['bytes'] == length, 'Original immutable marker length required')
    require(type(receipt['cohort']) is list and receipt['cohort'] == COHORT
            and all(type(ep) is int for ep in receipt['cohort']), 'Frozen random cohort required')
    states = receipt['states']
    require(type(states) is list and len(states) == len(COHORT), 'All four statuses required')
    for ep, row in zip(COHORT, states):
        require(type(row) is dict and type(row.get('episode')) is int and row['episode'] == ep,
                'Original cohort order required')
        if ep in COMPLETE:
            require(set(row) == {'episode', 'status', 'QA_passed'} and row['status'] == 'complete'
                    and type(row['QA_passed']) is bool, 'Both completed candidates, including QA failures, required')
        else:
            require(row.get('status') in {'failed', 'not_reconstructed_in_this_ablation'}
                    and {'episode', 'status', 'reason'} <= set(row)
                    and set(row) <= {'episode', 'status', 'reason', 'phase'}
                    and type(row['reason']) is str and 0 < len(row['reason']) <= 512
                    and (row.get('phase') is None or type(row['phase']) is str and len(row['phase']) <= 128),
                    'Unsupported/failed episodes must remain in the denominator')
    rows = receipt['files']
    require(type(rows) is list and len(rows) == 4, 'Exactly two MP4/JPEG pairs required')
    allowed = {}
    for row in rows:
        require(type(row) is dict and set(row) == {'name', 'etag', 'mime', 'bytes', 'sha256', 'episode_index'}
                and type(row['episode_index']) is int and row['episode_index'] in COMPLETE,
                'Only original completed preview records allowed')
        ep = row['episode_index']
        for ext, mime, cap in (('mp4', 'video/mp4', base.MAX_VIDEO), ('jpg', 'image/jpeg', base.MAX_POSTER)):
            if row['name'] == f'full4d-{producer_revision}/episode_{ep:06d}.{ext}':
                path = f'/episode-{ep}.{ext}'
                pin({key: row[key] for key in ('bytes', 'sha256')}, cap)
                require(row['mime'] == mime and type(row['etag']) is str
                        and re.fullmatch(r'"[A-Za-z0-9-]{1,100}"', row['etag']) and path not in allowed,
                        'Unique pinned MIME/ETag media identity required')
                allowed[path] = row
                break
        else:
            raise ValueError('Only exact private publication media names allowed')
    require(set(allowed) == {f'/episode-{ep}.{ext}' for ep in COMPLETE for ext in ('mp4', 'jpg')},
            'Missing original completed preview pair')
    return states, allowed


def load_receipt(path, producer_revision, **upstream):
    path = Path(path)
    require(path.is_absolute() and path.resolve() == path
            and not any(p.is_symlink() for p in (path, *path.parents)), 'Canonical sealed tiny receipt required')
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode) and not before.st_mode & 0o222 and before.st_nlink == 1
            and 0 < before.st_size <= MAX_RECEIPT, 'Only immutable bounded receipt metadata may be local')
    with path.open('rb') as stream: raw = stream.read(MAX_RECEIPT + 1)
    after = path.lstat()
    require(len(raw) == before.st_size and all(getattr(before, key) == getattr(after, key) for key in
            ('st_dev', 'st_ino', 'st_mode', 'st_nlink', 'st_size', 'st_mtime_ns', 'st_ctime_ns')),
            'Receipt changed during bounded read')
    states, allowed = records(base.strict(raw), producer_revision, **upstream)
    return states, allowed, hashlib.sha256(raw).hexdigest()


def index_html(states):
    sections = []
    for row in states:
        ep = row['episode']
        if row['status'] == 'complete':
            qa = 'QA PASS — GT pending' if row['QA_passed'] else 'QA FAIL — not adopted'
            content = ('<div class="columns"><span>Original RGB</span><span>Baseline complete (052)</span>'
                       f'<span>Native joint — {html.escape(qa)}</span></div>'
                       f'<video controls playsinline preload="none" aria-label="Épisode {ep}: original, baseline 052, candidat natif">'
                       f'<source src="/episode-{ep}.mp4" type="video/mp4"></video>'
                       f'<p><a href="/episode-{ep}.jpg" target="_blank" rel="noopener noreferrer">'
                       'Voir la planche fixe à la demande</a></p>')
        else:
            content = '<p>Échec / non reconstruit dans cette ablation — conservé dans le bilan.</p>'
        sections.append(f'<section><h2>Épisode {ep:02d}</h2>{content}</section>')
    return ('<!doctype html><html lang="fr"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>World Reward — diagnostic end2end</title>'
            '<style>:root{color-scheme:light dark}body{font:16px system-ui;max-width:1100px;margin:24px auto;padding:0 16px}'
            'h1,h2{font-weight:500}h1{font-size:24px}h2{font-size:18px}section{margin:28px 0}'
            '.columns{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;font-size:14px}'
            'video{width:100%;display:block}p{line-height:1.5}</style>'
            '<h1>World Reward — comparaison end2end</h1>'
            '<p>États figés : 2/4 reconstructions disponibles, y compris celles qui échouent à la QA. '
            'Qualité 4D et gain face à CARI4D non vérifiés. Même échelle inférée, caméra et sol virtuel fixes.</p>'
            '<p>Trois colonnes synchronisées : original RGB · baseline 052 intacte · candidat natif. '
            'Lecture et planche uniquement à la demande, sans vidéo enregistrée localement. '
            'La lecture consomme les petits aperçus sur votre connexion.</p>'
            '<main>' + ''.join(sections) + '</main></html>').encode()


def make_server(states, allowed, *, reader=None):
    """Reuse the existing same-origin range implementation without changing it."""
    page = index_html(states)
    parent = base.handler_factory([], allowed, reader or base.AzureReader())
    class Handler(parent):
        def serve(self, head=False):
            if self.path != '/':
                return super().serve(head=head)
            try:
                self.valid_local_request()
                self.response_headers(200, 'text/html; charset=utf-8', len(page))
                if not head: self.wfile.write(page)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception:
                self.failure(502); self.close_connection = True
    server = base.Proxy(('127.0.0.1', 0), Handler)
    server.deadline = time.monotonic() + base.MAX_TTL
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--render-revision', required=True)
    parser.add_argument('--candidate-revision', required=True)
    parser.add_argument('--ttl', type=int, default=base.MAX_TTL)
    args = parser.parse_args()
    require(60 <= args.ttl <= base.MAX_TTL, 'Viewer TTL must be between60and3600seconds')
    states, allowed, _ = load_receipt(args.receipt, args.revision,
        render_revision=args.render_revision, candidate_revision=args.candidate_revision)
    server = make_server(states, allowed); server.deadline = time.monotonic() + args.ttl
    timer = threading.Timer(args.ttl, server.shutdown); timer.daemon = True; timer.start()
    print(f'END2END_STREAM_URL http://127.0.0.1:{server.server_port}/', flush=True)
    try: server.serve_forever(poll_interval=.5)
    except KeyboardInterrupt: pass
    finally: timer.cancel(); server.server_close()


if __name__ == '__main__':
    main()
