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
CONTINUATION_SCHEMA = 'world_reward.end2end_preview_publication.v2'
SQP_SCHEMA = 'world_reward.end2end_preview_publication.v3'
SMOOTHING_SCHEMA = 'world_reward.end2end_preview_publication.v4'
SQP_STATUSES = {'accepted_native_pose_sqp', 'dynamic_A_fallback_no_improvement'}
SMOOTHING_STATUSES = {'accepted_rigid_projection_pending_independent_QA', 'dynamic_A_fallback_no_improvement'}
CONTINUATION_STATUSES = {'accepted_native_continuation', 'dynamic_A_fallback_no_improvement'}
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


def records(receipt, producer_revision, *, render_revision=None, candidate_revision=None, candidate_kind=None):
    """Accept exactly the frozen cohort and two actual, unfiltered candidates."""
    revision(producer_revision)
    fields = {'schema', 'status', 'producer_revision', 'render_revision', 'candidate_revision',
              'baseline_revision', 'render_report_pin', 'cohort', 'states', 'files', 'source_binding',
              'endpoint', 'private_container_verified', 'public_access_changed', 'account_keys_used',
              'quality_verified', 'heavy_data_uploaded'}
    continuation = type(receipt) is dict and receipt.get('schema') == CONTINUATION_SCHEMA
    sqp = type(receipt) is dict and receipt.get('schema') == SQP_SCHEMA
    smoothing = type(receipt) is dict and receipt.get('schema') == SMOOTHING_SCHEMA
    tagged = continuation or sqp or smoothing
    if tagged: fields.add('candidate_kind')
    kind = 'smoothing' if smoothing else 'sqp' if sqp else 'continuation' if continuation else 'joint'
    outcomes = SMOOTHING_STATUSES if smoothing else SQP_STATUSES if sqp else CONTINUATION_STATUSES
    require(candidate_kind is None or candidate_kind in ('joint', 'continuation', 'sqp', 'smoothing'), 'Explicit valid candidate kind required')
    require(candidate_kind is None or kind == candidate_kind, 'Pinned preview method differs')
    require(type(receipt) is dict and set(receipt) == fields
            and receipt['schema'] == (SMOOTHING_SCHEMA if smoothing else SQP_SCHEMA if sqp else CONTINUATION_SCHEMA if continuation else SCHEMA)
            and (not tagged or receipt['candidate_kind'] == kind)
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
            require(set(row) == ({'episode', 'status', 'QA_passed', 'candidate_status'} if tagged else
                                {'episode', 'status', 'QA_passed'}) and row['status'] == 'complete'
                    and (not tagged or row['candidate_status'] in outcomes)
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


def candidate_label(row):
    """Method identity is separate from QA/adoption; green is never a pass."""
    status = row.get('candidate_status')
    if status == 'dynamic_A_fallback_no_improvement':
        return 'Baseline fallback / no gain — référence dynamique conservée'
    if status == 'accepted_native_pose_sqp':
        return 'C sparse pose-SQP — diagnostic natif'
    if status == 'accepted_native_continuation':
        return 'C contact-continuation — diagnostic natif'
    if status == 'accepted_rigid_projection_pending_independent_QA':
        return 'Lissage contraint — diagnostic, GT à vérifier'
    return 'Native joint — diagnostic natif'


def qa_label(row):
    return ('QA PASS — GT pending · qualité 4D non validée' if row['QA_passed'] else
            'QA FAIL — not adopted · régression à examiner')


def view_path(episode, focus=None):
    return f'/view-episode-{episode}' + (f'/{focus}' if focus else '')


STYLES = '''
:root{color-scheme:light dark;--wr-bg:#f8fafc;--wr-fg:#172033;--wr-muted:#46546b;
--wr-line:#ccd5e1;--wr-blue:#1257b3;--wr-amber:#865000;--wr-green:#096640;--wr-red:#ad2535}
@media(prefers-color-scheme:dark){:root{--wr-bg:#101722;--wr-fg:#edf2f8;--wr-muted:#bac7d8;
--wr-line:#40516a;--wr-blue:#86baff;--wr-amber:#ffd084;--wr-green:#80dfb0;--wr-red:#ffa7b2}}
*{box-sizing:border-box}body{background:var(--wr-bg);color:var(--wr-fg);font:18px/1.55 system-ui;
max-width:1600px;margin:32px auto;padding:0 24px}h1,h2,h3,p{margin:0 0 16px}
h1{font-size:clamp(28px,3.6vw,46px);line-height:1.15;letter-spacing:-.03em}
h2{font-size:clamp(23px,2.2vw,32px);line-height:1.2}h3{font-size:22px;line-height:1.25}
a{color:inherit;text-underline-offset:4px}a:hover{text-decoration-thickness:3px}
.eyebrow{color:var(--wr-muted);font-size:16px;letter-spacing:.07em;text-transform:uppercase}
.subtitle,.fineprint{color:var(--wr-muted)}.fineprint{font-size:16px}
.warning{border-left:5px solid var(--wr-amber);padding-left:16px;margin:24px 0}
.qa{font-weight:650;margin:16px 0 24px}.qa-fail{color:var(--wr-red)}
.clip-list{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:28px;margin:32px 0}
.clip-summary{border-top:2px solid var(--wr-line);padding-top:20px;min-width:0}
.columns{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px;margin:24px 0 12px}
.column-label{min-width:0;border-top:6px solid currentColor;padding-top:14px}
.column-label h2{font-size:clamp(18px,2.1vw,32px);letter-spacing:.02em;margin-bottom:8px}
.column-label p{font-size:16px;margin-bottom:6px}.original{color:var(--wr-blue)}
.column-label .position{display:none}
.baseline{color:var(--wr-amber)}.candidate{color:var(--wr-green)}
.view-nav,.clip-nav{display:flex;flex-wrap:wrap;gap:12px 24px;margin:24px 0}
.view-nav a,.clip-nav a,.clip-summary>a{padding:10px 0;display:inline-block;min-height:44px}
.view-nav [aria-current=page]{font-weight:750;text-decoration-thickness:3px}
video{width:100%;height:auto;aspect-ratio:24/7;display:block;background:var(--wr-bg)}
.focus-view{max-width:960px;margin:24px auto}.focus-view video{aspect-ratio:8/7;object-fit:cover}
.focus-original video{object-position:left center}.focus-baseline video{object-position:center center}
.focus-candidate video{object-position:right center}.focus-heading{border-top:6px solid currentColor;padding-top:16px}
.inspection{margin:24px 0}.inspection strong{display:block;margin-bottom:6px}
details{margin:24px 0}summary{min-height:44px;padding:10px 0}
@media(max-width:700px){body{margin:20px auto;padding:0 16px}.clip-list{grid-template-columns:1fr}
.columns{grid-template-columns:1fr;gap:10px}.column-label{border-top:0;border-left:6px solid currentColor;
padding:4px 0 4px 12px}.column-label h2{font-size:20px;margin:0}.column-label p{display:none}
.column-label .position{display:inline}.view-nav,.clip-nav{gap:6px 18px}}
'''


def document(title, content):
    return ('<!doctype html><html lang="fr"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{html.escape(title)} — World Reward</title>'
            '<style>' + STYLES + '</style></head><body>' + content + '</body></html>').encode()


def index_html(states):
    """Metadata-only landing page: no media element or background Azure read."""
    summaries = []
    complete = sum(row['status'] == 'complete' for row in states)
    for row in states:
        ep = row['episode']
        if row['status'] == 'complete':
            detail = (f'<p>{html.escape(candidate_label(row))}</p>'
                      f'<p class="qa {"qa-fail" if not row["QA_passed"] else ""}">{qa_label(row)}</p>')
            action = 'Comparer original / avant / après'
        else:
            detail = '<p>Reconstruction absente de cette livraison — conservée dans le bilan.</p>'
            action = 'Voir le blocage enregistré'
        summaries.append(f'<section class="clip-summary"><h2>Épisode {ep:02d}</h2>{detail}'
                         f'<a href="{view_path(ep)}">{action} →</a></section>')
    return document('Comparaison 4D',
        '<header><p class="eyebrow">World Reward · diagnostic end2end</p>'
        '<h1>Le mouvement réel.<br>La reconstruction avant et après.</h1>'
        '<p class="subtitle">Un clip par page. Trois vues synchronisées, avec agrandissement individuel.</p></header>'
        f'<p class="warning"><strong>{complete}/{len(states)} reconstructions disponibles dans cette livraison figée.</strong> '
        'Qualité 4D et gain face à CARI4D non vérifiés. Les régressions restent visibles.</p>'
        '<main class="clip-list">' + ''.join(summaries) + '</main>'
        '<p class="fineprint">Bleu : ORIGINAL · ambre : BASELINE · vert : CORRECTIF. '
        'La couleur identifie la méthode, pas sa qualité.</p>'
        '<p class="fineprint">Aucune vidéo chargée ici. Lecture et planche uniquement à votre demande, '
        'sans vidéo enregistrée localement. La lecture consomme les petits aperçus sur votre connexion.</p>')


def episode_html(states, episode, *, focus=None):
    """One lazy, unchanged synchronized MP4 per page, optional CSS-only crop."""
    require(focus in (None, 'original', 'baseline', 'candidate'), 'Only exact display crops allowed')
    row = next((row for row in states if row['episode'] == episode), None)
    require(row is not None, 'Episode must belong to the frozen cohort')
    title = f'Épisode {episode:02d}'
    header = ('<header><p class="eyebrow"><a href="/">← Tous les clips</a> · World Reward</p>'
              f'<h1>{title} — regarder le mouvement, pas seulement le score</h1></header>')
    if row['status'] != 'complete':
        return document(title, header +
            '<main><h2>Reconstruction absente de cette livraison</h2>'
            '<p>Échec / non reconstruit dans cette ablation — conservé dans le bilan, sans clip remplacé.</p>'
            f'<p><strong>Cause enregistrée :</strong> {html.escape(row["reason"])}</p>'
            '<p class="fineprint">Cet état est figé au moment de la publication ; il ne décrit pas '
            'nécessairement les travaux en cours.</p></main>')
    labels = {'original': ('ORIGINAL', 'Ce qui se passe réellement dans la vidéo'),
              'baseline': ('BASELINE', 'Avant · référence 052 intacte'),
              'candidate': ('CORRECTIF', 'Après · candidat, pas encore adopté')}
    nav = []
    for key, label in [(None, 'Comparer les 3 vues'), ('original', 'Original agrandi'),
                       ('baseline', 'Baseline agrandie'), ('candidate', 'Correctif agrandi')]:
        current = ' aria-current="page"' if key == focus else ''
        nav.append(f'<a href="{view_path(episode, key)}"{current}>{label}</a>')
    context = (f'<p>{html.escape(candidate_label(row))}</p>'
               '<p class="fineprint">Le correctif n’est pas encore adopté : vérification physique et GT restantes.</p>'
               f'<p class="qa {"qa-fail" if not row["QA_passed"] else ""}">{qa_label(row)}</p>'
               '<nav class="view-nav" aria-label="Agrandir une vue">' + ''.join(nav) + '</nav>')
    video = (f'<video controls playsinline preload="none" aria-label="Épisode {episode}: '
             + (f'{labels[focus][0]}, vue agrandie du même aperçu synchronisé' if focus else
                'original RGB, baseline 052, correctif diagnostic synchronisés')
             + f'"><source src="/episode-{episode}.mp4" type="video/mp4"></video>')
    if focus is None:
        positions = dict(original='gauche', baseline='centre', candidate='droite')
        headings = ''.join(f'<div class="column-label {key}"><h2>{name} '
                           f'<span class="position">· {positions[key]}</span></h2><p>{description}</p></div>'
                           for key, (name, description) in labels.items())
        media = '<div class="columns">' + headings + '</div>' + video
    else:
        name, description = labels[focus]
        media = (f'<div class="focus-view focus-{focus}"><div class="focus-heading {focus}">'
                 f'<h2>{name} — vue agrandie</h2><p>{description}</p></div>' + video + '</div>'
                 '<p class="fineprint">Agrandissement d’une colonne du même fichier vidéo, sans nouvelle '
                 'reconstruction ni retiming. Le plein écran natif peut montrer les trois colonnes.</p>')
    position = next(i for i, item in enumerate(states) if item['episode'] == episode)
    previous = states[(position - 1) % len(states)]['episode']
    following = states[(position + 1) % len(states)]['episode']
    footer = ('<div class="inspection"><strong>À examiner : tremblements · contact main–objet · '
              'glissement au sol · mouvement conservé.</strong>'
              'Même timeline complète, échelle inférée, caméra et sol virtuel fixes. '
              'Ni vérité terrain ni qualité 4D vérifiées.</div>'
              f'<p><a href="/episode-{episode}.jpg" target="_blank" rel="noopener noreferrer">'
              'Ouvrir la planche fixe, uniquement à la demande</a></p>'
              '<p class="fineprint">Bleu / ambre / vert identifient les vues, pas leur réussite. '
              'Une seule vidéo à la demande, sans enregistrement local ni chargement anticipé.</p>'
              '<nav class="clip-nav" aria-label="Changer de clip">'
              f'<a href="{view_path(previous)}">← Épisode {previous:02d}</a>'
              '<a href="/">Tous les clips</a>'
              f'<a href="{view_path(following)}">Épisode {following:02d} →</a></nav>')
    return document(title, header + '<main>' + context + media + footer + '</main>')


def make_server(states, allowed, *, reader=None):
    """Reuse the existing same-origin range implementation without changing it."""
    pages = {'/': index_html(states)}
    for row in states:
        ep = row['episode']
        pages[view_path(ep)] = episode_html(states, ep)
        if row['status'] == 'complete':
            for focus in ('original', 'baseline', 'candidate'):
                pages[view_path(ep, focus)] = episode_html(states, ep, focus=focus)
    parent = base.handler_factory([], allowed, reader or base.AzureReader())
    class Handler(parent):
        def serve(self, head=False):
            if self.path not in pages:
                return super().serve(head=head)
            try:
                self.valid_local_request()
                page = pages[self.path]
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
    parser.add_argument('--candidate-kind', choices=('joint', 'continuation', 'sqp', 'smoothing'), default='joint')
    parser.add_argument('--ttl', type=int, default=base.MAX_TTL)
    args = parser.parse_args()
    require(60 <= args.ttl <= base.MAX_TTL, 'Viewer TTL must be between60and3600seconds')
    states, allowed, _ = load_receipt(args.receipt, args.revision,
        render_revision=args.render_revision, candidate_revision=args.candidate_revision, candidate_kind=args.candidate_kind)
    server = make_server(states, allowed); server.deadline = time.monotonic() + args.ttl
    timer = threading.Timer(args.ttl, server.shutdown); timer.daemon = True; timer.start()
    print(f'END2END_STREAM_URL http://127.0.0.1:{server.server_port}/', flush=True)
    try: server.serve_forever(poll_interval=.5)
    except KeyboardInterrupt: pass
    finally: timer.cancel(); server.server_close()


if __name__ == '__main__':
    main()
