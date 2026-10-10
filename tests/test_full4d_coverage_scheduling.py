"""Wrapper-only scheduling tests; no datasets, service control, models or GPU."""
from pathlib import Path
import subprocess

import pytest

ROOT=Path(__file__).parents[1]


def test_independent_coverage_waits_entire_terminal_preview_before_unchanged_compute_budget():
    wrapper=(ROOT/'infra/run_full4d_coverage.sh').read_text()
    assert '"$5" == --after-terminal' in wrapper
    assert '"$CODE/infra/terminal_success.py" "$WAIT_FOR" --allow-failed-terminal' in wrapper
    assert wrapper.index('"$CODE/infra/terminal_success.py" "$WAIT_FOR"')<wrapper.index('28920s')
    assert '"$(source_identity)" == "$BEFORE"' in wrapper
    assert '! -e "$ROOT/experiments/full4d-v1-$REV"' in wrapper
    assert 'set -- "$1" "$2" "$3" "$4"' in wrapper # strip queue args from unchanged producer ABI
    assert '--after-gpu-lock' not in wrapper and 'systemctl restart' not in wrapper


@pytest.mark.parametrize('extra,ok',[([],True),(['--after-terminal','world-reward-preview.service'],True),
    (['--after-terminal','unknown.service'],False),(['--after-gpu-lock'],False),
    (['--after-terminal','world-reward-x','--after-terminal','world-reward-y'],False)])
def test_coverage_parser_retains_original_exact_baseline_pins(extra,ok):
    wrapper=(ROOT/'infra/run_full4d_coverage.sh').read_text().split('ROOT="${WR_ROOT',1)[0]
    args=['--baseline-report-bytes','49601','--baseline-report-sha256','a'*64,*extra]
    r=subprocess.run(['bash','-c',wrapper+'\nprintf "%s" "$#"','wrapper',*args],capture_output=True,text=True)
    assert (r.returncode==0)==ok
    if ok:assert r.stdout=='4'


def test_coverage_failure_tolerant_wait_does_not_weaken_render_or_reference_success_gates():
    coverage=(ROOT/'infra/run_full4d_coverage.sh').read_text()
    preview=(ROOT/'infra/run_end2end_preview.sh').read_text()
    assert coverage.count('--allow-failed-terminal')==1
    assert '--allow-failed-terminal' not in preview
    assert 'Coverage is an independent experiment' in coverage
    assert 'prior preview technical failure' in coverage
    assert '--baseline-report-bytes' in coverage and '--baseline-report-sha256' in coverage
