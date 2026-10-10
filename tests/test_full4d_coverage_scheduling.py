"""Wrapper-only scheduling tests; no datasets, service control, models or GPU."""
from pathlib import Path
import subprocess

import pytest

ROOT=Path(__file__).parents[1]


def test_coverage_waits_entire_successful_preview_before_unchanged_compute_budget():
    wrapper=(ROOT/'infra/run_full4d_coverage.sh').read_text()
    assert '"$5" == --after-terminal' in wrapper
    assert '"$CODE/infra/terminal_success.py" "$WAIT_FOR"' in wrapper
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
