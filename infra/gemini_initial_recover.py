"""Recover only immutable initial-call transport failures; no semantic rerolls."""
import os
from pathlib import Path
import sys
from mediapipe_cpu_runtime_verify import strict,require
from gemini_initial import native,run,HELPERS as ORIGINAL_HELPERS

ENTRY='run_gemini_initial_recover'
CONFIG='configs/gemini_retry_v1.json'
HELPERS=(*ORIGINAL_HELPERS,'infra/gemini_initial_recover.py','infra/run_gemini_initial_recover.sh',CONFIG)

def settings(code):
    c=strict((code/CONFIG).read_bytes())
    require(c['schema']=='world_reward.gemini_retry.v1' and c['max_total_attempts']==3
        and c['request_changes_allowed'] is False and c['retry_successful_responses'] is False
        and c['ground_truth_used'] is False and c['manual_labels'] is False
        and c['quality_verified'] is False,'Same-request bounded recovery only')
    return c

if __name__=='__main__':
    if len(sys.argv)==4 and sys.argv[1]=='--native':
        code=Path(sys.argv[2]);native(code,Path(sys.argv[3]),settings(code))
    else:run(settings(Path(os.environ['WR_CODE'])),ENTRY,HELPERS)
