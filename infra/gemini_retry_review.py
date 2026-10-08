"""Render recovered saved predictions, no API call/GT/GPU access."""
from pathlib import Path
import sys
from gemini_initial_zoom import native,run,HELPERS as ORIGINAL_HELPERS
ENTRY='run_gemini_retry_review'
CONFIG='configs/gemini_retry_review_v1.json'
HELPERS=(*ORIGINAL_HELPERS,'infra/gemini_retry_review.py','infra/run_gemini_retry_review.sh',CONFIG)
if __name__=='__main__':
    if len(sys.argv)==4 and sys.argv[1]=='--native':native(Path(sys.argv[2]),Path(sys.argv[3]),CONFIG)
    else:run(ENTRY,HELPERS,CONFIG)
