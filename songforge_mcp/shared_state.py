"""Single shared instances used by more than one tool module.

Exists to avoid two problems that come from each tool module creating its
own instance instead: (1) SeparatorClient's internal lock only serializes
GPU-bound separation calls made through the same instance — a second,
independently-created instance would let two separations race for GPU
memory at once, the same bug class fixed for ACEStepClient's cross-process
generate lock. (2) JobRegistry needs to be the same registry regardless of
which tool created a job, so check_vocal_track_status can poll a job
created by split_vocal_stems (or any future tool) and not just ones
created by generate_vocal_track.

acestep_client specifically: extract_track's automatic checkpoint swap
(see acestep_client.py) only runs inside ACEStepClient's own generate lock
- a second instance would let a generate_vocal_track call and an
extract_track call race the exact same GPU/server at once, exactly the
problem this module exists to prevent.
"""
from songforge_mcp.acestep_client import ACEStepClient
from songforge_mcp.job_registry import JobRegistry
from songforge_mcp.separator_client import SeparatorClient

jobs = JobRegistry()
separator_client = SeparatorClient()
acestep_client = ACEStepClient()
