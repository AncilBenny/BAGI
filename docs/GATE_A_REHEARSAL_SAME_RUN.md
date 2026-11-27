# Gate A — same-run fork repair for live rehearsal

Prospective amendment to docs/GATE_A_REHEARSAL.md. The original
e01a2c8 run 45991409620 passed Ruff and 954 tests but failed at
the exact cached first-B trace comparison (after first seed's
8192 B actions, before A rehearsal). No output artifact was
retained; the exact divergence and root cause are not established.
Initial frozen traces matched. Pinned dependencies and historical
archive/summary digests match. Cross-run training reproducibility
must not be assumed; possible runner floating-point effects or
other training-state differences remain unproven.

Replace the cached trained comparator with a prospective same-run
paired fork. Keep the initial exact cached frozen trace check,
record exact mismatch counts, first mismatched monitor seeds and
success deltas against the historical first-B evaluation, then
use the actual common B=8192 checkpoint as the comparator origin.
Both branches independently load the exact same source-bound
checkpoint and must reproduce all four common monitor traces
exactly. Native smoke verifies two loads produce exactly the
same subsequent executed rollout, policy/Adam and counters.
Do not weaken within-run restore checks.

Per seed: common B8192 once; branch rehearsal A2048 then three
B8192+A2048 cycles; branch B-only four B8192 blocks. Both branches
have 40960 total phase actions, rehearsal B32768/A8192 and control
B40960. Preserve original reset streams, Adam-only initial reset,
persistent policy/Adam and original frozen evaluation criteria.
Record actual branch counters separately from common prefix; shared
actions are not counted twice. New unique actions per seed 73728,
across 3 seeds 221184 (A24576/B196608), optimizer steps 13824,
on-policy presentations 884736. 7200 actual evaluation episodes:
per seed 2 common points, 2 restore points, 7 further rehearsal
points, 4 further B-only points, 160 episodes per point.
Cached carry/fresh curves descriptive; same-run B-only primary
control at matched B exposures (through 32768) and matched total
phase actions (40960). Original registered reliability criteria
unchanged. Full training and evaluation traces flushed incrementally;
failed step uploads partial evidence with separate name, never
a completed study artifact. Full CI before execution.

No historical replay, no maintained continual-learning baseline,
no independent physical groups. No positive transfer claim from
extra live A practice. Full alternating-schedule resume remains
unimplemented (both cursors recorded, native boundary saved).
No change to main, PR #1 or scheduling.
