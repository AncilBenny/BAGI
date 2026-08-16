# BAGI Autonomous Research Log

## 2026-08-16 — Oracle search ceiling diagnostic

- Hypothesis: the Workshop v0.2 planner failure may come from random-shooting search rather than from learned-model rollout error.
- Evidence entering this cycle: the 50-guided learned planner reaches the switch substantially more often than it crosses the door, and increasing horizon from 5 to 8 did not restore success. The existing hourly research workflow was still rerunning the older sample-efficiency study rather than the current planner diagnostic.
- Change: added `bagi/experiments/oracle_search_ceiling.py`, `configs/oracle_search_ceiling.yaml`, and the `make oracle-search-ceiling` target. The diagnostic replaces the learned world model with exact Workshop v0.2 transition dynamics while preserving the same random-shooting MPC structure, horizon, discount, event weights, candidate counts, and planner seeds.
- Result: implementation is committed; a repository-run result has not yet been observed in this cycle, so no oracle-success claim is recorded yet.
- Next measurement: run `make check && make oracle-search-ceiling`. If oracle planning succeeds while learned planning fails under the same search budget, compare learned-vs-oracle candidate sequence scores/event predictions to localize rollout calibration error. If oracle planning also fails, focus on search/scoring itself.
