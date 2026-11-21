# BAGI M14 — independently verified native open-door traversal and physical goals

**2026-11-12 UTC. Preregistered M14 pilot PASS.** A PUBLIC-observation-only, hand-specified finite-memory reference controller completed **59/72 independent DoorKey physical goals** with a single *source-visible* open-door traversability correction; the unchanged M13 controller completed **0/72** on the same new layouts. These are real native simulated physical goal events, but NO learned-model training, policy improvement or task-independent transfer was performed.

## Provenance and reuse

- Frozen registration: `docs/GENERAL_BRAIN_M14_OPEN_DOOR_PREREG.md`, `configs/m14_public_open_door.yaml`, implementation `bagi/experiments/m14_public_open_door.py` at source commit `634d7b02b9480e3c4b62c5ce8d54e37165523e22`; serialized workflow/prereg log at `e31ca4baf9a6553339a58900a515c614bdbe53dd`. Exact one-file ADD-only request/source commit **`945346874b7713cc2c29886ecacc0f0b7d1f2f8c`**.
- Existing gated [BAGI Research Loop run 43240275682](https://github.com/AncilBenny/BAGI/actions/runs/43240275682) completed SUCCESS, both source-admission and compute jobs passed. Source-parent push CI 43239743358 / PR CI 43239756310 PASS Ruff, pytest and PR merge provenance; request push CI 43240275862 / PR CI 43240289323 PASS. No new workflow or schedule change.
- Artifact `bagi-research-652` ID **13955750128**, ZIP SHA256 **`ac0138af95a9ccbfd9cdeac71d45fc59b1bdc2244c69660fd7b9490551d3e6d5`**, independently compared to exact GitHub Actions digest, ZIP CRC PASS, internal study `m14` and source commit match. Config SHA256 **`6a124ec84cfece3c2a9481add0660b50c3f17c9e0ab28f05d841c6f6eab56ce4`**. Measured dependencies Gymnasium1.4.0, Farama MiniGrid3.2.0, NumPy2.4.6. Reuse official native DoorKey and BAGI's independent evaluator-only NativeDoorKeyAudit; no unsafe model checkpoint loading, secret, oracle or private input to acting controller.

## Actual simulator and resource accounting

- **TRAIN-MONITOR:** DoorKey6x6 fresh layouts191001–191012 and DoorKey8x8 201001–201012, TWO controllers per layout: **48 episodes, 9,554 native actions**. Blocked baseline 6x6 3,072 actions, 8x8 3,072; corrected controller6x6 1,355, 8x8 2,055.
- **INDEPENDENT FINAL:** wholly unused layout seeds 6x6 **192001–192036**, 8x8 **202001–202036**, both controllers per layout: **144 episodes, 28,129 native actions**. Blocked6x6 9,216; corrected6x6 3,871; blocked8x8 9,216; corrected8x8 5,826. Prior M9–M13 final layouts remain untouched/closed.
- **Combined new M14:** **192 native episodes, 37,683 real native actions**, 0 optimizer-step calls, 0 model fits, 0 teacher queries; measured in-run native execution **15.832 s**, excluding GitHub setup/install. Every per-episode seven-action histogram reconciles to actual executed steps; physical goal claims have a genuine pickup preceding effective opening preceding rewarded terminal, with actual door-cell traversal.
- Both controllers use **the exact same per-layout PRNG initialization** and the same M13 public-memory logic, except the corrected subclass selects native forward2 when its current PUBLIC 7×7×3 front tile has object `door` and state code0 **open**. Native private coordinates/true key possession/door state never enter actions or memory updates. An independent auditor checks the actual open-door tile entry and native key→door→goal events after actions, separate from reward. **All 72 paired independent seed layouts had identical pre-first-door-opening executed-action SHA256 digests**; this narrows the intervention to post-opening behavior. The earlier M13 independent data is NOT re-run or relabelled.

## Independent final physical outcomes

| Task and public controller | Actual key pickup episodes | Effective door opening episodes | Entered physically open door cell | Native full physical key→door→rewarded goal |
|---|---:|---:|---:|---:|
| DoorKey6x6 M13 original public-memory | 36/36 | 35/36 | **0/36** | **0/36** |
| DoorKey6x6 open-door traversable | 36/36 | 35/36 | **35/36** | **33/36** |
| DoorKey8x8 M13 original public-memory | 36/36 | 36/36 | **0/36** | **0/36** |
| DoorKey8x8 open-door traversable | 36/36 | 36/36 | **36/36** | **26/36** |

- **Corrected pooled independent 59/72 physical goals versus blocked 0/72**; per-task gains **+33/36** (6x6) and **+26/36** (8x8). All 59 corrected rewards correspond to the independently audited true physical pickup→effective opening→native goal chain; NONE is a vacuous or reward-only count. Native rewarded terminals equal physically verified goal episodes. Door-cell entry corrected35/36 6x6,36/36 8x8, blocked0/36 in each. Pre-opening action hashes equal for all pairs.
- Preregistered primary gate required matched preopening sequences, >=12 actual door-cell entries per size, at least one additional physical full goal per size and +2 pooled. **PASS all criteria by a wide margin**. This strongly supports the specific mechanism that the old public controller's open-door-as-blocked navigation choice was a major task bottleneck. It is not evidence that memory alone caused success: memory controller and traversal correction form a TASK-SPECIFIC scripted control policy.
- TRAIN-MONITOR outcomes (NOT independent): baseline0/12 physical goals per size; corrected10/12 6x6 and8/12 8x8. These do not enlarge final denominators or justify post-hoc tuning.

## Scientific interpretation and next open question

The original M13 result remains frozen: **0/216** independent physical task goals across three controls; do not relabel it as positive because M14 fixes a code issue on NEW layouts. M14 demonstrates that the M13 public-only controller already obtained keys and opened doors, but a single incorrect *post-door action eligibility* rule blocked end-to-end completion. The comparison is much better controlled than post-hoc reward correlation because the whole prefix through physical door opening is hash-matched for each paired seed; nevertheless different post-opening trajectories could have additional causal influences and the correction is hand-specified by a researcher.

**Next necessary learned-agent study (not yet executed here):** train an existing, maintained supervised or RL policy from genuinely public observation/action demonstrations collected by this physically competent PUBLIC reference controller. The fitting policy must see only public mission, partial image and its own **actual executed action history** (not privileged key carrying or teacher labels at evaluation). Compare against equal-interaction untrained/random and frozen reference on entirely new layout seeds, and physically verify pickup→door→goal rates and retention across task sizes/goal families. Use episode-disjoint train/final layouts, safe numeric-only checkpoints and measured gradient optimizer updates; do not re-use M14 FINAL 192001–192036 / 202001–202036, nor M9–M13 final. A cloned scripted controller is integration/competence replication, **not** novel AGI or task-independent continual learning; Workshop and XLand remain diagnostics until genuine independent transfer is tested.

**Governance:** existing serialized workflow and schedule unchanged, no duplicate study job, generated outputs ignored, append-only research log and past M1–M13 artifacts preserved. PR #1 remains without submitted reviews, so main merge HOLD pending reviewed path and postmerge main CI.
