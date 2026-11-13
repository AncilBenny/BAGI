# BAGI-M5 — completed public-goal continual-retention pilot

**Independent artifact audit: 2026-11-11 UTC. Retrospective result, not a changed preregistration.**

## Provenance and exact executed work

- Registered protocol: `docs/GENERAL_BRAIN_M5_CONTINUAL_TASK_RETENTION_PREREG.md`; admitted GitHub Actions [42972628418](https://github.com/AncilBenny/BAGI/actions/runs/42972628418) SUCCESS. Source/request commit `0a4b1e49148d2bdee78da70b1343786a960c88e0`; preregistered configuration `configs/m5_continual_retention.yaml` SHA256 `918c3025701fefa0d607f88f3c933eb394f8a0be0359304068ad5e8dadcfefc6`.
- Downloaded artifact `bagi-research-636`, ID **13834627167**, ZIP SHA256 **370ee27c3dfb0f485073d84f1382d3b52aedb56b9741883d652cf1a3057d0f90**, matching GitHub-provided digest. Original source/study markers exactly match `0a4b1e4...` / `m5`. Artifact timestamp `2026-11-11T06:41:53Z`. Eight pure numeric NPZ checkpoint SHA256s and all array member names, shapes, dtypes, and allowed non-object types were independently verified with `allow_pickle=False`; no model weights were committed.
- Fresh actual execution: **786,432 native training actions**, **24,576 intercepted optimizer.step calls**, **98,574 actual evaluation actions**, **480 independent monitor+heldout evaluation resets**, **8 finished PPO fits**, **0 teacher queries**, and **572.241 seconds** measured study wall-time. Training breakdown: 4 joint blocked/interleaved fits × 131,072 native actions and 4 task-only fits × 65,536; every joint model consumed 65,536 native actions PER task. PPO rollout128, batch64, epochs2, CPU, 15,880 policy parameters for every model. Versions: Gymnasium 1.4.0, MiniGrid 3.2.0, stable-baselines3 2.9.0.
- Independent heldout split: MemoryS7 seeds 102001–102012, DoorKey-8x8 seeds 103001–103012, each 12 episodes/model, 256-native-action cap. All 480 episode records and executed-action histogram sums, native step/reset totals, and goal predicates were independently recomputed from the JSON: rewarded goal requires actual terminal success AND positive public reward. DoorKey *physical* success requires separate evaluator-only observed key pickup + locked door opening + rewarded goal, not reward or attempted commands alone.
- The archived last `runs/m5_continual_retention/progress.json` intentionally retains its stage name `partial_not_a_completed_study` even after summary finalization. The completed study is instead established by the separate complete summary file, matching 8-model roster, counters, workflow SUCCESS and all archived NPZ hashes. Do not rewrite that historical progress marker.

## Final independent goal completions

| Policy schedule | MemoryS7 (two seeds ×12) | DoorKey-8x8 rewarded goals | Physically verified full DoorKey goals |
|---|---:|---:|---:|
| blocked Memory→DoorKey | 0/24 | 0/24 | 0/24 |
| interleaved Memory/DoorKey | 4/24 | 0/24 | 0/24 |
| Memory-only | 12/24 | 0/24 | 0/24 |
| DoorKey-only | 0/24 | 0/24 | 0/24 |

Per model seed (MemoryS7, DoorKey rewarded/physical): blocked71 (0/12,0/12,0/12), blocked83 (0/12,0/12,0/12); interleaved71 (4/12,0/12,0/12), interleaved83 (0/12,0/12,0/12); Memory-only71 (4/12,0/12,0/12), Memory-only83 (8/12,0/12,0/12); DoorKey-only71/83 (0/12,0/12,0/12). Memory-only is the competence ceiling here, not a retention claim. The interleaved71 and Memory-only71 successes occur on the SAME four heldout Memory layouts (102003, 102005, 102006, 102008).

Across **all 96 final DoorKey native episodes**, evaluator-only physical state reads recorded **one effective key pickup total** (DoorKey-only seed83), **zero observed locked→opened door transitions**, and **zero rewarded DoorKey terminal goals**. There was no physically verified key+door+goal episode. Training public rewarded DoorKey terminations totaled only **6** over the four joint models and 0 over the DoorKey-only models; raw training rewards are NOT independent success.

**Predeclared H-M5 gate FAILS.** Interleaved Memory minus blocked Memory was +4/24, meeting only that arithmetic subcriterion; DoorKey difference was 0/24, but **neither interleaved policy seed had any physically verified DoorKey goal**. The conjunctive success condition was not met. There is no credible task-independent continual learning or even reliable DoorKey competence claim.

## Diagnosed methodological confound — hypothesis for next controlled study

Source `bagi/experiments/m5_continual_retention.py` calls upstream `agent.set_env(training[task], force_reset=True)` before **every 128-native-action PPO learn block**, including when the task is unchanged. Under Stable-Baselines3, `force_reset=True` drops `_last_obs` and forces a fresh environment reset at the next train call; confirmed from upstream source https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/base_class.py and the maintained API docs https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/common/base_class.html . Thus the training stream was often cut at the update boundary, distinct from the independent evaluator's 256-action episode horizon. DoorKey's official sparse key-pickup→door-unlock→goal mechanism is documented at https://minigrid.farama.org/main/environments/minigrid/DoorKeyEnv/ .

Measured native DoorKey reset counts were approximately one per 128-action block: blocked seed71/83 (515/512), interleaved seed71/83 (514/513), DoorKey-only seed71/83 (512/512). These data support repeated training reset exposure but do NOT prove uninterrupted episodes alone would solve DoorKey. It is also important that the `set_env(force_reset=True)` approach matches M5's registered *control schedules*; this new finding qualifies interpretation of its broader continual-learning hypothesis and motivates a small controlled fix, not rewriting M5 post hoc.

**Open scientific question:** with identical public inputs, native training actions, PPO update settings, RNG seeds and 128-action learn calls, does preserving native episode continuity between PPO blocks improve actual DoorKey key/door/goal outcomes compared with the exact M5-style reset mechanism? A dedicated independent M6 control will test that before any new multitask "general intelligence" claim. The M5 heldout seeds must not be reused in training or model selection.

## Safeguards and limitations

This study used only two independent model seeds and one simulator family; mission text was hashed public words, not learned language understanding; physical state was external evaluator-only. This result does NOT establish effective multi-task retention, physical transfer to Workshop/XLand, or open-ended task-independent learning. Preserve all historical M1–M4/M5 failures and gates. Generated checkpoints and run outputs remain outside Git. Main integration still requires reviewed PR, current exact-head Ruff/pytest, independent push+PR merge-provenance CI and final main verification. No M6 fitting is recorded by this report.
