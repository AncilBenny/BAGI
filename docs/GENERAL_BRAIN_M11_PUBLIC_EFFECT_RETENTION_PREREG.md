# BAGI M11 — preregistered public-action effect learning and retention

**Stage:** source preregistration *before* any M11 native collection or fitting. This is a minimal G2-like supervised learning component, not an agent controller or proof of physical task competence.

## Gap and established research

BAGI M10-P1 independently verified only 2/72 full simulated DoorKey goals for the upstream one-hot PPO arm, both from the same initialization; its preregistered robustness gate failed. It did not establish a reusable causal state representation from actual executed actions. To pursue task-independent continual learning, first ask whether a persistent public-only observation-action effect predictor can adapt to a second goal/task while retaining old prediction quality. **Do not repurpose M10-P1 held-out layouts for training or tuning.**

- Reuse Farama MiniGrid `MemoryS7` and `DoorKey8x8` (same public image/mission/native action API): https://minigrid.farama.org/main/environments/minigrid/MemoryEnv/ and https://minigrid.farama.org/main/environments/minigrid/DoorKeyEnv/ .
- Reuse PyTorch maintained `nn.GRU` and `Adam`: https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html . No new optimizer, world model or reward shaping. Input is the **current publicly visible** 243-vector from BAGI's established encoder plus the **actually executed** native action one-hot; hidden state resets at real episode boundaries. Target: whether the next **PUBLIC 7x7x3 image** changes, observed after the action; private physical key/door state and simulator `info` are not predictors or training labels.
- Replay principle: Rolnick et al. NeurIPS2019 https://proceedings.neurips.cc/paper/2019/hash/fa7cdfad1a5aaf8370ebeda47a1ff1c3-Abstract.html . This study uses elementary cached-episode *supervised* rehearsal. It is NOT the CLEAR off-policy RL algorithm.

## Frozen hypothesis, fair controls and evaluation gate

Two independent model initialization seeds **317,331**. For each task collect genuine random executed native transitions on 24 TRAIN episodes: Memory layout seeds 131101–131124, DoorKey 141101–141124, native episode horizon 256. Separate 12 FINAL episode seeds/task: Memory 132101–132112, DoorKey 142101–142112; **acquire FINAL only after all fresh model fits and numeric integrity finish**. Native interactions are performed once and reused by all learner arms; preserve exact episode boundaries and actual step/action counts. No teacher, oracle or private label. All old M1–M10 held-out layouts remain untouched.

PyTorch GRU hidden32 + sigmoid-logit binary head, Adam lr0.001, four source epochs and four target epochs. Source learning on Memory then copy source weights for **naive target fine-tune** and **cached Memory rehearsal**; independently initialized same-seed target-only scratch GRU is an additional reference. Both target continuations see identical native new-task examples and take identical optimizer calls; rehearsal additionally computes equal-weight loss on one cached old-task TRAIN episode per target optimizer step. This has **more compute and old-task presentations per step**, reported as a confound; no additional real actions. All phase fits preserve a source-only frozen reference.

After all fits, read independent final episodes ONCE and evaluate each model on both tasks with **transition-weighted Brier score**, accuracy, positive image-change prevalence, zero-change constant baseline and TRAIN-fit empirical **action-conditioned** baseline. Action-driven camera changes may dominate; the supervised target is only a PUBLIC perceptual event, not a physical pickup, opened door, goal achievement or proven causal understanding.

**Primary pilot gate:** for BOTH model seeds, rehearsal reduces old-task final Brier score by at least **0.015** relative to naive fine-tune, with at most **+0.020** additional target-task Brier, and both tasks exhibit real positive public image-change events. Secondary task-only scratch Brier and action-only rate baseline. A failure is evidence of limited retention, task aliasing, poor label information or insufficient training; no post-hoc retuning on final episode seeds. A pass supports only a small same-benchmark continual predictive-learning result, not policy success, cross-environment generality or AGI.

## Governance

Register `m11` inside the EXISTING serialized `bagi-research-loop.yml` (unchanged schedule), separate ADD-only protected one-shot request after new push/PR Ruff+pytest and exact merge-provenance. Outputs `runs/m11_public_effect_retention/` and numeric-only NPZ artifact with SHA256; no generated files or credentials committed. Actual native steps, optimizer calls, seed-stream digests, measured runtime and negative results must be recorded.

**At preregistration:** 0 newly executed M11 simulator actions, 0 optimizer calls, 0 fresh fits, 0 independent effect tests, 0 teacher queries. Main integration remains HOLD pending reviewed PR.
