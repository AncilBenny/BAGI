# M20 preregistration: physical Fetch transfer after DoorKey

## Decision and established reuse

M18 (DoorKey→GoToDoor) and M19 (GoToDoor→DoorKey) failed their
preregistered primary gates; their final layouts remain closed. M19's
pooled retention signal was concentrated in two of four model seeds.
A different physical objective is required before claiming cross-mechanics
transfer. This is an **evaluation/adaptation**, not a novel algorithm.

Reuse the maintained Farama MiniGrid `MiniGrid-DoorKey-6x6-v0` and
`MiniGrid-Fetch-5x5-N2-v0`, Gymnasium time limits, PyTorch M15 public
behavior cloning, M19 exposure-matched replay, public finite-memory
DoorKey demonstrator, and native physical evaluator. References:
- https://minigrid.farama.org/environments/minigrid/FetchEnv/
- https://github.com/Farama-Foundation/Minigrid
- Rolnick et al., *Experience Replay for Continual Learning*, NeurIPS 2019,
  https://proceedings.neurips.cc/paper/2019/hash/fa7cdfad1a5aaf8370ebeda47a1ff1c3-Abstract.html
- Ross et al., *A Reduction of Imitation Learning and Structured Prediction to
  No-Regret Online Learning*, AISTATS 2011, https://proceedings.mlr.press/v15/ross11a.html

## Falsifiable hypothesis and controls

Does sequentially training on **public-action DoorKey experience** increase
genuinely nonvacuous native Fetch target pickup compared with fresh
Fetch-only scratch, without losing source DoorKey competence? Are replay's
source-retention effects present across at least three independent model seeds
when matched against equal additional *target* cached-label presentations?

Frozen tasks: DoorKey-6x6 source, Fetch-5x5-N2 target. Five conditions:
source frozen, target scratch, naive source→target, source→target with equal
extra target labels, source→target with same extra source labels.
Same M15 PyTorch model and feature contract, native seven actions, same
initialized seed within a contrast, 10 training epochs per fit, Adam 0.003,
batch 128, and matched target exposures/update calls across contrasts.
20 fresh model fits if feasibility succeeds. This fixed preregistration
is study-specific comparability, **not** an administrative compute ceiling.

Model seeds: 751, 757, 761, 769. Source TRAIN layouts 311001–311048;
target TRAIN 321001–321048. Train-only MONITOR layouts
312001–312012, 322001–322012. Independently frozen FINAL source
313001–313032, target 323001–323032. Each range is disjoint from the
earlier M15–M19 sealed finals. The final split is never used for
selection, tuning, demonstrations, or teacher feasibility adjustments.

**Train-only feasibility gate before fitting:** across 48 actual Fetch
public-teacher episodes require at least 8 independently witnessed
nonvacuous correct-object pickups (not merely game reward). If absent,
publish teacher coverage/actions and stop *this* study before model fits and
before opening final layouts. This is a measurement-validity condition,
not a ban on new research. Never substitute a privileged teacher to force pass.

## External physical success witness and registered gate

Fetch reward can arise after a correct pickup, but reward alone is not this
study's success label. `NativeFetchAudit` (EVALUATOR ONLY) checks:
requested mission type/color, actual identity change from grid/front object
into `carrying` after native executed pickup action 3, positive native
terminal reward, at least one effective forward movement, and target
not initially adjacent to agent. Report raw reward, correct/wrong pickups,
initial adjacency, nonvacuous physical completion, and real action histograms
separately. No unwrapped state is passed to teacher, model, or features.

Registered primary **conjunctive** gate: pooled Fetch target naive−scratch
≥8/128 physical goals and >0 on ≥3/4 model seeds; pooled DoorKey old-source
replay−equal-extra-target ≥8/128 and >0 on ≥3/4 seeds; source-replay
Fetch target deficit versus naive ≤8/128. Compare native teacher, random
reference, source, scratch, naive, extra-target and source-replay on all final
layouts. Failure is recorded as failure, not posthoc threshold relaxation.
No claim of task-independent transfer even on a pass: both are MiniGrid,
the source teacher is scripted, and model features are engineered.

## Execution, provenance, interpretation

Source code/config and these gates must pass Ruff/pytest and independent
current push/PR merge CI. Execute through the existing serialized BAGI
Research Loop with `m20` and `force_retrain`/exact one-shot request; no
new competing workflow. Store only ignored `runs/m20_fetch_rehearsal/`
outputs and immutable artifact, source commit, config hash, versions,
actual native actions, unique actual-action traces, fit/update counts,
numeric checkpoint hashes, independent outcomes, and limitations.
Append results to `docs/autonomous-research-log.md` without overwriting
M1–M19. Main integration requires an independent reviewed PR. Workshop
and XLand remain diagnostics. No physical-world or general intelligence
claim is implied by simulator reward.
