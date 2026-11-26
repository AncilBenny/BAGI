# M23 stage 7 preregistration — frozen effect models versus public action-history rule under policy shift

**Status: prospective 2026-11-15, before any new stage-7 source MONITOR actions.**
Stage 6's registered source native-effect gate PASSED, but its TRAIN and
MONITOR used the same 0.35 uniform-probe action policy on the same three
source task families. The ~0.999 macro-F1 does NOT establish transferable
causal knowledge or action selection. A policy change and a stronger
non-learning public-memory rule are the most useful next controls.

## Reuse-first and exact frozen source

Use maintained Farama MiniGrid 3.2.0 / Gymnasium, native seven-action
interface, BAGI M23 public encoder and actual native-effect evaluator,
and the **already fitted** source stage-6 action-aware and success-only
control PyTorch models. Do NOT invent another dynamics algorithm.
Source experiment https://github.com/AncilBenny/BAGI/actions/runs/43647434248
at commit `472ed53f37af64ae6c7e2ed9c12f74412f7ef34f`, artifact
`bagi-research-674` ID **14134639518**, GitHub-reported full ZIP
SHA-256
`de2270383c7de3d52dea54f4be208e0383b83d52e33001a27ba85cbff512e82e`.
Download into ignored isolated run directory; independently validate
the ZIP SHA, internal source commit/mode, stage-6 success gate, per-seed
`checkpoint_sha256` and exact numeric tensor shapes before loading with
`numpy.load(allow_pickle=False)` and `torch.load_state_dict`.
No pickle, arbitrary code, remote checkpoint execution or checkpoint
weight updates. **Eight paired model seeds 941,947,953,967,971,977,983,991**
are fixed from prior study.

## New independent OOD source MONITOR, no source TRAIN or target FINAL

Freeze previously unopened source layouts:
- DoorKey-6x6: **602001–602032**.
- Fetch-5x5-N2: **612001–612032**.
- PutNear-6x6-N2: **622001–622032**.

All 96 are disjoint from original/Stage-6 TRAIN, previous M23 MONITOR,
source sealed FINAL and UnlockPickup 453001–453032 FINAL. The new
collector uses the same public-only source controller, but changes
the action-intervention probability from **0.35 to 0.65**.
On an intervention, choose from the **public front cell only**:
pickup (3) when key/ball/box visible, drop (4) when front is
floor/empty, toggle (5) when locked door visible, otherwise
uniformly choose a native 3/4/5 attempt. Do NOT inspect private
carrying, native door locks, full grids, hidden state, or future
observations to decide an action. Each chosen action is genuinely
executed once. The independent native evaluator records exact
effective pickup/drop/unlock and no-effect witnesses, public
before/after proxy, native action sequence hashes, override counts
and task-specific episode completions separately.

## Baselines and hard-negative design

Compare on exactly the same new public source MONITOR transitions:

1. **Frozen probe-trained learned model** (eight numeric-only stage-6 fits).
2. **Frozen success-only matched model** (eight numeric-only stage-6 fits).
3. **Public front-cell precondition rule** from Stage 6.
4. **Public observation-history rule**: maintain a *publicly inferred*
   carried-object type (or none) from previously executed actions plus the
   visible front-object disappearance/appearance after actual pickup
   and drop. Do not peek at native carrying. It may predict a drop
   only when it has previously observed a successful pickup, and
   pickup only when it has not. Treat locked-door toggling as eligible
   only when the public memory indicates carrying a key.
   If the public effect is ambiguous, retain uncertainty and report
   errors; this is not a privileged oracle.

Report the **rule-eligible subset** where Stage-6 public
precondition rule predicts an effect: it must contain at least
**16 real native successes and 16 physically ineffective actions**
to support the primary hard-negative discrimination test.
Also report all native action3/4/5 events, native effects by task,
positive recall, no-effect specificity, class support, macro-F1 and
source task physical successes (not merely reward). The new
action policy could create too few source successes: if the frozen
nonvacuity minimums fail, mark HOLD and do not substitute previous
monitor layouts or retune the intervention probability.

## Frozen decision gates, no posthoc tuning

- **Robust model gate** requires BOTH nonvacuous supports and mean
  source hard-negative-subset macro-F1 for probe model > **0.75**,
  with macro-F1 exceeding the success-only control and front-cell
  rule by > **0.05** in ≥**6/8** paired seeds each, and correct
  native-effect-positive recall >= **0.70** on the rule-eligible
  subset. Report seed-cluster 95% bootstrap and episode-correlation
  limitations; do not claim reliable superiority if the interval
  includes zero.
- **Reuse decision gate:** if the public-history rule achieves
  macro-F1 >= **0.95** on the same nonvacuous eligible subset and
  matches or exceeds the frozen model's mean macro-F1, prioritize
  integrating that established public-memory mechanism into the
  next source-only planner feasibility test instead of fitting
  additional redundant neural event classifiers. That is an
  engineering preference, not a neural research pass.

No retraining: **0 new model fits and 0 optimizer updates** by
design, despite newly executed native source MONITOR actions.
Historical 41,052 stage-6 native actions and 16 fits remain
cached/historical and cannot be counted as stage-7 work.
Any checkpoint inference/forward calls and actual environment
executed steps must be independently recorded. Still **zero
UnlockPickup target actions, zero compositional physical successes,
zero independent FINAL task episodes and zero planner-imagined steps**.
Workshop/XLand remain diagnostics.

## Source provenance, integration and failure interpretations

Reuse the existing serialized research workflow and dispatcher with
one source-HEAD-bound `m23stress` study request, only after tested
source code passes Ruff and pytest plus independent current push/PR
CI. No duplicate workflow or schedule adjustment. Put output under
ignored `runs/m23_rule_shift/`, preserve per-task public native
effects in numeric-only files, exact input/output SHA-256, versions,
action histograms and per-seed results in an uploadable artifact.
Append outcomes to `docs/autonomous-research-log.md`, retaining all
previous results, and defer main integration until PR review and
postmerge CI. Unresolved outcome should guide either (a) reuse of
effective public action-history memory or (b) better observational
state estimation; do not bypass failed old gates or open target FINAL.
