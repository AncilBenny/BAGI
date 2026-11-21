# BAGI M10-P1 — independent public-representation comparison: executed result

**Date:** 2026-11-12 UTC. **Status:** completed native six-fit experiment; **preregistered primary gate FAIL**. This is new measured evidence, not a training plan, archive-only replay, or demonstration of task-independent continual learning.

## Executed source, provenance, and reuse

- **Admitted research source:** commit `778582135aace0f85c7b910c89cc1fde2ffa7fe1` (ADD-only exact-parent request `.github/bagi-m10p1-once.json`, parent tested implementation `b3ea38601546cebf9292b1bf6717699631bb6b36`). Study `m10p1`. Research ran in the EXISTING serialized [BAGI Research Loop run 43198019630](https://github.com/AncilBenny/BAGI/actions/runs/43198019630), push event, completed SUCCESS, both source/PR CI admission and the real compute job passed. Exact request CI: push [43198019735](https://github.com/AncilBenny/BAGI/actions/runs/43198019735) PASS Ruff/pytest, PR [43198029001](https://github.com/AncilBenny/BAGI/actions/runs/43198029001) PASS Ruff/pytest/merge provenance; source-parent push 43197618202 and PR 43197626509 PASS.
- **Research artifact:** `bagi-research-647`, ID **13934712131**; downloaded ZIP SHA-256 **`132de45ffaedc4df5de97220be29b2cae5767ab07610fac8a030ee17c9f195e3`**, matching the GitHub Actions artifact digest; ZIP CRC returned no errors. The internal source-commit marker and study-mode marker match the summary. Frozen config `configs/m10_public_representation.yaml`, SHA-256 **`a99ea20cc9d4daa2318509c8aa2f668ef0df1ea52caf85d50f9415868e109ea8`**. Numeric-only NPZ checkpoints inspected with `allow_pickle=False`; SHA-256, member shapes, dtypes and finite values match **all six** manifests. No untrusted pickle checkpoint loading.
- **Pinned measured dependencies:** Gymnasium 1.4.0, MiniGrid 3.2.0, Stable-Baselines3 2.9.0; upstream official `OneHotPartialObsWrapper` and PPO, native sparse DoorKey8x8 environment, CPU with one Torch thread. Contributions are maintained-implementation integration and controlled evaluation, **not a new RL algorithm**. Primary references: [MiniGrid official wrappers](https://github.com/Farama-Foundation/Minigrid/blob/v3.2.0/minigrid/wrappers.py) and [Schulman et al. PPO](https://arxiv.org/abs/1707.06347).
- **Actual training protocol:** arms `raw_history`, `raw_direction`, `onehot_direction`; freshly initialized model seeds **211/223** for each arm; **131072 native train actions and 4096 intercepted optimizer.step calls per fit**, total **786432 training actions, 24576 optimizer steps, six fresh fits**. Native rollout128, minibatch64, two PPO epochs, entropy0, actor/critic hidden widths32, native episode horizon256; **zero premature incomplete episode resets**. Actual source-recorded first reset seeds 106500000 (seed211) and 112500000 (seed223); reset stream hashes recorded per model in the artifact. Training reward-only goal terminations by arm and model seed: raw (0,1), raw+direction (2,1), one-hot+direction (3,3). These TRAIN rewards are **not audited physical goal-completion evidence**.
- **Policy input/parameter counts:** raw image+public mission+four actually executed actions =243 input features, **15880 parameters**; raw+PUBLIC heading =247 inputs, **16136 parameters**; upstream one-hot partial image+mission+actual actions+heading =1080 inputs, **69448 parameters**. Action semantics 0–6 and original reward unchanged. Simulator-private key/door/goal snapshots belong **only to external audit** and are never PPO inputs, training rewards, or policy selection signals.

## Total actual actions and evaluation integrity

- **TRAIN-MONITOR** (not independent): untouched-at-start 119101..119106, before-fit deterministic, after-fit deterministic and sampled; **46080** additional native actions, **180** resets/episodes. M10-D3 train-monitor layouts 119001..119012 were excluded.
- **INDEPENDENT FINAL**: separate untouched-at-start 120001..120012, sampled three times per layout and deterministic once per arm/model. **73670** native actions, **288** episodes/resets. Stochastic repeats on the same layout are **correlated** and do not create new independent layouts. Final split was gated until all six fits passed training integrity. Only two final episodes ended early by completing a physical goal, explaining 58 fewer final actions than the 73728 maximum.
- **Total new native interactions, this M10-P1 run:** 786432 training + 46080 monitor + 73670 final = **906182** actual native steps; **468** total evaluation episodes (180 train-monitor and 288 independent); **0 teacher queries**. In-run measured duration **491.5 seconds** (separate GitHub setup/install overhead). Archived M9 and executed M10-D3 step counts are NOT counted as new M10-P1 work.
- Independent read-only ZIP/JSON/NPZ audit: all 468 per-episode action histograms sum to native actions and the per-model/group sums reconcile with all run totals; all six model checkpoints match their numeric manifests. All recorded effective door-opening attempts respect valid opportunity counts; physical-goal claims have effective pickup strictly before door opening and before goal termination. No hidden held-out seeds accessed for tuning and no generated model output or private state trajectory committed.

## Independent physical outcomes (two model seeds, 12 layout seeds each)

These are **sampled** totals pooled over 2 initializations × 12 distinct final layouts × 3 correlated action-sampling repeats = 72 sampled episodes per arm:

| Native outcome | Raw image/history | Raw + public heading | One-hot + public heading |
|---|---:|---:|---:|
| Episodes with effective key pickup | 46/72 | 42/72 | 52/72 |
| Episodes with valid locked-door toggle opportunity | 3/72 | 6/72 | 12/72 |
| Valid opportunity **steps** | 21 | 22 | 55 |
| Valid native toggle **attempts** | 0 | 1 | 2 |
| Episodes with effective door opening | 0/72 | 1/72 | 2/72 |
| **Full key→door→native rewarded goal episodes** | **0/72** | **0/72** | **2/72** |
| Sampled >=95%-one-action concentration | 0/72 | 0/72 | 0/72 |

The one-hot arm's two independent physical successes came **only from model seed223**, on **distinct previously untouched final layouts**:
- Layout120006, sampled repeat1, actual key pickup at action20, effective door opening at action181, terminal physical goal at action209; reward0.70609375.
- Layout120012, sampled repeat3, actual key pickup at action9, effective door opening at action73, terminal physical goal at action245; reward0.65546875.

Both events are real native simulated key→door→goal successes, not reward-only or hand-scripted outcomes. **Seed211 one-hot achieved 0/36 sampled full goals**. Raw and raw+direction achieved zero for both model seeds.

In **deterministic independent** evaluation (24 episodes per arm), physical full goals were **0/24 across all three arms**. >=95%-single-action concentration occurred in 23/24 raw, 24/24 raw+direction, and 24/24 one-hot episodes; no deterministic key pickups or effective door openings were measured.

## Preregistered decision: H-M10-P1 FAIL

The frozen gate required (1) at least **+4/72** sampled physical goals for one-hot+direction versus raw, (2) **at least one** independently physical goal in **each** one-hot initialization, and (3) all training integrity checks. The actual one-hot difference is **+2/72**, all from seed223; training integrity checks passed, but criteria (1) and (2) failed. This is a **negative/insufficient result**, not an established representation improvement or a statistical causal effect.

One-hot+direction produced more sampled valid door opportunities (12/72 episodes versus 3/72 raw and 6/72 raw+heading), followed by two effective openings and goals. Those observations support additional investigation but cannot identify the mechanism: the one-hot network has about 4.4 times the raw network's parameters, the only successes came from one model seed, and repeated action-sampling rollouts correlate within layouts. Public direction alone did not yield full independent goals. All three deterministic policies remain severely action-concentrated, while stochastic task completion remains rare. The task is one sparse MiniGrid family: **no robust transfer, retention, independent cross-domain competence, real-world physical performance or general intelligence has been demonstrated**.

**Next falsifiable research decision (not executed here):** keep these M10-P1 final layouts excluded from future training/tuning. Test a train-only, observation-history-based **action-effect/precondition predictor** or another established policy-control baseline before deciding whether to scale exploration. Require separate independent goal/task distributions and an explicit retained-skill baseline when pursuing genuine task-independent continual learning; Workshop and XLand remain diagnostics, not substitutes for cross-task evidence. Freeze matched model capacity/inputs and control any attempted-policy-selection changes separately. A new preregistration, not posthoc interpretation of M10-P1 final layouts, is needed.

**Governance:** existing research workflow, schedule, negative historical gates and old artifacts preserved. Main integration remains **HOLD** while [PR #1](https://github.com/AncilBenny/BAGI/pull/1) lacks reviewed approval; this result documentation needs independent new push/PR CI before any reviewed integration.
