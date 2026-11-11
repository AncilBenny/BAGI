# BAGI M10-D0 — archived M9 deterministic action-concentration audit (2026-11-11 UTC)

**Status:** secondary analysis of *existing* M9 held-out evaluation, NOT a fresh experiment or a new M10 result. Original training source `943e7b2963a45d62b5f6ed25e0950be1f5ce3492`; GitHub Actions run 43014639399; retained artifact `bagi-research-642` ID 13851889475; verified ZIP SHA256 `95543df0c664eb80f367956538716ab17f359a38488880a6c4c780844aa83a88`. An external read-only analysis script was replayed against that ZIP before this source-only recording.

**Definition:** an evaluation episode is *action-concentrated* when one native action accounts for >=95% of actually executed actions. This threshold was selected **post hoc**; it is a descriptive diagnostic, not a preregistered hypothesis test or an independently validated gate.

| Arm | Seed | Deterministic concentrated / episodes | Sampled concentrated / episodes |
| --- | ---: | ---: | ---: |
| PPO reactive_history | 151 | 12/12 | 0/36 |
| PPO reactive_history | 167 | 12/12 | 0/36 |
| RecurrentPPO recurrent_history | 151 | 12/12 | 0/36 |
| RecurrentPPO recurrent_history | 167 | 12/12 | 0/36 |
| **Total** | | **48/48** | **0/144** |

All deterministic episodes lasted 256 native actions, 3072 per model; sampled episodes also lasted 256, 9216 per model. The existing M9 physical goal result remains **0/72 sampled for PPO and 0/72 for RecurrentPPO**, and **0/24 deterministic per arm**. Sampled action diversity is not equivalent to competent physical control. The 12 final layouts are shared across models, and three sampled repeats on each layout are correlated. Only two independent model initializations per arm.

**Interpretation:** deterministic action concentration is consistent with a possible action-selection bottleneck, but does not isolate its cause. Observation representation, learned logits, training dynamics, partial observability and task geometry are confounded. The archive has no counterfactual intervention that would establish why the dominant action was selected. Do not tune on M9 final layouts 118001–118012 or turn this post-hoc threshold into an M10 primary success metric.

**Next falsifiable test:** preregister an action-concentration metric *before* new M10 fits alongside the train-only key→door opportunity funnel. Evaluate the M10 public one-hot/direction intervention on fresh untouched layouts with native physical key pickup, locked-door opening and rewarded goal as the primary endpoint; match native interactions and count actual optimizer updates. Evaluator-only private state remains outside policy inputs and rewards. Preserve historical negative gates.

**Accounting for this documentation-only milestone:** 0 new simulator actions, 0 optimizer updates, 0 fits, 0 independent episodes, 0 teacher queries, 0 workflow changes. No generated model or archive assets committed. Main integration remains subject to reviewed PR #1, exact-source push/PR CI and local Ruff/pytest.
