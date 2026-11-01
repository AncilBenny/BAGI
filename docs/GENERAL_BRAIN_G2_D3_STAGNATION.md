# G2-D3: Frozen-policy public-stagnation recovery (prospective native trial)

**Scope:** a single bounded comparison on the existing general-purpose BAGI
`PublicStagnationEscape` option. This option uses only public observations,
action probabilities from the SAME frozen model and actual executed actions.
BLUE-square placement is a physical evaluator criterion, NOT core policy logic.

## Evidence and hypothesis

Independent D2 run `40129855962`, artifact `12574088395`, yielded 5/16 genuinely
physical BLUE placements. A follow-up audit of those EXISTING, not newly
sampled traces found 8/11 failures ended in >=20 repeated DROP actions,
although an unchanged public view may still conceal hidden physical progress.
Hypothesis: after three identical commands with unchanged public observation
and zero native reward, selecting the highest-probability different action
from the SAME BC policy may escape a harmful loop. Alternative: it may disrupt
valid progress or merely move errors elsewhere. This is not a trained
world-model policy or a proven generally intelligent skill.

## Predeclared methods, inputs and budgets

The append-only research log preregistered this G2-D3 design BEFORE any
G2-D3 native action: original model seeds `17,29`, unchanged verified 787-row
12-epoch upstream `imitation.BC` (291 original TRAIN labels+248 old-skill
rehearsal+248 physical TRAIN labels), native original tasks
`hold,near,putdown`, untouched source-disjoint layouts `11700..11709`,
conditions `direct` and `stagnation_escape`, and a 40-action episode cap.

**Arithmetic-only correction to the original preregistration:** the frozen
120 episodes ×40 maximum actions equals **4,800**, not 2,400. No seeds,
conditions, learners, interaction cap or task are expanded by correcting
that multiplication, and this correction was made before executing D3.
No teacher queries, new TRAIN simulator steps or composition expert labels.
A source-verified original BC training procedure is compared in both arms,
not a different optimizer or model architecture. No discarded counterfactual
simulator actions occur in G2-D3.

The general `PublicStagnationEscape` object consumes a legal observation,
ranked probabilities from the same verified policy, the actual command
and the next public observation/reward. The domain adapter removes ONLY
the final 28 true-history bits from its public-stagnation signature; the BC
policy still sees all 115 original image+heading+public goal/rules+FOUR
truly executed prior actions. Every action-probability vector must sum to
one and its max must agree with upstream deterministic `policy.predict`.
No hidden simulator info, pocket or grid can influence selection.

External evaluator ONLY (after each action): effective BLUE pickup, later
effective BLUE drop, initial nonadjacency, final adjacency and native
reward, plus hidden-state changes when public observations stay constant.
Upstream game reward without BLUE manipulation is NOT a physical placement.
Report hold/near retention, action4 trailing-streak distribution, public
stagnation alerts/escapes, episode eligibility, all real native action
counts and zero teacher interactions. Never replace ineligible layouts.

## Decision gate

Require improvement of verified physical placement **without** hold/near
regression, with consistent behavior on both seeds and independent
confirmation before calling a recovered skill robust. If negative, this
falsifies this exact public-action plateau override on these layouts,
NOT learned world models, memory, goal-conditioned exploration or AGI
research broadly. Any learned world-model/spatial-belief replacement
must be trained only on TRAIN-eligible public observations and evaluated
separately in real Workshop with *learned* policies; the current
Workshop-symbolic tests of the common core do not prove cross-domain
learned transfer.

Established relevant methods: DAgger (Ross et al. 2011,
https://proceedings.mlr.press/v15/ross11a.html) and episodic novelty
exploration (Badia et al. 2020, https://arxiv.org/abs/2002.06038).
This public-stagnation controller is a small BAGI-specific ablation,
NOT an implementation of either full published algorithm.
