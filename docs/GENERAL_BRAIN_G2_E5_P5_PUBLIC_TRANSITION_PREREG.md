# G2-E5-P5 — TRAIN-only empirical public transition identifiability (preregistered)

**Status: prospective diagnostic, not yet executed on archived real TRAIN arrays.**
Frozen on 2026-10-26 after the P4 upstream RND integration STOP. This is a
small, simulator-free, evaluator-free measurement of what a task-neutral
observation/action model could learn, **not** a new controller, policy update,
RND substitute, held-out TEST evaluation, causal identification or evidence
of AGI/transfer. No new native action or workflow is authorized by this plan.

## Question and existing work reused

A reactive public-observation predictor can only predict reliably if repeated
identical public view + executed action has a sufficiently consistent next
public view. Can the already-collected **TRAIN** traces empirically distinguish
next-public outcomes at identical inputs? Does a four-action *executed*
history disambiguate some repeats, or mostly fragment support? Alternatives:
no repeated keys due to sparse high-dimensional observations; no observed
conflicts despite latent aliasing; ambiguity from stochasticity rather than
hidden state. All are nonpromotion outcomes, not a reason to invent an oracle.

Reuse `bagi/experiments/g2_e5_train_history_audit.py` for exact ZIP SHA256
verification, numeric-only `np.load(..., allow_pickle=False)` and the strict
TRAIN key whitelist. Reuse existing source-only G2-E1 and G2-E2-P0 artifacts:

- E1 native run `40165691302`, artifact `12588790887`, ZIP SHA256
  `aaf8eb011763bf47e1d196771e257eb13c4eb35294bbeed2b71757ade525f073`.
  Read only `xland_train_{observations,actions,groups}` and
  `workshop_train_{observations,actions,groups}`; never E1 TEST arrays.
- E2 native run `40185492543`, artifact `12596191672`, ZIP SHA256
  `b000a02a2eacfb8aa07befa151e0bbfb2c5bf7fd430f8bd896bd11f90f57436a`.
  Read only original TRAIN `actor_index==0`, independently for policy seeds
  17 and 29, with current/next 5x5 public views, executed action and episode
  index. Never use E2 TEST actor arrays, evaluator events or model weights.

## Frozen arithmetic and measurements

Exactly four source/behavior datasets: E1 XLand TRAIN (all existing frozen BC
and random behavior episodes, **not** an independent test); E1 Workshop TRAIN
(including duplicated fixed-script episodes, explicitly nonindependent); E2
Workshop DQN TRAIN seed17; E2 Workshop DQN TRAIN seed29. No balancing,
retraining, policy-selection, new sampling or seed substitution. For E1, use
consecutive observations only *within the same episode*, dropping each final
row because its successor is unavailable. For E2 use stored executed next
public view. XLand 28 executed-history feature suffix bits are excluded only
from the **next-view target**, not current legal actor features; Workshop has
no such suffix. Compare exact public bytes + executed action with history
widths **0 and 4**, where history is padded and reset per episode. Never use
teacher-intended actions or privileged position, full grid, key/door state,
physical BLUE evaluator or `info.events`.

Report per dataset and history width: TRAIN rows and episodes, number of
unique exact public-action keys, rows in repeated keys, keys/rows with >1
observed next public view, and cross-episode conflicting keys/rows. Inference
is **descriptive**; identical views can hide different states even when no
conflicting outcome is sampled. No repeated keys = **inconclusive**, not a
Markov proof. E1 fixed-script duplicates cannot establish independent
support. Never interpret a next-view change as native physical success.

## Gate and next decision

No automatic promotion threshold or learned-policy claim: P5 can only justify
whether a separately preregistered public-only recurrent/effect-model study
has enough repeated TRAIN support to be interpretable. If conflicts appear,
report them without claiming they are caused by hidden state. If none appear,
report repeat coverage and stop; do not claim fully observed dynamics. The
next policy experiment, if any, requires a new prospective matched-control
protocol on source-disjoint XLand and **independently learned** Workshop
policies with actual native reward/effect gates, not scripts or BLUE-specific
rules. The existing P4 native RND study remains STOP pending a compatible
hash-verified wheel, complete dependency resolution and counted CPU/SB3
integration; P5 does not relax any P4 preregistration.

**Execution command after obtaining the two already-published verified ZIPs:**

```bash
python -m bagi.experiments.g2_e5_p5_public_transition_identifiability \
  --e1-zip /path/to/g2_e1.zip --e2-zip /path/to/g2_e2.zip \
  --output /path/to/g2_e5_p5_train_only.json
```

Before running on actual archives, verify current research HEAD/CI and archive
hashes; retain exact result and action-accounting log. Synthetic tests alone
are **not** a real-data result. No native actions/resets, teachers, policy or
predictor updates, TEST reads, unsafe pickle, workflow changes, billing,
deployments, main-branch writes or duplicate automations.
