# Cumulative Transfer Gate A — initial native feasibility pilot

Status: engineering candidate, not scientific PASS. Based on validated
convergence source `53bc120e73cb3136840e29730dd2e0b1a2c6edb0`. Keep PR #1 reviewable; isolated branch
`setup/cumulative-transfer-a` until its own tests pass and integration is safe.

## Question and reuse decision

Does prior experience accelerate acquisition of a new task while retaining
competence on the old one? Use the existing common AgentRuntime and PyTorch
GRU, standard target-network TD learning and whole-episode replay. No new
algorithm claim. The original frozen DQN file remains untouched. Reuse the
maintained Farama MiniGrid and Gymnasium native environments; verified
MiniGrid registration in https://github.com/Farama-Foundation/Minigrid/blob/main/minigrid/__init__.py.
SB3-Contrib RecurrentPPO is already supported (see reuse inventory), but
on-policy PPO does not provide off-policy episode replay. AMAGO is a stronger
future reference, not a dependency of this small diagnostic.

Research basis: [Continual World](https://papers.nips.cc/paper/2021/hash/ef8446f35513a8d6aa2308357a268a7e-Abstract.html),
[CORA](https://proceedings.mlr.press/v199/powers22b.html),
[CLEAR](https://proceedings.neurips.cc/paper_files/paper/2019/hash/fa7cdfad1a5aaf8370ebeda47a1ff1c3-Abstract.html),
[primacy bias](https://proceedings.mlr.press/v162/nikishin22a.html),
[plasticity loss](https://proceedings.mlr.press/v232/abbas23a.html),
[AMAGO](https://github.com/UT-Austin-RPL/amago).
The boundary head reset is a single inspired ablation, not a faithful
implementation of CLEAR or the full primacy-bias method.

## Pilot scope

A: MiniGrid-Empty-Random-6x6-v0, local image only, native goal arrival.
B: CartPole-v1, public four-vector, native survival for 100 steps without
termination (modified horizon for feasibility, not standard 500-step score).
Separate observation slots, explicit public interface indicator and legal
native action count; previous executed action included. The shared GRU
and output head are carried, not separate per-task networks. The adapter
does not provide hidden physics or success truth to the learner. CartPole
and navigation have dissimilar dynamics, so positive transfer is not assumed.
Both are intentionally easy probes; Workshop is a later diagnostic.

Three pilot seeds 41/43/47; each A trained once, four B conditions forked
from its state or fresh initialization: fresh, frozen, carry, head_reset.
Default 16 blocks x 256 native actions per phase, four optimizer updates
per train block. Frozen executes matched exposure but no optimizer updates.
Sample full trajectories with balanced domain sampling. At every B block
boundary measure frozen B success, before/after A success. Report B success
AUC, A change, actual updates and replay presentations. Seed streams in a
pilot-only 7,000,000 namespace; repeated evaluation seeds across checkpoints
are paired exploratory monitoring, not untouched final data. MiniGrid has
limited layouts and seed separation is not layout separation. A native
episode cut by a block boundary is not a physical goal; targets may bootstrap
at that artificial cut. The pilot is intentionally not powered for a
scientific decision. No competence or positive-transfer claim from CI smoke.

Command after admission:
```sh
python -m bagi.experiments.cumulative_transfer --output runs/gate-a-pilot
```

Before confirmation: demonstrate single-family competence, inspect actual
rewards/learning curves and throughput; compare against the existing DQN and
a maintained recurrent/continual baseline, test checkpoint disk resume and
plasticity probe, allocate independent layout/initial-condition groups (not
just reset seeds), freeze disjoint test manifest, power on pilot variance.
Predeclare primary B physical success AUC gain with paired seed confidence
interval above zero, practical target 15% fewer interactions to competence
and at most 10 percentage-point A degradation; these are design targets
not yet registered confirmation thresholds. Reverse order and untouched
third family only after configuration freeze. Negative or underpowered
results change the next decision; no world-model expansion before evidence.

No pilot or confirmation run is claimed at candidate publication. CI tests
exercise native actions, actual TD updates, frozen evaluation, equal budgets,
and non-overwrite. Source and outputs must be pinned and independently
verified before interpreting a full pilot.
