# Gate A — matched categorical public observation diagnostic

Predeclared after frozen observation probe #45942181415 at
d1044ca8b6633d1a238456f1343ec668bc9956af, before new training or evaluation.
All 180 live/stale paired stochastic paths identical; no evidence of
effective changing-image use. Reuse Farama OneHotPartialObsWrapper
(https://minigrid.farama.org/_modules/minigrid/wrappers/) rather than
invent a vision architecture. Its categorical image contains the same
public object/color/state data, not global positions or task solutions.
Transform only the existing local image; recover the original integer
categories from the scalar /10 encoding and retain all 14 tail slots.
Unit test checks equality with the maintained raw observation wrapper.
No changes to reward, horizon, action space, learning algorithm or
recurrent hidden size; wider input means more model parameters, so
this is a representation-plus-capacity comparison, not isolated
causal evidence about categorical encoding.

Fresh matched scalar and categorical RecurrentPPO 2.9.0 controls,
seeds 101/103/107, 65536 actions per model per arm, 256-action rollouts,
64 batch, 4 epochs, 64 LSTM hidden units. Train seed stream
41000000 + model-index*1000000; exploratory paired monitor stream
500000 later, 40 resets per seed, same reset/policy-sampling streams
for both arms. At 0,16384,32768,65536 native steps evaluate greedy
and frozen stochastic controllers; final stale-first-image and
memoryless last-TRAIN-interval action prior. Frozen optimizer/weights/RNG
invariance checked. Report per-seed native goal success at horizon 100,
success by 20, restricted mean actions (failure=100), actual traces and
paired differences; retain policy and Adam tensors with digests. New
fit is justified by changed representation and matched fresh scalar
control, not a repeat of the failed prior experiment. Do not open
confirmatory physical layout/start-state groups.

Prospective diagnostic target: categorical stochastic final >=80%
physical success per seed, >=10 percentage-point gain in success by
20 versus its own action prior AND stale control for every seed;
report scalar contrast on same seeds and both greedy curves.
The additional efficiency target was selected prospectively from
the previous frozen observation-dependence result. If target fails,
do not expand world models or continue arbitrary tuning; audit
observation sufficiency / task difficulty and select a genuinely
compatible second task family with a maintained baseline. If target
passes, validate common action semantics and persistent state before
a sequential curriculum and independent physical split. Neither arm
is a continual-learning baseline or a confirmatory experiment.

Existing read-only push CI marker [gate-a-categorical] installs pinned
RL libraries, executes full Ruff/pytest including smoke, then runs
the diagnostic with exact source checks. No additional schedule.
