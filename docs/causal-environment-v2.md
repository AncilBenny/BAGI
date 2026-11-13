# Workshop v0.2: Mandatory causal gates

## Why this revision exists

The original Workshop map contained a switch, door, key, lock, and goal, but the geometry did not require the agent to use that chain to reach the goal. A direct route could bypass all four causal objects. Results collected in that environment remain useful as navigation/model-planning diagnostics, but they must not be interpreted as evidence that the agent learned a necessary causal chain.

## v0.2 geometry

Workshop v0.2 adds two internal barriers while preserving the existing guided curriculum trajectory:

1. The start and switch are above the first barrier.
2. The only opening through the first barrier is a door. The switch must be activated before that door is passable.
3. The key lies between the two barriers.
4. The only opening through the second barrier is the lock. The key must be collected before that lock is passable.
5. The goal lies beyond the second barrier.

Therefore every successful trajectory must satisfy the structural order:

`switch -> door -> key -> lock -> goal`

The observable events remain `switch_activated`, `key_picked_up`, `lock_opened`, and `goal_reached`.

## Research interpretation

Do not compare v0.1 and v0.2 success rates as if they were measurements from the same task. v0.2 is a stricter environment and begins a new experimental series. The first fixed-background sample-efficiency run is written to `runs/sample_efficiency_causal_v2` so older artifacts are preserved.

Before any causal-learning claim is made from v0.2, the protocol should again establish fixed collection backgrounds, multiple training seeds, planner-seed separation, and eventually held-out layout generalization.
