# Fixed-background multi-seed sample-efficiency study

The primary sample-efficiency sweep keeps the random background experience fixed across guided-data budgets. A separate multi-seed runner varies only the world-model/planner training seed while keeping collection seeds and evaluation episode seeds fixed.

Run the lightweight single-seed study with `make sample-efficiency`. Run the more expensive 3-seed study with `make sample-efficiency-multiseed` only when variance estimates are needed.

Report individual seed results plus mean and standard deviation. Keep real environment steps separate from imagined planner steps.
