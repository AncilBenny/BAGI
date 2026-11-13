# Experiment seed policy

Collection, training, planner sampling, and evaluation are treated as separate sources of randomness. Guided and random collection use independent fixed seed ranges. Multi-seed studies vary the training/planner seed while keeping collection and evaluation seeds fixed across conditions.

This separation is required before interpreting changes in planner success as an effect of guided experience.
