# M23 stage 2: source-only world model — registered prediction gate FAIL

Executed 2026-11-14 UTC on source commit `1b0c72eedc17aa3e37b92dad482c8b5988dc044b`. Existing serialized workflow [#43633738861](https://github.com/AncilBenny/BAGI/actions/runs/43633738861) completed, with push CI #43633738813 and PR CI #43633743956 successful. The research job passed Ruff and pytest: 767 passed, 10 skipped, 44 subtests passed.

## Provenance and resource accounting

- Reused **16,832 cached actual native source TRAIN transitions** from verified artifact [run #43631440997](https://github.com/AncilBenny/BAGI/actions/runs/43631440997), executed source `594c670a0d501fbd3cafe78ef47d59bef4733c0a`. Input archive checksum verification in the training job reported OK. These are not new stage-2 interactions.
- **8 fresh PyTorch model fits** at seeds 941,947,953,967,971,977,983,991. Each fit: 6 epochs, batch 256, Adam 0.001, **396 actual optimizer steps**, **100,992 cached transition presentations**. Combined **3,168 optimizer steps** and **807,936 sample presentations**. There was no target training.
- **2,331 new native source MONITOR actions** (shared across all model evaluations), containing **72 executed interaction action 3/4/5 steps**. No UnlockPickup target actions, no target FINAL evaluations, no imagined planning, and no physical composition success was evaluated.
- Immutable GitHub artifact `bagi-research-670`, ID **14129455984**, 8,197,290 bytes, GitHub-reported SHA-256 `947c697d455711624f71a7185fa5cfbd557109beeddcd6275786a7fad7ea9b78`, expiry 2026-12-14 19:49:35 UTC. Output archive interior was **not independently extracted** here.

## Numeric evaluation (preregistered metric)

| Seed | Learned native-interaction next-object macro-F1 | Persistence macro-F1 |
|---|---:|---:|
| 941 | .411148 | .604251 |
| 947 | .444012 | .604251 |
| 953 | .387835 | .604251 |
| 967 | .474261 | .604251 |
| 971 | .397803 | .604251 |
| 977 | .476296 | .604251 |
| 983 | .478285 | .604251 |
| 991 | .474320 | .604251 |

**Mean difference −0.1612557568**; required **>+.05** mean and on **at least 6/8 seeds**; achieved 0/8 positive seeds: **FAIL**. Next-object accuracy across all cells was .914–.921 compared with persistence .771816, but this class-imbalanced aggregate is **not evidence of event-level dynamics correctness**. The 72 native interaction steps give limited effective support; eight models shared one evaluation set. Model and artifact numeric hashes are expected in immutable output but not individually checked here.

**Scientific decision:** source-dynamics learning has not passed its prerequisite, so no composed-task or task-independent continual-learning claim is supported. Investigate rare-class balance and action-conditioned effect prediction on fresh, separately registered *source-only* MONITOR layouts, with matched training exposures and no earlier FINAL layouts. No retroactive gate change, target-final opening, duplicate workflow, or main merge. M22 remains separate.
