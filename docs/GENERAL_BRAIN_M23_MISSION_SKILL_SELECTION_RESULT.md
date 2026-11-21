# M23 Stage 9 — public-mission source skill-selection native task results

**Executed 2026-11-15 06:05:26–06:05:43 UTC. Registered numerical gate: PASS; no independently novel action sequence relative to actor-only.** This source-only, hand-authored public mission-relevance baseline was preregistered *before* execution in `docs/GENERAL_BRAIN_M23_MISSION_SKILL_SELECTION_PREREG.md` at source revision `aec0867ee5b074b4a290b48b47d943f64e74301c`. Do **not** conflate this PASS with learned skill selection, improved source actor competence, or zero-shot UnlockPickup composition.

## Verifiable source, validation and immutable outputs

- Exact full native execution source: **`b1e151d978f18878807591661e53175ecbca0d54`** (`setup/initial-platform`), adding only the source-bound one-shot request `.github/bagi-m23skill-once.json`. Implemented and fixed source revision `d47f41df72e73baf7848250ae37a97eb4a7c8204` underwent successful independent push CI **43683328054** and PR CI **43683331556**.
- Existing serialized [BAGI Research Loop #43683527598](https://github.com/AncilBenny/BAGI/actions/runs/43683527598): CI/source audit and real research job both **SUCCESS**; exact-request push CI **43683527564** and PR CI **43683531217** passed. Research job ran Ruff and **803 pytest passed, 10 skipped, 44 subtests passed**.
- New output artifact **`bagi-research-679`**, GitHub artifact ID **14149467912**, 168,599-byte ZIP, expires **2026-12-15 06:05:43 UTC**. **Independently downloaded, extracted and recomputed ZIP SHA256** `f1769b1dd62fd758a1d9c6db651c27f6cbebf9e0270beb3afe84e4e35ccd7a11`, exactly matching GitHub-reported digest. Both `runs/m23_mission_skill_selection/summary.json` and duplicate research-loop copy have identical bytes (JSON SHA256 `0f8957f574492b01fe2e830426ce9bce01f4d4a0527eb971e6e4bbbd0d15c077`). Extracted source marker `research_branch_commit.txt` matches the exact source SHA and `study_mode.txt` is `m23skill`. Per-episode action hashes and native physical witnesses inspected from this JSON.
- Real execution duration within runner's source study: **15.946 seconds** (separately from CI/install time). Versions: numpy **2.4.6**, gymnasium **1.4.0**, minigrid **3.2.0**.

## True native episode and interaction accounting

Source tasks only: native `MiniGrid-DoorKey-6x6-v0` and `MiniGrid-Fetch-5x5-N2-v0`. Four arms (actor-only, public front effect, public history effect, hand-written mission-relevance gate), same maintained public source actors, native seven-action MiniGrid interface, fixed override probability 0.65 for intervened arms, 256-step wrapper. Fresh source TRAIN pilot DoorKey **681001–681064**, Fetch **691001–691064**; untouched source MONITOR DoorKey **682001–682064**, Fetch **692001–692064**. Each split ran **512** genuine native episodes (64 layouts × two tasks × four arms).

**Actual newly executed actions:** **45,674 TRAIN pilot + 45,511 independent MONITOR = 91,185** native source actions. Actual new neural fits **0**, optimizer steps **0**, imagined planner actions **0**, new target UnlockPickup native actions **0**, target FINAL episodes **0**. No Stage 6–8 actions are recounted. Private environment state was accessible only to independent physical-goal auditors.

## Independently recorded MONITOR outcomes: 64 native layouts per task and arm

| Source task and primary strict *physical* goal | Actor only | Front effect rule | History effect rule | Mission relevance gate |
| --- | ---: | ---: | ---: | ---: |
| DoorKey full goals | **57/64** | 9/64 | 19/64 | **57/64** |
| Fetch full goals | **40/64** | 23/64 | 21/64 | **40/64** |
| Combined physically audited full goals | **97/128** | 32/128 | 40/128 | **97/128** |

**Original preregistered numerical decision is PASS**: mission minus history paired strict full goals **+38 DoorKey** and **+19 Fetch**, total **+57/128**, above the preregistered +12 and positive in both tasks. Required public-mission versus history *proposal eligibility* disagreements **2,009** (1,946 DoorKey, 63 Fetch) exceed minimum 32; mission has 97 full goals versus 97 for actor-only (no deficit). Prespecified task-stratified paired-bootstrap 95% interval for total mission-minus-history gain **[+46, +68]**, above zero. This is an effect of filtering *disruptive interventions* relative to weaker override filters, not a learned transfer result.

**Reward versus physical state is not interchangeable.** DoorKey reward-positive 57/9/19/57 equals external audited strict full goals. Fetch rewarded terminations were **63 actor-only**, **39 front**, **34 history**, **63 mission**, whereas strictly audited physical full goals were **40**, **23**, **21**, **40** respectively; Fetch reward-only/nonqualifying episodes were **23/16/13/23**. The registered decision uses only real nonvacuous physical goals, never those reward-only episodes.

## Critical independent post-run trajectory-equivalence diagnostic

The ZIP contains each actually executed native action-sequence SHA-256 per layout and mode. Independent paired recomputation found:

| Source task | TRAIN-pilot mission-vs-actor identical action hashes | MONITOR mission-vs-actor identical action hashes |
| --- | ---: | ---: |
| DoorKey | **64/64** | **64/64** |
| Fetch | **64/64** | **64/64** |
| **Both tasks** | **128/128** | **128/128** |

The mission gate accepted 89 override proposals in DoorKey MONITOR and 43 in Fetch MONITOR, but **the complete executed sequences, native action counts, and strict physical outcomes were identical to actor-only across every paired layout**. The Stage 9 runner's `override_changed_from_controller` field is hard-coded zero, not an independently measured diagnostic; do **not** use it as evidence of distinct interventions. Action-sequence SHA and count equality are the independently checked evidence.

**Meaning:** the mission filter restored behavior of already strong, maintained public task-specific controllers under synthetic perturbations; it did **not** demonstrate task improvement over source actors, changed effective action choice versus actor-only, learning, causal compositional planning, generalization to unseen mission families, or autonomous novel skill selection. The prespecified numerical gate permits this actor-equivalent PASS, which reveals a missing *actual behavioral distinction* requirement in Stage 9's preregistration. Do not retroactively change that gate; annotate the inference limitation. The positive bootstrap interval measures mission filter versus **worse randomized history filter**, not an advantage versus actor-only.

## Decision and next distinct prospective question

**Registered numerical result: PASS; independent novel skill/action-selection contribution: NOT SHOWN.** Stage 8 remains **FAIL**, untouched. Reuse the established source actor as a strong baseline and do **not** merely create a task-specific hand-written gate that returns its own action path. A subsequent study must prospectively require independently witnessed *executed action-sequence differences from actor-only* on fresh source TRAIN/MONITOR layouts and physically audited gains or meaningful new task coverage. Better, use maintained BabyAI/Farama/RL machinery to train source-only mission-conditioned policy/skill arbitration with matched exposure and legitimate public observations; distinguish source task expert reuse from learned general skill selection. No tuning or new study on now-open Stage 9 682xxx/692xxx MONITOR, or Stage 8 652xxx/662xxx MONITOR; keep original M23 target `UnlockPickup` FINAL453001–453032 and prior M20/M21 finals sealed. Any later target composition study needs its own independently preregistered learnable planner, source effect coverage, checkpoints, controls and untouched final.

Existing research schedules and concurrency groups are unchanged. No generated checkpoints, private native state, credentials or trajectories committed into source. `setup/initial-platform` remains the research branch; **do not merge PR #1 to `main` without submitted independent review and current CI/source/merge provenance**.
