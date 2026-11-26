"""M1 is an opt-in mode of the EXISTING gated research runner, not a new CI job."""

from pathlib import Path

import yaml

WORKFLOW = (
    Path(__file__).resolve().parents[1]
    / ".github" / "workflows" / "bagi-research-loop.yml"
)


def test_m1_and_legacy_share_ci_and_one_serialized_runner_without_daily_quota():
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    assert workflow["concurrency"]["group"] == "bagi-coordinated-development"
    # PyYAML may parse the YAML 1.1 key "on" as the boolean True.
    triggers = workflow.get("on", workflow.get(True))
    dispatch = triggers["workflow_dispatch"]["inputs"]["study"]
    assert dispatch["options"] == [
        "legacy", "m1", "m2", "m3", "m4", "m5", "m6", "m7",
        "m8", "m9", "m10", "m10p1", "m11", "m12", "m13", "m14", "m15", "m16", "m17",
        "m18", "m19", "m20", "m21", "m23", "m23wm", "m23balance", "m23memory", "m23effect", "m23probe", "m23stress", "m23action", "m23skill", "m23bc", "m23vision", "m23correct", "m23intervene", "m23recurrent", "m23actionmemory", "m23context", "m23continuation", "m23router", "m23eventmemory", "m23postcorrect", "m23rescue", "m23recoverylearn", "m23specialist", "m23mixedrecovery",
    ]
    assert dispatch["default"] == "legacy"
    assert dispatch["required"] is True
    assert triggers["schedule"] == [{"cron": "17 * * * *"}]
    assert triggers["push"] == {
        "branches": ["setup/initial-platform"],
        "paths": [
            ".github/bagi-m1-once.json",
            ".github/bagi-m2-once.json",
            ".github/bagi-m3-once.json",
            ".github/bagi-m4-once.json",
            ".github/bagi-m5-once.json",
            ".github/bagi-m6-once.json",
            ".github/bagi-m6-repair-once.json",
            ".github/bagi-m7-once.json",
            ".github/bagi-m8-once.json",
            ".github/bagi-m9-once.json",
            ".github/bagi-m10-once.json",
            ".github/bagi-m10p1-once.json",
            ".github/bagi-m11-once.json",
            ".github/bagi-m12-once.json",
            ".github/bagi-m13-once.json",
            ".github/bagi-m14-once.json",
            ".github/bagi-m15-once.json",
            ".github/bagi-m15-repair-once.json",
            ".github/bagi-m16-once.json",
            ".github/bagi-m17-once.json",
            ".github/bagi-m18-once.json",
            ".github/bagi-m19-once.json",
            ".github/bagi-m20-once.json",
            ".github/bagi-m21-once.json",
            ".github/bagi-m23-once.json",
            ".github/bagi-m23wm-once.json",
            ".github/bagi-m23balance-once.json",
            ".github/bagi-m23memory-once.json",
            ".github/bagi-m23effect-once.json",
            ".github/bagi-m23probe-once.json",
            ".github/bagi-m23stress-once.json",
            ".github/bagi-m23action-once.json",
            ".github/bagi-m23skill-once.json",
            ".github/bagi-m23bc-once.json",
            ".github/bagi-m23vision-once.json",
            ".github/bagi-m23correct-once.json",
            ".github/bagi-m23intervene-once.json",
            ".github/bagi-m23recurrent-once.json",
            ".github/bagi-m23actionmemory-once.json",
            ".github/bagi-m23context-once.json",
            ".github/bagi-m23continuation-once.json",
            ".github/bagi-m23router-once.json",
            ".github/bagi-m23eventmemory-once.json",
            ".github/bagi-m23postcorrect-once.json",
            ".github/bagi-m23rescue-once.json",
            ".github/bagi-m23recoverylearn-once.json",
            ".github/bagi-m23specialist-once.json",
            ".github/bagi-m23mixedrecovery-once.json",
        ],
    }
    jobs = workflow["jobs"]
    assert list(jobs) == ["audit", "research"]
    assert jobs["research"]["needs"] == "audit"
    assert "timeout-minutes" not in jobs["research"]
    assert jobs["research"]["if"] == (
        "needs.audit.result == 'success' && github.event_name != 'schedule'"
    )
    assert triggers["workflow_dispatch"]["inputs"]["force_retrain"]["type"] == "boolean"
    assert jobs["research"]["env"]["BAGI_RESEARCH_STUDY"] == (
        "${{ needs.audit.outputs.study }}"
    )
    assert jobs["audit"]["outputs"]["study"] == "${{ steps.mode.outputs.study }}"
    assert next(step for step in jobs["audit"]["steps"]
                if step.get("name") == "Fail closed on missing or unknown manual study"
                )["id"] == "mode"
    assert "budget" not in jobs
    assert jobs["research"]["env"]["BAGI_RESEARCH_SOURCE_SHA"] == "${{ needs.audit.outputs.research_sha }}"
    assert "ci_snapshot(api, sha)" in str(jobs["audit"]["steps"])
    assert "bagi_wait_for_research_ci.py" in str(jobs["audit"]["steps"])
    assert workflow["jobs"]["audit"]["steps"][0]["with"]["fetch-depth"] == 2
    assert "ci_snapshot(api, sha)" in str(jobs["research"]["steps"])


def test_m1_installs_pinned_upstream_rl_only_when_selected_and_no_legacy_fit():
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    steps = {row["name"]: row for row in workflow["jobs"]["research"]["steps"]
             if "name" in row}
    m1 = steps["Run admitted M1 public-observation goal-learning trial"]
    assert "timeout-minutes" not in m1
    assert "env.BAGI_RESEARCH_STUDY == 'm1'" in m1["if"]
    assert "steps.evaluated.outputs.cache-hit != 'true'" in m1["if"]
    assert "--execute" in m1["run"]
    install = steps["Install BAGI with pinned existing RL implementations (M1)"]
    assert "env.BAGI_RESEARCH_STUDY == 'm1'" in install["if"]
    assert "env.BAGI_RESEARCH_STUDY == 'm2'" in install["if"]
    assert "env.BAGI_RESEARCH_STUDY == 'm3'" in install["if"]
    assert "env.BAGI_RESEARCH_STUDY == 'm4'" in install["if"]
    assert "env.BAGI_RESEARCH_STUDY == 'm5'" in install["if"]
    assert "env.BAGI_RESEARCH_STUDY == 'm6'" in install["if"]
    assert "env.BAGI_RESEARCH_STUDY == 'm7'" in install["if"]
    assert "env.BAGI_RESEARCH_STUDY == 'm8'" in install["if"]
    assert "env.BAGI_RESEARCH_STUDY == 'm9'" in install["if"]
    assert "env.BAGI_RESEARCH_STUDY == 'm10'" in install["if"]
    assert "env.BAGI_RESEARCH_STUDY == 'm10p1'" in install["if"]
    m2 = steps["Run admitted M2 extended goal-learning and sampling diagnostic"]
    assert "env.BAGI_RESEARCH_STUDY == 'm2'" in m2["if"]
    assert "configs/m2_goal_learning.yaml" in m2["run"]
    assert "--execute" in m2["run"]
    m3 = steps["Run admitted M3 independent goal-memory replication"]
    assert "env.BAGI_RESEARCH_STUDY == 'm3'" in m3["if"]
    assert "configs/m3_goal_learning.yaml" in m3["run"]
    m4 = steps["Run admitted M4 evaluation-only frozen policy diagnostic"]
    assert "env.BAGI_RESEARCH_STUDY == 'm4'" in m4["if"]
    assert "m4_frozen_policy_eval" in m4["run"]
    m5 = steps["Run admitted M5 continual task-retention PPO study"]
    assert "env.BAGI_RESEARCH_STUDY == 'm5'" in m5["if"]
    assert "m5_continual_retention" in m5["run"]
    m6 = steps["Run admitted M6 DoorKey episode-continuity ablation"]
    assert "env.BAGI_RESEARCH_STUDY == 'm6'" in m6["if"]
    assert "m6_doorkey_continuity" in m6["run"]
    assert "--execute" in m6["run"]
    m7 = steps["Run admitted M7 frozen DoorKey native-action sampling diagnostic"]
    assert "m7_frozen_doorkey_sampling" in m7["run"]
    assert "--execute" in m7["run"]
    m8 = steps["Run admitted M8 fresh DoorKey PPO entropy ablation"]
    assert "m8_doorkey_entropy" in m8["run"]
    assert "--execute" in m8["run"]
    m9 = steps["Run admitted M9 fresh DoorKey recurrent PPO comparison"]
    assert "m9_recurrent_doorkey" in m9["run"]
    assert "--execute" in m9["run"]
    m10 = steps["Run admitted M10 fresh train-only physical-action diagnostic"]
    assert "m10_train_only_diagnostic" in m10["run"]
    assert "--execute" in m10["run"]
    m10p1 = steps["Run admitted M10-P1 fresh public representation PPO comparison"]
    assert "m10_public_representation" in m10p1["run"]
    assert "--execute" in m10p1["run"]
    m11 = steps["Run admitted M11 fresh public-effect retention study"]
    assert "m11_public_effect_retention" in m11["run"]
    assert "--execute" in m11["run"]
    m12 = steps["Run admitted M12 public native forward-effect prediction"]
    assert "m12_public_front_effect" in m12["run"]
    assert "--execute" in m12["run"]
    m13 = steps["Run admitted M13 public-only native physical-interaction coverage"]
    assert "m13_public_interaction_coverage" in m13["run"]
    assert "--execute" in m13["run"]
    m14 = steps["Run admitted M14 public native open-door goal replication"]
    assert "m14_public_open_door" in m14["run"]
    assert "--execute" in m14["run"]
    mixed = steps["Run admitted M23 diverse 48-layout 64:64 native mixed recovery rehearsal"]
    assert "env.BAGI_RESEARCH_STUDY == 'm23mixedrecovery'" in mixed["if"]
    assert "m23_source_diverse_mixed_recovery" in mixed["run"]
    assert "Install BAGI M23 diverse source mixed recovery rehearsal" in steps
    specialist = steps["Run admitted M23 learned public post-open recovery specialist"]
    assert "env.BAGI_RESEARCH_STUDY == 'm23specialist'" in specialist["if"]
    assert "m23_source_learned_recovery_specialist" in specialist["run"]
    assert "Install BAGI M23 learned public recovery specialist" in steps
    recovery = steps["Run admitted M23 actually executed recovery learned teacher-free GRUs"]
    assert "env.BAGI_RESEARCH_STUDY == 'm23recoverylearn'" in recovery["if"]
    assert "m23_source_executed_recovery_learning" in recovery["run"]
    assert "Install BAGI M23 actual-executed recovery learning" in steps
    rescue = steps["Run admitted M23 native genuinely executed teacher-after-open rescue diagnostic"]
    assert "env.BAGI_RESEARCH_STUDY == 'm23rescue'" in rescue["if"]
    assert "m23_source_executed_rescue_diagnostic" in rescue["run"]
    assert "Install BAGI M23 genuine post-open public teacher rescue" in steps
    post_correct = steps["Run admitted M23 matched source post-open on-policy recurrent correction"]
    assert "env.BAGI_RESEARCH_STUDY == 'm23postcorrect'" in post_correct["if"]
    assert "m23_source_post_open_onpolicy_bc" in post_correct["run"]
    assert "Install BAGI M23 equal-native-action post-open recurrent correction" in steps
    event_memory = steps["Run admitted M23 native public witnessed event-memory GRU"]
    assert "env.BAGI_RESEARCH_STUDY == 'm23eventmemory'" in event_memory["if"]
    assert "m23_source_public_event_memory_bc" in event_memory["run"]
    assert "Install BAGI M23 public source event-memory GRU" in steps
    router = steps["Run admitted M23 native public learned phase router"]
    assert "env.BAGI_RESEARCH_STUDY == 'm23router'" in router["if"]
    assert "m23_source_public_phase_router" in router["run"]
    assert "Install BAGI M23 public learned phase router" in steps
    continuation = steps["Run admitted M23 native public post-open continuation GRU BC"]
    assert "env.BAGI_RESEARCH_STUDY == 'm23continuation'" in continuation["if"]
    assert "m23_source_post_open_continuation_bc" in continuation["run"]
    assert "Install BAGI M23 source post-open continuation recurrent BC" in steps
    context = steps["Run admitted M23 native source interaction-context GRU BC"]
    assert "env.BAGI_RESEARCH_STUDY == 'm23context'" in context["if"]
    assert "m23_source_interaction_context_bc" in context["run"]
    assert "Install BAGI M23 source interaction-context recurrent BC" in steps
    action_memory = steps["Run admitted M23 native public executed-action-conditioned recurrent BC"]
    assert "--execute" in action_memory["run"]
    assert "m23_source_action_memory_bc" in action_memory["run"]
    assert "env.BAGI_RESEARCH_STUDY == 'm23actionmemory'" in steps[
        "Install BAGI M23 public native action-conditioned recurrent BC"
    ]["if"]
    recurrent = steps["Run admitted M23 native source recurrent-versus-stateless public GRU BC"]
    assert "--execute" in recurrent["run"]
    assert "m23_source_recurrent_bc" in recurrent["run"]
    assert "env.BAGI_RESEARCH_STUDY == 'm23recurrent'" in steps[
        "Install BAGI M23 public source recurrent and stateless GRU"
    ]["if"]
    intervention = steps["Run admitted M23 source equal-native-budget public intervention BC"]
    assert "--execute" in intervention["run"]
    assert "m23_source_intervention_bc" in intervention["run"]
    assert "env.BAGI_RESEARCH_STUDY == 'm23intervene'" in steps[
        "Install BAGI M23 source public native intervention BC"
    ]["if"]
    correction = steps["Run admitted M23 source on-policy corrective BC experiment"]
    assert "--execute" in correction["run"]
    assert "m23_source_corrective_bc" in correction["run"]
    assert "env.BAGI_RESEARCH_STUDY == 'm23correct'" in steps[
        "Install BAGI M23 source on-policy corrective BC"
    ]["if"]
    vision = steps["Run admitted M23 source public vision and rare-action BC factorial"]
    assert "--execute" in vision["run"]
    assert "m23_source_public_vision_bc" in vision["run"]
    assert "env.BAGI_RESEARCH_STUDY == 'm23vision'" in steps[
        "Install BAGI M23 source public vision and rare-action BC"
    ]["if"]
    m23bc = steps["Run admitted M23 source trained public mission BC study"]
    assert "--execute" in m23bc["run"]
    assert "m23_source_mission_bc" in m23bc["run"]
    assert "env.BAGI_RESEARCH_STUDY == 'm23bc'" in steps[
        "Install BAGI M23 source trained public mission BC"
    ]["if"]
    m23skill = steps["Run admitted M23 source public mission skill-selection study"]
    assert "--execute" in m23skill["run"]
    assert "m23_mission_skill_selection" in m23skill["run"]
    assert "env.BAGI_RESEARCH_STUDY == 'm23skill'" in steps[
        "Install BAGI M23 source public mission skill-selection study"
    ]["if"]
    m23action = steps["Run admitted M23 source public-memory real-task action selection"]
    assert "--execute" in m23action["run"]
    assert "m23_effect_action_selection" in m23action["run"]
    assert "env.BAGI_RESEARCH_STUDY == 'm23action'" in steps[
        "Install BAGI M23 source task physical-goal memory action selection"
    ]["if"]
    m23stress = steps["Run admitted M23 frozen out-of-policy native-effect stress test"]
    assert "--execute" in m23stress["run"]
    assert "m23_rule_shift" in m23stress["run"]
    assert "sha256sum -c -" in steps[
        "Fetch verified frozen M23 source effect checkpoint artifact"
    ]["run"]
    assert "env.BAGI_RESEARCH_STUDY == 'm23stress'" in steps[
        "Install BAGI M23 frozen source intervention policy-shift study"
    ]["if"]
    m23probe = steps["Run admitted M23 physical source-probe data and matched effect fits"]
    assert "--execute" in m23probe["run"]
    assert "m23_probe_effects" in m23probe["run"]
    assert "sha256sum -c -" in steps[
        "Fetch verified original M23 source TRAIN for randomized probe study"
    ]["run"]
    assert "env.BAGI_RESEARCH_STUDY == 'm23probe'" in steps[
        "Install BAGI M23 randomized source interaction probe study"
    ]["if"]
    m23effect = steps["Run admitted M23 source-only native interaction-effect prediction"]
    assert "--execute" in m23effect["run"]
    assert "m23_event_effects" in m23effect["run"]
    assert "sha256sum -c -" in steps[
        "Fetch verified M23 source TRAIN for native-effect prediction"
    ]["run"]
    assert "env.BAGI_RESEARCH_STUDY == 'm23effect'" in steps[
        "Install BAGI M23 native-effect source prediction study"
    ]["if"]
    m23memory = steps["Run admitted M23 source-only recurrent belief observation ablation"]
    assert "--execute" in m23memory["run"]
    assert "m23_belief_memory" in m23memory["run"]
    assert "sha256sum -c -" in steps[
        "Fetch verified M23 source TRAIN for memory comparison"
    ]["run"]
    assert "env.BAGI_RESEARCH_STUDY == 'm23memory'" in steps[
        "Install BAGI M23 source observation-memory prediction study"
    ]["if"]
    m23balance = steps["Run admitted M23 paired source world-model loss ablation"]
    assert "m23_class_balance" in m23balance["run"]
    assert "--execute" in m23balance["run"]
    assert "m23_class_balance/input/source.zip" in steps[
        "Fetch verified M23 source-only train transitions for balance comparison"
    ]["run"]
    assert "env.BAGI_RESEARCH_STUDY == 'm23balance'" in steps[
        "Install BAGI M23 class-balance source prediction study"
    ]["if"]
    m23wm = steps["Run admitted M23 fresh source public world-model fits"]
    assert "m23_public_world_model" in m23wm["run"]
    assert "--execute" in m23wm["run"]
    assert "sha256sum -c -" in steps[
        "Fetch verified source-only M23 public action trajectories"
    ]["run"]
    assert "env.BAGI_RESEARCH_STUDY == 'm23wm'" in steps[
        "Install BAGI M23 public world-model prediction study"
    ]["if"]
    m23 = steps["Run admitted M23 TRAIN-only causal source acquisition"]
    assert "m23_source_feasibility" in m23["run"]
    assert "--execute" in m23["run"]
    assert "env.BAGI_RESEARCH_STUDY == 'm23'" in steps[
        "Install BAGI M23 source-only physical-mechanism acquisition"
    ]["if"]
    m21 = steps["Run admitted M21 independent physical Fetch replication"]
    assert "configs/m21_fetch_replication.yaml" in m21["run"]
    assert "--execute" in m21["run"]
    assert "env.BAGI_RESEARCH_STUDY == 'm21'" in steps[
        "Install BAGI independent Fetch replication (M21)"
    ]["if"]
    m20 = steps["Run admitted M20 DoorKey to Fetch physical-object retention"]
    assert "configs/m20_fetch_rehearsal.yaml" in m20["run"]
    assert "--execute" in m20["run"]
    assert "env.BAGI_RESEARCH_STUDY == 'm20'" in steps[
        "Install BAGI physical Fetch transfer study (M20)"
    ]["if"]
    m19 = steps["Run admitted M19 reverse-order native physical retention"]
    assert "configs/m19_reverse_rehearsal.yaml" in m19["run"]
    assert "--execute" in m19["run"]
    assert "env.BAGI_RESEARCH_STUDY == 'm19'" in steps[
        "Install BAGI reverse-sequence rehearsal study (M19)"
    ]["if"]
    m18 = steps["Run admitted M18 cross-objective physical BC transfer"]
    assert "configs/m18_matched_replay.yaml" in m18["run"]
    assert "--execute" in m18["run"]
    assert "env.BAGI_RESEARCH_STUDY == 'm18'" in steps[
        "Install BAGI cross-objective BC study (M18)"
    ]["if"]
    m17 = steps["Run admitted M17 matched source versus target replay study"]
    assert "configs/m17_matched_replay.yaml" in m17["run"]
    assert "--execute" in m17["run"]
    assert "env.BAGI_RESEARCH_STUDY == 'm17'" in steps[
        "Install BAGI matched extra-label replay study (M17)"
    ]["if"]
    m16 = steps["Run admitted M16 sequential physical public BC retention"]
    assert "m16_sequential_public_bc" in m16["run"]
    assert "--execute" in m16["run"]
    assert "env.BAGI_RESEARCH_STUDY == 'm16'" in steps[
        "Install BAGI sequential public behavior cloning (M16)"
    ]["if"]
    m15 = steps["Run admitted M15 learned public-action behavior cloning"]
    assert "m15_public_action_bc" in m15["run"]
    assert "--execute" in m15["run"]
    assert "env.BAGI_RESEARCH_STUDY == 'm15'" in steps[
        "Install BAGI native public action cloning (M15)"
    ]["if"]
    assert "env.BAGI_RESEARCH_STUDY == 'm14'" in steps[
        "Install BAGI public open-door replication (M14)"
    ]["if"]
    assert "env.BAGI_RESEARCH_STUDY == 'm13'" in steps[
        "Install BAGI native physical coverage study (M13)"
    ]["if"]
    assert "env.BAGI_RESEARCH_STUDY == 'm12'" in steps[
        "Install BAGI for public front-cell calibration (M12)"
    ]["if"]
    assert "env.BAGI_RESEARCH_STUDY == 'm11'" in steps[
        "Install BAGI with maintained public-effect GRU (M11)"
    ]["if"]
    assert "sha256sum -c -" in steps[
        "Retrieve verified M6 numeric checkpoint artifact for M7"
    ]["run"]
    assert "--execute" in m5["run"]
    assert "--execute" in m4["run"]
    assert "sha256sum -c -" in steps[
        "Retrieve verified M3 numeric checkpoint artifact for M4"
    ]["run"]
    assert "--execute" in m3["run"]
    assert "pip install -e '.[dev,rl]'" in install["run"]
    for name in (
        "Restore causal-v2 checkpoint cache",
        "Build causal-v2 checkpoint on cache miss",
        "Run oracle search ceiling",
        "Run learned planner candidate sweep",
        "Run grounded imagination-gap diagnostic",
        "Install BAGI (legacy study)",
    ):
        assert "env.BAGI_RESEARCH_STUDY == 'legacy'" in steps[name]["if"]
    assert "make check" in steps["Verify research platform"]["run"]


def test_outputs_preserve_source_provenance_partial_evidence_and_study_cache():
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    steps = {row["name"]: row for row in workflow["jobs"]["research"]["steps"]
             if "name" in row}
    assert "m1_goal_learning/summary.json" in steps["Capture research metadata"]["run"]
    assert "study_mode.txt" in steps["Capture research metadata"]["run"]
    artifact = steps["Upload research results"]["with"]
    assert "runs/m1_goal_learning/*.npz" in artifact["path"]
    assert "runs/m1_goal_learning/progress.json" in artifact["path"]
    assert "runs/m2_goal_learning/progress.json" in artifact["path"]
    assert "runs/m2_goal_learning/*.npz" in artifact["path"]
    assert "runs/m3_goal_learning/progress.json" in artifact["path"]
    assert "runs/m3_goal_learning/*.npz" in artifact["path"]
    assert "runs/m3_goal_learning/summary.json" in steps["Capture research metadata"]["run"]
    assert "runs/m4_frozen_policy/summary.json" in steps["Capture research metadata"]["run"]
    assert "runs/m4_frozen_policy/progress.json" in artifact["path"]
    assert "runs/m5_continual_retention/progress.json" in artifact["path"]
    assert "runs/m5_continual_retention/*.npz" in artifact["path"]
    assert "runs/m5_continual_retention/summary.json" in steps[
        "Capture research metadata"
    ]["run"]
    assert "runs/m6_doorkey_continuity/summary.json" in steps[
        "Capture research metadata"
    ]["run"]
    assert "runs/m6_doorkey_continuity/progress.json" in artifact["path"]
    assert "runs/m6_doorkey_continuity/*.npz" in artifact["path"]
    assert "runs/m7_frozen_doorkey_sampling/progress.json" in artifact["path"]
    assert "runs/m8_doorkey_entropy/progress.json" in artifact["path"]
    assert "runs/m8_doorkey_entropy/*.npz" in artifact["path"]
    assert "runs/m9_recurrent_doorkey/progress.json" in artifact["path"]
    assert "runs/m9_recurrent_doorkey/*.npz" in artifact["path"]
    assert "runs/m9_recurrent_doorkey/summary.json" in steps[
        "Capture research metadata"
    ]["run"]
    assert "runs/m10_train_only_diagnostic/progress.json" in artifact["path"]
    assert "runs/m10_train_only_diagnostic/*.npz" in artifact["path"]
    assert "runs/m10_train_only_diagnostic/summary.json" in steps[
        "Capture research metadata"
    ]["run"]
    assert "runs/m10_public_representation/progress.json" in artifact["path"]
    assert "runs/m10_public_representation/*.npz" in artifact["path"]
    assert "runs/m10_public_representation/summary.json" in steps[
        "Capture research metadata"
    ]["run"]
    assert "runs/m11_public_effect_retention/progress.json" in artifact["path"]
    assert "runs/m11_public_effect_retention/*.npz" in artifact["path"]
    assert "runs/m11_public_effect_retention/summary.json" in steps[
        "Capture research metadata"
    ]["run"]
    assert "runs/m12_public_front_effect/progress.json" in artifact["path"]
    assert "runs/m12_public_front_effect/*.npz" in artifact["path"]
    assert "runs/m12_public_front_effect/summary.json" in steps[
        "Capture research metadata"
    ]["run"]
    assert "runs/m13_public_interaction_coverage/progress.json" in artifact["path"]
    assert "runs/m13_public_interaction_coverage/summary.json" in steps[
        "Capture research metadata"
    ]["run"]
    assert "runs/m14_public_open_door/progress.json" in artifact["path"]
    assert "runs/m14_public_open_door/summary.json" in steps[
        "Capture research metadata"
    ]["run"]
    assert "runs/m15_public_action_bc/progress.json" in artifact["path"]
    assert "runs/m15_public_action_bc/*.npz" in artifact["path"]
    assert "runs/m15_public_action_bc/summary.json" in steps[
        "Capture research metadata"
    ]["run"]
    assert "runs/m16_sequential_public_bc/summary.json" in steps[
        "Capture research metadata"
    ]["run"]
    assert "runs/m16_sequential_public_bc/progress.json" in artifact["path"]
    assert "runs/m16_sequential_public_bc/*.npz" in artifact["path"]
    assert "runs/m21_fetch_replication/summary.json" in steps[
        "Capture research metadata"
    ]["run"]
    assert "runs/m21_fetch_replication/progress.json" in artifact["path"]
    assert "runs/m21_fetch_replication/*.npz" in artifact["path"]
    assert "runs/m20_fetch_rehearsal/summary.json" in steps[
        "Capture research metadata"
    ]["run"]
    assert "runs/m20_fetch_rehearsal/progress.json" in artifact["path"]
    assert "runs/m20_fetch_rehearsal/*.npz" in artifact["path"]
    assert "runs/m19_reverse_rehearsal/summary.json" in steps[
        "Capture research metadata"
    ]["run"]
    assert "runs/m19_reverse_rehearsal/progress.json" in artifact["path"]
    assert "runs/m19_reverse_rehearsal/*.npz" in artifact["path"]
    assert "runs/m18_matched_replay/summary.json" in steps[
        "Capture research metadata"
    ]["run"]
    assert "runs/m18_matched_replay/progress.json" in artifact["path"]
    assert "runs/m18_matched_replay/*.npz" in artifact["path"]
    assert "runs/m17_matched_replay/summary.json" in steps[
        "Capture research metadata"
    ]["run"]
    assert "runs/m17_matched_replay/progress.json" in artifact["path"]
    assert "runs/m17_matched_replay/*.npz" in artifact["path"]
    assert "runs/m7_frozen_doorkey_sampling/summary.json" in steps[
        "Capture research metadata"
    ]["run"]
    restore = next(row for row in workflow["jobs"]["research"]["steps"]
                   if row.get("id") == "evaluated")
    assert "env.BAGI_RESEARCH_STUDY" in restore["with"]["key"]
    assert "env.BAGI_RESEARCH_STUDY" in steps.get(
        "Reuse any successful evaluation of this exact revision", restore
    )["with"]["key"]


def test_dispatch_guard_selects_study_before_training():
    import os
    import subprocess
    import sys

    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["audit"]["steps"]
    guard = next(row for row in steps
                 if row.get("name") == "Fail closed on missing or unknown manual study")
    assert guard["env"]["BAGI_DISPATCH_STUDY"] == "${{ inputs.study }}"
    assert guard["run"] == "python3 scripts/bagi_research_dispatch.py"
    assert steps.index(guard) < next(i for i, row in enumerate(steps)
                                     if row.get("id") == "verify")
    script = WORKFLOW.parents[2] / "scripts" / "bagi_research_dispatch.py"
    cases = (
        ("schedule", "", "refs/heads/main", True),
        ("workflow_dispatch", "legacy", "refs/heads/setup/initial-platform", True),
        ("workflow_dispatch", "m1", "refs/heads/setup/initial-platform", True),
        ("workflow_dispatch", "m2", "refs/heads/setup/initial-platform", True),
        ("workflow_dispatch", "m3", "refs/heads/setup/initial-platform", True),
        ("workflow_dispatch", "m4", "refs/heads/setup/initial-platform", True),
        ("workflow_dispatch", "m5", "refs/heads/setup/initial-platform", True),
        ("workflow_dispatch", "m6", "refs/heads/setup/initial-platform", True),
        ("workflow_dispatch", "m7", "refs/heads/setup/initial-platform", True),
        ("workflow_dispatch", "m8", "refs/heads/setup/initial-platform", True),
        ("workflow_dispatch", "m9", "refs/heads/setup/initial-platform", True),
        ("workflow_dispatch", "m10", "refs/heads/setup/initial-platform", True),
        ("workflow_dispatch", "m10p1", "refs/heads/setup/initial-platform", True),
        ("workflow_dispatch", "m11", "refs/heads/setup/initial-platform", True),
        ("workflow_dispatch", "m12", "refs/heads/setup/initial-platform", True),
        ("workflow_dispatch", "m13", "refs/heads/setup/initial-platform", True),
        ("workflow_dispatch", "m14", "refs/heads/setup/initial-platform", True),
        ("workflow_dispatch", "m15", "refs/heads/setup/initial-platform", True),
        ("workflow_dispatch", "m16", "refs/heads/setup/initial-platform", True),
        ("workflow_dispatch", "m17", "refs/heads/setup/initial-platform", True),
        ("workflow_dispatch", "m18", "refs/heads/setup/initial-platform", True),
        ("workflow_dispatch", "m19", "refs/heads/setup/initial-platform", True),
        ("workflow_dispatch", "m20", "refs/heads/setup/initial-platform", True),
        ("workflow_dispatch", "m21", "refs/heads/setup/initial-platform", True),
        ("workflow_dispatch", "", "refs/heads/setup/initial-platform", False),
        ("workflow_dispatch", "unregistered", "refs/heads/setup/initial-platform", False),
        ("workflow_dispatch", "m1", "refs/heads/main", False),
        ("push", "m1", "refs/heads/setup/initial-platform", False),
    )
    for event, study, ref, admitted in cases:
        env = {**os.environ, "GITHUB_EVENT_NAME": event,
               "BAGI_DISPATCH_STUDY": study, "GITHUB_REF": ref}
        run = subprocess.run([sys.executable, str(script)], env=env,
                             capture_output=True, text=True, timeout=5, check=False)
        assert (run.returncode == 0) is admitted
        assert ("BAGI research dispatch HOLD" in run.stderr) is not admitted

def test_one_shot_push_request_rejects_modified_or_misbound_files(tmp_path):
    import json
    import os
    import subprocess
    import sys

    script = WORKFLOW.parents[2] / "scripts" / "bagi_research_dispatch.py"
    ref = "refs/heads/setup/initial-platform"

    def git(*args):
        return subprocess.check_output(
            ["git", *args], cwd=tmp_path, text=True, timeout=5
        ).strip()

    git("init", "-q")
    git("config", "user.name", "BAGI source regression")
    git("config", "user.email", "noreply@example.invalid")
    (tmp_path / "README.md").write_text("test\n", encoding="utf-8")
    git("add", "README.md")
    git("commit", "-qm", "test parent")
    parent = git("rev-parse", "HEAD")
    request_file = tmp_path / ".github" / "bagi-m1-once.json"
    request_file.parent.mkdir()
    request = {
        "study": "m1", "force_retrain": True,
        "expected_parent_sha": parent,
        "request_id": "BAGI-M1-2026-11-11",
    }
    request_file.write_text(json.dumps(request) + "\n", encoding="utf-8")
    git("add", ".github/bagi-m1-once.json")
    git("commit", "-qm", "research(m1): request one fresh fit")
    sha = git("rev-parse", "HEAD")
    event_file = tmp_path / "event.json"
    event_file.write_text(json.dumps({
        "ref": ref, "before": parent, "after": sha,
    }), encoding="utf-8")
    env = {
        **os.environ, "GITHUB_EVENT_NAME": "push", "GITHUB_REF": ref,
        "GITHUB_SHA": sha, "GITHUB_EVENT_PATH": str(event_file),
        "BAGI_DISPATCH_STUDY": "",
    }
    def invoke():
        return subprocess.run(
            [sys.executable, str(script)], env=env, cwd=tmp_path,
            capture_output=True, text=True, check=False, timeout=6
        )

    assert invoke().returncode == 0
    request["force_retrain"] = False
    request_file.write_text(json.dumps(request), encoding="utf-8")
    assert invoke().returncode == 2
    request["force_retrain"] = True
    request_file.write_text(json.dumps(request), encoding="utf-8")
    env["GITHUB_SHA"] = "a" * 40
    assert invoke().returncode == 2
    env["GITHUB_SHA"] = sha
    event_file.write_text(json.dumps({
        "ref": ref, "before": "f" * 40, "after": sha,
    }), encoding="utf-8")
    assert invoke().returncode == 2


def test_one_shot_m2_request_requires_exact_source_and_expected_payload(tmp_path):
    import json
    import os
    import subprocess
    import sys

    script = WORKFLOW.parents[2] / "scripts" / "bagi_research_dispatch.py"

    def git(*args):
        return subprocess.check_output(
            ["git", *args], cwd=tmp_path, text=True, timeout=5
        ).strip()

    git("init", "-q")
    git("config", "user.name", "BAGI source regression")
    git("config", "user.email", "noreply@example.invalid")
    (tmp_path / "README.md").write_text("test\n", encoding="utf-8")
    git("add", "README.md")
    git("commit", "-qm", "test parent")
    parent = git("rev-parse", "HEAD")
    one_shot = tmp_path / ".github" / "bagi-m2-once.json"
    one_shot.parent.mkdir()
    payload = {
        "study": "m2",
        "force_retrain": True,
        "expected_parent_sha": parent,
        "request_id": "BAGI-M2-2026-11-11",
    }
    one_shot.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    git("add", ".github/bagi-m2-once.json")
    git("commit", "-qm", "research(m2): request fresh extended fit")
    sha = git("rev-parse", "HEAD")
    event = tmp_path / "event.json"
    event.write_text(json.dumps({
        "ref": "refs/heads/setup/initial-platform",
        "before": parent, "after": sha,
    }), encoding="utf-8")
    env = {
        **os.environ,
        "GITHUB_EVENT_NAME": "push",
        "GITHUB_REF": "refs/heads/setup/initial-platform",
        "GITHUB_SHA": sha, "GITHUB_EVENT_PATH": str(event),
        "GITHUB_OUTPUT": str(tmp_path / "output.txt"),
    }

    def invoke():
        return subprocess.run(
            [sys.executable, str(script)], cwd=tmp_path, env=env,
            capture_output=True, text=True, timeout=6, check=False
        )

    assert invoke().returncode == 0
    assert (tmp_path / "output.txt").read_text() == "study=m2\n"
    payload["request_id"] = "unauthorized-repeat"
    one_shot.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    assert invoke().returncode == 2
    payload["request_id"] = "BAGI-M2-2026-11-11"
    one_shot.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    env["GITHUB_REF"] = "refs/heads/main"
    assert invoke().returncode == 2


def test_one_shot_m3_push_is_bound_to_request_only_new_commit(tmp_path):
    import json
    import os
    import subprocess
    import sys

    script = WORKFLOW.parents[2] / "scripts" / "bagi_research_dispatch.py"

    def git(*args):
        return subprocess.check_output(
            ["git", *args], cwd=tmp_path, text=True, timeout=5
        ).strip()

    git("init", "-q")
    git("config", "user.name", "BAGI test")
    git("config", "user.email", "noreply@example.invalid")
    (tmp_path / "README.md").write_text("replication\n", encoding="utf-8")
    git("add", "README.md")
    git("commit", "-qm", "initial")
    parent = git("rev-parse", "HEAD")
    request = tmp_path / ".github" / "bagi-m3-once.json"
    request.parent.mkdir()
    payload = {
        "study": "m3", "force_retrain": True,
        "expected_parent_sha": parent, "request_id": "BAGI-M3-2026-11-11",
    }
    request.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    git("add", ".github/bagi-m3-once.json")
    git("commit", "-qm", "one-shot independent replication")
    sha = git("rev-parse", "HEAD")
    event = tmp_path / "event.json"
    event.write_text(json.dumps({
        "ref": "refs/heads/setup/initial-platform",
        "before": parent, "after": sha,
    }), encoding="utf-8")
    env = {
        **os.environ, "GITHUB_EVENT_NAME": "push",
        "GITHUB_REF": "refs/heads/setup/initial-platform",
        "GITHUB_SHA": sha, "GITHUB_EVENT_PATH": str(event),
        "GITHUB_OUTPUT": str(tmp_path / "output.txt"),
    }

    def invoke():
        return subprocess.run(
            [sys.executable, str(script)], cwd=tmp_path, env=env,
            capture_output=True, text=True, timeout=6, check=False,
        )

    assert invoke().returncode == 0
    assert (tmp_path / "output.txt").read_text() == "study=m3\n"
    payload["force_retrain"] = False
    request.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    assert invoke().returncode == 2
    payload["force_retrain"] = True
    request.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    env["GITHUB_REF"] = "refs/heads/main"
    assert invoke().returncode == 2


def test_one_shot_m4_push_is_add_only_and_exact_commit_bound(tmp_path):
    import json
    import os
    import subprocess
    import sys

    script = WORKFLOW.parents[2] / "scripts" / "bagi_research_dispatch.py"

    def git(*args):
        return subprocess.check_output(
            ["git", *args], cwd=tmp_path, text=True, timeout=5
        ).strip()

    git("init", "-q")
    git("config", "user.name", "BAGI M4 test")
    git("config", "user.email", "noreply@example.invalid")
    (tmp_path / "README.md").write_text("base\n", encoding="utf-8")
    git("add", "README.md")
    git("commit", "-qm", "base")
    parent = git("rev-parse", "HEAD")
    request_path = tmp_path / ".github" / "bagi-m4-once.json"
    request_path.parent.mkdir()
    payload = {
        "study": "m4", "force_retrain": True,
        "expected_parent_sha": parent, "request_id": "BAGI-M4-2026-11-11",
    }
    request_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    git("add", ".github/bagi-m4-once.json")
    git("commit", "-qm", "M4 frozen evaluation request")
    sha = git("rev-parse", "HEAD")
    event_path = tmp_path / "event.json"
    event_path.write_text(json.dumps({
        "ref": "refs/heads/setup/initial-platform", "before": parent, "after": sha,
    }), encoding="utf-8")
    env = {
        **os.environ, "GITHUB_EVENT_NAME": "push",
        "GITHUB_REF": "refs/heads/setup/initial-platform",
        "GITHUB_SHA": sha, "GITHUB_EVENT_PATH": str(event_path),
        "GITHUB_OUTPUT": str(tmp_path / "output.txt"),
    }

    def invoke():
        return subprocess.run(
            [sys.executable, str(script)], cwd=tmp_path, env=env,
            capture_output=True, text=True, timeout=6, check=False,
        )

    assert invoke().returncode == 0
    assert (tmp_path / "output.txt").read_text() == "study=m4\n"
    payload["force_retrain"] = False
    request_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    assert invoke().returncode == 2
    payload["force_retrain"] = True
    request_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    env["GITHUB_REF"] = "refs/heads/main"
    assert invoke().returncode == 2


def test_one_shot_m5_push_is_bound_to_single_added_file_and_exact_parent(tmp_path):
    import json
    import os
    import subprocess
    import sys

    script = WORKFLOW.parents[2] / "scripts" / "bagi_research_dispatch.py"

    def git(*args):
        return subprocess.check_output(
            ["git", *args], cwd=tmp_path, text=True, timeout=5
        ).strip()

    git("init", "-q")
    git("config", "user.name", "M5 source guard")
    git("config", "user.email", "noreply@example.invalid")
    (tmp_path / "README.md").write_text("parent\n", encoding="utf-8")
    git("add", "README.md")
    git("commit", "-qm", "init")
    parent = git("rev-parse", "HEAD")
    path = tmp_path / ".github" / "bagi-m5-once.json"
    path.parent.mkdir()
    payload = {
        "study": "m5", "force_retrain": True,
        "expected_parent_sha": parent, "request_id": "BAGI-M5-2026-11-11",
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    git("add", ".github/bagi-m5-once.json")
    git("commit", "-qm", "request M5 training")
    head = git("rev-parse", "HEAD")
    event = tmp_path / "event.json"
    event.write_text(json.dumps({
        "ref": "refs/heads/setup/initial-platform",
        "before": parent, "after": head,
    }), encoding="utf-8")
    environ = {
        **os.environ, "GITHUB_EVENT_NAME": "push",
        "GITHUB_REF": "refs/heads/setup/initial-platform",
        "GITHUB_SHA": head, "GITHUB_EVENT_PATH": str(event),
        "GITHUB_OUTPUT": str(tmp_path / "output.txt"),
    }

    def invoke():
        return subprocess.run(
            [sys.executable, str(script)], cwd=tmp_path, env=environ,
            capture_output=True, text=True, timeout=6, check=False,
        )

    assert invoke().returncode == 0
    assert (tmp_path / "output.txt").read_text() == "study=m5\n"
    payload["force_retrain"] = False
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    assert invoke().returncode == 2
    payload["force_retrain"] = True
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    environ["GITHUB_REF"] = "refs/heads/main"
    assert invoke().returncode == 2


def test_one_shot_m6_push_is_add_only_and_exact_source_bound(tmp_path):
    import json
    import os
    import subprocess
    import sys

    script = WORKFLOW.parents[2] / "scripts" / "bagi_research_dispatch.py"

    def git(*args):
        return subprocess.check_output(
            ["git", *args], cwd=tmp_path, text=True, timeout=5
        ).strip()

    git("init", "-q")
    git("config", "user.name", "M6 test")
    git("config", "user.email", "noreply@example.invalid")
    (tmp_path / "README.md").write_text("source\n", encoding="utf-8")
    git("add", "README.md")
    git("commit", "-qm", "init")
    parent = git("rev-parse", "HEAD")
    request = tmp_path / ".github" / "bagi-m6-once.json"
    request.parent.mkdir()
    payload = {
        "study": "m6", "force_retrain": True,
        "expected_parent_sha": parent, "request_id": "BAGI-M6-2026-11-11",
    }
    request.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    git("add", ".github/bagi-m6-once.json")
    git("commit", "-qm", "request M6 native study")
    head = git("rev-parse", "HEAD")
    event_path = tmp_path / "event.json"
    event_path.write_text(json.dumps({
        "ref": "refs/heads/setup/initial-platform",
        "before": parent, "after": head,
    }), encoding="utf-8")
    environ = {
        **os.environ, "GITHUB_EVENT_NAME": "push",
        "GITHUB_REF": "refs/heads/setup/initial-platform",
        "GITHUB_SHA": head, "GITHUB_EVENT_PATH": str(event_path),
        "GITHUB_OUTPUT": str(tmp_path / "output.txt"),
    }

    def invoke():
        return subprocess.run(
            [sys.executable, str(script)], cwd=tmp_path, env=environ,
            capture_output=True, text=True, timeout=6, check=False,
        )

    assert invoke().returncode == 0
    assert (tmp_path / "output.txt").read_text() == "study=m6\n"
    payload["force_retrain"] = False
    request.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    assert invoke().returncode == 2
    payload["force_retrain"] = True
    request.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    environ["GITHUB_REF"] = "refs/heads/main"
    assert invoke().returncode == 2


def test_one_shot_m6_repair_is_new_add_only_request_and_source_bound(tmp_path):
    import json
    import os
    import subprocess
    import sys

    script = WORKFLOW.parents[2] / "scripts" / "bagi_research_dispatch.py"

    def git(*args):
        return subprocess.check_output(
            ["git", *args], cwd=tmp_path, text=True, timeout=5
        ).strip()

    git("init", "-q")
    git("config", "user.name", "M6 repair test")
    git("config", "user.email", "noreply@example.invalid")
    (tmp_path / "README.md").write_text("source\n", encoding="utf-8")
    git("add", "README.md")
    git("commit", "-qm", "initial")
    parent = git("rev-parse", "HEAD")
    request = tmp_path / ".github" / "bagi-m6-repair-once.json"
    request.parent.mkdir()
    payload = {
        "study": "m6", "force_retrain": True,
        "expected_parent_sha": parent,
        "request_id": "BAGI-M6-REPAIR-2026-11-11",
    }
    request.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    git("add", ".github/bagi-m6-repair-once.json")
    git("commit", "-qm", "request M6 corrected runner")
    head = git("rev-parse", "HEAD")
    event_path = tmp_path / "event.json"
    event_path.write_text(json.dumps({
        "ref": "refs/heads/setup/initial-platform",
        "before": parent, "after": head,
    }), encoding="utf-8")
    env = {
        **os.environ, "GITHUB_EVENT_NAME": "push",
        "GITHUB_REF": "refs/heads/setup/initial-platform",
        "GITHUB_SHA": head, "GITHUB_EVENT_PATH": str(event_path),
        "GITHUB_OUTPUT": str(tmp_path / "output.txt"),
    }

    def invoke():
        return subprocess.run(
            [sys.executable, str(script)], cwd=tmp_path, env=env,
            capture_output=True, text=True, timeout=6, check=False,
        )

    assert invoke().returncode == 0
    assert (tmp_path / "output.txt").read_text() == "study=m6\n"
    payload["request_id"] = "BAGI-M6-2026-11-11"
    request.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    assert invoke().returncode == 2
    payload["request_id"] = "BAGI-M6-REPAIR-2026-11-11"
    payload["force_retrain"] = False
    request.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    assert invoke().returncode == 2


def test_m15_repair_request_keeps_same_add_only_dispatch_contract(tmp_path):
    import json
    import os
    import subprocess
    import sys

    script = WORKFLOW.parents[2] / "scripts" / "bagi_research_dispatch.py"

    def git(*args):
        return subprocess.check_output(
            ["git", *args], cwd=tmp_path, text=True, timeout=5
        ).strip()

    git("init", "-q")
    git("config", "user.name", "BAGI source regression")
    git("config", "user.email", "noreply@example.invalid")
    (tmp_path / "README.md").write_text("test\n", encoding="utf-8")
    git("add", "README.md")
    git("commit", "-qm", "test parent")
    parent = git("rev-parse", "HEAD")
    request_path = ".github/bagi-m15-repair-once.json"
    request_file = tmp_path / request_path
    request_file.parent.mkdir()
    request = {
        "study": "m15", "force_retrain": True,
        "expected_parent_sha": parent,
        "request_id": "BAGI-M15-REPAIR-2026-11-12",
    }
    request_file.write_text(json.dumps(request) + "\n", encoding="utf-8")
    git("add", request_path)
    git("commit", "-qm", "test repair request")
    sha = git("rev-parse", "HEAD")
    event = tmp_path / "event.json"
    event.write_text(json.dumps({
        "ref": "refs/heads/setup/initial-platform",
        "before": parent, "after": sha,
    }), encoding="utf-8")
    env = {
        **os.environ, "GITHUB_EVENT_NAME": "push",
        "GITHUB_REF": "refs/heads/setup/initial-platform",
        "GITHUB_SHA": sha, "GITHUB_EVENT_PATH": str(event),
        "GITHUB_OUTPUT": str(tmp_path / "output.txt"),
    }

    def invoke():
        return subprocess.run(
            [sys.executable, str(script)], cwd=tmp_path, env=env,
            capture_output=True, text=True, timeout=6, check=False
        )

    assert invoke().returncode == 0
    assert (tmp_path / "output.txt").read_text() == "study=m15\n"
    request["request_id"] = "unregistered-replay"
    request_file.write_text(json.dumps(request) + "\n", encoding="utf-8")
    assert invoke().returncode == 2
    request["request_id"] = "BAGI-M15-REPAIR-2026-11-12"
    request_file.write_text(json.dumps(request) + "\n", encoding="utf-8")
    env["GITHUB_SHA"] = "f" * 40
    assert invoke().returncode == 2


def test_m16_add_only_request_is_exact_parent_bound_and_replay_resistant(tmp_path):
    import json
    import os
    import subprocess
    import sys

    script = WORKFLOW.parents[2] / "scripts" / "bagi_research_dispatch.py"

    def git(*args):
        return subprocess.check_output(
            ["git", *args], cwd=tmp_path, text=True, timeout=5
        ).strip()

    git("init", "-q")
    git("config", "user.name", "BAGI M16 contract test")
    git("config", "user.email", "noreply@example.invalid")
    (tmp_path / "README.md").write_text("test\n", encoding="utf-8")
    git("add", "README.md")
    git("commit", "-qm", "source parent")
    parent = git("rev-parse", "HEAD")
    request_path = ".github/bagi-m16-once.json"
    request_file = tmp_path / request_path
    request_file.parent.mkdir()
    payload = {
        "study": "m16", "force_retrain": True,
        "expected_parent_sha": parent, "request_id": "BAGI-M16-2026-11-12",
    }
    request_file.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    git("add", request_path)
    git("commit", "-qm", "source-bound M16")
    head = git("rev-parse", "HEAD")
    event = tmp_path / "event.json"
    event.write_text(json.dumps({
        "ref": "refs/heads/setup/initial-platform",
        "before": parent, "after": head,
    }), encoding="utf-8")
    env = {
        **os.environ, "GITHUB_EVENT_NAME": "push",
        "GITHUB_REF": "refs/heads/setup/initial-platform",
        "GITHUB_SHA": head, "GITHUB_EVENT_PATH": str(event),
        "GITHUB_OUTPUT": str(tmp_path / "output.txt"),
    }

    def invoke():
        return subprocess.run(
            [sys.executable, str(script)], cwd=tmp_path, env=env,
            capture_output=True, text=True, check=False, timeout=6
        )

    assert invoke().returncode == 0
    assert (tmp_path / "output.txt").read_text() == "study=m16\n"
    payload["force_retrain"] = False
    request_file.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    assert invoke().returncode == 2
    payload["force_retrain"] = True
    payload["request_id"] = "duplicate-run"
    request_file.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    assert invoke().returncode == 2
    payload["request_id"] = "BAGI-M16-2026-11-12"
    request_file.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    env["GITHUB_SHA"] = "f" * 40
    assert invoke().returncode == 2


def test_m17_exact_parent_bound_one_shot_request(tmp_path):
    import json
    import os
    import subprocess
    import sys

    script = WORKFLOW.parents[2] / "scripts" / "bagi_research_dispatch.py"

    def git(*args):
        return subprocess.check_output(
            ["git", *args], cwd=tmp_path, text=True, timeout=5
        ).strip()

    git("init", "-q")
    git("config", "user.name", "BAGI M17 regression")
    git("config", "user.email", "noreply@example.invalid")
    (tmp_path / "README.md").write_text("test\n", encoding="utf-8")
    git("add", "README.md")
    git("commit", "-qm", "parent")
    parent = git("rev-parse", "HEAD")
    request_path = ".github/bagi-m17-once.json"
    request_file = tmp_path / request_path
    request_file.parent.mkdir()
    metadata = {
        "study": "m17", "force_retrain": True,
        "expected_parent_sha": parent,
        "request_id": "BAGI-M17-2026-11-13",
    }
    request_file.write_text(json.dumps(metadata) + "\n", encoding="utf-8")
    git("add", request_path)
    git("commit", "-qm", "m17 request")
    head = git("rev-parse", "HEAD")
    event = tmp_path / "event.json"
    event.write_text(json.dumps({
        "ref": "refs/heads/setup/initial-platform",
        "before": parent, "after": head,
    }), encoding="utf-8")
    env = {
        **os.environ, "GITHUB_EVENT_NAME": "push",
        "GITHUB_REF": "refs/heads/setup/initial-platform",
        "GITHUB_SHA": head, "GITHUB_EVENT_PATH": str(event),
        "GITHUB_OUTPUT": str(tmp_path / "output.txt"),
    }

    def call():
        return subprocess.run(
            [sys.executable, str(script)], cwd=tmp_path, env=env,
            capture_output=True, text=True, timeout=6, check=False
        )

    assert call().returncode == 0
    assert (tmp_path / "output.txt").read_text() == "study=m17\n"
    metadata["request_id"] = "replay-rejected"
    request_file.write_text(json.dumps(metadata) + "\n", encoding="utf-8")
    assert call().returncode == 2
    metadata["request_id"] = "BAGI-M17-2026-11-13"
    request_file.write_text(json.dumps(metadata) + "\n", encoding="utf-8")
    env["GITHUB_SHA"] = "f" * 40
    assert call().returncode == 2


def test_m18_exact_parent_bound_one_shot_request(tmp_path):
    import json
    import os
    import subprocess
    import sys

    script = WORKFLOW.parents[2] / "scripts" / "bagi_research_dispatch.py"

    def git(*args):
        return subprocess.check_output(
            ["git", *args], cwd=tmp_path, text=True, timeout=5
        ).strip()

    git("init", "-q")
    git("config", "user.name", "BAGI M18 regression")
    git("config", "user.email", "noreply@example.invalid")
    (tmp_path / "README.md").write_text("test\n", encoding="utf-8")
    git("add", "README.md")
    git("commit", "-qm", "parent")
    parent = git("rev-parse", "HEAD")
    request_path = ".github/bagi-m18-once.json"
    request_file = tmp_path / request_path
    request_file.parent.mkdir()
    metadata = {
        "study": "m18", "force_retrain": True,
        "expected_parent_sha": parent,
        "request_id": "BAGI-M18-2026-11-13",
    }
    request_file.write_text(json.dumps(metadata) + "\n", encoding="utf-8")
    git("add", request_path)
    git("commit", "-qm", "m18 request")
    head = git("rev-parse", "HEAD")
    event = tmp_path / "event.json"
    event.write_text(json.dumps({
        "ref": "refs/heads/setup/initial-platform",
        "before": parent, "after": head,
    }), encoding="utf-8")
    env = {
        **os.environ, "GITHUB_EVENT_NAME": "push",
        "GITHUB_REF": "refs/heads/setup/initial-platform",
        "GITHUB_SHA": head, "GITHUB_EVENT_PATH": str(event),
        "GITHUB_OUTPUT": str(tmp_path / "output.txt"),
    }

    def call():
        return subprocess.run(
            [sys.executable, str(script)], cwd=tmp_path, env=env,
            capture_output=True, text=True, timeout=6, check=False
        )

    assert call().returncode == 0
    assert (tmp_path / "output.txt").read_text() == "study=m18\n"
    metadata["request_id"] = "replay-rejected"
    request_file.write_text(json.dumps(metadata) + "\n", encoding="utf-8")
    assert call().returncode == 2
    metadata["request_id"] = "BAGI-M18-2026-11-13"
    request_file.write_text(json.dumps(metadata) + "\n", encoding="utf-8")
    env["GITHUB_SHA"] = "f" * 40
    assert call().returncode == 2

