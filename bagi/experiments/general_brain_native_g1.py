"""G1: the SAME public-evidence General Brain on native Workshop and XLand.

Workshop uses transparent scripted adapter options solely to validate the
control/effect boundary. XLand uses SAME archived, upstream trained imitation.BC
policies under direct and capped fallback conditions. No evaluator hidden
state, pocket or full grid enters the acting brain. Never treat a native
putdown reward alone as physical placement.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bagi.agents.general_brain import GeneralBrain, Goal, NoPlanError, Skill
from bagi.agents.public_evidence import PublicEvidenceSensor

WORKSHOP_MODES = ("default", "mirror_x")
WORKSHOP_CONDITIONS = ("direct_script", "assumed_effect", "verified_replan")
XLAND_LAYOUTS = tuple(range(11300, 11305))
XLAND_CONDITIONS = ("direct_expanded_bc", "verified_option_fallback")
MODEL_SEEDS = (17, 29)
TASKS = ("hold", "near", "putdown")
XLAND_CAP = 40
OPTION_SPLIT = 20


def verify_registration() -> None:
    """Frozen finite budget and source-disjoint independent test task family."""
    old = set(range(10400, 10424)) | set(range(10600, 10610))
    old |= set(range(10700, 10710)) | set(range(10800, 10810))
    old |= set(range(10900, 10972)) | set(range(11000, 11020))
    old |= set(range(11100, 11120)) | set(range(11200, 11220))
    if (XLAND_LAYOUTS != tuple(range(11300, 11305))
            or set(XLAND_LAYOUTS) & old or MODEL_SEEDS != (17, 29)
            or TASKS != ("hold", "near", "putdown")
            or WORKSHOP_MODES != ("default", "mirror_x")
            or OPTION_SPLIT * 2 != XLAND_CAP or XLAND_CAP != 40):
        raise ValueError("G1 source, task or finite native action budget changed")


def workshop_public_direction(observation: np.ndarray) -> tuple[int, int]:
    """Orient an adapter script using a PUBLIC switch tile, not layout_mode."""
    switch_token = 4  # Workshop SWITCH public cell encoding; adapter-only.

    view = np.asarray(observation)
    if view.shape != (5, 5):
        raise ValueError("Unexpected Workshop public-view size")
    if view[2, 4] == switch_token:
        return (1, 3)  # toward, away
    if view[2, 0] == switch_token:
        return (3, 1)
    raise ValueError("Workshop first switch is not publicly observable")


def workshop_scripts(observation: np.ndarray) -> dict[str, tuple[int, ...]]:
    """Fixed adapter scripts are CONTROL FIXTURES, not learned navigation."""
    toward, away = workshop_public_direction(observation)
    return {
        "a_bad_switch": (away, away),
        "b_good_switch": (toward, toward),
        "c_reach_goal": (away, away, 2, 2, 2, 2,
                         toward, toward, toward, toward, toward, toward, 2, 2),
    }


class FixedScript:
    def __init__(self, actions: tuple[int, ...]):
        self.actions = actions
        self.position = 0

    def __call__(self, observation, facts: frozenset[str]) -> int:
        if self.position >= len(self.actions):
            raise RuntimeError("Script exhausted before skill outcome")
        action = self.actions[self.position]
        self.position += 1
        return action


def workshop_sensor() -> PublicEvidenceSensor:
    """DOOR_OPEN is visually visible; reward is public; no hidden info."""
    from bagi.worlds.workshop import DOOR_CLOSED, DOOR_OPEN

    def door(observation):
        view = np.asarray(observation)
        if np.any(view == DOOR_OPEN):
            return True
        if np.any(view == DOOR_CLOSED):
            return False
        return None

    return PublicEvidenceSensor({"door_open": door}, reward_fact="native_reward_observed")


def workshop_brain(observation: np.ndarray, mode: str) -> GeneralBrain:
    scripts = workshop_scripts(observation)
    skills = (
        Skill("a_bad_switch", frozenset(), frozenset({"door_open"}),
              FixedScript(scripts["a_bad_switch"]), max_actions=2),
        Skill("b_good_switch", frozenset(), frozenset({"door_open"}),
              FixedScript(scripts["b_good_switch"]), max_actions=2),
        Skill("c_reach_goal", frozenset({"door_open"}),
              frozenset({"native_reward_observed"}),
              FixedScript(scripts["c_reach_goal"]), max_actions=14),
    )
    if mode == "direct_script":
        # Compare against a competent, handcrafted, non-learning adapter.
        skills = (
            Skill("direct", frozenset(), frozenset({"native_reward_observed"}),
                  FixedScript(scripts["b_good_switch"] + scripts["c_reach_goal"]),
                  max_actions=16),
        )
    elif mode != "verified_replan":
        raise ValueError("Only direct and evidence-verified brains are supported")
    return GeneralBrain(skills, workshop_sensor(), max_episode_actions=40,
                        max_attempts_per_skill=1)


def workshop_native_one(mode: str, condition: str) -> dict:
    """Real native Workshop environment with only raw local public views."""
    from bagi.worlds import WorkshopEnv
    from bagi.worlds.workshop import SWITCH

    if SWITCH != 4:
        raise ValueError("Workshop public switch encoding changed")
    verify_registration()
    world = WorkshopEnv(size=9, view_size=5, max_steps=40,
                        start_mode="fixed", layout_mode=mode)
    observation, _hidden_info = world.reset(seed=11400)
    scripts = workshop_scripts(observation)
    moves: list[int] = []
    detected_open = False
    finished = False
    reward_seen = False
    evidence = workshop_sensor()
    if condition == "assumed_effect":
        # Explicit UNSAFE control: assume the failed option opened the door,
        # then execute the remainder without public outcome confirmation.
        for action in scripts["a_bad_switch"] + scripts["c_reach_goal"]:
            observation, reward, finished, truncated, _hidden_info = world.step(action)
            moves.append(action)
            reward_seen |= reward > 0
            detected_open |= "door_open" in evidence.initial(observation)
            if finished or truncated:
                break
        verified = bool(reward_seen)
        option_failures = 0
    else:
        brain = workshop_brain(observation, condition)
        brain.reset(Goal("workshop_goal", frozenset({"native_reward_observed"})),
                    observation)
        while not brain.complete and not brain.exhausted and not finished:
            try:
                decision = brain.decide()
            except NoPlanError:
                break
            action = int(decision.action)
            after, reward, finished, truncated, _hidden_info = world.step(action)
            brain.observe(action, after, float(reward))
            moves.append(action)
            observation = after
            reward_seen |= reward > 0
            detected_open |= "door_open" in brain.facts
            if truncated:
                break
        verified = brain.complete
        option_failures = sum(x.failures for x in brain.experience.values())
    # The evaluator-only native flags are read AFTER the episode. Never pass
    # hidden info, position, door_open or has_key into the acting controller.
    actual_hidden = {"door_open": bool(world._door_open),  # noqa: SLF001
                     "has_key": bool(world._has_key)}  # noqa: SLF001
    return {
        "layout_mode": mode, "condition": condition,
        "native_actions": len(moves), "goal_rewarded": reward_seen,
        "public_goal_verified": bool(verified),
        "public_door_evidence_seen": bool(detected_open),
        "failed_options": option_failures,
        "evaluator_only_causal_chain": actual_hidden,
        "actions": moves,
    }


def workshop_trials() -> list[dict]:
    return [workshop_native_one(layout, condition)
            for layout in WORKSHOP_MODES for condition in WORKSHOP_CONDITIONS]


def xland_reward_sensor() -> PublicEvidenceSensor:
    """Public reward verifies game outcome ONLY, not physical manipulation."""
    return PublicEvidenceSensor({}, reward_fact="native_reward_observed")


def xland_policy_skill(policy, name: str, cap: int) -> Skill:
    def act(public_observation, facts: frozenset[str]):
        from bagi.experiments.xland_gate_b_exact_collection import native_action
        chosen, _state = policy.predict(
            np.asarray(public_observation, dtype=np.float32), deterministic=True,
        )
        return native_action(np.asarray(chosen))

    return Skill(name, frozenset(), frozenset({"native_reward_observed"}),
                 act, max_actions=cap)


def xland_brain(expanded_policy, frozen_policy, condition: str) -> GeneralBrain:
    if condition == "direct_expanded_bc":
        skills = (xland_policy_skill(expanded_policy, "direct_expanded", XLAND_CAP),)
    elif condition == "verified_option_fallback":
        skills = (
            xland_policy_skill(expanded_policy, "a_expanded", OPTION_SPLIT),
            xland_policy_skill(frozen_policy, "b_original_fallback", OPTION_SPLIT),
        )
    else:
        raise ValueError("Unknown G1 XLand control condition")
    return GeneralBrain(skills, xland_reward_sensor(),
                        max_episode_actions=XLAND_CAP, max_attempts_per_skill=1)


def xland_episode(brain: GeneralBrain, timestep, *, native_step, observe,
                  eligibility: bool, physical, adjacent) -> dict:
    """Published BC actions, 115 public floats, evaluator-only physical metrics."""
    from bagi.experiments.xland_complete_demo_bc import actual_history
    from bagi.experiments.xland_gate_b_exact_collection import INPUT_WIDTH, native_action

    if not eligibility:
        return {"eligible": False, "native_actions": 0, "native_reward": False,
                "physical_placement": False, "effective_blue_pickups": 0,
                "effective_blue_drops": 0, "public_reward_claim": False}
    history = (6, 6, 6, 6)
    observation = np.asarray(observe(timestep, history), dtype=np.float32)
    if observation.shape != (INPUT_WIDTH,) or not np.isfinite(observation).all():
        raise ValueError("Invalid public observation at XLand reset")
    brain.reset(Goal("native_task", frozenset({"native_reward_observed"})), observation)
    seen_reward = seen_pickup = ordered_drop = False
    pickups = drops = 0
    chosen_options: list[str] = []
    while not bool(timestep.last()) and not brain.exhausted and not brain.complete:
        try:
            decision = brain.decide()
        except NoPlanError:
            break
        taken = native_action(np.asarray(decision.action))
        if not chosen_options or chosen_options[-1] != decision.skill_id:
            chosen_options.append(decision.skill_id)
        before = timestep
        after = native_step(before, taken)
        picked, dropped = physical(before, after, taken)
        pickups += int(picked)
        drops += int(dropped)
        ordered_drop |= bool(dropped and seen_pickup)
        seen_pickup |= bool(picked)
        seen_reward |= float(after.reward) > 0
        history = actual_history(history, taken)
        public_after = np.asarray(observe(after, history), dtype=np.float32)
        if public_after.shape != (INPUT_WIDTH,) or not np.isfinite(public_after).all():
            raise ValueError("Invalid public XLand transition features")
        brain.observe(taken, public_after, float(after.reward))
        timestep = after
    verified = bool(seen_reward and pickups > 0 and drops > 0
                    and ordered_drop and adjacent(timestep))
    return {"eligible": True, "native_actions": brain.actual_actions,
            "native_reward": bool(seen_reward),
            "public_reward_claim": brain.complete,
            "physical_placement": verified,
            "effective_blue_pickups": pickups, "effective_blue_drops": drops,
            "ordered_blue_drop": bool(ordered_drop),
            "option_sequence": chosen_options,
            "cumulative_option_outcomes": {
                k: {"successes": v.successes, "failures": v.failures}
                for k, v in brain.experience.items()
            }}


def source_policies(source_dir: Path, old_dir: Path, new_dir: Path):
    """Reconstruct SHA-verified, previous-recipe upstream imitation.BC weights."""
    from bagi.experiments import xland_diverse_physical_demos as diverse
    from bagi.experiments.xland_complete_demo_confirmation import load_frozen_teacher
    from bagi.experiments.xland_diversity_confirmation import (
        load_frozen_diversity,
        require_previous_training,
    )
    from bagi.experiments.xland_existing_bc import TRAIN_TASKS, get_rules
    from bagi.experiments.xland_gate_b_native_collection import source_record_checks
    from bagi.experiments.xland_instruction_action_history import assemble, check_true_history
    from bagi.experiments.xland_upstream_instruction_bc import (
        load_verified_source, task_instructions,
    )

    old_meta, original = load_frozen_teacher(old_dir)
    prior, diverse_teacher = load_frozen_diversity(new_dir)
    if old_meta["status"] != "complete":
        raise ValueError("Old verified physical TRAIN source not complete")
    expanded = {key: np.concatenate((original[key], diverse_teacher[key]))
                for key in ("obs", "expert_label", "done")}
    summary = json.loads((source_dir / "summary.json").read_text(encoding="utf-8"))
    source_record_checks(summary)
    descriptions = {
        task: task_instructions(get_rules(task).goal, get_rules(task).rules, enabled=True)
        for task in TRAIN_TASKS
    }
    policies = {}
    trainings = {}
    for seed in MODEL_SEEDS:
        model = [m for m in summary["model_results"] if m["model_seed"] == seed]
        if len(model) != 1:
            raise ValueError("Original source policy seed missing or duplicated")
        data, tasks, checksum = load_verified_source(
            source_dir / f"action_provenance_seed{seed}.npz", model[0], seed
        )
        check_true_history(data)
        base = {
            "obs": assemble(
                data["obs"], np.stack([descriptions[task] for task in tasks]),
                data["prior_executed_actions"], "real_executed_action_history"
            ),
            "expert_label": data["expert_label"], "done": data["done"],
        }
        policies[seed] = {}
        trainings[seed] = {}
        for arm, teacher in zip(diverse.ARMS, (original, expanded)):
            policy, train = diverse.fit_original_bc(base, teacher, seed)
            require_previous_training(prior, seed, arm, train, checksum)
            policies[seed][arm] = policy
            trainings[seed][arm] = train
    return policies, trainings


def native_xland_trials(source_dir: Path, old_dir: Path, new_dir: Path) -> dict:
    """Genuine XLand fresh episodes; evaluator alone sees grid/pocket."""
    import jax
    import xminigrid
    from xminigrid.core.constants import TILES_REGISTRY, Colors, Tiles
    from xminigrid.wrappers import (
        DirectionObservationWrapper,
        RulesAndGoalsObservationWrapper,
    )

    from bagi.experiments import xland_diverse_physical_demos as diverse
    from bagi.experiments.xland_existing_bc import get_rules
    from bagi.experiments.xland_gate_b_exact_collection import public_features
    from bagi.experiments.xland_gate_b_native_collection import screen_source_layout
    from bagi.experiments.xland_putdown_reward_validity import (
        blue_square_transition,
        tiles_adjacent,
    )

    verify_registration()
    models, training = source_policies(source_dir, old_dir, new_dir)
    room, params0 = xminigrid.make("XLand-MiniGrid-R1-9x9")
    visible = RulesAndGoalsObservationWrapper(DirectionObservationWrapper(room))
    reset, step = jax.jit(visible.reset), jax.jit(visible.step)
    blue = np.asarray(TILES_REGISTRY[Tiles.SQUARE, Colors.BLUE])
    red = np.asarray(TILES_REGISTRY[Tiles.BALL, Colors.RED])
    empty = int(Tiles.EMPTY)
    results = []
    total = 0
    for seed in MODEL_SEEDS:
        expanded_policy = models[seed][diverse.ARMS[1]]
        frozen_policy = models[seed][diverse.ARMS[0]]
        for condition in XLAND_CONDITIONS:
            # The same goal controller retains public-reward option reliability
            # across fresh episodes and goals, while reset clears episode facts.
            brain = xland_brain(expanded_policy, frozen_policy, condition)
            by_task = {}
            for task in TASKS:
                params = params0.replace(ruleset=get_rules(task), max_steps=XLAND_CAP)
                cases = []
                for layout in XLAND_LAYOUTS:
                    ts = reset(params, jax.random.key(layout))
                    eligible = (task != "putdown"
                                or screen_source_layout(ts, blue, red))

                    def native_step(before, taken, *, params=params):
                        return step(params, before, taken)

                    def physical(before, after, taken, *, task=task):
                        if task != "putdown":
                            return False, False
                        return blue_square_transition(
                            taken, np.asarray(before.state.agent.pocket),
                            np.asarray(after.state.agent.pocket),
                            np.asarray(before.state.grid),
                            np.asarray(after.state.grid), blue, empty,
                        )

                    def adjacent(ts):
                        return tiles_adjacent(np.asarray(ts.state.grid), blue, red)

                    result = xland_episode(
                        brain, ts, native_step=native_step,
                        observe=public_features, eligibility=eligible,
                        physical=physical, adjacent=adjacent,
                    )
                    total += result["native_actions"]
                    cases.append({"layout": layout, **result})
                by_task[task] = {
                    "eligible": sum(int(x["eligible"]) for x in cases),
                    "native_rewarded": sum(int(x["native_reward"]) for x in cases),
                    "physical_placements": sum(int(x["physical_placement"]) for x in cases),
                    "effective_blue_pickups": sum(x["effective_blue_pickups"] for x in cases),
                    "effective_blue_drops": sum(x["effective_blue_drops"] for x in cases),
                    "native_actions": sum(x["native_actions"] for x in cases),
                    "cases": cases,
                }
            results.append({"model_seed": seed, "condition": condition,
                            "training": training[seed], "tasks": by_task})
    return {"native_actions": total, "new_teacher_queries": 0,
            "new_train_simulator_actions": 0,
            "predeclared_layouts": list(XLAND_LAYOUTS), "results": results}


def run(source_dir: Path, old_dir: Path, new_dir: Path,
        output_dir: Path) -> dict:
    verify_registration()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "summary.json"
    report = {
        "experiment": "G1 GeneralBrain native Workshop and published-XLand BC adapters",
        "status": "running",
        "public_input_only": True,
        "workshop_unique_geometry_layouts": list(WORKSHOP_MODES),
        "xland_new_eval_layouts": list(XLAND_LAYOUTS),
        "xland_models": list(MODEL_SEEDS), "xland_conditions": list(XLAND_CONDITIONS),
        "new_teacher_queries": 0, "new_training_native_actions": 0,
        "new_workshop_native_actions": 0, "new_xland_native_actions": 0,
        "workshop": [], "xland": None,
    }
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    workshop = workshop_trials()
    report["workshop"] = workshop
    report["new_workshop_native_actions"] = sum(x["native_actions"] for x in workshop)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    xland = native_xland_trials(source_dir, old_dir, new_dir)
    report["xland"] = xland
    report["new_xland_native_actions"] = xland["native_actions"]
    report["status"] = "complete"
    report["limitations"] = [
        "Same GeneralBrain core reused; Workshop options are scripts, NOT learned skill discovery.",
        "Workshop uses just 2 fixed geometries, not statistical layout generalization.",
        "XLand options reuse exact frozen upstream BC labels and training recipe.",
        "BC weights retrained only from archived numeric TRAIN data, no new teacher queries.",
        "Two XLand option policies switch at action20 after no PUBLIC reward evidence.",
        "The controller's public reward fact is not a physical pickup/drop verifier.",
        "Only the EXTERNAL evaluator sees XLand pocket/full grid for BLUE checks.",
        "Matched XLand 40-action cap, but direct/fallback differ in policy mix and costs.",
        "A positive result cannot show generality, learned causal facts or option discovery.",
    ]
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--old-teacher-dir", type=Path, required=True)
    parser.add_argument("--diverse-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/g1_real_worlds"))
    args = parser.parse_args()
    run(args.source_dir, args.old_teacher_dir, args.diverse_dir, args.output_dir)
