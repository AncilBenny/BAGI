"""Runtime contracts; these tests contain no native training or transfer proof."""
import json

import pytest

from bagi.agents.general_brain import GeneralBrain, Goal, PublicFactSensor, Skill
from bagi.runtime import AgentRuntime, StepFeedback


def make_runtime(identity="mock-v1"):
    skills = (
        Skill("a_bad", frozenset(), frozenset({"checked"}), lambda o, f: 0, 1),
        Skill("b_good", frozenset(), frozenset({"checked"}), lambda o, f: 1, 1),
    )
    brain = GeneralBrain(skills, PublicFactSensor(), max_attempts_per_skill=1)
    return AgentRuntime(brain, identity)


def step(action):
    facts = frozenset({"checked"}) if action == 1 else frozenset()
    return StepFeedback(action, facts)


def goal():
    return Goal("check", frozenset({"checked"}))


def test_resume_same_outcome_learning_across_goals(tmp_path):
    first = make_runtime()
    initial = first.run_episode(goal(), frozenset(), step)
    assert initial.actions == (0, 1) and initial.goal_verified
    checkpoint = tmp_path / "state.json"
    first.save(checkpoint)
    resumed = make_runtime()
    resumed.load(checkpoint)
    assert resumed.snapshot() == first.snapshot()
    assert resumed.run_episode(goal(), frozenset(), step).actions == (1,)
    assert first.run_episode(goal(), frozenset(), step).actions == (1,)
    assert resumed.snapshot() == first.snapshot()
    assert make_runtime().run_episode(goal(), frozenset(), step).actions == (0, 1)


def test_frozen_evaluation_does_not_update_outcome_counts():
    runtime = make_runtime()
    before = runtime.snapshot()
    result = runtime.run_episode(goal(), frozenset(), step, learn=False)
    assert result.goal_verified and result.actions == (0, 1)
    assert runtime.snapshot() == before


def test_termination_and_actual_action_mismatch():
    runtime = make_runtime()
    result = runtime.run_episode(
        goal(), frozenset(), lambda a: StepFeedback(a, frozenset(), 1.0, terminated=True)
    )
    assert not result.goal_verified and result.reason == "terminated"
    assert result.actions == (0,)
    with pytest.raises(ValueError, match="provenance"):
        make_runtime().run_episode(goal(), frozenset(), lambda a: StepFeedback(1, frozenset()))


def test_checkpoint_rejects_incompatible_or_invalid_state_atomically(tmp_path):
    runtime = make_runtime()
    path = tmp_path / "state.json"
    runtime.save(path)
    with pytest.raises(ValueError, match="Incompatible"):
        make_runtime("other").load(path)
    before = runtime.snapshot()
    data = runtime.snapshot()
    data["experience"]["a_bad"]["successes"] = 4
    data["experience"]["b_good"]["failures"] = True
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid"):
        runtime.load(path)
    assert runtime.snapshot() == before
