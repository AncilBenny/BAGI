"""Native integration contracts; no scientific success claim."""
import pytest
import torch


def test_native_active_objective():
    pytest.importorskip("sb3_contrib")
    from stable_baselines3.common.distributions import MultiCategoricalDistribution

    from bagi.agents.active_ppo import ActiveFactorDistribution, train_active_block
    from bagi.agents.plasticity import reset_adam_history
    from bagi.experiments.cumulative_transfer import TASKS
    from bagi.experiments.gate_a_competence import BLOCK
    from bagi.experiments.gate_a_shared_interface import (
        SharedPPO,
        capture_rng,
        check_boundary,
        restore_rng,
    )

    torch.set_num_threads(1)
    c = SharedPPO(TASKS[0], 211, 99501)
    try:
        c.train(BLOCK)
        c.env.rows = []
        assert reset_adam_history(c.model.policy)
        model, policy = c.model, c.model.policy
        original, optimizer = policy.action_dist, policy.optimizer
        assert type(original) is MultiCategoricalDistribution
        # Both collection and evaluation compute the same marginal likelihood.
        obs = torch.as_tensor(model._last_obs)
        starts = torch.ones(1)
        states = model._last_lstm_states
        rng = capture_rng()
        with torch.no_grad():
            expected, _, _, _ = policy(obs, states, starts)
            restore_rng(rng)
            policy.action_dist = ActiveFactorDistribution(0)
            actions, _, old_log_prob, _ = policy(obs, states, starts)
            _, new_log_prob, entropy = policy.evaluate_actions(obs, actions, states, starts)
            torch.testing.assert_close(actions, expected, rtol=0, atol=0)
            torch.testing.assert_close(old_log_prob, new_log_prob)
            torch.testing.assert_close(entropy, policy.action_dist.distribution[0].entropy())
        policy.action_dist = original
        restore_rng(rng)
        c.switch_domain(TASKS[1], 99601)
        assert model._last_obs is None and c.env.episodes == 0
        with pytest.raises(ValueError, match="completed rollout"):
            check_boundary(c)
        c.env.rows.append({})
        with pytest.raises(ValueError, match="untouched post-switch"):
            train_active_block(c, BLOCK)
        c.env.rows.clear()
        assert c.training_actions == BLOCK and c.env.episodes == 0
        before = policy.action_net.weight[:7].detach().clone()
        bias = policy.action_net.bias[:7].detach().clone()
        train_active_block(c, BLOCK)
        assert policy.action_dist is original and policy.optimizer is optimizer
        torch.testing.assert_close(policy.action_net.weight[:7], before, rtol=0, atol=0)
        torch.testing.assert_close(policy.action_net.bias[:7], bias, rtol=0, atol=0)
        assert c.training_actions == 2 * BLOCK
        assert c.updates == 32 and c.presentations == 8 * BLOCK
        assert sum(r["actions"] for r in c.env.rows) == BLOCK
        assert c.env.rows[0]["seed"] == 99601
        c.env.rows.clear()
        check_boundary(c)
    finally:
        c.model_env.close()


@pytest.mark.parametrize("index", [0, 1])
def test_buffer_likelihood_and_active_resume(tmp_path, index):
    pytest.importorskip("sb3_contrib")
    from bagi.agents.active_ppo import ActiveFactorDistribution, train_active_block
    from bagi.experiments.cumulative_transfer import TASKS
    from bagi.experiments.gate_a_competence import BLOCK
    from bagi.experiments.gate_a_sequential_pilot import same_tree
    from bagi.experiments.gate_a_shared_interface import SharedPPO, load_boundary, save_boundary

    torch.set_num_threads(1)
    c = SharedPPO(TASKS[1 - index], 223, 99701)
    first = second = None
    try:
        c.train(BLOCK)
        c.env.rows.clear()
        c.switch_domain(TASKS[index], 99801)
        policy = c.model.policy
        buffer = c.model.rollout_buffer
        original_get = buffer.get
        checked = []

        def checked_get(*args, **kwargs):
            for batch in original_get(*args, **kwargs):
                if not checked:
                    assert isinstance(policy.action_dist, ActiveFactorDistribution)
                    assert policy.action_dist.active == index
                    with torch.no_grad():
                        _, lp, _ = policy.evaluate_actions(
                            batch.observations, batch.actions, batch.lstm_states,
                            batch.episode_starts,
                        )
                    mask = batch.mask > 1e-8
                    assert mask.any()
                    torch.testing.assert_close(lp[mask], batch.old_log_prob[mask],
                                               rtol=1e-5, atol=1e-5)
                    checked.append(True)
                yield batch

        buffer.get = checked_get
        try:
            train_active_block(c, BLOCK)
        finally:
            buffer.get = original_get
        assert checked == [True]
        c.env.rows.clear()
        path = tmp_path / "active.pt"
        digest = save_boundary(c, path)
        first = load_boundary(path, digest)
        train_active_block(first, BLOCK)
        second = load_boundary(path, digest)
        train_active_block(second, BLOCK)
        assert first.env.rows == second.env.rows
        assert same_tree(first.model.policy.state_dict(), second.model.policy.state_dict())
        assert same_tree(first.model.policy.optimizer.state_dict(),
                         second.model.policy.optimizer.state_dict())
        assert first.training_actions == second.training_actions == 3 * BLOCK
        assert first.updates == second.updates == 48
        assert first.presentations == second.presentations == 12 * BLOCK
    finally:
        c.model_env.close()
        for branch in (first, second):
            if branch is not None:
                branch.model_env.close()
