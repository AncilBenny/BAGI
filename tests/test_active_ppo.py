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
    from bagi.experiments.gate_a_shared_interface import SharedPPO, capture_rng, restore_rng

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
        before = policy.action_net.weight[:7].detach().clone()
        bias = policy.action_net.bias[:7].detach().clone()
        train_active_block(c, BLOCK)
        assert policy.action_dist is original and policy.optimizer is optimizer
        torch.testing.assert_close(policy.action_net.weight[:7], before, rtol=0, atol=0)
        torch.testing.assert_close(policy.action_net.bias[:7], bias, rtol=0, atol=0)
        assert c.training_actions == 2 * BLOCK
        assert c.updates == 32 and c.presentations == 8 * BLOCK
        assert sum(r["actions"] for r in c.env.rows) == BLOCK
    finally:
        c.model_env.close()
