import pytest

from bagi.experiments.multi_seed_sample_efficiency import summarize_seed_results


def test_summarize_seed_results_reports_mean_std_and_fixed_real_steps():
    seed_results = [
        {
            "conditions": [
                {
                    "guided_episodes": 0,
                    "training_real_steps": 100,
                    "planner_success_rate": 0.4,
                    "planner_evaluation_real_steps": 50,
                    "planner_imagined_steps": 1000,
                    "validation_cell_accuracy": 0.8,
                    "validation_reward_mae": 0.1,
                    "validation_event_f1": 0.5,
                }
            ]
        },
        {
            "conditions": [
                {
                    "guided_episodes": 0,
                    "training_real_steps": 100,
                    "planner_success_rate": 0.8,
                    "planner_evaluation_real_steps": 40,
                    "planner_imagined_steps": 1200,
                    "validation_cell_accuracy": 0.9,
                    "validation_reward_mae": 0.05,
                    "validation_event_f1": 0.7,
                }
            ]
        },
    ]

    summary = summarize_seed_results(seed_results)

    assert summary[0]["training_real_steps"] == 100
    assert summary[0]["seed_count"] == 2
    assert summary[0]["planner_success_rate_mean"] == pytest.approx(0.6)
    assert summary[0]["planner_success_rate_std"] == pytest.approx(0.2828427125)


def test_summarize_seed_results_rejects_mismatched_real_experience():
    seed_results = [
        {
            "conditions": [
                {
                    "guided_episodes": 5,
                    "training_real_steps": 100,
                    "planner_success_rate": 0.5,
                    "planner_evaluation_real_steps": 50,
                    "planner_imagined_steps": 1000,
                    "validation_cell_accuracy": 0.8,
                    "validation_reward_mae": 0.1,
                    "validation_event_f1": 0.5,
                }
            ]
        },
        {
            "conditions": [
                {
                    "guided_episodes": 5,
                    "training_real_steps": 101,
                    "planner_success_rate": 0.5,
                    "planner_evaluation_real_steps": 50,
                    "planner_imagined_steps": 1000,
                    "validation_cell_accuracy": 0.8,
                    "validation_reward_mae": 0.1,
                    "validation_event_f1": 0.5,
                }
            ]
        },
    ]

    with pytest.raises(ValueError, match="Training real-step count changed"):
        summarize_seed_results(seed_results)
