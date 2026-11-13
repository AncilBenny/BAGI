from bagi.worlds import WorkshopEnv


def test_workshop_reset_and_step():
    env = WorkshopEnv(size=9)
    obs, info = env.reset(seed=1)
    assert obs.shape == (5, 5)
    assert info["position"].tolist() == [1, 1]

    obs, reward, terminated, truncated, info = env.step(1)
    assert obs.shape == (5, 5)
    assert info["position"].tolist() == [1, 2]
    assert reward == 0.0
    assert not terminated
    assert not truncated
