from bagi.worlds import WorkshopEnv


def main():
    env = WorkshopEnv()
    observation, info = env.reset(seed=42)
    print("BAGI Workshop v0")
    print(env.render())
    print("\nLocal observation:\n", observation)

    for _ in range(5):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        print(f"\naction={action} reward={reward} position={info['position'].tolist()}")
        print(env.render())
        if terminated or truncated:
            break


if __name__ == "__main__":
    main()
