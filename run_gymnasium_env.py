import gymnasium as gym
import gymnasium_env  # This will register "gymnasium_env/Traffic-v0"

def main():
    # Create the environment using the registered ID
    env = gym.make("gymnasium_env/Traffic-v0")

    # Fixed seed = EV always spawns at same time (deterministic)
    # Can also implement stochastic process, EV probability scales with dt
    obs, _ = env.reset(seed=None) 
    print("Initial observation:", obs)

    for t in range(30):
        # Sample a random action from the action space
        action = env.action_space.sample()
        
        # Take a step in the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the environment (ASCII)
        env.render()
        
        # Print step info
        print(f"Step {t}: Action={action}, Reward={reward:.2f}, Info={info}")
        
        if terminated or truncated:
            print("Episode finished")
            break

    env.close()

if __name__ == "__main__":
    main()


