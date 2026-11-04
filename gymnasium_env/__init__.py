from gymnasium.envs.registration import register

register(
    id="gymnasium_env/Traffic-v0",
    entry_point="gymnasium_env.envs.traffic_world:TrafficIntersectionEnv",
)
