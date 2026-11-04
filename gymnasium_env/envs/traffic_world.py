# traffic_world.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Any

class TrafficIntersectionEnv(gym.Env):
    """
    Gymnasium environment for a 4-way intersection with turning (left) and straight movements.
    Action space (12 discrete actions):
        0-2   : NS Left -> [Green, Yellow, Red]
        3-5   : NS Straight -> [Green, Yellow, Red]
        6-8   : EW Left -> [Green, Yellow, Red]
        9-11  : EW Straight -> [Green, Yellow, Red]

    Observations: Dict
        - lights: array of length 8 (order: Left_N, Right_N, Left_S, Right_S, Left_E, Right_E, Left_W, Right_W)
                    each element in {0=Red, 1=Yellow, 2=Green}
        - queues: array of length 4 (N, S, E, W) integers (vehicles waiting)
        - emergency: array of length 4 flags (emergency waiting N,S,E,W)
        - last_switch_times: array length 4 (last time any ns-left / ns-straight / ew-left / ew-straight switched) [for internal use]
        - time: scalar (elapsed sim time)
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 1}

    def __init__(
        self,
        max_queue: int = 50,
        dt: float = 1.0,
        max_time: float = 3600.0,
        arrival_rates: Dict[str, float] = None,  # vehicles per second for N,S,E,W
        flow_rate: int = 2,  # max vehicles that can pass per second when corresponding light is green
        throughput_threshold: int = 5,
        throughput_interval: float = 10.0,
        switch_protect_time: float = 10.0,
        emergency_penalty_interval: float = 1.0,
        emergency_reward_when_served: float = 5.0,
        switch_penalty: float = -3.0,
        queue_wait_penalty_coef: float = -0.01,
        seed: int = None,
    ):
        super().__init__()

        self.rng = np.random.default_rng(seed)

        # Simulation params
        self.dt = dt
        self.max_time = max_time
        self.time = 0.0

        # Movement labeling
        # Light indexing scheme: 0 Left_N, 1 Straight_N, 2 Left_S, 3 Straight_S, 4 Left_E, 5 Straight_E, 6 Left_W, 7 Straight_W
        # But to match the user's list, we'll expose an ordering mapping in observation.
        self.light_count = 8

        # Spaces
        # lights: each in {0,1,2} -> Red, Yellow, Green
        lights_low = np.zeros(self.light_count, dtype=np.int64)
        lights_high = np.full(self.light_count, 2, dtype=np.int64)  # up to 2 (Green)
        # queues: N,S,E,W lengths
        self.num_dirs = 4
        self.max_queue = max_queue

        self.observation_space = spaces.Dict(
            {
                "lights": spaces.MultiDiscrete([3] * self.light_count),
                "queues": spaces.Box(low=0, high=self.max_queue, shape=(self.num_dirs,), dtype=np.float32),
                "emergency": spaces.MultiBinary(self.num_dirs),
                "time": spaces.Box(low=0.0, high=np.finfo(np.float32).max, shape=(), dtype=np.float32),
            }
        )

        # 12 discrete actions
        self.action_space = spaces.Discrete(12)

        # arrival rates per direction (N,S,E,W)
        default_arrival = {"N": 0.3, "S": 0.3, "E": 0.2, "W": 0.2}  # veh/sec defaults
        if arrival_rates is None:
            arrival_rates = default_arrival
        self.arrival_rates = arrival_rates

        # passage model parameters
        self.flow_rate = flow_rate

        # reward / penalty params
        self.throughput_threshold = throughput_threshold
        self.throughput_interval = throughput_interval
        self.switch_protect_time = switch_protect_time
        self.emergency_penalty_interval = emergency_penalty_interval
        self.emergency_reward_when_served = emergency_reward_when_served
        self.switch_penalty = switch_penalty
        self.queue_wait_penalty_coef = queue_wait_penalty_coef

        # internal counters
        self.last_switch_time_by_group = np.full(4, -1e9)  # groups: 0=NS_left,1=NS_straight,2=EW_left,3=EW_straight
        self.group_to_light_indices = {
            0: [0, 2],  # NS Left -> Left_N (0), Left_S (2)
            1: [1, 3],  # NS Straight -> Right_N (1), Right_S (3)
            2: [4, 6],  # EW Left -> Left_E (4), Left_W (6)
            3: [5, 7],  # EW Straight -> Right_E (5), Right_W (7)
        }

        # For throughput reward: track vehicles passed in current throughput interval
        self.passed_count_window = 0
        self.time_since_throughput_window_start = 0.0

        # Emergency wait timers per direction
        self.emergency_wait_time = np.zeros(self.num_dirs, dtype=np.float32)

        # queues array order: [N, S, E, W]
        self.queues = np.zeros(self.num_dirs, dtype=np.int32)

        # lights initial (all red)
        self.lights = np.zeros(self.light_count, dtype=np.int64)  # 0=Red,1=Yellow,2=Green

        # emergency flags per direction
        self.emergency_flags = np.zeros(self.num_dirs, dtype=np.int8)

        # time limit
        self.seed(seed)

        # Performance Metrics
        self.total_wait_time = 0.0
        self.total_cars_served = 0
        self.total_emergency_wait = 0.0
        self.total_emergency_cleared = 0

    def seed(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def _action_to_group_and_color(self, action: int) -> Tuple[int, int]:
        """
        Map 0..11 action to (group 0..3, color in {2(G),1(Y),0(R)}).
        Use ordering: (Green, Yellow, Red) repeated per group.
        """
        group = action // 3
        idx = action % 3
        # idx=0 => Green, 1=>Yellow, 2=>Red (per user's order G,Y,R)
        # convert idx to color code
        if idx == 0:
            color = 2
        elif idx == 1:
            color = 1
        else:
            color = 0
        return group, color

    def reset(self, *, seed: int | None = None, options: dict = None):
        if seed is not None:
            self.seed(seed)
        self.time = 0.0
        self.lights = np.zeros(self.light_count, dtype=np.int64)  # all red
        self.queues = np.zeros(self.num_dirs, dtype=np.int32)
        self.emergency_flags = np.zeros(self.num_dirs, dtype=np.int8)
        self.last_switch_time_by_group = np.full(4, -1e9)
        self.passed_count_window = 0
        self.time_since_throughput_window_start = 0.0
        self.emergency_wait_time = np.zeros(self.num_dirs, dtype=np.float32)

        # Optionally seed initial queues randomly (small)
        for i, key in enumerate(["N", "S", "E", "W"]):
            # initialize small random queue
            self.queues[i] = int(self.rng.poisson(1))

        obs = self._get_obs()
        return obs, {}

    def _get_obs(self) -> Dict[str, Any]:
        return {
            "lights": self.lights.copy(),
            "queues": self.queues.astype(np.float32).copy(),
            "emergency": self.emergency_flags.copy(),
            "time": np.float32(self.time),
        }

    def step(self, action: int):
        """
        Perform given action, simulate dt seconds.
        Returns (obs, reward, terminated, truncated, info)
        """
        reward = 0.0
        info = {}

        # 1) interpret action
        group, color = self._action_to_group_and_color(action)

        # 2) switching penalty if switched within protect time
        time_since_last = self.time - self.last_switch_time_by_group[group]
        switched = False
        # apply new color to corresponding light indices
        for li in self.group_to_light_indices[group]:
            if self.lights[li] != color:
                switched = True
                self.lights[li] = color

        if switched:
            # check switch penalty
            if time_since_last < self.switch_protect_time:
                reward += self.switch_penalty
                info["switch_penalty"] = self.switch_penalty
            self.last_switch_time_by_group[group] = self.time

        # 3) arrivals (Poisson)
        arrivals = np.zeros(self.num_dirs, dtype=int)
        dir_keys = ["N", "S", "E", "W"]
        for i, d in enumerate(dir_keys):
            lam = self.arrival_rates.get(d, 0.0) * self.dt
            # Poisson using RNG
            arrivals[i] = self.rng.poisson(lam)
            self.queues[i] = min(self.max_queue, self.queues[i] + arrivals[i])

        info["arrivals"] = arrivals.tolist()

        # 4) emergency random events: we will assume small probability of an emergency appearing this step
        # (You can set emergencies externally via info or a wrapper, too.)
        # For demonstration, a small chance per direction:
        for i in range(self.num_dirs):
            if self.emergency_flags[i] == 0:
                if self.rng.random() < 0.010:  # 1.0% chance per second by default (tunable)
                    self.emergency_flags[i] = 1
                    self.emergency_wait_time[i] = 0.0
                    info.setdefault("emergency_started", []).append(i)

        # 5) vehicles passing based on lights
        passed_this_step = 0
        # Map lights to which queue they allow:
        # NS Left group (lights indices 0 and 2) correspond to left-turning vehicles from N and S respectively:
        # map lights: index->direction queue index: Left_N(0)->N(0), Straight_N(1)->N(0), Left_S(2)->S(1), Straight_S(3)->S(1),
        # Left_E(4)->E(2), Straight_E(5)->E(2), Left_W(6)->W(3), Straight_W(7)->W(3)
        light_to_queue = {0:0, 1:0, 2:1, 3:1, 4:2, 5:2, 6:3, 7:3}

        for li in range(self.light_count):
            if self.lights[li] == 2:  # green
                qidx = light_to_queue[li]
                # vehicles can pass at up to flow_rate per second PER MOVEMENT light. To avoid double-counting from two greens for same direction,
                # we allocate capacity proportionally by active green movements for that queue.
                # Count how many green lights are currently for that queue:
                active_greens_for_queue = sum(
                    1 for l in range(self.light_count) if (light_to_queue[l] == qidx and self.lights[l] == 2)
                )
                if active_greens_for_queue <= 0:
                    continue
                # capacity share:
                capacity = max(1, int(self.flow_rate / active_greens_for_queue))
                # vehicles that actually pass
                passing = min(self.queues[qidx], capacity)
                self.queues[qidx] -= passing
                passed_this_step += passing

                # If an emergency is waiting in this direction and we've turned green, reward emergency served.
                if self.emergency_flags[qidx] == 1 and passing > 0:
                    reward += self.emergency_reward_when_served
                    info.setdefault("emergency_served", []).append(qidx)
                    # clear emergency
                    self.emergency_flags[qidx] = 0
                    self.emergency_wait_time[qidx] = 0.0

        self.passed_count_window += passed_this_step

        self.total_wait_time += float(self.queues.sum()) * self.dt
        self.total_cars_served += passed_this_step
        self.total_emergency_wait += float(self.emergency_wait_time.sum())
        self.total_emergency_cleared += len(info.get("emergency_served", []))

        # 6) throughput reward: +1 every time throughput threshold passed within throughput_interval
        self.time_since_throughput_window_start += self.dt
        reward += (self.passed_count_window / self.throughput_threshold) # OCTOBER 9TH
        if self.time_since_throughput_window_start >= self.throughput_interval:
            # check threshold
            if self.passed_count_window >= self.throughput_threshold:
                reward += 1.0
                info["throughput_reward"] = 1.0
            # reset
            self.passed_count_window = 0
            self.time_since_throughput_window_start = 0.0
        for i in range(self.num_dirs):        #OCTOBER 9TH
            if self.emergency_flags[i] == 1:
                self.emergency_wait_time[i] += self.dt

        # Penalize for waiting
                reward -= (self.dt / self.emergency_penalty_interval) * 1.0
                info.setdefault("emergency_penalty_accumulated", 0)
                info["emergency_penalty_accumulated"] += (self.dt / self.emergency_penalty_interval)

        # Extra reward if light turns green for that direction
                if (self.lights[i*2] == 2 or self.lights[i*2+1] == 2):
                    reward += self.emergency_reward_when_served
                    self.emergency_flags[i] = 0
                    self.emergency_wait_time[i] = 0.0
                    info.setdefault("emergency_served", []).append(i)



        # 8) queue waiting penalty: negative proportional to total waiting time for total queue length
        total_queue = int(self.queues.sum())
        # penalty is scaled by queue_wait_penalty_coef * total_queue * dt
        q_pen = self.queue_wait_penalty_coef * total_queue * self.dt
        reward += q_pen
        info["queue_penalty"] = q_pen

        info["avg_wait_time"] = (
            self.total_wait_time / max(1, self.total_cars_served)
        )

        info["emergency_response_efficiency"] = (
            self.total_emergency_cleared / max(1, (self.total_emergency_wait + 1e-6))
        )


        # 9) switching confusion penalty already applied earlier if any

        # advance time
        self.time += self.dt

        # Construct observation
        obs = self._get_obs()

        # Termination / truncation
        terminated = False
        truncated = False
        if self.time >= self.max_time:
            truncated = True

        return obs, float(reward), terminated, truncated, info

    def render(self, mode="human"):
        # Simple ASCII rendering
        lights_human = []
        mapping = {0: "R", 1: "Y", 2: "G"}
        for i in range(self.light_count):
            lights_human.append(mapping[int(self.lights[i])])
        s = (
            f"Time: {self.time:.1f}s | Lights (L_N,R_N,L_S,R_S,L_E,R_E,L_W,R_W): "
            + ",".join(lights_human)
            + f" | Queues N,S,E,W: {self.queues.tolist()} | Emergencies: {self.emergency_flags.tolist()}"
        )
        print(s)

    def close(self):
        pass


# Example usage:
if __name__ == "__main__":
    env = TrafficIntersectionEnv(seed=0)
    obs, _ = env.reset()
    print("Initial obs:", obs)
    for t in range(20):
        a = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(a)
        env.render()
        print("action", a, "reward", reward, "info", info)
        if term or trunc:
            break
