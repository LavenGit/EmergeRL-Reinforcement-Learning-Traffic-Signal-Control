import numpy as np
import gymnasium as gym
from gymnasium import spaces


class TrafficIntersectionEnv(gym.Env):
    """Four-way intersection with safe signal phases and movement-specific queues.

    Actions: 0 NS left, 1 NS straight, 2 EW left, 3 EW straight.
    Observation queues are ordered: N-left, N-straight, S-left, S-straight,
    E-left, E-straight, W-left, W-straight.
    """

    metadata = {"render_modes": ["human"], "render_fps": 1}
    PHASE_LIGHTS = ((0, 2), (1, 3), (4, 6), (5, 7))
    DIR_NAMES = ("N", "S", "E", "W")

    def __init__(self, max_queue=50, dt=1.0, max_time=3600.0,
                 arrival_rates=None, flow_rate=2, left_turn_fraction=0.3,
                 min_green_time=10.0, yellow_time=3.0, all_red_time=1.0,
                 emergency_rate=0.002, emergency_reward_when_served=5.0,
                 queue_wait_penalty_coef=-0.05, emergency_wait_penalty=-0.5,
                 switch_penalty=-0.2, seed=None):
        super().__init__()
        if dt <= 0 or max_time <= 0 or max_queue <= 0 or flow_rate <= 0:
            raise ValueError("dt, max_time, max_queue and flow_rate must be positive")
        if not 0 <= left_turn_fraction <= 1:
            raise ValueError("left_turn_fraction must be in [0, 1]")
        if min_green_time < 0 or yellow_time < 0 or all_red_time < 0:
            raise ValueError("Signal durations cannot be negative")
        if emergency_rate < 0:
            raise ValueError("emergency_rate cannot be negative")

        self.max_queue = int(max_queue)
        self.dt = float(dt)
        self.max_time = float(max_time)
        self.arrival_rates = arrival_rates or {"N": 0.3, "S": 0.3, "E": 0.2, "W": 0.2}
        self.flow_rate = float(flow_rate)
        self.left_turn_fraction = float(left_turn_fraction)
        self.min_green_time = float(min_green_time)
        self.yellow_time = float(yellow_time)
        self.all_red_time = float(all_red_time)
        self.emergency_rate = float(emergency_rate)
        self.emergency_reward_when_served = float(emergency_reward_when_served)
        self.queue_wait_penalty_coef = float(queue_wait_penalty_coef)
        self.emergency_wait_penalty = float(emergency_wait_penalty)
        self.switch_penalty = float(switch_penalty)
        self.rng = np.random.default_rng(seed)

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Dict({
            "lights": spaces.MultiDiscrete([3] * 8),
            "queues": spaces.Box(0, self.max_queue, shape=(8,), dtype=np.float32),
            "emergency": spaces.MultiBinary(4),
            "time": spaces.Box(0.0, np.float32(self.max_time), shape=(), dtype=np.float32),
            "phase": spaces.Discrete(5),  # 0..3 active, 4 transition/all red
            "transition_remaining": spaces.Box(0.0, np.float32(self.yellow_time + self.all_red_time), shape=(), dtype=np.float32),
        })
        self.reset(seed=seed)

    def _set_lights(self, phase, color):
        self.lights[:] = 0
        if phase is not None:
            for idx in self.PHASE_LIGHTS[phase]:
                self.lights[idx] = color

    def _get_obs(self):
        return {
            "lights": self.lights.copy(),
            "queues": self.queues.astype(np.float32).copy(),
            "emergency": self.emergency_flags.copy(),
            "time": np.float32(self.time),
            "phase": 4 if self.stage != "green" else int(self.current_phase),
            "transition_remaining": np.float32(self.stage_remaining),
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.time = 0.0
        self.queues = np.zeros(8, dtype=np.int32)
        for d in range(4):
            total = int(self.rng.poisson(1))
            left = int(self.rng.binomial(total, self.left_turn_fraction))
            self.queues[2*d] = left
            self.queues[2*d + 1] = total - left
        self.emergency_flags = np.zeros(4, dtype=np.int8)
        self.emergency_lane = np.full(4, -1, dtype=np.int8)
        self.emergency_wait_time = np.zeros(4, dtype=np.float32)
        self.current_phase = 1  # NS straight initially green
        self.pending_phase = None
        self.stage = "green"
        self.stage_remaining = 0.0
        self.green_elapsed = 0.0
        self.lights = np.zeros(8, dtype=np.int64)
        self._set_lights(self.current_phase, 2)
        self.total_wait_time = 0.0  # queue-vehicle-seconds, includes still-waiting vehicles
        self.total_cars_served = 0
        self.total_emergency_wait = 0.0
        self.total_emergency_cleared = 0
        self.emergency_clearance_wait_sum = 0.0
        self.total_arrivals = 0
        self.total_dropped = 0
        self.total_switches = 0
        return self._get_obs(), {}

    def _advance_signals(self, requested_phase, elapsed):
        """Advance a signal state machine by elapsed simulated seconds."""
        remaining = elapsed
        while remaining > 1e-10:
            if self.stage == "green":
                # Hold green until the minimum duration has elapsed.
                if requested_phase != self.current_phase and self.green_elapsed >= self.min_green_time:
                    self.pending_phase = requested_phase
                    self.stage = "yellow"
                    self.stage_remaining = self.yellow_time
                    self._set_lights(self.current_phase, 1)
                    self.total_switches += 1
                    continue
                if requested_phase != self.current_phase:
                    until_eligible = max(0.0, self.min_green_time - self.green_elapsed)
                    if until_eligible > 0:
                        advance = min(remaining, until_eligible)
                        self.green_elapsed += advance
                        remaining -= advance
                        continue
                self.green_elapsed += remaining
                remaining = 0.0
            else:
                advance = min(remaining, self.stage_remaining)
                self.stage_remaining -= advance
                remaining -= advance
                if self.stage_remaining > 1e-10:
                    break
                if self.stage == "yellow":
                    self.stage = "all_red"
                    self.stage_remaining = self.all_red_time
                    self._set_lights(None, 0)
                else:
                    self.current_phase = self.pending_phase
                    self.pending_phase = None
                    self.stage = "green"
                    self.green_elapsed = 0.0
                    self.stage_remaining = 0.0
                    self._set_lights(self.current_phase, 2)
        # Complete zero-duration transitions, if configured.
        for _ in range(2):
            if self.stage == "yellow" and self.stage_remaining <= 1e-10:
                self.stage = "all_red"
                self.stage_remaining = self.all_red_time
                self._set_lights(None, 0)
            elif self.stage == "all_red" and self.stage_remaining <= 1e-10:
                self.current_phase = self.pending_phase
                self.pending_phase = None
                self.stage = "green"
                self.green_elapsed = 0.0
                self.stage_remaining = 0.0
                self._set_lights(self.current_phase, 2)
            else:
                break

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"Action must be an integer from 0 to 3; got {action!r}")
        if self.time >= self.max_time:
            raise RuntimeError("Episode has ended; call reset() before step()")

        step_dt = min(self.dt, self.max_time - self.time)
        previous_switches = self.total_switches
        self._advance_signals(int(action), step_dt)
        reward = self.switch_penalty * (self.total_switches - previous_switches)

        arrivals = np.zeros(8, dtype=np.int32)
        for d, name in enumerate(self.DIR_NAMES):
            total = int(self.rng.poisson(self.arrival_rates.get(name, 0.0) * step_dt))
            left = int(self.rng.binomial(total, self.left_turn_fraction))
            arrivals[2*d] = left
            arrivals[2*d + 1] = total - left
        self.total_arrivals += int(arrivals.sum())
        available = self.max_queue - self.queues
        accepted = np.minimum(arrivals, available)
        self.total_dropped += int((arrivals - accepted).sum())
        self.queues += accepted

        # One emergency per direction, attached to an occupied movement queue.
        for d in range(4):
            if self.emergency_flags[d] == 0 and self.queues[2*d:2*d+2].sum() > 0:
                if self.rng.random() < 1.0 - np.exp(-self.emergency_rate * step_dt):
                    candidates = [lane for lane in (2*d, 2*d+1) if self.queues[lane] > 0]
                    self.emergency_lane[d] = int(self.rng.choice(candidates))
                    self.emergency_flags[d] = 1
                    self.emergency_wait_time[d] = 0.0

        passed = np.zeros(8, dtype=np.int32)
        if self.stage == "green":
            for lane in self.PHASE_LIGHTS[self.current_phase]:
                capacity = int(self.rng.poisson(self.flow_rate * step_dt))
                passed[lane] = min(int(self.queues[lane]), capacity)
                self.queues[lane] -= passed[lane]

        served_emergencies = []
        for d in range(4):
            if self.emergency_flags[d]:
                lane = int(self.emergency_lane[d])
                # Approximation: an emergency departs when its assigned lane serves a vehicle.
                if lane >= 0 and passed[lane] > 0:
                    self.emergency_clearance_wait_sum += float(self.emergency_wait_time[d])
                    self.total_emergency_cleared += 1
                    self.emergency_flags[d] = 0
                    self.emergency_lane[d] = -1
                    self.emergency_wait_time[d] = 0.0
                    served_emergencies.append(d)
                else:
                    self.emergency_wait_time[d] += step_dt
                    self.total_emergency_wait += step_dt

        served = int(passed.sum())
        self.total_cars_served += served
        waiting = int(self.queues.sum())
        self.total_wait_time += waiting * step_dt
        reward += float(served)
        reward += self.queue_wait_penalty_coef * waiting * step_dt
        reward += self.emergency_wait_penalty * int(self.emergency_flags.sum()) * step_dt
        reward += self.emergency_reward_when_served * len(served_emergencies)
        self.time += step_dt

        info = {
            "arrivals": arrivals.tolist(),
            "passed": passed.tolist(),
            "passed_this_step": served,
            "total_cars_served": self.total_cars_served,
            "total_queue": waiting,
            "total_dropped": self.total_dropped,
            "queue_vehicle_seconds": self.total_wait_time,
            "queue_wait_per_departure_proxy": self.total_wait_time / max(1, self.total_cars_served),
            "emergency_served": served_emergencies,
            "emergency_cleared": self.total_emergency_cleared,
            "avg_cleared_emergency_wait": self.emergency_clearance_wait_sum / max(1, self.total_emergency_cleared),
            "switches": self.total_switches,
        }
        return self._get_obs(), float(reward), False, self.time >= self.max_time, info

    def render(self):
        labels = {0: "R", 1: "Y", 2: "G"}
        print(f"t={self.time:.0f}s phase={self.current_phase} stage={self.stage} "
              f"lights={''.join(labels[x] for x in self.lights)} "
              f"queues={self.queues.tolist()} emergencies={self.emergency_flags.tolist()}")

    def close(self):
        pass


if __name__ == "__main__":
    env = TrafficIntersectionEnv(seed=42, max_time=30)
    obs, _ = env.reset(seed=42)
    for _ in range(30):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        env.render()
        if terminated or truncated:
            break
    env.close()
