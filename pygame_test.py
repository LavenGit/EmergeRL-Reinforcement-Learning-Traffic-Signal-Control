# traffic_sim_pygame_smooth.py
import pygame
import sys
import time
import numpy as np
from tabulate import tabulate
from gymnasium_env.envs import TrafficIntersectionEnv  
from sarsa_agent import SarsaAgent
from mappo_agent import MAPPOAgent  # import your MAPPO class

import math
import torch
# -----------------------------
# Pygame setup
# -----------------------------
pygame.init()
WIDTH, HEIGHT = 800, 800
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Traffic Intersection Simulation")
CLOCK = pygame.time.Clock()
FPS = 20 # simulation steps per second

# Colors
GRAY = (50, 50, 50)
WHITE = (255, 255, 255)
RED = (220, 0, 0)
YELLOW = (220, 220, 0)
GREEN = (0, 220, 0)
BLUE = (0, 0, 220)
ORANGE = (255, 128, 0)
BLACK = (0,0,0)

CAR_IMG = pygame.image.load("assets/car.png").convert()
CAR_IMG.set_colorkey((255, 255, 255))
CAR_IMG = pygame.transform.scale(CAR_IMG, (40, 20))

EMERGENCY_IMG = pygame.image.load("assets/emergency.png").convert_alpha()
EMERGENCY_IMG = pygame.transform.scale(EMERGENCY_IMG, (45, 35))

# Stop line positions (x or y coordinate for each lane)
STOP_LINES = {
    'N': HEIGHT//2 - 50,  # vertical road, stopping before intersection
    'S': HEIGHT//2 + 50,
    'E': WIDTH//2 + 50,
    'W': WIDTH//2 - 50
}

# -----------------------------
# Vehicle class with stop line
# -----------------------------
class Vehicle:
    def __init__(self, direction, turning=False, emergency=False, lane_choice="right"):
        self.direction = direction
        self.turning = turning
        self.emergency = emergency
        self.speed = 5  # pixels per frame
        self.progress = 0.0
        self.stopped = False
        self.done = False

        # Lane Positions
        CENTER_X, CENTER_Y = WIDTH // 2, HEIGHT // 2
        L_OUTER = 37   # outer lane (straight)
        L_INNER = 12   # inner lane (left turn)
        CAR_HALF = 20
        OFFSET_OUT = 140  # spawn/exit past screen
        TURN_R = 65

        def line_path(x0, y0, x1, y1, steps=100):
            xs = np.linspace(x0, x1, steps)
            ys = np.linspace(y0, y1, steps)
            return list(zip(xs, ys))

        def arc_path(xc, yc, r, start_angle, end_angle, steps=40, x_offset=0, y_offset=0):
            ang = np.linspace(start_angle, end_angle, steps)
            xs = xc + r * np.cos(ang) + x_offset
            ys = yc + r * np.sin(ang) + y_offset
            return list(zip(xs, ys))

        # Build a full path (approach -> arc -> exit) for left turns
        def build_left_turn_path(dir_):
            if dir_ == "N":
                # North to East
                y_start = -OFFSET_OUT
                y_stop = STOP_LINES['N'] - CAR_HALF
                # Approach from North inner lane
                approach = line_path(CENTER_X - L_INNER, y_start,
                                     CENTER_X - L_INNER, y_stop)

                # Arc pivot: **inside** intersection, bottom-right of approach lane
                xc, yc = CENTER_X + TURN_R, CENTER_Y + TURN_R
                arc = arc_path(xc, yc, TURN_R, np.pi, np.pi/2, x_offset=-10, y_offset=-120)

                # Exit: move right along East inner lane
                exit_seg = line_path(CENTER_X + TURN_R, CENTER_Y + L_INNER,
                                     WIDTH + OFFSET_OUT, CENTER_Y + L_INNER)

            elif dir_ == "S":
                # South to West
                y_start = HEIGHT + OFFSET_OUT
                y_stop = STOP_LINES['S'] + CAR_HALF
                # Approach from South inner lane
                approach = line_path(CENTER_X + L_INNER, y_start,
                                     CENTER_X + L_INNER, y_stop)

                # Arc pivot: inside intersection, upper-left of approach lane
                xc, yc = CENTER_X - TURN_R, CENTER_Y + TURN_R
                arc = arc_path(xc, yc, TURN_R, 0, -np.pi/2, x_offset=10, y_offset=-10)

                # Exit: move left along West inner lane
                exit_seg = line_path(CENTER_X - TURN_R, CENTER_Y - L_INNER,
                                     -OFFSET_OUT, CENTER_Y - L_INNER)

            elif dir_ == "E":
                # East to South
                x_start = WIDTH + OFFSET_OUT
                x_stop = STOP_LINES['E'] + CAR_HALF
                # Approach from East inner lane
                approach = line_path(x_start, CENTER_Y - L_INNER,
                                     x_stop,  CENTER_Y - L_INNER)

                # Arc pivot: inside intersection, upper-left of approach lane
                xc, yc = CENTER_X + TURN_R, CENTER_Y - TURN_R
                arc = arc_path(xc, yc, TURN_R, 3*np.pi/2, np.pi, x_offset=-10, y_offset=120)

                # Exit: move down along South inner lane
                exit_seg = line_path(CENTER_X - L_INNER + 25, CENTER_Y - TURN_R,
                                     CENTER_X - L_INNER + 25, -OFFSET_OUT)

            elif dir_ == "W":
                # West to North
                x_start = -OFFSET_OUT
                x_stop = STOP_LINES['W'] - CAR_HALF
                # Approach from West inner lane
                approach = line_path(x_start, CENTER_Y + L_INNER,
                                     x_stop,  CENTER_Y + L_INNER)

                # Arc pivot: inside intersection, bottom-right of approach lane
                xc, yc = CENTER_X - TURN_R, CENTER_Y + TURN_R
                arc = arc_path(xc, yc, TURN_R, np.pi/2, 0, x_offset=10, y_offset=-120)

                # Exit: move up along North inner lane
                exit_seg = line_path(CENTER_X + L_INNER - 25, CENTER_Y + TURN_R,
                                     CENTER_X + L_INNER - 25, HEIGHT + OFFSET_OUT)

            else:
                approach = arc = exit_seg = []

            return approach + arc + exit_seg
        
        # Build path
        if self.turning:
            self.path = build_left_turn_path(self.direction)
        else:
            # Straight uses the outer lane the whole way
            if self.direction == "N":
                self.path = line_path(CENTER_X - L_OUTER, -OFFSET_OUT,
                                      CENTER_X - L_OUTER, HEIGHT + OFFSET_OUT)
            elif self.direction == "S":
                self.path = line_path(CENTER_X + L_OUTER, HEIGHT + OFFSET_OUT,
                                      CENTER_X + L_OUTER, -OFFSET_OUT)
            elif self.direction == "E":
                self.path = line_path(WIDTH + OFFSET_OUT, CENTER_Y - L_OUTER,
                                      -OFFSET_OUT,        CENTER_Y - L_OUTER)
            elif self.direction == "W":
                self.path = line_path(-OFFSET_OUT,        CENTER_Y + L_OUTER,
                                      WIDTH + OFFSET_OUT, CENTER_Y + L_OUTER)
            else:
                self.path = []

        # Initial position
        self.x, self.y = self.path[0] if self.path else (0, 0)

        # Stop line coordinates based on direction
        if self.direction == 'N':
            lane_x = (CENTER_X - (L_INNER if self.turning else L_OUTER))
            self.stop_line = (lane_x, STOP_LINES['N'])
        elif self.direction == 'S':
            lane_x = (CENTER_X + (L_INNER if self.turning else L_OUTER))
            self.stop_line = (lane_x, STOP_LINES['S'])
        elif self.direction == 'E':
            lane_y = (CENTER_Y - (L_INNER if self.turning else L_OUTER))
            self.stop_line = (STOP_LINES['E'], lane_y)
        elif self.direction == 'W':
            lane_y = (CENTER_Y + (L_INNER if self.turning else L_OUTER))
            self.stop_line = (STOP_LINES['W'], lane_y)
        else:
            self.stop_line = None

    def update(self, light_green, vehicles_ahead):
        if self.emergency:
            light_green = True

        CAR_HALF = 20
        MIN_GAP = 50

        # Check stop line and control movement
        if self.direction == 'N':
            stop_pos = self.stop_line[1] - CAR_HALF
            before_stop = self.y < stop_pos
        elif self.direction == 'S':
            stop_pos = self.stop_line[1] + CAR_HALF
            before_stop = self.y > stop_pos
        elif self.direction == 'E':
            stop_pos = self.stop_line[0] + CAR_HALF
            before_stop = self.x > stop_pos
        elif self.direction == 'W':
            stop_pos = self.stop_line[0] - CAR_HALF
            before_stop = self.x < stop_pos
        else:
            before_stop = False  # default

        # Correct can_move logic:
        if before_stop:
            can_move = light_green      # Must obey light BEFORE stop line
        else:
            can_move = True 

        # Queue spacing
        if can_move:
            for o in vehicles_ahead:
                if self.direction == 'N' and 0 < o.y - self.y < MIN_GAP:
                    can_move = False
                elif self.direction == 'S' and 0 < self.y - o.y < MIN_GAP:
                    can_move = False
                elif self.direction == 'E' and 0 < self.x - o.x < MIN_GAP:
                    can_move = False
                elif self.direction == 'W' and 0 < o.x - self.x < MIN_GAP:
                    can_move = False
                if not can_move:
                    break

        # Advance along path
        if can_move and self.path:
            self.stopped = False
            self.progress += self.speed / 10.0
            idx_f = self.progress
            idx = int(idx_f)
            frac = idx_f - idx

            # When car reaches end of path, mark it as done
            if idx >= len(self.path) - 1:
                self.done = True
                return

            # Normal movement
            x1, y1 = self.path[idx]
            x2, y2 = self.path[idx + 1]

            self.x = x1 + frac * (x2 - x1)
            self.y = y1 + frac * (y2 - y1)

            dx = x2 - x1
            dy = y2 - y1
            self.angle = -math.degrees(math.atan2(dy, dx))

        else:
            self.stopped = True


    def draw(self, screen):
        img = EMERGENCY_IMG if self.emergency else CAR_IMG
        angle = getattr(self, "angle", 0)
        if img == EMERGENCY_IMG:
            angle -= 90
        # Pygame rotation: rotate image by current heading
        rotated = pygame.transform.rotate(img, angle)
        rect = rotated.get_rect(center=(int(self.x), int(self.y)))
        
        screen.blit(rotated, rect)

# -----------------------------
# Traffic simulation
# -----------------------------
class TrafficSim:
    def __init__(self,agent_type="MAPPO"):
        # --- Initialize environment ---
        self.env = TrafficIntersectionEnv(
            seed=None,  # same scenario each time if seed is fixed
            arrival_rates={"N": 0.35, "S": 0.35, "E": 0.35, "W": 0.35}
        )
        self.obs, _ = self.env.reset()

        # --- Simulation state ---
        self.vehicles = []
        self.total_reward = 0.0
        self.last_light_change = time.time()
        self.light_interval = 10.0  # seconds between light changes

        # --- Define state/action sizes for RL agent ---
        state_size = 16   # lights + queues + emergency
        action_size = 12  # 12 discrete light combinations

        # --- Initialize RL Agent (MAPPO) ---
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.agent_type = agent_type
        if agent_type == "MAPPO":
            self.agent = MAPPOAgent(num_agents=1, obs_dim=17, action_dim=12)
        elif agent_type == "SARSA":
            from sarsa_agent import SarsaAgent
            self.agent = SarsaAgent(state_size=17, action_size=12)
        else:
            self.agent = None 

        # --- Link environment with agent control ---
        self.env.agent = True  
        # --- Light control setup ---
        self.current_group = 0
        self.obs['lights'] = np.zeros(8, dtype=np.int64)  # all red to start
        self.group_to_light_indices = {
            0: [0, 2],  # NS Left
            1: [1, 3],  # NS Straight
            2: [4, 6],  # EW Left
            3: [5, 7],  # EW Straight
        }
        # Counter for throttled debug printing
        self.update_counter = 0

        # Reward normalization
        self.reward_mean = 0.0
        self.reward_m2 = 1.0
        self.reward_count = 0
        self.reward_clip_range = 5.0
    
    def normalize_reward(self, reward):
        # clip large rewards
        r = np.clip(reward, -self.reward_clip_range, self.reward_clip_range)

        # update running stats
        self.reward_count += 1
        delta = r - self.reward_mean
        self.reward_mean += delta / self.reward_count
        delta2 = r - self.reward_mean
        self.reward_m2 += delta * delta2

        # standard deviation
        std = 1.0
        if self.reward_count > 1:
            std = np.sqrt(self.reward_m2 / self.reward_count)

        # normalize
        if std < 1e-6:
            out = r - self.reward_mean
        else:
            out = (r - self.reward_mean) / std

        # clip final value
        return float(np.clip(out * 0.5, -3.0, 3.0))

    def get_state_vector(self, obs=None):
        """
        Convert observation into 1D state vector for RL agent.
        If obs is None, use current self.obs.
        """
        obs = self.obs if obs is None else obs
        lights = obs['lights'].astype(np.float32)
        queues = obs['queues'].astype(np.float32)
        emergency = obs['emergency'].astype(np.float32)
        state = np.concatenate([lights, queues, emergency])

        return state
    def spawn_vehicles(self):
        dirs = ['N','S','E','W']
        for i,d in enumerate(dirs):
            target_count = int(self.obs['queues'][i])
            existing = sum(1 for v in self.vehicles if v.direction == d)
            to_spawn = max(0, target_count - existing)
            MIN_SPAWN_GAP = 60  # pixels between vehicles
            
            for _ in range(to_spawn):
                turning = np.random.rand() < 0.5
                emergency = self.obs['emergency'][i] == 1

                lane_choice = "left" if turning else "right"
                new_vehicle = Vehicle(d, turning, emergency, lane_choice)

                same_dir = [v for v in self.vehicles if v.direction == d and v.turning == turning]
                too_close = False
                for v in same_dir:
                    dx = abs(v.x - new_vehicle.x)
                    dy = abs(v.y - new_vehicle.y)
                    if np.hypot(dx, dy) < MIN_SPAWN_GAP:
                        too_close = True
                        break

                if not too_close:
                    self.vehicles.append(new_vehicle)

    def get_light_status_for_vehicle(self, vehicle):
        light_idx = None
        if vehicle.direction=='N':
            light_idx = 0 if vehicle.turning else 1
        elif vehicle.direction=='S':
            light_idx = 2 if vehicle.turning else 3
        elif vehicle.direction=='E':
            light_idx = 4 if vehicle.turning else 5
        elif vehicle.direction=='W':
            light_idx = 6 if vehicle.turning else 7
        return self.obs['lights'][light_idx] == 2  # green

    def draw_intersection(self):
        SCREEN.fill('darkgreen')
        pygame.draw.rect(SCREEN, GRAY, (0, HEIGHT//2 - 50, WIDTH, 100))
        pygame.draw.rect(SCREEN, GRAY, (WIDTH//2 -50, 0, 100, HEIGHT))

        # Draw median lines
        pygame.draw.line(SCREEN, YELLOW, (WIDTH // 2, 0), (WIDTH // 2, HEIGHT // 2 - 50), 1)
        pygame.draw.line(SCREEN, YELLOW, (WIDTH // 2, HEIGHT // 2 + 50), (WIDTH // 2, HEIGHT), 1)
        pygame.draw.line(SCREEN, YELLOW, (0, HEIGHT // 2), (WIDTH // 2 - 50, HEIGHT // 2), 1)
        pygame.draw.line(SCREEN, YELLOW, (WIDTH // 2 + 50, HEIGHT // 2), (WIDTH, HEIGHT // 2), 1)
        

        # Draw lane lines
        pygame.draw.line(SCREEN, WHITE, (WIDTH // 2 - 25, 0), (WIDTH // 2 - 25, HEIGHT // 2 - 50), 1)
        pygame.draw.line(SCREEN, WHITE, (WIDTH // 2 + 25, 0), (WIDTH // 2 + 25, HEIGHT // 2 - 50), 1)
        pygame.draw.line(SCREEN, WHITE, (WIDTH // 2 - 25, HEIGHT // 2 + 50), (WIDTH // 2 - 25, HEIGHT), 1)
        pygame.draw.line(SCREEN, WHITE, (WIDTH // 2 + 25, HEIGHT // 2 + 50), (WIDTH // 2 + 25, HEIGHT), 1)
        pygame.draw.line(SCREEN, WHITE, (0, HEIGHT // 2 - 25), (WIDTH // 2 - 50, HEIGHT // 2 - 25), 1)
        pygame.draw.line(SCREEN, WHITE, (0, HEIGHT // 2 + 25), (WIDTH // 2 - 50, HEIGHT // 2 + 25), 1)
        pygame.draw.line(SCREEN, WHITE, (WIDTH // 2 + 50, HEIGHT // 2 - 25), (WIDTH, HEIGHT // 2 - 25), 1)
        pygame.draw.line(SCREEN, WHITE, (WIDTH // 2 + 50, HEIGHT // 2 + 25), (WIDTH, HEIGHT // 2 + 25), 1)

        # Erase parts of lane lines
        dash_spacing = 20
        dash_size = 10
        # Vertical lines (use y)
        for y in range(0, HEIGHT, dash_spacing):
            pygame.draw.rect(SCREEN, GRAY, (WIDTH // 2 - 26, y + dash_size, 3, dash_size))
            pygame.draw.rect(SCREEN, GRAY, (WIDTH // 2 + 24, y + dash_size, 3, dash_size))
        # Horizontal lines (use x)
        for x in range(0, WIDTH, dash_spacing):
            pygame.draw.rect(SCREEN, GRAY, (x + dash_size, HEIGHT // 2 - 26, dash_size, 3))
            pygame.draw.rect(SCREEN, GRAY, (x + dash_size, HEIGHT // 2 + 24, dash_size, 3))

        # Draw lights
        light_positions = {
            0: (WIDTH//2, HEIGHT//2 - 60),
            1: (WIDTH//2 - 50, HEIGHT//2 - 60),
            2: (WIDTH//2, HEIGHT//2 + 60),
            3: (WIDTH//2 + 50, HEIGHT//2 + 60),
            4: (WIDTH//2 + 60, HEIGHT//2),
            5: (WIDTH//2 + 60, HEIGHT//2 - 50),
            6: (WIDTH//2 - 60, HEIGHT//2),
            7: (WIDTH//2 - 60, HEIGHT//2 + 50),
        }
        color_map = {0:RED, 1:YELLOW, 2:GREEN}
        for i,pos in light_positions.items():
            c = color_map[int(self.obs['lights'][i])]
            pygame.draw.circle(SCREEN, c, pos, 7.5)

        # Draw stop lines
        pygame.draw.line(SCREEN, WHITE, (WIDTH//2 - 50, STOP_LINES['N']), (WIDTH//2, STOP_LINES['N']), 3)
        pygame.draw.line(SCREEN, WHITE, (WIDTH//2, STOP_LINES['S']), (WIDTH//2 + 50, STOP_LINES['S']), 3)
        pygame.draw.line(SCREEN, WHITE, (STOP_LINES['E'], HEIGHT//2 - 50), (STOP_LINES['E'], HEIGHT//2), 3)
        pygame.draw.line(SCREEN, WHITE, (STOP_LINES['W'], HEIGHT//2), (STOP_LINES['W'], HEIGHT//2 + 50), 3)

    def update(self, dt):
        self.counter = getattr(self, "counter", 0) + 1
        self.update_counter = getattr(self, "update_counter", 0) + 1

        # Traffic lights automatic switching (if no RL agent)
        if not self.agent:
            if self.counter * dt > self.light_interval:
                self.current_group = (self.current_group + 1) % 4
                self.last_light_change = time.time()
                self.obs['lights'][:] = 0
                for idx in self.group_to_light_indices[self.current_group]:
                    self.obs['lights'][idx] = 2

        # RL agent update
        if self.agent:
            if self.agent_type == "SARSA":
                self._update_sarsa()
            elif self.agent_type == "MAPPO":
                self._update_mappo()

        # Spawn & update vehicles
        self._update_vehicles()
    def _update_sarsa(self):
        try:
            state = self.get_state_vector()
            action = self.agent.choose_action(state)

            next_obs, reward, terminated, truncated, info = self.env.step(action)
            next_state = self.get_state_vector(next_obs)
            next_action = self.agent.choose_action(next_state)
            done = terminated or truncated

            self.agent.learn(state, action, reward, next_state, next_action, done)
            self.agent.update_epsilon()

            self.obs = next_obs
            self.total_reward += reward

        except Exception as e:
            print(f"Error during SARSA update: {e}")
        if self.update_counter % 100 == 0:
            self._print_summary(action, reward, info)

    def _update_mappo(self):
        try:
            state_dict = self.obs
            with torch.no_grad():
                action_list, log_probs, obs_vec = self.agent.select_actions(state_dict)
                action = action_list[0]

            next_obs, reward, terminated, truncated, info = self.env.step(action)
            next_obs_vec = self.agent.flatten_obs(next_obs)
            reward += -0.1 * info.get("avg_wait_time", 0)
            reward += 0.5 * info.get("emergency_response_efficiency", 0)
            normalized_reward = self.normalize_reward(reward)

            self.agent.store_transition(
                obs_vec,
                action_list,
                log_probs,
                normalized_reward,
                next_obs_vec,
                done=terminated or truncated
            )

            if self.update_counter % 32 == 0:
                self.agent.update()

            self.obs = next_obs
            self.total_reward += reward

        except Exception as e:
            print(f"Error during MAPPO update: {e}")
        if self.update_counter % 100 == 0:
            self._print_summary(action, reward, info)
    def _update_vehicles(self):
        # your existing vehicle update code
        self.spawn_vehicles()
        for d in ['N','S','E','W']:
            same_dir = [v for v in self.vehicles if v.direction==d]
            if d=='N': same_dir.sort(key=lambda v: v.y, reverse=True)
            elif d=='S': same_dir.sort(key=lambda v: v.y)
            elif d=='E': same_dir.sort(key=lambda v: v.x)
            elif d=='W': same_dir.sort(key=lambda v: v.x, reverse=True)
            for i, v in enumerate(same_dir):
                light_green = self.get_light_status_for_vehicle(v)
                vehicles_ahead = same_dir[:i]
                v.update(light_green, vehicles_ahead)
        self.vehicles = [v for v in self.vehicles if not getattr(v,"done",False)]
    def _print_summary(self, action, reward, info):
        light_states = ["Red", "Yellow", "Green"]
        directions = ["North", "South", "East", "West"]

        env_headers = ["Step", "Action", "Reward", "Total", "Active Vehicles", "Avg Wait", "Emergency Efficiency"]
        env_data = [[
            self.update_counter,
            action,
            f"{reward:.2f}",
            f"{self.total_reward:.2f}",
            len(self.vehicles),
            f"{info.get('avg_wait_time', 0):.2f}",
            f"{info.get('emergency_response_efficiency', 0):.2f}"
        ]]
        env_table = tabulate(env_data, headers=env_headers, tablefmt="github", stralign="center", numalign="center")

    # Traffic info
        traffic_data = []
        for i, direction in enumerate(directions):
            queue = self.obs["queues"][i] if i < len(self.obs["queues"]) else 0
            emergency = self.obs["emergency"][i] if i < len(self.obs["emergency"]) else 0
            traffic_data.append([
                direction,
                f"{light_states[int(self.obs['lights'][i * 2])]}", 
                f"{light_states[int(self.obs['lights'][i * 2 + 1])]}", 
                queue,
                emergency
            ])
        traffic_table = tabulate(traffic_data, headers=["Direction", "Left", "Straight", "Queue", "Emergency Flags"],
                             tablefmt="github", stralign="center", numalign="center")

    # Print tables
        print(env_table, "\n")
        print(traffic_table, "\n")
        print("=" * 59, "\n")

    # Reset total reward for next batch
        self.total_reward = 0

    def draw_vehicles(self):
        for v in self.vehicles:
            v.draw(SCREEN)

# -----------------------------
# Main loop
# -----------------------------
def main():
    agent_type = "MAPPO" # write "SARSA"  or "MAPPO" for the RL Algorithm interaction
    sim = TrafficSim(agent_type=agent_type)

    # Print which algorithm is running
    print(f"Running {agent_type} algorithm...")

    running = True
    last_time = time.time()
    while running:
        dt = time.time() - last_time
        last_time = time.time()
        CLOCK.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        sim.update(dt)
        sim.draw_intersection()
        sim.draw_vehicles()
        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()