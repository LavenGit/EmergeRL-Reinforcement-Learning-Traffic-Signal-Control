import numpy as np

class SarsaAgent:
    def __init__(self, 
                 state_size: int, 
                 action_size: int, 
                 alpha: float = 0.1, 
                 gamma: float = 0.9, 
                 epsilon: float = 0.1,
                 epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995,
                 debug: bool = False):

        self.state_size = state_size
        self.action_size = action_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table = {}  # key = state tuple, value = action values array
        self.debug = debug
        self.update_count = 0

    def _state_to_key(self, state: np.ndarray) -> tuple:
        """
        Convert state to a hashable key for Q-table.
        Rounds floats to 4 decimals to reduce table size.
        """
        return tuple(np.round(state, 4))

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Get Q-values for a state, initializing if necessary."""
        key = self._state_to_key(state)
        if key not in self.q_table:
            self.q_table[key] = np.zeros(self.action_size, dtype=np.float32)
        return self.q_table[key]

    def choose_action(self, state: np.ndarray) -> int:
        """Epsilon-greedy action selection."""
        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.action_size)
            if self.debug:
                print(f"Random action {action} selected (epsilon={self.epsilon:.3f})")
            return action
        q_values = self.get_q_values(state)
        action = int(np.argmax(q_values))
        if self.debug:
            print(f"Greedy action {action} selected with Q={q_values[action]:.3f}")
        return action

    def learn(self, state: np.ndarray, action: int, reward: float, 
              next_state: np.ndarray, next_action: int, done: bool):
        """
        SARSA Q-table update:
            Q(s,a) <- Q(s,a) + alpha * [r + gamma * Q(s',a') - Q(s,a)]
        """
        q_values = self.get_q_values(state)
        next_q = self.get_q_values(next_state)[next_action] if not done else 0.0
        target = reward + self.gamma * next_q
        q_values[action] += self.alpha * (target - q_values[action])

        self.update_count += 1
        if self.debug:
            print(f"Update {self.update_count}: s={state}, a={action}, r={reward}, s'={next_state}, a'={next_action}")
            print(f"Q[{action}] -> {q_values[action]:.3f}")

    def update_epsilon(self, method: str = "exponential"):
        """
        Decay epsilon to reduce exploration over time.

        method: 'exponential' (default) or 'linear'
        """
        if method == "exponential":
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        elif method == "linear":
            self.epsilon = max(self.epsilon_min, self.epsilon - (1 - self.epsilon_min) * 0.001)
        if self.debug:
            print(f"Epsilon updated to {self.epsilon:.4f}")
