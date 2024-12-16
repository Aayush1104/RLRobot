import numpy as np

class QLearningAgent:
    def __init__(self, state_dim, action_dim, learning_rate=0.1, discount_factor=0.95, epsilon=0.1):
        self.q_table = {}
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.action_dim = action_dim
        # Initialize Q-table with forward-biased policy
        self.initialize_q_table()

    def initialize_q_table(self):
        # Encourage forward movement by default
        for state in self.generate_possible_states():
            self.q_table[state] = np.array([1.0, 0.0])  # Forward motion with no rotation

    def generate_possible_states(self):
        # Generate a grid of possible states for initialization
        distance_range = np.linspace(0, 2.0, 10)  # Example range for distance
        angle_range = np.linspace(-np.pi, np.pi, 10)  # Example range for angle
        states = []
        for d in distance_range:
            for a in angle_range:
                states.append((round(d, 1), round(a, 1)))
        return states
    def get_action(self, state):
        state_key = self.discretize_state(state)
        if np.random.rand() < self.epsilon:
            # Bias random actions toward forward movement
            return np.array([np.random.uniform(0.5, 1.0), np.random.uniform(-0.5, 0.5)])  # Forward motion
        return self.q_table.get(state_key, np.array([0.5, 0.0]))

    def update_q_table(self, state, action, reward, next_state):
        state_key = self.discretize_state(state)
        next_state_key = self.discretize_state(next_state)
        
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_dim)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.action_dim)

        current_q = self.q_table[state_key]
        max_next_q = np.max(self.q_table[next_state_key])
        new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state_key] = new_q

    def discretize_state(self, state):
        # Discretize continuous state for Q-table lookup
        return tuple(np.round(state, decimals=1))
