import numpy as np

class RobotEnvironment:
    def __init__(self):
        self.state_dim = 2  # distance and angle to target
        self.action_dim = 2  # linear and angular velocity
        self.target_position = np.array([-0.08, 0.06, 0.53])  # Human position
        self.robot_start_position = np.array([0.00, -0.95, 0.14])  # Robot start position
        self.current_position = self.robot_start_position.copy()
        self.proximity_threshold = 0.05  # Close proximity threshold
        self.frame_view_angle = np.pi / 3  # 60-degree field of view
        
        # Define wall boundaries
        self.wall_bounds = [
            {
                'x_min': -0.271567 * 0.42,
                'x_max': 1.014147 * 0.42,
                'y_min': -1.185,
                'y_max': -1.185 + (221 * 0.005)
            },
            {
                'x_min': -1 * 0.42,
                'x_max': -0.3333333 * 0.42,
                'y_min': -1.185,
                'y_max': -1.185 + (307 * 0.005)
            },
            {
                'x_min': -1.011905 * 0.42,
                'x_max': 1.011905 * 0.42,
                'y_min': -1.185,
                'y_max': -1.185 + (474 * 0.005)
            }
        ]

    def reset(self):
        # Update internal state
        self.current_position = self.robot_start_position.copy()
        return self.calculate_state()


    def step(self, action):
        # Update position based on action
        linear_scale = 0.5  # Adjust for meaningful linear motion
        angular_scale = np.pi / 4  # Adjust for meaningful angular motion

        prev_position = self.current_position.copy()
        self.current_position[0] += action[0] * linear_scale  # x position
        self.current_position[1] += action[1] * angular_scale  # y position

        # Calculate new state
        new_state = self.calculate_state()
        
        # Check for wall collision
        if self.check_wall_collision():
            return new_state, -10.0, True  # Large negative reward for wall collision

        # Calculate reward
        reward = self.calculate_reward(prev_position)
        
        # Check if goal reached
        done = self.is_goal_reached(new_state)
        if done:
            reward += 10.0  # Bonus reward for reaching goal

        return new_state, reward, done

    def calculate_state(self):
        # Calculate distance and angle to target
        diff = self.target_position[:2] - self.current_position[:2]
        distance = np.linalg.norm(diff)
        angle = np.arctan2(diff[1], diff[0])
        return np.array([distance, angle])

    def calculate_reward(self, prev_position):
        prev_dist = np.linalg.norm(self.target_position[:2] - prev_position[:2])
        curr_dist = np.linalg.norm(self.target_position[:2] - self.current_position[:2])
        reward = prev_dist - curr_dist

        # Penalize moving backward
        if reward > 0:
            reward *= 2  # Amplify positive progress

        if self.current_position[0] < prev_position[0]:  # Assuming forward is positive x
            reward -= 0.5

        return reward

    def is_goal_reached(self, state):
        # Check proximity
        if state[0] < self.proximity_threshold:
            return True
        
        # Check if target is within robot's frame
        if abs(state[1]) < self.frame_view_angle / 2:
            return True
        
        return False

    def check_wall_collision(self):
        x, y = self.current_position[:2]
        for wall in self.wall_bounds:
            if (x >= wall['x_min'] and x <= wall['x_max'] and
                y >= wall['y_min'] and y <= wall['y_max']):
                return True
        return False
