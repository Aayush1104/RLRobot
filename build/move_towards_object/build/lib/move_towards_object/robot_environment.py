import numpy as np
import cv2

class RobotEnvironment:
    def __init__(self):
        self.state_dim = 22  # 20 lidar readings + 2 goal distances
        self.action_dim = 1  # steering angle
        self.vel = 3.0  # constant velocity
        self.goal = [24.0, -24.0]  # goal position
        self.robot_start_position = np.array([0.0, 0.0, 0.0])  # x, y, theta
        self.current_position = self.robot_start_position.copy()
        self.lidar_data = np.zeros(20)
        self.prev_dist = 0
        
        # Image parameters for collision detection
        self.image_size = 720
        self.pix2m = 0.1
        self.collision_image = np.zeros((self.image_size, self.image_size), np.uint8)

    def reset(self):
        self.current_position = self.robot_start_position.copy()
        self.prev_dist = 0
        self.collision_image[:] = 0
        return np.zeros(self.state_dim)

    def step(self, action, time_steps, max_episode_steps):
        # Update position
        self.current_position[2] = 2 * action[0]  # Update theta
        
        # Calculate distance to goal
        distance = np.sqrt((self.current_position[0] - self.goal[0])**2 + 
                         (self.current_position[1] - self.goal[1])**2)
        
        # Create state vector
        state = np.zeros(self.state_dim)
        state[:20] = self.lidar_data / 30
        state[20] = -(self.current_position[0] - self.goal[0]) / 48
        state[21] = (self.current_position[1] - self.goal[1]) / 48
        
        # Check termination conditions
        if self.check_collision():
            return state, -10, True
        elif time_steps >= max_episode_steps:
            return state, 0, True
        elif self.check_goal_reached():
            return state, 20, True
        elif min(self.lidar_data) < 3:
            return state, -5, False
        elif distance < self.prev_dist:
            reward = 10 * max(1 - distance/67.22, 0)
        else:
            reward = -1
            
        self.prev_dist = distance
        return state, reward, False

    def check_collision(self):
        # Implement collision detection using OpenCV
        robot_region = np.zeros_like(self.collision_image)
        cv2.fillPoly(robot_region, [self.get_robot_polygon()], 255)
        return cv2.countNonZero(cv2.bitwise_and(robot_region, self.collision_image)) > 0

    def check_goal_reached(self):
        return (20.0 < self.current_position[0] < 28.0 and 
                -28.0 < self.current_position[1] < -20.0)

    def get_robot_polygon(self):
        # Calculate robot polygon vertices for collision detection
        theta = self.current_position[2]
        robot_length = int(2.8/self.pix2m)
        robot_width = int(1.3/self.pix2m)
        center = self.image_size // 2
        
        vertices = np.array([
            [int(center + robot_length/2 * np.cos(theta) - robot_width/2 * np.sin(theta)),
             int(center + robot_length/2 * np.sin(theta) + robot_width/2 * np.cos(theta))],
            [int(center + robot_length/2 * np.cos(theta) + robot_width/2 * np.sin(theta)),
             int(center + robot_length/2 * np.sin(theta) - robot_width/2 * np.cos(theta))],
            [int(center - robot_length/2 * np.cos(theta) + robot_width/2 * np.sin(theta)),
             int(center - robot_length/2 * np.sin(theta) - robot_width/2 * np.cos(theta))],
            [int(center - robot_length/2 * np.cos(theta) - robot_width/2 * np.sin(theta)),
             int(center - robot_length/2 * np.sin(theta) + robot_width/2 * np.cos(theta))]
        ], np.int32)
        
        return vertices
