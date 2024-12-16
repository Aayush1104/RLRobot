import rclpy
from rclpy.node import Node
import tf2_ros
import geometry_msgs.msg
from geometry_msgs.msg import Twist, TransformStamped
import math
import numpy as np
import time
from .robot_environment import RobotEnvironment
from .q_learning_agent import QLearningAgent

class MoveTowardsObject(Node):
    def __init__(self):
        super().__init__('move_towards_object')
        
        # Publishers
        self.velocity_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.teleport_publisher = self.create_publisher(TransformStamped, '/teleport', 10)
        
        # Transform handling
        self.tf_buffer = tf2_ros.Buffer(cache_time=rclpy.duration.Duration(seconds=60.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Initialize environment and agent
        self.environment = RobotEnvironment()
        self.agent = QLearningAgent(state_dim=22, action_dim=1)  # Modified dimensions
        
        # Training parameters
        self.max_episode_steps = 120
        self.episode_steps = 0
        self.total_steps = 0
        self.max_total_steps = 1000001
        self.updates = 0
        self.evaluation_interval = 10
        
        # Initialize state
        self.current_state = self.environment.reset()
        
        # Control loop
        self.timer = self.create_timer(0.05, self.control_loop)
        
        # Performance tracking
        self.episode_reward = 0.0
        self.prev_distance = 0.0

    def control_loop(self):
        try:
            # Get action from agent
            action = self.agent.get_action(self.current_state)
            
            # Execute action
            next_state, reward, done = self.environment.step(action, self.episode_steps, self.max_episode_steps)
            
            # Update agent
            self.agent.update_q_table(self.current_state, action, reward, next_state)
            
            # Publish velocity command
            twist = Twist()
            twist.linear.x = self.environment.vel  # Constant velocity
            twist.angular.z = 2 * action[0]  # Steering from action
            self.velocity_publisher.publish(twist)
            
            # Update episode tracking
            self.episode_steps += 1
            self.total_steps += 1
            self.episode_reward += reward
            self.current_state = next_state

            # Check episode completion
            if done or self.episode_steps >= self.max_episode_steps:
                self.get_logger().info(f"Episode completed: Steps={self.episode_steps}, Reward={self.episode_reward:.2f}")
                self.episode_steps = 0
                self.episode_reward = 0
                self.current_state = self.environment.reset()
                self.teleport_to_start()

            # Check training completion
            if self.total_steps >= self.max_total_steps:
                self.get_logger().info("Training completed!")
                self.timer.cancel()

        except Exception as e:
            self.get_logger().error(f'Control loop error: {str(e)}')

    def teleport_to_start(self):
        teleport = TransformStamped()
        teleport.header.stamp = self.get_clock().now().to_msg()
        teleport.header.frame_id = 'map'
        teleport.child_frame_id = 'base_link'
        
        start_pos = self.environment.robot_start_position
        teleport.transform.translation.x = float(start_pos[0])
        teleport.transform.translation.y = float(start_pos[1])
        teleport.transform.translation.z = float(start_pos[2])
        teleport.transform.rotation.w = 1.0
        
        self.teleport_publisher.publish(teleport)
        time.sleep(0.5)
        
        stop_cmd = Twist()
        self.velocity_publisher.publish(stop_cmd)
