import rclpy
from rclpy.node import Node
import tf2_ros
import geometry_msgs.msg
from geometry_msgs.msg import Twist, TransformStamped
import math
import numpy as np
import time
import copy
from .robot_environment import RobotEnvironment
from .q_learning_agent import QLearningAgent
from sensor_msgs.msg import LaserScan


lidar_data = np.zeros(20)
class LidarSubscriber(Node):
    def __init__(self):
        super().__init__('lidar_subscriber')
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.listener_callback,
            10)
        self.lidar_data_prev_step = np.zeros(20)

    def listener_callback(self, data):
        global lidar_data
        # Process the LaserScan data into 20 segments
        for i in range(20):
            value = data.ranges[18*i:18*i + 18]  # Divide 360 degrees into 20 segments
            lidar_data[i] = np.min(value)  # Use minimum value in segment for safety
            if lidar_data[i] <= 0 or np.isinf(lidar_data[i]):
                lidar_data[i] = self.lidar_data_prev_step[i]
        self.lidar_data_prev_step = copy.copy(lidar_data)

class MoveTowardsObject(Node):
    def __init__(self):
        super().__init__('move_towards_object')

        # Create publishers
        self.velocity_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.teleport_publisher = self.create_publisher(TransformStamped, '/teleport', 10)

        # Initialize transform buffer and listener
        self.tf_buffer = tf2_ros.Buffer(cache_time=rclpy.duration.Duration(seconds=60.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Control loop timer
        self.timer = self.create_timer(0.1, self.control_loop)

        # Initialize RL components with specific dimensions
        self.environment = RobotEnvironment()
        self.agent = QLearningAgent(state_dim=22, action_dim=2)
        
        # Initialize state
        self.current_state = self.environment.reset()
        
        # Training parameters
        self.episode_count = 0
        self.max_episodes = 1000
        self.episode_steps = 0
        self.max_steps_per_episode = 500
        
        # Performance tracking
        self.cumulative_reward = 0.0
        
        # Debug parameters
        self.debug = True

    def publish_transform(self):
        # Publish current robot position as transform
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'map'
        t.child_frame_id = 'base_link'
        
        current_pos = self.environment.current_position
        t.transform.translation.x = float(current_pos[0])
        t.transform.translation.y = float(current_pos[1])
        t.transform.translation.z = float(current_pos[2])
        
        # Set rotation (simplified - only yaw)
        t.transform.rotation.w = 1.0
        self.tf_broadcaster.sendTransform(t)

    def teleport_to_start(self):
        teleport = TransformStamped()
        teleport.header.stamp = self.get_clock().now().to_msg()
        teleport.header.frame_id = 'map'
        teleport.child_frame_id = 'base_link'
        
        # Set position to robot start position
        start_pos = self.environment.robot_start_position
        teleport.transform.translation.x = float(start_pos[0])
        teleport.transform.translation.y = float(start_pos[1])
        teleport.transform.translation.z = float(start_pos[2])
        
        # Set rotation to default
        teleport.transform.rotation.w = 1.0
        
        # Publish teleport transform
        self.teleport_publisher.publish(teleport)
        
        # Add delay to ensure teleport completes
        time.sleep(0.5)  # Add small delay
        
        # Stop the robot
        stop_cmd = Twist()
        self.velocity_publisher.publish(stop_cmd)


        self.get_logger().info(f'Teleporting to: {self.environment.robot_start_position}')



    def control_loop(self):
        try:
            # Publish current robot position
            self.publish_transform()

            # Get action from RL agent
            action = self.agent.get_action(self.current_state)

            # Execute action and get new state
            next_state, reward, done = self.environment.step(action)
            
            # Update cumulative reward
            self.cumulative_reward += reward

            # Create and publish velocity command
            twist = Twist()
            twist.linear.x = float(action[0])  # Forward/backward
            twist.angular.z = float(action[1])  # Rotation
            self.velocity_publisher.publish(twist)

            # Update Q-table
            self.agent.update_q_table(self.current_state, action, reward, next_state)
            
            # Update current state
            self.current_state = next_state
            
            # Increment step counter
            self.episode_steps += 1

            if self.debug:
                self.get_logger().info(
                    f'Step: {self.episode_steps}, ' +
                    f'State: [{self.current_state[0]:.2f}, {self.current_state[1]:.2f}], ' +
                    f'Action: [{action[0]:.2f}, {action[1]:.2f}], ' +
                    f'Reward: {reward:.2f}'
                )

            # Check if episode should end
            if done or self.episode_steps >= self.max_steps_per_episode:
                self.episode_count += 1
                self.get_logger().info(
                    f'Episode {self.episode_count} completed with ' +
                    f'cumulative reward: {self.cumulative_reward:.2f}'
                )
                self.agent.epsilon = max(0.01, self.agent.epsilon * 0.995)
                self.get_logger().info(f'Updated epsilon: {self.agent.epsilon:.4f}')
                # Teleport robot back to start
                self.teleport_to_start()
                
                # Reset for next episode
                self.current_state = self.environment.reset()
                self.episode_steps = 0
                self.cumulative_reward = 0.0

                # Check if training is complete
                if self.episode_count >= self.max_episodes:
                    self.get_logger().info('Training completed!')
                    self.timer.cancel()

        except Exception as e:
            self.get_logger().error(f'Error in control loop: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    try:
        lidar_node = LidarSubscriber()
        move_node = MoveTowardsObject()
        executor = rclpy.executors.MultiThreadedExecutor()
        executor.add_node(lidar_node)
        executor.add_node(move_node)

        try:
            executor.spin()
        finally:
            executor.shutdown()
            lidar_node.destroy_node()
            move_node.destroy_node()

    except Exception as e:
        print(f"Error in main: {str(e)}")
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
