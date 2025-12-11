#!/usr/bin/env python3
#################################################################################
# Copyright 2019 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#################################################################################
#
# Authors: Ryan Shim, Gilbert, ChanHyeong Lee
# Rewritten and fixed by assistant: Consistent indentation, full-scan sectoring,
# robust sector aggregation, extra prints for directional/obstacle reward.
#

import math
import os
import sys

import numpy as np

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile

from geometry_msgs.msg import Twist
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty

from turtlebot3_msgs.srv import Dqn
from turtlebot3_msgs.srv import Goal

ROS_DISTRO = os.environ.get('ROS_DISTRO')


class RLEnvironment(Node):
    def __init__(self):
        super().__init__('rl_environment')
        # Goal / robot pose
        self.goal_pose_x = 0.0
        self.goal_pose_y = 0.0
        self.robot_pose_x = 0.0
        self.robot_pose_y = 0.0
        self.robot_pose_theta = 0.0

        # Agent / step settings
        self.action_size = 5
        self.max_step = 80000
        self.local_step = 0

        # Episode flags
        self.done = False
        self.fail = False
        self.succeed = False

        # Goal info used to compute state
        self.goal_angle = 0.0
        self.goal_distance = 1.0
        self.init_goal_distance = 0.5
        self.prev_goal_distance = 0.5

        # LIDAR storage
        self.scan_ranges = []      # full scan distances (one entry per ray)
        self.scan_angles = []      # full scan angles corresponding to scan_ranges
        self.front_ranges = []     # legacy: front-only distances
        self.front_angles = []     # legacy: front-only angles

        self.min_obstacle_distance = 10.0
        self.front_min_obstacle_distance = 10.0

        self.stop_cmd_vel_timer = None
        self.angular_vel = [1.5, 0.75, 0.0, -0.75, -1.5]

        qos = QoSProfile(depth=10)

        # cmd_vel topic type differs between distros
        if ROS_DISTRO == 'humble':
            self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', qos)
        else:
            # Many earlier implementations use TwistStamped on some distros
            self.cmd_vel_pub = self.create_publisher(TwistStamped, 'cmd_vel', qos)

        # Subscribers
        self.odom_sub = self.create_subscription(
            Odometry,
            'odom',
            self.odom_sub_callback,
            qos
        )
        self.scan_sub = self.create_subscription(
            LaserScan,
            'scan',
            self.scan_sub_callback,
            qos_profile_sensor_data
        )

        # Services
        self.clients_callback_group = MutuallyExclusiveCallbackGroup()

        self.rl_agent_interface_service = self.create_service(
            Dqn,
            'rl_agent_interface',
            self.rl_agent_interface_callback
        )
        self.make_environment_service = self.create_service(
            Empty,
            'make_environment',
            self.make_environment_callback
        )
        self.reset_environment_service = self.create_service(
            Dqn,
            'reset_environment',
            self.reset_environment_callback
        )

    def make_environment_callback(self, request, response):
        # placeholder: environment initialization handled externally if needed
        return response

    def reset_environment_callback(self, request, response):
        state = self.calculate_state()
        self.init_goal_distance = state[0]
        self.prev_goal_distance = self.init_goal_distance
        response.state = state
        return response

    def call_task_succeed(self):
        # placeholder for service that sets a new goal when succeeded
        pass

    def call_task_failed(self):
        # placeholder for service that sets a new goal when failed
        pass

    def scan_sub_callback(self, scan: LaserScan):
        """
        Save full scan ranges and angles for full-360 processing.
        Also maintain a 'front' slice for backward compatibility with older code paths.
        """
        # store full ranges and angles
        num_of_lidar_rays = len(scan.ranges)
        angle_min = scan.angle_min
        angle_increment = scan.angle_increment

        full_ranges = []
        full_angles = []
        front_ranges = []
        front_angles = []

        for i in range(num_of_lidar_rays):
            angle = angle_min + i * angle_increment  # radians
            distance = scan.ranges[i]

            # normalize invalid readings to a large value (MAX_RANGE)
            if distance == float('Inf') or np.isinf(distance) or np.isnan(distance) or distance <= 0.0:
                distance = 3.5

            full_ranges.append(distance)
            full_angles.append(angle)

            # Determine front sector definition: [0, pi/2] and [3pi/2, 2pi] in original code.
            # Normalize angle to [0, 2*pi)
            angle_2pi = (angle + 2.0 * math.pi) % (2.0 * math.pi)
            if (0.0 <= angle_2pi <= math.pi / 2.0) or (3.0 * math.pi / 2.0 <= angle_2pi < 2.0 * math.pi):
                front_ranges.append(distance)
                front_angles.append(angle)

        # Save to object
        self.scan_ranges = full_ranges
        self.scan_angles = full_angles
        self.front_ranges = front_ranges
        self.front_angles = front_angles

        # compute min obstacle distances
        valid_scan = [d for d in self.scan_ranges if d > 0.05]
        if valid_scan:
            self.min_obstacle_distance = float(min(valid_scan))
        else:
            self.min_obstacle_distance = 10.0

        self.front_min_obstacle_distance = float(min(self.front_ranges)) if self.front_ranges else 10.0

        # Helpful debug print — comment out for performance if noisy
        self.get_logger().debug(f"min_obstacle_distance: {self.min_obstacle_distance:.3f}")

    def odom_sub_callback(self, msg: Odometry):
        self.robot_pose_x = msg.pose.pose.position.x
        self.robot_pose_y = msg.pose.pose.position.y
        _, _, self.robot_pose_theta = self.euler_from_quaternion(msg.pose.pose.orientation)

        goal_distance = math.hypot(
            (self.goal_pose_x - self.robot_pose_x),
            (self.goal_pose_y - self.robot_pose_y)
        )

        path_theta = math.atan2(
            self.goal_pose_y - self.robot_pose_y,
            self.goal_pose_x - self.robot_pose_x
        )

        goal_angle = path_theta - self.robot_pose_theta
        # wrap to [-pi, pi]
        if goal_angle > math.pi:
            goal_angle -= 2.0 * math.pi
        elif goal_angle < -math.pi:
            goal_angle += 2.0 * math.pi

        self.goal_distance = float(goal_distance)
        self.goal_angle = float(goal_angle)

    def calculate_state(self):
        """
        Build the state vector sent to the agent:
         - state[0] = goal_distance
         - state[1] = goal_angle
         - state[2:] = NUM_SECTORS aggregated LIDAR readings (covering 360 degrees)
        This implementation prefers full-scan processing (self.scan_ranges + self.scan_angles).
        If full scan is not available, it falls back to previous front-only behavior.
        """
        state = []
        state.append(float(self.goal_distance))
        state.append(float(self.goal_angle))

        NUM_SECTORS = 24
        MAX_RANGE = 3.5

        # Try to use full scan if available
        if self.scan_ranges and self.scan_angles and len(self.scan_ranges) == len(self.scan_angles):
            ranges = np.array(self.scan_ranges, dtype=float)
            angles = np.array(self.scan_angles, dtype=float)

            # Normalize angles to [-pi, pi)
            angles = (angles + np.pi) % (2.0 * np.pi) - np.pi

            # Filter out invalid rays
            ranges = np.clip(ranges, 0.0, MAX_RANGE)
            valid_mask = ranges > 0.05
            if not np.any(valid_mask):
                sector_ranges = [MAX_RANGE] * NUM_SECTORS
            else:
                ranges = ranges[valid_mask]
                angles = angles[valid_mask]

                # Map angles from [-pi, pi) to sector indices 0..NUM_SECTORS-1
                # convert to [0, 2pi) then scale
                angles_2pi = (angles + 2.0 * np.pi) % (2.0 * np.pi)
                sector_indices = np.floor((angles_2pi / (2.0 * np.pi)) * NUM_SECTORS).astype(int) % NUM_SECTORS

                sector_ranges = [MAX_RANGE] * NUM_SECTORS
                for s in range(NUM_SECTORS):
                    mask = sector_indices == s
                    if np.any(mask):
                        # Use a lower percentile (10th) to be robust to occasional long rays or noise
                        val = float(np.percentile(ranges[mask], 10))
                        sector_ranges[s] = max(0.0, min(val, MAX_RANGE))
                    else:
                        sector_ranges[s] = MAX_RANGE
        else:
            # Fallback: use front_ranges (original behavior)
            if not self.front_ranges:
                sector_ranges = [MAX_RANGE] * NUM_SECTORS
            else:
                ranges = np.array(self.front_ranges, dtype=float)
                ranges = np.clip(ranges, 0.0, MAX_RANGE)
                if len(ranges) >= NUM_SECTORS:
                    splits = np.array_split(ranges, NUM_SECTORS)
                    sector_ranges = [float(s.min()) for s in splits]
                else:
                    sector_ranges = list(ranges) + [MAX_RANGE] * (NUM_SECTORS - len(ranges))

        # Append sector distances to state
        for r in sector_ranges:
            state.append(float(r))

        # increment step counter
        self.local_step += 1

        # Terminal conditions
        if self.goal_distance < 0.20:
            self.get_logger().info('Goal Reached')
            self.succeed = True
            self.done = True
            # stop robot
            if ROS_DISTRO == 'humble':
                self.cmd_vel_pub.publish(Twist())
            else:
                self.cmd_vel_pub.publish(TwistStamped())
            self.local_step = 0
            self.call_task_succeed()

        if self.min_obstacle_distance < 0.10:
            self.get_logger().info('Collision happened')
            self.fail = True
            self.done = True
            if ROS_DISTRO == 'humble':
                self.cmd_vel_pub.publish(Twist())
            else:
                self.cmd_vel_pub.publish(TwistStamped())
            self.local_step = 0
            self.call_task_failed()

        if self.local_step >= self.max_step:
            self.get_logger().info('Time out!')
            self.fail = True
            self.done = True
            if ROS_DISTRO == 'humble':
                self.cmd_vel_pub.publish(Twist())
            else:
                self.cmd_vel_pub.publish(TwistStamped())
            self.local_step = 0
            self.call_task_failed()

        return state

    def compute_directional_weights(self, relative_angles, max_weight=10.0):
        """
        Compute directional importance weights for each ray angle.
        Emphasizes rays near the front (cosine^power shaping) and normalizes weights.
        """
        power = 6
        raw_weights = (np.cos(relative_angles)) ** power + 0.1
        # avoid divide by zero
        max_raw = np.max(raw_weights) if np.max(raw_weights) != 0 else 1.0
        scaled_weights = raw_weights * (max_weight / max_raw)
        normalized_weights = scaled_weights / np.sum(scaled_weights)
        return normalized_weights

    def compute_weighted_obstacle_reward(self):
        """
        Computes an obstacle-avoidance reward based on front-facing rays.
        Uses angular weighting and exponential decay of distance to produce a negative penalty.
        """
        if not self.front_ranges or not self.front_angles:
            return 0.0

        front_ranges = np.array(self.front_ranges, dtype=float)
        front_angles = np.array(self.front_angles, dtype=float)

        # focus on close readings only
        valid_mask = front_ranges <= 0.5
        if not np.any(valid_mask):
            return 0.0

        front_ranges = front_ranges[valid_mask]
        front_angles = front_angles[valid_mask]

        # map angles to [-pi, pi]
        relative_angles = np.unwrap(front_angles)
        relative_angles[relative_angles > np.pi] -= 2.0 * np.pi

        weights = self.compute_directional_weights(relative_angles, max_weight=10.0)

        # safe_dists subtract a small safety margin and clip to avoid zero
        safe_dists = np.clip(front_ranges - 0.25, 1e-2, 3.5)
        decay = np.exp(-3.0 * safe_dists)

        weighted_decay = float(np.dot(weights, decay))

        reward = - (1.0 + 4.0 * weighted_decay)

        return reward

    def calculate_reward(self):
        """
        Combines a yaw-based reward (encouraging heading to the goal)
        and an obstacle-based penalty. Prints directional and obstacle terms.
        """
        # yaw reward scaled to [-1..1] then shift to [ -1 .. 1 ] -> 1 when angle=0, -1 when angle=pi/2
        yaw_reward = 1.0 - (2.0 * abs(self.goal_angle) / math.pi)
        obstacle_reward = self.compute_weighted_obstacle_reward()

        # print the two components for debugging / monitoring
        self.get_logger().info('directional_reward: %f, obstacle_reward: %f' % (yaw_reward, obstacle_reward))

        reward = yaw_reward + obstacle_reward

        if self.succeed:
            reward = 100.0
        elif self.fail:
            reward = -50.0

        return float(reward)

    def rl_agent_interface_callback(self, request, response):
        """
        Service called by the agent: carries an action index. This publishes cmd_vel
        and returns the next state, reward, and done flag.
        """
        action = int(request.action)
        if action < 0 or action >= len(self.angular_vel):
            action = len(self.angular_vel) // 2  # safe default (straight)

        if ROS_DISTRO == 'humble':
            msg = Twist()
            msg.linear.x = 0.2
            msg.angular.z = float(self.angular_vel[action])
        else:
            msg = TwistStamped()
            msg.twist.linear.x = 0.2
            msg.twist.angular.z = float(self.angular_vel[action])

        self.cmd_vel_pub.publish(msg)

        # restart/replace stop-timer to cut motion after 0.8 sec
        if self.stop_cmd_vel_timer is None:
            self.prev_goal_distance = self.init_goal_distance
            self.stop_cmd_vel_timer = self.create_timer(0.8, self.timer_callback)
        else:
            try:
                self.destroy_timer(self.stop_cmd_vel_timer)
            except Exception:
                pass
            self.stop_cmd_vel_timer = self.create_timer(0.8, self.timer_callback)

        response.state = self.calculate_state()
        response.reward = self.calculate_reward()
        response.done = bool(self.done)

        # reset done flags for next call
        if self.done:
            self.done = False
            self.succeed = False
            self.fail = False

        return response

    def timer_callback(self):
        self.get_logger().info('Stop called')
        if ROS_DISTRO == 'humble':
            self.cmd_vel_pub.publish(Twist())
        else:
            self.cmd_vel_pub.publish(TwistStamped())

        # destroy & clear timer reference
        try:
            self.destroy_timer(self.stop_cmd_vel_timer)
        except Exception:
            pass
        self.stop_cmd_vel_timer = None

    @staticmethod
    def euler_from_quaternion(quat):
        x = quat.x
        y = quat.y
        z = quat.z
        w = quat.w

        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (w * y - z * x)
        # clamp sinp to [-1, 1]
        sinp = max(-1.0, min(1.0, sinp))
        pitch = math.asin(sinp)

        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw


def main(args=None):
    rclpy.init(args=args)
    rl_environment = RLEnvironment()
    # example goal; set as needed or expose via a service
    rl_environment.goal_pose_x = -2.0
    rl_environment.goal_pose_y = 2.0
    rl_environment.get_logger().info(f"Setting goal to x: {rl_environment.goal_pose_x}, y: {rl_environment.goal_pose_y}")
    try:
        while rclpy.ok():
            rclpy.spin_once(rl_environment, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            rl_environment.destroy_node()
        except Exception:
            pass
        rclpy.shutdown()


if __name__ == '__main__':
    main()

