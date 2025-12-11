#!/usr/bin/env python3
#################################################################################
# Copyright ...
#################################################################################

import collections
import datetime
import json
import math
import os
import random
import sys
import time

import gymnasium as gym
from gymnasium import spaces
import numpy as np

import numpy
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from std_srvs.srv import Empty
import tensorflow
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.optimizers import Adam

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from turtlebot3_msgs.srv import Dqn

# Disable GPU
tensorflow.config.set_visible_devices([], 'GPU')

LOGGING = True
current_time = datetime.datetime.now().strftime('[%m-%d %H:%M]')


# --------------------- METRIC CLASS ------------------------------------
class DQNMetric(tensorflow.keras.metrics.Metric):
    def __init__(self, name='dqn_metric'):
        super().__init__(name=name)
        self.loss = self.add_weight(name='loss', initializer='zeros')
        self.episode_step = self.add_weight(name='step', initializer='zeros')

    def update_state(self, y_true, y_pred=0, sample_weight=None):
        self.loss.assign_add(y_true)
        self.episode_step.assign_add(1)

    def result(self):
        return self.loss / self.episode_step

    def reset_states(self):
        self.loss.assign(0)
        self.episode_step.assign(0)


# --------------------- TENSORBOARD CALLBACK ------------------------------------
class TensorboardRewardCallback(BaseCallback):
    """
    Custom callback to write per-episode reward to TensorBoard
    """
    def __init__(self, agent, verbose=0):
        super().__init__(verbose)
        self.agent = agent

    def _on_step(self):
        # SB3 injects 'infos' list: one per environment
        for info in self.locals.get("infos", []):
            if "episode" in info:
                reward = info["episode"]["r"]

                # Write to TensorBoard live
                with self.agent.dqn_reward_writer.as_default():
                    tensorflow.summary.scalar(
                        'episode_reward',
                        reward,
                        step=self.num_timesteps,
                    )
        return True


# --------------------- MAIN AGENT (ROS2 Node + PPO Wrapper) ---------------------
class DQNAgent(Node):
    def __init__(self, stage_num, max_training_episodes):
        super().__init__('dqn_agent')

        self.stage = int(stage_num)
        self.max_training_episodes = int(max_training_episodes)

        self.state_size = 26
        self.action_size = 5

        # # Directory to save PPO models
        # self.model_dir_path = os.path.join(
        #     os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
        #     'saved_model_ppo'
        # )
        # Fixed directory for storing PPO trained models
        self.model_dir_path = os.path.expanduser('~/turtlebot3_models/ppo')
        os.makedirs(self.model_dir_path, exist_ok=True)

        print("\n[INFO] PPO models will be saved to:", self.model_dir_path, "\n")

        os.makedirs(self.model_dir_path, exist_ok=True)

        # TensorBoard logging initialization
        if LOGGING:
            tensorboard_file_name = current_time + f'_ppo_stage{self.stage}'
            home_dir = os.path.expanduser('~')
            dqn_reward_log_dir = os.path.join(
                home_dir, 'turtlebot3_new_ppo_logs', 'tensorboard', tensorboard_file_name
            )
            self.dqn_reward_writer = tensorflow.summary.create_file_writer(dqn_reward_log_dir)
            self.dqn_reward_metric = DQNMetric()

        # ROS2 service clients
        self.rl_agent_interface_client = self.create_client(Dqn, 'rl_agent_interface')
        self.make_environment_client = self.create_client(Empty, 'make_environment')
        self.reset_environment_client = self.create_client(Dqn, 'reset_environment')

        # ROS2 publishers
        self.action_pub = self.create_publisher(Float32MultiArray, '/get_action', 10)
        self.result_pub = self.create_publisher(Float32MultiArray, 'result', 10)

        # Begin training gradient_tape/[12m08d-13:39] dqn_stage1_reward
        self.process()

    # ------------------ PPO TRAINING LOOP ------------------
    def process(self):
        self.env_make()
        time.sleep(1.0)

        env = TurtlebotGymEnv(self)
        max_episode_steps = 800

        print("Creating PPO model...")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log="ppo_logs_new_shaurya"
        )

        print("Training started...")
        total_steps = self.max_training_episodes * max_episode_steps

        callback = TensorboardRewardCallback(self)

        # Train PPO with TensorBoard callback
        model.learn(
            total_timesteps=int(total_steps),
            progress_bar=True,
            callback=callback
        )

        save_path = os.path.join(
            self.model_dir_path,
            f'stage{self.stage}_episode{self.max_training_episodes}'
        )
        model.save(save_path)
        print(f"Model saved to {save_path}")

    # ------------------ ROS SERVICE WRAPPERS ------------------
    def env_make(self):
        while not self.make_environment_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("Waiting for make_environment service...")
        self.make_environment_client.call_async(Empty.Request())

    def reset_environment(self):
        while not self.reset_environment_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("Waiting for reset_environment...")

        future = self.reset_environment_client.call_async(Dqn.Request())
        rclpy.spin_until_future_complete(self, future)

        if future.result():
            state = numpy.reshape(numpy.asarray(future.result().state), [1, self.state_size])
            return state
        else:
            self.get_logger().error(f"reset_environment failed: {future.exception()}")
            return numpy.zeros((1, self.state_size))

    def step(self, action):
        req = Dqn.Request()
        req.action = int(action)

        while not self.rl_agent_interface_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('Waiting for rl_agent_interface...')

        future = self.rl_agent_interface_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        if future.result():
            next_state = numpy.reshape(numpy.asarray(future.result().state), [1, self.state_size])
            return next_state, future.result().reward, future.result().done
        else:
            self.get_logger().error(f"step() failed: {future.exception()}")
            return numpy.zeros((1, self.state_size)), 0.0, True


# --------------------- GYM ENV WRAPPER ---------------------
class TurtlebotGymEnv(gym.Env):
    def __init__(self, agent: DQNAgent):
        super().__init__()
        self.agent = agent
        self.action_space = spaces.Discrete(agent.action_size)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(1, agent.state_size),
            dtype=float
        )
        self.episode_reward = 0

    def reset(self, seed=None, options=None):
        state = self.agent.reset_environment()
        self.episode_reward = 0
        time.sleep(0.5)
        return state, {}

    def step(self, action):
        obs, reward, done = self.agent.step(action)
        self.episode_reward += reward

        info = {}
        if done:
            info["episode"] = {"r": self.episode_reward}

        time.sleep(0.01)

        return obs, reward, done, done, info


# --------------------- MAIN ---------------------
def main(args=None):
    if args is None:
        args = sys.argv

    stage_num = args[1] if len(args) > 1 else '1'
    max_training_episodes = args[2] if len(args) > 2 else '1000'

    rclpy.init(args=args)

    agent = DQNAgent(stage_num, max_training_episodes)
    rclpy.spin(agent)

    agent.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

