

import collections
import datetime
import json
import math
import os
import random
import sys
import time

import numpy
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from std_srvs.srv import Empty
import tensorflow
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env

from turtlebot3_msgs.srv import Dqn

import scipy.signal

tensorflow.config.set_visible_devices([], 'GPU')

LOGGING = True
current_time = datetime.datetime.now().strftime('[%mm%dd-%H:%M]')


class DQNMetric(tensorflow.keras.metrics.Metric):

    def __init__(self, name='dqn_metric'):
        super(DQNMetric, self).__init__(name=name)
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


class DQNAgent(Node):

    def __init__(self, stage_num, max_training_episodes):
        super().__init__('dqn_agent')

        self.stage = int(stage_num)
        self.train_mode = True
        self.state_size = 26
        self.action_size = 5
        self.max_training_episodes = int(max_training_episodes)

        self.done = False
        self.succeed = False
        self.fail = False

        self.discount_factor = 0.99
        self.learning_rate = 0.0007
        self.epsilon = 1.0
        self.step_counter = 0
        self.epsilon_decay = 6000 * self.stage
        self.epsilon_min = 0.05
        self.batch_size = 128

        self.replay_memory = collections.deque(maxlen=500000)
        self.min_replay_memory_size = 5000

        self.model = self.create_qnetwork()
        self.target_model = self.create_qnetwork()
        self.update_target_model()
        self.update_target_after = 5000
        self.target_update_after_counter = 0

        self.load_model = False
        self.load_episode = 0
        self.model_dir_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
            'saved_model_double_dqn'
        )
        self.model_path = os.path.join(
            self.model_dir_path,
            'stage' + str(self.stage) + '_episode' + str(self.load_episode) + '.h5'
        )

        if self.load_model:
            self.model.set_weights(load_model(self.model_path).get_weights())
            with open(os.path.join(
                    self.model_dir_path,
                    'stage' + str(self.stage) + '_episode' + str(self.load_episode) + '.json'
            )) as outfile:
                param = json.load(outfile)
                self.epsilon = param.get('epsilon')
                self.step_counter = param.get('step_counter')

        if LOGGING:
            tensorboard_file_name = current_time + ' dqn_stage' + str(self.stage) + '_reward'
            home_dir = os.path.expanduser('~')
            dqn_reward_log_dir = os.path.join(
                home_dir, 'turtlebot3_dqn_logs', 'gradient_tape', tensorboard_file_name
            )
            self.dqn_reward_writer = tensorflow.summary.create_file_writer(dqn_reward_log_dir)
            self.dqn_reward_metric = DQNMetric()

        self.rl_agent_interface_client = self.create_client(Dqn, 'rl_agent_interface')
        self.make_environment_client = self.create_client(Empty, 'make_environment')
        self.reset_environment_client = self.create_client(Dqn, 'reset_environment')

        self.action_pub = self.create_publisher(Float32MultiArray, '/get_action', 10)
        self.result_pub = self.create_publisher(Float32MultiArray, 'result', 10)

        self.process()

    def process(self):
        self.env_make()
        time.sleep(1.0)

        env = TurtlebotGymEnv(self)
        # check_env(env)

        max_episode_steps = 800
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)

        print("Creating DQN model...")
        # Instantiate the agent

        #####################
        # model = DQN("MlpPolicy", env, verbose=1)

        load_from_checkpoint = False
        checkpoint_file = os.path.join(
            self.model_dir_path,
            f"stage{self.stage}_episode{self.load_episode}.zip"
        )

        # if we want to load from episode
        if load_from_checkpoint and os.path.exists(checkpoint_file):
            print(f"Loading model from checkpoint: {checkpoint_file}")
            model = DQN.load(checkpoint_file, env=env, tensorboard_log="~/turtlebot3_dqn_logs/sb3/")

        else:
            model = DQN("MlpPolicy", env, verbose=1, tensorboard_log="~/turtlebot3_dqn_logs/sb3/")

        #######################

        # Train the agent and display a progress bar
        print("Training started...")
        total_steps = self.max_training_episodes * max_episode_steps  # assuming max_episode_steps per episode
        model.learn(total_timesteps=int(total_steps), progress_bar=True)

        self.model_path = os.path.join(
            self.model_dir_path,
            'stage' + str(self.stage) + '_episode' + str(self.max_training_episodes))
        # Save the agent
        model.save(self.model_path)

    def env_make(self):
        while not self.make_environment_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn(
                'Environment make client failed to connect to the server, try again ...'
            )

        self.make_environment_client.call_async(Empty.Request())

    def reset_environment(self):
        while not self.reset_environment_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn(
                'Reset environment client failed to connect to the server, try again ...'
            )

        future = self.reset_environment_client.call_async(Dqn.Request())

        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            state = future.result().state
            state = numpy.reshape(numpy.asarray(state), [1, self.state_size])
        else:
            self.get_logger().error(
                'Exception while calling service: {0}'.format(future.exception()))

        return state

    def get_action(self, state):
        if self.train_mode:
            self.step_counter += 1
            self.epsilon = self.epsilon_min + (1.0 - self.epsilon_min) * math.exp(
                -1.0 * self.step_counter / self.epsilon_decay)
            lucky = random.random()
            if lucky > (1 - self.epsilon):
                result = random.randint(0, self.action_size - 1)
            else:
                result = numpy.argmax(self.model.predict(state))
        else:
            result = numpy.argmax(self.model.predict(state))

        return result

    def step(self, action):
        req = Dqn.Request()
        req.action = action

        while not self.rl_agent_interface_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('rl_agent interface service not available, waiting again...')

        future = self.rl_agent_interface_client.call_async(req)

        rclpy.spin_until_future_complete(self, future)

        if future.result() is not None:
            next_state = future.result().state
            next_state = numpy.reshape(numpy.asarray(next_state), [1, self.state_size])
            reward = future.result().reward
            done = future.result().done
        else:
            self.get_logger().error(
                'Exception while calling service: {0}'.format(future.exception()))

        return next_state, reward, done

    def create_qnetwork(self):
        model = Sequential()
        model.add(Input(shape=(self.state_size,)))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=self.learning_rate))
        model.summary()

        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
        self.target_update_after_counter = 0
        print('*Target model updated*')

    def append_sample(self, transition):
        self.replay_memory.append(transition)

    def train_model(self, terminal):
        if len(self.replay_memory) < self.min_replay_memory_size:
            return
        data_in_mini_batch = random.sample(self.replay_memory, self.batch_size)

        current_states = numpy.array([transition[0] for transition in data_in_mini_batch])
        current_states = current_states.squeeze()
        current_qvalues_list = self.model.predict(current_states)

        next_states = numpy.array([transition[3] for transition in data_in_mini_batch])
        next_states = next_states.squeeze()
        next_qvalues_list = self.target_model.predict(next_states)

        x_train = []
        y_train = []

        for index, (current_state, action, reward, _, done) in enumerate(data_in_mini_batch):
            current_q_values = current_qvalues_list[index]

            if not done:
                future_reward = numpy.max(next_qvalues_list[index])
                desired_q = reward + self.discount_factor * future_reward
            else:
                desired_q = reward

            current_q_values[action] = desired_q
            x_train.append(current_state)
            y_train.append(current_q_values)

        x_train = numpy.array(x_train)
        y_train = numpy.array(y_train)
        x_train = numpy.reshape(x_train, [len(data_in_mini_batch), self.state_size])
        y_train = numpy.reshape(y_train, [len(data_in_mini_batch), self.action_size])

        self.model.fit(
            tensorflow.convert_to_tensor(x_train, tensorflow.float32),
            tensorflow.convert_to_tensor(y_train, tensorflow.float32),
            batch_size=self.batch_size, verbose=0
        )
        self.target_update_after_counter += 1

        if self.target_update_after_counter > self.update_target_after and terminal:
            self.update_target_model()


# todo: create a gym class for turtlebot3. Pass DQNAgent as a constructor parameter to the gym class.
# Define methods in the gym class by delegating to DQNAgent methods.


import gymnasium as gym
from gymnasium import spaces
import numpy as np


class TurtlebotGymEnv(gym.Env):
    def __init__(self, dqnAgent: DQNAgent):
        print("Initializing TurtlebotGymEnv...")
        super().__init__()

        # Define action space
        self.action_space = spaces.Discrete(dqnAgent.action_size)

        # Define observation space (1D array)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(dqnAgent.state_size,), dtype=float)
        self.dqnAgent = dqnAgent

    def reset(self, seed=None, options=None):
        # print("Resetting environment...")
        observation = self.dqnAgent.reset_environment()
        # Convert 2D observation to 1D for gym compatibility
        observation = observation.flatten()
        time.sleep(1.0)
        return observation, {}

    def step(self, action):
        # print(f"Taking action: {action}")
        (observation, reward, done) = self.dqnAgent.step(int(action))
        # Convert 2D observation to 1D for gym compatibility
        observation = observation.flatten()

        terminated = bool(done)  # Episode ends when goal reached
        truncated = bool(done)  # For time limits

        time.sleep(0.01)

        return observation, float(reward), terminated, truncated, {}

    def render(self):  # Optional: For visualization
        pass

    def close(self):  # Optional: Clean up
        pass


def main(args=None):
    if args is None:
        args = sys.argv
    stage_num = args[1] if len(args) > 1 else '1'
    max_training_episodes = args[2] if len(args) > 2 else '1000'
    rclpy.init(args=args)

    dqn_agent = DQNAgent(stage_num, max_training_episodes)
    rclpy.spin(dqn_agent)

    dqn_agent.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
