'''
Added new metrics to track:
'collisions_avoided': 0,  # New field to track collisions avoided
'near_misses': 0,         # New field to track near misses

Added tracking variables:
prev_min_distance = 10.0  # Initialize with a large value

Added collision avoidance logic:
# Extract minimum obstacle distance from state
# State format: [goal_distance, goal_angle, 24 lidar readings]
if len(next_state) >= 26:
    # Get the minimum value from the lidar readings (indices 2-25)
    current_min_distance = min(next_state[2:26])

    # Track near misses (when robot comes very close to obstacles)
    if current_min_distance < 0.25:  # Near miss threshold
        cur_results["near_misses"] += 1

    # Track collisions avoided
    # If robot was in danger zone but moved away without colliding
    if prev_min_distance < 0.2 and current_min_distance > 0.3:
        cur_results["collisions_avoided"] += 1

    # Update previous distance for next iteration
    prev_min_distance = current_min_distance

'''

import collections
import os
import sys
import time

import numpy
import rclpy
from rclpy.node import Node
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop

from turtlebot3_msgs.srv import Dqn

import json


class DQNTest(Node):

    def __init__(self, stage, load_episode):
        super().__init__('dqn_test')

        self.stage = int(stage)
        self.load_episode = int(load_episode)

        self.state_size = 26
        self.action_size = 5

        self.total_test = 100

        self.results = []

        self.memory = collections.deque(maxlen=1000000)

        self.model = self.build_model()
        model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
            'saved_model',
            f'stage{self.stage}_episode{self.load_episode}.h5'
        )

        loaded_model = load_model(
            model_path, compile=False, custom_objects={'mse': MeanSquaredError()}
        )
        self.model.set_weights(loaded_model.get_weights())

        self.rl_agent_interface_client = self.create_client(Dqn, 'rl_agent_interface')
        self.reset_environment_client = self.create_client(Dqn, 'reset_environment')

        self.run_test()

    def build_model(self):
        model = Sequential()
        # Add explicit Input layer to avoid warning
        model.add(Input(shape=(self.state_size,)))
        model.add(Dense(512, activation='relu', kernel_initializer='lecun_uniform'))
        model.add(Dense(256, activation='relu', kernel_initializer='lecun_uniform'))
        model.add(Dense(128, activation='relu', kernel_initializer='lecun_uniform'))
        model.add(Dense(self.action_size, activation='linear', kernel_initializer='lecun_uniform'))
        model.compile(loss=MeanSquaredError(), optimizer=RMSprop(learning_rate=0.00025))
        return model

    def get_action(self, state):
        state = numpy.asarray(state)
        # Ensure state has correct shape before prediction
        if len(state) != self.state_size:
            # Pad state if it's too short
            if len(state) < self.state_size:
                padding = [3.5] * (self.state_size - len(state))
                state = numpy.concatenate([state, padding])
            # Truncate state if it's too long
            else:
                state = state[:self.state_size]

        q_values = self.model.predict(state.reshape(1, -1), verbose=0)
        return int(numpy.argmax(q_values[0]))

    def reset_environment(self):
        """Reset the environment between episodes"""
        req = Dqn.Request()
        req.action = 0  # Default action
        req.init = True  # Signal that this is a reset

        while not self.reset_environment_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('reset environment service not available, waiting again...')

        future = self.reset_environment_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        if future.done() and future.result() is not None:
            return future.result().state
        else:
            self.get_logger().error(f'Reset environment service call failure: {future.exception()}')
            return None

    def run_test(self):
        cur_test = 1

        while cur_test <= self.total_test:
            self.get_logger().info(f'Starting test {cur_test}/{self.total_test}')

            cur_results = {
                'test_id': cur_test,
                'commulative_reward': 0,
                'timeout_or_error': '',
                'goal_reached': None,
                'collision_happened': None,
                'episode_length': None,
                'collisions_avoided': 0,  # New field to track collisions avoided
                'near_misses': 0,  # New field to track near misses
            }

            done = False
            init = True
            score = 0
            local_step = 0
            next_state = []
            prev_min_distance = 10.0  # Initialize with a large value

            # Reset environment for each test (except the first one)
            if cur_test > 1:
                next_state = self.reset_environment()
                if next_state is None:
                    cur_results['timeout_or_error'] = 'Failed to reset environment'
                    self.results.append(cur_results)
                    cur_test += 1
                    continue

            time.sleep(1.0)

            while not done:
                local_step += 1
                action = 2 if local_step == 1 else self.get_action(next_state)

                req = Dqn.Request()
                req.action = action
                req.init = init

                while not self.rl_agent_interface_client.wait_for_service(timeout_sec=1.0):
                    self.get_logger().warn('rl_agent interface service not available, waiting again...')

                future = self.rl_agent_interface_client.call_async(req)
                rclpy.spin_until_future_complete(self, future)

                if future.done() and future.result() is not None:
                    # episode finished
                    next_state = future.result().state
                    reward = future.result().reward
                    done = future.result().done
                    score += reward
                    init = False

                    cur_results["episode_length"] = local_step

                    # Extract minimum obstacle distance from state
                    # State format: [goal_distance, goal_angle, 24 lidar readings]
                    if len(next_state) >= 26:
                        # Get the minimum value from the lidar readings (indices 2-25)
                        current_min_distance = min(next_state[2:26])

                        # Track near misses (when robot comes very close to obstacles)
                        if current_min_distance < 0.25:  # Near miss threshold
                            cur_results["near_misses"] += 1

                        # Track collisions avoided
                        # If robot was in danger zone but moved away without colliding
                        if prev_min_distance < 0.2 and current_min_distance > 0.3:
                            cur_results["collisions_avoided"] += 1

                        # Update previous distance for next iteration
                        prev_min_distance = current_min_distance

                    # Check if the episode ended due to goal reached or collision
                    if done:
                        if reward == 100.0:  # Goal reached
                            cur_results["goal_reached"] = True
                        elif reward == -50.0:  # Collision or timeout
                            cur_results["collision_happened"] = True
                else:
                    self.get_logger().error(f'Service call failure: {future.exception()}')
                    cur_results['timeout_or_error'] = f'Service call failure: {future.exception()}'
                    done = True  # End the episode on service failure

                time.sleep(0.01)

            # After the episode ends
            cur_results["commulative_reward"] = score
            self.results.append(cur_results)
            cur_test += 1  # Move to next test

        # After all tests are done, save the results
        filename = f"PPO_results_stage{self.stage}_{self.load_episode}.json"
        self.get_logger().info(f'Saving results to {filename}')
        with open(filename, "w") as f:
            json.dump(self.results, f, indent=4)


def main(args=None):
    rclpy.init(args=args if args else sys.argv)
    stage = sys.argv[1] if len(sys.argv) > 1 else '1'
    load_episode = sys.argv[2] if len(sys.argv) > 2 else '600'
    node = DQNTest(stage, load_episode)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()