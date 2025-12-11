# TurtleBot3 Machine Learning Simulation (RL Training)

This repository provides a complete setup for training **Reinforcement Learning (RL)** policies for TurtleBot3 in simulation, including **DQN**, **PPO**, and real-robot **physical testing**. Follow the steps below to install TurtleBot3 simulation packages, run Gazebo, launch RL training scripts, and use Docker for a simplified environment setup.

The setup is similar to the [TurtleBot3 Official eManual](https://emanual.robotis.com/docs/en/platform/turtlebot3/machine_learning/).  
We have trained PPO and DQN using the SOTA library **stable-baselines3**.

**DQN training modification:**  
[View dqn_agent.py (DQN version)](https://github.com/Alive-Rehan/RL_Project/blob/dqn_simulation/turtlebot3_dqn/turtlebot3_dqn/dqn_agent.py#L242)

**PPO training modification:**  
[View dqn_agent.py (PPO version)](https://github.com/Alive-Rehan/RL_Project/blob/ed321d02390a70d58362defdb936c7e00a4f0c5a/turtlebot3_dqn/turtlebot3_dqn/PPO_train.py#L141)

---

## 🚨 Important Branches

Make sure to explore the following branches depending on your workflow:

* **`DQN_simulation`** → DQN training in Gazebo
* **`PPO_simulation`** → PPO training in Gazebo
* **`Physical_testing`** → Deploy trained RL policies on physical TurtleBot3

Switch branches using:

```bash
git checkout <branch-name>
```

---

# 🤖 2. RL Training Setup
## Run Machine Learning

Following code works for both the branches:

**Bring the stage in Gazebo map:**

```
ros2 launch turtlebot3_gazebo turtlebot3_dqn_${stage_num}.launch.py
```

**Run Gazebo environment node**
This node manages the Gazebo environment. It regenerates the goal and initializes the TurtleBot’s location when an episode starts anew.

```
ros2 run turtlebot3_dqn dqn_gazebo ${stage_num}
```

**Run DQN environment node**
This node manages the DQN environment. It calculates the state of the TurtleBot and uses it to determine rewards, success, and failure.

```
ros2 run turtlebot3_dqn dqn_environment
```

**Run DQN agent node**
This node trains the TurtleBot. It trains TurtleBot with calculated rewards and determines its next behavior.

```
ros2 run turtlebot3_dqn dqn_agent ${stage_num} ${max_training_episodes}
```

**Test trained model**
After training, to test the trained model, run this node instead of the DQN agent node.

```
ros2 run turtlebot3_dqn dqn_test ${stage_num} ${load_episode}
```


**Run TensorBoard**

To start TensorBoard and view the training curves:

```
tensorboard --logdir=~/turtlebot3_dqn_logs/gradient_tape
```

---

# 🐳 3. Docker Image for TurtleBot3 RL Training

To avoid dependency issues, you can use a Docker image with ROS, Gazebo, and machine learning libraries pre‑installed.

## 3.1 Pull the Docker Image

```bash
docker pull rehanf369/tb3_stablebaseline:DQN2
```

## 3.2 Run the Container with GUI Support

```bash
docker run -it --privileged --gpus all -e NVIDIA_DRIVER_CAPABILITIES=all -e NVIDIA_VISIBLE_DEVICES=all -e ROS_DOMAIN_ID=12 -e "DISPLAY=$DISPLAY" -e ACCEPT_EULA=Y -v /tmp/.X11-unix:/tmp/.X11-unix -v $HOME/.Xauthority:/root/.Xauthority -v /dev:/dev --name <container name> <images name or ID>
```

Allow Docker to access X11:

```bash
xhost +local:docker
```

# 🎯 Summary

This repository provides:

* TurtleBot3 Gazebo simulation setup
* Training scripts for DQN & PPO
* Docker support for easy installation
* Physical robot deployment instructions
* Branch-wise separation for clean workflow

For any issues, refer to the TurtleBot3 documentation or open an issue in this repository.




