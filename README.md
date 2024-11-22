# ME5418_Machine_Learning

## Project Overview

This project aims to use reinforcement learning to train the Multi-Modal Mobility Morphobot (M4), a wheel-legged robot, to overcome complex environments with obstacles, such as slopes and holes. The primary algorithm used is PPO (Proximal Policy Optimization), which iteratively optimizes the policy to help M4 learn how to deal with various terrain challenges.

## Steps to Set Up

### Step1: Download Repository

```
mkdir your_workspace
cd your_workspace/
git clone https://github.com/YukiKuma111/ME5418_ML.git
cd ME5418_ML/
```

### Step2: Environment Setup

This project relies on several key libraries. To ensure compatibility, please create a conda environment using the following commands:

```
conda env create -f environment.yaml
```

The main libraries and versions used in this project are as follows:

- __Python 3.8__
- __PyTorch 2.0.0__
- __TensorFlow 2.13.1__
- __Gym 0.26.2__
- __Matplotlib 3.7.5__
- __Pygame 2.6.1__
- __Box2D__

## Project Declare
Our network was defined in the model_custom.py file and our learning agent is the ppo_custom.py. 
The M4_PPO_custom_train.ipynb is the jupyter notebook used to train our network, but after we trained our network by the M4_PP0_custom_train.ipynb file
```
# Start Jupyter Notebook
jupyter notebook M4_PPO_custom_train.ipynb
```
We found the rewards of our network cannot converge as the following figure shows, so we changed our network and ppo agent according to a [BipedalWalker Project](https://github.com/Rafael1s/Deep-Reinforcement-Learning-Algorithms/tree/master/BipedalWalker-PPO-VectorizedEnv) 
The new network is the model_ref.py and new ppo agent is the ppo_ref.py
![rewards changes of our network](./dir_save/custom_training_result/Average%20Score_20241122_155812.png)
__The "Running custom Project" section is to run our original project, 
and the "Running the Reference Project" section is to run the project changed 
according to the [BipedalWalker Project](https://github.com/Rafael1s/Deep-Reinforcement-Learning-Algorithms/tree/master/BipedalWalker-PPO-VectorizedEnv)__ 


## Running the Custom Project
### Step 1: Test M4 Environment

```
# Test if the environment is set up correctly
python env/group24_env.py
```

You can switch M4's movement mode by modifying the `state` defined in line 1113 of the [`env/group24_env.py`](./env/group24_env.py) file.

If you wish to remove obstacles, change `hardcore: bool = True` to __False__ in line 149 of the [`env/group24_env.py`](./env/group24_env.py) file.

### Step 2: Run the Project

The project runs through the M4_PPO_ref_train.ipynb file which will train the network:

```
# Start Jupyter Notebook
jupyter notebook M4_PPO_custom_train.ipynb
```

In the notebook, you can execute the code step-by-step to monitor the training process and model performance.

### Step 3: Visualizing Training with TensorBoard

A bash script, [`view_tensorboard.sh`](./view_tensorboard.sh), is included for real-time tracking of training progress. To launch TensorBoard for monitoring the training metrics, run:

```
bash view_tensorboard.sh
```

This visualization tool provides insights into key metrics: Average Total Reward and Average Score, helping users assess and optimize the training process effectively.

## Running the Reference Project

### Step 1: Test M4 Environment

```
# Test if the environment is set up correctly
python env/group24_env.py
```

You can switch M4's movement mode by modifying the `state` defined in line 1113 of the [`env/group24_env.py`](./env/group24_env.py) file.

If you wish to remove obstacles, change `hardcore: bool = True` to __False__ in line 149 of the [`env/group24_env.py`](./env/group24_env.py) file.

### Step 2: Run the Project

The project runs through the M4_PPO_ref_train.ipynb file which will train the network:

```
# Start Jupyter Notebook
jupyter notebook M4_PPO_ref_train.ipynb
```

In the notebook, you can execute the code step-by-step to monitor the training process and model performance.

### Step 3: Training Configuration and Parameters

The neural network training process in Jupyter Notebook can be customized with the following key parameters:

- __Gamma__ (default = 0.99): This discount factor helps determine the importance of future rewards. A higher gamma (close to 1) makes the model more focused on long-term rewards.
- __Tau__ (default = 0.95): This parameter is used in Generalized Advantage Estimation (GAE) to smooth the advantage estimates, balancing bias and variance.

These parameters can be adjusted to fine-tune the model’s learning process. Additionally:

- __num_updates__ (default = 1000): This controls the number of training updates (iterations). Increasing this value can allow for longer training, potentially improving performance but also increasing computation time.
- __np.mean (scores_deque)__ (default > 1100): This specifies the stopping criterion for training. Training will end if the average score over recent episodes exceeds 800. This threshold can be adjusted depending on the desired performance level.

### Step 4: Visualizing Training with TensorBoard

A bash script, [`view_tensorboard.sh`](./view_tensorboard.sh), is included for real-time tracking of training progress. To launch TensorBoard for monitoring the training metrics, run:

```
bash view_tensorboard.sh
```

This visualization tool provides insights into key metrics: Average Total Reward and Average Score, helping users assess and optimize the training process effectively.

## Expected Output and Results Analysis

After running the project, the expected output will be `.pth` files stored in the `dir_save` directory, containing the weights and biases of the actor and critic neural networks.

The main reward function of the project are as follows:

- Reward for Successful Forward Movement: M4 receives a reward for every step forward.
- Reward for Wheel-Ground Contact: A reward is given for stable ground contact of the wheels.
- Penalty for High Leg Angle: M4 receives a penalty if its leg angle is too high, preventing it from losing balance.
- Penalty for Lander Ground Contact: M4 is penalized heavily if its lander touches the ground.

By optimizing these rewards and penalties, M4 can eventually navigate complex terrains with SLOPE and HOLE, as shown in the gif below.
Or if you want to run our train result please run the M4_PPO_test.ipynb file:
```
# Start Jupyter Notebook
jupyter notebook M4_PPO_ref_test.ipynb
```

![Watch the video](./M4_PPO_vis.gif)