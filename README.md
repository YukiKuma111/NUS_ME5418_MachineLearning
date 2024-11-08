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

## Running the Project

### Step 1: Test M4 Environment

```
# Test if the environment is set up correctly
python env/group24_env.py
```

You can switch M4's movement mode by modifying the `state` defined in line 1113 of the [`env/group24_env.py`](./env/group24_env.py) file.

If you wish to remove obstacles, change `hardcore: bool = True` to __False__ in line 149 of the [`env/group24_env.py`](./env/group24_env.py) file.

### Step 2: Run the Project

The project runs through the M4_PPO.ipynb file:

```
# Start Jupyter Notebook
jupyter notebook M4_PPO.ipynb
```

In the notebook, you can execute the code step-by-step to monitor the training process and model performance.

### Step 3: Training Configuration and Parameters

The neural network training process in Jupyter Notebook can be customized with the following key parameters:

- __Gamma__ (default = 0.99): This discount factor helps determine the importance of future rewards. A higher gamma (close to 1) makes the model more focused on long-term rewards.
- __Tau__ (default = 0.95): This parameter is used in Generalized Advantage Estimation (GAE) to smooth the advantage estimates, balancing bias and variance.

These parameters can be adjusted to fine-tune the model’s learning process. Additionally:

- __num_updates__ (default = 1000): This controls the number of training updates (iterations). Increasing this value can allow for longer training, potentially improving performance but also increasing computation time.
- __np.mean (scores_deque)__ (default > 1000): This specifies the stopping criterion for training. Training will end if the average score over recent episodes exceeds 800. This threshold can be adjusted depending on the desired performance level.

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

![Watch the video](./M4_PPO_vis.gif)