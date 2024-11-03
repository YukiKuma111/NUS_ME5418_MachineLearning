#load_ext autoreload
#autoreload 2

import gym
import torch
import numpy as np
import time
import os
from datetime import datetime

from parallelEnv import parallelEnv
from model import PolicyNetwork as actor
from model import ValueNetwork as critic
from ppo import ppo_agent
from storage import RolloutStorage
from gym.vector import SyncVectorEnv
from collections import deque
import matplotlib.pyplot as plt
from envs import make_vec_envs
from utils import get_render_func
from torch.utils.tensorboard import SummaryWriter
# from stable_baselines3.common.env_util import make_vec_env
# matplotlib inline

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

debug = True

print('gym version: ', gym.__version__)
print('torch version: ', torch.__version__)
print('cuda version: ', torch.version.cuda)
print('cuda available: ', torch.cuda.is_available())
print("CUDA device count: ", torch.cuda.device_count())
print("Current CUDA device: ", torch.cuda.current_device())

seed = 0
gamma=0.99
num_processes=16
device = torch.device("cpu")
# device = torch.device("cuda:0")
print('device: ', device)

envs = parallelEnv('Group24M4-v0', n=num_processes, seed=seed)

max_steps = envs.max_steps
print('max_steps: ', max_steps)

if debug:
    action = envs.action_space.sample()
    observation = envs.observation_space.sample()
    ac_size = envs.action_space
    ob_size = envs.observation_space.shape

    print("action", action)
    print("observation", observation)
    print("ac_size", ac_size)
    print("ob_size", ob_size)

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
# init the nn
hidden_size_gru = 35
hidden_size_MLP =35
# actor
actor_nn = actor(envs.observation_space.shape,hidden_size_gru,hidden_size_MLP, envs.action_space.shape[0])
# critic
critic_nn = critic(envs.observation_space.shape, hidden_size_gru,hidden_size_MLP)
if debug:
    print("actor_nn: ", actor_nn)
    print("critic_nn: ", critic_nn)
    print("envs.observation_space.shape: ", envs.observation_space.shape)
    print("envs.action_space: ", envs.action_space.shape[0])

actor_nn.to(device)
critic_nn.to(device)

agent = ppo_agent(actor=actor_nn,critic=critic_nn, ppo_epoch=16, num_mini_batch=16,\
                lr=0.001, eps=1e-5, max_grad_norm=0.5)

rollouts = RolloutStorage(num_steps=max_steps, num_processes=num_processes, \
                        obs_shape=envs.observation_space.shape, action_space=envs.action_space, \
                        recurrent_hidden_state_size=policy.recurrent_hidden_state_size)

obs = envs.reset()
print('type obs: ', type(obs), ', shape obs: ', obs.shape)
obs_t = torch.tensor(obs)
print('type obs_t: ', type(obs_t), ', shape obs_t: ', obs_t.shape)

rollouts.obs[0].copy_(obs_t)
rollouts.to(device)


def save(model, directory, filename, suffix):
    if not os.path.exists(directory):
        os.makedirs(directory)
    torch.save(model.base.actor.state_dict(), '%s/%s_actor_%s.pth' % (directory, filename, suffix))
    torch.save(model.base.critic.state_dict(), '%s/%s_critic_%s.pth' % (directory, filename, suffix))
    torch.save(model.base.critic_linear.state_dict(), '%s/%s_critic_linear_%s.pth' % (directory, filename, suffix))


# limits = [-300, -160, -100, -70, -50, 0, 20, 30, 40, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
limits = [-2000, -900, -600, -500, -400, -300, -160, -100, -70, -50, 0, 20, 30, 40, 60, 90, 120, 150, 180, 210, 240,
          270, 300, 330]


def return_suffix(j):
    suf = '0'
    for i in range(len(limits) - 1):
        if j > limits[i] and j < limits[i + 1]:
            suf = str(limits[i + 1])
            break

        i_last = len(limits) - 1
        if j > limits[i_last]:
            suf = str(limits[i_last])
            break
    return suf


num_updates = 1000000
gamma = 0.99
tau = 0.95
save_interval = 30
log_interval = 1

# Define the log directory
log_dir = 'runs/ppo_experiment'

# Get the current time as a string for tagging
current_time_tag = datetime.now().strftime('%Y%m%d_%H%M%S')


def ppo_vec_env_train(envs, agent, policy, num_processes, num_steps, rollouts):
    # Create a new SummaryWriter
    writer = SummaryWriter(log_dir=log_dir)

    time_start = time.time()

    n = len(envs.ps)
    envs.reset()

    # start all parallel agents
    print('Number of agents: ', n)
    envs.step([[1] * 4] * n)

    indices = []
    for i in range(n):
        indices.append(i)

    s = 0

    scores_deque = deque(maxlen=100)
    scores_array = []
    avg_scores_array = []

    for i_episode in range(num_updates):

        total_reward = np.zeros(n)
        timestep = 0

        for timestep in range(num_steps):
            with torch.no_grad():
                value, actions, action_log_prob, recurrent_hidden_states = \
                    policy.act(
                        rollouts.obs[timestep],
                        rollouts.recurrent_hidden_states[timestep],
                        rollouts.masks[timestep])

            obs, rewards, done, infos = envs.step(actions.cpu().detach().numpy())
            # obs, rewards, done, truncs, infos = envs.step(actions.cpu().detach().numpy())

            total_reward += rewards  ## this is the list by agents

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            obs_t = torch.tensor(obs)
            ## Add one dimnesion to tensor, otherwise does not work
            ## This is (unsqueeze(1)) solution for:
            ## RuntimeError: The expanded size of the tensor (1) must match the existing size...
            rewards_t = torch.tensor(rewards).unsqueeze(1)
            rollouts.insert(obs_t, recurrent_hidden_states, actions, action_log_prob, \
                            value, rewards_t, masks)

        avg_total_reward = np.mean(total_reward)
        scores_deque.append(avg_total_reward)
        scores_array.append(avg_total_reward)

        with torch.no_grad():
            next_value = policy.get_value(rollouts.obs[-1],
                                          rollouts.recurrent_hidden_states[-1],
                                          rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, gamma, tau)

        agent.update(rollouts)

        rollouts.after_update()

        avg_score = np.mean(scores_deque)
        avg_scores_array.append(avg_score)

        # TensorBoard Logging
        writer.add_scalar(f'Average Total Reward/{current_time_tag}', avg_total_reward, i_episode)
        writer.add_scalar(f'Average Score/{current_time_tag}', avg_score, i_episode)

        if i_episode > 0 and i_episode % save_interval == 0:
            print('Saving model, i_episode: ', i_episode, '\n')
            suf = return_suffix(avg_score)
            save(policy, 'dir_save', 'we0', suf)

        if i_episode % log_interval == 0 and len(scores_deque) > 1:
            prev_s = s
            s = (int)(time.time() - time_start)
            t_del = s - prev_s
            print('Ep. {}, Timesteps {}, Score.Agents: {:.2f}, Avg.Score: {:.2f}, Time: {:02}:{:02}:{:02}, \
Interval: {:02}:{:02}' \
                  .format(i_episode, timestep + 1, \
                          avg_total_reward, avg_score, s // 3600, s % 3600 // 60, s % 60, t_del % 3600 // 60,
                          t_del % 60))

            # if len(scores_deque) == 100 and np.mean(scores_deque) > 300.5:
        # if len(scores_deque) == 100 and np.mean(scores_deque) > 300.0:
        if len(scores_deque) == 100 and np.mean(scores_deque) > 280:
            # if np.mean(scores_deque) > 20:
            print('Environment solved with Average Score: ', np.mean(scores_deque))
            break

    writer.close()

    return scores_array, avg_scores_array

scores, avg_scores = ppo_vec_env_train(envs, agent, policy, num_processes, max_steps, rollouts)
print('length of scores: ', len(scores), ', len of avg_scores: ', len(avg_scores))

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores, label="Score")
plt.plot(np.arange(1, len(avg_scores)+1), avg_scores, label="Avg on 100 episodes")
plt.legend(bbox_to_anchor=(1.05, 1))
plt.ylabel('Score')
plt.xlabel('Episodes #')
plt.show()

#--------------- make_vec_envs ----------------
## we continue with the same model, model Policy uses MLPBase, but with new environment env_venv

device = torch.device("cpu")
print('device: ', device)

seed = 0
num_processes=1

env_venv = make_vec_envs('Group24M4-v0', \
                    seed + 1000, num_processes,
                    None, None, False, device='cpu', allow_early_resets=False)

policy = policy.to(device)

print('env_venv.observation_space.shape: ', env_venv.observation_space.shape, \
      ', len(obs_shape): ', len(env_venv.observation_space.shape))
print('env_venv.action_space: ',  env_venv.action_space, \
      ', action_space.shape[0]: ', env_venv.action_space.shape[0])


## No CUDA, only CPU

def play_VecEnv(env, model, num_episodes):
    for name, param in model.named_parameters():
        print(name, param)

    obs = env.reset()
    obs = torch.Tensor(obs)
    obs = obs.float()

    recurrent_hidden_states = torch.zeros(1, model.recurrent_hidden_state_size)

    masks = torch.zeros(1, 1)

    scores_deque = deque(maxlen=100)

    render_func = get_render_func(env)

    for i_episode in range(1, num_episodes + 1):

        time_start = time.time()
        total_reward = np.zeros(num_processes)
        timestep = 0

        while True:

            with torch.no_grad():
                value, action, _, recurrent_hidden_states = \
                    model.act(obs, recurrent_hidden_states, masks, \
                              deterministic=False)  # obs = state

            render_func()

            obs, reward, done, _ = env.step(action)
            obs = torch.Tensor(obs)
            obs = obs.float()

            reward = reward.detach().numpy()

            masks.fill_(0.0 if done else 1.0)

            total_reward += reward[0]

            # if timestep < 800:
            #    print('timestep: ', timestep, 'reward: ', reward[0])

            timestep += 1

            if timestep + 1 == 1600:  ##   envs.max_steps:
                break

        s = (int)(time.time() - time_start)

        scores_deque.append(total_reward[0])

        avg_score = np.mean(scores_deque)

        print('Episode {} \tScore: {:.2f}, Avg.Score: {:.2f}, \tTime: {:02}:{:02}:{:02}' \
              .format(i_episode, total_reward[0], avg_score, s // 3600, s % 3600 // 60, s % 60))  

env_venv.close()
obs = env_venv.reset()
# obs.close()
env_venv.close()