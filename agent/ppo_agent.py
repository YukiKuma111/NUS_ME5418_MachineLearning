import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.testing._internal.distributed.rpc.examples.reinforcement_learning_rpc_test import Agent

from nerual_network.rnn import PolicyNetwork as actor
from nerual_network.rnn import ValueNetwork as critic

PPO_CLIP = 0.2

class PPOAgent():
    def __init__(self,actor,critic,ppo_epoch,num_mini_batch,
                 lr, eps, max_grad_norm):
        self.actor = actor
        self.critic = critic
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch
        self.actor_loss_fn = nn.MSELoss()
        self.critic_loss_fn = nn.MSELoss()
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

    def get_action(self):


    def learning(self):


