import torch
import torch.nn as nn
import torch.optim as optim
from storage import RolloutStorage

from model import PolicyNetwork as actor
from model import ValueNetwork as critic


PPO_CLIP = 0.2

class ppo_agent():
    def __init__(self,
                 actor, critic,
                 ppo_epoch,
                 num_mini_batch,
                 lr = None,
                 eps = None,
                 max_grad_norm=None):
        # network init
        self.actor = actor
        self.critic = critic
        self.num_mini_batch = num_mini_batch

        # ppo init
        self.max_grad_norm = max_grad_norm
        self.ppo_epoch = ppo_epoch
        #self.actor_loss_fn = nn.MSELoss()
        self.critic_loss_fn = nn.MSELoss()
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

    def update(self,rollouts):
        #, action_log_probs, dist_entropy, values
        # advantage function updating
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / advantages.std()

        # ppo internal epoch
        for epoch in range(self.ppo_epoch):
            # get hidden state, sample action, action probability distribution, states, entropy
            # of actions of each parallel environment
            data_generator = rollouts.feed_forward_generator(
                advantages, self.num_mini_batch)

            # analyze each parallel envs to update the network
            for sample in data_generator:
                # state, hidden state, sample action, reward,masks,
                # old log probs of actions, target advantage
                obs_batch, recurrent_hidden_states_batch, actions_batch, \
                    return_batch, masks_batch, old_action_log_probs_batch, \
                    adv_targ = sample

                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy, states = self.actor.evaluate_actions(
                    obs_batch, recurrent_hidden_states_batch,
                    masks_batch)

                # the ratio between action probability distribution and the previous one
                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - PPO_CLIP, 1.0 + PPO_CLIP) * adv_targ

                # loss calculation
                actor_loss = -torch.min(surr1, surr2).mean()- 0.01 * dist_entropy
                critic_loss = self.critic_loss_fn(values, return_batch)

                # restrict gradient boom
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)

                # clean gradient
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()

                # backpropgation
                actor_loss.backward()
                critic_loss.backward()

                # gradient update
                self.actor_optimizer.step()
                self.critic_optimizer.step()
