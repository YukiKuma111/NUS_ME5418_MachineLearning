import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.distributions import Categorical
from distributions import DiagGaussian

from utils import init, init_normc_


# class of critic
class ValueNetwork(nn.Module):
    def __init__(self, obs_space, hidden_size_gru, hidden_size_MLP):
        super(ValueNetwork, self).__init__()
        self.gru_value_1 = nn.GRUCell(input_size=obs_space,hidden_size= hidden_size_gru)

        # gru layer init
        nn.init.orthogonal_(self.gru_value_1.weight_ih.data)
        nn.init.orthogonal_(self.gru_value_1.weight_hh.data)
        self.gru_value_1.bias_ih.data.fill_(0)
        self.gru_value_1.bias_hh.data.fill_(0)

        # define the init function for the weight and bias init of MLPs
        self.init_ = lambda m: init(m, init_normc_, lambda x: nn.init.constant_(x, 0))

        # two MLP layer to get action with normalization to improve the training speed
        self.MLP_1 = self.init_(nn.Linear(hidden_size_gru, hidden_size_MLP))
        self.MLP_2 = self.init_(nn.Linear(hidden_size_MLP, 1))

        self.train()

    def forward_critic(self, x, hidden_state,masks):
        # gru layer
        #output_set_gru,_ = self.gru_value_1(input)
        #squ_gru,batch_gru,output_gru = output_set_gru.shape
        #inputs_of_MLP_1 = output_set_gru.view(squ_gru * batch_gru, output_gru)

        batch_size, input_size = x.shape
        if x.shape[0] == hidden_state.shape[0]:
            masked_hidden_state = hidden_state*masks
            hidden_state = self.gru_value_1(x, masked_hidden_state)

        elif x.shape[0] > hidden_state.shape[0]:
            batch_size_of_hidden_state = hidden_state.shape[0]
            seq_len = int(batch_size/batch_size_of_hidden_state)

            x = x.view(seq_len, batch_size_of_hidden_state, x.size(1))
            masks = masks.view(seq_len,batch_size_of_hidden_state,hidden_state.size(1))

            output = []

            for i in range(seq_len):
                masked_hidden_state = hidden_state*masks[i]
                out = hidden_state = self.gru_policy_1(x[i], masked_hidden_state)
                output.append(out)

            x = torch.stack(output, dim=0)
            x = x.view(seq_len*batch_size_of_hidden_state,-1)

        else:
            return NotImplementedError

        # MLP layer 1
        output_of_MLP_1 = torch.tanh(self.MLP_1(x))

        # MLP layer 2
        value = torch.tanh(self.MLP_2(output_of_MLP_1))

        #value = value.view(squ_gru,batch_gru,-1)

        return value, hidden_state

    def get_value(self, x, hidden_state, masks):
        value, hidden_state = self.forward_critic(x, hidden_state, masks)
        return value

# class of actor
class PolicyNetwork(nn.Module):
    def __init__(self,
                 obs_space,
                 hidden_size_gru,
                 hidden_size_MLP,
                 action_space):
        super(PolicyNetwork, self).__init__()

        self.input_of_Gaussian = hidden_size_MLP
        self.action_space = action_space

        # Embed ValueNetwork as a component instead of inheriting from it
        self.value_network = ValueNetwork(obs_space, hidden_size_gru, hidden_size_MLP)

        # init hidden state
        #self.hidden_state = torch.zeros(batch_size, hidden_size_gru).to(device)

        # gru layer definition
        self.gru_policy_1 = nn.GRUCell(input_size=obs_space,
                                       hidden_size=hidden_size_gru)

        # gru layer init
        nn.init.orthogonal_(self.gru_policy_1.weight_ih.data)
        nn.init.orthogonal_(self.gru_policy_1.weight_hh.data)
        self.gru_policy_1.bias_ih.data.fill_(0)
        self.gru_policy_1.bias_hh.data.fill_(0)

        # define the init function for the weight and bias init of MLPs
        self.init_ = lambda m: init(m, init_normc_, lambda x: nn.init.constant_(x, 0))

        # two MLP layer to get action with normalization to improve the training speed
        self.MLP_1 = self.init_(nn.Linear(hidden_size_gru, hidden_size_MLP))
        self.MLP_2 = self.init_(nn.Linear(hidden_size_MLP, hidden_size_MLP))

        self.train()
        # init the action probability distribution calculator
        #self.dist = DiagGaussian(hidden_size_gru, action_space)

    def forward_actor(self, x, hidden_state,masks):
        # input = (batch_size,input_size)
        # hidden_state = (batch_size,hidden_size_gru)
        batch_size,input_size = x.shape
        if x.shape[0] == hidden_state.shape[0]:
            masked_hidden_state = hidden_state * masks
            hidden_state = self.gru_policy_1(x, masked_hidden_state)

        elif x.shape[0] > hidden_state.shape[0]:
            # the batch size of the input x must be larger than the batch size of the hidden state
            batch_size_of_hidden_state = hidden_state.shape[0]
            seq_len = int(batch_size/batch_size_of_hidden_state)

            # 2D to 3D adjustment for the input x and the masks
            x = x.view(seq_len,batch_size_of_hidden_state,x.size(1))
            masks = masks.view(seq_len,batch_size_of_hidden_state,hidden_state.size(1))

            output = []

            for i in range(seq_len):
                masked_hidden_state = hidden_state * masks[i]
                out = hidden_state = self.gru_policy_1(x[i], masked_hidden_state)
                output.append(out)

            x = torch.stack(output, dim=0)
            x = x.view(seq_len*batch_size_of_hidden_state,-1)
        else:
            return NotImplementedError


        # gru layer
        #output_set_gru,_ = self.gru_policy_1(input)
        #squ_gru,batch_gru,output_gru = output_set_gru.shape
        #inputs_of_MLP_1 = output_set_gru.view(squ_gru * batch_gru,output_gru)

        # first MLP layer
        output_of_MLP_1 = torch.tanh(self.MLP_1(x))

        # second MLP layer
        action = torch.tanh(self.MLP_2(output_of_MLP_1))

        # #action = action.view(squ_gru, batch_gru, -1)
        # # normalize the action, to get the probability distribution of the four actions
        # action = F.softmax(action, dim=-1)

        return action, hidden_state


    # used to the choose which action should be taken
    # the input - action is the output of the network, the probability distribution of the four actions
    def act(self, x, hidden_state, masks, deterministic=False):

        action_feature,hidden_state = self.forward_actor(x, hidden_state, masks)
        value, hidden_state = self.value_network.forward_critic(x, hidden_state, masks)

        # print(f'the output of act in the network {action_feature}')

        self.dist = DiagGaussian(self.input_of_Gaussian,self.action_space)
        dist = self.dist(action_feature)

        if deterministic:
            sample_action = dist.mode()
        else:
            sample_action = dist.sample()

        # print(f'\nthe output of sample action in the network {sample_action}')

        # return the index of the action which should be taken
        action_log_probs = dist.log_probs(sample_action)
        # print(f'\nthe output of action_log_prob in the network {action_log_probs}')
        # calculate the log probability of hte sample action for the ppo calculation
        # check the gradient tracking
        # print(f"Action feature requires grad: {action_feature.requires_grad}")
        # print(f"Sample action requires grad: {sample_action.requires_grad}")

        return value, sample_action, action_log_probs, hidden_state

    # used to
    def evaluate_actions(self, x, hidden_state, masks,deterministic=False):

        action_feature, hidden_state = self.forward_actor(x, hidden_state, masks)
        value, hidden_state = self.value_network.forward_critic(x, hidden_state, masks)

        self.dist = DiagGaussian(self.input_of_Gaussian, self.action_space)
        dist = self.dist(action_feature)

        if deterministic:
            sample_action = dist.mode()
        else:
            sample_action = dist.sample()

        action_log_probs = dist.log_probs(sample_action)
        # calculate the log probability of hte sample action for the ppo calculation

        dist_entropy = dist.entropy().mean()
        # the entropy of the four actions ,for the ppo calculation

        return value, action_log_probs, dist_entropy, hidden_state

if __name__ == '__main__':

    # device definition
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # target_action = torch.tensor([[0.1,0.2,0.1,0.5],
    #                               [0.2,0.1,0.5,0.4],
    #                               [0.3,0.8,0.4,0.2],
    #                               [0.1,0.5,0.3,0.4],
    #                               [0.1,0.2,0.6,0.8]],dtype=torch.float32).to(device)
    # target_value = torch.tensor([[0.5],
    #                              [0.2],
    #                              [0.4],
    #                              [0.1],
    #                              [0.3]],dtype=torch.float32).to(device)

    # network params
    obs_space = 35
    hidden_size_gru = 35
    hidden_size_MLP = 64
    action_space = 4
    num_layers = 1

    epoch = 5

    # actor and critic setup
    myactor = PolicyNetwork(obs_space, hidden_size_gru, hidden_size_MLP, action_space).to(device)
    mycritic = ValueNetwork(obs_space, hidden_size_gru, hidden_size_MLP).to(device)

    # optimizer definition
    actor_optimizer = torch.optim.Adam(myactor.parameters(), lr = 1e-3)
    critic_optimizer = torch.optim.Adam(mycritic.parameters(), lr = 1e-3)

    # loss function definition
    actor_loss_fn = nn.MSELoss()
    critic_loss_fn = nn.MSELoss()

    # assume the input data
    batch_size = 1
    seq_len = 10
    input_data = torch.randn(batch_size, obs_space).to(device)  # 随机生成输入数据
    x = input_data

    # init the hidden_state of the actor and the critic
    batch_size_hidden = batch_size
    hidden_state_actor = torch.zeros(batch_size_hidden, hidden_size_gru).to(device)
    hidden_state_critic = torch.zeros(1, 1).to(device)

    # masks init
    masks = torch.ones(batch_size_hidden, hidden_size_gru,dtype=torch.float32).to(device)

    # random target
    target_action = torch.randn(batch_size, action_space).to(device)
    target_value = torch.randn(batch_size, 1).to(device)

    for i in range(epoch):
        print(f"{i + 1} times train")
        # my_actor.train()
        # my_critic.train()

        # # get action and value
        # action, hidden_state_actor = myactor.forward(x= x, hidden_state=hidden_state_actor,masks = masks)
        # value, hidden_state_critic = mycritic.forward(x=x, hidden_state=hidden_state_critic,masks = masks)
        # print(f'Epoch {i + 1}, Actions: {action} \nValues: {value}')

        # 1. get the index of the action which should be taken,
        # 2. get the log probability of the sample action
        value, action,log_probs_action,hidden_state_actor = myactor.act(x=x,hidden_state=hidden_state_actor,masks=masks,deterministic=False)
        print(f"value by act fn {value}, sample action {action}, log probs {log_probs_action}, hidden state {hidden_state_actor}")

        # # the entropy of the four actions
        # value_evaluate, action_log_probs, dist_entropy, hidden_state = myactor.evaluate_action(x=x,hidden_state=hidden_state_actor,masks=masks,sample_action = sample_action)
        # print(f"value by evaluate_action fn {value_evaluate}, action log_probs {action_log_probs}, dist_entropy {dist_entropy}, hidden state {hidden_state}")
        #
        # # get value
        # value = mycritic.get_value(x=x,hidden_state=hidden_state_critic,masks=masks)
        action = action.requires_grad_()

        actor_loss = actor_loss_fn(action, target_action.to(device).float())
        critic_loss = critic_loss_fn(value, target_value.to(device).float())

        actor_optimizer.zero_grad()
        critic_optimizer.zero_grad()

        actor_loss.backward()
        critic_loss.backward()

        actor_optimizer.step()
        critic_optimizer.step()

        print(f'Epoch {i + 1} train end, loss: {actor_loss.item()}, loss: {critic_loss.item()}')