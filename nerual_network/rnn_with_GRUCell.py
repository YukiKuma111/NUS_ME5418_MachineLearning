import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from docutils.nodes import target
from torch.ao.quantization import obs_or_fq_ctr_equals
from distribution.DiagGaussian import DiagGaussian
from torch.distributions import Categorical


# device definition
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# class of actor
class PolicyNetwork(nn.Module):
    def __init__(self,
                 obs_space,
                 hidden_size_gru,
                 hidden_size_MLP,
                 action_space,
                 num_layers):
        super(PolicyNetwork, self).__init__()
        # init hidden state
        self.hidden_state = torch.zeros(batch_size, self.hidden_size_gru).to(device)

        # gru layer definition
        self.gru_policy_1 = nn.GRUCell(input_size=obs_space,
                                       hidden_size=hidden_size_gru)

        # gru layer init
        nn.init.orthogonal_(self.gru_policy_1.weight_ih.data)
        nn.init.orthogonal_(self.gru_policy_1.weight_hh.data)
        self.gru_policy_1.bias_ih.data.fill_(0)
        self.gru_policy_1.bias_hh.data.fill_(0)

        # two MLP layer to get action with normalization to improve the training speed
        self.MLP_1 = nn.Linear(hidden_size_gru, hidden_size_MLP)
        self.MLP_2 = nn.Linear(hidden_size_MLP, action_space)

    def forward(self, input):# input = (seq_len,batch_size,input_size)
        seq_len,batch_size,input_size = input.shape
        hidden_state = torch.zeros(batch_size,self.gru_policy_1.hidden_size).to(device)


        for t in range(seq_len):
            hidden_state = self.gru_policy_1(input[t],hidden_state)

        # gru layer
        #output_set_gru,_ = self.gru_policy_1(input)
        #squ_gru,batch_gru,output_gru = output_set_gru.shape
        #inputs_of_MLP_1 = output_set_gru.view(squ_gru * batch_gru,output_gru)

        # first MLP layer
        output_of_MLP_1 = torch.tanh(self.MLP_1(hidden_state))

        # second MLP layer
        action = torch.tanh(self.MLP_2(output_of_MLP_1))

        #action = action.view(squ_gru, batch_gru, -1)

        return action





# class of critic
class ValueNetwork(nn.Module):
    def __init__(self, obs_space, hidden_size_gru, hidden_size_MLP, num_layers):
        super(ValueNetwork, self).__init__()
        self.gru_value_1 = nn.GRUCell(input_size=obs_space,hidden_size= hidden_size_gru)

        # gru layer init
        nn.init.orthogonal_(self.gru_value_1.weight_ih.data)
        nn.init.orthogonal_(self.gru_value_1.weight_hh.data)
        self.gru_value_1.bias_ih.data.fill_(0)
        self.gru_value_1.bias_hh.data.fill_(0)

        # two MLP layer to get action with normalization to improve the training speed
        self.MLP_1 = nn.Linear(hidden_size_gru, hidden_size_MLP)
        self.MLP_2 = nn.Linear(hidden_size_MLP, 1)

    def forward(self, input):
        # gru layer
        #output_set_gru,_ = self.gru_value_1(input)
        #squ_gru,batch_gru,output_gru = output_set_gru.shape
        #inputs_of_MLP_1 = output_set_gru.view(squ_gru * batch_gru, output_gru)

        seq_len, batch_size, input_size = input.shape
        hidden_state = torch.zeros(batch_size, self.gru_value_1.hidden_size).to(device)

        for t in range(seq_len):
            hidden_state = self.gru_value_1(input[t], hidden_state)


        # MLP layer 1
        output_of_MLP_1 = torch.tanh(self.MLP_1(hidden_state))

        # MLP layer 2
        value = torch.tanh(self.MLP_2(output_of_MLP_1))

        #value = value.view(squ_gru,batch_gru,-1)

        return value



if __name__ == '__main__':
    target_action = torch.tensor([1,0,1,0],dtype=torch.float32).to(device)
    target_value = torch.tensor([0.5],dtype=torch.float32).to(device)
    # network params
    obs_space = 35
    hidden_size_gru = 35
    hidden_size_MLP =35
    action_space = 4
    num_layers = 1
    masks = 1

    epoch = 15

    # actor and critic setup
    myactor = PolicyNetwork(obs_space, hidden_size_gru, hidden_size_MLP, action_space, num_layers)
    mycritic = ValueNetwork(obs_space, hidden_size_gru, hidden_size_MLP, num_layers)

    # optimizer definition
    actor_optimizer = torch.optim.Adam(myactor.parameters(), lr = 1e-3)
    critic_optimizer = torch.optim.Adam(mycritic.parameters(), lr = 1e-3)

    # loss function definition
    actor_loss_fn = nn.MSELoss()
    critic_loss_fn = nn.MSELoss()

    # assume the input data
    batch_size = 5
    seq_len = 10
    input_data = torch.randn(seq_len, batch_size, obs_space).to(device)  # 随机生成输入数据

    for i in range(epoch):
        print(f"{i + 1} times train")
        myactor.train()
        mycritic.train()

        # get action and value
        action = myactor(input_data).to(device)
        value = mycritic(input_data).to(device)

        action = action[-1]
        value = value[-1]
        print(f'Epoch {i + 1}, Actions: {action}, \n Values: {value}')

        actor_loss = actor_loss_fn(action, target_action.to(device).float())
        critic_loss = critic_loss_fn(value, target_value.to(device).float())
        actor_optimizer.zero_grad()
        critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        actor_optimizer.step()
        critic_optimizer.step()
        print(f'Epoch {i + 1} train end, loss: {actor_loss.item()}, loss: {critic_loss.item()}')












