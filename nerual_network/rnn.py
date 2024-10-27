import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from docutils.nodes import target
from torch.ao.quantization import obs_or_fq_ctr_equals



# class of actor
class PolicyNetwork(nn.Module):
    def __init__(self, obs_space, hidden_size_gru, hidden_size_MLP, action_space, num_layers):
        super(PolicyNetwork, self).__init__()
        # gru layer definition
        self.gru_policy_1 = nn.GRU(input_size=obs_space, hidden_size=hidden_size_gru, num_layers=num_layers)

        # gru layer init
        #nn.init.orthogonal_(self.gru_policy_1.weight_ih.data)
        #nn.init.orthogonal_(self.gru_policy_1.weight_hh.data)
        #self.gru_policy_1.bias_ih.data.fill_(0)
        #self.gru_policy_1.bias_hh.data.fill_(0)

        # two MLP layer to get action with normalization to improve the training speed
        self.MLP_1 = nn.Linear(hidden_size_gru, hidden_size_MLP)
        self.MLP_2 = nn.Linear(hidden_size_MLP, action_space)

    def forward(self, input):# input = (seq_len,batch_size,input_size)
        # gru layer
        output_set_gru,_ = self.gru_policy_1(input)
        squ_gru,batch_gru,output_gru = output_set_gru.shape
        inputs_of_MLP_1 = output_set_gru.view(squ_gru * batch_gru,output_gru)

        # first MLP layer
        output_of_MLP_1 = torch.tanh(self.MLP_1(inputs_of_MLP_1))

        # second MLP layer
        action = torch.tanh(self.MLP_2(output_of_MLP_1))

        action = action.view(squ_gru, batch_gru, -1)

        return action


# class of critic
class ValueNetwork(nn.Module):
    def __init__(self, obs_space, hidden_size_gru, hidden_size_MLP, num_layers):
        super(ValueNetwork, self).__init__()
        self.gru_value_1 = nn.GRU(input_size=obs_space,hidden_size= hidden_size_gru, num_layers=num_layers)

        # gru layer init
        #nn.init.orthogonal_(self.gru_value_1.weight_ih.data)
        #nn.init.orthogonal_(self.gru_value_1.weight_hh.data)
        #self.gru_value_1.bias_ih.data.fill_(0)
        #self.gru_value_1.bias_hh.data.fill_(0)

        # two MLP layer to get action with normalization to improve the training speed
        self.MLP_1 = nn.Linear(hidden_size_gru, hidden_size_MLP)
        self.ln_1 = nn.LayerNorm(hidden_size_MLP)
        self.MLP_2 = nn.Linear(hidden_size_MLP, 1)

    def forward(self, input):
        # gru layer
        output_set_gru,_ = self.gru_value_1(input)
        squ_gru,batch_gru,output_gru = output_set_gru.shape
        inputs_of_MLP_1 = output_set_gru.view(squ_gru * batch_gru, output_gru)

        # MLP layer 1
        output_of_MLP_1 = F.tanh(self.MLP_1(inputs_of_MLP_1))

        # MLP layer 2
        value = F.tanh(self.MLP_2(output_of_MLP_1))

        value = value.view(squ_gru,batch_gru,-1)

        return value

if __name__ == '__main__':
    target_action = torch.tensor([[1,0,1,0],
                                  [1,0,2,1],
                                  [1,0,0.2,0.5],
                                  [1,0.1,0.3,0.4],
                                  [0.2,0.3,0.4,0.8]])
    target_value = torch.tensor([[0.5],
                                 [0.4],
                                 [0.8],
                                 [0.4],
                                 [0.2]])
    # network params
    obs_space = 35
    hidden_size_gru = 20
    hidden_size_MLP =20
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
    actor_loss_fn = nn.CrossEntropyLoss()
    critic_loss_fn = nn.MSELoss()

    # assume the input data
    batch_size = 5
    seq_len = 10
    input_data = torch.randn(seq_len, batch_size, obs_space)  # 随机生成输入数据

    for i in range(epoch):
        print(f"{i + 1} times train")
        myactor.train()
        mycritic.train()

        # get action and value
        action = myactor(input_data)
        value = mycritic(input_data)

        action = action[-1]
        value = value[-1]
        print(f'Epoch {i + 1}, Actions: {action}, \n Values: {value}')

        actor_loss = actor_loss_fn(action, target_action)
        critic_loss = critic_loss_fn(value, target_value)
        actor_optimizer.zero_grad()
        critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        actor_optimizer.step()
        critic_optimizer.step()
        print(f'Epoch {i + 1} train end, loss: {actor_loss.item()}, loss: {critic_loss.item()}')












