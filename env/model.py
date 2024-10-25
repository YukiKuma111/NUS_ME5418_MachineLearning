import torch
import torch.nn as nn

from distributions import DiagGaussian
from utils import init, init_normc_


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, base_kwargs=None):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}

        if len(obs_shape) == 1:
            self.base = MLPBase(obs_shape[0], **base_kwargs)
        else:
            raise NotImplementedError

        num_outputs = action_space.shape[0]
        self.dist = DiagGaussian(self.base.output_size, num_outputs)

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):  # 实际前向传播是act和evaluate_actions
        raise NotImplementedError

    # 从策略网络中获取动作（action）以及相关信息，如动作的概率对数（action_log_probs）和对应的状态值（value）
    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)  # 创建动作分布（高斯）

        # deterministic: 是否以确定性方式选取动作。
        # 如果为True，会直接选择动作分布的均值（mode）【确定性策略】，否则会根据动作分布进行采样【随机探索】。
        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)  # 动作的概率对数

        return value, action, action_log_probs, rnn_hxs

    # 获取状态的价值估计（value）
    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    # 评估给定动作的表现，计算其对数概率、熵值和状态价值
    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()  # 计算熵。熵越高，策略越随机，越倾向于探索

        return value, action_log_probs, dist_entropy, rnn_hxs


class NNBase(nn.Module):

    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRUCell(recurrent_input_size, hidden_size)
            nn.init.orthogonal_(self.gru.weight_ih.data)
            nn.init.orthogonal_(self.gru.weight_hh.data)
            self.gru.bias_ih.data.fill_(0)
            self.gru.bias_hh.data.fill_(0)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x = hxs = self.gru(x, hxs * masks)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N, 1)

            outputs = []
            for i in range(T):
                hx = hxs = self.gru(x[i], hxs * masks[i])
                outputs.append(hx)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.stack(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)

        return x, hxs


class Print(nn.Module):
    def __init__(self):
        super(Print, self).__init__()

    def forward(self, x):
        print('layer input:', x.shape)
        return x


class MLPBase(NNBase):
    def __init__(self, num_inputs, recurrent=True, hidden_size=64):
        super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size)

        if recurrent:
            num_inputs = hidden_size

        # 初始化网络层的权重和偏置
        init_ = lambda m: init(m,
                               init_normc_,
                               lambda x: nn.init.constant_(x, 0))

        # 策略网络（用于选择动作）
        # 最后的输出会被用作生成动作的输入
        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)),
            nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh()
        )

        # 用于计算状态价值
        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)),
            nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh()
        )

        # 全连接层（线性层），将critic网络的输出转换为标量值（状态的价值）
        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_critic = self.critic(x)  # 与价值函数相关
        hidden_actor = self.actor(x)  # 与策略（动作生成）相关

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs

