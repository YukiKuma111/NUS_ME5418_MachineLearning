import torch
import gym
import numpy as np
from model import Policy  # 确保model.py中Policy类已经实现

# 假设obs_shape和action_space已经定义
obs_shape = (35,)  # 假设一维输入，128个特征（根据实际情况调整）
action_space = gym.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)  # 使用np.float32

# 1. 初始化Policy模型
model = Policy(obs_shape, action_space)

# 2. 加载示例数据（假设使用随机数据模拟）
example_input = torch.rand(1, *obs_shape)  # 示例输入（批大小为1）
rnn_hxs = torch.zeros(1, model.recurrent_hidden_state_size)  # 初始隐藏状态
masks = torch.ones(1, 1)  # 掩码（1表示有效输入）

# 3. 进行推理（动作生成）
with torch.no_grad():
    value, action, action_log_probs, _ = model.act(example_input, rnn_hxs, masks, deterministic=False)

# 4. 打印推理结果
print("Value:", value)
print("Action:", action)
print("Action Log Probabilities:", action_log_probs)
