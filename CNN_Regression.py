"""
    作者：Troublemaker
    日期：2020/4/11 10:59
    脚本：rnn_regression.py
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class LSTM(nn.Module):
    """搭建rnn网络"""
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,)
        self.output_layer = nn.Linear(in_features=hidden_size, out_features=output_size)

    def forward(self, x, hc):
        # x (batch, time_step, input_size)
        # h_state (n_layers, batch, hidden_size)
        # rnn_out (batch, time_step, hidden_size)
        rnn_out, hc = self.lstm(x, hc)   # h_state是之前的隐层状态

        out = []
        for time in range(rnn_out.size(1)):
            every_time_out = rnn_out[:, time, :]       # 相当于获取每个时间点上的输出，然后过输出层
            temp = self.output_layer(every_time_out)
            out.append(temp)
            # print(f'Time={time}', rnn_out.shape, every_time_out.shape, temp.shape, len(out))
        return torch.stack(out, dim=1), hc       # torch.stack扩成[1, output_size, 1]

class RNN(nn.Module):
    """搭建rnn网络"""
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,)
        self.output_layer = nn.Linear(in_features=hidden_size, out_features=output_size)

    def forward(self, x, hc):
        # x (batch, time_step, input_size)
        # h_state (n_layers, batch, hidden_size)
        # rnn_out (batch, time_step, hidden_size)
        rnn_out, hc = self.rnn(x, hc)   # h_state是之前的隐层状态

        out = []
        for time in range(rnn_out.size(1)):
            every_time_out = rnn_out[:, time, :]       # 相当于获取每个时间点上的输出，然后过输出层
            temp = self.output_layer(every_time_out)
            out.append(temp)
            # print(f'Time={time}', rnn_out.shape, every_time_out.shape, temp.shape, len(out))
        return torch.stack(out, dim=1), hc       # torch.stack扩成[1, output_size, 1]

# 设置超参数
input_size = 1
output_size = 1
num_layers = 1
hidden_size = 32
learning_rate = 0.02
train_step = 100
time_step = 10

# 准备数据
steps = np.linspace(0, 2*np.pi, 100, dtype=np.float32)
x_np = np.sin(steps)
y_np = np.cos(steps)

# plt.plot(steps, y_np, 'r-', label='target (cos)')
# plt.plot(steps, x_np, 'b-', label='input (sin)')
# plt.legend(loc='best')
# plt.show()

# rnn = RNN()
# print(rnn)

lstm = LSTM()
print(lstm)

# 设置优化器和损失函数
# optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
loss_function = nn.MSELoss()

plt.figure(1, figsize=(12, 5))
plt.ion()

# 训练
h_state = None   # 初始化隐藏层状态

for step in range(train_step):
    start, end = step * np.pi, (step+1) * np.pi
    steps = np.linspace(start, end, time_step, dtype=np.float32)
    x_np = np.sin(steps)
    y_np = np.cos(steps)

    x = torch.from_numpy(x_np[np.newaxis, :, np.newaxis])
    y = torch.from_numpy(y_np[np.newaxis, :, np.newaxis])
    # print('x y shape:', x.shape, y.shape)
    
    # pridect, h_state = rnn(x, h_state)
    # h_state = h_state.detach()     # 重要！！！ 需要将该时刻隐藏层的状态作为下一时刻rnn的输入

    pridect, h_state = lstm(x, h_state)
    h_state = (h_state[0].detach(), h_state[1].detach())
    # print('shape:', pridect.shape, y.shape)
    # exit()
    # if step == 50:
    #     exit()
    loss = loss_function(pridect, y)
    optimizer.zero_grad()

    loss.backward()
    optimizer.step()

    # plotting
    plt.plot(steps, y_np.flatten(), 'r-')
    plt.plot(steps, pridect.detach().numpy().flatten(), 'b-')
    plt.draw()
    plt.pause(0.05)

plt.ioff()
plt.show()


