import os

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from utils import load_data, plot_roc, evaluate

data_dir = "data/500hz_csi_data/human_count/run_circle"
labels = os.listdir(data_dir)
random_state = 0
batch_size = 32
device = 'cpu'


class MyNN(nn.Module):
    def __init__(self, seq_len, d_in, d_out, d_hidden, lstm_layers=4):
        super(MyNN, self).__init__()
        self.in_proj = nn.Linear(in_features=d_in, out_features=d_hidden)  # 输入映射
        self.lstm = nn.LSTM(d_hidden, d_hidden, num_layers=lstm_layers)  # RNN网络
        self.aggregate = nn.Conv1d(in_channels=seq_len, out_channels=1, kernel_size=1)  # 聚合函数
        self.out_proj = nn.Linear(d_hidden, d_out)  # 输出映射
        #
        self.reset_parameters()

    def forward(self, x):
        x = self.in_proj(x)
        x, (_, _) = self.lstm(x)
        x = self.aggregate(x)
        x = self.out_proj(x)
        return torch.softmax(x[:, 0], dim=-1)

    def fit(self, data_train: tuple, data_test: tuple, epochs=100):
        # ndarray to tenser
        X_train = torch.from_numpy(data_train[0]).float()
        y_train = torch.from_numpy(data_train[1]).long()
        X_test = torch.from_numpy(data_test[0]).float()
        y_test = torch.from_numpy(data_test[1]).long()
        data_train = DataLoader(TensorDataset(X_train, y_train), batch_size, shuffle=True)
        data_test = DataLoader(TensorDataset(X_test, y_test), batch_size, shuffle=False)
        #
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=1e-3)
        #
        outputs = None
        for epoch in range(epochs):
            L_train, _ = self._run_epoch(data_train, loss_fn, optimizer, mode='train')
            L_test, outputs = self._run_epoch(data_test, loss_fn, optimizer, mode='test')
            accuracy = torch.sum(torch.argmax(outputs, -1) == y_test) / len(outputs)
            print(f'epoch = {epoch}, loss_train = {L_train:.3f}, loss_test = {L_test:.3f}, accuracy = {accuracy:.3f}')
        return outputs.numpy()

    def reset_parameters(self):
        for param in self.parameters():
            if param.ndim >= 2:
                nn.init.xavier_normal_(param)
            else:
                f_out = param.size(0)
                nn.init.uniform_(param, -(f_out ** -0.5), f_out ** -0.5)

    def _run_epoch(self, data: DataLoader, loss_fn: nn.Module, optimizer: optim.Optimizer, mode='train'):
        is_train = mode == 'train'
        with torch.set_grad_enabled(is_train):
            outputs = []
            L_acc, L_ave = 0, 0
            for batch_idx, (inputs, target) in enumerate(data):
                inputs, target = inputs.to(device), target.to(device)
                output = self(inputs)
                loss = loss_fn(output, target)
                if is_train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                outputs += [output]
                L_acc += loss.item()
                L_ave = L_acc / (batch_idx + 1)
        return L_ave, torch.cat(outputs)


if __name__ == "__main__":
    # 加载数据
    X, y, Y = load_data(data_dir, labels, length=300, stride=30)
    # 切分数据
    X_train, X_test = train_test_split(X, test_size=0.4, random_state=random_state)
    y_train, y_test = train_test_split(y, test_size=0.4, random_state=random_state)
    Y_train, Y_test = train_test_split(Y, test_size=0.4, random_state=random_state)
    # 训练分类器
    net = MyNN(seq_len=300, d_in=30, d_out=5, d_hidden=64).to(device)
    Y_score = net.fit((X_train, y_train), (X_test, y_test))
    # 评估模型
    y_pred = np.argmax(Y_score, -1)
    evaluate(y_test, y_pred)
    # 画出ROC曲线
    plot_roc(Y_test, Y_score, labels, title="LSTM - human_count/run_circle", out_file="out/lstm.roc.png")
