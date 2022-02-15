from typing import Tuple

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

batch_size = 32


class MyLSTM(nn.Module):
    def __init__(self, seq_len: int, d_in: int, d_out: int, d_hidden: int, lstm_layers: int = 4):
        super(MyLSTM, self).__init__()
        #
        self.in_proj = nn.Linear(in_features=d_in, out_features=d_hidden)  # 输入映射
        self.norm = nn.LayerNorm(normalized_shape=d_hidden)
        self.lstm = nn.LSTM(d_hidden, d_hidden, num_layers=lstm_layers)  # RNN网络
        self.aggr = nn.Conv1d(in_channels=seq_len, out_channels=1, kernel_size=1)  # 聚合函数
        self.out_proj = nn.Linear(in_features=d_hidden, out_features=d_out)  # 输出映射
        #
        self.X_mean, self.X_std = None, None
        #
        self.reset_parameters()

    def forward(self, x: torch.Tensor):
        x = self.in_proj(x)
        x = self.norm(x)
        x, _ = self.lstm(x)
        x = self.aggr(x)
        x = self.out_proj(x)
        return torch.softmax(x[:, 0], dim=-1)

    def predict(self, X: np.ndarray, device: str = "cuda:0"):
        proba = self.predict_proba(X, device=device)
        return np.argmax(proba, axis=-1)

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray, device: str = "cuda:0"):
        X_norm = torch.from_numpy(X).to(device=device, dtype=torch.float)
        X_norm -= self.X_mean
        X_norm /= self.X_std
        #
        outputs = torch.cat([self(x).cpu() for x in DataLoader(TensorDataset(X_norm), batch_size, shuffle=False)])
        return outputs.numpy().astype(float)

    def save(self, file: str):
        states = {
            "net": self.state_dict(),
            "X_std": self.X_std,
            "X_mean": self.X_mean,
        }
        torch.save(states, file)

    def load(self, file: str):
        states = torch.load(file)
        self.load_state_dict(states["net"])
        self.X_std, self.X_mean = states["X_std"], states["X_mean"]

    def fit(
        self,
        D_train: Tuple[torch.Tensor, torch.Tensor],
        D_test: Tuple[torch.Tensor, torch.Tensor],
        epochs: int = 100,
        device: str = "cuda:0",
    ):
        #
        X_train = torch.from_numpy(D_train[0]).to(device=device, dtype=torch.float)
        y_train = torch.from_numpy(D_train[1]).to(device=device, dtype=torch.long)
        #
        self.X_std, self.X_mean = torch.std_mean(X_train, dim=0, keepdim=True)
        #
        X_train = (X_train - self.X_mean) / self.X_std
        D_train = DataLoader(TensorDataset(X_train, y_train), batch_size, shuffle=True)
        #
        X_test = torch.from_numpy(D_test[0]).to(device=device, dtype=torch.float)
        y_test = torch.from_numpy(D_test[1]).to(device=device, dtype=torch.long)
        #
        X_test = (X_test - self.X_mean) / self.X_std
        D_test = DataLoader(TensorDataset(X_test, y_test), batch_size, shuffle=False)
        #
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        #
        epoch = 1
        while True:
            L_train, _ = self._run_epoch(D_train, loss_fn, optimizer)
            L_test, outputs = self._run_epoch(D_test, loss_fn)
            accuracy = torch.sum(torch.argmax(outputs, -1) == y_test) / len(outputs)
            print(f"epoch = {epoch}, loss_train = {L_train:.3f}, loss_test = {L_test:.3f}, accuracy = {accuracy:.3f}")
            if epoch >= epochs:
                return outputs.numpy().astype(float)
            epoch += 1

    def reset_parameters(self):
        for param in self.parameters():
            if param.ndim >= 2:
                nn.init.xavier_normal_(param)
            else:
                f_out = param.size(0)
                nn.init.uniform_(param, -(f_out ** -0.5), f_out ** -0.5)

    def _run_epoch(self, data: DataLoader, loss_fn: nn.Module, optimizer: optim.Optimizer = None):
        is_train = optimizer is not None
        with torch.set_grad_enabled(is_train):
            loss_acc, loss_ave, outputs = 0, 0, []
            for batch_idx, (x, y) in enumerate(data):
                output = self(x)
                loss = loss_fn(output, y)
                if is_train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                loss_acc += loss.item()
                loss_ave = loss_acc / (batch_idx + 1)
                outputs += [output.cpu()]
        return loss_ave, torch.cat(outputs)
