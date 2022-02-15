import os

from sklearn.model_selection import train_test_split
import numpy as np
from lstm import MyLSTM
from utils import evaluate, load_data, plot_roc

device = "cuda:0"
#
data_name = "500hz_csi_data/human_count/run_circle"
#
file_path = "data/500hz_csi_data/human_count/run_circle/1_circle/target/1ren_3m_interval0.002s_circle_0wave.csv"

if __name__ == "__main__":
    classes = os.listdir(f"data/{data_name}")
    # 加载数据
    length, stride = 300, 10
    arr = np.loadtxt(file_path, delimiter=",")
    X = np.array([arr[s : s + length] for s in np.arange(0, len(arr) - length + 1, stride)])
    # 加载分类器
    net = MyLSTM(seq_len=300, d_in=30, d_out=5, d_hidden=64).to(device)
    net.load("out/lstm.pth")
    # 预测
    y = net.predict(X, device=device)
    print(y)
    c = np.bincount(y).argmax()
    print(f"Prediction result: {classes[c]}")
