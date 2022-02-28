from lstm import MyLSTM
from utils import evaluate, load_data, plot_history, plot_roc

device = "cuda:0"
#
random_state = 0
data_name = "human_count/run_free"
#

if __name__ == "__main__":
    # 加载数据
    classes = [f"{i}_free" for i in range(6)]
    print(classes)
    data = load_data(f"data/{data_name}", classes, seq_len=200, seq_step=200)
    # 训练分类器
    net = MyLSTM(seq_len=200, d_in=30, d_out=len(classes), d_hidden=64).to(device)
    his = net.fit((data["X_train"], data["y_train"]), (data["X_validate"], data["y_validate"]), device=device)
    net.evaluate((data["X_evaluate"], data["y_evaluate"]), device=device)
    # 评估模型
    y_pred = net.predict(data["X_evaluate"], device=device)
    evaluate(data["y_evaluate"], y_pred)
    # 画出ROC曲线
    Y_score = net.predict_proba(data["X_evaluate"], device=device)
    plot_roc(data["Y_evaluate"], Y_score, classes, title=f"LSTM({data_name})", out_file="out/lstm.roc.png")
    #
    plot_history(his, out_file="out/lstm.history.png")
