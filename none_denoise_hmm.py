import os
import numpy as np
import librosa
from hmmlearn import hmm
from sklearn.metrics import accuracy_score
from collections import Counter
from joblib import Parallel, delayed  # ✅ 新增导入

# 参数
n_mfcc = 13
n_states = 12
n_models = 80
random_seeds = [i for i in range(n_models)]

# 特征提取
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc.T

# 数据加载
def load_data(base_path):
    train_data = {}
    test_data = {}
    for digit in range(10):
        digit_path = os.path.join(base_path, f'digit_{digit}')
        files = sorted([f for f in os.listdir(digit_path) if f.endswith('.wav')])
        train_data[digit] = [extract_features(os.path.join(digit_path, f)) for f in files[:15]]
        test_data[digit] = [extract_features(os.path.join(digit_path, f)) for f in files[15:]]
    return train_data, test_data

# ✅ 单个模型训练函数（供并行用）
def train_model_for_seed(seed, train_data):
    model_dict = {}
    for digit in range(10):
        model = hmm.GaussianHMM(n_components=n_states, covariance_type='diag', n_iter=100, random_state=seed)
        X = np.vstack(train_data[digit])
        lengths = [len(x) for x in train_data[digit]]
        model.fit(X, lengths)
        model_dict[digit] = model
    return model_dict

# ✅ 并行训练所有模型
def train_multiple_models_parallel(train_data):
    model_dicts = Parallel(n_jobs=-1)(  # 使用所有CPU核心
        delayed(train_model_for_seed)(seed, train_data) for seed in random_seeds
    )
    return model_dicts

# 多模型投票预测
def predict_with_voting(model_dicts, test_data):
    y_true = []
    y_pred = []
    for digit in range(10):
        for sample in test_data[digit]:
            y_true.append(digit)
            scores_per_model = []
            for model_dict in model_dicts:
                scores = {i: model_dict[i].score(sample) for i in range(10)}
                predicted = max(scores, key=scores.get)
                scores_per_model.append(predicted)
            final_vote = Counter(scores_per_model).most_common(1)[0][0]
            y_pred.append(final_vote)
    return y_true, y_pred

if __name__ == '__main__':
    train_data, test_data = load_data('Numbers')
    model_dicts = train_multiple_models_parallel(train_data)  # ✅ 替换为并行训练函数
    y_true, y_pred = predict_with_voting(model_dicts, test_data)

    print("真实值:", y_true)
    print("预测值:", y_pred)
    acc = accuracy_score(y_true, y_pred)
    print("准确率: {:.2f}%".format(acc * 100))
