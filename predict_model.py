import os
import numpy as np
import librosa
from joblib import load
from collections import Counter, defaultdict
from sklearn.metrics import accuracy_score
from tqdm import tqdm

n_mfcc = 13
model_dir = 'model'
test_data_dir = 'Numbers_denoise_aug'

# 特征提取函数
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc.T

# 加载测试数据：digit_0 ~ digit_9 中的后5个文件
def load_test_data(base_path):
    test_data = {}
    for digit in range(10):
        digit_path = os.path.join(base_path, f'digit_{digit}')
        files = sorted([f for f in os.listdir(digit_path) if f.endswith('.wav')])
        selected = files[60:65]  # 后5个文件
        test_data[digit] = [extract_features(os.path.join(digit_path, f)) for f in selected]
    return test_data

# 加载所有模型
def load_all_models(model_dir='model'):
    model_dicts = []
    files = sorted([f for f in os.listdir(model_dir) if f.endswith('.pkl')])
    for file in tqdm(files, desc='加载模型'):
        path = os.path.join(model_dir, file)
        model_dicts.append(load(path))
    return model_dicts

# 多模型投票预测
def predict_with_voting(model_dicts, test_data):
    y_true = []
    y_pred = []
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    print("开始预测...")
    for digit in range(10):
        for sample in test_data[digit]:
            y_true.append(digit)
            class_total[digit] += 1

            votes = []
            for model_dict in model_dicts:
                scores = {i: model_dict[i].score(sample) for i in range(10)}
                votes.append(max(scores, key=scores.get))
            final = Counter(votes).most_common(1)[0][0]
            y_pred.append(final)

            if final == digit:
                class_correct[digit] += 1

    return y_true, y_pred, class_correct, class_total

if __name__ == '__main__':
    print("[1] 正在加载测试数据...")
    test_data = load_test_data(test_data_dir)

    print("[2] 正在加载模型...")
    model_dicts = load_all_models(model_dir)

    print("[3] 正在进行多模型投票预测...")
    y_true, y_pred, class_correct, class_total = predict_with_voting(model_dicts, test_data)

    acc = accuracy_score(y_true, y_pred)
    print("\n✅ 总体准确率: {:.2f}%".format(acc * 100))

    print("\n📊 每类预测准确率:")
    for digit in range(10):
        correct = class_correct[digit]
        total = class_total[digit]
        acc_digit = (correct / total) * 100 if total else 0
        print(f"Digit {digit}: {correct}/{total} 正确，准确率 = {acc_digit:.2f}%")
