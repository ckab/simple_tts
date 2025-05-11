import os
import librosa
import soundfile as sf
import numpy as np
import noisereduce as nr
from scipy.signal import convolve

# 参数设置
root_dir = "wave"
output_root = "wave_denoise"
subfolders = ["wave_1", "wave_2", "wave_3"]
digit_words = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
sr_target = 22050
max_duration = 3.0

# 移动平均去噪
def moving_average(signal, window_size=5):
    window = np.ones(window_size) / window_size
    return convolve(signal, window, mode='same')

# 处理每个文件夹
for folder in subfolders:
    input_dir = os.path.join(root_dir, folder)
    output_dir = os.path.join(output_root, folder)
    os.makedirs(output_dir, exist_ok=True)

    for word in digit_words:
        filepath = os.path.join(input_dir, f"{word}.wav")
        y, sr = librosa.load(filepath, sr=sr_target, duration=max_duration)

        # 方法1: 静音裁剪
        y_trim, _ = librosa.effects.trim(y, top_db=60)

        # 方法2: 频谱去噪
        y_nr = nr.reduce_noise(y=y_trim, sr=sr)

        # 方法3: 移动平均滤波
        y_final = moving_average(y_nr, window_size=5)

        # 写入输出文件
        output_path = os.path.join(output_dir, f"{word}.wav")
        sf.write(output_path, y_final, sr)

print("去噪处理完成，文件保存至 wave_denoise 目录。")
