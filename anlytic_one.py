import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from scipy.io import wavfile

# 读取音频文件路径
wav_path = r"D:\pth_one\speech.wav"

# 使用 librosa 加载音频
y, sr = librosa.load(wav_path, sr=None)  # 保留原始采样率

# 保留前3秒的数据
duration = 3  # 秒
max_len = int(sr * duration)
y = y[:max_len]

# 时间轴（用于时间-幅度图）
time = np.linspace(0, len(y) / sr, num=len(y))

# 绘图
plt.figure(figsize=(12, 6))

# --- 时间-幅度分析 ---
plt.subplot(2, 1, 1)
plt.plot(time, y, color='blue')
plt.title("Time-Amplitude Waveform")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)

# --- 频谱分析 ---
plt.subplot(2, 1, 2)
D = np.abs(librosa.stft(y, n_fft=1024, hop_length=256))
DB = librosa.amplitude_to_db(D, ref=np.max)
librosa.display.specshow(DB, sr=sr, hop_length=256, x_axis='time', y_axis='hz', cmap='magma')
plt.colorbar(format="%+2.0f dB")
plt.title("Spectrogram (dB)")

plt.tight_layout()
plt.show()
