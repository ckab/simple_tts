import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft

# 设置基本参数
root_folder = "wave"
subfolders = ["wave_1", "wave_2", "wave_3"]
digit_words = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
max_duration = 3.0  # 取前 3 秒
sr_target = 22050   # 统一采样率

def plot_waveform(ax, y, sr, title):
    t = np.linspace(0, len(y) / sr, num=len(y))
    ax.plot(t, y)
    ax.set_title(title, fontsize=8)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude")

def plot_spectrum(ax, y, sr, title):
    N = len(y)
    Y = np.abs(fft(y))[:N // 2]
    freqs = np.linspace(0, sr / 2, N // 2)
    ax.plot(freqs, Y)
    ax.set_title(title, fontsize=8)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Magnitude")

# 主循环
for folder in subfolders:
    wav_dir = os.path.join(root_folder, folder)

    # 画时间幅度图
    fig1, axs1 = plt.subplots(2, 5, figsize=(18, 6))
    fig1.suptitle(f"Waveform - {folder}", fontsize=14)

    # 画频谱图
    fig2, axs2 = plt.subplots(2, 5, figsize=(18, 6))
    fig2.suptitle(f"Spectrum - {folder}", fontsize=14)

    for i, word in enumerate(digit_words):
        filepath = os.path.join(wav_dir, f"{word}.wav")
        y, sr = librosa.load(filepath, sr=sr_target, duration=max_duration)

        row, col = divmod(i, 5)

        # 时间幅度图
        plot_waveform(axs1[row, col], y, sr, f"{word}")

        # 频谱图
        plot_spectrum(axs2[row, col], y, sr, f"{word}")

    # 保存图像
    fig1.tight_layout(rect=[0, 0, 1, 0.95])
    fig2.tight_layout(rect=[0, 0, 1, 0.95])
    fig1.savefig(os.path.join(wav_dir, f"waveform_{folder}.png"))
    fig2.savefig(os.path.join(wav_dir, f"spectrum_{folder}.png"))
    plt.close(fig1)
    plt.close(fig2)

print("分析完成，图像已保存到各文件夹。")
