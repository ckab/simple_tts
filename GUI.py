import os
import subprocess
import tempfile
import tkinter as tk
from tkinter import messagebox
from playsound import playsound
import soundfile as sf
import numpy as np

# 有效输入和模型路径
valid_words = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
model_10374 = "D:/pth/best_model_10374.pth"
model_11058 = "D:/pth/best_model_11058.pth"
config_path = "D:/pth/config.json"
sample_rate = 22050
duration_sec = 1.0 # 每段音频取前1.5秒


def synthesize_single_word(word):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    model_path = model_11058 if word == "nine" else model_10374

    cmd = [
        "tts",
        "--text", word,
        "--model_path", model_path,
        "--config_path", config_path,
        "--out_path", tmp_path
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        audio, sr = sf.read(tmp_path)
        trimmed = audio[:int(duration_sec * sr)]
        return trimmed
    except Exception as e:
        raise RuntimeError(f"{word} 合成失败: {e}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def synthesize_and_play_sequence(text):
    words = text.strip().lower().split()
    if not all(word in valid_words for word in words):
        messagebox.showwarning("无效输入", "请只输入 zero 至 nine 的英文单词，中间用空格分隔。")
        return

    try:
        segments = [synthesize_single_word(word) for word in words]
        full_audio = np.concatenate(segments)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_out:
            out_path = tmp_out.name
            sf.write(out_path, full_audio, sample_rate)

        playsound(out_path)
    except Exception as e:
        messagebox.showerror("错误", f"合成或播放失败:\n{e}")
    finally:
        if 'out_path' in locals() and os.path.exists(out_path):
            os.remove(out_path)


def on_submit():
    user_input = entry.get()
    synthesize_and_play_sequence(user_input)


# GUI 界面
root = tk.Tk()
root.title("数字语音合成器")

label = tk.Label(root, text="请输入 zero 到 nine 的英文单词（可多个，用空格隔开）：")
label.pack(pady=10)

entry = tk.Entry(root, width=50)
entry.pack(pady=5)

submit_btn = tk.Button(root, text="合成并播放", command=on_submit)
submit_btn.pack(pady=10)

root.mainloop()
