import os
import subprocess
import tempfile
import tkinter as tk
from tkinter import messagebox
from playsound import playsound

# 允许的输入
valid_words = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]

# 模型与配置路径
model_10374 = "D:/pth/best_model_10374.pth"
model_11058 = "D:/pth/best_model_11058.pth"
config_path = "D:/pth/config.json"

def synthesize_and_play(text):
    # 临时文件路径
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    # 选择模型
    if text == "nine":
        model_path = model_11058
    else:
        model_path = model_10374

    # 构建 tts 命令
    cmd = [
        "tts",
        "--text", text,
        "--model_path", model_path,
        "--config_path", config_path,
        "--out_path", tmp_path
    ]

    try:
        subprocess.run(cmd, check=True)
        playsound(tmp_path)
    except Exception as e:
        messagebox.showerror("错误", f"语音合成或播放失败:\n{e}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def on_submit():
    user_input = entry.get().strip().lower()
    if user_input not in valid_words:
        messagebox.showwarning("无效输入", "请输入 zero 至 nine 之间的英文单词。")
    else:
        synthesize_and_play(user_input)

# 创建 GUI
root = tk.Tk()
root.title("数字语音合成")

label = tk.Label(root, text="请输入 zero 至 nine 的英文单词：")
label.pack(pady=10)

entry = tk.Entry(root, width=30)
entry.pack(pady=5)

submit_btn = tk.Button(root, text="合成并播放", command=on_submit)
submit_btn.pack(pady=10)

root.mainloop()
