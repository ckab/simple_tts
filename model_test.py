import os
import subprocess

# 要合成的文本（数字0~9）
texts = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]

# 模型配置
models = [
    {
        "model_path": "D:/pth/best_model_10374.pth",
        "config_path": "D:/pth/config.json",
        "output_dir": "wave/wave_1"
    },
    {
        "model_path": "D:/pth/best_model_11058.pth",
        "config_path": "D:/pth/config.json",
        "output_dir": "wave/wave_2"
    }
]

# 创建输出目录
for model in models:
    os.makedirs(model["output_dir"], exist_ok=True)

# 合成语音
for model_index, model in enumerate(models, start=1):
    for i, text in enumerate(texts):
        output_path = os.path.join(model["output_dir"], f"{text}.wav")
        cmd = [
            "tts",
            "--text", text,
            "--model_path", model["model_path"],
            "--config_path", model["config_path"],
            "--out_path", output_path
        ]
        print(f"[模型{model_index}] 正在合成: '{text}' -> {output_path}")
        subprocess.run(cmd, check=True)

print("全部语音合成完成。")
