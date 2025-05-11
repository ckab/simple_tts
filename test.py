import os
from TTS.api import TTS

# 创建输出目录
output_dir = "wave/wave_3"
os.makedirs(output_dir, exist_ok=True)

# 文本与文件名映射
digit_words = [
    "zero", "one", "two", "three", "four",
    "five", "six", "seven", "eight", "nine"
]

# 初始化 TTS 模型
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=True, gpu=True)

# 依次合成语音
for word in digit_words:
    output_path = os.path.join(output_dir, f"{word}.wav")
    tts.tts_to_file(text=word, file_path=output_path)
    print(f"Saved: {output_path}")
