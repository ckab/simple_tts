import os
import csv

# 设置输入和输出路径
wav_dir = "digit_one/wavs"
output_csv = "digit_one/metadata.csv"

# 获取所有 .wav 文件
wav_files = [f for f in os.listdir(wav_dir) if f.endswith(".wav")]

# 按照文件名排序（可选）
wav_files.sort()

# 生成 metadata.csv 内容
with open(output_csv, "w", encoding="utf-8", newline="") as csvfile:
    writer = csv.writer(csvfile, delimiter="|")
    for wav_file in wav_files:
        base_name = os.path.splitext(wav_file)[0]
        text = "one"
        writer.writerow([base_name, text])

print(f"metadata.csv 已生成，包含 {len(wav_files)} 条记录。")
