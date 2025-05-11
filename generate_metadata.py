import os
import csv

SOURCE_DIR = "Numbers_denoise_aug_re"  # 目标数据集路径
CSV_FILE_PATH = os.path.join(SOURCE_DIR, "metadata.csv")  # CSV文件保存路径

# 数字到英文的映射
digit_to_word = {
    0: "zero",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine"
}

# 获取wavs文件夹路径
wavs_dir = os.path.join(SOURCE_DIR, "wavs")

# 生成CSV内容
metadata = []

for filename in os.listdir(wavs_dir):
    if filename.endswith(".wav"):
        # 提取数字部分
        digit_part = filename.split('_')[0][5]  # 提取文件名中的数字部分，文件名格式为 "digitX_"
        digit = int(digit_part)  # 获取数字

        # 根据数字获取对应的英文单词
        text = digit_to_word[digit]

        # 移除 .wav 后缀
        filename_without_ext = os.path.splitext(filename)[0]

        # 创建一行数据，格式为: 文件名(无后缀) | 对应的英文数字
        metadata.append([filename_without_ext, text])

# 保存到CSV文件
with open(CSV_FILE_PATH, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f, delimiter='|')
    writer.writerow(["filename", "text"])  # 写入标题行
    writer.writerows(metadata)

print(f"[✓] CSV文件已生成并保存在 {CSV_FILE_PATH}")
