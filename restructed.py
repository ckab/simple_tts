import os
import shutil

SOURCE_DIR = "Numbers_denoise_aug"  # 原始数据集路径
DEST_DIR = "Numbers_denoise_aug_re"  # 目标目录路径

# 创建目标目录
wavs_dir = os.path.join(DEST_DIR, "wavs")
if not os.path.exists(wavs_dir):
    os.makedirs(wavs_dir)

# 遍历源数据集中的所有文件夹和文件
for digit in range(10):  # 假设有10个数字文件夹（digit_0, digit_1, ..., digit_9）
    src_folder = os.path.join(SOURCE_DIR, f"digit_{digit}")

    # 如果该文件夹存在，开始处理其中的文件
    if os.path.exists(src_folder):
        for file in os.listdir(src_folder):
            if file.endswith(".wav"):  # 只处理.wav文件
                # 源文件路径
                src_file_path = os.path.join(src_folder, file)

                # 目标文件名：digitX_1_0.wav 等等
                new_file_name = f"digit{digit}_{file}"
                dest_file_path = os.path.join(wavs_dir, new_file_name)

                # 移动文件
                shutil.copy(src_file_path, dest_file_path)
                print(f"[✓] 文件 {file} 已成功移动到 {dest_file_path}")
