import os
import csv

# 原始 CSV 文件路径
original_csv_path = "Numbers_denoise_aug_re/metadata.csv"
# 新的 CSV 文件路径
new_csv_path = "Numbers_denoise_aug_re/new_metadata.csv"

# 读取原始 CSV 文件并创建新的 CSV 文件
with open(original_csv_path, 'r', encoding='utf-8') as f_in, open(new_csv_path, 'w', newline='',
                                                                  encoding='utf-8') as f_out:
    reader = csv.reader(f_in, delimiter='|')
    writer = csv.writer(f_out, delimiter='|')

    # 处理每一行，确保格式符合 (文件路径|文本)
    for row in reader:
        if len(row) == 2:  # 确保原始文件是正确的
            file_name = row[0]
            text = row[1].strip()  # 去掉前后空白符
            # 将文件路径转换为相对路径
            file_path = os.path.join("wavs", file_name)
            # 写入新的 CSV 文件
            writer.writerow([file_path, text])

print(f"New CSV file saved as {new_csv_path}")
