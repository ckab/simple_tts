import os

from TTS.config.shared_configs import BaseAudioConfig
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.tacotron2_config import Tacotron2Config
from TTS.utils.capacitron_optimizer import CapacitronOptimizer

# 数据集和音频配置
data_path = "./Numbers_denoise_aug_re"  # 数据集路径

dataset_config = BaseDatasetConfig(
    formatter="ljspeech",  # 使用 LJSpeech 格式
    meta_file_train="metadata.csv",  # 元数据文件路径
    path=data_path,  # 数据集目录
)

# 音频配置
audio_config = BaseAudioConfig(
    sample_rate=22050,  # 采样率
    do_trim_silence=True,  # 去除静音
    trim_db=60.0,  # 剪辑的 DB 阈值
    signal_norm=True,  # 信号归一化
    mel_fmin=0.0,  # 最小频率
    mel_fmax=8000,  # 最大频率
    spec_gain=1.0,  # 增益
    log_func="np.log",  # 对数函数
    ref_level_db=20,  # 参考级别
    preemphasis=0.0,  # 预加重
)

# Tacotron2 模型配置
config = Tacotron2Config(
    run_name="digits_tacotron2",  # 训练名称
    audio=audio_config,  # 音频配置
    batch_size=8,  # 批大小
    max_audio_len=10 * 22050,  # 最大音频长度
    min_audio_len=0,  # 最小音频长度
    eval_batch_size=8,  # 评估批大小
    num_loader_workers=8,  # 数据加载工作者数量
    num_eval_loader_workers=12,  # 评估数据加载工作者数量
    precompute_num_workers=24,  # 预计算工作者数量
    epochs=20,  # 训练周期数
    lr=1e-3,  # 学习率
    # optimizer="CapacitronOptimizer",  # 优化器
    # optimizer_params={
    #     "RAdam": {"betas": [0.9, 0.998], "weight_decay": 1e-6},
    #     "SGD": {"lr": 1e-5, "momentum": 0.9},
    # },
    attention_type="dynamic_convolution",  # 使用动态卷积
    grad_clip=0.0,  # 梯度裁剪
    text_cleaner="basic_cleaners",  # 基本文本清理
    use_phonemes=False,  # 不使用音素
    use_capacitron_vae=False,  # 关闭 Capacitron VAE
    mixed_precision=False,  # 不使用混合精度
    output_path="./output",  # 输出路径
    datasets=[dataset_config],  # 数据集配置
)

# 如果需要，可以打印此配置并保存为文件
print(config)
