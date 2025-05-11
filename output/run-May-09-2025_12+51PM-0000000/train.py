import os

from trainer import Trainer, TrainerArgs
from TTS.config.shared_configs import BaseAudioConfig
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.tacotron2_config import Tacotron2Config
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.tacotron2 import Tacotron2
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

# 设置输出路径为当前文件夹下的 output 子目录
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

# 数据路径
data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Numbers_denoise_aug_re/")

# 配置数据集
dataset_config = BaseDatasetConfig(
    formatter="ljspeech",
    meta_file_train="metadata.csv",
    path=data_path,
)

# 音频配置
audio_config = BaseAudioConfig(
    sample_rate=22050,
    do_trim_silence=True,
    trim_db=60.0,
    signal_norm=False,
    mel_fmin=0.0,
    mel_fmax=8000,
    spec_gain=1.0,
    log_func="np.log",
    ref_level_db=20,
    preemphasis=0.0,
)

# Tacotron2 配置（小数据集优化）
config = Tacotron2Config(
    audio=audio_config,
    batch_size=8,
    eval_batch_size=4,
    num_loader_workers=2,
    num_eval_loader_workers=2,
    run_eval=True,
    test_delay_epochs=-1,
    r=2,
    epochs=24,  # 限制训练轮数
    text_cleaner="basic_cleaners",  # 不使用 phoneme
    use_phonemes=False,
    precompute_num_workers=4,
    print_step=10,
    print_eval=True,
    mixed_precision=False,
    output_path=output_path,
    eval_split_max_size=50,
    eval_split_size=0.1,  # 自动分出每类约 5 条数据
    datasets=[dataset_config],
    max_audio_len=10 * 22050,  # 最大音频长度
    min_audio_len=0,  # 最小音频长度
    lr=1e-3,
    attention_type="dynamic_convolution",  # 使用动态卷积
    use_capacitron_vae=False,  # 关闭 Capacitron VAE
    grad_clip=0.0,  # 梯度裁剪
    max_decoder_steps=1000,
)

# 初始化 audio processor 和 tokenizer
ap = AudioProcessor.init_from_config(config)
tokenizer, config = TTSTokenizer.init_from_config(config)

# 加载数据样本
train_samples, eval_samples = load_tts_samples(
    dataset_config,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
)

# 初始化模型
model = Tacotron2(config, ap, tokenizer, speaker_manager=None)

# 训练器
trainer = Trainer(
    TrainerArgs(), config, output_path, model=model, train_samples=train_samples, eval_samples=eval_samples, training_assets={"audio_processor":ap},
)

if __name__ == "__main__":
    print("Starting training...")
    trainer.fit()  # 开始训练
