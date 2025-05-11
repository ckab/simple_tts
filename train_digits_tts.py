import os
from trainer import Trainer, TrainerArgs
from TTS.utils.audio import AudioProcessor
from TTS.tts.models.tacotron2 import Tacotron2
#from TTS.tts.configs.shared_configs import BaseAudioConfig
# from TTS.tts.configs.tacotron2_config import Tacotron2Config
from TTS.tts.datasets import load_tts_samples
from TTS.tts.utils.text.tokenizer import TTSTokenizer
#from TTS.utils.generic_utils import download_model
import torch
from TTS.utils.capacitron_optimizer import CapacitronOptimizer

torch.set_num_threads(8)
# 设置数据路径和模型输出路径
data_path = "Numbers_denoise_aug_re"  # 数据路径
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")  # 输出路径为当前脚本所在路径下的output文件夹

# 如果输出目录不存在，则创建
if not os.path.exists(output_path):
    os.makedirs(output_path)

# 加载配置文件
from config import config  # 引入修改后的配置

# 加载音频处理器
ap = AudioProcessor(**config.audio.to_dict())

# 初始化文本分词器
tokenizer, config = TTSTokenizer.init_from_config(config)

# 加载数据集样本
dataset_config = config.datasets[0]  # 获取数据集的配置信息
train_samples, eval_samples = load_tts_samples(dataset_config, eval_split=True)
if not train_samples:
    raise RuntimeError("Train samples could not be loaded. Check dataset path or metadata.csv format.")

# 初始化模型
model = Tacotron2(config, ap, tokenizer, speaker_manager=None)

# 创建训练器
trainer = Trainer(
    TrainerArgs(),
    config,
    output_path,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
    training_assets={"audio_processor": ap},
)

# 开始训练
if __name__ == "__main__":
    print("Starting training...")
    trainer.fit()  # 开始训练
