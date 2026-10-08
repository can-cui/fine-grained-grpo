"""
config.py — exp8: span pooling tone head（不经过 cross-attn）

相比 exp5:
  - tone_head 改为 span pooling 输入（word_audio + tail_audio + post_audio）
  - 新增 TAIL_RATIO / POST_FRAMES / TONE_INPUT_DIM
  - 新增 TEXT_DROPOUT（训练时随机 mask 文本 token）
  - 删除 F0 相关配置
"""
from pathlib import Path

# =====================================================================
# 数据路径
# =====================================================================
DATA_ROOT = Path("/train29/tts/permanent/sjliu18/02-data/onnx/Tone_anno/En_corpus")
EXTRACT_SCRIPT = "/train29/tts/permanent/jyxu24/extraction_fbk/get_filterbank_feature_24k.py"

# =====================================================================
# 输出目录
# =====================================================================
OUTPUT_DIR = Path("/train29/tts/permanent/jyxu24/exp8_span_tone_3class_vc_0627")
STATS_NPZ  = OUTPUT_DIR / "stats.npz"
CKPT_DIR   = OUTPUT_DIR / "checkpoints"
LOG_DIR    = OUTPUT_DIR / "logs"
VOCAB_PATH = OUTPUT_DIR / "vocab.json"

# =====================================================================
# 标签规则
# =====================================================================
TONE_MAPPING = {
    "L*+H": 0, "L+H*": 0, "LH-": 0, "L-H%": 0,
    "H*+L": 1, "HL-": 1, "H-L%": 1,
}
TONE_NAMES = ["rising", "falling", "flat"]
N_TONES = 3

# =====================================================================
# 数据划分
# =====================================================================
SPLIT_RATIO = {"train": 0.9, "val": 0.05, "test": 0.05}
SPLIT_SEED  = 42

# =====================================================================
# 特征参数
# =====================================================================
FBANK_DIM    = 80
AUDIO_SR     = 24000
FBANK_HOP    = 240       # 10ms @ 24kHz
FRAME_DUR_S  = 0.01
CONV_STRIDE  = 4         # Conv frontend 总下采样倍数

# =====================================================================
# 模型结构
# =====================================================================
# Acoustic Encoder（和 exp5 一样）
CONV_DIM       = 128 #256 #128
LSTM_HIDDEN    = 256#512 #256
LSTM_LAYERS    = 6 #8 #6
ACOUSTIC_DIM   = LSTM_HIDDEN * 2  # 1024
ENCODER_TYPE   = "bilstm"

# Text Encoder（和 exp5 一样）
TEXT_EMBED_DIM = 128
TEXT_N_HEADS   = 4
TEXT_N_LAYERS  = 4
TEXT_FF_DIM    = 1024

# Cross-Attention（只用于 break_head）
CROSS_N_HEADS  = 8

# Span Pooling for Tone Head（新增）
TAIL_RATIO     = 0.3     # 词尾 30% 帧作为 tail_audio
POST_FRAMES    = 8       # 词后取 8 帧（约 80ms @ 10ms/frame after 4x downsample = 320ms raw）
TONE_INPUT_DIM = ACOUSTIC_DIM * 3  # word_audio + tail_audio + post_audio = 1536

# Text Dropout（训练时随机 mask 文本 token 为 <unk>）
TEXT_DROPOUT   = 0.3     # 30% 概率（可通过 --text-dropout 调）

# =====================================================================
# 训练超参
# =====================================================================
MAX_FRAMES     = 600
MAX_WORDS      = 80
BATCH_SIZE     = 64
EPOCHS         = 50
LR             = 1e-3
WEIGHT_DECAY   = 1e-4
PATIENCE       = 5
DROPOUT        = 0.2
GRAD_CLIP      = 1.0
WARMUP_EPOCHS  = 3

# Loss 权重
ALPHA_TONE     = 1.0
ALPHA_ALIGN    = 0.0

# Best model 选择权重（0521_gold_train: 只看 tone）
BEST_SCORE_W_BREAK = 0.0
BEST_SCORE_W_TONE  = 1.0

# 在线数据增强
USE_PEAK_NORM  = True
NOISE_PROB     = 0.4
SNR_MIN        = 10.0
SNR_MAX        = 25.0

# =====================================================================
# 预处理
# =====================================================================
TRIM_EDGE_SILENCE       = True
EDGE_SIL_MARGIN_FRAMES  = 10
SILENCE_PHONES = {"sil", "silv", "sp", "pau", "silb", "sile", "#", ""}

# =====================================================================
# 置信度阈值
# =====================================================================
CONFIDENCE_THRESHOLD = 0.75
