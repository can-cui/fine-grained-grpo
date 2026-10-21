"""
config.py — exp8: span-pooling tone head (without cross-attention)

Compared with exp5:
  - Change the tone_head input to span pooling (word_audio + tail_audio + post_audio)
  - Add TAIL_RATIO / POST_FRAMES / TONE_INPUT_DIM
  - Add TEXT_DROPOUT (randomly mask text tokens during training)
  - Remove F0-related configuration
"""
from pathlib import Path

# =====================================================================
# Data paths
# =====================================================================
DATA_ROOT = Path("/train29/tts/permanent/sjliu18/02-data/onnx/Tone_anno/En_corpus")
EXTRACT_SCRIPT = "/train29/tts/permanent/jyxu24/extraction_fbk/get_filterbank_feature_24k.py"

# =====================================================================
# Output directory
# =====================================================================
OUTPUT_DIR = Path("/train29/tts/permanent/jyxu24/exp8_span_tone_3class_vc_0627")
STATS_NPZ  = OUTPUT_DIR / "stats.npz"
CKPT_DIR   = OUTPUT_DIR / "checkpoints"
LOG_DIR    = OUTPUT_DIR / "logs"
VOCAB_PATH = OUTPUT_DIR / "vocab.json"

# =====================================================================
# Label rules
# =====================================================================
TONE_MAPPING = {
    "L*+H": 0, "L+H*": 0, "LH-": 0, "L-H%": 0,
    "H*+L": 1, "HL-": 1, "H-L%": 1,
}
TONE_NAMES = ["rising", "falling", "flat"]
N_TONES = 3

# =====================================================================
# Data splits
# =====================================================================
SPLIT_RATIO = {"train": 0.9, "val": 0.05, "test": 0.05}
SPLIT_SEED  = 42

# =====================================================================
# Feature parameters
# =====================================================================
FBANK_DIM    = 80
AUDIO_SR     = 24000
FBANK_HOP    = 240       # 10ms @ 24kHz
FRAME_DUR_S  = 0.01
CONV_STRIDE  = 4         # Total downsampling factor of the Conv frontend

# =====================================================================
# Model architecture
# =====================================================================
# Acoustic Encoder (same as exp5)
CONV_DIM       = 128 #256 #128
LSTM_HIDDEN    = 256#512 #256
LSTM_LAYERS    = 6 #8 #6
ACOUSTIC_DIM   = LSTM_HIDDEN * 2  # 1024
ENCODER_TYPE   = "bilstm"

# Text Encoder (same as exp5)
TEXT_EMBED_DIM = 128
TEXT_N_HEADS   = 4
TEXT_N_LAYERS  = 4
TEXT_FF_DIM    = 1024

# Cross-attention (used only by break_head)
CROSS_N_HEADS  = 8

# Span pooling for the tone head (new)
TAIL_RATIO     = 0.3     # Use the last 30% of word frames as tail_audio
POST_FRAMES    = 8       # Take 8 frames after the word (about 80 ms at 10 ms/frame after 4x downsampling = 320 ms raw)
TONE_INPUT_DIM = ACOUSTIC_DIM * 3  # word_audio + tail_audio + post_audio = 1536

# Text dropout (randomly mask text tokens as <unk> during training)
TEXT_DROPOUT   = 0.3     # 30% probability (configurable via --text-dropout)

# =====================================================================
# Training hyperparameters
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

# Loss weights
ALPHA_TONE     = 1.0
ALPHA_ALIGN    = 0.0

# Best model selection weights (0521_gold_train: tone only)
BEST_SCORE_W_BREAK = 0.0
BEST_SCORE_W_TONE  = 1.0

# Online data augmentation
USE_PEAK_NORM  = True
NOISE_PROB     = 0.4
SNR_MIN        = 10.0
SNR_MAX        = 25.0

# =====================================================================
# Preprocessing
# =====================================================================
TRIM_EDGE_SILENCE       = True
EDGE_SIL_MARGIN_FRAMES  = 10
SILENCE_PHONES = {"sil", "silv", "sp", "pau", "silb", "sile", "#", ""}

# =====================================================================
# Confidence thresholds
# =====================================================================
CONFIDENCE_THRESHOLD = 0.75
