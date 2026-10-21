import sys, torch
from . import register_reward, register_reward_target_type, register_reward_norm_type
import Levenshtein
from concurrent.futures import ThreadPoolExecutor
import os
import numpy as np
import soundfile as sf
import torchaudio
from scipy.signal import resample_poly
os.environ["NUMBA_DISABLE_JIT"] = "1"
from transformers import AutoModelForCTC, AutoProcessor
import pronouncing
import re
from difflib import SequenceMatcher

# Path to the phoneme ASR model
PHONEME_MODEL_DIR = "Fine-Grained-GRPO/models/phone_asr/model/best_by_target_metrics_0623"
TARGET_SAMPLE_RATE = 16000
SAMPLE_RATE = 16000  # WhisperX uses 16kHz

# Path to the WhisperX model
WHISPERX_DIR = "Fine-Grained-GRPO/models/intonation_model/exp8_span_tone_3class/whisperX-main"
WHISPERX_MAIN_DIR = "Fine-Grained-GRPO/models/intonation_model/exp8_span_tone_3class/whisperX-main"

# Add the paths
sys.path.insert(0, WHISPERX_DIR)
sys.path.insert(0, WHISPERX_MAIN_DIR)
import whisperx

# Phoneme inventory used by the Huper phoneme model
MODEL_PHONES = {
    "AA", "AE", "AH", "AW", "AY", "B", "CH", "D", "DH", "EH", "ER", "EY",
    "F", "G", "HH", "IH", "IY", "JH", "K", "L", "M", "N", "NG", "OW", "OY",
    "P", "R", "S", "SH", "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH"
}

PHONE_MAP = {
    "AO": ["AA"],
    "AX": ["AH"],
    "AXR": ["ER"],
    "IX": ["IH"],
    "EL": ["AH", "L"],
    "EM": ["AH", "M"],
    "EN": ["AH", "N"],
    "ENG": ["IH", "NG"],
}

WORD_OVERRIDES = {
    "a": ["AH"],
    "an": ["AE", "N"],
    "and": ["AE", "N", "D"],
    "to": ["T", "UW"],
    "of": ["AH", "V"],
    "wasn't": ["W", "AH", "Z", "AH", "N", "T"],
    "don't": ["D", "OW", "N", "T"],
    "i'm": ["AY", "M"],
    "we've": ["W", "IY", "V"],
    "what's": ["W", "AH", "T", "S"],
}

LEXICAL_OVERRIDES = {
    "surprisingly": ["S", "AH", "P", "R", "AY", "Z", "IH", "NG", "L", "IY"],
    "visual": ["V", "IH", "ZH", "UW", "AH", "L"],
}

VOWEL_LETTERS = set("aeiou")

# Special tokens for mapping tokens to words
# Assume the GPT-generated token sequence contains special separators marking word boundaries
# Determine these using the actual tokenizer; the following tokens are placeholders for word boundaries
WORD_SEPARATOR_TOKENS = {"", ""}  # Configure this according to the actual tokenizer


def _normalize_word(word):
    """Normalize a word (remove punctuation and lowercase it) for fuzzy matching."""
    return re.sub(r'[^/w]', '', word.lower())


def _merge_whisperx_to_user_words(user_words, whisperx_segments):
    """Map WhisperX fine-grained tokens to the user-provided coarse-grained tokens.

    Args:
        user_words: Sequence of user-provided words
        whisperx_segments: Output from WhisperX

    Returns:
        word_boundaries: List of [start_cs, end_cs] pairs, with length len(user_words)
    """
    user_norm = [_normalize_word(w) for w in user_words]
    whisperx_norm = [_normalize_word(seg["word"]) for seg in whisperx_segments]

    word_boundaries = []
    wx_idx = 0

    for i, user_w_norm in enumerate(user_norm):
        if not user_w_norm:
            return None

        matched_segs = []
        accumulated = ""

        while wx_idx < len(whisperx_norm):
            wx_word_norm = whisperx_norm[wx_idx]

            if wx_word_norm:
                accumulated += wx_word_norm

            matched_segs.append(whisperx_segments[wx_idx])
            wx_idx += 1

            if user_w_norm in accumulated:
                break

            if len(accumulated) >= len(user_w_norm) and user_w_norm in accumulated:
                break

            if len(accumulated) > len(user_w_norm) * 2:
                return None

        if not matched_segs:
            return None

        start_s = min(seg["start"] for seg in matched_segs)
        end_s = max(seg["end"] for seg in matched_segs)
        start_cs = int(start_s * 100)
        end_cs = int(end_s * 100)
        word_boundaries.append([start_cs, end_cs])

    if len(word_boundaries) != len(user_words):
        return None

    return word_boundaries


def clean_phone(phone: str):
    """Strip stress and map CMUdict-only phones into the Huper label set."""
    phone = re.sub(r"[0-2]$", "", phone)
    if phone in PHONE_MAP:
        return PHONE_MAP[phone]
    if phone in MODEL_PHONES:
        return [phone]
    return []


def clean_arpabet(phones: str):
    cleaned = []
    for phone in phones.split():
        cleaned.extend(clean_phone(phone))
    return cleaned


def starts_with_vowel_sound(word: str):
    phones_list = pronouncing.phones_for_word(word.lower())
    if phones_list:
        phones = clean_arpabet(phones_list[0])
        return bool(phones and phones[0] in {
            "AA", "AE", "AH", "AW", "AY", "EH", "ER", "EY",
            "IH", "IY", "OW", "OY", "UH", "UW"
        })
    return bool(word and word[0].lower() in VOWEL_LETTERS)


def choose_pronunciation(word: str):
    for phones in pronouncing.phones_for_word(word.lower()):
        cleaned = clean_arpabet(phones)
        if cleaned:
            return cleaned
    return ["<UNK>"]


def word_to_huper_phones(word: str, next_word: str = None):
    """
    Convert one word to a huper29/huper_recognizer-style phone sequence.
    Keep true T/D, remove stress digits, and do not emit DX.
    """
    word = word.lower()
    if word == "the":
        if next_word and starts_with_vowel_sound(next_word):
            return ["DH", "IY"]
        return ["DH", "AH"]
    if word in WORD_OVERRIDES:
        return WORD_OVERRIDES[word]
    if word in LEXICAL_OVERRIDES:
        return LEXICAL_OVERRIDES[word]
    return choose_pronunciation(word)


def tokenize_text(text: str):
    """Tokenize text while preserving common English contractions."""
    text = re.sub(r"【[^】]*】", " ", text)
    text = text.replace("&", " and ")
    text = re.sub(r"[-_/]+", " ", text)
    text = text.replace("'", "'").replace("'", "'")
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text)


def text_to_huper_phones(text: str):
    """
    Convert sentence text to a huper29/huper_recognizer-style phone sequence.
    No word boundary markers are added, and T/D are not replaced with DX.
    """
    words = tokenize_text(text)
    phone_seq = []
    for i, word in enumerate(words):
        next_word = words[i + 1] if i + 1 < len(words) else None
        phones = word_to_huper_phones(word, next_word=next_word)
        if phones:
            phone_seq.extend(phones)
    return " ".join(phone_seq)


def normalize_spaces(phoneme_str: str):
    """Replace consecutive spaces with a single space."""
    return re.sub(r' +', ' ', phoneme_str).strip()

# Configure the offline environment
def configure_offline_env(root):
    os.environ.setdefault("HF_HOME", os.path.join(root, "hf_cache"))
    os.environ.setdefault("HF_HUB_CACHE", os.path.join(root, "hf_cache", "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(root, "hf_cache", "transformers"))
    os.environ.setdefault("TORCH_HOME", os.path.join(root, "torch_cache"))
    os.environ.setdefault("TMPDIR", os.path.join(root, "temp"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def load_audio_16k_mono(path):
    """Load audio and resample it to 16 kHz mono."""
    audio, sample_rate = sf.read(str(path), always_2d=False)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sample_rate != TARGET_SAMPLE_RATE:
        audio = resample_poly(audio, TARGET_SAMPLE_RATE, sample_rate).astype(np.float32)
        sample_rate = TARGET_SAMPLE_RATE
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak
    return audio, sample_rate


def decode_phoneme_labels(model, predicted_ids):
    """Decode phoneme labels and remove CTC repeats and blanks."""
    ids = predicted_ids[0].detach().cpu().tolist()
    blank_id = getattr(model.config, "pad_token_id", None)
    id2label = getattr(model.config, "id2label", {})
    labels = []
    previous = None
    for item in ids:
        if blank_id is not None and item == blank_id:
            previous = item
            continue
        if item == previous:
            continue
        label = id2label.get(item, str(item))
        if label not in {"<pad>", "[PAD]", "<s>", "</s>", "<unk>"}:
            labels.append(label)
        previous = item
    return " ".join(labels).strip().replace("|", " ")


def infer_phoneme_one(processor, model, device, audio):
    """Recognize phonemes in a single audio file."""
    # audio, sample_rate = load_audio_16k_mono(wav_path)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        logits = model(**inputs).logits
        predicted_ids = torch.argmax(logits, dim=-1)
    return decode_phoneme_labels(model, predicted_ids)

def count_deletions(ref_phoneme: str, hyp_phoneme: str):
    ref = ref_phoneme.split()
    hyp = hyp_phoneme.split()

    n, m = len(ref), len(hyp)

    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(
                    dp[i - 1][j] + 1,      # deletion
                    dp[i][j - 1] + 1,      # insertion
                    dp[i - 1][j - 1] + 1,  # substitution
                )

    # Count deletions during backtracing
    i, j = n, m
    deletions = 0

    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1]:
            i -= 1
            j -= 1
        elif (
            i > 0
            and dp[i][j] == dp[i - 1][j] + 1
        ):
            deletions += 1
            i -= 1
        elif (
            j > 0
            and dp[i][j] == dp[i][j - 1] + 1
        ):
            j -= 1
        else:
            i -= 1
            j -= 1

    return deletions

def count_phone_errors(gt_phones, pred_phones):
    """
    Args:
        gt_phones: list[str]
        pred_phones: list[str]

    Returns:
        deletions
        replacements
    """
    ops = Levenshtein.editops(
        gt_phones,
        pred_phones
    )

    deletions = 0
    replacements = 0

    for op, i, j in ops:
        if op == "delete":
            deletions += 1
        elif op == "replace":
            replacements += 1

    return deletions, replacements


@register_reward('phoneasrnormMarkedTokenSpeed')
class PhoneASRWordLevelSpeed(torch.nn.Module):
    """Word-level phoneme ASR reward with speed penalty.

    Returns:
        - sentence-level reward (for backward compatibility)
        - word-level rewards with speed penalty (for token-level loss in trainer)
    """

    def __init__(self) -> None:
        super().__init__()

        # Configure the offline environment
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        phoneme_root = "/train34/tts/permanent/cancui11/RL/models/phoneme_asr"
        configure_offline_env(phoneme_root)

        # Load the phoneme ASR model
        self.processor = AutoProcessor.from_pretrained(PHONEME_MODEL_DIR, local_files_only=True)
        self.model = AutoModelForCTC.from_pretrained(PHONEME_MODEL_DIR, local_files_only=True)
        self.model.to("cuda")
        self.model.eval()

        # Load the WhisperX alignment model
        self.align_model, self.align_metadata = whisperx.load_align_model(
            "en", "cuda", model_dir='/train34/tts/permanent/cancui11/RL/models/whisperX-main/checkpoints'
        )

        # Speech-rate parameters (consistent with asr_phone_norm_marked_speed.py)
        self.speed_threshold = 10.0  # Letters per second
        self.speed_penalty_factor = -0.5

    def whisperx_align_words(self, audio_16k, words):
        """Align with WhisperX and obtain word boundaries in centiseconds."""
        try:
            duration_s = len(audio_16k) / SAMPLE_RATE
            text_str = " ".join(words)
            transcript = [{"start": 0.0, "end": duration_s, "text": text_str}]

            result = whisperx.align(transcript, self.align_model, self.align_metadata,
                                    audio_16k, "cuda", return_char_alignments=False)

            if "word_segments" not in result or not result["word_segments"]:
                return None

            whisperx_segments = result["word_segments"]
            word_boundaries = _merge_whisperx_to_user_words(words, whisperx_segments)

            if word_boundaries is not None:
                for i in range(len(word_boundaries) - 1):
                    word_boundaries[i][1] = word_boundaries[i + 1][0]
            return word_boundaries
        except Exception as e:
            print(f"[warn] whisperx align failed: {e}")
            return None

    def cal_speech_rate(self, text, wav_duration):
        """Compute speech rate in English letters per second."""
        num_letters = len(re.findall(r"[A-Za-z]", text))
        if wav_duration <= 0:
            return 0.0
        return num_letters / wav_duration

    def cal_word_speed_reward(self, word, word_duration):
        """Compute the speech-rate reward for a single word.

        Args:
            word: Word text
            word_duration: Word duration in seconds

        Returns:
            speed_reward: Speech-rate reward (negative when speech is too slow)
        """
        num_letters = len(re.findall(r"[A-Za-z]", word))
        if word_duration <= 0:
            return 0.0

        speech_rate = num_letters / word_duration
        if speech_rate < self.speed_threshold:
            return (self.speed_threshold - speech_rate) * self.speed_penalty_factor
        return 0.0

    def cal_reward(self, focus_pred, focus_gt):
        """Compute the sentence-level phoneme ASR reward."""
        deletions, replacements = count_phone_errors(focus_gt, focus_pred)
        reward = -(deletions + 0.5 * replacements) / max(len(focus_gt), 1)
        return reward

    def _get_focus_phones(self, text, intos, pred_phone):
        """Get the ground-truth and predicted phones for the focus words."""
        words = text.split()
        gold_phones = []
        word_phone_spans = []

        offset = 0
        for w in words:
            phones = text_to_huper_phones(w).split()
            s = offset
            e = offset + len(phones)
            word_phone_spans.append((s, e))
            gold_phones.extend(phones)
            offset = e

        focus_gt_idx = []
        for i, flag in enumerate(intos):
            if flag != 1:
                continue
            s, e = word_phone_spans[i]
            focus_gt_idx.extend(range(s, e))

        pred_phones = pred_phone.split()

        sm = SequenceMatcher(None, gold_phones, pred_phones)
        gold2pred = {}

        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                for g, p in zip(range(i1, i2), range(j1, j2)):
                    gold2pred[g] = p
            elif tag == "replace":
                m = min(i2 - i1, j2 - j1)
                for k in range(m):
                    gold2pred[i1 + k] = j1 + k

        focus_gt_phones = [gold_phones[i] for i in focus_gt_idx]
        focus_pred_idx = [gold2pred[i] for i in focus_gt_idx if i in gold2pred]
        focus_pred_phones = [pred_phones[i] for i in focus_pred_idx]

        return focus_gt_phones, focus_pred_phones

    @torch.no_grad()
    def forward(self, wav_batch, labels, wav_masks=None, num_thread=1, infer_bsz=1):
        """Run batch inference.

        Returns:
            rewards: tensor of shape [batch_size, 1], sentence-level reward
            word_rewards_list: list of list of float, word-level rewards per sample
            word_boundaries_list: list of list of [start_cs, end_cs], word boundaries per sample
        """
        if num_thread <= 1:
            return self._forward(wav_batch, labels, wav_masks, infer_bsz)

        bsz = wav_batch.size(0)
        total_batch = bsz // infer_bsz
        if bsz % infer_bsz != 0:
            total_batch += 1
        results = []
        with ThreadPoolExecutor(max_workers=num_thread) as pool:
            for i in range(total_batch):
                results.append(pool.submit(
                    self.reward_func,
                    wav_batch[i*infer_bsz:(i+1)*infer_bsz],
                    labels[i*infer_bsz:(i+1)*infer_bsz],
                    wav_masks[i*infer_bsz:(i+1)*infer_bsz] if wav_masks is not None else None
                ))
        results = [f.result() for f in results]

        rewards = []
        word_rewards_list = []
        word_boundaries_list = []
        for r in results:
            rewards.append(r["reward"])
            word_rewards_list.append(r["word_rewards"])
            word_boundaries_list.append(r["word_boundaries"])

        rewards_tensor = torch.tensor(rewards).unsqueeze(-1).to(wav_batch.device)
        return rewards_tensor, word_rewards_list, word_boundaries_list

    def _forward(self, wav_batch, labels, wav_masks=None, infer_bsz=1):
        """Run batch inference in a single thread."""
        preds = []
        audios_16k = []
        wav_durations = []
        bsz = wav_batch.size(0)
        total_batch = bsz // infer_bsz
        if bsz % infer_bsz != 0:
            total_batch += 1

        for i in range(total_batch):
            wav = wav_batch[i*infer_bsz:(i+1)*infer_bsz]
            wav_mask = wav_masks[i*infer_bsz:(i+1)*infer_bsz] if wav_masks is not None else None
            valid_wav = wav[wav_mask.bool()] if wav_mask is not None else wav

            # Compute audio duration (original sample rate: 24 kHz)
            wav_durations.append(valid_wav.shape[0] / 24000.0)

            audio = torchaudio.functional.resample(valid_wav.unsqueeze(0), orig_freq=24000, new_freq=16000).squeeze(0)
            audio = audio.detach().float().cpu().numpy()

            pred_phoneme = infer_phoneme_one(self.processor, self.model, "cuda", audio)
            preds.extend([pred_phoneme])
            audios_16k.append(audio)

        results = []
        for idx, (pred_text, tgt_text, audio_16k, wav_dur) in enumerate(zip(preds, labels, audios_16k, wav_durations)):
            text, intos = tgt_text

            words = text.split()
            word_boundaries = self.whisperx_align_words(audio_16k, words)

            # Compute the sentence-level reward
            focus_gt, focus_pred = self._get_focus_phones(
                text=text,
                intos=intos,
                pred_phone=normalize_spaces(pred_text),
            )
            sentence_reward = self.cal_reward(focus_pred, focus_gt)

            if word_boundaries is None:
                word_boundaries = [[i * 100, (i + 1) * 100] for i in range(len(words))]

            # Initialize word_rewards to zero; use sentence_reward + speed_reward where intos == 1
            final_word_rewards = [0.0] * len(words)
            for i, flag in enumerate(intos):
                if flag == 1:
                    base_reward = sentence_reward
                    # Add the speech-rate penalty for this word
                    if word_boundaries and i < len(word_boundaries):
                        start_cs, end_cs = word_boundaries[i]
                        word_duration = (end_cs - start_cs) / 100.0  # Convert centiseconds to seconds
                        speed_reward = self.cal_word_speed_reward(words[i], word_duration)
                        base_reward = base_reward + speed_reward
                    final_word_rewards[i] = base_reward

            results.append({
                "reward": sentence_reward,
                "word_rewards": final_word_rewards,
                "word_boundaries": word_boundaries,
            })

        rewards = [r["reward"] for r in results]
        word_rewards_list = [r["word_rewards"] for r in results]
        word_boundaries_list = [r["word_boundaries"] for r in results]

        rewards_tensor = torch.tensor(rewards).unsqueeze(-1).to(wav_batch.device)
        return rewards_tensor, word_rewards_list, word_boundaries_list

    def reward_func(self, wav, label, wav_masks=None):
        """Compute the reward for a single sample."""
        valid_wav = wav[wav_masks.bool()] if wav_masks is not None else wav
        wav_duration = valid_wav.shape[0] / 24000.0

        audio = torchaudio.functional.resample(valid_wav.unsqueeze(0), orig_freq=24000, new_freq=16000).squeeze(0)
        audio = audio.detach().float().cpu().numpy()

        pred_phoneme = infer_phoneme_one(self.processor, self.model, "cuda", audio)
        text, intos = label

        words = text.split()
        word_boundaries = self.whisperx_align_words(audio, words)

        # Compute the sentence-level reward
        focus_gt, focus_pred = self._get_focus_phones(
            text=text,
            intos=intos,
            pred_phone=normalize_spaces(pred_phoneme),
        )
        sentence_reward = self.cal_reward(focus_pred, focus_gt)

        if word_boundaries is None:
            word_boundaries = [[i * 100, (i + 1) * 100] for i in range(len(words))]

        # Initialize word_rewards to zero; use sentence_reward + speed_reward where intos == 1
        final_word_rewards = [0.0] * len(words)
        for i, flag in enumerate(intos):
            if flag == 1:
                base_reward = sentence_reward
                if word_boundaries and i < len(word_boundaries):
                    start_cs, end_cs = word_boundaries[i]
                    word_duration = (end_cs - start_cs) / 100.0
                    speed_reward = self.cal_word_speed_reward(words[i], word_duration)
                    base_reward = base_reward + speed_reward
                final_word_rewards[i] = base_reward

        return {
            "reward": sentence_reward,
            "word_rewards": final_word_rewards,
            "word_boundaries": word_boundaries,
        }


register_reward_target_type('phoneasrnormMarkedTokenSpeed', 'text_with_into')
register_reward_norm_type('phoneasrnormMarkedTokenSpeed', 'max_division')