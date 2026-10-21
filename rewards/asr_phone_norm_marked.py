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
@register_reward('phoneasrnormMarked')
class PhoneASR(torch.nn.Module):
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



    def get_focus_phones(self,text,
                        intos,
                        pred_phone,
                        text_to_huper_phones):
        """
        Args:
            text: Original text
            intos: Word-level mask; 1 means include and -1 means ignore
            pred_phone: Phoneme string predicted by ASR
            text_to_huper_phones: Function that converts text to HuPER phonemes

        Returns:
            focus_gt_phones: list[str]
            focus_pred_phones: list[str]
        """

        words = text.split()

        #
        # 1. Build the complete ground-truth phone sequence
        #
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

        #
        # 2. Find the ground-truth phone indices for the target words
        #
        focus_gt_idx = []

        for i, flag in enumerate(intos):
            if flag != 1:
                continue

            s, e = word_phone_spans[i]
            focus_gt_idx.extend(range(s, e))

        #
        # 3. pred phones
        #
        pred_phones = pred_phone.split()

        #
        # 4. Align the gold and predicted sequences
        #
        sm = SequenceMatcher(
            None,
            gold_phones,
            pred_phones
        )

        gold2pred = {}

        for tag, i1, i2, j1, j2 in sm.get_opcodes():

            if tag == "equal":
                for g, p in zip(
                        range(i1, i2),
                        range(j1, j2)):
                    gold2pred[g] = p

            elif tag == "replace":
                m = min(i2 - i1, j2 - j1)

                for k in range(m):
                    gold2pred[i1 + k] = j1 + k

            # delete:
            # The gold phone exists, but the predicted phone does not
            # Do not create a mapping

            # insert:
            # The prediction contains an extra phone
            # Do not create a mapping

        #
        # 5. Extract ground-truth phones for the target words
        #
        focus_gt_phones = [
            gold_phones[i]
            for i in focus_gt_idx
        ]

        #
        # 6. Extract predicted phones for the target words
        #
        focus_pred_idx = [
            gold2pred[i]
            for i in focus_gt_idx
            if i in gold2pred
        ]

        focus_pred_phones = [
            pred_phones[i]
            for i in focus_pred_idx
        ]

        return focus_gt_phones, focus_pred_phones
    def cal_reward(self, asr_phoneme, tgt_text):
        """
        Compute the phoneme ASR reward
        TODO: Implement phoneme-level reward computation as needed

        Args:
            asr_phoneme: Phoneme sequence recognized by ASR
            tgt_text: Original text to be converted into phonemes

        Returns:
            reward: Reward value
        """
        # # Normalize consecutive spaces in ASR output
        # asr_phoneme = normalize_spaces(asr_phoneme)
        # # Convert the original text to HuPER phonemes
        # golden_phoneme = text_to_huper_phones(tgt_text)
        # deletions = count_deletions(tgt_text, asr_phoneme)
        deletions, replacements = count_phone_errors(
            tgt_text,
            asr_phoneme
        )
        reward = -(deletions + 0.5*replacements) / len(tgt_text)
        # if (deletions + replacements)>0:
        #     import pdb;pdb.set_trace()
        return reward

    @torch.no_grad()
    def forward(self, wav_batch, labels, wav_masks=None, num_thread=1, infer_bsz=1):
        """Run batch inference."""
        if num_thread <= 1:
            return self._forward(wav_batch, labels, wav_masks, infer_bsz)

        bsz = wav_batch.size(0)
        total_batch = bsz // infer_bsz
        if bsz % infer_bsz != 0:
            total_batch += 1
        rewards = []
        with ThreadPoolExecutor(max_workers=num_thread) as pool:
            for i in range(total_batch):
                rewards.append(pool.submit(
                    self.reward_func,
                    wav_batch[i*infer_bsz:(i+1)*infer_bsz],
                    labels[i*infer_bsz:(i+1)*infer_bsz],
                    wav_masks[i*infer_bsz:(i+1)*infer_bsz] if wav_masks is not None else None
                ))
        rewards = [feature.result() for feature in rewards]
        rewards = torch.tensor(rewards).unsqueeze(-1).to(wav_batch)
        return rewards

    def _forward(self, wav_batch, labels, wav_masks=None, infer_bsz=1):
        """Run batch inference in a single thread."""
        preds = []
        bsz = wav_batch.size(0)
        total_batch = bsz // infer_bsz
        if bsz % infer_bsz != 0:
            total_batch += 1

        for i in range(total_batch):
            wav = wav_batch[i*infer_bsz:(i+1)*infer_bsz]
            wav_mask = wav_masks[i*infer_bsz:(i+1)*infer_bsz] if wav_masks is not None else None
            valid_wav = wav[wav_mask.bool()] if wav_mask is not None else wav

            # Resample to 16 kHz (the original audio may be 24 kHz)
            audio = torchaudio.functional.resample(valid_wav.unsqueeze(0), orig_freq=24000, new_freq=16000).squeeze(0)
            audio = audio.detach().float().cpu().numpy()

            # Write a temporary file for inference
            # import tempfile
            # with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            #     sf.write(f.name, audio, TARGET_SAMPLE_RATE)
            pred_phoneme = infer_phoneme_one(self.processor, self.model, "cuda", audio)
            preds.extend([pred_phoneme])

        rewards = []
        for idx, (pred_text, tgt_text) in enumerate(zip(preds, labels)):
            text, intos = tgt_text
            focus_gt, focus_pred = self.get_focus_phones(
                text=text,
                intos=intos,
                pred_phone=normalize_spaces(pred_text),
                text_to_huper_phones=text_to_huper_phones,
            )
            # import pdb;pdb.set_trace()
            try:
                reward = self.cal_reward(focus_pred, focus_gt)
            except:
                import pdb;pdb.set_trace()

            rewards.append(reward)

        rewards = torch.tensor(rewards).unsqueeze(-1).to(wav_batch)
        return rewards

    def reward_func(self, wav, label, wav_masks=None):
        """Compute the reward for a single sample."""
        valid_wav = wav[wav_masks.bool()] if wav_masks is not None else wav
        audio = torchaudio.functional.resample(valid_wav.unsqueeze(0), orig_freq=24000, new_freq=16000).squeeze(0)
        audio = audio.detach().float().cpu().numpy()

        pred_phoneme = infer_phoneme_one(self.processor, self.model, "cuda", audio)
        text, intos = labels
        # text, intos = tgt_text
        focus_gt, focus_pred = self.get_focus_phones(
            text=text,
            intos=intos,
            pred_phone=normalize_spaces(pred_phoneme),
            text_to_huper_phones=text_to_huper_phones,
        )
        # import pdb;pdb.set_trace()
        reward = self.cal_reward(focus_pred, focus_gt)
        
        # import pdb;pdb.set_trace()/
        # reward = self.cal_reward(pred_phoneme, label)
        # import pdb;pdb.set_trace()
        return reward


register_reward_target_type('phoneasrnormMarked', 'text_with_into')
register_reward_norm_type('phoneasrnormMarked', 'max_division')