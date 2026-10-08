"""
model.py — exp8: Span Pooling Tone Head

相比 exp5:
  - tone_head 不再吃 cross-attn 输出
  - 改为直接从 acoustic_feat 上做 span pooling（用 word_boundaries 定位）
  - break_head 保持不变（继续吃 cross-attn 输出）
  - 删除 F0 相关模块

架构:
    fbank → Conv×2 (↓4x) → BiLSTM → acoustic_feat (B, T/4, 512)
    text  → Embed + Transformer → text_feat (B, N, 128)

    Break path:
        CrossAttn(Q=text, K/V=acoustic) → word_repr (B, N, 512)
        break_head(word_repr) → (B, N)

    Tone path:
        span_pool(acoustic_feat, word_spans) → tone_input (B, N, 1536)
        tone_head(tone_input) → (B, N, 2)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import config3c as C


class BiLSTMEncoder(nn.Module):
    def __init__(self, in_dim=C.CONV_DIM, hidden=C.LSTM_HIDDEN,
                 n_layers=C.LSTM_LAYERS, dropout=C.DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden,
            num_layers=n_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.out_dim = hidden * 2

    def forward(self, x, mask=None):
        out, _ = self.lstm(x)
        return out


class ResidualBiLSTMLayer(nn.Module):
    def __init__(self, dim, dropout=C.DROPOUT):
        super().__init__()
        assert dim % 2 == 0
        self.lstm = nn.LSTM(
            input_size=dim, hidden_size=dim // 2,
            num_layers=1, bidirectional=True, batch_first=True,
        )
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h, _ = self.lstm(self.norm(x))
        h = self.dropout(h)
        return x + h


class ResidualBiLSTMEncoder(nn.Module):
    def __init__(self, in_dim=C.CONV_DIM, hidden=C.LSTM_HIDDEN,
                 n_layers=C.LSTM_LAYERS, dropout=C.DROPOUT):
        super().__init__()
        dim = hidden * 2
        self.in_proj = nn.Linear(in_dim, dim)
        self.layers = nn.ModuleList([
            ResidualBiLSTMLayer(dim, dropout=dropout) for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(dim)
        self.out_dim = dim

    def forward(self, x, mask=None):
        x = self.in_proj(x)
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)


class TransformerAcousticEncoder(nn.Module):
    def __init__(self, in_dim=C.CONV_DIM, d_model=C.ACOUSTIC_DIM,
                 n_heads=8, n_layers=4, ff_dim=1024, dropout=C.DROPOUT):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=ff_dim, dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        self.out_dim = d_model

    def forward(self, x, mask=None):
        x = self.proj(x)
        if mask is not None:
            src_key_padding_mask = (mask < 0.5)
        else:
            src_key_padding_mask = None
        return self.transformer(x, src_key_padding_mask=src_key_padding_mask)


class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=C.TEXT_EMBED_DIM,
                 n_heads=C.TEXT_N_HEADS, n_layers=C.TEXT_N_LAYERS,
                 ff_dim=C.TEXT_FF_DIM, dropout=C.DROPOUT):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads,
            dim_feedforward=ff_dim, dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        self.out_dim = embed_dim

    def forward(self, text_tokens, text_mask):
        x = self.embedding(text_tokens)
        src_key_padding_mask = (text_mask < 0.5)
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        return x


class CrossAttentionAlignment(nn.Module):
    def __init__(self, text_dim=C.TEXT_EMBED_DIM, acoustic_dim=C.ACOUSTIC_DIM,
                 n_heads=C.CROSS_N_HEADS):
        super().__init__()
        self.query_proj = nn.Linear(text_dim, acoustic_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=acoustic_dim, num_heads=n_heads, batch_first=True,
        )

    def forward(self, text_feat, acoustic_feat, acoustic_mask=None):
        query = self.query_proj(text_feat)
        key_padding_mask = (acoustic_mask < 0.5) if acoustic_mask is not None else None
        word_acoustic, attn_weights = self.attn(
            query=query, key=acoustic_feat, value=acoustic_feat,
            key_padding_mask=key_padding_mask,
        )
        return word_acoustic, attn_weights


def span_pool(acoustic_feat, word_spans, tail_ratio=C.TAIL_RATIO,
              post_frames=C.POST_FRAMES):
    """从 acoustic_feat 上按 word_spans 做 span pooling（向量化版）。

    Args:
        acoustic_feat: (B, T, D)
        word_spans: (B, N, 2) — 每个词的 [start, end) 帧索引（已经是下采样后的）
        tail_ratio: 取词尾多少比例作为 tail_audio
        post_frames: 词后取多少帧作为 post_audio

    Returns:
        tone_input: (B, N, D*3)
    """
    B, T, D = acoustic_feat.shape
    N = word_spans.size(1)
    device = acoustic_feat.device

    # cumsum trick: sum over [s, e) = cumsum[e] - cumsum[s]
    # 在 T 维前面 pad 一个零，方便切片
    zero = torch.zeros(B, 1, D, device=device, dtype=acoustic_feat.dtype)
    cumsum = torch.cat([zero, acoustic_feat.cumsum(dim=1)], dim=1)  # (B, T+1, D)

    s = word_spans[..., 0].clamp(min=0, max=T)   # (B, N)
    e = word_spans[..., 1].clamp(min=0, max=T)   # (B, N)
    e = torch.maximum(e, s + 1).clamp(max=T)     # 保证 e > s

    span_len = (e - s).clamp(min=1).unsqueeze(-1).to(acoustic_feat.dtype)  # (B, N, 1)

    def _pool(s_idx, e_idx):
        # gather cumsum at s_idx and e_idx → (B, N, D)
        s_exp = s_idx.unsqueeze(-1).expand(-1, -1, D)
        e_exp = e_idx.unsqueeze(-1).expand(-1, -1, D)
        return cumsum.gather(1, e_exp) - cumsum.gather(1, s_exp)

    # word_audio: [s, e)
    word_sum = _pool(s, e)
    word_audio = word_sum / span_len

    # tail_audio: [s + (1-tail_ratio)*(e-s), e)
    tail_start = s + ((e - s).float() * (1.0 - tail_ratio)).long()
    tail_start = torch.maximum(tail_start, s)
    tail_start = torch.minimum(tail_start, e - 1)
    tail_len = (e - tail_start).clamp(min=1).unsqueeze(-1).to(acoustic_feat.dtype)
    tail_sum = _pool(tail_start, e)
    tail_audio = tail_sum / tail_len

    # post_audio: [e, min(e + post_frames, T))
    post_end = (e + post_frames).clamp(max=T)
    # 处理 post 段为空的情况：fallback 到最后一帧
    empty = post_end <= e
    post_start = torch.where(empty, (e - 1).clamp(min=0), e)
    post_end_safe = torch.where(empty, e.clamp(max=T), post_end)
    post_len = (post_end_safe - post_start).clamp(min=1).unsqueeze(-1).to(acoustic_feat.dtype)
    post_sum = _pool(post_start, post_end_safe)
    post_audio = post_sum / post_len

    return torch.cat([word_audio, tail_audio, post_audio], dim=-1)


class IntonationV4Model(nn.Module):
    def __init__(self, vocab_size, encoder_type=C.ENCODER_TYPE):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(C.FBANK_DIM, C.CONV_DIM, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Dropout(C.DROPOUT),
            nn.Conv1d(C.CONV_DIM, C.CONV_DIM, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Dropout(C.DROPOUT),
        )

        if encoder_type == "bilstm":
            self.acoustic_encoder = BiLSTMEncoder()
        elif encoder_type == "bilstm_residual":
            self.acoustic_encoder = ResidualBiLSTMEncoder()
        elif encoder_type == "transformer":
            self.acoustic_encoder = TransformerAcousticEncoder()
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")
        acoustic_dim = self.acoustic_encoder.out_dim

        self.text_encoder = TextEncoder(vocab_size)

        self.cross_attn = CrossAttentionAlignment(
            text_dim=self.text_encoder.out_dim,
            acoustic_dim=acoustic_dim,
        )

        # Break head: 吃 cross-attn 输出（和 exp5 一样）
        self.break_head = nn.Linear(acoustic_dim, 1)

        # Tone head: 吃 span pooling 输出（新）
        tone_h1 = 256
        tone_h2 = 128
        tone_h3 = 64
        self.tone_head = nn.Sequential(
            nn.Linear(C.TONE_INPUT_DIM, tone_h1),
            nn.GELU(),
            nn.Dropout(C.DROPOUT),
            nn.Linear(tone_h1, tone_h2),
            nn.GELU(),
            nn.Dropout(C.DROPOUT),
            nn.Linear(tone_h2, tone_h3),
            nn.GELU(),
            nn.Dropout(C.DROPOUT),
            nn.Linear(tone_h3, C.N_TONES),
        )

    def forward(self, fbank, fbank_mask, text_tokens, text_mask, word_spans):
        """
        Args:
            fbank:       (B, T, 80)
            fbank_mask:  (B, T)
            text_tokens: (B, N)
            text_mask:   (B, N)
            word_spans:  (B, N, 2) — 下采样后的帧索引 [start, end)

        Returns:
            break_logits: (B, N)
            tone_logits:  (B, N, 2)
            attn_weights: (B, N, T')
        """
        # Acoustic encoder
        h = self.conv(fbank.transpose(1, 2)).transpose(1, 2)
        T_down = h.size(1)

        acoustic_mask = fbank_mask[:, ::4]
        if acoustic_mask.size(1) != T_down:
            if acoustic_mask.size(1) < T_down:
                pad = torch.zeros(acoustic_mask.size(0), T_down - acoustic_mask.size(1),
                                  device=acoustic_mask.device)
                acoustic_mask = torch.cat([acoustic_mask, pad], dim=1)
            else:
                acoustic_mask = acoustic_mask[:, :T_down]

        acoustic_feat = self.acoustic_encoder(h, acoustic_mask)

        # Text encoder
        text_feat = self.text_encoder(text_tokens, text_mask)

        # Break path: cross-attn
        word_acoustic, attn_weights = self.cross_attn(
            text_feat, acoustic_feat, acoustic_mask
        )
        break_logits = self.break_head(word_acoustic).squeeze(-1)

        # Tone path: span pooling（不经过 cross-attn）
        tone_input = span_pool(acoustic_feat, word_spans)
        tone_logits = self.tone_head(tone_input)

        return {
            "break_logits": break_logits,
            "tone_logits": tone_logits,
            "attn_weights": attn_weights,
        }


if __name__ == "__main__":
    vocab_size = 1000
    model = IntonationV4Model(vocab_size)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {n_params / 1e6:.2f}M")

    B, T, N = 2, 400, 12
    fbank = torch.randn(B, T, 80)
    fbank_mask = torch.ones(B, T)
    fbank_mask[1, 300:] = 0
    text_tokens = torch.randint(1, vocab_size, (B, N))
    text_mask = torch.ones(B, N)
    text_mask[1, 10:] = 0
    word_spans = torch.zeros(B, N, 2, dtype=torch.long)
    for i in range(N):
        word_spans[:, i, 0] = i * 8
        word_spans[:, i, 1] = (i + 1) * 8

    out = model(fbank, fbank_mask, text_tokens, text_mask, word_spans)
    print(f"break_logits: {out['break_logits'].shape}")
    print(f"tone_logits:  {out['tone_logits'].shape}")
    print(f"attn_weights: {out['attn_weights'].shape}")
    print("验证通过")
