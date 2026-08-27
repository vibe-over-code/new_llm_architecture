import math

import torch
from torch import nn
from torch.nn import functional as F

from .config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype) * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int, theta: float):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        positions = torch.arange(max_seq_len).float()
        freqs = torch.outer(positions, inv_freq)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        length = q.shape[-2]
        cos = self.cos[:length].to(q.device, q.dtype)[None, None]
        sin = self.sin[:length].to(q.device, q.dtype)[None, None]

        def rotate(x):
            x1, x2 = x[..., ::2], x[..., 1::2]
            return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)

        return rotate(q), rotate(k)


class SlidingWindowAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.hidden_size % cfg.num_heads == 0
        self.heads = cfg.num_heads
        self.head_dim = cfg.hidden_size // cfg.num_heads
        self.window = cfg.sliding_window
        self.qkv = nn.Linear(cfg.hidden_size, 3 * cfg.hidden_size, bias=False)
        self.out = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("mask", torch.empty(0, dtype=torch.bool), persistent=False)

    def _mask(self, length: int, device: torch.device) -> torch.Tensor:
        if self.mask.shape != (length, length) or self.mask.device != device:
            pos = torch.arange(length, device=device)
            # True means allowed for scaled_dot_product_attention.
            self.mask = (pos[None, :] <= pos[:, None]) & ((pos[:, None] - pos[None, :]) < self.window)
        return self.mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        shape = (batch, length, self.heads, self.head_dim)
        q, k, v = (z.view(shape).transpose(1, 2) for z in (q, k, v))
        q, k = self.rope(q, k)
        mask = self._mask(length, x.device)
        # The CUDA flash/efficient SDPA kernels in some PyTorch versions do
        # not implement double backward. Meta-TTT needs double backward
        # through the inner update, so use the plain differentiable path.
        # This is intentionally kept here as a small, easy-to-replace backend
        # seam for a future memory-efficient local-attention kernel.
        scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        y = torch.matmul(scores.softmax(dim=-1).to(v.dtype), v)
        return self.out(y.transpose(1, 2).reshape(batch, length, width))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.hidden_size)
        self.attn = SlidingWindowAttention(cfg)
        self.norm2 = RMSNorm(cfg.hidden_size)
        self.w1 = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.w3 = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.w2 = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.w2(F.silu(self.w1(self.norm2(x))) * self.w3(self.norm2(x)))
        return x


class CausalTTTModel(nn.Module):
    """Decoder-only LM; adaptation is normally applied to ``blocks.*.w*``."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.num_layers)])
        self.norm = RMSNorm(cfg.hidden_size)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.norm(x))

    def loss(self, input_ids: torch.Tensor) -> torch.Tensor:
        logits = self(input_ids)
        return F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.size(-1)), input_ids[:, 1:].reshape(-1))
