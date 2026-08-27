from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 32000
    hidden_size: int = 512
    intermediate_size: int = 1376
    num_layers: int = 8
    num_heads: int = 8
    max_seq_len: int = 32768
    sliding_window: int = 8192
    dropout: float = 0.0
    rope_theta: float = 500000.0


@dataclass
class TrainConfig:
    batch_size: int = 2
    seq_len: int = 2048
    steps: int = 1000
    lr: float = 3e-4
    ttt_lr: float = 1e-2
    ttt_steps: int = 1
    grad_accum: int = 1
    mode: str = "pretrain"  # pretrain | meta
    device: str = "cuda"
    amp_dtype: str = "bf16"
