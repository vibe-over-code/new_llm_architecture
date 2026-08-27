import argparse

import torch
from torch.amp import autocast

from .config import ModelConfig, TrainConfig
from .model import CausalTTTModel
from .adaptation import meta_loss


def main():
    p = argparse.ArgumentParser(description="Minimal CUDA PyTorch TTT-E2E trainer")
    p.add_argument("--mode", choices=("pretrain", "meta"), default="pretrain")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--vocab-size", type=int, default=32000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--data", help="optional .npy file with a 1-D array of token ids")
    args = p.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Install a CUDA-enabled PyTorch build or use --device cpu.")
    device = torch.device(args.device)
    model_cfg = ModelConfig(vocab_size=args.vocab_size, max_seq_len=args.seq_len)
    train_cfg = TrainConfig(mode=args.mode, steps=args.steps, batch_size=args.batch_size, seq_len=args.seq_len)
    model = CausalTTTModel(model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr)
    amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    token_data = None
    if args.data:
        import numpy as np

        token_data = torch.from_numpy(np.load(args.data, mmap_mode="r")).long()
        if token_data.ndim != 1 or token_data.numel() < train_cfg.seq_len:
            raise ValueError("--data must be a 1-D .npy token array at least as long as --seq-len")
    model.train()
    for step in range(train_cfg.steps):
        if token_data is None:
            tokens = torch.randint(low=0, high=model_cfg.vocab_size, size=(train_cfg.batch_size, train_cfg.seq_len), device=device)
        else:
            starts = torch.randint(0, token_data.numel() - train_cfg.seq_len + 1, (train_cfg.batch_size,))
            tokens = torch.stack([token_data[s : s + train_cfg.seq_len] for s in starts]).to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
            if train_cfg.mode == "meta":
                split = train_cfg.seq_len // 2
                loss = meta_loss(model, tokens[:, :split], tokens[:, split - 1:], lr=train_cfg.ttt_lr)
            else:
                loss = model.loss(tokens)
        loss.backward()
        optimizer.step()
        if step % 10 == 0:
            print(f"step={step:05d} loss={loss.item():.4f}")


if __name__ == "__main__":
    main()
