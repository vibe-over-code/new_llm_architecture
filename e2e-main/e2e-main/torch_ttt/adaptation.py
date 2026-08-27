from collections.abc import Iterable

import torch
from torch import nn
from torch.func import functional_call, grad


def _selected(model: nn.Module, names: Iterable[str]):
    names = tuple(names)
    return {name: p for name, p in model.named_parameters() if any(name.startswith(prefix) for prefix in names)}


def adapt(model: nn.Module, context: torch.Tensor, *, lr: float, steps: int = 1, parameter_prefixes=("blocks.",)) -> nn.Module:
    """Clone and adapt a model on ``context``; the original model is untouched."""
    adapted = type(model)(model.cfg).to(device=context.device, dtype=next(model.parameters()).dtype)
    adapted.load_state_dict(model.state_dict())
    for p in adapted.parameters():
        p.requires_grad_(False)
    chosen = _selected(adapted, parameter_prefixes)
    for p in chosen.values():
        p.requires_grad_(True)
    optimizer = torch.optim.SGD(chosen.values(), lr=lr)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        adapted.loss(context).backward()
        optimizer.step()
    return adapted


def meta_loss(model: nn.Module, context: torch.Tensor, query: torch.Tensor, *, lr: float, steps: int = 1,
              parameter_prefixes=("blocks.",)) -> torch.Tensor:
    """Differentiable inner loop for end-to-end/meta TTT training."""
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    selected = set(_selected(model, parameter_prefixes))

    def loss_with(p, tokens):
        logits = functional_call(model, (p, buffers), (tokens,))
        return torch.nn.functional.cross_entropy(logits[:, :-1].float().reshape(-1, logits.size(-1)), tokens[:, 1:].reshape(-1))

    for _ in range(steps):
        inner = loss_with(params, context)
        grads = torch.autograd.grad(inner, [params[n] for n in selected], create_graph=True)
        params = dict(params)
        for name, g in zip(selected, grads):
            params[name] = params[name] - lr * g
    return loss_with(params, query)
