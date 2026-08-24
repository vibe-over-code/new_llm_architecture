import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# V9
#
# CONTINUOUS ONLINE LEARNING
#
# prediction -> error -> fast update -> next prediction
#
# NO USERS
# NO ROUTER
# NO SLOTS
# NO explicit learn(target)
# ============================================================


SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

D_MODEL = 128
N_HEADS = 4
N_LAYERS = 3
FF_DIM = 384

MAX_LEN = 64

PRETRAIN_STEPS = 10000
PRETRAIN_LR = 3e-4

FAST_LR = 0.03

print("=" * 72)
print("V9 — CONTINUOUS ONLINE LEARNING")
print("=" * 72)

print("device:", DEVICE)


# ============================================================
# VOCAB
# ============================================================

TEXT = """
the cat is blue .
the cat is small .
the cat likes fish .
the cat likes milk .
the dog is brown .
the dog is small .
the dog likes meat .
the dog likes water .
the bird is red .
the bird is small .
the bird likes seeds .
the bird can fly .
paris is a city .
paris is beautiful .
berlin is a city .
berlin is large .
london is a city .
london is old .
alex likes games .
alex likes pizza .
alex lives in paris .
boris likes music .
boris lives in berlin .
charlie likes books .
charlie lives in london .
what is the cat ?
what color is the cat ?
what does the cat like ?
what is the dog ?
what color is the dog ?
where does alex live ?
what does alex like ?
hello .
hi .
yes .
no .
and
a
an
is
are
my
your
name
favorite
game
food
city
"""

VOCAB = list(
    dict.fromkeys(
        TEXT.split()
    )
)

VOCAB = [
    "<PAD>",
    "<BOS>",
    "<EOS>",
    "<UNK>",
] + VOCAB

stoi = {
    x: i
    for i, x in enumerate(VOCAB)
}

itos = {
    i: x
    for x, i in stoi.items()
}

VOCAB_SIZE = len(VOCAB)

PAD_ID = stoi["<PAD>"]
BOS_ID = stoi["<BOS>"]
EOS_ID = stoi["<EOS>"]
UNK_ID = stoi["<UNK>"]


def encode(text):

    return [
        BOS_ID,
        *[
            stoi.get(
                x,
                UNK_ID
            )
            for x in text.lower().split()
        ],
        EOS_ID
    ]


# ============================================================
# PRETRAINING DATA
# ============================================================

PRETRAIN_CORPUS = [

    "the cat is blue .",
    "the cat is small .",
    "the cat likes fish .",
    "the cat likes milk .",

    "the dog is brown .",
    "the dog is small .",
    "the dog likes meat .",
    "the dog likes water .",

    "the bird is red .",
    "the bird is small .",
    "the bird likes seeds .",
    "the bird can fly .",

    "paris is a city .",
    "paris is beautiful .",

    "berlin is a city .",
    "berlin is large .",

    "london is a city .",
    "london is old .",

    "alex likes games .",
    "alex likes pizza .",
    "alex lives in paris .",

    "boris likes music .",
    "boris lives in berlin .",

    "charlie likes books .",
    "charlie lives in london .",

    "what is the cat ?",
    "what color is the cat ?",
    "what does the cat like ?",

    "what is the dog ?",
    "what color is the dog ?",

    "where does alex live ?",
    "what does alex like ?",

    "hello .",
    "hi .",
]


# ============================================================
# TRANSFORMER
# ============================================================

class TinyTransformer(nn.Module):

    def __init__(self):

        super().__init__()

        self.embedding = nn.Embedding(
            VOCAB_SIZE,
            D_MODEL
        )

        self.position = nn.Embedding(
            MAX_LEN,
            D_MODEL
        )

        layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=N_HEADS,
            dim_feedforward=FF_DIM,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            layer,
            N_LAYERS
        )

        self.norm = nn.LayerNorm(
            D_MODEL
        )

        self.lm_head = nn.Linear(
            D_MODEL,
            VOCAB_SIZE
        )

    def hidden(self, x):

        B, T = x.shape

        pos = torch.arange(
            T,
            device=x.device
        )[None, :]

        h = (
            self.embedding(x)
            +
            self.position(pos)
        )

        mask = torch.full(
            (T, T),
            float("-inf"),
            device=x.device
        )

        mask = torch.triu(
            mask,
            diagonal=1
        )

        h = self.transformer(
            h,
            mask=mask
        )

        return self.norm(h)

    def forward(self, x):

        return self.lm_head(
            self.hidden(x)
        )


model = TinyTransformer().to(DEVICE)


print(
    "parameters:",
    sum(
        p.numel()
        for p in model.parameters()
    )
)


# ============================================================
# PRETRAIN
# ============================================================

print()
print("=" * 72)
print("PRETRAINING SLOW NETWORK")
print("=" * 72)


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=PRETRAIN_LR
)


for step in range(
    PRETRAIN_STEPS
):

    text = random.choice(
        PRETRAIN_CORPUS
    )

    ids = encode(text)

    x = torch.tensor(
        ids[:-1],
        dtype=torch.long,
        device=DEVICE
    )[None]

    y = torch.tensor(
        ids[1:],
        dtype=torch.long,
        device=DEVICE
    )[None]

    logits = model(x)

    loss = F.cross_entropy(
        logits.reshape(
            -1,
            VOCAB_SIZE
        ),
        y.reshape(-1)
    )

    optimizer.zero_grad()

    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        1.0
    )

    optimizer.step()

    if step % 1000 == 0:

        print(
            f"step={step:5d} "
            f"loss={loss.item():.4f}"
        )


print("pretraining finished")


# ============================================================
# FREEZE SLOW WEIGHTS
# ============================================================

for p in model.parameters():

    p.requires_grad_(False)


# ============================================================
# FAST WEIGHTS
#
# This is the important part.
#
# They are NOT trained by Adam.
#
# They are the dynamic state of the agent.
# ============================================================


class FastWeights:

    def __init__(self, d):

        self.d = d

        # Fast transformation
        self.W = torch.zeros(
            d,
            d,
            device=DEVICE
        )

        self.b = torch.zeros(
            d,
            device=DEVICE
        )

    def reset(self):

        self.W.zero_()
        self.b.zero_()

    def apply(self, h):

        return (
            h
            +
            h @ self.W.T
            +
            self.b
        )

    def update(
        self,
        h,
        target,
        lr
    ):

        # ----------------------------------------------------
        # Current fast prediction
        # ----------------------------------------------------

        prediction = self.apply(
            h
        )

        # ----------------------------------------------------
        # Delta
        # ----------------------------------------------------

        error = (
            target
            -
            prediction
        )

        # ----------------------------------------------------
        # Online weight update
        #
        # W <- W + lr * error * h
        #
        # This happens DURING inference.
        # ----------------------------------------------------

        self.W += (
            lr
            *
            error.T
            @
            h
        )

        self.b += (
            lr
            *
            error.mean(
                dim=0
            )
        )

        # ----------------------------------------------------
        # Stability
        # ----------------------------------------------------

        norm = self.W.norm()

        if norm > 20:

            self.W *= (
                20.0 / norm
            )

        bnorm = self.b.norm()

        if bnorm > 10:

            self.b *= (
                10.0 / bnorm
            )


fast = FastWeights(
    D_MODEL
)


# ============================================================
# CONTINUOUS AGENT
# ============================================================


def process_stream(
    text,
    report_every=20
):

    ids = encode(
        text
    )

    losses = []

    correct = 0

    total = 0

    print()
    print("=" * 72)
    print("ONLINE STREAM")
    print("=" * 72)

    for t in range(
        len(ids) - 1
    ):

        # ----------------------------------------------------
        # Current token
        # ----------------------------------------------------

        current = torch.tensor(
            [[ids[t]]],
            dtype=torch.long,
            device=DEVICE
        )

        target_id = ids[t + 1]

        target = torch.tensor(
            [target_id],
            dtype=torch.long,
            device=DEVICE
        )

        # ----------------------------------------------------
        # Frozen Transformer
        # ----------------------------------------------------

        with torch.no_grad():

            h = model.hidden(
                current
            )

        h = h[:, -1]

        # ----------------------------------------------------
        # FAST WEIGHTS AFFECT THE MODEL
        # ----------------------------------------------------

        fast_h = fast.apply(
            h
        )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        logits = (
            fast_h
            @
            model.lm_head.weight.T
            +
            model.lm_head.bias
        )

        loss = F.cross_entropy(
            logits,
            target
        )

        prediction = int(
            logits.argmax(
                dim=-1
            )
        )

        ok = (
            prediction
            ==
            target_id
        )

        correct += int(ok)

        total += 1

        losses.append(
            float(loss)
        )

        # ----------------------------------------------------
        # TARGET REPRESENTATION
        #
        # We use the slow network's embedding of the
        # ACTUAL next token.
        #
        # The agent doesn't receive this until AFTER
        # making its prediction.
        # ----------------------------------------------------

        with torch.no_grad():

            target_h = model.embedding(
                target
            )

        # ----------------------------------------------------
        # IMMEDIATELY CHANGE FAST WEIGHTS
        # ----------------------------------------------------

        fast.update(
            h,
            target_h,
            FAST_LR
        )

        # ----------------------------------------------------
        # PRINT
        # ----------------------------------------------------

        if (
            t % report_every == 0
        ):

            pred_word = itos[
                prediction
            ]

            target_word = itos[
                target_id
            ]

            print(
                f"t={t:4d} "
                f"input={itos[ids[t]]:10s} "
                f"target={target_word:10s} "
                f"pred={pred_word:10s} "
                f"loss={loss.item():.3f} "
                f"acc={correct / total:.3f}"
            )

    return {
        "loss": sum(losses)
        / len(losses),

        "accuracy":
            correct / total
    }


# ============================================================
# IMPORTANT:
#
# BEFORE THE STREAM, FAST MEMORY IS EMPTY.
# ============================================================

fast.reset()


print()
print("=" * 72)
print("FAST MEMORY BEFORE STREAM")
print("=" * 72)

print(
    "W norm:",
    fast.W.norm().item()
)

print(
    "b norm:",
    fast.b.norm().item()
)


# ============================================================
# STREAM
#
# This is the actual experiment.
#
# Every token:
#
# prediction
#     ↓
# error
#     ↓
# fast weight update
#     ↓
# next token
#
# ============================================================


stream = """
the cat is blue .
the cat is small .
the cat likes fish .
the dog is brown .
the dog is small .
the dog likes water .
the bird is red .
the bird can fly .
paris is a city .
paris is beautiful .
berlin is a city .
berlin is large .
"""


result = process_stream(
    stream
)


print()
print("=" * 72)
print("STREAM RESULT")
print("=" * 72)

print(
    "average loss:",
    result["loss"]
)

print(
    "accuracy:",
    result["accuracy"]
)


# ============================================================
# FAST MEMORY STATE
# ============================================================

print()
print("=" * 72)
print("FAST MEMORY AFTER STREAM")
print("=" * 72)

print(
    "W norm:",
    fast.W.norm().item()
)

print(
    "W max:",
    fast.W.abs().max().item()
)

print(
    "b norm:",
    fast.b.norm().item()
)


# ============================================================
# REPLAY TEST
#
# Give the model a sequence again.
#
# The slow weights are unchanged.
# Only fast memory contains the experience.
# ============================================================


def evaluate_stream(
    text
):

    ids = encode(
        text
    )

    correct = 0
    total = 0

    with torch.no_grad():

        for t in range(
            len(ids) - 1
        ):

            current = torch.tensor(
                [[ids[t]]],
                dtype=torch.long,
                device=DEVICE
            )

            target_id = ids[t + 1]

            h = model.hidden(
                current
            )[:, -1]

            fast_h = fast.apply(
                h
            )

            logits = (
                fast_h
                @
                model.lm_head.weight.T
                +
                model.lm_head.bias
            )

            prediction = int(
                logits.argmax(
                    dim=-1
                )
            )

            correct += (
                prediction
                ==
                target_id
            )

            total += 1

    return (
        correct / total
    )


print()
print("=" * 72)
print("REPLAY AFTER ONLINE LEARNING")
print("=" * 72)

replay_acc = evaluate_stream(
    stream
)

print(
    "replay accuracy:",
    replay_acc
)


# ============================================================
# LONG DISTRACTOR
#
# We now give the agent lots of unrelated information.
#
# Then test whether the original stream is still represented
# in fast weights.
# ============================================================


DISTRACTOR = """
hello .
hi .
the dog is small .
the bird is small .
hello .
the cat is small .
paris is beautiful .
hello .
the dog likes meat .
the bird likes seeds .
hello .
berlin is large .
the dog is brown .
hello .
the cat likes milk .
"""


print()
print("=" * 72)
print("DISTRACTOR STREAM")
print("=" * 72)


process_stream(
    DISTRACTOR,
    report_every=1000
)


print()
print("=" * 72)
print("REPLAY AFTER DISTRACTOR")
print("=" * 72)

replay_acc_after = evaluate_stream(
    stream
)

print(
    "replay accuracy:",
    replay_acc_after
)


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 72)
print("FINAL")
print("=" * 72)

print(
    "W norm:",
    fast.W.norm().item()
)

print(
    "W max:",
    fast.W.abs().max().item()
)

print(
    "b norm:",
    fast.b.norm().item()
)

print(
    "initial replay:",
    replay_acc
)

print(
    "after distractor:",
    replay_acc_after
)

print()
print("Experiment finished.")