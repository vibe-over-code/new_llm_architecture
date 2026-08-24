import math
import random
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# CONFIG
# ============================================================

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Размеры намеренно маленькие.
# Даже CPU должен справиться.
D_MODEL = 128
N_HEADS = 4
N_LAYERS = 2
FF_DIM = 256

N_MEMORY_SLOTS = 8
MEMORY_DIM = D_MODEL

N_USERS = 100
PRETRAIN_STEPS = 20_000

ONLINE_STEPS = 5_000

PRETRAIN_LR = 3e-4
MEMORY_LR = 0.15

print("=" * 60)
print("Device:", DEVICE)
print("=" * 60)


# ============================================================
# SYNTHETIC WORLD
#
# Мы создаём маленький искусственный мир.
#
# В реальном эксперименте здесь потом будет токенизатор
# и настоящий текст.
# ============================================================

FACTS = {
    "name": [
        "alex",
        "boris",
        "charlie",
        "david",
        "eva",
        "frank",
        "george",
        "helen",
    ],

    "animal": [
        "cat",
        "dog",
        "bird",
        "fish",
    ],

    "game": [
        "doom",
        "minecraft",
        "quake",
        "factorio",
    ],

    "city": [
        "paris",
        "moscow",
        "berlin",
        "london",
    ],
}

FACT_TYPES = list(FACTS.keys())

VALUES = sorted(
    set(
        value
        for values in FACTS.values()
        for value in values
    )
)

VALUE_TO_ID = {
    value: i
    for i, value in enumerate(VALUES)
}

ID_TO_VALUE = {
    i: value
    for value, i in VALUE_TO_ID.items()
}

TYPE_TO_ID = {
    name: i
    for i, name in enumerate(FACT_TYPES)
}


# ============================================================
# USER PROFILE
# ============================================================

@dataclass
class Profile:
    name: str
    animal: str
    game: str
    city: str

    def get(self, fact_type):
        return getattr(self, fact_type)


def random_profile():
    return Profile(
        name=random.choice(FACTS["name"]),
        animal=random.choice(FACTS["animal"]),
        game=random.choice(FACTS["game"]),
        city=random.choice(FACTS["city"]),
    )


# ============================================================
# MEMORY
#
# У пользователя НЕ хранится:
#
#     {"name": "alex"}
#
# Здесь только числа.
#
# Именно эти числа будут изменяться во время online learning.
# ============================================================

class UserMemory:

    def __init__(self):

        self.slots = torch.randn(
            N_MEMORY_SLOTS,
            MEMORY_DIM,
            device=DEVICE
        ) * 0.02

        self.slots.requires_grad_(True)

    def reset(self):

        with torch.no_grad():
            self.slots.normal_(0.0, 0.02)

        self.slots.requires_grad_(True)

    def update(self, loss):

        gradient = torch.autograd.grad(
            loss,
            self.slots,
            retain_graph=False
        )[0]

        with torch.no_grad():

            # Нормализация градиента делает эксперимент
            # стабильнее.
            norm = gradient.norm()

            if norm > 1.0:
                gradient = gradient / norm

            self.slots -= MEMORY_LR * gradient

        self.slots.requires_grad_(True)


# ============================================================
# POSITIONAL ENCODING
# ============================================================

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=32):

        super().__init__()

        position = torch.arange(
            max_len,
            dtype=torch.float
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0,
                d_model,
                2,
                dtype=torch.float
            )
            * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(
            max_len,
            d_model
        )

        pe[:, 0::2] = torch.sin(
            position * div_term
        )

        pe[:, 1::2] = torch.cos(
            position * div_term
        )

        self.register_buffer(
            "pe",
            pe.unsqueeze(0)
        )

    def forward(self, x):

        return x + self.pe[:, :x.size(1)]


# ============================================================
# SHARED MODEL
#
# ВАЖНО:
#
# Эта модель будет обучена ДО online learning.
#
# После pretraining все её параметры замораживаются.
# ============================================================

class OnlineMemoryTransformer(nn.Module):

    def __init__(self):

        super().__init__()

        # Query representing the fact we want.
        self.fact_embedding = nn.Embedding(
            len(FACT_TYPES),
            D_MODEL
        )

        # Memory slot positions.
        self.slot_embedding = nn.Embedding(
            N_MEMORY_SLOTS,
            D_MODEL
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=N_HEADS,
            dim_feedforward=FF_DIM,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=N_LAYERS
        )

        self.norm = nn.LayerNorm(D_MODEL)

        self.output = nn.Linear(
            D_MODEL,
            len(VALUES)
        )

    def forward(
        self,
        fact_type,
        memory
    ):
        """
        fact_type:
            [batch]

        memory:
            [batch, slots, dim]
        """

        batch_size = memory.size(0)

        # ----------------------------------------------------
        # Memory tokens
        # ----------------------------------------------------

        slot_ids = torch.arange(
            N_MEMORY_SLOTS,
            device=DEVICE
        )

        slot_ids = slot_ids.unsqueeze(0)

        slot_ids = slot_ids.expand(
            batch_size,
            -1
        )

        slot_pos = self.slot_embedding(
            slot_ids
        )

        memory_tokens = memory + slot_pos

        # ----------------------------------------------------
        # Query token
        # ----------------------------------------------------

        query = self.fact_embedding(
            fact_type
        ).unsqueeze(1)

        # ----------------------------------------------------
        # Sequence:
        #
        # [QUERY] [MEM0] [MEM1] ... [MEM7]
        # ----------------------------------------------------

        x = torch.cat(
            [
                query,
                memory_tokens
            ],
            dim=1
        )

        x = self.transformer(x)

        # We only read the query position.
        query_output = x[:, 0]

        query_output = self.norm(
            query_output
        )

        logits = self.output(
            query_output
        )

        return logits


# ============================================================
# MODEL
# ============================================================

model = OnlineMemoryTransformer().to(DEVICE)

print(
    "Parameters:",
    sum(p.numel() for p in model.parameters())
)


# ============================================================
# PRETRAINING
#
# Здесь модель учится:
#
#     "если memory содержит информацию,
#      то по query можно извлечь правильный value"
#
# Это НЕ обучение пользователей.
# ============================================================

print()
print("Pretraining shared model...")


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=PRETRAIN_LR
)


# ------------------------------------------------------------
# Чтобы pretraining был осмысленным, создаём "teacher memory".
#
# У каждого fact type есть свой обучаемый прототип.
#
# Это лишь способ научить общую сеть пользоваться memory.
# После pretraining эти прототипы исчезнут.
# ------------------------------------------------------------

teacher_memory = nn.Parameter(
    torch.randn(
        len(VALUES),
        MEMORY_DIM,
        device=DEVICE
    ) * 0.5
)

teacher_optimizer = torch.optim.AdamW(
    list(model.parameters()) + [teacher_memory],
    lr=PRETRAIN_LR
)


def make_teacher_memory(profile):
    """
    Создаёт memory из фактов профиля.

    ВАЖНО:
    это используется ТОЛЬКО во время pretraining.

    Во время online learning такой функции не будет.
    """

    memory = torch.zeros(
        N_MEMORY_SLOTS,
        MEMORY_DIM,
        device=DEVICE
    )

    facts = [
        profile.name,
        profile.animal,
        profile.game,
        profile.city,
    ]

    for slot, value in enumerate(facts):

        value_id = VALUE_TO_ID[value]

        memory[slot] = teacher_memory[value_id]

    return memory


for step in range(PRETRAIN_STEPS):

    profile = random_profile()

    fact_type = random.choice(
        FACT_TYPES
    )

    target_value = profile.get(
        fact_type
    )

    fact_id = torch.tensor(
        [TYPE_TO_ID[fact_type]],
        device=DEVICE
    )

    target_id = torch.tensor(
        [VALUE_TO_ID[target_value]],
        device=DEVICE
    )

    memory = make_teacher_memory(
        profile
    ).unsqueeze(0)

    logits = model(
        fact_id,
        memory
    )

    loss = F.cross_entropy(
        logits,
        target_id
    )

    teacher_optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        list(model.parameters()) + [teacher_memory],
        1.0
    )

    teacher_optimizer.step()

    if step % 2000 == 0:

        accuracy = (
            logits.argmax(dim=-1) == target_id
        ).float().mean().item()

        print(
            f"step={step:6d} "
            f"loss={loss.item():.4f} "
            f"accuracy={accuracy:.2f}"
        )


print("Pretraining finished.")


# ============================================================
# FREEZE SHARED MODEL
#
# ОТ ЭТОГО МОМЕНТА:
#
# model.parameters() НИКОГДА БОЛЬШЕ НЕ МЕНЯЮТСЯ.
#
# Меняться могут только UserMemory.slots.
# ============================================================

for parameter in model.parameters():
    parameter.requires_grad_(False)

teacher_memory.requires_grad_(False)


# ============================================================
# USER PROFILES
# ============================================================

alice_profile = Profile(
    name="alex",
    animal="cat",
    game="doom",
    city="paris",
)

bob_profile = Profile(
    name="boris",
    animal="dog",
    game="minecraft",
    city="moscow",
)


users = {
    "Alice": UserMemory(),
    "Bob": UserMemory(),
}


# ============================================================
# ONLINE LEARNING
# ============================================================

def learn_fact(
    user_name,
    fact_type,
    target_value
):

    user_memory = users[user_name]

    fact_id = torch.tensor(
        [TYPE_TO_ID[fact_type]],
        device=DEVICE
    )

    target_id = torch.tensor(
        [VALUE_TO_ID[target_value]],
        device=DEVICE
    )

    # --------------------------------------------------------
    # INFERENCE
    # --------------------------------------------------------

    logits = model(
        fact_id,
        user_memory.slots.unsqueeze(0)
    )

    loss = F.cross_entropy(
        logits,
        target_id
    )

    prediction = logits.argmax(
        dim=-1
    ).item()

    confidence = F.softmax(
        logits,
        dim=-1
    )[0, prediction].item()

    # --------------------------------------------------------
    # ONLINE LEARNING
    # --------------------------------------------------------

    user_memory.update(loss)

    return (
        ID_TO_VALUE[prediction],
        confidence,
        loss.item()
    )


# ============================================================
# ASK
# ============================================================

def ask(
    user_name,
    fact_type
):

    user_memory = users[user_name]

    fact_id = torch.tensor(
        [TYPE_TO_ID[fact_type]],
        device=DEVICE
    )

    with torch.no_grad():

        logits = model(
            fact_id,
            user_memory.slots.unsqueeze(0)
        )

        probabilities = F.softmax(
            logits,
            dim=-1
        )

        prediction = logits.argmax(
            dim=-1
        ).item()

        confidence = probabilities[
            0,
            prediction
        ].item()

    return (
        ID_TO_VALUE[prediction],
        confidence
    )


# ============================================================
# TRAIN A USER
# ============================================================

def teach_user(
    user_name,
    profile,
    repetitions=100
):

    print()
    print(
        f"Teaching {user_name}..."
    )

    facts = {
        "name": profile.name,
        "animal": profile.animal,
        "game": profile.game,
        "city": profile.city,
    }

    for _ in range(repetitions):

        fact_type = random.choice(
            FACT_TYPES
        )

        value = facts[fact_type]

        learn_fact(
            user_name,
            fact_type,
            value
        )


# ============================================================
# EVALUATION
# ============================================================

def evaluate_user(
    user_name,
    profile
):

    facts = {
        "name": profile.name,
        "animal": profile.animal,
        "game": profile.game,
        "city": profile.city,
    }

    correct = 0
    total = 0

    print()
    print(
        f"========== {user_name} =========="
    )

    for fact_type, expected in facts.items():

        prediction, confidence = ask(
            user_name,
            fact_type
        )

        ok = prediction == expected

        if ok:
            correct += 1

        total += 1

        print(
            f"{fact_type:8s} "
            f"expected={expected:10s} "
            f"got={prediction:10s} "
            f"confidence={confidence:.3f} "
            f"{'OK' if ok else 'FAIL'}"
        )

    accuracy = correct / total

    print(
        f"accuracy = {accuracy:.2%}"
    )

    return accuracy


# ============================================================
# INITIAL TEST
# ============================================================

print()
print("=" * 60)
print("INITIAL MEMORY TEST")
print("=" * 60)

evaluate_user(
    "Alice",
    alice_profile
)

evaluate_user(
    "Bob",
    bob_profile
)


# ============================================================
# TEACH USERS
# ============================================================

teach_user(
    "Alice",
    alice_profile,
    repetitions=200
)

teach_user(
    "Bob",
    bob_profile,
    repetitions=200
)


# ============================================================
# TEST AFTER LEARNING
# ============================================================

print()
print("=" * 60)
print("AFTER LEARNING")
print("=" * 60)

alice_before = evaluate_user(
    "Alice",
    alice_profile
)

bob_before = evaluate_user(
    "Bob",
    bob_profile
)


# ============================================================
# MIXED STREAM
#
# Теперь самое важное.
#
# Alice и Bob говорят одновременно.
#
# Их сообщения перемешаны.
# ============================================================

print()
print("=" * 60)
print("MIXED ONLINE STREAM")
print("=" * 60)


alice_facts = [
    ("name", alice_profile.name),
    ("animal", alice_profile.animal),
    ("game", alice_profile.game),
    ("city", alice_profile.city),
]

bob_facts = [
    ("name", bob_profile.name),
    ("animal", bob_profile.animal),
    ("game", bob_profile.game),
    ("city", bob_profile.city),
]


for step in range(ONLINE_STEPS):

    # randomly choose user
    if random.random() < 0.5:

        user = "Alice"
        fact_type, value = random.choice(
            alice_facts
        )

    else:

        user = "Bob"
        fact_type, value = random.choice(
            bob_facts
        )

    learn_fact(
        user,
        fact_type,
        value
    )

    if step % 1000 == 0:

        print(
            f"online step {step}"
        )


# ============================================================
# FINAL TEST
# ============================================================

print()
print("=" * 60)
print("FINAL TEST")
print("=" * 60)


alice_accuracy = evaluate_user(
    "Alice",
    alice_profile
)

bob_accuracy = evaluate_user(
    "Bob",
    bob_profile
)


# ============================================================
# CROSS USER TEST
#
# Здесь проверяем не только:
#
#     "помнит ли Alice?"
#
# но и:
#
#     "не получила ли Alice память Bob?"
# ============================================================

print()
print("=" * 60)
print("CROSS USER TEST")
print("=" * 60)

print()
print("Alice:")
evaluate_user(
    "Alice",
    bob_profile
)

print()
print("Bob:")
evaluate_user(
    "Bob",
    alice_profile
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 60)
print("RESULT")
print("=" * 60)

print(
    f"Alice accuracy: {alice_accuracy:.2%}"
)

print(
    f"Bob accuracy:   {bob_accuracy:.2%}"
)

print()
print("If Alice remembers Alice and Bob remembers Bob,")
print("while cross-user accuracy stays low,")
print("the basic personalized memory mechanism works.")