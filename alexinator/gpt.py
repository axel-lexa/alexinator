"""Alexinator-110M : notre propre GPT décodeur (voir le document « Modélisation de notre LLM »).

Ce module est partagé par les scripts d'entraînement (dossier entrainement/) et par le jeu
(backend « alexinator » dans llm.py). Il contient :
- GPTConfig / GPT : le modèle (pre-norm RMSNorm, RoPE, attention causale FlashAttention, SwiGLU,
  embedding et tête de sortie partagés) ;
- les tokens spéciaux et la mise en forme des conversations ;
- la génération (échantillonnage token par token, comme au lab 5).

Différences avec le GPT du lab 5 : voir le tableau du document de modélisation.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Tokens spéciaux, réservés dès l'entraînement du tokenizer
# ---------------------------------------------------------------------------
FIN_TEXTE = "<|fin_texte|>"  # sépare deux documents pendant le pré-entraînement
PAD = "<|pad|>"  # remplissage des batchs de la phase 2
SYSTEME = "<|systeme|>"  # début du prompt système (consignes + fiche du film)
JOUEUR = "<|joueur|>"  # début d'un message du joueur
GARDIEN = "<|gardien|>"  # début d'une réponse du gardien
FIN = "<|fin|>"  # fin d'un message : le gardien apprend à s'arrêter là
SPECIAL_TOKENS = [FIN_TEXTE, PAD, SYSTEME, JOUEUR, GARDIEN, FIN]
ROLE_TOKENS = {"system": SYSTEME, "user": JOUEUR, "assistant": GARDIEN}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class GPTConfig:
    vocab_size: int = 32000
    block_size: int = 1024  # fenêtre de contexte
    n_layer: int = 12
    n_head: int = 12
    d_model: int = 768
    d_ff: int = 2048  # couche cachée du MLP SwiGLU (≈ 8/3 × d_model)
    dropout: float = 0.0  # 0 en pré-entraînement, 0.1 en phase 2
    rope_base: float = 10000.0

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Briques du modèle
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    """Normalise chaque vecteur par sa norme quadratique moyenne (sans soustraire la moyenne)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        xf = x.float()  # calcul en float32 pour la stabilité, même sous bfloat16
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return xf.type_as(x) * self.weight


def rope_tables(head_dim: int, max_len: int, base: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Tables cos/sin de RoPE : la paire de dimensions i tourne d'un angle position × base^(-2i/d)."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    angles = torch.outer(torch.arange(max_len).float(), inv_freq)  # (max_len, head_dim / 2)
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    """Fait tourner les paires (x[2i], x[2i+1]) d'un angle qui dépend de la position.

    x : (batch, têtes, positions, head_dim). Le produit scalaire entre une requête et une clé tournées
    ne dépend alors que de la distance entre leurs positions.
    """
    T = x.size(-2)
    cos, sin = cos[:T], sin[:T]
    x1, x2 = x[..., 0::2].float(), x[..., 1::2].float()
    rotated = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)
    return rotated.type_as(x)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.d_model // cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)  # requêtes, clés, valeurs d'un coup
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, têtes, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        # is_causal=True : chaque position ne voit que les précédentes, à l'entraînement comme en génération.
        # PyTorch choisit FlashAttention quand c'est possible : la matrice T x T n'est jamais stockée.
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class SwiGLU(nn.Module):
    """MLP « porte » : w2( silu(w1 x) * w3 x ). Trois matrices au lieu de deux."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.w1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w3 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w2 = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class Block(nn.Module):
    """Pre-norm : on normalise AVANT chaque sous-couche, et la connexion résiduelle reste intacte."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, cos, sin):
        x = x + self.drop(self.attn(self.norm1(x), cos, sin))
        x = x + self.mlp(self.norm2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # poids partagés : 24,6 M de paramètres en moins

        cos, sin = rope_tables(cfg.d_model // cfg.n_head, cfg.block_size, cfg.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        # Initialisation de GPT-2 : N(0, 0.02), et les projections qui retournent dans le flux
        # résiduel réduites de 1/sqrt(2 x nombre de blocs) pour que sa variance reste stable.
        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith("attn.proj.weight") or name.endswith("mlp.w2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())  # l'embedding partagé n'est compté qu'une fois

    def forward(self, idx, targets: torch.Tensor | None = None):
        """idx : (B, T) identifiants de tokens. targets : (B, T), -100 = position ignorée par la loss."""
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"séquence de {T} tokens > fenêtre de {self.cfg.block_size}"
        x = self.drop(self.tok_emb(idx))
        for block in self.blocks:
            x = block(x, self.rope_cos, self.rope_sin)
        logits = self.lm_head(self.norm_f(x))
        loss = None
        if targets is not None:
            # Cross-entropy en float32 : les logits bfloat16 sont trop imprécis pour la loss.
            loss = F.cross_entropy(
                logits.float().view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100
            )
        return logits, loss

    def configure_optimizer(self, lr: float, weight_decay: float, betas=(0.9, 0.95)):
        """AdamW. Le weight decay ne s'applique qu'aux matrices (pas aux poids des normalisations)."""
        decay = [p for p in self.parameters() if p.requires_grad and p.dim() >= 2]
        no_decay = [p for p in self.parameters() if p.requires_grad and p.dim() < 2]
        groups = [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
        fused = torch.cuda.is_available()
        return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=fused)

    @torch.no_grad()
    def generate(
        self, idx, max_new_tokens: int, temperature: float = 0.8, top_k: int = 50, stop_id: int | None = None
    ):
        """Ajoute des tokens un par un (comme gpt_generate du lab 5), jusqu'à stop_id ou max_new_tokens."""
        was_training = self.training
        self.eval()
        for _ in range(max_new_tokens):
            context = idx[:, -self.cfg.block_size :]
            logits, _ = self(context)
            logits = logits[:, -1, :].float()
            if temperature <= 0:
                next_id = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k:
                    kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
                    logits[logits < kth] = -float("inf")
                next_id = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
            if stop_id is not None and next_id.item() == stop_id:
                break
        self.train(was_training)
        return idx


# ---------------------------------------------------------------------------
# Mise en forme des conversations (phase 2 et jeu)
# ---------------------------------------------------------------------------
def sanitize(text: str) -> str:
    """Empêche un joueur d'écrire lui-même un token spécial (« <|gardien|> ») pour piéger le modèle."""
    return text.replace("<|", "< |")


def encode_conversation(tokenizer, messages: list[dict[str, str]], add_generation_prompt: bool = False):
    """Transforme [{"role", "content"}, ...] en (ids, labels).

    Format :  <|systeme|> ... <|fin|> <|joueur|> ... <|fin|> <|gardien|> ... <|fin|>
    labels = -100 partout, sauf sur le contenu des réponses du gardien et leur <|fin|> :
    le modèle n'apprend qu'à répondre (masque de la loss, comme dans le notebook QLoRA).
    """
    ids, labels = [], []
    fin = tokenizer.token_to_id(FIN)
    for message in messages:
        role_id = tokenizer.token_to_id(ROLE_TOKENS[message["role"]])
        content = tokenizer.encode(sanitize(message["content"])).ids
        learn = message["role"] == "assistant"
        ids += [role_id] + content + [fin]
        labels += [-100] + (content + [fin] if learn else [-100] * (len(content) + 1))
    if add_generation_prompt:
        ids.append(tokenizer.token_to_id(GARDIEN))
        labels.append(-100)
    return ids, labels


def load_checkpoint(path: str, device: str = "cpu", dropout: float | None = None) -> tuple[GPT, dict]:
    """Recharge un modèle sauvegardé par pretrain.py ou sft.py."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = GPTConfig(**ckpt["config"])
    if dropout is not None:
        cfg.dropout = dropout
    model = GPT(cfg)
    model.load_state_dict(ckpt["model"])
    return model.to(device), ckpt
