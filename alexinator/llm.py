"""Accès aux LLM.

Tout le reste du code ne connaît qu'une méthode : `chat(messages) -> str`.
- `LLMClient` parle à n'importe quelle API compatible OpenAI (Ollama, vLLM, fournisseur en ligne...).
- `MockLLM` est un faux LLM pour les tests : aucune connexion réseau.
Changer de modèle = changer trois variables dans `.env`, sans toucher au code.
"""

from __future__ import annotations

from collections.abc import Callable

Message = dict[str, str]  # {"role": "system" | "user" | "assistant", "content": "..."}


class LLMClient:
    def __init__(self, base_url: str, api_key: str, model: str):
        # Import tardif : les tests (MockLLM) tournent sans le paquet openai.
        from openai import OpenAI

        self.model = model
        self._client = OpenAI(base_url=base_url, api_key=api_key)

    def chat(self, messages: list[Message], temperature: float = 0.7, max_tokens: int = 400) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return (response.choices[0].message.content or "").strip()


class MockLLM:
    """Faux LLM. `reply` est soit un texte fixe, soit une fonction messages -> texte."""

    def __init__(self, reply: str | Callable[[list[Message]], str] = "...", model: str = "mock"):
        self.model = model
        self._reply = reply
        self.calls: list[list[Message]] = []  # historique des appels, pratique dans les tests

    def chat(self, messages: list[Message], temperature: float = 0.7, max_tokens: int = 400) -> str:
        self.calls.append(messages)
        return self._reply(messages) if callable(self._reply) else self._reply


class TransformersLLM:
    """LLM chargé directement sur le GPU avec transformers, sans serveur.

    Sert à jouer avec NOTRE modèle entraîné : `model` est le modèle de base et `adapter` le dossier
    de l'adaptateur LoRA produit par finetune/entrainement_qlora.ipynb (vide = modèle de base seul).
    """

    def __init__(self, model: str, adapter: str = ""):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.model = f"{model}+{adapter}" if adapter else model
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(adapter or model)
        quantization = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            model, quantization_config=quantization, device_map="auto"
        )
        if adapter:
            from peft import PeftModel

            self._model = PeftModel.from_pretrained(self._model, adapter)
        self._model.eval()

    def chat(self, messages: list[Message], temperature: float = 0.7, max_tokens: int = 400) -> str:
        inputs = self._tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(self._model.device)
        sampling = (
            {"do_sample": True, "temperature": temperature} if temperature > 0 else {"do_sample": False}
        )
        with self._torch.no_grad():
            output = self._model.generate(**inputs, max_new_tokens=max_tokens, **sampling)
        new_tokens = output[0][inputs["input_ids"].shape[1] :]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


class AlexinatorLLM:
    """Notre propre modèle (Alexinator-110M, entraîné avec entrainement/pretrain.py puis sft.py)."""

    def __init__(self, checkpoint: str):
        import torch
        from tokenizers import Tokenizer

        from .gpt import FIN, encode_conversation, load_checkpoint

        self._torch = torch
        self._encode = encode_conversation
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model, ckpt = load_checkpoint(checkpoint, self._device)
        self._model.eval()
        self._tokenizer = Tokenizer.from_file(ckpt["tokenizer"])
        self._fin = self._tokenizer.token_to_id(FIN)
        self.model = f"alexinator-110m:{checkpoint}"

    def chat(self, messages: list[Message], temperature: float = 0.7, max_tokens: int = 400) -> str:
        # Le gardien répond en 3 phrases : on garde de la place pour le prompt.
        max_tokens = min(max_tokens, 160)
        ids, _ = self._encode(self._tokenizer, messages, add_generation_prompt=True)
        block = self._model.cfg.block_size
        # On garde la fin si la conversation dépasse la fenêtre (le prompt système est alors tronqué).
        idx = self._torch.tensor([ids[-(block - max_tokens):] if len(ids) > block - max_tokens else ids],
                                 device=self._device)
        out = self._model.generate(idx, max_tokens, temperature=temperature, top_k=50, stop_id=self._fin)
        return self._tokenizer.decode(out[0, idx.size(1):].tolist(), skip_special_tokens=True).strip()


_CACHE: dict = {}


def make_llm(settings, model: str = "", aux: bool = False):
    """Crée le LLM décrit par la configuration (.env). Un même modèle n'est chargé qu'une fois.

    aux=True pour les rôles auxiliaires (censeur, détecteur, devineur, professeur) : ils suivent
    ALEXINATOR_AUX_BACKEND, ce qui permet d'avoir notre petit modèle en gardien et un modèle plus
    gros pour le reste.
    """
    model = model or settings.model
    backend = settings.aux_backend if aux else settings.backend
    key = (backend, model, settings.adapter, settings.checkpoint)
    if key not in _CACHE:
        if backend == "alexinator":
            _CACHE[key] = AlexinatorLLM(settings.checkpoint)
        elif backend == "local":
            _CACHE[key] = TransformersLLM(model, settings.adapter)
        else:
            _CACHE[key] = LLMClient(settings.base_url, settings.api_key, model)
    return _CACHE[key]
