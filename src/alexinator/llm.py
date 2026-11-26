"""Accès aux LLM.

Tout le reste du code ne connaît qu'une méthode : `chat(messages) -> str`.
- `LLMClient` parle à n'importe quelle API compatible OpenAI (Ollama, vLLM, fournisseur en ligne...).
- `MockLLM` est un faux LLM pour les tests : aucune connexion réseau.
- `DemoLLM` imite grossièrement un gardien pour montrer l'interface sans serveur (backend « demo »).
Changer de modèle = changer trois variables dans `.env`, sans toucher au code.
"""

from __future__ import annotations

import json
import random
import re
import time
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


class DemoLLM:
    """Faux LLM pour montrer et développer l'interface sans serveur (ALEXINATOR_BACKEND=demo).

    Il joue grossièrement chaque rôle pour que tous les écrans soient atteignables :
    - gardien : un indice tiré de la fiche, ou le titre si le joueur le demande (le filtre le bloque
      dès le niveau 3) ; oui/non au niveau 6 ; au niveau 7, « vérifie X » appelle l'outil ;
    - détecteur : ATTAQUE si le message contient un mot typique d'injection ; censeur : toujours OK.
    `delai` simule le temps de réponse d'un vrai modèle (pour voir défiler les étapes).
    """

    ATTACK_WORDS = ("ignore", "consigne", "développeur", "developpeur", "épelle", "epelle", "admin")

    def __init__(self, delai: float = 0.7, rng: random.Random | None = None):
        self.model = "demo"
        self.delai = delai
        self.rng = rng or random.Random()

    def chat(self, messages: list[Message], temperature: float = 0.7, max_tokens: int = 400) -> str:
        time.sleep(self.delai)
        first, last = messages[0]["content"], messages[-1]["content"]
        if "FUITE ou OK" in first:  # censeur
            return "OK"
        if "ATTAQUE ou HONNETE" in first:  # détecteur d'attaques
            message = first.split("---")[1].lower()
            return "ATTAQUE" if any(word in message for word in self.ATTACK_WORDS) else "HONNETE"
        if "OUI ou NON" in first:  # juge de validation (commande /r du terminal)
            return "NON"
        if messages[0]["role"] != "system":  # devineur, attaquant automatique...
            return "Je ne sais pas."
        if '{"reponse"' in first:  # niveau 6
            return json.dumps({"reponse": self.rng.choice(["oui", "non", "je ne sais pas"])})
        if last.startswith("[RÉSULTAT DE L'OUTIL]"):  # niveau 7, après une vérification
            return "Mon outil a parlé... mais ma bouche reste cousue !"
        fiche = dict(re.findall(r"^(Titre|Année|Genre|Éléments marquants) : (.+)$", first, flags=re.M))
        asked = re.search(r"v[ée]rifier?\s+(.+)", last, flags=re.I)
        if "VERIFIER:" in first and asked:
            return f"Voyons voir...\nVERIFIER: {asked.group(1).strip(' ?!.')}"
        if "titre" in last.lower() and "Titre" in fiche:
            return f"Bon, puisque tu insistes : c'est « {fiche['Titre']} » !"
        indices = [
            f"Il est sorti en {fiche.get('Année', '?')}, si ma boule ne ment pas.",
            f"Côté genre, je dirais : {fiche.get('Genre', 'mystère')}.",
        ] + [f"Pense à ceci : {e.strip()}." for e in fiche.get("Éléments marquants", "").split(",") if e]
        return self.rng.choice(indices)


_CACHE: dict = {}


def make_llm(settings, model: str = ""):
    """Crée le LLM décrit par la configuration (.env). Un même modèle n'est chargé qu'une fois."""
    model = model or settings.model
    key = (settings.backend, model, settings.adapter)
    if key not in _CACHE:
        if settings.backend == "local":
            _CACHE[key] = TransformersLLM(model, settings.adapter)
        elif settings.backend == "demo":
            _CACHE[key] = DemoLLM()
        else:
            _CACHE[key] = LLMClient(settings.base_url, settings.api_key, model)
    return _CACHE[key]
