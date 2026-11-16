"""Configuration du projet : chemins et réglages du LLM.

Les réglages se lisent dans les variables d'environnement, ou dans un fichier `.env`
placé à la racine du projet (voir `.env.example`). Aucun paquet externe n'est nécessaire.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Racine du projet = dossier qui contient app.py, data/, levels/...
ROOT = Path(__file__).resolve().parent.parent
FILMS_PATH = ROOT / "data" / "films.json"
LEVELS_DIR = ROOT / "levels"
JOURNAL_PATH = ROOT / "data" / "journal.db"


def load_dotenv(path: Path = ROOT / ".env") -> None:
    """Charge les lignes `CLE=valeur` du fichier .env dans os.environ.

    Une variable déjà définie dans l'environnement n'est pas écrasée.
    """
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass
class Settings:
    base_url: str   # adresse de l'API compatible OpenAI (Ollama par défaut)
    api_key: str    # clé d'API ("ollama" suffit pour un modèle local)
    model: str      # modèle du gardien
    aux_model: str  # modèle des défenses (détecteur, censeur) : peut être plus petit
    backend: str = "api"       # "api" (serveur compatible OpenAI) ou "local" (transformers, sur GPU)
    adapter: str = ""          # backend local : dossier de l'adaptateur LoRA entraîné (optionnel)
    teacher_model: str = ""    # modèle « professeur » pour fabriquer le jeu de données d'entraînement


def get_settings() -> Settings:
    load_dotenv()
    model = os.environ.get("ALEXINATOR_MODEL", "llama3.1:8b")
    return Settings(
        base_url=os.environ.get("ALEXINATOR_BASE_URL", "http://localhost:11434/v1"),
        api_key=os.environ.get("ALEXINATOR_API_KEY", "ollama"),
        model=model,
        aux_model=os.environ.get("ALEXINATOR_AUX_MODEL", model),
        backend=os.environ.get("ALEXINATOR_BACKEND", "api"),
        adapter=os.environ.get("ALEXINATOR_ADAPTER", ""),
        teacher_model=os.environ.get("ALEXINATOR_TEACHER_MODEL", model),
    )
