from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import re
from typing import Dict, List


OEUVRES_DIR = Path("oeuvres")


def _slugify(nom: str) -> str:
    slug = (nom or "").strip().lower()
    slug = slug.replace("-", "_").replace(" ", "_")
    slug = re.sub(r"[^a-z0-9_]", "", slug)
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "oeuvre"


@dataclass
class OeuvreManager:
    nom: str
    slug: str
    base_dir: Path

    @classmethod
    def charger_ou_creer(cls, nom: str) -> "OeuvreManager":
        slug = _slugify(nom)
        base_dir = OEUVRES_DIR / slug

        if base_dir.exists():
            return cls._load_existing(base_dir)

        print("Œuvre introuvable. Créer ? (o/n)")
        if input().strip().lower() != "o":
            raise SystemExit(0)

        resume = input("Résumé de l'histoire (Entrée pour passer) :").strip()

        instance = cls(nom=nom.strip() or slug, slug=slug, base_dir=base_dir)
        instance._create_files(resume)
        return instance

    @classmethod
    def _load_existing(cls, base_dir: Path) -> "OeuvreManager":
        meta_path = base_dir / "meta.json"
        nom = base_dir.name
        slug = base_dir.name

        if meta_path.exists():
            try:
                with meta_path.open("r", encoding="utf-8") as f:
                    meta = json.load(f)
                nom = str(meta.get("nom") or nom)
                slug = str(meta.get("slug") or slug)
            except Exception:
                pass

        instance = cls(nom=nom, slug=slug, base_dir=base_dir)
        instance._ensure_structure()
        return instance

    def _ensure_structure(self):
        self.base_dir.mkdir(parents=True, exist_ok=True)

        if not (self.base_dir / "meta.json").exists():
            self._write_json(
                self.base_dir / "meta.json",
                {
                    "nom": self.nom,
                    "slug": self.slug,
                    "source_lang": "en",
                    "cree_le": datetime.now().isoformat(),
                },
            )

        if not (self.base_dir / "glossaire.json").exists():
            self._write_json(self.base_dir / "glossaire.json", {"termes": []})

        if not (self.base_dir / "historique.json").exists():
            self._write_json(self.base_dir / "historique.json", {"chapitres": {}})

        if not (self.base_dir / "contexte.txt").exists():
            (self.base_dir / "contexte.txt").write_text("", encoding="utf-8")

    def _create_files(self, resume: str):
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self._write_json(
            self.base_dir / "meta.json",
            {
                "nom": self.nom,
                "slug": self.slug,
                "source_lang": "en",
                "cree_le": datetime.now().isoformat(),
            },
        )
        self._write_json(self.base_dir / "glossaire.json", {"termes": []})
        self._write_json(self.base_dir / "historique.json", {"chapitres": {}})
        (self.base_dir / "contexte.txt").write_text(resume, encoding="utf-8")

    def _write_json(self, path: Path, payload: Dict):
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _read_json(self, path: Path, fallback: Dict) -> Dict:
        if not path.exists():
            return fallback
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
        return fallback

    def ajouter_termes_glossaire_interactif(self):
        termes = self.get_termes_glossaire()
        print(f"Termes existants ({len(termes)}): {termes}")

        while True:
            terme = input("Ajouter un terme (Entrée pour terminer) :").strip()
            if not terme:
                break
            termes.append(terme)
            self._write_json(self.base_dir / "glossaire.json", {"termes": termes})

    def chapitre_deja_traduit(self, nom_chap: str) -> bool:
        historique = self._read_json(self.base_dir / "historique.json", {"chapitres": {}})
        chapitres = historique.get("chapitres", {})
        return nom_chap in chapitres

    def marquer_chapitre_traduit(self, nom_chap: str, nb_images: int):
        historique = self._read_json(self.base_dir / "historique.json", {"chapitres": {}})
        chapitres = historique.setdefault("chapitres", {})
        chapitres[nom_chap] = {
            "traduit_le": datetime.now().isoformat(),
            "statut": "done",
            "nb_images": int(nb_images),
        }
        self._write_json(self.base_dir / "historique.json", historique)

    def get_termes_glossaire(self) -> List[str]:
        data = self._read_json(self.base_dir / "glossaire.json", {"termes": []})
        termes = data.get("termes", [])
        if not isinstance(termes, list):
            return []
        return [str(t) for t in termes if str(t).strip()]

    def get_contexte(self) -> str:
        path = self.base_dir / "contexte.txt"
        if not path.exists():
            return ""
        try:
            return path.read_text(encoding="utf-8")
        except Exception:
            return ""
