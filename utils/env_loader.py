# -*- coding: utf-8 -*-
"""Chargement du .env tolérant à l'encodage.

PowerShell écrit ses fichiers en UTF-16LE avec BOM quand on fait
`echo ... > .env`. `python-dotenv` ne lit que l'UTF-8 : il échoue sur un
`UnicodeDecodeError` et abandonne le fichier SANS RIEN DIRE. La clé API est
alors invisible, le traducteur repart en silence sur le texte source, et on
obtient des planches à moitié en anglais sans le moindre message d'erreur.

Diagnostiqué le 2026-08-18 après avoir cru successivement à une limite de
payload, à un filtre SFX, puis à une clé absente.
"""

from pathlib import Path
from typing import Optional


def load_env(path: Optional[Path] = None) -> bool:
    """Charge le .env en essayant plusieurs encodages. True si quelque chose
    a été chargé."""
    try:
        from dotenv import load_dotenv
    except Exception:
        return False

    target = Path(path) if path else Path(".env")
    if not target.exists():
        return False

    for enc in ("utf-8", "utf-16", "utf-16-le", "latin-1"):
        try:
            if load_dotenv(target, encoding=enc, override=False):
                return True
        except (UnicodeDecodeError, UnicodeError):
            continue
        except Exception:
            break
    print(
        f"⚠️ {target} illisible dans tous les encodages testés — "
        f"les variables qu'il contient ne seront PAS chargées."
    )
    return False
