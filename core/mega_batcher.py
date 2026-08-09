"""
═══════════════════════════════════════════════════════════════════════════════
MEGA-BATCHER — Hack de Quota Gemini (20 RPD / 250K TPM)

Accumule le texte de 5-10 chapitres en une seule requête API.
IDs composites : {chapitre_id}_{index_bulle} (ex: CH01_001, CH02_045)
Un envoi de ~10 chapitres (~1500 textes) ≈ 45K tokens = 1 seul ticket RPD.
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class BubbleEntry:
    """Une bulle individuelle avec son ID composite."""
    composite_id: str          # ex: "CH01_001"
    chapter_id: str            # ex: "CH01"
    local_index: int           # index dans le chapitre (0-based)
    text_original: str         # texte OCR brut
    class_name: str = ""       # bulle, system, sfx, etc.
    text_translated: str = ""  # rempli après traduction
    font_key: str = "STANDARD"


@dataclass
class ChapterPayload:
    """Toutes les bulles d'un chapitre."""
    chapter_id: str
    entries: List[BubbleEntry] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


class MegaBatcher:
    """
    Accumule les chapitres et déclenche un envoi groupé quand le seuil est atteint.

    Usage:
        batcher = MegaBatcher(translate_fn=my_gemini_call, batch_size=8)
        batcher.add_chapter("CH01", texts_ch01, classes_ch01)
        batcher.add_chapter("CH02", texts_ch02, classes_ch02)
        ...
        # Quand batch_size atteint, translate_fn est appelé automatiquement.
        # Ou forcer le flush:
        batcher.flush()
    """

    def __init__(
        self,
        translate_fn: Callable[[str], Optional[Dict]],
        batch_size: int = 8,
        max_tokens_estimate: int = 200_000,
        build_prompt_fn: Optional[Callable] = None,
    ):
        self.translate_fn = translate_fn
        self.batch_size = max(1, min(batch_size, 15))
        self.max_tokens_estimate = max_tokens_estimate
        self.build_prompt_fn = build_prompt_fn

        self._pending: List[ChapterPayload] = []
        self._results: Dict[str, BubbleEntry] = {}  # composite_id → entry traduit

        # Stats
        self.total_api_calls = 0
        self.total_texts_sent = 0
        self.total_chapters_processed = 0

    # ── PUBLIC API ────────────────────────────────────────────────────────

    def add_chapter(
        self,
        chapter_id: str,
        texts: List[str],
        class_names: Optional[List[str]] = None,
    ) -> bool:
        """
        Ajoute un chapitre au buffer.
        Retourne True si un flush automatique a été déclenché.
        """
        if class_names is None:
            class_names = [""] * len(texts)

        payload = ChapterPayload(chapter_id=chapter_id)

        for i, (txt, cls) in enumerate(zip(texts, class_names)):
            composite_id = f"{chapter_id}_{i:03d}"
            entry = BubbleEntry(
                composite_id=composite_id,
                chapter_id=chapter_id,
                local_index=i,
                text_original=txt.strip(),
                class_name=cls,
            )
            payload.entries.append(entry)

        self._pending.append(payload)

        # Auto-flush si seuil atteint
        if len(self._pending) >= self.batch_size:
            self.flush()
            return True

        # Auto-flush si estimation tokens trop haute
        total_chars = sum(
            len(e.text_original) for p in self._pending for e in p.entries
        )
        estimated_tokens = total_chars // 3  # ~3 chars/token en moyenne
        if estimated_tokens > self.max_tokens_estimate * 0.8:
            self.flush()
            return True

        return False

    def flush(self) -> Dict[str, BubbleEntry]:
        """
        Envoie tous les chapitres accumulés en UNE requête.
        Retourne le dict composite_id → BubbleEntry avec traductions.
        """
        if not self._pending:
            return {}

        # Construire la liste numérotée avec IDs composites
        all_entries: List[BubbleEntry] = []
        for payload in self._pending:
            all_entries.extend(payload.entries)

        # Filtrer les vides et les SFX purs
        to_send = [e for e in all_entries if e.text_original.strip()]

        if not to_send:
            self._pending.clear()
            return {}

        # Construire le prompt
        numbered_lines = "\n".join(
            f"{e.composite_id}: {e.text_original}" for e in to_send
        )

        # Appeler la fonction de traduction (1 seul appel API)
        result = self.translate_fn(numbered_lines)
        self.total_api_calls += 1
        self.total_texts_sent += len(to_send)
        self.total_chapters_processed += len(self._pending)

        # Parser les résultats et remplir les entries
        if result and isinstance(result, dict):
            trad_list = result.get("traductions", [])
            trad_map = {}
            font_map = {}
            for item in trad_list:
                if isinstance(item, dict):
                    id_val = str(item.get("id", ""))
                    trad_map[id_val] = item.get("fr", "")
                    font_map[id_val] = item.get("font_key", "STANDARD")

            for entry in to_send:
                fr = trad_map.get(entry.composite_id, "")
                if fr:
                    entry.text_translated = fr
                    entry.font_key = font_map.get(entry.composite_id, "STANDARD")
                else:
                    entry.text_translated = entry.text_original

                self._results[entry.composite_id] = entry

        # Cleanup
        batch_results = {e.composite_id: e for e in all_entries}
        self._pending.clear()

        return batch_results

    def get_chapter_results(self, chapter_id: str) -> List[BubbleEntry]:
        """Récupère les traductions pour un chapitre spécifique."""
        return [
            entry for entry in self._results.values()
            if entry.chapter_id == chapter_id
        ]

    def get_result(self, composite_id: str) -> Optional[BubbleEntry]:
        """Récupère une bulle traduite par son ID composite."""
        return self._results.get(composite_id)

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def pending_texts_count(self) -> int:
        return sum(len(p.entries) for p in self._pending)

    def stats(self) -> Dict[str, Any]:
        return {
            "api_calls": self.total_api_calls,
            "texts_sent": self.total_texts_sent,
            "chapters_processed": self.total_chapters_processed,
            "pending_chapters": self.pending_count,
            "pending_texts": self.pending_texts_count,
            "avg_texts_per_call": (
                self.total_texts_sent / max(1, self.total_api_calls)
            ),
        }
