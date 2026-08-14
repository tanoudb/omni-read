# Guide d'utilisation — Webtoon Translator V5

Tout a été vérifié dans le code (lecture des sources + `--help` réel des scripts) le 2026-08-14.
Ce qui n'a pas pu être vérifié ou qui n'existe pas est marqué explicitement.

Toutes les commandes ci-dessous s'exécutent depuis `A:\omni read`, avec l'environnement
`.venv311` actif (ou en appelant directement `.venv311\Scripts\python.exe`).

---

## 1. Prérequis

### 1.1 Deux environnements Python distincts

Le projet utilise **deux venvs séparés**, pas un seul :

- **`.venv311`** (Python 3.11, à la racine) — c'est l'environnement dans lequel tu lances
  `main.py`, `run_all_series.py`, les scripts `scratch/`, etc. Il contient torch, ultralytics,
  transformers, rapidocr, simple-lama-inpainting, wordsegment...
- **`.venv_paddleocr`** — un venv **séparé**, dédié à PaddleOCR (PaddlePaddle GPU + PP-OCRv5
  server). Tu ne le lances jamais toi-même : `core/backends/paddleocr_vl_v15.py` démarre
  automatiquement un **worker subprocess persistant** (`_paddle_vl_worker.py`) via
  `.venv_paddleocr/Scripts/python.exe`, et communique avec lui en JSON sur stdin/stdout. Ce
  worker est le moteur OCR **primaire** ; RapidOCR (in-process, dans `.venv311`) sert de
  fallback si le subprocess échoue.
  Si `.venv_paddleocr` n'existe pas ou est mal installé, le code lève explicitement :
  `RuntimeError("PaddleOCR: .venv_paddleocr introuvable. Vérifiez que .venv_paddleocr existe à la racine du projet.")`

Il n'existe pas de `requirements.txt` dédié à `.venv_paddleocr` dans le dépôt. Les seules
versions connues (`paddleocr==3.4.0`, `paddlepaddle==3.1.1`) apparaissent dans
`requirements_from_user.txt` (un dump d'environnement, pas un fichier d'install officiel) —
à vérifier/adapter si tu dois reconstruire ce venv.

### 1.2 `wordsegment` — piège silencieux

`core/ocr.py` utilise le paquet `wordsegment` pour recoller les mots que l'OCR rend collés
(polices BD serrées : `YOUCOMEBACK` → `YOU COME BACK`). **Ce paquet doit être installé dans
l'interpréteur qui exécute réellement la pipeline** (`.venv311`, ou tout autre venv depuis
lequel tu lances `main.py`). S'il est absent :

- avant le 2026-08-14 : l'échec était avalé silencieusement par un `except Exception` large
  → le découpeur de mots collés tournait à vide, sans aucun log.
- maintenant : `core/ocr.py::_wordsegment()` logge explicitement un avertissement :
  `⚠️ wordsegment indisponible (...) — le découpage des mots collés OCR (...) est DÉSACTIVÉ.`

Vérifie-le avant toute session de travail sur une nouvelle série ou un nouveau venv :

```bash
python -c "import wordsegment; wordsegment.load(); print('OK')"
```

S'il manque :

```bash
pip install wordsegment==1.3.1
```

### 1.3 Modèles attendus

- **YOLO** : `assets/models/manhwa_v4.pt` (chemin par défaut, contrôlé par
  `WEBTOON_YOLO_MODEL`, défaut `"manhwa_v4.pt"`). Présent sur cette machine, avec aussi
  `manhwa_v1.pt`/`v2.pt`/`v3.pt` dans le même dossier. Le mode ensemble v3+v4 est
  **désactivé par défaut** (voir `config/settings.py`, commentaire : détections en double sur
  une même bulle avec des classes différentes) — pour le réactiver, définir
  `WEBTOON_YOLO_MODEL_SECONDARY=manhwa_v3.pt`.
  Si le modèle est absent, `main.py` s'arrête avec :
  `Modèle YOLO introuvable: <chemin>` suivi de `Placez le modèle manhwa_v2.pt dans assets/models/`.
  **Attention** : ce message d'aide mentionne `manhwa_v2.pt`, alors que le modèle réellement
  attendu par défaut est `manhwa_v4.pt` — c'est un message d'erreur obsolète dans le code, pas
  une consigne à suivre.
- **LaMa (inpainting)** : `simple_lama_inpainting.SimpleLama` télécharge et met en cache
  automatiquement `big-lama.pt`. Sur cette machine il est déjà présent dans
  `C:\Users\<user>\.cache\torch\hub\checkpoints\big-lama.pt`. Rien à faire manuellement sauf
  si ce cache est absent (premier lancement = téléchargement automatique).
- **Qwen local** (traduction sans API) : `assets/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf`,
  déjà présent.
- **NLLB** : `assets/models/nllb-200-3.3b-ct2` (CTranslate2) et
  `assets/models/models--facebook--nllb-200-distilled-600M`, déjà présents.

---

## 2. Clé API Gemini

Vérifié dans `main.py` et `core/translator_gemini.py` :

- La clé est lue depuis les variables d'environnement `GEMINI_API_KEY` (ou en repli
  `GOOGLE_API_KEY`).
- `main.py` fait `from dotenv import load_dotenv; load_dotenv()` en tout début de fichier :
  donc un fichier **`.env` à la racine** (`A:\omni read\.env`) est chargé automatiquement,
  pas besoin d'exporter la variable manuellement dans le shell.
- Un `.env.example` existe à la racine comme modèle. `.env` est listé dans `.gitignore`
  (`.env`, `.env.*`, avec exception explicite pour `!.env.example`) — il n'est donc pas
  versionné.

Pour configurer :

```bash
cp .env.example .env
```

Puis éditer `.env` et renseigner `GEMINI_API_KEY=<ta_clé>` (récupérable sur
`https://aistudio.google.com/apikey`). **Ne mets jamais la clé en dur dans un fichier
versionné.**

**Point important vérifié dans le code, qui contredit le `README.md` actuel** : le README dit
qu'avoir `GEMINI_API_KEY` dans `.env` fait basculer automatiquement `main.py` sur Gemini. Ce
n'est plus vrai dans le `main.py` actuel : la variable `has_gemini_key` (ligne ~207) est bien
calculée à partir de l'environnement, mais **elle n'est utilisée nulle part ensuite** — c'est
du code mort. La sélection réelle du mode de traduction (lignes ~209-217) est :

- `--api` fourni → mode `gemini`
- sinon `--translation-mode <mode>` fourni → ce mode
- sinon → **NLLB par défaut, coût API zéro**, quelle que soit la présence d'une clé Gemini.

Donc : **pour utiliser Gemini, il faut explicitement passer `--api`** (voir section 3), même
si la clé est bien présente dans `.env`.

---

## 3. Mode normal / production

### 3.1 Une image unique

```bash
python main.py --image "input/mon_image.png" --output "output" --debug
```

`--debug` sauvegarde l'image annotée des détections + les crops dans `output/debug/`.

### 3.2 Un dossier d'images (chapitre non structuré en séries)

```bash
python main.py --input "input/mon_dossier" --output "output"
```

Sans `--input`, `main.py` traite `input/` → `output/` par défaut.

### 3.3 Traduction via l'API Gemini (quasi gratuite)

```bash
python main.py --input "input/mon_dossier" --api
```

Nécessite `GEMINI_API_KEY` dans `.env` (voir section 2). Sans `--api`, le mode par défaut est
NLLB local (coût zéro, plus lent).

### 3.4 Choisir explicitement un mode de traduction local

```bash
python main.py --input "input/mon_dossier" --translation-mode qwen
```

Valeurs valides du parser (`--help` vérifié) : `hybrid`, `hybrid_quality`, `nllb`, `qwen`.

### 3.5 Une série complète (mode « Série »)

```bash
python main.py --series nom-de-la-serie
```

(alias français équivalent : `--serie nom-de-la-serie`)

Comportement vérifié dans `main.py` :

- Cherche les chapitres dans `manhwa/<slug>/<Chapitre XXX>/*.png|jpg|jpeg`
  (slug = nom en minuscules, espaces remplacés par `_`, avec recherche floue si le dossier
  exact n'existe pas).
- Écrit le résultat dans `manhwa_trad/<slug>/`.
- **Demande une confirmation interactive** (`Lancer la traduction pour N chapitres et M
  images ? (o/n)`) avant de démarrer — donc pas utilisable tel quel dans un script non
  interactif sans répondre au prompt.
- Charge/actualise le glossaire de série (`data/series/<slug>/`, via `utils/series_db.py`) et,
  en fin de traitement, exécute une vérification de cohérence (noms, phrases récurrentes).

Pour un chapitre précis dans ce mode (contexte narratif du glossaire) :

```bash
python main.py --series nom-de-la-serie --chapter 3
```

Pour initialiser une nouvelle série (crée les métadonnées dans `data/series/<slug>/`) :

```bash
python main.py --series nom-de-la-serie --init-series "Nom complet de la série"
```

### 3.6 Toutes les séries d'un coup, sans API (coût zéro)

```bash
python run_all_series.py
```

Vérifié via `--help` et lecture du code :

- Scanne `manhwa/<serie>/<Chapitre XXX>/` et détecte automatiquement les séries et chapitres.
- **Force toujours le mode Qwen local** (`config.translation.translation_mode = "qwen"` codé
  en dur dans `run_series()`) — il n'y a **pas d'option pour passer par Gemini** dans ce
  script, contrairement à `main.py --api`.
- Saute les chapitres déjà traduits (détecte un fichier `*_translated.*` dans le dossier de
  sortie `manhwa_trad/<slug>/<chapitre>/`).
- Demande confirmation interactive avant de lancer (sauf `--dry-run` ou `--list`).
- Écrit un rapport JSON en fin d'exécution : `manhwa_trad/run_all_series_report.json`.

Options vérifiées :

```bash
python run_all_series.py --dry-run
```

```bash
python run_all_series.py --serie nom-de-la-serie
```

```bash
python run_all_series.py --force
```

```bash
python run_all_series.py --list
```

### 3.7 Voir la configuration active sans rien lancer

```bash
python main.py --show-config
```

### 3.8 Scripts qu'il vaut mieux ignorer / ne pas utiliser tels quels

- `run_dry_all.py` : malgré son nom, ce n'est **pas** un harnais dry-run générique — c'est un
  script jetable qui boucle sur une liste de séries codées en dur et relance
  `test_corrections.py` (lui-même codé pour une seule série/chapitre en dur,
  `SERIE = "hellogin"`, `CHAPITRE = "Chapitre 001"`) via `subprocess`, en réécrivant le fichier
  à chaque itération. À ne pas lancer sans le relire d'abord.
- `run_dummy_test.py` : lance `main.py` sur un chemin d'entrée codé en dur
  (`manhwa/i-married-the-dragon-i-killed/Chapitre 001`) avec un patch qui neutralise NLLB
  (retourne le texte source tel quel). Utile comme exemple, pas comme outil générique.
- `utils/dry_run_all.py`, `utils/dry_run_fast.py` : listes de séries codées en dur
  (`utils/dry_run_fast.py` ne traite que `path-of-vengeance`), pas d'arguments CLI.
- `utils/dry_run_first_chapters.py` : celui-ci a un vrai CLI (`--serie`, `--bubble-debug`) et
  traite le premier chapitre de chaque série d'une liste codée en dur
  (`TARGET_SERIES` dans le fichier). Réinjecte le texte OCR source (pas de traduction), écrit
  un rapport JSON détaillé (`ocr_skip_reasons`, `render_errors`, `ghost_risk_bboxes`) dans
  `tests/dry_run_out/`. Exemple :
  ```bash
  python utils/dry_run_first_chapters.py --serie hellogin --bubble-debug
  ```
  Le harnais **générique et destiné à l'itération manuelle** reste celui de `scratch/`
  (section 4 ci-dessous).

---

## 4. Mode dry-run / sans traduction (`scratch/`)

Ce harnais réinjecte le texte OCR source à la place d'une traduction : il isole
détection + OCR + effacement + rendu, sans dépendre de la qualité de traduction. Chaque script
a été vérifié avec `--help` réel (sauf mention contraire).

### 4.1 Pipeline complète sur une image, avant/après par bulle

```bash
python scratch/render_iterate.py "manhwa/hellogin/Chapitre 001/page01.png" "scratch/out/hellogin_ch1_p01" --margin 50
```

Signature exacte (positionnels obligatoires, `--margin` optionnel, défaut `50`) :
`render_iterate.py [-h] [--margin MARGIN] image out_dir`

Écrit dans `out_dir` :
- `page_before.png`, `page_erased.png` (effacement seul, **avant** réinjection — c'est le seul
  endroit où juger l'effacement sans que le texte réinjecté le recouvre), `page_after.png`
- `bubbles/NN_before.png`, `NN_erased.png`, `NN_after.png` (crops par bulle avec marge)
- `bubbles_meta.json` (bbox, classe, texte OCR, score, `ghost_risk` par bulle)

### 4.2 Planches-contact avant/après pour revue visuelle rapide

```bash
python scratch/build_contact_sheet.py "scratch/out/hellogin_ch1_p01"
```

**Attention** : ce script n'a **pas** de parsing `argparse` — il lit `sys.argv[1]` directement
comme `run_dir` et ne supporte donc pas `--help` (un `--help` littéral serait interprété comme
un chemin de dossier et provoquerait une erreur `FileNotFoundError`, vérifié). Il attend que
`render_iterate.py` ait déjà tourné sur ce `run_dir` (a besoin de `bubbles_meta.json` et du
dossier `bubbles/`). Écrit les planches dans `<run_dir>/contact_sheets/sheet_NN.png`.

### 4.3 Tuiles pleine page avant | après

```bash
python scratch/side_by_side.py "scratch/out/hellogin_ch1_p01" --tile 2000 --scale 1.0
```

Signature vérifiée : `side_by_side.py [-h] [--tile TILE] [--scale SCALE] run_dir`
(défauts : `--tile 2000`, `--scale 1.0`). Nécessite `page_before.png` et `page_after.png` déjà
générés par `render_iterate.py` dans ce `run_dir`. Écrit dans `<run_dir>/side_by_side/`.

### 4.4 Choisir une tranche représentative d'un strip très haut

```bash
python scratch/pick_slice.py "input/strip_tres_haut.jpg" "scratch/out/slice.png" --height 6000
```

Signature vérifiée : `pick_slice.py [-h] [--height HEIGHT] strip out` (défaut `--height 6000`).
Fait tourner YOLO pour choisir la fenêtre la plus dense en bulles avec la plus grande variété
de tailles, puis écrit uniquement cette tranche.

### 4.5 Banc d'essai « effacement seul » (itération rapide sans repasser par YOLO/OCR/SAM)

```bash
python scratch/erase_lab.py --build "manhwa/hellogin/Chapitre 001/page01.png" "scratch/cache/hellogin_p01"
```

```bash
python scratch/erase_lab.py "scratch/cache/hellogin_p01" "scratch/out/hellogin_p01_erase"
```

Signature vérifiée : `erase_lab.py [-h] [--build] a b`. En phase `--build`, fait tourner
détection + OCR + segmentation une fois et met le résultat en cache sur disque
(`page.png`, `dets.pkl`). Sans `--build`, recharge ce cache et ne rejoue que l'effacement — ce
qui permet d'itérer sur l'algo d'inpainting en quelques secondes. Écrit
`<out_dir>/page_erased.png`, des crops avant/après par zone dans `<out_dir>/crops/`, et
`meta.json`.

---

## 5. Mode Google Drive

**Ce mode n'existe pas dans le dépôt.** Recherche effectuée (grep sur tout `*.py` du dépôt,
hors venvs) pour `rclone`, `drive.google`, `gdrive`, `GoogleDrive`, `pydrive`,
`google-api-python-client` : aucune correspondance dans `core/`, `utils/`, `scripts/`, ni à la
racine. Le seul faux positif trouvé (mot « google » dans
`webtoon-translator-native/src-frontend/src/styles.css`) est sans rapport (référence à une
police Google Fonts, pas à Drive).

Il n'y a **aucun mécanisme de téléchargement/synchronisation automatique** des chapitres depuis
Drive ou un service cloud. L'arrivée des images sur le disque est **entièrement manuelle** :
tu dois toi-même déposer les images dans `manhwa/<slug>/<Chapitre XXX>/` (ou dans un dossier
passé à `--input`). De même, les résultats ne repartent nulle part automatiquement : ils
restent dans `manhwa_trad/<slug>/` (mode Série) ou dans le dossier passé à `--output`.

À ne pas confondre avec (vérifié dans le code, sans rapport avec Drive) :
- `data/series/<slug>/` : métadonnées de la série pour la cohérence de traduction (glossaire
  `glossary.json`, personnages `series.json`, résumés par chapitre `chapters/chNNN.json`),
  géré par `utils/series_db.py`. Aucune image n'y est stockée.
- `data/gemini_state/<slug>/` et `cache/gemini_cache.json` : état/cache internes à la
  traduction via l'API Gemini (`core/translator_gemini.py`), pas un mécanisme de stockage de
  chapitres.
- `gemini_watcher.py` : un script autonome et **déconnecté du pipeline principal**
  (`main.py`/`pipeline.py`) qui surveille un dossier local (`ocr_input/` par défaut) pour des
  fichiers JSON déjà passés par l'OCR, et les traduit via Gemini. Ce n'est pas un
  téléchargeur/synchroniseur Drive — il attend des fichiers déjà présents localement, écrits
  par un autre outil.

---

## 6. Configuration — variables d'environnement `WEBTOON_*`

Liste établie par `grep os.environ` dans `config/settings.py` (vérifiée exhaustivement pour ce
fichier). Seules celles avec un impact direct et visible sur le résultat sont détaillées ;
les autres (seuils fins de binarisation OCR, etc.) sont listées en bref.

### Détection (YOLO)

| Variable | Défaut | Effet visible |
|---|---|---|
| `WEBTOON_YOLO_MODEL` | `manhwa_v4.pt` | Modèle de détection principal (cherché dans `assets/models/`) |
| `WEBTOON_YOLO_MODEL_SECONDARY` | *(vide → désactivé)* | Active un 2e modèle en ensemble (désactivé par défaut, cause historique de doublons de bulles) |
| `WEBTOON_MAX_HEIGHT` | `0` (désactivé) | Si >0, redimensionne l'image avant détection pour tenir sous cette hauteur |
| `WEBTOON_INTER_CLASS_IOU` | `0.5` | Seuil de dédoublonnage entre classes différentes sur la même zone (évite le rendu en double d'une bulle) |
| `WEBTOON_ENSEMBLE_DEDUPE_IOU` | `0.30` | Seuil de dédoublonnage après fusion multi-modèles |
| `WEBTOON_BLACK_PADDING_RATIO` | `0.03` | Padding noir ajouté avant détection |
| `WEBTOON_USE_BLACK_PADDING` / `WEBTOON_BLACK_BARS_ENABLED` | `true` | Active/désactive ce padding |

### OCR

| Variable | Défaut | Effet visible |
|---|---|---|
| `WEBTOON_OCR_BACKEND` / `WEBTOON_OCR_PRIMARY` | `paddleocr-vl-v1.5` | Backend OCR principal (voir §1.1 — en réalité PP-OCRv5 server via worker `.venv_paddleocr`, pas le VL doc-parser) |
| `WEBTOON_OCR_FALLBACKS` | `rapidocr-ppocrv5` | Backend(s) de repli si le primaire échoue |
| `WEBTOON_OCR_FALLBACK_MIN_CONF` | `0.72` | Confiance mini avant repli |
| `WEBTOON_USE_VL15` | `true` | (voir doc interne — flag legacy VL1.5) |
| `WEBTOON_VL15_MIN_FREE_VRAM_GB` | `4.0` | VRAM libre mini exigée |
| `WEBTOON_OCR_MULTIPASS_ENABLED` | `false` | Preprocessing multi-passes des crops |

### Segmentation

| Variable | Défaut | Effet visible |
|---|---|---|
| `WEBTOON_ENABLE_PRECISE_MASKS` | `true` | Masques précis (pixel) vs bbox brute pour l'effacement |
| `WEBTOON_SEGMENTATION_BACKEND` | auto (`hybrid` si `sam2_b.pt` présent) | `hybrid` \| `sam2` \| `ocr_regions` |
| `WEBTOON_SEGMENTATION_DILATE` | `9` | Dilatation du masque (marge d'effacement) |

### Traduction

| Variable | Défaut | Effet visible |
|---|---|---|
| `WEBTOON_TRANSLATION_BACKEND` | `local_llm` | Backend de traduction (écrasé par `main.py` selon `--api`/`--translation-mode`, voir §2) |
| `WEBTOON_TRANSLATION_MODE` | `hybrid` | idem — mais `main.py` force `nllb` par défaut si ni `--api` ni `--translation-mode` |
| `WEBTOON_LLM_MODEL` | `assets/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf` | Modèle LLM local |
| `WEBTOON_LLM_REQUIRE_CUDA` | `true` | Si `true` et pas de GPU CUDA visible → erreur fatale plutôt que repli CPU silencieux (voir §7) |
| `WEBTOON_USE_BITSANDBYTES` / `WEBTOON_BNB_4BIT` / `WEBTOON_BNB_8BIT` | `false` / `true` / `false` | Quantization du LLM local |
| `WEBTOON_CONTEXT_DISTANCE_THRESHOLD` / `WEBTOON_MAX_GROUP_SIZE` | `420` / `7` | Regroupement de bulles pour traduction avec contexte |

### Rendu (impact visuel direct)

| Variable | Défaut | Effet visible |
|---|---|---|
| `WEBTOON_MIN_FONT_SIZE` / `WEBTOON_MAX_FONT_SIZE` | `11` / `80` | Bornes de la taille de police auto-ajustée |
| `WEBTOON_FOLLOW_TEXT_ANGLE` | `true` | Le texte réinjecté suit l'inclinaison du texte source (pancartes, papiers de travers) |
| `WEBTOON_PRESERVE_TEXT_COLOR` | `true` | Réutilise la couleur de texte détectée dans l'image source |
| `WEBTOON_AUTO_STYLE_TYPESETTING` | `true` | Style typographique auto (gras/contour selon contexte) |
| `WEBTOON_LOCK_TEXT_TO_OCR_REGIONS` / `WEBTOON_LOCK_TEXT_SYSTEM_ONLY` | `true` / `true` | Verrouille le texte rendu dans les régions OCR détectées |
| `WEBTOON_INPAINT_MASK_DILATE` / `WEBTOON_BUBBLE_INPAINT_MASK_DILATE` / `WEBTOON_OUT_TEXT_DILATE` | `7` / `7` (plafonné à 9) / `11` | Marge de dilatation du masque avant inpainting — trop bas laisse des résidus de texte, trop haut mange le dessin autour |
| `WEBTOON_DIFFUSION_FALLBACK` | `false` | Repli Navier-Stokes si l'inpainting laisse un fantôme (désactivé — cause désormais des bandes grises, voir commentaire dans le code) |
| `WEBTOON_INPAINTING_MODEL_ID` | `dreMaz/AnimeMangaInpainting` | 2e modèle d'inpainting (fonds d'artwork complexes) |
| `WEBTOON_OUTPUT_FORMAT` / `WEBTOON_OUTPUT_QUALITY` | `png` / `95` | Format de sortie (`png`\|`jpg`\|`webp`) |

### Divers

- `WEBTOON_PRIMARY_FONT_PATH` : chemin de police forcé (sinon découverte auto via
  `_discover_font_paths`).
- `WEBTOON_FONT_PATHS` : liste de chemins de polices additionnels.
- `WEBTOON_NLLB_*`, `WEBTOON_RAPIDOCR_*`, `WEBTOON_OCR_ADAPTIVE_*` : réglages fins,
  généralement pas à toucher en usage courant.

---

## 7. Sorties

Vérifié via `config/settings.py` et les scripts :

- **Images traduites (usage direct, `--image`/`--input`)** : dossier passé à `--output`
  (défaut `output/` à la racine, `config.OUTPUT_DIR`).
- **Mode Série (`--series`)** : `manhwa_trad/<slug>/` (et par chapitre :
  `manhwa_trad/<slug>/<Chapitre XXX>/`).
- **`run_all_series.py`** : même arborescence `manhwa_trad/<slug>/<chapitre>/`, plus un
  rapport global `manhwa_trad/run_all_series_report.json`.
- **Mode debug (`--debug`)** : `<output>/debug/` (image annotée des détections + crops).
- **Logs** : `logs/webtoon_v5.log` (run standard `main.py`) ou `logs/series_<slug>_<ts>.log`
  (par série dans `run_all_series.py`), dossier contrôlé par `config.LOGS_DIR` (`logs/` à la
  racine).
- **Caches** :
  - `assets/cache/ocr_weights/` (poids OCR/segmentation, ex. `sam2_b.pt`)
  - `assets/cache/translation_models/` et fichiers `translation_cache*.json` (cache de
    traduction, activé par défaut)
  - `assets/cache/inpainting_models/` (2e modèle d'inpainting)
  - `cache/gemini_cache.json` (cache des traductions Gemini)
  - `C:\Users\<user>\.cache\torch\hub\checkpoints\` (LaMa, etc. — cache torch standard, hors
    du dépôt)
- **État Série / Gemini** : `data/series/<slug>/` (glossaire, personnages, résumés par
  chapitre) et `data/gemini_state/<slug>/` (état narratif auto-évolutif pour Gemini).
- **Sorties du harnais `scratch/`** : dans le `out_dir`/`run_dir` que tu passes en argument
  (rien n'est écrit ailleurs par défaut) — voir section 4.

---

## 8. Dépannage

Erreurs de démarrage identifiées directement dans le code :

- **`Modèle YOLO introuvable: <chemin>`** (dans `main.py`, avant tout traitement) : le
  fichier pointé par `WEBTOON_YOLO_MODEL` (défaut `manhwa_v4.pt`) n'est pas dans
  `assets/models/`. Le message d'aide affiché (« Placez le modèle manhwa_v2.pt... ») est
  obsolète — c'est bien `manhwa_v4.pt` qu'il faut, pas `manhwa_v2.pt`.
- **`PaddleOCR: .venv_paddleocr introuvable. Vérifiez que .venv_paddleocr existe à la racine
  du projet.`** (`core/backends/paddleocr_vl_v15.py`) : le venv `.venv_paddleocr` est absent
  ou son `python.exe`/`python` introuvable aux emplacements attendus
  (`.venv_paddleocr/Scripts/python.exe` sous Windows). L'OCR primaire ne peut pas démarrer —
  vérifie que ce venv a bien été créé et contient `paddleocr` + `paddlepaddle` (GPU).
- **`CUDA demandé pour le LLM mais torch ne voit pas de GPU (build CPU ou drivers absents).`**
  (`core/translator.py` / `core/translation/translator.py`) : levée quand
  `WEBTOON_LLM_REQUIRE_CUDA=true` (défaut) et qu'aucun GPU CUDA n'est visible pour le LLM
  local (mode `qwen`/`hybrid`). Deux options : installer/réparer les drivers/CUDA, ou mettre
  `WEBTOON_LLM_REQUIRE_CUDA=false` pour accepter un repli CPU (nettement plus lent).
  Note : le mode NLLB (`translator_nllb.py`) fait un repli CPU **silencieux** (juste un print
  `[NLLB] CUDA indisponible: fallback HF sur CPU`), sans lever d'erreur.
- **Découpeur de mots collés inactif sans erreur visible** : voir §1.2 — symptôme = mots OCR
  du type `YOUCOMEBACK` jamais reséparés dans le texte rendu. Vérifier que `wordsegment` est
  installé dans l'interpréteur utilisé (`pip show wordsegment` dans ce venv précis, pas juste
  présent dans `requirements.txt`).
- **Dossier `manhwa/<slug>` introuvable** en mode `--series` : `main.py` s'arrête avec
  `Dossier manhwa/<slug> (ou similaire) introuvable.` — vérifie l'orthographe du slug (espaces
  → `_`) ou dépose les images dans `manhwa/<slug>/<Chapitre XXX>/`.

---

## 9. Ce qui n'existe pas

- **Mode Google Drive / synchronisation cloud** : aucun code, aucune dépendance (`rclone`,
  API Google Drive) trouvée dans le dépôt. Tout est manuel (voir section 5).
- **Bascule automatique vers Gemini sur simple présence de `GEMINI_API_KEY`** : le code qui
  ferait ça (`has_gemini_key` dans `main.py`) est mort/inutilisé ; il faut passer `--api`
  explicitement (voir section 2).
- **`run_dry_all.py` en tant que harnais dry-run générique** : malgré son nom, ce n'est pas
  un outil réutilisable tel quel (voir §3.8).
