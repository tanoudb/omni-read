# Webtoon Translator V5 — Spécification MVP UI Éditeur Visuel (Tauri)

Version: 1.0  
Date: 2026-02-17  
Statut: Prêt à implémenter

## 1) Scope MVP (verrouillé)

### 1.1 Objectif produit
Transformer l’UI Tauri actuelle en éditeur visuel de traduction manga/webtoon (style comic-translate), en conservant le pipeline backend existant et en ajoutant des APIs d’édition interactive.

### 1.2 Contraintes validées
- Frontend: React + TypeScript.
- UI: CSS custom (pas de UI kit au MVP).
- Plateforme MVP: Windows uniquement.
- Offline strict.
- Tauri: une seule fenêtre.
- Langue UI: FR uniquement.
- Thème: dark unique.
- Persistance settings: locale machine.
- Intégration backend: HTTP FastAPI (pas de commandes Tauri custom).
- Architecture frontend: modulaire par features.

### 1.3 UX MVP obligatoire
- Layout 3 colonnes: liste images | canvas édition | settings.
- Mode Auto (pipeline complet) + Mode Manuel (étapes séparées).
- Édition bboxes: draw, move, resize, delete.
- Override texte source/traduit par bulle.
- Debug mapping LLM + remap manuel.
- Rerender rapide.
- Save/load projet.

### 1.4 Hors scope MVP (explicite)
- Sélection multiple.
- Rotation bbox.
- Snap/grille/guides.
- Tiling / virtualisation très grandes images (limite entrée < 5000px hauteur).
- WebSocket/SSE.
- Multi-job concurrent (1 job max).
- Stockage artefacts lourds dans le projet.

---

## 2) Modèle de données projet

## 2.1 Arborescence projet

```text
<project-root>/
  project.json
  assets/
    originals/
      page_001.png
      page_002.png
    previews/
      page_001_translated.png
```

Notes:
- `project.json` contient l’état éditable complet.
- Les artefacts lourds (masques raster intermédiaires, debug images, réponses brutes volumineuses) ne sont pas persistés comme fichiers séparés dans le projet MVP.
- Les previews peuvent être régénérées au load.

## 2.2 Principes d’identité
- Chaque bulle a un `id` UUID stable (source de vérité).
- Les indices LLM (`llm_input_index`, `llm_output_index`) sont des métadonnées de mapping, jamais des identités.

## 2.3 Schéma JSON (Draft 2020-12)

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://webtoon-translator.local/schemas/project.v1.json",
  "title": "Webtoon Translator Visual Editor Project",
  "type": "object",
  "required": [
    "schema_version",
    "project_id",
    "name",
    "created_at",
    "updated_at",
    "settings",
    "pages"
  ],
  "properties": {
    "schema_version": { "type": "string", "const": "1.0.0" },
    "project_id": { "type": "string", "format": "uuid" },
    "name": { "type": "string", "minLength": 1 },
    "created_at": { "type": "string", "format": "date-time" },
    "updated_at": { "type": "string", "format": "date-time" },
    "settings": {
      "type": "object",
      "required": ["source_lang", "target_lang", "cache_enabled"],
      "properties": {
        "source_lang": { "type": "string", "default": "auto" },
        "target_lang": { "type": "string", "default": "fr" },
        "cache_enabled": { "type": "boolean", "default": true },
        "render": {
          "type": "object",
          "properties": {
            "skip_inpainting_on_text_only": { "type": "boolean", "default": true }
          },
          "additionalProperties": true
        }
      },
      "additionalProperties": true
    },
    "pages": {
      "type": "array",
      "minItems": 1,
      "items": {
        "type": "object",
        "required": [
          "id",
          "index",
          "image_path",
          "width",
          "height",
          "viewport",
          "bubbles"
        ],
        "properties": {
          "id": { "type": "string", "format": "uuid" },
          "index": { "type": "integer", "minimum": 0 },
          "image_path": { "type": "string" },
          "preview_path": { "type": ["string", "null"] },
          "width": { "type": "integer", "minimum": 1 },
          "height": { "type": "integer", "minimum": 1 },
          "viewport": {
            "type": "object",
            "required": ["zoom", "pan_x", "pan_y", "show_translated"],
            "properties": {
              "zoom": { "type": "number", "minimum": 0.05, "maximum": 20 },
              "pan_x": { "type": "number" },
              "pan_y": { "type": "number" },
              "show_translated": { "type": "boolean" }
            }
          },
          "bubbles": {
            "type": "array",
            "items": { "$ref": "#/$defs/Bubble" }
          },
          "llm_debug": { "$ref": "#/$defs/LlmDebug" }
        },
        "additionalProperties": false
      }
    }
  },
  "$defs": {
    "BBox": {
      "type": "object",
      "required": ["x", "y", "w", "h"],
      "properties": {
        "x": { "type": "number", "minimum": 0 },
        "y": { "type": "number", "minimum": 0 },
        "w": { "type": "number", "exclusiveMinimum": 0 },
        "h": { "type": "number", "exclusiveMinimum": 0 }
      },
      "additionalProperties": false
    },
    "MaskStroke": {
      "type": "object",
      "required": ["id", "size", "points"],
      "properties": {
        "id": { "type": "string", "format": "uuid" },
        "size": { "type": "number", "minimum": 1, "maximum": 256 },
        "points": {
          "type": "array",
          "minItems": 1,
          "items": {
            "type": "object",
            "required": ["x", "y"],
            "properties": {
              "x": { "type": "number" },
              "y": { "type": "number" }
            },
            "additionalProperties": false
          }
        }
      },
      "additionalProperties": false
    },
    "TextStyle": {
      "type": "object",
      "required": ["font_family", "font_size", "align", "color"],
      "properties": {
        "font_family": { "type": "string" },
        "font_size": { "type": "number", "minimum": 8, "maximum": 180 },
        "align": { "type": "string", "enum": ["left", "center", "right"] },
        "color": { "type": "string", "pattern": "^#([0-9a-fA-F]{6})$" }
      },
      "additionalProperties": false
    },
    "Bubble": {
      "type": "object",
      "required": [
        "id",
        "bbox",
        "class",
        "source_text",
        "translated_text",
        "llm_input_index",
        "llm_output_index",
        "text_style",
        "mask_strokes",
        "errors"
      ],
      "properties": {
        "id": { "type": "string", "format": "uuid" },
        "bbox": { "$ref": "#/$defs/BBox" },
        "class": { "type": "string" },
        "source_text": { "type": "string" },
        "translated_text": { "type": "string" },
        "source_override": { "type": ["string", "null"] },
        "translated_override": { "type": ["string", "null"] },
        "llm_input_index": { "type": ["integer", "null"], "minimum": 0 },
        "llm_output_index": { "type": ["integer", "null"], "minimum": 0 },
        "detection_confidence": { "type": ["number", "null"], "minimum": 0, "maximum": 1 },
        "ocr_confidence": { "type": ["number", "null"], "minimum": 0, "maximum": 1 },
        "text_style": { "$ref": "#/$defs/TextStyle" },
        "mask_strokes": {
          "type": "array",
          "items": { "$ref": "#/$defs/MaskStroke" }
        },
        "errors": {
          "type": "array",
          "items": {
            "type": "object",
            "required": ["code", "message"],
            "properties": {
              "code": { "type": "string" },
              "message": { "type": "string" }
            },
            "additionalProperties": false
          }
        }
      },
      "additionalProperties": false
    },
    "LlmDebug": {
      "type": "object",
      "properties": {
        "payload": { "type": ["object", "null"] },
        "raw_response": { "type": ["string", "null"] },
        "parsed_mapping": {
          "type": "array",
          "items": {
            "type": "object",
            "required": ["input_index", "output_index", "bubble_id"],
            "properties": {
              "input_index": { "type": "integer", "minimum": 0 },
              "output_index": { "type": "integer", "minimum": 0 },
              "bubble_id": { "type": "string", "format": "uuid" }
            },
            "additionalProperties": false
          }
        }
      },
      "additionalProperties": false
    }
  },
  "additionalProperties": false
}
```

## 2.4 Compatibilité metadata existante

Règles d’import depuis metadata pipeline actuelle:
- `detections[].bbox` (x1,y1,x2,y2) → `bbox` (x,y,w,h).
- `detections[].original` → `source_text`.
- `detections[].translated` → `translated_text`.
- `detections[].class` → `class`.
- Générer `id` UUID si absent.
- `llm_input_index` / `llm_output_index` initialisés selon l’ordre importé.
- `viewport` initial: `zoom=1`, `pan_x=0`, `pan_y=0`, `show_translated=true`.

---

## 3) Contrats API backend (MVP)

Base URL: `http://127.0.0.1:8000/api/v1`  
Règle de concurrence MVP: une seule opération de job active; sinon `409`.

## 3.1 Types partagés (TypeScript)

```ts
export type UUID = string;

export interface BBox {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface BubbleDTO {
  id: UUID;
  bbox: BBox;
  class: string;
  source_text: string;
  translated_text: string;
  llm_input_index: number | null;
  llm_output_index: number | null;
  detection_confidence?: number | null;
  ocr_confidence?: number | null;
  errors?: { code: string; message: string }[];
}

export interface ApiError {
  code: string;
  message: string;
  details?: unknown;
  bubble_id?: UUID;
}
```

## 3.2 Endpoints

### 3.2.1 Health / capacité

`GET /health`

Réponse:

```json
{
  "status": "ok",
  "localhost_only": true,
  "single_job_mode": true,
  "features": {
    "detect": true,
    "ocr": true,
    "translate": true,
    "render": true,
    "remap": true
  }
}
```

### 3.2.2 Mode Auto (pipeline complet)

`POST /jobs/auto`

Request:

```json
{
  "image_path": "A:/omni read/image test/image1.png",
  "output_dir": "A:/omni read/output image test",
  "debug": true,
  "cache_enabled": true
}
```

Response:

```json
{ "job_id": "uuid", "status": "queued" }
```

`GET /jobs/{job_id}?offset=0`

Response:

```json
{
  "job_id": "uuid",
  "status": "running",
  "logs": [{ "ts": "...", "level": "INFO", "message": "..." }],
  "next_offset": 42,
  "result": {
    "page": {
      "width": 800,
      "height": 13255,
      "bubbles": []
    },
    "preview_path": ".../image1_translated.png"
  },
  "error": null
}
```

### 3.2.3 Mode Manuel — Détection

`POST /detect`

Request:

```json
{
  "image_path": "A:/.../page_001.png",
  "classes": ["bulle", "out_text", "System"],
  "debug": false
}
```

Response:

```json
{
  "page": { "width": 800, "height": 4000 },
  "bubbles": [
    {
      "id": "uuid",
      "bbox": { "x": 120, "y": 300, "w": 220, "h": 90 },
      "class": "bulle",
      "source_text": "",
      "translated_text": "",
      "llm_input_index": null,
      "llm_output_index": null,
      "detection_confidence": 0.89,
      "ocr_confidence": null,
      "errors": []
    }
  ],
  "errors": []
}
```

### 3.2.4 Mode Manuel — OCR

`POST /ocr`

Request:

```json
{
  "image_path": "A:/.../page_001.png",
  "bubbles": [
    {
      "id": "uuid",
      "bbox": { "x": 120, "y": 300, "w": 220, "h": 90 },
      "class": "bulle",
      "source_text": "",
      "translated_text": "",
      "llm_input_index": null,
      "llm_output_index": null
    }
  ]
}
```

Response:

```json
{
  "bubbles": [
    {
      "id": "uuid",
      "source_text": "SAVE ME!!!",
      "ocr_confidence": 0.95,
      "errors": []
    }
  ],
  "errors": []
}
```

### 3.2.5 Mode Manuel — Traduction

`POST /translate`

Request:

```json
{
  "bubbles": [
    {
      "id": "uuid",
      "source_text": "SAVE ME!!!",
      "translated_text": "",
      "llm_input_index": 0,
      "llm_output_index": null
    }
  ],
  "cache_enabled": true,
  "return_llm_debug": true
}
```

Response:

```json
{
  "bubbles": [
    {
      "id": "uuid",
      "translated_text": "Sauvez-moi !",
      "llm_input_index": 0,
      "llm_output_index": 0,
      "errors": []
    }
  ],
  "llm_debug": {
    "payload": { "items": [{ "index": 0, "text": "SAVE ME!!!" }] },
    "raw_response": "{\"0\": \"Sauvez-moi !\"}",
    "parsed_mapping": [
      { "input_index": 0, "output_index": 0, "bubble_id": "uuid" }
    ]
  },
  "errors": []
}
```

### 3.2.6 Debug Mapping — Remap dédié

`POST /translate/remap`

Request:

```json
{
  "page_id": "uuid",
  "remap": [
    { "bubble_id": "uuid-a", "output_index": 3 },
    { "bubble_id": "uuid-b", "output_index": 1 }
  ]
}
```

Response:

```json
{
  "bubbles": [
    { "id": "uuid-a", "translated_text": "...", "llm_output_index": 3, "errors": [] },
    { "id": "uuid-b", "translated_text": "...", "llm_output_index": 1, "errors": [] }
  ],
  "errors": []
}
```

### 3.2.7 Rendu

`POST /render`

Request:

```json
{
  "image_path": "A:/.../page_001.png",
  "bubbles": [
    {
      "id": "uuid",
      "bbox": { "x": 120, "y": 300, "w": 220, "h": 90 },
      "translated_text": "Sauvez-moi !",
      "text_style": {
        "font_family": "Anime Ace",
        "font_size": 28,
        "align": "center",
        "color": "#FFFFFF"
      },
      "mask_strokes": [
        {
          "id": "uuid-stroke",
          "size": 24,
          "points": [{ "x": 10, "y": 12 }, { "x": 16, "y": 18 }]
        }
      ]
    }
  ],
  "text_only": true,
  "skip_inpainting": true
}
```

Response:

```json
{
  "preview_path": "A:/.../assets/previews/page_001_translated.png",
  "timings": {
    "text_render_ms": 180,
    "inpaint_ms": 0,
    "total_ms": 220
  },
  "errors": []
}
```

### 3.2.8 Cache toggle

`POST /cache`

Request:

```json
{ "enabled": true }
```

Response:

```json
{ "enabled": true }
```

## 3.3 Sémantique erreurs
- Erreur globale: HTTP non-2xx + payload `ApiError`.
- Erreur partielle par bulle: HTTP 200 avec `errors[]` au niveau bulle.
- Exemples de `code`: `ocr_empty`, `translation_failed`, `render_out_of_bounds`, `job_locked`.

## 3.4 Performance contract (MVP)
- Rerender texte seul cible: < 500ms (images dans contraintes MVP).
- Rerender avec inpaint cible: < 3000ms.

---

## 4) Architecture frontend (React + TS)

## 4.1 Stack
- React 18 / TypeScript
- Tauri v2 (Bureau Windows)
- Zustand (Immer middlewares pour historique Undo/Redo)
- React-Konva pour le canvas central.
- **Tailwind CSS** pour le styling global de l'interface (remplace le CSS personnalisé strict d'origine) + Lucide React pour les icônes.
- Communication backend via fetch local (`http://127.0.0.1:8000`).

## 4.2 Structure modules

```text
src/
  app/
    App.tsx
    routes.ts
  features/
    project/
      projectStore.ts
      projectSerializer.ts
      projectAutosave.ts
      projectImporter.ts
    image-list/
      ImageListPanel.tsx
      imageListStore.ts
    canvas-editor/
      CanvasEditor.tsx
      KonvaLayers.tsx
      tools/
        selectTool.ts
        drawTool.ts
        moveResizeTool.ts
        deleteTool.ts
        panTool.ts
        brushTool.ts
      canvasStore.ts
    bubble-inspector/
      BubbleInspectorPanel.tsx
      textOverrideForm.tsx
      styleForm.tsx
    llm-mapping-debug/
      LlmMappingPanel.tsx
      mappingStore.ts
      remapTable.tsx
    render/
      renderStore.ts
      renderService.ts
    jobs/
      jobsStore.ts
      pollingService.ts
    settings/
      SettingsPanel.tsx
      settingsStore.ts
  shared/
    api/
      client.ts
      endpoints.ts
      types.ts
    history/
      historyStore.ts
      patchHistory.ts
    components/
    utils/
```

## 4.3 Stores (Zustand)

### ProjectStore
Responsabilités:
- Charger/sauver `project.json`.
- Maintenir `schema_version` + migrations.
- Import metadata legacy vers format projet.
- Autosave toutes les 10s si dirty.

State minimal:
- `project`, `isDirty`, `lastSavedAt`, `isSaving`, `saveError`.

Actions clés:
- `createProjectFromImages()`
- `loadProject(path)`
- `saveProject()`
- `markDirty()`
- `applyMigrationIfNeeded()`

### CanvasStore
Responsabilités:
- Page active, bulle active, mode outil.
- Viewport (zoom/pan).
- Overlay toggle OCR/traduit.

State minimal:
- `activePageId`, `activeBubbleId`, `tool`, `viewport`, `showTranslated`.

Actions clés:
- `setTool()`
- `setViewport()`
- `selectBubble()`
- `updateBubbleBBox()`
- `addBubble()`
- `removeBubble()`
- `addMaskStroke()`

### MappingStore
Responsabilités:
- Conserver `llm_debug` par page.
- Edit table remap input/output index ↔ bubble UUID.

Actions clés:
- `setLlmDebug()`
- `setOutputIndex(bubbleId, outputIndex)`
- `applyRemap()` (appel API `/translate/remap`)

### JobsStore
Responsabilités:
- Exécuter mode Auto + étapes manuelles.
- Polling d’état.
- Verrou 1 job actif.

State minimal:
- `activeJobId`, `status`, `logs`, `lastError`.

### HistoryStore
Responsabilités:
- Undo/redo incrémental (patches Immer).
- Limite 50 actions.

Actions clés:
- `pushPatch()`
- `undo()`
- `redo()`

Note: historique non persisté dans `project.json` (choix validé).

### SettingsStore (persist local machine)
Responsabilités:
- URL API, cache on/off, options UI locales.
- Persist local (`localStorage` Tauri webview).

---

## 5) Flows UX principaux

## 5.1 Mode Auto
1. Sélection images.
2. Lancer job auto.
3. Poll logs/statut.
4. Recevoir bulles + preview.
5. Édition manuelle si nécessaire.
6. Save projet.

## 5.2 Mode Manuel
1. `detect`.
2. Ajuster bboxes.
3. `ocr` sur bboxes actuelles.
4. Override source/traduit.
5. `translate`.
6. Corriger mapping via panneau debug + `remap`.
7. `render` texte seul ou complet.

## 5.3 Panneau debug mapping LLM
- Table colonnes: `bubble_id`, `llm_input_index`, `llm_output_index`, `source_text`, `translated_text`.
- Afficher payload LLM + raw response.
- Bouton “Appliquer remap” → endpoint dédié.
- Répercussion immédiate sur preview après rerender.

---

## 6) Raccourcis clavier MVP
- `V`: Select.
- `B`: Draw bbox.
- `Espace` (maintenu): Pan.
- `Suppr`: delete bulle active.
- `Ctrl+Z`: Undo.
- `Ctrl+Y`: Redo.
- `Ctrl+S`: Save projet.

---

## 7) Backlog MVP priorisé

## 7.1 P0 — Fondations (bloquant)
1. Initialiser app React + TypeScript dans Tauri frontend.
2. Mettre en place layout 3 colonnes responsive desktop.
3. Mettre en place API client HTTP typé + gestion erreurs standard.
4. Créer stores Zustand de base (`project`, `canvas`, `jobs`, `settings`, `history`, `mapping`).
5. Implémenter import metadata legacy → format projet v1.

Critères d’acceptation:
- L’app démarre dans Tauri, charge une image/page, affiche les 3 panneaux.

## 7.2 P1 — Fonctionnalités non négociables
1. Canvas Konva avec outils bbox (select/draw/move/resize/delete).
2. Inspector bulle: override source/traduit.
3. Panneau debug mapping LLM + remap manuel.
4. Save/load projet (`project.json` + assets).
5. Rerender instantané (`/render` texte seul prioritaire).

Critères d’acceptation:
- Les 5 features non négociables sont utilisables de bout en bout sur au moins 1 page.

## 7.3 P2 — Mode Auto + Mode Manuel complet
1. Intégrer `jobs/auto` + polling logs.
2. Intégrer endpoints manuels `detect`, `ocr`, `translate`, `render`.
3. Gestion erreurs par bulle dans UI.
4. Toggle cache backend (`/cache`).

Critères d’acceptation:
- Un utilisateur peut exécuter auto ou manuel et corriger localement sans rerun complet.

## 7.4 P3 — Édition avancée MVP
1. Brush inpainting binaire par bulle (taille variable).
2. Typo par bulle: taille, align, police, couleur.
3. Toggle avant/après.
4. Undo/redo incrémental (50 actions).
5. Autosave 10s + save manuel.

Critères d’acceptation:
- Actions éditoriales principales réversibles, sauvegardées et restaurables.

---

## 8) Définition de terminé (MVP)

Le MVP est considéré prêt si:
- Les 5 features non négociables sont livrées.
- Les deux modes (Auto, Manuel) fonctionnent.
- Le bug de permutation d’index est corrigeable via UI sans relancer tout le pipeline.
- Les performances cibles de rerender sont respectées sur la cible matérielle MVP.
- Le format projet v1 est stable, versionné, et backward-compatible metadata actuelle.
