# Journal des modifications — phase « qualité du rendu »

Traduction désactivée : le harnais `scratch/render_iterate.py` réinjecte le
texte OCR d'origine, donc toute différence visible entre AVANT et APRÈS est
imputable à l'effacement ou au rendu, jamais à la traduction.

Série de travail : **path-of-vengeance**, `Chapitre 001_merged_part01.jpg`
(690 × 44 830 px, 38 zones détectées, 37 rendues).

## Outils ajoutés (scratch/)

| Fichier | Rôle |
|---|---|
| `render_iterate.py` | Pipeline complète sans traduction, crops avant/après par bulle |
| `build_contact_sheet.py` | Planches-contact AVANT \| APRÈS (vue 2) |
| `side_by_side.py` | Tuiles pleine page ORIGINAL \| RÉSULTAT (vue 1) |
| `pick_slice.py` | Choisit une tranche représentative dans un strip très haut |
| `erase_lab.py` | Cache détection+OCR+segmentation, puis rejoue **l'effacement seul** (itération en secondes au lieu de minutes) |
| `diag_erase.py` | Trace la branche réellement empruntée dans `inpaint_region` |
| `exp_inpaint.py` / `exp_mask.py` / `exp_glow.py` / `exp_texture.py` | Bancs de comparaison de stratégies d'effacement |

## Corrections

### E1 — Masque de glyphe inversé sur texte d'impact
`pipeline.py::_ocr_mask_from_regions` — le seuillage d'Otsu ne retient qu'un
côté. Sur un cartouche (corps noir + gros contour blanc) posé sur un fond
sombre, « l'encre » désignée était le CONTOUR, et le corps de la lettre restait
hors masque. Ajout de `_fill_glyph_holes` : les trous fermés du masque (bornés
par la hauteur de ligne au carré) sont comblés.

### E2 — Contexte d'inpainting indexé sur la taille de la zone
`core/renderer.py::inpaint_region` — la marge de crop était fixe (30 px, 60 px
pour out_text). Sur un cartouche de 423 × 284, LaMa voyait 39 % de trou et
rendait une bouillie violette. Marge portée à `2 × hauteur` (out_text) et
`1 × hauteur` (bulle) : part masquée du crop 39 % → 6 %.

### E3 — Masque au BLOC pour les cartouches out_text
`core/renderer.py::_block_mask_from_regions` — le texte d'impact porte une
LUEUR externe qui déborde largement du glyphe ; un masque au glyphe, même
parfait, laissait le rectangle du bloc visible en violet. Le masque devient les
polygones de ligne OCR dilatés de 0,30 × hauteur de ligne. Vérifié sur les six
cartouches de la planche : aucune trace, sur fond noir comme sur rayures rouges
ou éclairs.

### E4 — Repli « diffusion Navier-Stokes » désactivé
`core/renderer.py::_diffusion_is_safe` + `config.rendering.diffusion_fallback_enabled`
(défaut `false`, réactivable par `WEBTOON_DIFFUSION_FALLBACK=true`). Ce repli
datait de l'époque où LaMa ne voyait que 30 px de contexte. Il fusionnait les
lignes d'un bloc en larges bandes grises — mesuré sur la bulle à trames de
vitesse « YOU LITTLE-! YOU'RE WAY TOO CASUAL ABOUT THIS! », où LaMa reconstruit
les trames sans défaut au même endroit.

### R1 — Centre de référence horizontal faux
`core/renderer.py::_wrap_text_by_mask` — `default_center = mask_x_origin + inner_w/2`
mélangeait l'origine de la BBOX et la largeur de la zone INTÉRIEURE : le centre
était décalé vers la gauche de la moitié de la marge, et tout le bloc avec lui.
Remplacé par le centre horizontal réel du ballon, mesuré sur le masque.

### R2 — Marge intérieure sur le masque de wrap
`core/renderer.py::_bubble_shape_mask` — le masque décrit le ballon trait
compris (85 % de la bbox mesuré, contre ~78 % pour l'ovale réel), donc les
largeurs de ligne amenaient le texte au contact du contour. Érosion de 6 % de
la plus petite dimension, annulée si elle ferait disparaître le masque.

### R3 — Nombre de lignes cible
`core/renderer.py::_fit_font_hard(target_lines=…)` — maximiser la taille de
police produisait un empilement de mots isolés qui « tient » mais ne ressemble
pas à du lettrage (six lignes d'un mot là où la planche en fait trois : la
police de rendu est plus large que celle du studio). On redescend d'un cran en
taille tant que le bloc dépasse le nombre de polygones de ligne OCR + 1.

### R4 — Centrage vertical optique
`core/renderer.py::_optical_center_y` — `total_h` compte des cadratins complets
alors qu'un texte tout en capitales n'occupe que la moitié haute du cadratin :
le bloc était systématiquement rendu trop haut. On centre l'encre réelle.

### R5 — Plancher de largeur de ligne
`core/renderer.py::_wrap_text_by_mask` — près des pointes d'un ovale, la largeur
de bande tombe à quelques dizaines de pixels et la ligne se réduit à un mot, ce
qui rallonge le bloc et pousse la ligne suivante encore plus près de la pointe.
Plancher calé sur la largeur TYPIQUE du ballon (médiane des bandes de sa moitié
centrale), pas sur le rectangle inscrit.

### R6 — Zone reprise en entier si le retrait « loin des voisines » étouffe le texte
`core/renderer.py::insert_text` — sur deux bulles qui se chevauchent, le retrait
amputait la zone au point que rien ne tenait ; le repli au plancher produisait
un bloc plus haut que la zone, qui démarrait AU-DESSUS de la bulle. Déborder de
sa propre bulle étant pire que déborder sur la zone d'une voisine, on reprend la
zone complète dans ce cas.

### R7 — Interligne
`config/settings.py` — `line_spacing_ratio` 0,25 → 0,18 (+14,8 % de hauteur de
bloc en moyenne sur 33 bulles mesurées, ce qui forçait une taille plus petite).

### R8 — Police de lettrage
`core/renderer.py::PREFERRED_FONTS` — l'heuristique retenait « le premier
fichier du dossier dont le nom contient bold », soit `buddychampionbold.ttf`
pour presque tous les styles : police techno à **zéro barré** (« LEVEL: 10 »
sortait « LEVEL: 1Ø », « TOKYO » sortait « TOKYØ ») et à dessin carré. Choix
explicite des polices de scanlation présentes dans `assets/fonts`
(CCWildWords, Anime Ace). `system_card` garde volontairement sa police techno.

### R9 — Épaisseur de contour indexée sur la taille du texte
`core/renderer.py::insert_text` — le résolveur de couleurs renvoie 2 px quelle
que soit la taille ; sur un cartouche à lettres de 60 px le texte paraissait
posé par-dessus l'image. Contour = `0,075 × hauteur de ligne source`.

### O1 — Découpeur de mots collés inactif
`core/ocr.py` — `wordsegment` n'était pas installé dans l'interpréteur qui
exécute réellement la pipeline, et un `except Exception` large avalait le
`ModuleNotFoundError` : le découpeur était un no-op silencieux. Installé, et
l'échec d'import est désormais journalisé bruyamment.

### O2 — Ponctuation finale mangée
`core/ocr.py::_strip_watermark_fragments_safe` — `utils/filters.py` terminait par
un `.strip(" .-—|")` inconditionnel, même sans motif de filigrane : « AKINA… »
→ « AKINA », « HEHE. » → « HEHE ». Ne rogne plus la ponctuation que si un motif
a effectivement matché.

### O3 — Mot collé derrière une contraction
`core/ocr.py::_split_after_contraction` — « YOU'REMY », « I'MGOINGTO »,
« KAZUKI'SON » : l'apostrophe interne faisait renoncer `_segment_token`.
Repli qui coupe après le suffixe de contraction. Tests : `tests/test_word_splitter.py`
(51/51), `tests/test_ocr_post_processing.py`.

### O4 — Confusion Y/4
`utils/ocr_cleaner.py` — « SEE YOU WIN » lu « SEE 40 U WIN » (biais du modèle sur
le Y stylisé de cette police). Motifs ciblés `40 U` → `YOU`, `4 EAH` → `YEAH`.

## Itération 2 — retours utilisateur (bulles 2, 3, 5, 9, 20, 36 + out_text illisibles)

### E5 — Contour de bulle entamé par l'effacement
`core/renderer.py::_extend_fill_mask(inside_box=…)` — le remplissage à plat
absorbe tout ce qui, dans un rayon de **21 px** autour des lettres, s'écarte du
fond et leur est connexe. Dès qu'une ligne de texte est large, le trait du
ballon est à moins de 21 px : il est sombre, donc « s'écarte du fond », il est
connexe aux lettres, et il finit repeint en blanc. Mesuré sur « IT MIGHT JUST
BE A NORMAL RUN, BUT… » : 574 px absorbés dont **48 sur le contour**, à gauche
et à droite à la même hauteur — le décrochement visible.

La longue portée n'est justifiée que HORS de la bbox (c'est là que vit le
morceau de glyphe coupé par la boîte de détection). À l'intérieur, l'antialias
tient dans 2–3 px. Portée ramenée à 5 px dans la bbox, 21 px conservés dehors.
Vérifié : 48 → **0** px de contour modifié sur cette bulle, et 0 sur les 37
bulles de la planche (les seules zones périphériques modifiées sont le texte des
bulles VOISINES présentes dans la marge du crop, ce qui est correct).

Trois pistes ont été essayées et écartées avant celle-là, chacune vérifiée
comme sans effet sur le défaut mesuré : éroder `det.mask_binary` (c'est un
masque de LETTRES, pas le ballon) ; filtrer les composantes trop hautes du
masque d'encre (le trait y est déjà tronqué à la hauteur du polygone, donc
indiscernable d'une lettre) ; déduire l'intérieur du ballon par
`_bubble_mask_from_image` (non fiable tant que le texte d'origine est encore là
— recouvrement nul avec l'encre, mesuré).

### R10 — Cartouches out_text trop petits et trop pâles
Trois causes cumulées, `core/renderer.py` :
- `_draw_exact_lines` ajustait chaque ligne à la largeur de SON polygone OCR,
  serré sur le texte d'origine ; la police de rendu étant plus large que celle
  du studio, la contrainte de largeur mordait avant celle de hauteur et chaque
  ligne rétrécissait. La largeur disponible devient celle du BLOC (enveloppe de
  tous les polygones) ; la hauteur du polygone reste la référence de corps.
- `font_hint` forcé à `bold` pour `out_text` : sur la planche d'origine ces
  cartouches sont gras.
- contour garanti pour `out_text` quand le résolveur de couleurs le supprimait :
  « bon contraste » n'a pas de sens sur un fond d'éclairs ou de flammes dont la
  luminosité change d'un bout à l'autre du cartouche.

## Série 2 — the-frontier-count's-10th-class-outcas (410 × 64 188, 88 bulles)

L'effacement passe cette série **sans aucune retouche** — fonds noirs, dégradés,
cartouches posés sur du décor. Les corrections ci-dessous portent toutes sur le
texte.

### C1 — Couleur du texte d'origine perdue
`core/renderer.py::extract_original_text_color(ink_mask=…)` + `pipeline.py::_prepare_render_style`.
Les cartouches rouges ressortaient BLANCS, le dialogue bleu ressortait NOIR.
Deux causes empilées :
- l'échantillonnage se faisait sur le POLYGONE de ligne OCR rempli, qui contient
  les lettres, le fond entre les lettres et le contour blanc → moyenne délavée
  (220, 183, 179) ;
- le k-means qui suit retient le groupe le plus ÉLOIGNÉ du fond. Avec un masque
  d'encre, « le fond » est le noir de la case, et le groupe le plus éloigné du
  noir est le CONTOUR BLANC, pas le rouge.
Correction : échantillonner le CŒUR du trait (masque d'encre érodé de 3 px) et
prendre la MÉDIANE, sans k-means. Mesuré : (153, 59, 52) le vrai cramoisi,
(83, 126, 182) le vrai bleu.

### C2 — Corps de texte variable dans un même cartouche
`core/renderer.py::_fit_block_lines`. `_draw_exact_lines` ajustait chaque ligne à
SON polygone : sur « IT WOULD GO QUIET, ONLY TO ERUPT AGAIN WITHOUT WARNING. »,
la 3e ligne sortait deux fois plus petite que la 1re. Prendre simplement la plus
petite des tailles ne marchait pas non plus (la répartition des mots par largeur
de polygone ne retrouve pas les coupures d'origine, une ligne surchargée tirait
tout vers le bas). On cherche maintenant la plus grande taille à laquelle le
texte, REDÉCOUPÉ sur la largeur du bloc, tient encore dans le nombre de lignes
disponibles.

### C3 — Bulles de cri : texte à la moitié du corps d'origine (défaut restant de la série 1)
Trois suspects ont été innocentés par la mesure (`_get_inner_zone` : dérive de
4-15 px seulement ; `_mask_row_span`/`_mask_row_center` : conformes à un recalcul
manuel ; l'érosion `_inset` : 17-19 % de perte de bande sur les bulles dentelées
contre 13-15 % sur les ovales, pas de seuil séparateur).

Le vrai coupable : `_shrink_zone_away_from_siblings`. Sur « TOO SLOW, KAZUKI! »
(bulle de 297 px de haut), la hauteur utile tombait à **113 px** — le retrait
raisonne sur des BBOX, et deux bulles de cri ont des bbox qui se recouvrent
largement alors que leurs contours se touchent à peine. Deux correctifs :
- plancher d'aire : au-delà de 40 % d'aire perdue, on renonce au retrait ;
- on garde celui des deux dégagements (X ou Y) qui préserve le plus d'aire, au
  lieu de trancher sur l'axe de moindre pénétration — mesuré, dégager en X ne
  gardait que 29 % de la zone contre 50 % en Y. Ce choix ne peut jamais faire
  moins bien que l'ancien, qui reste l'un des deux candidats.

Ajouté aussi : mise en page sur l'ENVELOPPE CONVEXE pour les bulles de cri. La
**solidité** (aire / aire de l'enveloppe) sépare les deux familles sans
ambiguïté sur les 30 bulles de la série 1 : 0,852 à 0,863 pour les cinq bulles
dentelées, ≥ 0,943 pour toutes les autres. Les pointes sont décoratives, le
lettreur d'origine y fait déborder le texte.

### C4 — Cicatrice d'effacement sur bulle en dégradé
`core/renderer.py::_smooth_fill`. Sur « A UNIQUE CONSTITUTION? », la 1re ligne
sortait propre et la 2e laissait une cicatrice grise nettement visible.
`_flat_fill_color` exige un fond quasi UNI et refuse un dégradé de bulle ; LaMa
prenait alors la main et y laissait ce résidu.

Nouveau chemin intermédiaire : modèle de fond LISSE (amorce par diffusion à
court rayon, puis flou d'écart-type ≈ 0,6 × épaisseur du trait). Il ne peut
représenter qu'une variation douce, donc il ne peut pas recopier la forme des
lettres.

Le test n'est pas une classification du fond — aucun critère mesuré ne sépare
proprement « dégradé » de « texturé », la couronne autour du texte contenant le
trait noir de la bulle qui domine toutes les statistiques. Il est AUTO-VALIDÉ :
on vérifie que le modèle explique les pixels NON masqués juste autour du texte.
Mesuré : dégradé de bulle → résidu médian 3, p75 4 (accepté) ; bulle à trames de
vitesse → médiane 12, p75 18 (refusé, rendu à LaMa qui la reconstruit très bien).

### C5 — Mots collés (complément)
`core/ocr.py::_KNOWN_WORD_DENYLIST` — même mécanisme que « ofcourse » : le corpus
web de `wordsegment` contient ces suites comme « mots connus », ce qui
court-circuitait le découpage. Ajoutés après vérification un par un : `ican`,
`backto`, `fromme`, `bigbrother`, `iwas`, `iam`, `buti`, `sizeof`, `ata`.
Tests : `test_word_splitter.py` 51/51, `test_ocr_post_processing.py` OK.

## OCR — les espaces se comptent dans l'image, pas dans un dictionnaire

### O5 — Noms propres détruits par le découpeur
`core/ocr.py::_estimate_word_count` + garde dans `post_process_text`.

« FERDA ROSNOVA. » était **correctement lu** par l'OCR et ressortait
« FERD A ROS NOVA. ». Ce n'est pas un défaut de reconnaissance : c'est le
découpeur statistique qui casse. `wordsegment` propose toujours une découpe
plausible, et rien dans le TEXTE ne distingue un nom propre d'un mot collé —
mesuré précédemment, « MORI » est plus fréquent que « KUROI », donc aucun seuil
de fréquence ne sépare « MORISHIGE » (à ne pas toucher) de « HEAVYWARRIOR » (à
découper).

L'information existe, mais dans les pixels : entre deux mots, le blanc est plus
large qu'une chasse entre lettres. On compte donc, par ligne détectée, les
blancs internes dépassant 0,32 × la hauteur de casse. Le découpeur n'est appelé
que si l'image montre PLUS de mots que le texte n'en contient.

Vérifié : `A UNIQUE CONSTITUTION?` 3=3 (intact), `IT MIGHT JUST BE A NORMAL RUN,
BUT...` 8=8 (intact), `MAKE SURE YOU COMEBACK SAFE, OKAY?` texte 6 / image 7
(découpage autorisé, à raison).

**Deux pistes essayées, mesurées, abandonnées** — elles sont documentées dans le
code pour qu'on ne les refasse pas :
1. *Desserrer les garde-fous quand le collage est prouvé* (longueur minimale de
   token à 4, court-circuit « mot connu » ignoré) : 4 gains contre
   8 dégradations — « AGO » → « A GO », « ABOUT » → « A BOUT », « MANA » →
   « MAN A », « INSIDE » → « I NSIDE ». La preuve qu'il MANQUE un espace ne dit
   pas OÙ il manque.
2. *Traiter « image < texte » comme une mesure peu fiable et découper quand
   même* : récupérait 4 lignes collées sur the-frontier-count, mais recassait
   « ROSNOVA ».

**Compromis assumé, mesuré sur deux séries** : i-married-the-dragon passe de
0/4 à **4/4 noms propres intacts** ; the-frontier-count perd l'aide du découpeur
sur **8 lignes**, toutes dans des cartouches à LUEUR, où le halo comble les
blancs et fait sous-compter la mesure. Un nom propre méconnaissable coûte plus
cher au traducteur qu'un mot collé, qu'il peut souvent recomposer.

*Prochaine étape identifiée* : la mesure se fait DÉJÀ par ligne, mais la garde
s'applique au bloc entier. L'appliquer ligne par ligne protégerait les lignes
propres tout en laissant le découpeur agir sur les lignes à lueur. Il faut pour
ça associer chaque portion de texte à sa ligne, ce que l'OCR ne renvoie pas
aujourd'hui sous cette forme.

## Exécution — un chapitre traité à 39 % sans que rien ne le signale

`pipeline.py::_process_chapters_mega_batch` et `run_translation_and_optimize.py`.
Le 2026-08-15, `i-married-the-dragon-i-killed` a planté au rendu de `part03`
(« array allocation size too large ») : l'exception remontait jusqu'à `main.py`
qui faisait `sys.exit(1)`, donc `part04`/`part05` n'ont jamais été tentés et
aucun `summary.json` n'a été écrit. Le lanceur ne lisait pas le code de sortie,
affichait « Traduction terminée. » et recopiait le dossier tronqué.
Preuves : `logs/webtoon_v5.log` (phases 1 et 2 terminées sur les 5 images, crash
en phase 3) et `qcheck_report.json` limité à part01/part02.

Corrigé : chaque image de la phase 3 est protégée individuellement (l'échec est
journalisé, compté, et le traitement continue), le nettoyage mémoire complet est
rétabli entre images (`MemoryManager.cleanup_medium()`, que le chemin
`process_directory` faisait déjà), et le lanceur vérifie le code de sortie puis
affiche un récapitulatif des séries en échec.

**Sécurité** : ce même lanceur contenait une clé API Gemini EN CLAIR et
réécrasait `.env` avec elle à chaque exécution. Fichier jamais commité (fuite
locale, pas publique). Il lit désormais `GEMINI_API_KEY` depuis l'environnement.

## O6 — Le modèle de reconnaissance était mal choisi

`_paddle_vl_worker.py`. Le worker chargeait `PP-OCRv5_server_rec` : le gros
modèle multilingue (81 Mo, précision publiée 86,4 %). Remplacé par
`latin_PP-OCRv5_mobile_rec` (14 Mo, précision publiée **inférieure**, 84,7 %) —
qui fait pourtant nettement mieux sur notre corpus, parce qu'il est spécialisé
sur l'alphabet latin là où le « server » partage sa capacité avec le chinois.

Banc de mesure créé pour trancher : `scratch/ocr_bench.py`, 16 bulles de 3
séries avec vérité terrain relevée à l'œil, score exact + taux d'erreur
caractère (CER).

| configuration | exacts | CER |
|---|---|---|
| **`latin_PP-OCRv5_mobile_rec`** | **10/16** | **0,0506** |
| `en_PP-OCRv5_mobile_rec` | 9/16 | 0,0524 |
| latin + détection min 640 px | 9/16 | 0,0628 |
| **`PP-OCRv5_server_rec` (avant)** | 7/16 | 0,0733 |
| latin + détection min 960 px | 9/16 | 0,0733 |
| `unclip_ratio` 1,2 | 7/16 | 0,0750 |
| crop agrandi ×2 | 5/16 | 0,0768 |
| crop agrandi ×3 | 5/16 | 0,1065 |

**Tous les réglages de détection ont été mesurés négatifs** une fois le bon
modèle en place — y compris forcer la résolution d'entrée, alors que la
détection ne redimensionne jamais nos crops par défaut
(`limit_type='min'`, `limit_side_len=64`, donc sans effet sur des crops déjà
plus grands). Les paramètres restent pilotables par variables d'environnement
(`PADDLE_DET_*`, `PADDLE_REC_MODEL`) pour pouvoir rejouer ces comparaisons.

Ce que ça corrige à la SOURCE, sans post-traitement :
- espaces perdus : « ITWOULDGOQUIET, ONLYTO ERUPT AGAIN WITHOUTWARNING » →
  exact ; « ITKEPTEVERYONE IN THE ESTATE ONEDGE. » → exact ; « BRINGBACKA
  SOUVENIR » → « BRING BACK A SOUVENIR » ; « IF YOULEAVE » → « IF YOU LEAVE » ;
- confusion Y/4 : « AREN'T 4 OU » → « AREN'T YOU » (le rustine regex de O4
  devient inutile) ;
- nom propre : « MORI SHIGE » → « MORISHIGE » (l'OCR ne le coupe plus, donc le
  découpeur n'a plus à être protégé de lui-même sur ce cas) ;
- casse : « you SHOULD » → « YOU SHOULD » ; « KAZUKI' SON » → « KAZUKI'S ON ».

Régressions honnêtes : « KAZUKI! » → « KAZUK! », « CANT LAND A » → « CANT
LANA », « THE TOKYO MAKAI » → « THe TOKYO MAkAl », « WILD... » → « WILD.. ».
Bilan sur la série 1 : nettement positif.

**Piste ouverte** : PaddleOCR **3.7.0** (juin 2026) apporte **PP-OCRv6**, dont
le benchmark comporte une catégorie « texte artistique » — exactement notre cas.
Le venv `.venv_paddleocr` est en **3.4.0** : la mise à jour n'a pas été faite
sans accord, elle touche un environnement qui fonctionne.

## O7 — PP-OCRv6, et le piège du parsing sur PaddleOCR-VL

Nouveau venv `.venv_paddle_next` (PaddleOCR 3.7.0 + PaddlePaddle GPU 3.2.2) ;
`.venv_paddleocr` (3.4.0) laissé intact. `PADDLE_VENV` permet de basculer, et le
worker choisit PP-OCRv6 si la version le permet, sinon le latin PP-OCRv5.

| configuration | exacts | CER |
|---|---|---|
| **PP-OCRv6 détection + reconnaissance** | **13/16** | **0,0401** |
| PaddleOCR-VL 1.6 | 12/16 | 0,0681 |
| PP-OCRv6 reconnaissance seule | 11/16 | 0,0384 |
| latin PP-OCRv5 | 10/16 | 0,0489 |
| PP-OCRv5 server (config d'origine) | 7/16 | 0,0733 |

**Le piège** : PaddleOCR-VL rendait VIDE sur les 16 bulles au premier essai — ce
qui confirmait à la fois le commentaire du code (« ne marche PAS sur des crops »)
et la recherche documentaire (~89 % d'erreur sur crops de bulles). Conclusion
toute faite… et fausse : **c'était le code de lecture des résultats**. VL ne
renvoie pas `rec_texts` mais `parsing_res_list[].block_content`. Corrigé, il fait
12/16 et réussit `DAMN IT... I CAN'T LAND A SINGLE HIT.` exactement, apostrophe
et points de suspension compris, là où toutes les versions de PP-OCR échouent.
Écarté malgré tout : il **hallucine** (un « も… » japonais surgi de nulle part,
un « RUN, BL » ailleurs) — quand un VLM se trompe, il invente du texte plausible
au lieu de rendre du charabia repérable.

Ce que PP-OCRv6 corrige à la source sur la série 2, sans une ligne de
post-traitement — dont la totalité des « défauts connus restants » listés plus
bas jusqu'ici :
« JEMAS JONAS! » → « JONAS! JONAS! » ; « ASYOU » → « AS YOU » ;
« ALONG TIME » → « A LONG TIME » ; « AMANA VESSEL » → « A MANA VESSEL » ;
« ATA SPEED » → « AT A SPEED » ; « THEMAN A INSIDE » → « THE MANA INSIDE » ;
« MANA DUT SIDE » → « MANA OUTSIDE » ; « MYSON » → « MY SON » ;
« ILL PLAY » → « I'LL PLAY » ; « YOu » → « YOU » ; « HAA. » → « HAA.. » ;
« MANY WEREKILLED ORINJURED. » → « MANY WERE KILLED OR INJURED. ».

## Défauts connus restants

1. **Mots collés de 5 lettres ou moins** : « ASYOU », « ANDNO », « AMANA »,
   « ATA », « ALONG » restent soudés. Ils tombent sous `_SEG_MIN_TOKEN_LEN = 6`,
   seuil délibéré qui protège les onomatopées courtes de l'émiettement. Cas
   particulier de « ALONG » (« THAT WAS ALONG TIME AGO ») : c'est un vrai mot
   anglais, seul le contexte permettrait de trancher — pas corrigeable par le
   découpeur statistique.
2. **Apostrophes et points de suspension manqués par l'OCR** sur certains crops
   (« DAMN IT… I CAN'T » → « DAMN ITI CANT ») : raté de reconnaissance, pas de
   post-traitement.
3. **PaddleOCR n'est pas déterministe** : deux passes sur les mêmes pixels
   peuvent donner des découpages différents (« BRING BACK A » vs « BRINGBACKA »).
4. **Noms propres sur-découpés** par `wordsegment` (« MORISHIGE » → « MORI
   SHIGE ») : à corriger par le glossaire de série, vide pour cette série.
5. Style des cartouches d'impact (graisse, contour, lueur) encore plus léger que
   l'original.
6. **Les textes longs rendent un cran plus petit que l'original.** La police de
   lettrage disponible est plus large par caractère que celle du studio, donc à
   hauteur de casse égale il faut plus de largeur, donc plus de lignes, donc un
   corps plus petit. J'ai tenté de mesurer le rapport largeur/casse du lettrage
   source pour choisir une police plus étroite (CCAskForMercy 0,66 et
   CCSamaritanTall 0,61 contre CCWildWords 0,72), mais la mesure côté source
   dépend de polygones OCR parfois tronqués et n'est pas assez fiable pour
   justifier un changement de police. À reprendre sur un corpus plus large.
