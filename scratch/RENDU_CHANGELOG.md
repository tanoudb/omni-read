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

## Mise en page face au foisonnement de la traduction (2026-08-18)

Tout le travail précédent avait été validé **traduction désactivée** — le texte
réinjecté était le texte source, donc de longueur identique. Le cas « la
traduction est plus longue que la VO » n'avait jamais été exercé. Le harnais
sait désormais traduire réellement (`--translate`) et simuler le foisonnement
(`--inflate`).

Trois correctifs, mesurés sur path-of-vengeance traduit par l'API (24 bulles) :

| | taille conservée (moy.) | pire cas |
|---|---|---|
| départ | 93,9 % | 48 % |
| + budget de lignes dynamique | 98,0 % | 48 % |
| + césure pyphen | 100,7 % | 48 % |
| + prédicat monotone | **102,7 %** | **79 %** |

### M1 — Budget de lignes indexé sur le foisonnement
Un budget constant (lignes source + 1) enfermait la traduction dans le
découpage de la planche : le moteur épuisait la réduction de taille avant
d'envisager une ligne de plus. Le budget suit maintenant le ratio de longueur
FR/EN, déduit du texte lui-même — donc valable pour toute langue cible.

### M2 — Césure syllabique (pyphen)
Un mot est insécable au sens de la mise en page : quand il ne tient pas, tout le
bloc rétrécit pour lui. `pyphen` plutôt que la géométrie — couper là où ça rentre
donnerait « RAVITA-ILLEMENT », les motifs Hunspell donnent « ra-vi-taille-ment ».
Changer le code langue suffit pour l'allemand.

Deux défauts de la première implémentation, tous deux mesurés :
- une seule coupure était tentée ; à 28 px la première césure laisse un reste de
  198 px qui déborde. Le reste est désormais recoupé tant qu'il ne tient pas ;
- la coupure n'était tentée que dans la place restante sur la ligne courante.

### M3 — Prédicat de mise en page NON MONOTONE (le vrai mur de #17)
Le défaut 2 ci-dessus faisait que la mise en page **tenait à 28 px mais pas à
24** : la césure ne se déclenchait qu'au-delà d'un certain corps. La recherche
dichotomique de `_fit_font_hard` suppose la monotonie — elle sautait donc
par-dessus la bonne taille et se rabattait sur 16 px.

Piste écartée en chemin : j'avais diagnostiqué un mur géométrique (retrait
« loin des voisines »). L'instrumentation l'a infirmé — zone utile de 147 × 238
sur un masque de 211 × 316, rien n'était amputé.

Conséquence : **l'A/B des polices de dialogue est annulé**. Le blocage écrasait
les quatre candidates de la même façon ; une fois levé, CCWildWords atteint
26 px sur #17. On garde sa largeur et son air qui respire, et on évite la
fatigue de lecture des variantes Bold condensées.

---

## Nuit du 2026-08-26/27 — Croissance depuis l'encre : ce qu'elle apporte réellement

Mission : remplacer la détection de forme de ballon par teinte
(`TextRenderer._bubble_mask_from_image`) par une **croissance depuis l'encre**,
au motif que la teinte produit des « coulées » sur les ballons sombres.

Le nouveau module est `core/bubble_shape.py`, le banc `scratch/test_mask_growth.py`
et `scratch/test_bubble_verdict.py`.

**Le résultat contredit la prémisse, et l'oriente ailleurs.**

### R1 — La prémisse elle-même était un artefact de bbox
La « coulée » observée sur le ballon rond sombre de i-married-the-dragon venait
d'une **bbox fausse**, pas de la polarité. Sur la bbox de production (score
0,898), la méthode par TEINTE donne 71,2 % de remplissage et 17,5 px de dérive
de centre (3,6 %) — elle fonctionne. Il n'y avait pas de coulée à corriger.

### R2 — Comme *forme*, la croissance ne gagne rien
Banc sur 12 ballons (1 sombre, 1 bulle de cri dentelée, 10 ovales classiques),
métriques : remplissage, dérive du centre bande par bande, asymétrie, IoU.

| | médiane | moyenne | pire |
|---|---|---|---|
| dérive du centre % — TEINTE | 7,42 | 10,07 | 27,26 |
| dérive du centre % — CROISSANCE | 9,13 | 11,06 | 27,65 |
| asymétrie % — TEINTE | 2,46 | 4,23 | 13,73 |
| asymétrie % — CROISSANCE | 3,22 | 4,51 | 13,90 |

IoU entre les deux : **0,937 à 0,983**. Elles décrivent le même objet. La
croissance ne gagne que sur le ballon sombre (dérive 3,59 % → 2,05 %,
asymétrie 0,96 % → 0,55 %) et perd de peu sur 10 cas sur 12.

**Remplacer la teinte serait donc une régression légère.** On ne le fait pas.

### R3 — Ma métrique de « dérive » mesurait la vraie forme du ballon
Les deux pires cas (pov#06 à 27 %, pov#23 à 22 %) sont pires **avec les deux
méthodes**, et l'inspection visuelle
(`scratch/mask_growth_visuel.png`) montre deux masques justes : pov#06 est une
bulle de cri qui se rétrécit vers le bas, pov#23 porte une queue en bas à
gauche. La dérive de bande y est l'asymétrie honnête du ballon, pas une erreur.
Ce n'est donc PAS un critère de qualité de masque — seulement un détecteur de
coulée.

### R4 — Le critère de fuite par contact avec les bords est faux
Première version : « la croissance touche ≥ 2 bords ⇒ fuite ⇒ enveloppe
convexe ». Elle amputait pov#23, un ballon parfaitement ordinaire, de 77,6 % à
47,7 % de remplissage. Mesure du recouvrement de bord : **pov#23 (légitime)
0,48, pov#06 (cri) 0,51** — le critère ne sépare rien. C'est mécanique : la
bbox étant dessinée autour du ballon, celui-ci touche ses bords par
construction.

En laissant simplement la croissance converger, **12 cas sur 12 se referment
d'eux-mêmes sur le trait**, bulle de cri dentelée comprise. Il n'y avait aucune
fuite à gérer. Le repli par enveloppe convexe demandé au cahier des charges
s'est révélé inutile ; le code de clôture le garde en option (`_close_contour(hull=True)`),
non branché.

### R5 — Le vrai discriminant est le REMPLISSAGE, et il sert à autre chose
Le seul signal qui sépare franchement les familles :

| | remplissage final |
|---|---|
| ballons (30 cas) | 52,9 – 79,5 % |
| textes libres `out_text` (6 cas) | 86,5 – 98,0 % |

Sans recouvrement. Seuil à **82 %** (`MAX_FILL`). Au-delà, la croissance ne
rend pas un masque : elle rend le verdict **« il n'y a pas de ballon ici »**.

C'est cette réponse-là qui a de la valeur, pas la forme. Sur les 36 détections
de path-of-vengeance :

| | accord avec l'étiquette de classe |
|---|---|
| TEINTE | 33/36 |
| CROISSANCE | **36/36** |

La teinte invente une forme de ballon sur pov#30, #31 et #33 — trois cartouches
`out_text`. Conséquence réelle et pas cosmétique : dans `insert_text`,
`has_mask_wrap` met `is_bubble` à vrai, ce qui **détourne le texte de
`_draw_exact_lines`** (le régime qui rejoue les lignes d'origine) vers le wrap
sur polygone. Sur pov#33 le masque fantôme ne fait que 27,3 % de la boîte : le
texte y est mis en page dans un blob qui n'existe pas.

### R6 — Le verdict exige le crop EFFACÉ (résultat négatif, utile)
Tenté : calculer le verdict sur le crop d'ORIGINE, ce qui aurait permis de le
loger dans `_prepare_render_style` (qui reçoit déjà l'image d'origine). Mesuré :
**30/36 d'accord seulement** — six vrais ballons (#03, #06, #14, #21, #26, #27)
basculent en « texte libre », parce que le dégagement de 9 px autour de l'encre
perce le contour là où le lettrage le frôle, et la croissance s'échappe. Le
veto doit donc se calculer dans la boucle de rendu, qui seule dispose des deux
images.

### R7 — Le veto est un gain sur les 3 cas, et ma première lecture était fausse
Mesure de l'effet réel : sur les 6 textes libres de path-of-vengeance, le veto
change **exactement** les 3 cas prédits ; les 3 autres sont identiques au pixel
près (0 pixel de différence). Aucun effet de bord.

J'ai d'abord annoncé une régression sur pov#31. C'était un artefact de mon banc :
je dessinais le rectangle de la bbox PAR-DESSUS le texte rendu, ce qui donnait
l'illusion d'une dernière ligne tranchée. Sans l'encadré, les trois cartouches
sont plus grands, mieux centrés, et pov#33 perd son alignement en escalier.

Débordement réel sous le plancher de la bbox, mesuré :

| | encre sous le plancher | débord max | hauteur bbox |
|---|---|---|---|
| pov#30 | 490 px | 2 px | 290 px |
| pov#31 | 3144 px | **11 px** | 233 px |
| pov#33 | 1833 px | **11 px** | 215 px |

Origine : `_fit_block_lines` tolère `hauteur_glyphe ≤ hauteur_polygone × 1,10`,
et `_draw_exact_lines` centre le glyphe dans son polygone — la moitié de
l'excédent passe donc dessous. Or les polygones OCR occupent TOUTE la hauteur de
la bbox (pov#31 : 0..233 pour une boîte de 233), donc le dernier polygone a son
bas sur le plancher. `_draw_exact_lines` ne reçoit pas la bbox : il ne peut pas
connaître ce plancher. C'est une contrainte ABSENTE, pas un calcul faux, et le
débordement des cartouches est autorisé.

### R8 — Les deux bibliothèques d'inpainting se disputaient le même fichier
`simple_lama_inpainting` et `lama_cleaner` téléchargent des builds DIFFÉRENTS
sous le même nom `~/.cache/torch/hub/checkpoints/big-lama.pt` :

| bibliothèque | source | md5 |
|---|---|---|
| `simple_lama_inpainting` | release *enesmsahin* | `19970cd5…` |
| `lama_cleaner` | release *Sanster* | `e3aa4aaa…` |

Conséquence, à CHAQUE construction de `TextRenderer` : `_init_anime_inpainter()`
trouvait le build enesmsahin, `torch.jit.load` échouait, `lama_cleaner`
SUPPRIMAIT le fichier — puis `SimpleLama`, qui ne vérifie aucun md5, le
retéléchargeait (205 Mo). Donc 205 Mo par lancement, et `anime_inpainter_ready`
définitivement à False.

Portée réelle, à ne pas surestimer : l'inpainter « anime » n'est qu'un repli de
3e rang, utilisé seulement si `SimpleLama` lève. `SimpleLama` fonctionnait. **Ce
n'était donc pas la cause des fantômes résiduels.**

Situation désormais stable et auto-réparante : le fichier est le build Sanster
(md5 conforme), `_init_anime_inpainter()` s'exécute AVANT `SimpleLama` dans le
constructeur, et `SimpleLama` réutilise sans broncher le fichier présent.

### R9 — Le contour mordu venait de la 2e passe, pas de la lueur
Diagnostic initial FAUX, à corriger : j'avais annoncé que la lueur rose de
« JUST KILL ME ALREADY!! » bavait jusqu'au trait et que le masque l'englobait.
L'instrumentation dit l'inverse. Masque réellement envoyé à LaMa, Dragon #0 :

| passe | part du crop masquée | trait du ballon avalé |
|---|---|---|
| 1 (chirurgical + lueur) | 11,7 % | **0 %** |
| 2 (bloc + halo_grow) | 40,6 % | **9,5 %** |

La passe 1 est PROPRE : elle épouse les lettres et leur lueur et laisse le
feston blanc intact. Les dégâts viennent entièrement de la deuxième passe, qui
masque `local_mask | block_mask` puis fait croître de 30 px. Or `block_mask`
dilate déjà les polygones de 0,30 × hauteur de ligne : sur une bulle de 595 px
dont deux lignes de ~150 px occupent presque toute la surface, il atteint le
trait AVANT la croissance, que `_halo_grow` ne peut donc plus éviter.

**Correctif** : borner la passe 2 par l'INTÉRIEUR du ballon, déduit par
`grow_from_ink` sur la sortie de la passe 1 — le texte y a disparu, le trait est
intact, c'est exactement l'image dont la croissance a besoin. Sur l'image
d'origine ce serait faux (R6 : 30/36).

La zone de calcul déborde la bbox de 20 % : les festons et la queue SORTENT de
la boîte de détection, et borner sur la bbox seule laissait encore 4,9 % du
trait dans le masque.

| variante | masque | trait avalé | ghost_score |
|---|---|---|---|
| passe 2 actuelle | 40,6 % | 9,5 % | 0,1681 |
| bornée, pad 0 % | 39,4 % | 4,9 % | — |
| bornée, pad 10 % | 38,6 % | 1,3 % | — |
| **bornée, pad 20 %** | **38,0 %** | **0 %** | **0,0363** |

Meilleur sur les deux critères à la fois, pour 2,6 points de couverture en moins.

**Piège de métrique à retenir** : sur `tex` et `t28` le `_ghost_score` MONTE
légèrement après correctif (2,6277 → 2,6371 ; 1,3990 → 1,4204) alors que
l'image est visiblement meilleure — le contour noir dentelé, aminci et émoussé
avant, ressort net. `_ghost_score` compte la structure résiduelle dans le
masque : préserver le trait AJOUTE de la structure. La métrique ne distingue
pas un fantôme de texte d'un contour légitime. Ne pas l'utiliser seul pour
arbitrer une protection de contour.

Non-régression : 19 détections sur 8 caches d'effacement ; la borne s'applique
sur 7, aucune dégradation visuelle constatée, 12 inchangées au pixel près.

### R10 — Le veto ne contredit plus une détection étiquetée `bulle`
Le seuil de remplissage de R5 (82 %) a été calibré sur UNE planche, où les
ballons plafonnaient à 79,5 %. Sur le chapitre entier il ne généralise pas :
9 vraies bulles remplissent 85 à 97 % de leur bbox — couronne hérissée, ou
boîte serrée sur le ballon — et se faisaient prendre pour du texte libre.

Le défaut MESURÉ que le veto corrige est ailleurs (la teinte qui invente une
géométrie sur des cartouches `out_text`). On le réserve donc à ce cas, et on
fait confiance à YOLO là où il annonce une bulle.

Mesure à travers le chemin de production, part03 (35 détections, 0 `out_text`) :

| | détections modifiées |
|---|---|
| veto AVEC garde | **0** |
| veto SANS garde | **5**, toutes `bulle` |

Et sur part01 (6 `out_text`), le veto agit toujours sur **6/6 cartouches et
0 bulle** — les trois non encore inspectés (#29, #32, #34) gagnent en corps
sans régresser.

### R11 — Le harnais ne reproduisait pas la production (piège de banc)
`scratch/render_iterate.py` appelait `insert_text` SANS `bubble_present`, alors
que les deux sites d'appel de `pipeline.py` le passent. Conséquence : un run
complet du chapitre « validait » un chemin que le pipeline n'emprunte pas, et
le veto n'y était jamais exercé — ni avant ni après le garde, d'où 0 pixel de
différence entre les deux runs, ce qui aurait pu passer pour « le correctif ne
sert à rien ».

J'ai d'abord annoncé « le veto a mal routé 9 vraies bulles pendant le run ».
C'était faux : c'était une mesure du classifieur HORS LIGNE, pas du rendu. Le
run n'a réellement validé que la borne d'effacement (R9), qui passe par
`_run_pre_inpainting`.

Règle : tout paramètre ajouté à `insert_text` doit être ajouté au harnais dans
le même commit, sinon le banc ment en silence.

## Phase « barème » — mesure chiffrée sur 8 séries

Outil : `scratch/bareme.py` (`build` / `score` / `report` / `sheet` / `compare`).
Corpus : la planche `_merged_part01` du premier chapitre des 8 séries de
`manhwa/`, soit **325 bulles**. Traduction désactivée comme dans
`render_iterate.py` : le texte OCR est réinjecté, donc tout écart mesuré entre
l'encre source et l'encre rendue est imputable au rendu.

`build` est le seul passage par YOLO/PaddleOCR (~70 s pour les 8 planches, mis
en cache) ; `score` rejoue effacement + rendu depuis ce cache et se relance
après chaque modification de `core/renderer.py`. `compare a b` donne le delta
par défaut et nomme les bulles qui basculent.

### M0 — Le banc doit passer CHAQUE argument de `pipeline.py`
Premier jet écrit d'après `render_iterate.py` d'avant son correctif : il
oubliait `bubble_present`, `outline_width_px` et `source_text`. Même piège que
celui déjà consigné, et même conséquence — noter un chemin que la production
n'emprunte pas. L'appel du barème est désormais copié sur celui de
`pipeline.py`, verdict `has_closed_bubble(img, after, …)` compris.

### M1 — Hook de mesure absent sur le chemin `_draw_exact_lines`
`core/renderer.py` — ce chemin (texte hors bulle) sortait sans renseigner
`last_layout_debug`, alors que `insert_text` le renseigne sur tous les siens :
38 zones sur 332 échappaient à toute mesure, dont 25 cartouches out_text. Le
texte y était bien dessiné (encre rendue > 0 sur les 38) ; c'est
l'instrumentation qui manquait. Hook ajouté, purement additif.

### M2 — `line_h_ratio` est un mauvais juge du corps de texte
Premier jet du barème : `font_size × 0,75 / source_line_height`, qui donnait
p50 = 0,67 et la conclusion « le texte sort à 70 % du corps du studio sur 8/8
séries ». **Faux.** Ce rapport compare la police à la hauteur des POLYGONES OCR,
plus lâches que l'encre réelle, et il réutilise la constante 0,75 em du
renderer — il ne peut donc pas l'arbitrer.

Remplacé par `cap_ratio` : hauteur médiane d'une bande d'encre (projection des
pixels sur les lignes), mesurée des DEUX côtés, sans aucune constante commune
avec le renderer. Résultat p50 = **1,00** : au médian, le corps est juste.

### M3 — La référence de débordement était fausse
Premier jet : encre rendue hors de l'enveloppe convexe du texte source. Mesuré
**0,4 % de débordement sur les bulles à corps trop petit contre 38 % sur celles
au bon corps** — l'inverse d'un défaut. À corps égal, un découpage de lignes
différent sort légitimement de cette enveloppe.

Remplacé par `bubble_overflow_pct` : encre rendue hors de l'intérieur du ballon,
détecté sur l'image EFFACÉE. E5 avait écarté la détection de ballon comme « non
fiable tant que le texte d'origine est encore là » — vrai sur l'originale, faux
après effacement où l'intérieur est lisse. Mesurable sur 262/325 zones ; rend
`None` ailleurs plutôt que d'inventer. Résultat : p50 = 0, **3 % des bulles**.

`mask_binary` a été essayé comme référence et écarté : 15–36 % de couverture de
bbox, c'est un masque de LETTRES, pas de ballon (déjà noté en E5).

### Ligne de base (run `actuel`, commit `b9b7e38`, 325 bulles)

| Métrique | p50 | p75 | Lecture |
|---|---|---|---|
| `erase_spill_pct` | 0,000 | 0,000 | effacement propre (p90 = 1,10) |
| `ghost_contrast` | 0,000 | 0,000 | 2 bulles sur 325 |
| `bubble_overflow_pct` | 0,000 | 0,000 | 3 % de bulles débordent |
| `cap_ratio` | 1,000 | 1,225 | corps juste au médian |
| `footprint_ratio` | 1,234 | 1,755 | pavé plus étalé que l'original |
| `rag_cv` | 0,192 | 0,293 | équilibre des lignes |

**L'effacement généralise.** `erase_spill` et `ghost` sont à zéro au p75 sur les
8 séries, dont 5 jamais travaillées. La partie difficile est derrière.

**Les cartouches out_text sont réglés.** Par chemin de rendu, `exact_lines`
donne p25 = 0,92 / p50 = 0,98 / p75 = 1,05. Le défaut connu n°5 (« cartouches
d'impact plus légers que l'original ») ne se mesure plus. Sur le code du
14 août, le même chemin donnait p50 = 0,71 : ce sont les 18 commits de
`rendu-ocr-2026-08-15` qui l'ont corrigé.

**Le corps de texte ne généralise pas.** `cap_ratio` médian par série :

| Série | p25 | `cap_ratio` | empreinte | orphelins |
|---|---|---|---|---|
| i-married-the-dragon | 0,57 | **0,75** | 0,75 | 21 % |
| path-of-vengeance | 0,65 | 0,85 | 1,11 | 35 % |
| the-frontier-count's | 0,78 | 0,93 | 1,21 | 30 % |
| 30-years-have-passed | 0,67 | 0,96 | 0,86 | 10 % |
| hellogin | 0,88 | 1,05 | 1,44 | 33 % |
| the-wind-mage | 1,03 | 1,05 | 1,65 | 12 % |
| rise-of-the-dragon | 1,05 | **1,29** | 2,08 | 24 % |
| the_cleaner | 1,19 | **1,38** | 1,83 | 23 % |

Facteur **1,8** entre la série la plus petite et la plus grosse, et la
dispersion INTERNE est du même ordre (chemin `wrap` : p25 = 0,76, p75 = 1,24).
Le défaut n'est donc pas « trop petit » ni « trop gros » : le choix de corps
n'est pas stable, ni d'une série à l'autre ni d'une bulle à l'autre. C'est
précisément ce qui empêche un rendu professionnel indépendant du manhwa.

Second défaut, indépendant : `footprint_ratio` p50 = 1,23 / p75 = 1,76 — à corps
égal le pavé s'étale plus que l'original — et **25 % des blocs multi-lignes
finissent sur un mot orphelin**.

### Défauts trouvés par les extrêmes du barème (hors périmètre mesuré)

La planche-contact (`sheet --metric cap_ratio`) a fait remonter quatre défauts
que le barème ne cherchait pas :
1. **Filigranes rendus** : « VORTEXSCANS.COM » → « SCANSREAD », et
   « READ THIS SERIES FIRST AT: VORTEXSCANS.COM » traduits et redessinés.
2. **Logos de titre traduits** : le logo « THE CLEANER » ressort en texte plat
   « IHE CLEANER ZORONGE YLAB ».
3. **Texte tronqué au grand corps** : « OR ACCEPT AND BE TORN TO PIECES » →
   « ACCFAND B TORNPIECES ».
4. **Ligne fantôme dupliquée** : « I MUST GET A COMBAT JOB! » suivi d'un
   « CUM BAT JUB! » parasite.

### T1 — Plafond de dimensionnement calé sur le POLYGONE au lieu de l'ENCRE
`pipeline.py::_measure_source_ink_height` (nouveau) + `_prepare_render_style`.

Le corps rendu variait d'un facteur 1,8 entre séries (0,75 pour i-married à
1,38 pour the_cleaner). Deux hypothèses testées :

1. *« Le rapport polygone/encre varie selon la série et transmet sa variance. »*
   **Écartée par la mesure** : ce rapport est remarquablement stable — p50 entre
   1,21 et 1,26 sur les 8 séries — et sa corrélation avec `cap_ratio` vaut
   r = −0,16, c'est-à-dire rien.

2. *« La taille est un `min` de trois contraintes ; l'écart vient de laquelle
   mord. »* **Confirmée.** En isolant la taille CHOISIE rapportée à la taille
   impliquée par l'encre source mesurée :

   | | n | part | taille/source |
   |---|---|---|---|
   | au plafond source | 44 | 14 % | **1,26** |
   | sous le plafond | 281 | 86 % | 0,84 |

   Le plafond vaut `source_line_height / 0,75 × 1,05` où `source_line_height`
   est la hauteur du POLYGONE OCR — soit 1,23 fois l'encre. Il surestime donc le
   corps d'origine d'environ un quart, et le texte sort trop gros partout où il
   mord. Les séries à texte court l'atteignent souvent (the_cleaner 49 %,
   rise-of-the-dragon 41 %), celles à texte long jamais (i-married 0 %,
   path-of-vengeance 0 %).

   **L'écart entre séries n'était donc pas un besoin de réglage par série, mais
   l'artefact bimodal d'un plafond mal calibré.**

Correctif : `source_line_height` est désormais mesurée sur l'ENCRE
(`chirurgical_mask`, disponible sur 325/325 zones) par projection sur les
lignes, avec repli sur l'ancienne mesure si le masque manque. Une seule ligne
change de sens, et elle se propage à tous les points qui s'en servent.

Mesuré, `compare actuel corps-encre` :

| | avant | après | |
|---|---|---|---|
| bulles conformes | 93 (29 %) | **133 (41 %)** | +40 |
| `corps_gros` | 70 | **6** | −64 |
| `empreinte` | 156 | **92** | −64 |
| `corps_petit` | 90 | 74 | −16 |
| `cap_ratio` p90 | 1,42 | **1,18** | |
| `footprint_ratio` p75 | 1,76 | **1,49** | |

**Réparées 40, cassées 0.** Les +3 / +2 / +1 sur décentrage et débordement
portent tous sur des bulles déjà fautives par ailleurs.

Le décompte « cassées 0 » est vrai mais masque les ÉCHANGES de défaut : une
bulle déjà fautive qui change de faute reste fautive. Comptabilité complète des
transitions de corps, cible [0,80 – 1,25] :

| avant → après | n |
|---|---|
| OK → OK | 163 |
| **gros → OK** | **64** |
| **petit → OK** | **18** |
| petit → petit (inchangé) | 71 |
| gros → gros | 5 |
| OK → petit | 2 |
| échange gros ↔ petit | 2 |

**Corps dans la cible : 165/325 (51 %) → 246/325 (76 %)** une fois T2 posé.
82 bulles réparées, 3 dégradées — et les 3 relèvent de D1, pas du changement
de mesure.

### M4 — Le critère de corps ne peut pas être `line_h_ratio`
Corollaire de T1 : `line_h_ratio` a `source_line_height` au dénominateur, donc
il change de sens dès qu'on touche à la façon de mesurer la source. Comparer
deux runs avec lui, c'est comparer deux barèmes. Le verdict se prononce
désormais sur `cap_ratio` (pixels des deux côtés, aucune dépendance au
renderer), avec un seuil haut ET un seuil bas — le défaut « trop gros » existe
autant que « trop petit », et n'était pas compté jusqu'ici.

Les verdicts sont maintenant **recalculés à la lecture** du run : changer un
seuil ne demande plus de renoter, et deux runs se comparent toujours au même
barème.

### Reste à traiter, par ordre de poids

1. `empreinte` 92 et `corps_petit` 74 — les 86 % de bulles SOUS le plafond, où
   c'est l'ajustement qui mord. Le levier est le découpage de lignes :
   `wrap_text` est un remplissage glouton, jamais équilibré, et **25 % des blocs
   multi-lignes finissent sur un mot orphelin**.
2. `erase_spill` 41 (13 %), concentré sur i-married (19). En hausse par rapport
   au code du 14 août (30) — à vérifier, possible régression des commits récents.
3. Les quatre défauts hors périmètre ci-dessus (filigranes, logos, troncature,
   ligne dupliquée).

### T2 — Garde-fou : la projection d'encre fusionne les lignes qui se touchent
`pipeline.py::_prepare_render_style`. Sur « START GAME> » (hellogin), la
projection rendait **101 px** là où les polygones en donnent 48 : deux lignes de
grandes capitales qui se touchent forment une seule bande. Le plafond MONTAIT au
lieu de descendre. L'encre d'une ligne ne pouvant pas être plus haute que le
polygone qui la borne, on prend `min(encre, polygone)` — sans effet dans le cas
normal (le rapport encre/polygone mesuré vaut 0,96 au p05, 1,00 au p50), correctif
dans le cas fusionné.

Portée réelle : actif sur **8 bulles sur 325**. « START GAME> » repasse de 101 à
48 px de plafond, et de `cap_ratio` 0,35 à 1,00. Sur le corpus entier le corps
dans la cible passe de 75 % à 76 % — modeste, mais gratuit et juste. Les deux
autres bulles dégradées par T1 ne bougent pas : elles relèvent de D1.

### D1 — Diagnostic : le choix de ROUTE de rendu n'est pas monotone
**Non corrigé — décision à prendre, le correctif est structurel.**

En traçant les 3 bulles dégradées par T1, on tombe sur bien plus gros :

| bulle | plafond | police | |
|---|---|---|---|
| `i-married #22` | 56 → **45** (baisse) | 28 → **59** (double) | |
| `30-years #15` | 48 → **37** (baisse) | 28 → **58** (double) | |
| `rise-of-the-dragon #7` | 27 → **22** (baisse) | 37 → **13** (s'effondre) | |
| `hellogin #4` | 48 → **101** (monte) | 75 → **42** (baisse) | |

La taille finale ne varie pas de façon monotone avec le plafond, et elle peut
varier d'un facteur 2 pour un plafond qui bouge de 20 %.

Cause, `core/renderer.py::insert_text` : deux portes décident de la ROUTE de
rendu, pas seulement de la taille —
- `exact_lines_ok` (ligne ~3242) : rendu ligne-à-ligne dans les polygones OCR
  plutôt que remise en page ;
- `anchor_box` (ligne ~3292) : mise en page ancrée sur les polygones plutôt que
  dans la boîte entière.

Les deux testent `probe.size < 0.6 * cap` où `cap = source_line_height/0,75 × 1,05`.
**Le seuil est proportionnel à la grandeur qu'il teste.** Baisser le plafond
abaisse le seuil d'autant, la porte devient plus facile à passer, et le rendu
CHANGE DE ROUTE — d'où des sauts de taille sans rapport avec l'amplitude du
changement de plafond.

C'est la vraie racine de l'instabilité du corps : ce n'est pas un réglage à
ajuster, c'est une décision binaire prise sur un critère qui bouge avec son
propre référentiel. Deux directions possibles, à trancher :
1. Rendre le seuil ABSOLU — comparer `probe.size` à la taille d'origine mesurée,
   pas à un `cap` qui en dérive.
2. Supprimer la bascule et rendre les deux routes continues (choisir la mise en
   page qui minimise un coût, au lieu de basculer sur un seuil).

Le barème donnera le verdict dans les deux cas : `compare` nomme les bulles qui
basculent, et la planche `sheet --vs` les montre.

### Limite connue du barème — `erase_spill` sur-compte les `out_text`
Le critère mesure les pixels repeints par l'effacement HORS de l'encre source
dilatée. Or E3 donne délibérément aux cartouches `out_text` un masque au BLOC
(polygones de ligne dilatés de 0,30 × hauteur de ligne) pour attraper la lueur
externe. Ce débordement est donc voulu, et le barème le compte comme bavure :
**29 des 41 zones signalées sont des `out_text`**, avec 10 000 à 23 000 px
repeints — cohérent avec un masque au bloc, pas avec un accident.

Les 10 `bulle` restantes, elles, sont à regarder : c'est là que vit le défaut E5
(contour de ballon entamé). Correctif du barème à faire : dilater la référence
selon la classe (0,30 × hauteur de ligne pour `out_text`, quelques px pour
`bulle`) au lieu du noyau fixe de 11 px.

## Phase « routage par coût continu »

### D1 (correctif) — les portes binaires deviennent un coût
`core/renderer.py::_route_cost`. Trois routes explicites, de la plus fidèle à la
planche à la plus libre : `exact_lines` (redessine dans les polygones OCR,
conserve les coupures d'origine), `anchor` (remise en page bornée à l'enveloppe
des polygones), `box` (remise en page dans la boîte ou la forme du ballon).

```
coût = |ln(taille / taille_source)| + infidélité[route] + rag + 2,0 si ça ne tient pas
```

Le logarithme rend l'écart **symétrique** : rendre au double ou à la moitié du
corps de la planche coûte pareil — ce que ne faisait aucun seuil, tous
unilatéraux. L'infidélité s'ajoute dans la même unité, ce qui rend l'arbitrage
lisible et réglable : `0,10` se lit « on accepte de perdre 10 % de corps pour
garder les coupures de la planche ».

Point de méthode : la route `box`, toujours disponible, est chiffrée AVANT les
autres et leur sert de référence. C'est ce qui supprime l'auto-référence — chaque
route se juge contre une alternative réelle, plus contre un seuil dérivé
d'elle-même.

**Effet mesuré : quasi nul.** 7 bulles sur 325 changent de route, +3
`corps_petit`, 0 conforme gagnée. Ce n'est pas un échec du modèle : l'usage
d'`exact_lines` reste conditionné en aval par
`not is_bubble and container is None`, donc pour une vraie bulle avec forme de
ballon, `box` est la seule route possible quoi que dise le coût. **Le routage ne
peut arbitrer que ~24 % des zones.** La route `anchor` est quasi morte (1 bulle).

Le correctif reste juste et supprime l'effet de falaise mesuré en D1 ; il fallait
simplement constater que le gros du problème n'était pas là.

### Deux impasses, notées pour ne pas les refaire

1. **Le balayage des constantes d'infidélité est dégénéré.** Minimiser « l'écart
   de corps de la route choisie » en faisant varier l'infidélité revient à
   optimiser l'objectif en supprimant le seul terme qui n'est pas l'objectif :
   il répond mécaniquement « infidélité = 0 ». Ces constantes ne se calibrent pas
   sur une mesure de taille — il leur faut un terme de qualité indépendant.

2. **La pénalité de césures placée dans le coût de ROUTE est inerte.** Résultats
   identiques au bit près, même répartition de routes. Conservée parce que le
   modèle est juste, mais elle ne mord pas tant que le mix de routes est
   contraint. À ne pas prendre pour un acquis.

### C1 — Rééquilibrage des lignes, là où le défaut est fabriqué
`core/renderer.py::_rebalance_lines`, appelé depuis `_layout_at_size`.

Les défauts restants ne vivent pas dans le choix de route mais DANS la route
`box` : **79/90 `empreinte`, 52/70 `corps_petit`, et 67 des 71 orphelins**.
`wrap_text` et `_wrap_text_by_mask` remplissent gloutonnement — chaque ligne
prend tout ce qu'elle peut et le reliquat tombe sur la dernière.

`_layout_at_size` est le point unique où les DEUX découpeurs produisent leurs
lignes, et où la largeur allouée à chacune est connue (pour une bulle, la largeur
du ballon à la hauteur de cette ligne). On y déplace des mots entre lignes
voisines tant que ça rapproche les largeurs, en refusant tout dépassement de la
largeur allouée.

**Le nombre de lignes ne bouge jamais** : la hauteur du bloc, et donc la taille
retenue par la dichotomie, restent valides. Vérifié sur le corpus — 0 bulle
change de nombre de lignes, 0 change de taille de police. C'est ce qui permet de
poser ce rééquilibrage sans toucher à la recherche de taille.

Mesuré, `compare routage-rag cesures` :

| | avant | après |
|---|---|---|
| **orphelins** | 71/282 (25 %) | **46/282 (16 %)** |
| `rag_cv` p50 | 0,194 | **0,163** |
| `empreinte` | 90 | 88 |
| bulles conformes | 133 | 135 |

**25 orphelins résolus, 0 nouveau.**

### Barème — deux corrections
- **L'orphelin est désormais un défaut COMPTÉ.** Il n'apparaissait qu'en bas de
  rapport : un pavé finissant sur un mot seul passait pour « conforme ».
- **L'équilibre se mesure en PIXELS.** `rag_cv` comptait des caractères alors que
  `_rebalance_lines` optimise des largeurs d'encre ; les deux divergent dès que
  la largeur moyenne des glyphes change d'une ligne à l'autre. Le renderer expose
  `line_widths_px`, le barème s'en sert, avec repli sur le décompte de caractères
  pour les runs antérieurs au hook.

### Trajectoire, tous les runs rejugés au même barème

| étape | conformes | orphelins | corps dans la cible | `rag_cv` |
|---|---|---|---|---|
| départ (`b9b7e38`) | 80 (25 %) | 72 (25 %) | 165 (51 %) | 0,192 |
| + T1/T2 plafond sur l'encre | 108 (33 %) | 70 (25 %) | **246 (76 %)** | 0,194 |
| + routage par coût | 108 (33 %) | 71 (25 %) | 249 (77 %) | 0,194 |
| + rag dans le coût (inerte) | 108 (33 %) | 71 (25 %) | 249 (77 %) | 0,194 |
| + rééquilibrage des lignes | **121 (37 %)** | **46 (16 %)** | 250 (77 %) | **0,163** |

## Phase « le budget de lignes devient un arbitrage »

Corpus étendu à **16 planches / 632 bulles** (2 par série). La `part01` porte le
titre et les crédits, donc elle sur-représente les cartouches `out_text` ; la
`part02` apporte le dialogue dense qui manquait.

### Diagnostic — d'où viennent vraiment les `corps_petit`
`_fit_font_hard` enregistre désormais quelle contrainte a bloqué la taille juste
au-dessus de celle retenue (hauteur, largeur, ou les deux). Instrumentation
purement additive, posée dans la référence elle-même.

Sur les **141 bulles `corps_petit`** :

| origine | n | |
|---|---|---|
| **rattrapage `target_lines`** | **84 (60 %)** | aucun blocage relevé, et 4/84 seulement au plafond |
| bloquées en hauteur | 37 | dont **29 refusent la taille supérieure pour exactement UNE ligne de plus** |
| bloquées en largeur | 19 | |

Le premier levier n'était donc ni la programmation dynamique ni le routage :
c'était le budget de lignes, une contrainte DURE. Dès que le bloc dépassait
`lignes_source + 1`, le code redescendait la taille jusqu'à rentrer dedans, à
n'importe quel prix — soit λ = ∞ sur le troc « corps de la planche contre
nombre de lignes de la planche ». `cap_ratio` médian de ces 84 bulles : 0,72.

### B1 — Le rattrapage devient un arbitrage chiffré
`core/renderer.py::_fit_font_hard`. Même unité que `_route_cost` :

```
coût = |ln(taille / taille_source)| + λ × lignes_au-delà_du_budget + rag
```

On descend en taille comme avant, mais au lieu de s'arrêter à la première mise
en page qui rentre dans le budget, on chiffre chaque candidate — **le bloc
initial compris** — et on garde la moins coûteuse. Balayage borné à 14 crans, et
arrêt dès que le budget est atteint (en dessous on ne gagne plus de lignes et on
perd du corps).

**Calibrage de λ** (`WEBTOON_PENALITE_LIGNE_SUP`), 632 bulles :

| λ | conformes | `corps_petit` | orphelins | corps dans la cible | `cap_ratio` p50 |
|---|---|---|---|---|---|
| ∞ (budget dur) | 298 (47 %) | 141 | 82 | 481 (76 %) | 1,000 |
| 0,45 | 298 (47 %) | 138 | 83 | 484 (77 %) | 1,000 |
| 0,25 | 305 (48 %) | 128 | 84 | 494 (78 %) | 1,000 |
| **0,12** | **314 (50 %)** | **114** | 87 | **508 (80 %)** | **1,000** |
| 0,06 | 314 (50 %) | 111 | 91 | 511 (81 %) | 1,032 |

0,12 est le genou : 0,06 ne gagne plus de conformes, ajoute 4 orphelins et fait
dépasser la médiane du corps. Retenu.

Sur les 46 bulles qui gagnent une ligne : corps **+0,22** en médiane, et
**17/46 → 44/46 dans la cible**.

**Coût de calcul inchangé** — 16 découpages et 976 mesures de largeur par bulle
avant comme après, 174 ms contre 182 ms. L'arbitrage REMPLACE l'ancienne
descente, qui pouvait aller jusqu'au plancher ; la nouvelle est bornée.

### M5 — Deux critères du barème étaient FAUX, et ils ont été retirés
Le premier verdict de B1 annonçait **−11 conformes** : `empreinte` +22 et
`trop_de_lignes` 0 → 46. Vérification sur les crops avant de conclure — les
métriques avaient tort.

| bulle | avant | après | source |
|---|---|---|---|
| `frontier p01 #59` | cap 0,67 · empr 0,84 | cap **1,11** · empr **2,17** | 2 lignes |
| `i-married p02 #7` | cap 0,74 · empr 0,94 | cap **1,06** · empr **1,70** | 3 lignes |
| `frontier p01 #46` | cap 0,76 · empr 0,88 | cap **1,12** · empr **2,09** | 2 lignes |

Le corps entre dans la cible ET l'empreinte double, parce que c'est
arithmétique : à `cap_ratio` égal, avec une police 11 % plus large par caractère
(0,80 contre 0,72 pour le studio, déjà mesuré), le même texte prend plus de
largeur, donc plus de lignes, donc plus de surface.

`empreinte` et `trop_de_lignes` demandaient au pavé rendu de **reproduire la
géométrie du pavé source**. C'est impossible à corps correct dès que la police
diffère : ils pénalisaient la justesse.

Retirés du verdict — non pas parce qu'ils accusaient le changement, mais parce
qu'ils sont FALSIFIÉS : cinq cas mesurés où la métrique se dégrade pendant que le
rendu s'améliore visiblement, avec un mécanisme arithmétique vérifiable
indépendamment. Ils restent mesurés et affichés, un écart énorme restant un
signal ; ils ne décident plus de la conformité.

Le verdict repose désormais sur des critères **insensibles à la police** :
effacement (bavure, fantôme), corps (`cap_ratio`, deux bornes), centrage,
débordement hors ballon, orphelin.

### La programmation dynamique est écartée, et pourquoi
Elle était le plan annoncé. Deux constats l'ont invalidée avant écriture :
1. `allowed[k]` ne dépend pas que de l'indice de ligne — le bloc étant centré
   verticalement, la bande mesurée dépend du nombre TOTAL de lignes. Le code
   résout ça par un point fixe. Une PD devrait tourner une fois par nombre de
   lignes candidat.
2. Surtout : **à largeurs données, le glouton minimise déjà le nombre de
   lignes**. Une PD ne peut donc pas rendre le texte plus gros — or c'est le
   nombre de lignes qui plafonne la taille. Elle n'aiderait que l'équilibre, que
   `_rebalance_lines` traite déjà à moindre risque sur une fonction portant
   quatre correctifs de monotonie documentés.

Piste restante pour les 29 cas « une ligne de trop » : forcer un bloc plus court
élargit les bandes dans un ovale (cycle vertueux que le point fixe ne cherche
jamais, puisqu'il ne descend pas en dessous de sa convergence).

## Phase « centrage vertical et fiabilisation du barème » (corpus 16 planches)

Défaut n°1 au barème corrigé : `decentre_y`, 127 bulles, jamais diagnostiqué.

### V1 — Offset vertical d'encre non compensé dans `_draw_exact_lines`
`core/renderer.py`. Ce chemin calculait l'ordonnée par `yp = y1 + (rh - line_h)//2`
où `line_h` est une hauteur d'ENCRE (`getbbox[3]-getbbox[1]`), alors que
`draw.text()` place son `y` au haut du CADRATIN. Chaque ligne était donc dessinée
`getbbox[1]` px trop bas. L'offset HORIZONTAL, lui, était déjà compensé
(`- offset_x`) : c'était une asymétrie, pas un choix.

Mesuré par police : **Allegre Sans (celle des `out_text`) a un offset d'encre de
20–23 % du corps** (5–9 px aux tailles usuelles), contre 0–5 % pour CCWildWords.
D'où `exact_lines` pesant 68 des 127 décentrages alors qu'il ne fait que 19 % du
corpus, et 78 des 127 décalages vers le bas. Compensé par `- offset_y`.

### V2 — Ancrage sur la bande d'encre SOURCE
`pipeline.py` mesure désormais, pour chaque ligne source, la bande d'encre réelle
(`ink_y0`/`ink_y1`) dans son polygone OCR — disponible sur 100 % des `out_text`.
`_draw_exact_lines` aligne la ligne rendue sur cette bande plutôt que de la centrer
dans le polygone, qui contient de la place de jambage inutilisée par des capitales
et remontait le texte. Écart médian `exact_lines` : 8,2 → **2,2 px**.

`decentre_y` : **127 → 85**. Pour la route `box`, aucune correction : le renderer
y centre dans sa zone à 2,5 px près (p90 4,5), l'écart résiduel venant de ce que la
planche elle-même ne centrait pas son texte.

### Barème — trois corrections de fiabilité

**`residu_pct` — le texte source encore VISIBLE.** `ghost_contrast` comparait deux
médianes, aveugle à un résidu partiel : sur `30-years p01 #14`, 70 % du texte
anglais restait visible et le critère ne bronchait pas. Nouvelle mesure : part de
l'encre source qui, sur l'image FINALE, reste sombre HORS du texte rendu. La
nuance « hors du texte rendu » est essentielle — un `out_text` est réinjecté à la
même position, donc un résidu d'effacement y est RECOUVERT et invisible (mesuré
sur `path-of-vengeance p02 #6` : résidu brut 86 %, rendu final impeccable). Après
recentrage sur le visible : 17 vrais résidus.

**`erase_spill` sensible à la classe.** E3 donne délibérément aux `out_text` un
masque au BLOC (polygones dilatés de 0,30 × hauteur de ligne) pour absorber la
lueur externe du texte d'impact. Compter ce débordement voulu comme bavure
faisait ressortir 29 `out_text` sur 41. La zone légitime est maintenant élargie à
ce bloc pour cette classe : **80 → 27** signalements, tous de vraies `bulle`.

**`bubble_overflow` — deux faux positifs.** Le crop élargi de 60 % contient le
texte des bulles VOISINES, compté comme encre de cette bulle hors de son ballon ;
exclu désormais. Et la couronne d'échantillonnage incluait le trait du ballon,
gonflant σ donc la tolérance — l'« intérieur » de `rise p01 #16` couvrait tout le
crop. Remplacé par un écart robuste (MAD). **37 → 18**.

### Bilan RENDU pur (départ `95b9e6f` vs final, MÊME barème final)

Isolé de l'effet barème en re-notant les deux rendus avec le code de mesure final :

| | départ | final |
|---|---|---|
| **bulles conformes** | 338 (53 %) | **366 (58 %)** |
| `decentre_y` | 127 | **85** |
| `corps_petit` | 114 | 107 |
| `residu` | 22 | 17 |
| `debordement` | 20 | 18 |

**+28 conformes**, aucune régression visuelle. Les deux `corps_gros` « nouveaux »
sont soit une amélioration mal étiquetée (`frontier #32` : cap 0,45 → 1,30, sortie
du trop-petit), soit un rendu visuellement correct dont la vraie hauteur d'encre
n'était révélée que par le bon placement (`30-years #15`).

## Bilan visuel et défauts flagrants résiduels (2026-08-28)

Après les correctifs de la journée (corps calé sur l'encre, routage par coût,
rééquilibrage, centrage vertical), inspection de planches COMPLÈTES rendues pour
juger la qualité perçue, que le barème ne capture qu'indirectement.

**i-married-the-dragon (pire série au barème, ~40 % conforme) rend en fait à un
niveau quasi-professionnel** : texte centré, couleurs d'origine préservées
(bordeaux, blanc sur cartouche sombre), cartouches nets, effacement propre. Le
barème strict (58 % global) sous-estime la qualité perçue : la plupart de ses
« défauts » (corps un cran petit, décentrage < 10 %, orphelin) sont
imperceptibles à l'œil.

### Défauts flagrants résiduels, tous RARES et HORS du scope « rendu »

Ce sont eux qui trahiraient l'automatisation, mais ils sont peu nombreux et leur
cause est en amont du rendu :

1. **Texte hors ballon** (~2-3 bulles) — `hellogin p02 #41` (texte 150 px sous
   sa bulle), `p01 #16` (texte à droite du ballon). Diagnostiqué : la **bbox de
   détection ne couvre pas le ballon** (décalée), le texte source lui-même est
   hors de la bbox. Le centroïde de `mask_binary` tombe au centre de la bbox
   (donc pas sur le vrai texte), et un recentrage dessus EMPIRE le cas (essayé,
   mesuré, annulé). Corrigeable seulement côté DÉTECTION (YOLO), pas rendu.

2. **Effacement tronqué** (~5-8 bulles, 1 %) — `30-years p01 #14` : le masque
   d'effacement s'arrête à x≈432 alors que l'encre va à ~600, laissant
   « RELATIONSHIP / CHARACTERS / WORK » (les fins de ligne) non effacées, qui
   transparaissent sous le nouveau texte. Le seuillage Otsu par ligne rate le
   texte noir sur la partie où le fond passe du blanc au beige sombre (arche).
   Mesuré : `chirurgical_mask` du segmenter est tronqué de la même façon.
   Un Otsu reconstruit capterait 98 %, mais l'effacement utilise
   `chirurgical_mask` (segmenter). Fix côté segmentation, hors rendu ; et 1 % de
   cas ne justifie pas de risquer les 99 % via un changement de masque.

3. **Filigranes et logos** (VORTEXSCANS.COM, logo « THE CLEANER ») — hors
   périmètre convenu.

**Conclusion** : le rendu est à un niveau quasi-professionnel sur la masse. Les
défauts bloquants pour un « 100 % pro » relèvent de la détection de boîte et de
la segmentation (amont), déjà identifiées comme PRIORITÉ 0 d'une session
antérieure et non résolues — un chantier distinct du travail de rendu.

### Métrique `residu` — faux positifs restants (barème)
Sur les 6 pires `residu`, 2 vrais défauts (texte dupliqué, hors ballon), 1 SFX
mineur, et **3 faux positifs** : textes à GLOW coloré (néon) dont le halo
résiduel après effacement déclenche la mesure alors que le rendu final est
propre. Distinguer un glyphe résiduel lisible d'un halo diffus demanderait une
mesure de structure (densité de bords) ; non fait, la métrique reste indicative.

## C2 — L'arbitrage de taille absorbe aussi les orphelins
`core/renderer.py::_fit_font_hard`. L'arbitrage par coût introduit pour le budget
de lignes (B1) ne se déclenchait que sur un DÉPASSEMENT de budget. Or un lettreur
qui voit un mot seul sur la dernière ligne descend d'un cran pour l'absorber —
exactement le compromis que le coût sait trancher, puisque `_rag_penalty` pénalise
déjà l'orphelin.

Déclenchement élargi : budget dépassé OU orphelin présent. La descente est bornée
à 8 crans (un orphelin coûte ~0,15-0,30, soit 2-4 crans de corps ; au-delà la
perte de corps l'emporte) et s'arrête deux crans après résorption.

Mesuré, `compare final orphelin2` :

| | avant | après |
|---|---|---|
| **orphelins** | 87 | **64** |
| bulles conformes | 366 (58 %) | **383 (61 %)** |
| `corps_petit` | 107 | 108 |
| `cap_ratio` p50 | 1,023 | 1,000 |

**23 orphelins résolus, 1 seule régression `corps_petit`.** Vérifié à l'œil :
« IF YOU'RE OKAY WITH TELLING IT. » passe de 4 lignes (« IT. » orphelin) à 3
lignes équilibrées ; « IS SOMETHING WRONG, ARAM? » de 3 lignes à 2. Le texte est
parfois un cran plus petit mais nettement mieux réparti — le geste du lettreur.

## L1 — Texte hors ballon : container faux rejeté (confort de lecture)
`core/renderer.py::insert_text`. Le défaut le plus destructeur pour la lecture —
le texte rendu HORS de sa bulle, dans le vide, l'œil qui le cherche.

Diagnostic sur hellogin p02 #41 (« WAIT… DID I FALL ASLEEP…? ») : `_container_box`
attrapait le FOND BLANC de la case (un aplat uni de 689 px de large, centré 176 px
SOUS le texte) comme « container », et ce container remplaçait la bbox de la
détection. Le texte, centré dans ce faux container, sortait de sa bulle. Ce
n'était donc ni la bbox de détection ni le masque de forme (deux fausses pistes
écartées, mesurées), mais le container.

Correctif : un container n'est retenu que s'il est CENTRÉ sur le texte source —
son centre vertical ne doit pas s'écarter des polygones OCR de plus d'une hauteur
de texte. Un vrai cartouche est serré sur son texte (écart quasi nul) ; un aplat
de fond happé par erreur est centré ailleurs (176 px ici) et se fait rejeter.

Mesuré : `dy` de #41 passe de **+153 px à +15 px** (texte dans la bulle), et sur
les 632 bulles **une SEULE bouge** — exactement le cas pathologique. Zéro
régression (decentre_y −1, debordement −1, aucune bulle cassée). Vérifié à l'œil.

## L2 — Effacement tronqué : détecteur de résidu hors-masque, essayé et ÉCARTÉ
Le second défaut de confort de lecture après le hors-ballon : du texte anglais
résiduel visible (30-years p01 #14, « RELATIONSHIP / CHARACTERS / WORK » à droite
des lignes).

Diagnostic confirmé : les polygones OCR — et donc `chirurgical_mask` et
`block_mask` — s'arrêtent à x≈432 alors que l'encre va à ~600. Le seuillage rate
le texte noir là où le fond passe du blanc (fenêtre) au beige (arche). Les deux
détecteurs de la 2e passe (`_erasure_failed`, `_ghost_remains`) regardent DANS le
masque, où tout est propre ; le résidu vit HORS du masque.

Essayé : (1) un détecteur `_residual_outside_mask` cherchant l'encre dans
`block_mask` mais hors du masque d'effacement, pour déclencher la 2e passe ;
(2) un rattrapage par contenu élargissant les bandes de lignes en X.

**Écarté, mesuré.** Le rattrapage ne résout pas #14 : `interior_limit` (intérieur
du ballon) exclut la zone droite, car #14 est un cartouche de narration sur DÉCOR
(arche + fenêtre) mal classé `bulle`, sans « intérieur » à droite. Et le détecteur
seul, au barème : residu 17 → 17 (aucun cas résolu) et erase_spill 27 → 28 (une
bulle sur-effacée). Effet net négatif. Les deux annulés.

La cause racine est l'OCR qui tronque ses polygones sur fond contrasté — un
problème de détection de texte, en amont du rendu. ~5-8 cas sur 632 (1 %). Non
traité côté rendu : le risque sur les 99 % dépasse le gain sur 1 %.

## Bilan confort de lecture (demande utilisateur)

Priorité recadrée par l'utilisateur : le CONFORT DE LECTURE (lire sans s'arrêter,
ne pas casser le rythme). Ce sont les défauts FLAGRANTS qui cassent le rythme, pas
le corps un cran petit.

- **Texte hors ballon** (le pire — l'œil cherche le texte) : CORRIGÉ (L1, container
  faux rejeté). hellogin p02 #41 : texte remis dans sa bulle.
- **Taille lisible** : médiane 36 px, 90 % du texte ≥ 20 px, 8/632 seulement sous
  16 px. Les `cap_ratio` très bas sont des out_text stylisés (source énorme) dont
  le rendu reste gros et lisible (fs 35-80). Pas de problème de lisibilité de masse.
- **Lignes équilibrées** (fluidité) : orphelins 25 % → 10 % sur la session.
- **Effacement propre** sauf ~5-8 cas d'OCR tronqué (L2, hors rendu).
