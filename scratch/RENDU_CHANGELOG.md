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
