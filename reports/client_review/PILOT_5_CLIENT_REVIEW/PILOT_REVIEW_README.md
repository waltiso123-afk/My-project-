# Pilot 5 images — package de revue client

## Objet
Ce package regroupe **5 images pilotes** annotées afin d'obtenir votre **confirmation AVANT** de passer
à l'échelle (~200 images). Il vous permet d'inspecter les toits sans ouvrir l'outil d'annotation.

## Comment ces annotations ont été produites (transparence)
1. **SAM2 (assist)** a été utilisé par **prompt par points** : un point positif est placé dans chaque pan de toit,
   SAM2 propose une segmentation.
2. Une **correction humaine** est appliquée lorsque nécessaire (repositionnement de point, polygone, gomme).
   Pour ce pilote, plusieurs pans ont dû être repositionnés (voir `special_cases/` et les fichiers `*_FIX_*`).
3. **Les masques par pan sont les sorties PRIMAIRES.** Un masque PNG binaire (fond 0, pan 255) par pan de toit,
   aux dimensions de l'image affichée (orientation EXIF corrigée). Les pans ne sont **pas** fusionnés.
4. **Les masques fusionnés (`merged/`) ne sont fournis que par commodité** de visualisation — ils ne remplacent
   jamais les masques par pan.

## Statut (important)
Il s'agit de **propositions pilote assistées par SAM2, en attente de validation humaine finale et de votre
confirmation**. Aucune image n'est marquée « approuvée / conforme définitive ». **Aucune annotation de masse
ni entraînement de modèle (checkpoints 50/100/150/200) n'a été réalisé** — et rien de tel n'est présenté comme fait.

## Cas spécial — image 0169
0169 est une **maison moderne à toits plats derrière parapets**. Depuis le sol, on ne voit quasiment aucun pan
de toit : seulement des **arases de parapet**, une **pergola à lames** et une **dalle en porte-à-faux**.
- Selon la spec, les toits plats derrière parapet s'éclairent le long du **bord SUPÉRIEUR du parapet**.
- **Ambiguïtés à confirmer par le client** : la **pergola à lames** et la **dalle en porte-à-faux** comptent-elles
  comme des plans à éclairer ? **Aucune règle de labeling nouvelle n'a été décidée en silence.**
Voir `special_cases/0169/` (original, masques, overlay, `WHY_SPECIAL.txt`).

## Structure du package
- `originals/<id>/` — image d'origine **non modifiée** (format et nom d'origine).
- `planes/<id>/plane-0N.png` — masques binaires par pan (sortie primaire).
- `merged/<id>.png` — masque fusionné (commodité uniquement).
- `overlays/<id>_overlay.png` — original + pans colorés + contours (transparence pour voir le toit).
- `before_after/<id>_before_after.png` — comparaison Original / Annoté côte à côte.
- `special_cases/<id>/` — cas notables pour votre retour (0169 parapet, 0050 occlusion, 0001 repositionnement).
- `metadata/<id>.json` — métadonnées par image (dimensions, EXIF, pans, provenance SAM2, statut, notes/ambiguïtés).
- `PILOT_REVIEW_SUMMARY.md` — tableau de synthèse.

## Correspondance des identifiants
| ID pilote | Fichier d'origine | Split |
|-----------|-------------------|-------|
| 0001 | 0001.webp | train |
| 0169 | b2-001.jpeg | val |
| 0050 | 0059.webp | train |
| 0138 | 0162.webp | holdout |
| 0173 | b2-005.jpeg | train |

## Ce que nous attendons de vous
Confirmer, pan par pan, que la séparation des pans et les limites (notamment la **ligne de gouttière/eave**)
correspondent à votre attente, et **trancher les ambiguïtés de 0169**. Une fois validé, nous passons à l'échelle.
