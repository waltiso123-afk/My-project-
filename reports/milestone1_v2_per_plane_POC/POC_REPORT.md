# Milestone 1 v2 — POC de modélisation PER-PLANE (validation technique)

> **But** : décider de la *formulation de sortie* (architecture) capable de prédire des **plans de toit
> individuels** — pas un masque toit fusionné — et de produire correctement les rooflines
> **eave (bas)** / **parapet (haut)**, AVANT de louer un GPU et de lancer le checkpoint 150 img @ 768 px.
> **Read-only** : aucune annotation approuvée ni le holdout figé (13 img) n'ont été modifiés.
> **Les chiffres 256 px ci-dessous sont DIAGNOSTIQUES, pas des métriques d'acceptation.**

Script : `scripts/poc_perplane.py` · Résultats : `poc_results.json` · Visuels : `overlays/`.

## 1. Protocole
- 101 labels approuvés existants ; holdout figé identique (13 : 0016,0023,0031,0032,0037,0044,0051,0054,0058,0064,0085,0120,0138) ; 76 img train pour l'entraînement rapide.
- Instanciation → appariement **hongrois par IoU** prédiction↔GT (seuil IoU 0.25 = « plan récupéré »).
- Par plan apparié : frontière **eave (max-y/colonne)** pour un plan normal, **parapet (min-y/colonne)** pour un plan `-parapet` ; puis **erreur perpendiculaire moyenne / largeur** et **couverture horizontale**.
- 3 images portent des plans parapet (0020, 0021 = train ; 0169 = val) → 0 dans le holdout ⇒ la branche parapet est démontrée sur ces images (diagnostic).

## 2. Formulations comparées (256 px, CPU)
| Formulation | Récup. instances | IoU moyen apparié | Prédictions qui **fusionnent ≥2 plans** | Perp err (norm) | Couverture |
|---|---|---|---|---|---|
| **A — binaire + composantes connexes** (baseline_100) | **0.516** | 0.515 | 6 | 0.0121 | 0.835 |
| **A — binaire + watershed** | **0.629** | 0.494 | 8 | 0.0143 | 0.756 |
| **B — 3 classes bord-conscient** (fond/corps/joint, 8 ép.) | **0.397** | 0.454 | 8 | 0.0152 | 0.829 |
| **Mask2Former (instance seg)** | *non entraîné — analyse conceptuelle §4* | | | | |

### Lecture (l'essentiel n'est PAS le meilleur chiffre)
- **Aucune** formulation légère ne préserve fiablement les instances : au mieux **63 %** des plans GT sont récupérés, avec **6–8 prédictions qui fusionnent plusieurs plans** par lot de 13 images.
- **A/watershed** récupère plus d'instances mais **sur-découpe** des plans lisses (IoU et couverture chutent) → instabilité géométrique.
- **B/3 classes** n'a pas convergé en 8 époques à 256 px : le joint inter-plans est trop fin/sous-représenté ; il faudrait beaucoup plus d'entraînement et une résolution supérieure, sans garantie de séparer des plans qui partagent une arête (faîtage/noue).
- **Cause racine** : une sortie **sémantique** (binaire ou multi-classe) fusionne par nature les plans **adjacents qui se touchent** (même faîtage/même gouttière). Les visuels `overlays/0031_*.png` le montrent : deux versants distincts (rooflines GT vertes) sont prédits comme **une seule région**.

## 3. Branche parapet (démonstration)
- **Extraction GT du bord supérieur = OK** : sur 0169 (`overlays/0169_A.png`) la polyligne **rouge** longe correctement le **haut du parapet** (min-y/colonne), et le bord vert (eave) est utilisé pour les plans normaux ailleurs. Le sélecteur upper/lower piloté par le suffixe `-parapet` fonctionne.
- **MAIS la prédiction des plans parapet échoue** avec la sortie actuelle : le modèle binaire (entraîné sur toits en pente) ne détecte quasiment **rien** sur les toits plats modernes (0020, 0021 → aucun appariement ; 0169 → couverture ~0). Un toit plat n'a pas de « surface de versant » visible depuis le sol : c'est intrinsèquement une **classe/instance distincte**, pas un sous-cas du masque toit binaire.

## 4. Mask2Former — analyse conceptuelle (pas d'entraînement dans ce POC)
Segmentation **d'instances** (ou panoptique) : le modèle émet un **nombre variable de masques d'instance**, chacun avec une **catégorie**. Pourquoi c'est le bon choix ici :
- **Sortie = 1 masque par plan** → correspond **1:1** à notre format GT (`plane-0N.png`) : aucune conversion d'annotation.
- **Nombre de plans variable** (nos images : 2 à 8 plans) géré nativement — pas de canaux fixes ni d'ordre imposé (le pièges des formulations multi-classes/par-canal).
- **Catégorie par instance = {plan normal, plan parapet}** → alimente directement le sélecteur eave/parapet (le suffixe `-parapet` devient un label de catégorie d'instance).
- **Appariement hongrois prédiction↔GT** = à la fois son objectif d'entraînement *et* notre protocole d'évaluation per-plane → cohérence totale avec la métrique d'acceptation client (erreur perpendiculaire + couverture **par plan**).
- Coût : plus lourd que SegFormer-B0 → **nécessite le GPU** déjà prévu pour le 768 px (T4/L4/A10) ; poids pré-entraînés COCO disponibles (`facebook/mask2former-swin-tiny/small-*-instance`).

## 5. Recommandation pour le checkpoint 150 img @ 768 px
**PRINCIPAL — Segmentation d'instances Mask2Former** (backbone Swin-Tiny/Small, 2 catégories : `roof_plane`, `parapet_plane`), à 768 px sur GPU.
- **Compatible sans re-labelliser** : nos masques per-plane + drapeau `-parapet` = exactement des données d'instance seg (masques + catégorie).
- **Aligné sur la métrique client** : produit un masque par plan → notre extraction de roofline (haut si parapet, bas sinon) et l'erreur perpendiculaire/couverture **par plan** s'appliquent telles quelles (`scripts/per_plane_eval.py` réutilisé au niveau instance).
- **Sélection du checkpoint** : inchangée — erreur perpendiculaire per-plane MINIMISÉE (primaire), couverture horizontale (secondaire, départage) ; Dice/IoU = diagnostics.

**REPLI léger (si contrainte GPU/temps)** — SegFormer **3 classes bord-conscient** (§ formulation B) **davantage entraîné à 768 px** + post-instanciation. À réserver comme solution dégradée : ce POC montre qu'il ne sépare pas encore fiablement les plans adjacents ; ne pas l'adopter sur la seule foi d'un chiffre 256 px.

**À NE PAS retenir** : binaire + composantes/watershed (plafonne à ~63 % de récupération, fusions fréquentes, aucune prédiction des plans parapet) et multi-classe par canal fixe (ordre/nombre de plans instable).

## 6. Prochaines étapes (bloqué GPU)
1. Provisionner GPU (T4 16 Go min ; L4/A10 24 Go recommandé) — déjà requis par le 768 px.
2. Atteindre ~150 labels approuvés (101 aujourd'hui ; holdout figé conservé).
3. Fine-tuner Mask2Former-instance (2 catégories) à 768 px ; évaluer per-plane parapet-aware sur le holdout figé.
4. Vérifier surtout la **récupération d'instances** et la géométrie eave/parapet, pas seulement Dice/IoU.

> Intégrité : aucune métrique fabriquée ; tous les chiffres proviennent de `poc_results.json`. Aucune annotation ni le holdout n'ont été modifiés. Workflow Humain + SAM2 uniquement (aucun VLM).
