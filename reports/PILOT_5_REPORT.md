# Pilot 5 images — rapport QA (SAM2 CPU réel)

> **Transparence (obligatoire).** Ce pilote a été exécuté par l'agent IA : SAM2 (CPU) réel a produit
> chaque proposition, l'agent a placé les points par pan et a inspecté visuellement chaque overlay.
> **Aucun chiffre n'est fabriqué** (tous mesurés). Les masques montrés sont exactement ce que SAM2 a
> renvoyé. **La validation finale conforme à la spec (précision au niveau de la gouttière/eave) reste
> une étape humaine** — conformément au human-in-the-loop et à PROJECT_SPEC.md §"Consistency beats cleverness".
> Le portail STOP est respecté : **aucune annotation de masse, aucun checkpoint 50/100/150/200, spec inchangée.**

Preuves visuelles : `/app/reports/pilot/overlays/` (combinés `*_00_COMBINED.jpg`, par pan, `*_FIX_*` = points repositionnés).
Masques binaires bruts : `/app/reports/pilot/masks/`. Métriques : `pilot/sam2_metrics.json`, `pilot/sam2_fix_metrics.json`.

## 1. Les 5 images du pilote
| ID | Batch | Split | EXIF | Taille affichée | Type de toit |
|----|-------|-------|------|-----------------|--------------|
| 0001 | batch1 | train | aucun | 1536×1152 (paysage) | Tuiles barrel méditerranéennes, croupes + tourelle |
| 0169 | batch2 | val | 6 (rot 90°) | 3024×4032 (**portrait**) | Moderne toits plats + parapets (cas difficile) |
| 0050 | batch1 | train | aucun | 1376×768 (paysage) | Tuiles, pignon 2 niveaux + croupe garage |
| 0138 | batch1 | holdout | aucun | 1536×831 (paysage) | Croupe tuiles 1 niveau, cadrage large |
| 0173 | batch2 | train | 1 (aucun) | 5712×4284 (paysage) | Tuiles, tourelle 2 niveaux + entrée |

## 2. Performance SAM2 CPU (mesurée, hiera-tiny)
| ID | Encode (embedding) | Predict / point après cache | Points | Cache embedding |
|----|--------------------|-----------------------------|--------|-----------------|
| 0001 | 3780 ms | 154 / 94 / 84 ms | 3 | ✅ 1 encode → N predicts |
| 0169 | 3732 ms | 294 / 155 / 163 / 217 ms | 4 | ✅ |
| 0050 | 3487 ms | 104 / 73 / 84 ms | 3 | ✅ |
| 0138 | 3488 ms | 150 / 87 / 93 ms | 3 | ✅ |
| 0173 | 3710 ms | 494 / 500 / 397 / 408 ms | 4 | ✅ |

- **Encode ≈ 3,5–3,8 s / image (une seule fois)**, puis chaque clic supplémentaire = **0,07–0,5 s**.
- Le **cache d'embedding fonctionne** : un seul encode par image, tous les points suivants réutilisent l'embedding.
- Coût CPU acceptable en usage interactif (le 1er clic sur une image attend l'encode, les suivants sont instantanés).

## 3. Vérification coordonnées / orientation (test critique)
Pour **chaque** image, le point cliqué visuellement dans un pan tombe **exactement** au même pixel physique
dans l'original (overlays combinés à l'appui). Vérifié :
- **0169 (portrait, EXIF=6, 4032×3024 → affiché 3024×4032)** : EXIF appliqué correctement, points alignés,
  masque aux dimensions affichées, overlay aligné. **RÉGRESSION ORIENTATION : PASS.**
- **0173 (5712×4284, EXIF=1)** : alignement parfait à pleine résolution. **PASS.**
- `db_size == display_size` pour les 5 (masques générés dans l'espace EXIF-appliqué = pixels vus par l'annotateur).
- **Aucun problème de coordonnées/orientation détecté.**

## 4. Résultat par image (proposition SAM2 → correction)

### 0138 — MEILLEUR CAS ✅
- 3 pans (croupe gauche, petit toit central, croupe droite garage) tous **captés proprement**.
- Bord bas du masque = bas des tuiles ≈ ligne de drip edge/eave. Voisin à droite **non** inclus.
- Correction humaine attendue : trim fin du bord bas sur la lèvre de gouttière. **Faible.**

### 0173 — TRÈS BON ✅
- 4 pans (tourelle, croupe gauche, croupe basse garage, toit d'entrée droite). SAM2 capte bien les bandes de tuiles.
- La croupe basse (garage) et le toit d'entrée sont excellents ; la tourelle capte la bande de faîte.
- Correction humaine : étendre légèrement certains pans jusqu'au débord, préciser l'eave. **Faible à moyenne.**

### 0050 — BON (avec occlusion) ⚠️
- Pan **droit du pignon** + **croupe garage** : excellents.
- Point "pan gauche" 1re passe → tombé sur le **mur de pignon** (à EXCLURE par spec) ; repositionné 2e passe →
  tombé sur l'**arbre fleuri** qui occulte réellement le pan gauche.
- Constat : le pan gauche du pignon est **occulté par la végétation** (occlusion à documenter dans meta).
- Unités latérales G/D = **villas voisines** → correctement exclues (sujet = maison centrale).

### 0001 — BON après repositionnement ⚠️
- 1re passe : pan droit (garage) OK ; **tourelle** → point tombé sur le **mur** ; **aile gauche** → sur la **végétation**.
- 2e passe (points repositionnés sur la tuile) : tourelle ✅ et aile gauche ✅ corrigées.
- Enseignement clé : **le placement du point est déterminant** ; SAM2 segmente la texture sous le point.
  Un annotateur place le point mieux et ajoute des points négatifs → masque fiable. Voisin à gauche exclu.

### 0169 — CAS DIFFICILE / AMBIGU ❌ (SAM2 insuffisant seul)
- Maison **moderne à toits plats derrière parapets**. Depuis le sol on ne voit **quasi pas de pan de toit** :
  seulement des **arases de parapet** (blanc sur blanc/ciel), une **pergola à lames** et une **dalle en porte-à-faux**.
- SAM2 attrape des murs / une fenêtre / la pergola (faible pertinence). Coordonnées OK, mais **contenu inadapté**.
- Par spec : les toits plats derrière parapet se **délimitent le long du HAUT du parapet** (bord *supérieur*,
  suffixe `-parapet`, `"parapets": ["plane-XX"]`). Ici cela exige un **polygone manuel complet** + jugement.
- **Ambiguïté à remonter au client** (spec §"ask rather than deciding quietly") : la **pergola à lames**
  compte-t-elle comme pan à éclairer ? La dalle en porte-à-faux est-elle un plan parapet ?

## 5. Compte de pans (proposition) & sorties
| ID | Pans proposés | Masque fusionné | Remarque |
|----|----------------|-----------------|----------|
| 0001 | 3 | union des pans | tourelle + aile corrigées 2e passe |
| 0169 | 4 (dont pergola/dalle ambiguës) | union | nécessite polygones parapet manuels |
| 0050 | 2 fiables (+1 occulté) | union | pan gauche occulté (arbre) |
| 0138 | 3 | union | propre |
| 0173 | 4 | union | propre |

> Le masque fusionné n'est produit **qu'en sortie de commodité** (spec : les masques par pan sont l'artefact
> primaire ; le fusionné seul est inacceptable car porche/toit principal se recouvrent).

## 6. Échecs SAM2 observés (honnête)
1. **Point sur mur/végétation/fenêtre** → SAM2 segmente le mauvais objet (0001 tourelle/aile, 0050 gauche, 0169 mid-block). Remède : repositionner + points négatifs (démontré efficace).
2. **Toits plats modernes / parapets (0169)** → aucun pan de toit visible depuis le sol ; SAM2 seul inadapté, polygone manuel requis.
3. **Bord bas au niveau de la gouttière/eave** → SAM2 s'arrête au **bas des tuiles**, souvent proche mais **pas exactement** sur la lèvre de gouttière → **la correction humaine reste nécessaire** sur la seule frontière scorée.

## 7. Portail qualité — la vraie question
> « Un annotateur peut-il transformer de façon fiable la proposition SAM2 en un masque conforme à la spec
> (eave/pan de toit) ? »

**OUI pour les toits en pente à tuiles** (0001, 0050, 0138, 0173 — la majorité du dataset) : SAM2 fournit un
excellent point de départ, le bas de la bande de tuiles est proche de l'eave, la correction humaine se limite au
trim de la lèvre de gouttière et à l'exclusion mur/faîte. Le placement du point + points négatifs suffit.

**NON de façon automatique pour les toits plats modernes derrière parapet (type 0169)** : polygone manuel complet
le long du haut du parapet, plus une clarification client sur les éléments ambigus (pergola, dalle).

## 8. Recommandation
- **Le workflow est PRÊT pour l'annotation à l'échelle sur les toits en pente à tuiles** avec la boucle
  SAM2-CPU (proposition) → correction humaine (point/polygone/gomme, trim eave). C'est le régime dominant.
- **Traiter les toits plats/parapet (type 0169) comme une catégorie à part** : prévoir un mode polygone manuel
  systématique et une file "à clarifier avec le client" pour les cas ambigus, **avant** de les compter dans les checkpoints.
- **Étape humaine restante avant l'échelle** : valider dans le studio la précision eave/gouttière sur ces 5,
  statuer sur les ambiguïtés 0169, puis approuver. Je n'ai **pas** marqué les 5 comme "approuvés/spec-final"
  car la précision au niveau de la gouttière est le jugement humain qui définit ce portail.

## 9. Ce que je n'ai PAS fait (portail STOP respecté)
- Pas d'annotation des ~200 autres images. Pas de checkpoints 50/100/150/200. Spec non modifiée. Aucune métrique inventée.
