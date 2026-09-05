# Pilot 5 images — synthèse

> Observations issues **uniquement** des données réelles du pilote. **Aucun score de qualité inventé.**
> Statut global : propositions pilote assistées SAM2, **en attente de validation humaine + confirmation client**.

| Image | Fichier d'origine | Pans de toit | Observation principale | Confirmation client requise ? |
|-------|-------------------|--------------|------------------------|-------------------------------|
| 0001 | 0001.webp | 3 | Pan droit (garage) bien capté ; tourelle et aile gauche ont nécessité un repositionnement du point (1re passe sur mur/végétation). Voisin gauche exclu. | Oui |
| 0169 | b2-001.jpeg | 4 | **Cas spécial** : toits plats modernes derrière parapets ; depuis le sol quasi aucun pan visible (arases, pergola à lames, dalle en porte-à-faux). Éclairage type parapet (bord supérieur). Pergola/dalle = **ambiguës**. | **Oui (obligatoire)** |
| 0050 | 0059.webp | 3 | Pan droit du pignon + croupe garage bien captés ; pan gauche du pignon **occulté par un arbre** ; unités latérales = villas voisines (exclues). | Oui |
| 0138 | 0162.webp | 3 | Croupe gauche (plane-01) et croupe garage droite (plane-03) propres et bien séparées ; **plane-02 (centre) est un raté SAM2** (point sur l'entrée/le palmier) → correction manuelle nécessaire. | Oui |
| 0173 | b2-005.jpeg | 4 | Tuiles : tourelle, croupe gauche, croupe basse (garage) et toit d'entrée droite captés proprement. Pleine résolution 5712×4284, EXIF géré. | Oui |

**Total : 5 images, 17 masques de pans.**

## Constats transversaux (factuels)
- **Coordonnées / orientation** : vérifiées sur les 5 (dont 0169 portrait EXIF=6 et 0173 5712×4284) — le point cliqué correspond au même pixel physique ; masques aux dimensions affichées.
- **Placement du point déterminant** : SAM2 segmente la texture sous le point ; un point mal placé (mur/végétation/fenêtre) donne un masque erroné → repositionnement/points négatifs nécessaires (0001, 0138 plane-02, 0050 gauche, 0169).
- **Toits en pente à tuiles** : SAM2 fournit un très bon point de départ (0173, 0138 plane-01/03, 0050 droite/garage, 0001 droite).
- **Toits plats / parapets (0169)** : SAM2 seul insuffisant → polygone manuel + clarification client.

## Recommandation
Prêt à l'échelle sur les **toits en pente à tuiles** avec la boucle SAM2 → correction humaine. Traiter les
**toits plats/parapets** (type 0169) comme une file distincte (polygone manuel + clarification client) avant de
les compter. Aucune décision de labeling n'a été prise en silence.
