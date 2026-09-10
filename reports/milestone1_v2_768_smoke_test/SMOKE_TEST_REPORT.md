# Milestone 1 v2 — 768px SMOKE TEST (Mask2Former instance-seg pipeline)

**SMOKE TEST ONLY — NOT FINAL ACCURACY.** Tiny subset, 1–2 epochs, CPU. Purpose = verify the
complete production pipeline runs end-to-end BEFORE any paid GPU time. No annotations, approved
outputs, or the fixed holdout were modified (read-only).

- **Date/heure** : 2026-09-10 11:42 UTC
- **Statut** : **PASS** (16/16 étapes)
- **Images de test** : **7** — train (4) : `0020`(parapet), `0001`, `0002`, `0009` ; eval (3) : `0169`(parapet), `0016`(holdout normal), `0021`(parapet)
- **Résolution** : **768 px**
- **Époques** : **2** (max)
- **Modèle / config** : `facebook/mask2former-swin-tiny-coco-instance`, **instance segmentation**, `num_labels=2` → catégories `{0: roof_plane, 1: parapet_plane}`, ~47.4 M paramètres, AdamW lr=1e-5, CPU, seed 0. Tête de classification ré-initialisée à 2 classes (attendu).

## Chemin de production testé (aucune voie simplifiée/mock)
| Étape | Résultat | Détail |
|---|---|---|
| dataset_integrity | PASS | approved=150 · frozen-13 holdout intact=True · leak groupe=NONE |
| load_dataset_768_perplane_gt | PASS | 7 imgs · masques GT per-plane @768 · 10 plans parapet au total |
| instance_conversion (GT per-plane → masques d'instance + class ids) | PASS | roof_plane/parapet_plane par instance |
| mask2former_init | PASS | 47.4M params, 2 catégories |
| forward + loss | PASS | pixel_values=(1,3,768,768), loss finie |
| backward | PASS | grad_norm fini (466.8) |
| optimizer_step | PASS | AdamW |
| train_2epochs + eval per epoch | PASS | 2 époques évaluées |
| inference_multiple_instances | PASS | jusqu'à 3 images renvoyant ≥2 instances |
| per_plane_matching (Hungarian IoU) | PASS | 20 lignes plan-apparié au total |
| normal_eave_boundary (bord bas) | PASS | extraction eave exécutée |
| parapet_upper_boundary (bord haut) | PASS | branche parapet exécutée (0169/0021, flag `-parapet` détecté) |
| roofline_error_and_coverage | PASS | erreur perpendiculaire + couverture calculées par plan apparié |
| checkpoint_selection_by_roofline_error | PASS | **sélection par MIN erreur perpendiculaire** (époque 2, perp=0.09355) — Dice/IoU NON utilisés |
| checkpoint_save | PASS | `model.safetensors`, `config.json` |
| checkpoint_reload + inference | PASS | rechargé + inférence OK (94 segments) |

## Évaluation per-plane (diagnostic, non-accuracy)
- 20 plans appariés prédiction↔GT sur les 3 images d'eval ; chaque ligne comporte : image, id d'instance prédite, GT plane id, label prédit, IoU, erreur perpendiculaire (norm. largeur), couverture horizontale.
- **Métrique de sélection PRIMAIRE = erreur perpendiculaire moyenne** (MINIMISER) ; couverture = secondaire ; Dice/IoU = diagnostics seulement (n'ont PAS déterminé le checkpoint). Sélection exécutée : époque 2.
- Chiffres (ex. perp 0.13→0.09, cov ~0.26) **non significatifs** : 2 époques sur 4 images. Objectif = valider l'exécution, pas la précision.

## Gestion des bords (les deux cas testés explicitement)
- **Plan normal** → bord **eave / bas** (max-y par colonne). Exécuté (ex. `0016`).
- **Plan parapet** → flag `-parapet` détecté → bord **parapet / haut** (min-y par colonne). Exécuté (`0169`, `0021`). Convention d'annotation inchangée.

## Intégrité des données (vérifiée avant exécution)
- 150 images approuvées présentes.
- **Holdout figé de 13 intact** : les 13 ids d'origine sont toujours en `holdout`.
- **Note/avertissement** : le split `holdout` compte désormais **19** images (les 49 nouveaux labels ont aussi été répartis) ; les 13 figés ne bougent pas. À trancher pour le vrai run : garder la comparabilité 50/100/150 sur les **13 figés** (recommandé) ou adopter les 19. **Aucune donnée approuvée n'a été modifiée.**
- Fuite groupe train/holdout = **NONE** (règle group-aware respectée).

## Avertissements / blocages
- Tête de classe ré-initialisée (81→3) : normal (fine-tuning 2 classes).
- Warning bénin `_max_size` du processor (ignoré, sans effet).
- CPU @768 : ~100 s/époque pour 4 images train + 3 eval → **confirme que le vrai run 150 img est impraticable sur CPU** (des heures). GPU requis.
- Aucun OOM (Mask2Former-tiny @768, batch 1).

## Estimation GPU pour le VRAI run 150 img @ 768 (smoke PASS ⇒ valable)
- **GPU** : T4 16 Go (minimum, bs1–2) ; **L4 / A10 24 Go recommandé** (bs2–4, marge).
- **VRAM estimée** : ~8–12 Go @768 bs2 (Mask2Former swin-tiny).
- **Accélération vs CPU** : ~50–100× → un checkpoint complet 150 img se compte en **minutes à ~1 h** (vs des heures sur CPU).
- **Pipeline prêt GPU** (device-agnostique : `.to('cuda')`, augmenter le batch). Aucun changement d'architecture requis.

## Décision
Le pipeline complet 768 px Mask2Former **passe le smoke test de bout en bout**. Le vrai run 150 img @ 768 sur GPU **n'a PAS été lancé** (conformément à la consigne). Prochaine étape = louer le GPU puis lancer le run réel (avec trancher 13-vs-19 holdout).

> **SMOKE TEST ONLY — NOT FINAL ACCURACY.** Aucune donnée de production modifiée. Humain + SAM2 uniquement (aucun VLM).
