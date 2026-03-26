Détection et Classification du Harcèlement en Ligne

Projet en cours — L3 MIASHS, Semestre 6, Sciences des Données 4 (2025–2026)

Projet de groupe réalisé en groupe de 4 dans le cadre du module Sciences des Données 4 en L3 MIASHS.

Objectif : Construire un pipeline complet de détection automatique du harcèlement en ligne à partir de messages textuels, en deux tâches :

1. Classification binaire : détecter si un message est haineux ou non
2. *Classification multi-classes : identifier la catégorie parmi 6 formes de harcèlement (Homophobie, Islamophobie, Racisme, Sexisme, Validisme, Xénophobie)

Données

- `Harcelement.csv` — base de 2 640 messages annotés (colonnes : Identifiant, Texte, Traduction, Types, Categories)
- Parfaitement équilibrée : 1 320 messages haineux / 1 320 neutres, 220 messages par catégorie

Contenu du repo

| Fichier | Description |
|---|---|
| `visu_final.Rmd` | Analyse exploratoire complète (EDA) en R — visualisations, analyse linguistique, stylistique et émotionnelle (NRC) |
| `visu_final.pdf` | Rapport PDF généré depuis le Rmd |
| `Harcelement_donnees.ipynb` | Notebook Python (Google Colab) — vérification de la qualité des données |
| `harcelement_ml2_cv.py` | Modèle ML en Python — TF-IDF + Régression Logistique, validation croisée 5-fold |
| `tf_idf_base_train2.py` | Modèle TF-IDF sur base d'entraînement étendue (Google Colab) — classification binaire + multi-classes |
| `Explication_TF.docx` | Explication détaillée des résultats TF-IDF (précision 98% binaire, 89% multi-classes) |
| `Harcelement.csv` | Base de données annotée |

Stack technique

- R : tidyverse, tidytext, wordcloud, syuzhet (NRC), ggplot2
- Python : scikit-learn (TF-IDF, LogisticRegression, StratifiedKFold), pandas, numpy

État d'avancement

- [x] Constitution et nettoyage de la base de données
- [x] Analyse exploratoire complète (EDA) en R
- [x] Vérification qualité des données (Python / Colab)
- [x] Modèle baseline : TF-IDF + Régression Logistique avec validation croisée
- [x] Modèle TF-IDF étendu sur base d'entraînement complète (98% binaire, 89% multi-classes)
- [x] Analyse et explication des résultats
- [ ] Comparaison de modèles (SVM, Random Forest, XGBoost)
- [ ] Features enrichies (NRC + longueur + score stylistique)
- [ ] Rapport final et conclusions

Reproduire l'analyse R

```r
# Installer les dépendances
install.packages(c("tidyverse", "tidytext", "tm", "wordcloud", "syuzhet", "RColorBrewer"))

# Placer Harcelement.csv dans le même dossier que visu_final.Rmd
# Puis knit le fichier depuis RStudio
```

Lancer le modèle Python

```bash
pip install scikit-learn pandas numpy
python harcelement_ml2_cv.py
```

> Le script attend les fichiers `Base_Entraînement - Feuille 1.csv` et `Base_Test - Feuille 1.csv` dans le même dossier.

