# Customer Churn Prediction — ML Pipeline & API de Prédiction

> Système de prédiction du churn client basé sur le Machine Learning, couvrant l'intégralité du cycle de vie : exploration des données, modélisation, évaluation métier et déploiement via une API REST.

---

## 📋 Table des matières

1. [Contexte & enjeux métier](#-contexte--enjeux-métier)
2. [Objectifs du projet](#-objectifs-du-projet)
3. [Architecture du projet](#-architecture-du-projet)
4. [Dataset](#-dataset)
5. [Pipeline Machine Learning](#-pipeline-machine-learning)
6. [Modèles & Évaluation](#-modèles--évaluation)
7. [Interprétabilité avec SHAP](#-interprétabilité-avec-shap)
8. [API REST – Endpoints](#-api-rest--endpoints)
9. [Résultats](#-résultats)
10. [Stack technologique](#-stack-technologique)
11. [Installation & Lancement](#-installation--lancement)
12. [Roadmap](#-roadmap)
13. [Cas d'usage métier](#-cas-dusage-métier)
14. [Compétences démontrées](#-compétences-démontrées)
15. [Auteur](#-auteur)

---

## Contexte & enjeux métier

La rétention client est un enjeu stratégique majeur pour toute entreprise. Acquérir un nouveau client coûte en moyenne **5 à 7 fois plus cher** que fidéliser un client existant.

Ce projet simule un **cas réel en entreprise** : à partir des données historiques clients, le système identifie automatiquement les profils à risque de départ afin de permettre une intervention proactive.

| Problème | Solution apportée |
|---|---|
| Identifier les clients susceptibles de partir | Score de probabilité de churn |
| Prioriser les actions de rétention | Classement par niveau de risque |
| Réduire les coûts d'acquisition | Ciblage précis des campagnes |

---

## Objectifs du projet

Ce projet couvre **l'intégralité du cycle de vie** d'un modèle ML en contexte professionnel :

- ✅ Exploration et analyse des données (EDA)
- ✅ Nettoyage et feature engineering
- ✅ Entraînement et validation de plusieurs algorithmes
- ✅ Évaluation avec métriques adaptées au contexte métier
- ✅ Interprétabilité du modèle (SHAP)
- ✅ Pipeline reproductible et modulaire
- ✅ Déploiement via API REST avec FastAPI
- ✅ Organisation professionnelle du code

---

## Architecture du projet

```
detection_churn/
│
├── data/
│   ├── raw/               # Données brutes originales
│   └── processed/         # Données nettoyées et transformées
│
├── models/
│   └── churn_pipeline.pkl          # Meilleur modèle sérialisé (Joblib)
│
├── src/
│   ├── config.py          # Paramètres et configurations globales
│   ├── preprocess.py      # Nettoyage et transformation des données
│   ├── split.py           # Séparation Train / Test
│   ├── train.py           # Entraînement et sélection du modèle
│   ├── evaluate.py        # Évaluation des performances
│   ├── predict.py         # Prédiction locale (hors API)
│   ├── decision.py        # Logique de seuil de décision
│   ├── curve.py           # Courbes ROC, Precision-Recall
│   ├── explain.py         # Interprétabilité SHAP
│   └── utils.py           # Fonctions utilitaires et métriques
│
├── api/
│   ├── app.py             # Application FastAPI
│   └── schema.py          # Validation des données entrantes (Pydantic)
│
├── notebooks/
│   └── eda.ipynb          # Analyse exploratoire interactive
│
├── requirements.txt
└── README.md
```

---

## Dataset

Le dataset contient des informations comportementales et démographiques sur les clients d'une institution financière.

| Variable | Type | Description |
|---|---|---|
| `customer_id` | Identifiant | Identifiant unique client |
| `credit_score` | Numérique | Score de crédit |
| `country` | Catégorielle | Pays de résidence |
| `gender` | Catégorielle | Genre |
| `age` | Numérique | Âge du client |
| `tenure` | Numérique | Ancienneté (années) |
| `balance` | Numérique | Solde du compte |
| `products_number` | Numérique | Nombre de produits souscrits |
| `credit_card` | Binaire | Possession d'une carte de crédit |
| `active_member` | Binaire | Statut d'activité du client |
| `estimated_salary` | Numérique | Salaire estimé |
| `churn` | **Cible** | 1 = churner / 0 = fidèle |

---

## Pipeline Machine Learning

```
Données brutes
    │
    ▼
1. Exploration & EDA (notebooks/eda.ipynb)
    │
    ▼
2. Nettoyage — gestion des valeurs manquantes, encodage
    │
    ▼
3. Feature Engineering — variables dérivées, scaling
    │
    ▼
4. Séparation Train / Test  →  python src/split.py
    │
    ▼
5. Entraînement multi-modèles  →  python src/train.py
    │
    ▼
6. Évaluation & sélection  →  python src/evaluate.py
    │
    ▼
7. Interprétabilité SHAP  →  python src/explain.py
    │
    ▼
8. Sérialisation  →  models/model.pkl
    │
    ▼
9. Déploiement API REST  →  uvicorn api.app:app
```

---

## Modèles & Évaluation

### Algorithmes comparés

| Modèle | Avantages |
|---|---|
| **Random Forest** | Robuste, gère bien les non-linéarités |
| **XGBoost** | Haute performance, gestion du déséquilibre |
| **Régression Logistique** | Interprétable, baseline solide |
| **SVM** | Efficace sur données de taille moyenne |

Le meilleur modèle est automatiquement sélectionné et sauvegardé dans `models/model.pkl`.

### Métriques d'évaluation

| Métrique | Rôle dans ce contexte |
|---|---|
| **Recall** | ⭐ Priorité absolue — minimise les churners non détectés |
| **ROC-AUC** | Performance globale de discrimination |
| **Precision** | Pertinence des alertes déclenchées |
| **F1-score** | Équilibre Precision / Recall |
| **Accuracy** | Vue d'ensemble (à nuancer avec le déséquilibre de classes) |

> **Pourquoi prioriser le Recall ?**  
> Dans un problème de churn, **rater un client qui va partir est beaucoup plus coûteux** que déclencher une action de rétention sur un faux positif. Le Recall mesure précisément la capacité du modèle à capturer tous les churners réels.

---

## Interprétabilité avec SHAP

Le module `src/explain.py` intègre **SHAP (SHapley Additive exPlanations)** pour expliquer les prédictions du modèle :

- Identification des **variables les plus influentes** à l'échelle globale
- **Explication locale** de chaque prédiction individuelle
- Visualisations : beeswarm plot, waterfall chart, summary plot

> Cette dimension est essentielle pour garantir la confiance des équipes métier dans le modèle et répondre aux exigences de transparence algorithmique.

---

## API REST – Endpoints

### `GET /` — Accueil

```http
GET /
```

### `GET /health` — Vérification de l'état

```http
GET /health
```

**Réponse :**

```json
{
  "status": "ok"
}
```

---

### `POST /predict` — Prédiction du churn

```http
POST /predict
Content-Type: application/json
```

**Corps de la requête :**

```json
{
  "credit_score": 500,
  "country": "germany",
  "gender": "Male",
  "age": 60,
  "tenure": 3,
  "balance": 0.0,
  "products_number": 1,
  "credit_card": 1,
  "active_member": 0,
  "estimated_salary": 50000
}
```

**Réponse :**

```json
{
  "churn_probability": 0.96,
  "prediction": 1
}
```

| Champ | Type | Description |
|---|---|---|
| `churn_probability` | `float [0, 1]` | Probabilité estimée de départ |
| `prediction` | `int (0 ou 1)` | 1 = churner prédit / 0 = fidèle prédit |

---

## Résultats

Performances obtenues sur le jeu de test :

| Métrique | Score |
|---|---|
| **Accuracy** | 0.77 |
| **Recall** | 0.82 |
| **ROC-AUC** | 0.88 |

**Interprétation :** Le modèle détecte correctement **82% des clients susceptibles de churner**, ce qui le rend directement exploitable dans un contexte business pour déclencher des actions de rétention ciblées.

---

## Stack technologique

| Outil | Usage |
|---|---|
| **Python** | Langage principal |
| **Pandas / NumPy** | Manipulation et transformation des données |
| **Scikit-learn** | Pipeline ML, modélisation, évaluation |
| **XGBoost** | Algorithme de boosting haute performance |
| **SHAP** | Interprétabilité et explicabilité du modèle |
| **Matplotlib / Seaborn** | Visualisations (EDA, courbes ROC) |
| **FastAPI** | Framework API REST |
| **Pydantic** | Validation des données entrantes |
| **Joblib** | Sérialisation du modèle |
| **Uvicorn** | Serveur ASGI de production |

---

## Installation & Lancement

### 1. Cloner le projet

```bash
git clone https://github.com/LProsper25/detection_churn.git
cd detection_churn
```

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 3. Exécuter le pipeline ML

```bash
python src/split.py      # Séparation des données
python src/train.py      # Entraînement et sélection du modèle
python src/evaluate.py   # Évaluation des performances
python src/explain.py    # Génération des graphiques SHAP (optionnel)
```

### 4. Lancer l'API

```bash
uvicorn api.app:app --reload
```

### 5. Accéder à la documentation interactive

```
http://localhost:8000/docs
```

> FastAPI génère automatiquement une interface **Swagger UI** pour tester les endpoints directement depuis le navigateur.

---

## Roadmap

| Amélioration | Statut |
|---|---|
| Hyperparameter tuning avancé (Optuna) | 🔜 Planifié |
| Feature engineering avancé | 🔜 Planifié |
| Gestion du déséquilibre de classes (SMOTE) | 🔜 Planifié |
| Déploiement cloud (Render / AWS / GCP) | 🔜 Planifié |
| Monitoring du modèle en production (Evidently) | 🔜 Planifié |
| Pipeline ML automatisé (MLflow / Prefect) | 🔜 Planifié |
| Dashboard de suivi du churn rate | 🔜 Planifié |

---

## Cas d'usage métier

| Secteur | Application |
|---|---|
| 🏦 Banque & Assurance | Rétention des clients inactifs |
| 📱 Télécommunications | Prévention du départ vers la concurrence |
| 🛍️ E-commerce & Retail | Réactivation des clients dormants |
| 🎮 SaaS & Abonnements | Réduction du taux de désabonnement |

---

## Compétences démontrées

Ce projet illustre une maîtrise end-to-end de la Data Science appliquée à un cas métier concret :

- **Data Science appliquée** — EDA, feature engineering, sélection de variables
- **Machine Learning** — comparaison multi-modèles, optimisation du seuil de décision
- **MLOps** — pipeline reproductible, sérialisation, déploiement API
- **Interprétabilité** — SHAP pour l'explicabilité des prédictions
- **Logique métier** — priorisation des métriques selon les enjeux business
- **Software Engineering** — organisation modulaire, validation des données, séparation des responsabilités

---

## 👤 Auteur

**Fantodji Prosper**  
*Data Scientist & Machine Learning Engineer*

[![GitHub](https://img.shields.io/badge/GitHub-LProsper25-black?style=flat&logo=github)](https://github.com/LProsper25)

---

## 📄 Licence

Projet à but éducatif et portfolio.