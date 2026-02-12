# Customer Churn Prediction – Machine Learning Pipeline & API

## 1. Contexte du projet
La rétention client est un enjeu stratégique pour les entreprises. Acquérir un nouveau client coûte souvent plus cher que conserver un client existant.
L’objectif de ce projet est de **prédire le churn (départ des clients)** à partir des données historiques afin d’aider une entreprise à :
* Identifier les clients à risque
* Metttre en place des actions de rétention ciblées
* Optimiser les coûts marketing

Ce projet simule un **cas réel en entreprise**, depuis l’exploration des données jusqu’au déploiement d’une API.

## 2. Objectifs
Ce projet couvre tout le cycle de vie d’un modèle :
* Préparation des données
* Feature engineering
* Entraînement et validation
* Évaluation avec métriques métier
* Pipeline reproductible
* API de prédiction avec FastAPI
* Organisation professionnelle du code

## 3. Stack technique
* Python
* Pandas, NumPy
* Scikit-learn
* Matplotlib / Seaborn
* FastAPI
* Uvicorn
* Joblib
* SHAP

## 4. Structure du projet

detection_churn/
│
├── data/
│   ├── raw/                # Données brutes
│   ├── processed/          # Données nettoyées
│
├── models/
│   └── model.pkl           # Modèle entraîné
│
├── src/
│   ├── config.py           # Les configurations
│   ├── preprocess.py       # Nettoyage et transformation
│   ├── split.py            # Séparation train/test
│   ├── train.py            # Entraînement du modèle
│   ├── evaluate.py         # Évaluation
│   ├── predict.py          # Prédiction locale
│   ├── decision.py         # Threshold decision
│   ├── curve.py            # Courbes de validation
│   ├── explain.py          # Feature importance avec SHAP
│   ├── utils.py            # Metrics
│
├── api/
│   ├── app.py              # API FastAPI
│   ├── schema.py           # Validation de données
│
├── notebooks/
│   └── eda.ipynb           # Analyse exploratoire
│
├── requirements.txt
└── README.md

## 5. Dataset
Le dataset contient des informations sur les clients :

Exemples de variables :
* costomer Id
* credit_score
* country
* gender
* age
* tenure
* balance
* products_number
* credit_card
* active_member
* estimated_salary
* churn (variable cible)

## 6. Étapes du pipeline

### 6.1 Data Cleaning
* Gestion des valeurs manquantes
* Encodage des variables catégorielles
* Normalisation des variables numériques

### 6.2 Feature Engineering
Exemples :
* Transformation de variables catégorielles
* Création de variables dérivées
* Scaling

### 6.3 Séparation des données
Script :
python src/split.py

Sorties :
* X_train
* X_test
* y_train
* y_test

### 6.4 Entraînement du modèle
Script :
python src/train.py

Modèles testés :
* Random Forest
* XGBoost
* Logistic Regression
* SVM

Le meilleur modèle est sauvegardé dans :
models/model.pkl

### 6.5 Évaluation
Script :
python src/evaluate.py

Métriques utilisées :
* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC

Pourquoi ces métriques ?
Dans un problème de churn, **le Recall est critique**, car manquer un client qui va partir coûte beaucoup cher.

## 7. Prédiction locale
Script :
python src/predict.py

Permet de tester le modèle sans API.

## 8. Déploiement avec FastAPI

### Lancer l’API
Dans le dossier du projet :
uvicorn api.app:app --reload

L'API sera accessible sur : http://localhost:8000

### Documentation interactive
Swagger UI: http://localhost:8000/docs

### Endpoint d'accueil
GET /

### Endpoint de santé
GET /health

Réponse :
{"status": "ok"}

### Endpoint de prédiction
POST /predict

Exemple de requête JSON :
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

Réponse :
{
    "Fraud_probability": 0.96,
    "Prediction": 1
}

## 9. Résultats
Exemple :

* Accuracy : 0.77
* Recall : 0.82
* ROC-AUC : 0.88

Interprétation :
Le modèle détecte correctement une grande proportion des clients susceptibles de churner, ce qui le rend exploitable dans un contexte business.

## 10. Cas d’usage métier
Ce modèle peut être utilisé pour :
* Déclencher des campagnes de rétention
* Prioriser les clients à risque
* Réduire le churn rate global

## 11. Améliorations possibles
* Hyperparameter tuning avancé
* Feature engineering plus avancé
* Déploiement cloud (Render, AWS, GCP)
* Monitoring du modèle
* Pipeline ML automatisé

## 12. Comment exécuter le projet

### 12.1 Cloner le repo
git clone <repo_url>
cd detection_churn

### 12.2 Installer les dépendances
pip install -r requirements.txt

### 12.3 Lancer le pipeline
python src/split.py
python src/train.py
python src/evaluate.py

### 12.4 Lancer l’API
uvicorn api.app:app --reload

## 13. Ce que ce projet démontre
Ce projet démontre des compétences en :
* Data Science appliqué
* Machine Learning appliqué
* Data preprocessing
* Feature engineering
* Validation de modèles
* Organisation de projet professionnel
* Déploiement d’API ML

## 14. Auteur
Nom : Prosper Fantodji
Domaine : Data Science / Machine Learning

## 15. Licence
Projet éducatif et portfolio.