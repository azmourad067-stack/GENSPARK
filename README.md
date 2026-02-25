# 🏇 PronoHippique AI

> Application de pronostics hippiques intelligente propulsée par l'IA Vision.

## 🚀 Déploiement sur Streamlit Cloud

### 1. Préparer le dépôt GitHub
```bash
git init
git add .
git commit -m "Initial commit - PronoHippique AI"
git remote add origin https://github.com/VOTRE_USERNAME/pronohippique-ai
git push -u origin main
```

### 2. Déployer sur Streamlit Cloud
1. Rendez-vous sur [share.streamlit.io](https://share.streamlit.io)
2. Connectez votre compte GitHub
3. Cliquez **"New app"**
4. Sélectionnez votre dépôt et `app.py` comme fichier principal
5. Cliquez **"Deploy!"**

### 3. Configurer les secrets API (recommandé)
Dans Streamlit Cloud → Settings → Secrets :
```toml
GEMINI_API_KEY = "AIza..."
OPENAI_API_KEY = "sk-..."
```

---

## 🏗️ Structure du Projet

```
pronohippique-ai/
├── app.py                          ← Application principale Streamlit
├── requirements.txt                ← Dépendances Python
├── README.md                       ← Ce fichier
├── .streamlit/
│   └── config.toml                 ← Thème et configuration Streamlit
└── modules/
    ├── __init__.py
    ├── ocr_extractor.py            ← Extraction OCR (Gemini/OpenAI/EasyOCR)
    ├── data_cleaner.py             ← Nettoyage & parsing des données
    ├── scorer.py                   ← Algorithme de scoring hippique
    ├── pronostic.py                ← Génération des pronostics
    └── visualizer.py               ← Graphiques Plotly
```

---

## 🤖 Moteurs OCR Supportés

| Moteur | Précision | Coût | Nécessite |
|--------|-----------|------|-----------|
| 🤖 Google Gemini 1.5 Flash | ⭐⭐⭐⭐⭐ | Faible | Clé API Gemini |
| 🧠 OpenAI GPT-4o | ⭐⭐⭐⭐⭐ | Moyen | Clé API OpenAI |
| 📷 EasyOCR | ⭐⭐⭐ | Gratuit | Rien |

---

## 📊 Algorithme de Scoring

| Critère | Poids |
|---------|-------|
| Musique récente (forme) | 20% |
| Record absolu (vitesse) | 18% |
| Réussite Driver | 12% |
| Cote PMU inversée | 9% |
| Réussite Entraîneur | 10% |
| Écart (fraîcheur) | 10% |
| Gains | 8% |
| Victoires Driver | 7% |
| Régularité | 6% |

---

## 📸 Types d'Images Supportés

- **Liste des Partants** : numéro, cheval, SA, driver, entraîneur, musique, gains, cotes
- **Records Absolus** : record chrono, date et lieu du record
- **Statistiques Drivers** : courses, victoires, écart, réussite %, musique driver
- **Statistiques Entraîneurs** : courses, victoires, écart, réussite %, musique entraîneur

---

## 🔮 Roadmap Machine Learning

L'application est conçue pour être **ML-ready** :
- Structure de données compatible scikit-learn
- Historique de résultats exportable en CSV
- Interface de feedback sur les pronostics (à ajouter)
- Modèle Random Forest ou XGBoost intégrable dans `modules/scorer.py`

---

## ⚠️ Avertissement

Les pronostics fournis sont à titre indicatif uniquement. Le jeu peut créer une dépendance. Jouez de manière responsable. [Joueurs Info Service : 09 74 75 13 13](https://www.joueurs-info-service.fr)
