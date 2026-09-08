# 🐎 HorseProba — Pronostics hippiques probabilistes (Streamlit)

Application Streamlit qui classe les partants d'une course hippique par **probabilité de
victoire et de placé**, à partir d'un **modèle logit conditionnel** (Bradley-Terry /
Plackett-Luce généralisé) combinant cotes, forme récente, jockey, entraîneur, corde, poids
et fraîcheur. Chaque classement est accompagné des statistiques et du raisonnement qui le
justifient, et l'interface rappelle explicitement les limites du modèle.

> ⚠️ Les paris hippiques sont incertains par nature. Ce projet est un outil d'analyse
> statistique et pédagogique ; il ne garantit aucun gain. Jouez de façon responsable.

---

## Fonctionnalités

| Fonction | État |
|---|---|
| Sélection d'une course du programme du jour (source web publique PMU, sans clé) | ✅ |
| Import CSV d'une course ou d'un historique (séparateur `,`/`;`, alias FR/EN) | ✅ |
| Saisie manuelle d'une course dans un tableau éditable | ✅ |
| Modèle logit conditionnel régularisé, estimé par max. de vraisemblance (L-BFGS) | ✅ |
| Probabilité de placé (top-k) par Plackett-Luce (exact ou Monte-Carlo) | ✅ |
| Comparaison modèle vs marché, cote « juste », détection d'écart | ✅ |
| Explications par cheval (contributions de chaque variable) | ✅ |
| Backtest walk-forward : log-loss, Brier, calibration, hit rate, ROI illustratif | ✅ |
| Données synthétiques réalistes pour la démo / les tests | ✅ |
| Export CSV du pronostic | ✅ |
| Gestion des erreurs (course introuvable, réseau, CSV invalide) sans crash | ✅ |
| Tests unitaires + CI GitHub Actions | ✅ |

### Non implémenté (pistes)
- Scraping des **résultats historiques** PMU en masse pour constituer automatiquement un
  historique d'entraînement (le module `pmu.fetch_results` fournit la brique de base).
- Intégration d'APIs sous licence (The Racing API, Equidia Pro) via `st.secrets`.
- Variables supplémentaires : aptitude terrain/distance par cheval, œillères, ferrage (trot),
  valeur handicap, forme de l'entraîneur sur 30 jours.
- Modèles alternatifs (gradient boosting avec softmax par course) et ensembles.
- Optimisation de mises (Kelly fractionnaire) — volontairement absente pour ne pas
  encourager le jeu.

---

## Structure du dépôt

```
.
├── app.py                        # Point d'entrée Streamlit
├── requirements.txt              # Dépendances (Streamlit Cloud)
├── requirements-dev.txt          # + pytest, ruff
├── README.md
├── .gitignore
├── .streamlit/
│   ├── config.toml               # Thème / serveur
│   └── secrets.toml.example      # Modèle de secrets (aucun secret obligatoire)
├── .github/workflows/ci.yml      # Tests automatiques
├── data/
│   └── sample_races.csv          # Jeu d'exemple (5 courses, résultats inclus)
├── horseproba/
│   ├── __init__.py
│   ├── schema.py                 # Schéma de colonnes canonique
│   ├── features.py               # Ingénierie des variables (musique, cotes, taux lissés…)
│   ├── model.py                  # Logit conditionnel + Plackett-Luce + explications
│   ├── evaluate.py               # Métriques probabilistes + backtest walk-forward
│   ├── data/
│   │   ├── __init__.py           # 📌 Documentation centralisée des sources de données
│   │   ├── loader.py             # Lecture/validation CSV
│   │   ├── synthetic.py          # Générateur d'historique synthétique
│   │   └── pmu.py                # Source web publique PMU (non officielle)
│   └── ui/
│       ├── __init__.py
│       └── components.py         # Tableaux, graphiques Plotly, avertissements
└── tests/
    └── test_pipeline.py
```

---

## Approche statistique (résumé)

**Pourquoi un logit conditionnel ?** Une course est un problème de *choix discret* : un seul
gagnant parmi *n*. Le modèle attribue à chaque cheval une force latente
`s_i = β·x_i` et pose `P(i gagne) = exp(s_i) / Σ_j exp(s_j)`. Les probabilités somment à 1
par construction, les features sont **centrées par course** (seules les différences entre
partants comptent), et les coefficients sont directement interprétables. C'est le modèle de
référence de la littérature (Bolton & Chapman 1986, Benter 1994). Justification détaillée
dans les commentaires de `horseproba/model.py`.

- **Estimation** : maximum de vraisemblance conditionnelle pénalisé (ridge centré sur des
  coefficients a priori) → fonctionne même avec peu ou pas d'historique.
- **Placé** : Plackett-Luce, énumération exacte des k-permutations (n ≤ 16) sinon Monte-Carlo
  (astuce de Gumbel).
- **Variables** : probabilité implicite du marché, score de forme pondéré (musique),
  taux de victoire/place récents et en carrière (lissage bayésien), gains par course,
  fraîcheur, réussite jockey/entraîneur (lissée), corde relative, poids relatif, âge relatif.
- **Évaluation** : log-loss et Brier vs marché et vs uniforme, calibration par tranches,
  backtest temporel sans fuite d'information.

---

## Sources de données

Documentées dans `horseproba/data/__init__.py` et dans l'onglet *Méthode & données*.

| Source | Clé requise | Rafraîchissement |
|---|---|---|
| **CSV utilisateur** (partants ou historique avec `finish_position`) | Non | Re-téléverser |
| **`data/sample_races.csv`** (exemple embarqué) | Non | Éditer dans le dépôt |
| **Synthétique** (`generate_history`) | Non | Graine / nombre de courses |
| **PMU web public** (`online.turfinfo.api.pmu.fr`, endpoints JSON consommés par pmu.fr — *non officiel*) | Non | Bouton « Actualiser », cache 2–10 min |
| APIs tierces sous licence | Oui (`st.secrets`) | Non intégré |

Usage responsable de la source web : User-Agent identifiable, délai entre requêtes, cache,
aucun contournement. La structure peut changer sans préavis : l'application affiche alors un
message et bascule sur l'import CSV.

### Format CSV

Obligatoire : `race_id`, `horse`. Recommandé : `odds` (cote décimale), `musique` (`1p 3p Da 2p`),
`jockey`, `trainer`, `draw`, `weight_kg`, `age`, `days_since_last_run`, `career_starts`,
`career_wins`, `career_places`, `earnings`. Contexte : `race_date`, `track`, `discipline`
(`plat|trot|obstacle`), `distance_m`, `going`. Cible (historique seulement) : `finish_position`.
Des alias français (`cheval`, `cote`, `entraineur`, `arrivee`…) sont reconnus.

---

## Installation locale

```bash
git clone https://github.com/<votre-compte>/horseproba.git
cd horseproba
python -m venv .venv && source .venv/bin/activate     # Windows : .venv\Scripts\activate
pip install -r requirements-dev.txt
pytest -q                                              # tests
streamlit run app.py
```

Secrets optionnels : copier `.streamlit/secrets.toml.example` → `.streamlit/secrets.toml`.

---

## Déploiement sur Streamlit Community Cloud

1. **Pousser le dépôt sur GitHub** (public ou privé) avec `app.py` et `requirements.txt` à la racine.
   ```bash
   git init && git add . && git commit -m "HorseProba v1.0"
   git branch -M main
   git remote add origin https://github.com/<votre-compte>/horseproba.git
   git push -u origin main
   ```
2. Aller sur **https://share.streamlit.io**, se connecter avec GitHub, cliquer **« Create app »**
   → **« Deploy a public app from GitHub »**.
3. Renseigner :
   - *Repository* : `<votre-compte>/horseproba`
   - *Branch* : `main`
   - *Main file path* : `app.py`
   - *App URL* : sous-domaine de votre choix
   - *Advanced settings* → Python version **3.11** (recommandé).
4. **Secrets (optionnels)** : *Advanced settings → Secrets*, coller le contenu de
   `.streamlit/secrets.toml.example` adapté (`HTTP_USER_AGENT` avec votre contact,
   `ENABLE_WEB_FETCH = false` pour désactiver la source web). Ils sont lus via `st.secrets`.
5. Cliquer **Deploy**. Le build installe `requirements.txt` ; l'application est en ligne en
   1–3 minutes à `https://<sous-domaine>.streamlit.app`.
6. **Mises à jour** : chaque `git push` sur `main` redéploie automatiquement. Le menu
   *Manage app* permet de voir les logs, redémarrer ou modifier les secrets.

Dépannage : si le build échoue, consulter les logs (*Manage app*) ; vérifier que
`requirements.txt` est à la racine et que la version Python est 3.10+.

---

## Données & modèles (référence technique)

- **Structures** : `pandas.DataFrame` au schéma `horseproba/schema.py` ; modèle sérialisable
  en JSON (`ConditionalLogitModel.to_json/from_json`).
- **Stockage** : aucun stockage serveur. Les caches Streamlit (`st.cache_data`,
  `st.cache_resource`) sont en mémoire et volatils. Les fichiers utilisateur restent côté
  session.
- **Entrées fonctionnelles** : `app.py` (page unique, trois onglets : Pronostic, Modèle &
  backtest, Méthode & données) ; barre latérale pour l'historique d'entraînement et λ.

---

## Limites (rappelées dans l'interface)

- Un cheval estimé à 30 % perd 7 fois sur 10 : ce sont des probabilités, pas des prédictions.
- Sans historique réel, les coefficients a priori ne sont pas calibrés sur vos courses.
- Les cotes de référence évoluent jusqu'au départ ; un « écart de valeur » peut disparaître.
- Le marché est un concurrent redoutable : battre systématiquement les cotes est rare.
- Jeu responsable : *Joueurs Info Service* 09 74 75 13 13. Interdit aux mineurs.

## Licence

MIT. Les données PMU restent la propriété de leurs éditeurs ; respectez leurs conditions d'utilisation.
