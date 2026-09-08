"""
Sous-paquet données.

SOURCES DE DONNÉES UTILISÉES (documentation centralisée)
=========================================================

1. Import CSV (source principale, toujours disponible)
   - Schéma : voir `horseproba/schema.py`.
   - Un jeu d'exemple est fourni dans `data/sample_races.csv`.
   - Rafraîchissement : l'utilisateur téléverse un nouveau fichier dans l'interface.

2. Données synthétiques (démonstration / tests)
   - `horseproba.data.synthetic.generate_history` génère un historique réaliste
     dont la « vérité terrain » est connue. Utile pour valider le pipeline et
     illustrer le backtest sans dépendre du réseau.

3. Source web publique : programme et résultats PMU (non officielle, sans clé)
   - Module `horseproba.data.pmu`.
   - Endpoints JSON publics utilisés par le site pmu.fr :
       https://online.turfinfo.api.pmu.fr/rest/client/1/programme/{JJMMAAAA}
       https://online.turfinfo.api.pmu.fr/rest/client/1/programme/{JJMMAAAA}/R{n}/C{m}/participants
       https://online.turfinfo.api.pmu.fr/rest/client/1/programme/{JJMMAAAA}/R{n}/C{m}/rapports-definitifs
   - Il ne s'agit PAS d'une API documentée : elle peut changer sans préavis.
     L'application se dégrade proprement (message + retour au mode CSV) en cas d'échec.
   - Bonnes pratiques respectées : User-Agent identifiable, délai entre requêtes,
     cache (st.cache_data) pour ne pas solliciter la source inutilement, aucun
     contournement de protection. Consultez les CGU du site avant tout usage
     intensif ou commercial.
   - Rafraîchissement : bouton « Actualiser » dans l'interface (vide le cache).

4. Alternatives payantes / documentées (non intégrées par défaut)
   - The Racing API (theracingapi.com), Equidia Pro, données Turf-FR par abonnement.
   - Prévues via `st.secrets` (voir `.streamlit/secrets.toml.example`).
"""

from .loader import (  # noqa: F401
    load_runners_csv,
    validate_runners,
    coerce_types,
    sample_dataset_path,
    load_sample_history,
)
