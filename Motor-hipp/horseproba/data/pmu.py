"""
Source web publique : programme, partants et résultats PMU (France).

AVERTISSEMENT
-------------
Les endpoints ci-dessous sont ceux consommés par le site public pmu.fr. Ils sont
accessibles sans authentification mais NE SONT PAS une API officielle documentée :
leur structure peut changer sans préavis. Ce module est donc écrit de manière
très défensive (chaque champ est lu avec un `.get()` et une valeur par défaut) et
toute erreur remonte comme `PMUFetchError` avec un message lisible, afin que
l'interface puisse basculer sur le mode CSV.

Usage responsable
-----------------
- Un User-Agent identifiable est envoyé (configurable via st.secrets["HTTP_USER_AGENT"]).
- Un délai minimal entre requêtes est respecté (RATE_LIMIT_SECONDS).
- Les résultats sont mis en cache côté application (voir app.py, st.cache_data).
- N'utilisez pas ce module pour aspirer massivement des données : pour cela,
  passez par un fournisseur de données sous licence.

Endpoints
---------
Programme du jour :
    GET {BASE}/programme/{JJMMAAAA}
Partants d'une course :
    GET {BASE}/programme/{JJMMAAAA}/R{reunion}/C{course}/participants
Rapports (résultats) définitifs :
    GET {BASE}/programme/{JJMMAAAA}/R{reunion}/C{course}/rapports-definitifs

Rafraîchissement : les fonctions ne mettent rien en cache elles-mêmes ; c'est
l'appelant (Streamlit) qui gère le cache et le bouton « Actualiser ».
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

BASE_URL = "https://online.turfinfo.api.pmu.fr/rest/client/1"
DEFAULT_TIMEOUT = 10  # secondes
RATE_LIMIT_SECONDS = 0.6
_last_call = 0.0


class PMUFetchError(RuntimeError):
    """Échec de récupération/lecture des données PMU (message destiné à l'utilisateur)."""


@dataclass
class RaceRef:
    day: date
    reunion: int
    course: int
    label: str
    track: str = ""
    discipline: str = ""
    distance_m: Optional[int] = None
    start_time: str = ""

    @property
    def race_id(self) -> str:
        return f"{self.day.isoformat()}_R{self.reunion}_C{self.course}"


def _date_path(day: date) -> str:
    return day.strftime("%d%m%Y")


def _get_json(url: str, user_agent: str = "HorseProbaApp/1.0 (educational)") -> Dict[str, Any]:
    """GET JSON avec limitation de débit, timeout et gestion d'erreurs propre."""
    global _last_call
    wait = RATE_LIMIT_SECONDS - (time.monotonic() - _last_call)
    if wait > 0:
        time.sleep(wait)
    try:
        resp = requests.get(url, headers={"User-Agent": user_agent, "Accept": "application/json"}, timeout=DEFAULT_TIMEOUT)
        _last_call = time.monotonic()
    except requests.Timeout as exc:
        raise PMUFetchError("La source PMU ne répond pas (délai dépassé).") from exc
    except requests.RequestException as exc:
        raise PMUFetchError(f"Erreur réseau lors de l'accès à la source PMU : {exc}") from exc
    if resp.status_code == 204 or not resp.content:
        raise PMUFetchError("Aucune donnée disponible pour cette date/course (réponse vide).")
    if resp.status_code >= 400:
        raise PMUFetchError(f"La source PMU a renvoyé une erreur HTTP {resp.status_code}.")
    try:
        return resp.json()
    except ValueError as exc:
        raise PMUFetchError("Réponse PMU illisible (JSON invalide) — la structure a peut-être changé.") from exc


def _map_discipline(specialite: str) -> str:
    s = (specialite or "").upper()
    if "TROT" in s:
        return "trot"
    if any(k in s for k in ("HAIE", "STEEPLE", "CROSS", "OBSTACLE")):
        return "obstacle"
    return "plat"


def fetch_program(day: date, user_agent: str = "HorseProbaApp/1.0 (educational)") -> List[RaceRef]:
    """Liste les courses du jour. Lève PMUFetchError en cas d'échec."""
    data = _get_json(f"{BASE_URL}/programme/{_date_path(day)}", user_agent)
    reunions = (data.get("programme") or {}).get("reunions") or []
    refs: List[RaceRef] = []
    for reunion in reunions:
        rnum = reunion.get("numOfficiel")
        track = ((reunion.get("hippodrome") or {}).get("libelleCourt")) or ""
        for course in reunion.get("courses") or []:
            cnum = course.get("numOrdre") or course.get("numExterne")
            if rnum is None or cnum is None:
                continue
            heure = course.get("heureDepart")
            start = ""
            if isinstance(heure, (int, float)):
                start = pd.to_datetime(int(heure), unit="ms", utc=True).tz_convert("Europe/Paris").strftime("%H:%M")
            refs.append(
                RaceRef(
                    day=day,
                    reunion=int(rnum),
                    course=int(cnum),
                    label=f"R{rnum}C{cnum} {start} – {track} – {course.get('libelle', '')}".strip(),
                    track=track,
                    discipline=_map_discipline(course.get("specialite", "")),
                    distance_m=course.get("distance"),
                    start_time=start,
                )
            )
    if not refs:
        raise PMUFetchError("Aucune course trouvée pour cette date.")
    return refs


def fetch_runners(ref: RaceRef, user_agent: str = "HorseProbaApp/1.0 (educational)") -> pd.DataFrame:
    """
    Récupère les partants d'une course et les convertit au schéma canonique.
    Les cotes utilisées sont les « rapports probables » (cote de référence PMU) si présents.
    """
    url = f"{BASE_URL}/programme/{_date_path(ref.day)}/R{ref.reunion}/C{ref.course}/participants"
    data = _get_json(url, user_agent)
    participants = data.get("participants") or []
    if not participants:
        raise PMUFetchError("Aucun partant renvoyé pour cette course.")

    rows = []
    for p in participants:
        if str(p.get("statut", "PARTANT")).upper() == "NON_PARTANT":
            continue
        odds = None
        rapport = p.get("dernierRapportDirect") or p.get("dernierRapportReference") or {}
        if isinstance(rapport, dict):
            odds = rapport.get("rapport")
        gains = (p.get("gainsParticipant") or {}).get("gainsCarriere")
        if gains is not None:
            gains = gains / 100.0  # centimes -> euros
        rows.append(
            {
                "race_id": ref.race_id,
                "race_date": ref.day.isoformat(),
                "track": ref.track,
                "discipline": ref.discipline,
                "distance_m": ref.distance_m,
                "going": "",
                "horse": p.get("nom", "").title(),
                "jockey": p.get("driver") or p.get("jockey") or "",
                "trainer": p.get("entraineur", ""),
                "draw": p.get("placeCorde") or p.get("numPmu"),
                "weight_kg": (p.get("handicapPoids") or 0) / 10.0 if p.get("handicapPoids") else None,
                "age": p.get("age"),
                "days_since_last_run": None,  # non fourni de façon fiable par cette source
                "musique": p.get("musique", ""),
                "career_starts": p.get("nombreCourses"),
                "career_wins": p.get("nombreVictoires"),
                "career_places": p.get("nombrePlaces"),
                "earnings": gains,
                "odds": odds,
                "num_pmu": p.get("numPmu"),
            }
        )
    if not rows:
        raise PMUFetchError("Tous les chevaux sont non-partants.")
    return pd.DataFrame(rows)


def fetch_results(ref: RaceRef, user_agent: str = "HorseProbaApp/1.0 (educational)") -> Dict[int, int]:
    """
    Récupère l'ordre d'arrivée (num PMU -> place). Dict vide si la course n'est pas
    encore courue. Utile pour compléter `finish_position` dans un historique.
    """
    url = f"{BASE_URL}/programme/{_date_path(ref.day)}/R{ref.reunion}/C{ref.course}/rapports-definitifs"
    try:
        data = _get_json(url, user_agent)
    except PMUFetchError:
        return {}
    arrival: Dict[int, int] = {}
    # Le pari "SIMPLE_GAGNANT" contient la combinaison gagnante ; l'ordre d'arrivée
    # complet est dans la clé "ordreArrivee" du programme si disponible.
    for pari in data if isinstance(data, list) else []:
        if pari.get("typePari") == "SIMPLE_GAGNANT":
            for rap in pari.get("rapports") or []:
                for i, num in enumerate(rap.get("combinaison", "").split("-")):
                    if num.strip().isdigit():
                        arrival[int(num)] = i + 1
    return arrival
