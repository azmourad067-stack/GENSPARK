"""
🏇 Application de Pronostics Hippiques Intelligente
Développée avec Streamlit — Prête pour déploiement sur Streamlit Cloud

Architecture :
  modules/ocr_extractor.py   → Extraction OCR (Gemini / OpenAI / EasyOCR)
  modules/data_cleaner.py    → Nettoyage & structuration des données
  modules/scorer.py          → Algorithme de scoring
  modules/pronostic.py       → Génération des pronostics
  modules/visualizer.py      → Graphiques Plotly
"""

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import io
import os
import time
import json

# Modules internes
from modules.ocr_extractor import extract_data_from_image, merge_extracted_data
from modules.data_cleaner import clean_horse_data, assess_data_quality
from modules.scorer import calculate_scores, get_score_breakdown
from modules.pronostic import generate_pronostic_report, format_combinations
from modules.visualizer import (
    plot_scores_bar,
    plot_radar_top3,
    plot_form_history,
    plot_driver_comparison,
    plot_confidence_gauge,
)

# ─────────────────────────────────────────────
# Configuration Streamlit
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="🏇 PronoHippique AI",
    page_icon="🏇",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CSS personnalisé
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Thème global */
    :root {
        --primary: #1a6b3c;
        --secondary: #2c9e5e;
        --accent: #f28a00;
        --bg-light: #f0f7f3;
        --bg-dark: #0d3320;
        --text: #1a1a2e;
    }

    /* En-tête principal */
    .main-header {
        background: linear-gradient(135deg, #0d3320 0%, #1a6b3c 50%, #2c9e5e 100%);
        color: white;
        padding: 2rem 2.5rem;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(13, 51, 32, 0.4);
    }
    .main-header h1 { font-size: 2.8rem; margin: 0; letter-spacing: 2px; }
    .main-header p  { font-size: 1.1rem; margin: 0.5rem 0 0; opacity: 0.88; }

    /* Cartes */
    .card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 16px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
        border-left: 5px solid #1a6b3c;
    }
    .card-accent { border-left-color: #f28a00; }
    .card-danger { border-left-color: #e74c3c; }
    .card-gold   { border-left-color: #ffd700; background: #fffef0; }

    /* Podium */
    .podium-1 { background: linear-gradient(135deg, #fff7d6, #ffe66d); border: 2px solid #ffd700; border-radius: 12px; padding: 1rem 1.5rem; }
    .podium-2 { background: linear-gradient(135deg, #f8f8f8, #e8e8e8); border: 2px solid #c0c0c0; border-radius: 12px; padding: 1rem 1.5rem; }
    .podium-3 { background: linear-gradient(135deg, #fff3e0, #ffcc90); border: 2px solid #cd7f32; border-radius: 12px; padding: 1rem 1.5rem; }

    /* Badges */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.15rem;
    }
    .badge-green  { background: #d4edda; color: #155724; }
    .badge-orange { background: #fff3cd; color: #856404; }
    .badge-blue   { background: #d1ecf1; color: #0c5460; }
    .badge-red    { background: #f8d7da; color: #721c24; }

    /* Bouton principal */
    .stButton > button {
        background: linear-gradient(135deg, #1a6b3c, #2c9e5e);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 700;
        cursor: pointer;
        transition: all 0.2s;
        box-shadow: 0 4px 12px rgba(26, 107, 60, 0.3);
        width: 100%;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #0d3320, #1a6b3c);
        box-shadow: 0 6px 20px rgba(26, 107, 60, 0.5);
        transform: translateY(-1px);
    }

    /* Upload zone */
    .upload-zone {
        border: 2px dashed #2c9e5e;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        background: #f0f7f3;
        margin: 1rem 0;
    }

    /* Table style */
    .dataframe { border-radius: 10px; overflow: hidden; }

    /* Combo */
    .combo-box {
        background: #f0f7f3;
        border: 1px solid #2c9e5e;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        margin: 0.25rem 0;
        font-family: monospace;
        font-size: 1rem;
        font-weight: 600;
        color: #0d3320;
    }
    .combo-box:nth-child(1) { background: #fff7d6; border-color: #ffd700; }
    .combo-box:nth-child(2) { background: #f8f8f8; border-color: #c0c0c0; }

    /* Sidebar */
    .sidebar-section {
        background: #f0f7f3;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #2c9e5e;
    }

    /* Séparateur */
    hr { border: none; border-top: 2px solid #e8f5ee; margin: 1.5rem 0; }

    /* Score bar */
    .score-fill {
        height: 8px;
        border-radius: 4px;
        background: linear-gradient(90deg, #1a6b3c, #5ab87e);
    }

    /* Metrics */
    [data-testid="metric-container"] {
        background: white;
        border: 1px solid #e8f5ee;
        border-radius: 10px;
        padding: 0.75rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Initialisation session state
# ─────────────────────────────────────────────
if "df_cleaned" not in st.session_state:
    st.session_state.df_cleaned = None
if "df_scored" not in st.session_state:
    st.session_state.df_scored = None
if "pronostic" not in st.session_state:
    st.session_state.pronostic = None
if "raw_extractions" not in st.session_state:
    st.session_state.raw_extractions = []
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False


# ─────────────────────────────────────────────
# SIDEBAR — Configuration
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:1rem 0;'>
        <span style='font-size:3rem;'>🏇</span>
        <h2 style='color:#1a6b3c; margin:0.5rem 0;'>PronoHippique AI</h2>
        <p style='color:#666; font-size:0.85rem;'>Pronostics intelligents par IA</p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── Configuration OCR ──
    st.markdown("### ⚙️ Configuration OCR")

    ocr_engine = st.radio(
        "Moteur d'extraction",
        ["🤖 Google Gemini (Recommandé)", "🧠 OpenAI GPT-4o", "📷 EasyOCR (Local)"],
        help="Gemini et OpenAI nécessitent une clé API. EasyOCR est entièrement local (moins précis)."
    )

    gemini_key = ""
    openai_key = ""

    if "Gemini" in ocr_engine:
        gemini_key = st.text_input(
            "Clé API Google Gemini",
            type="password",
            placeholder="AIza...",
            help="Obtenez votre clé sur https://aistudio.google.com"
        )
        if not gemini_key:
            st.info("💡 Sans clé API, l'app utilisera EasyOCR en fallback.")

    elif "OpenAI" in ocr_engine:
        openai_key = st.text_input(
            "Clé API OpenAI",
            type="password",
            placeholder="sk-...",
            help="Obtenez votre clé sur https://platform.openai.com"
        )

    st.divider()

    # ── Type de course ──
    st.markdown("### 🎯 Type de Course")
    race_type = st.selectbox(
        "Sélectionner le type",
        ["default", "quinté", "prix", "trot"],
        format_func=lambda x: {
            "default": "🏇 Course standard",
            "quinté": "🌟 Quinté+",
            "prix": "🏆 Grand Prix",
            "trot": "🏇 Trot Attelé",
        }.get(x, x)
    )

    st.divider()

    # ── Paramètres avancés ──
    with st.expander("🔧 Paramètres Avancés"):
        n_recent_musique = st.slider("Courses récentes pour musique", 3, 10, 5)
        show_raw_data = st.checkbox("Afficher données brutes OCR", False)
        show_score_detail = st.checkbox("Afficher détail des scores", True)
        confidence_min = st.slider("Score minimum à afficher", 0.0, 5.0, 0.0, 0.5)

    st.divider()

    # ── Infos ──
    st.markdown("""
    <div class='sidebar-section'>
        <strong>📋 Types d'images supportés</strong><br>
        <small>
        ✅ Liste des partants<br>
        ✅ Records absolus<br>
        ✅ Statistiques drivers<br>
        ✅ Statistiques entraîneurs<br>
        ✅ Plusieurs images par course
        </small>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# EN-TÊTE PRINCIPAL
# ─────────────────────────────────────────────
st.markdown("""
<div class='main-header'>
    <h1>🏇 PronoHippique AI</h1>
    <p>Intelligence Artificielle pour vos pronostics hippiques — Analysez, Scorez, Gagnez !</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SECTION 1 — UPLOAD DES IMAGES
# ─────────────────────────────────────────────
st.markdown("## 📤 Étape 1 — Téléchargez vos captures d'écran")

st.markdown("""
<div class='card'>
    <h3>💡 Instructions</h3>
    <p>Téléchargez <strong>1 à 4 captures d'écran</strong> de statistiques hippiques de la même course :</p>
    <ul>
        <li>📊 <strong>Liste des partants</strong> (avec cotes, musique, gains)</li>
        <li>🏆 <strong>Records absolus</strong> (meilleure performance)</li>
        <li>🏇 <strong>Statistiques drivers</strong> (courses, victoires, %)</li>
        <li>👨‍🏫 <strong>Statistiques entraîneurs</strong> (courses, victoires, %)</li>
    </ul>
    <p><em>Plus vous uploadez d'images, plus l'analyse sera précise !</em></p>
</div>
""", unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "📷 Glissez vos images ici ou cliquez pour sélectionner",
    type=["png", "jpg", "jpeg", "webp"],
    accept_multiple_files=True,
    help="Formats supportés: PNG, JPG, JPEG, WEBP — Maximum 4 images recommandé",
    label_visibility="visible",
)

# Aperçu des images
if uploaded_files:
    st.markdown(f"**{len(uploaded_files)} image(s) chargée(s)** ✅")
    cols = st.columns(min(len(uploaded_files), 4))
    for i, f in enumerate(uploaded_files):
        with cols[i % 4]:
            img = Image.open(f)
            st.image(img, caption=f.name, use_column_width=True)
            st.markdown(f"<small>📐 {img.size[0]}×{img.size[1]}px</small>", unsafe_allow_html=True)

st.divider()


# ─────────────────────────────────────────────
# SECTION 2 — BOUTON ANALYSER
# ─────────────────────────────────────────────
st.markdown("## 🧠 Étape 2 — Lancer l'Analyse")

col_btn, col_info = st.columns([2, 3])
with col_btn:
    analyze_clicked = st.button("🚀 Analyser la Course", use_container_width=True)

with col_info:
    if not uploaded_files:
        st.warning("⚠️ Veuillez d'abord télécharger au moins une image.")
    elif "Gemini" in ocr_engine and not gemini_key:
        st.warning("⚠️ Aucune clé Gemini fournie → EasyOCR sera utilisé (précision réduite).")
    elif "OpenAI" in ocr_engine and not openai_key:
        st.warning("⚠️ Aucune clé OpenAI fournie → EasyOCR sera utilisé (précision réduite).")
    else:
        st.success("✅ Prêt pour l'analyse !")


# ─────────────────────────────────────────────
# LOGIQUE PRINCIPALE D'ANALYSE
# ─────────────────────────────────────────────
if analyze_clicked and uploaded_files:
    st.session_state.analysis_done = False

    # ── Barre de progression ──
    progress_bar = st.progress(0)
    status_text = st.empty()

    extractions = []
    total_steps = len(uploaded_files) + 3  # OCR + nettoyage + scoring + pronostic

    for i, f in enumerate(uploaded_files):
        status_text.markdown(f"🔍 **Extraction OCR** — Image {i+1}/{len(uploaded_files)}: `{f.name}`...")
        progress_bar.progress(int(i / total_steps * 100))

        img = Image.open(f).convert("RGB")

        # Choisir l'engine OCR
        result = extract_data_from_image(
            image=img,
            gemini_api_key=gemini_key,
            openai_api_key=openai_key,
            use_easyocr=True,
        )
        extractions.append(result)
        time.sleep(0.2)

    st.session_state.raw_extractions = extractions

    # ── Fusion des données ──
    step = len(uploaded_files)
    progress_bar.progress(int(step / total_steps * 100))
    status_text.markdown("🔀 **Fusion** des données extraites...")
    merged = merge_extracted_data(extractions)
    time.sleep(0.3)

    # ── Nettoyage ──
    step += 1
    progress_bar.progress(int(step / total_steps * 100))
    status_text.markdown("🧹 **Nettoyage** et structuration des données...")
    df_clean = clean_horse_data(merged.get("chevaux", []))
    st.session_state.df_cleaned = df_clean
    time.sleep(0.3)

    # ── Scoring ──
    step += 1
    progress_bar.progress(int(step / total_steps * 100))
    status_text.markdown("📊 **Calcul des scores**...")
    df_scored = calculate_scores(df_clean, race_type=race_type)
    st.session_state.df_scored = df_scored
    time.sleep(0.3)

    # ── Pronostic ──
    step += 1
    progress_bar.progress(100)
    status_text.markdown("🎯 **Génération du pronostic**...")
    pronostic = generate_pronostic_report(df_scored)
    st.session_state.pronostic = pronostic
    st.session_state.analysis_done = True
    time.sleep(0.2)

    progress_bar.empty()
    status_text.success("✅ Analyse terminée avec succès !")
    time.sleep(0.5)
    status_text.empty()

    st.rerun()


# ─────────────────────────────────────────────
# SECTION 3 — RÉSULTATS
# ─────────────────────────────────────────────
if st.session_state.analysis_done and st.session_state.df_scored is not None:

    df = st.session_state.df_scored
    pronostic = st.session_state.pronostic
    n_partants = len(df)

    st.divider()
    st.markdown("## 📊 Résultats de l'Analyse")

    # ── Métriques globales ──
    qual = assess_data_quality(df)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🐎 Partants détectés", n_partants)
    with col2:
        score_max = df["score_global"].max() if not df.empty else 0
        favori_name = df.loc[df["score_global"].idxmax(), "cheval"] if not df.empty else "-"
        st.metric("🏆 Favori IA", f"#{int(df.loc[df['score_global'].idxmax(), 'numero'])}" if not df.empty else "-", favori_name)
    with col3:
        st.metric("📈 Qualité données", f"{qual.get('qualite', 0)}%")
    with col4:
        engine_used = st.session_state.raw_extractions[0].get("ocr_engine", "?") if st.session_state.raw_extractions else "?"
        st.metric("🤖 Moteur OCR", engine_used.split(" ")[0] if engine_used else "?")

    st.divider()

    # ─────────────────────────────────────────
    # ONGLETS RÉSULTATS
    # ─────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🏆 Pronostic", "📊 Classement", "📈 Graphiques",
        "🔍 Données", "🎰 Combinaisons", "📋 Détail Scores"
    ])

    # ─────────────────────────────────────────
    # TAB 1 — PRONOSTIC
    # ─────────────────────────────────────────
    with tab1:
        st.markdown("### 🎯 Pronostic Intelligent")

        # Confiance
        confiance = pronostic.get("confiance", "?")
        col_gauge, col_conf = st.columns([1, 2])
        with col_gauge:
            gap = df["score_global"].max() - df["score_global"].nlargest(2).iloc[-1] if len(df) >= 2 else 0
            gauge_val = min(10.0, 5.0 + gap * 1.5)
            st.plotly_chart(
                plot_confidence_gauge(round(gauge_val, 1), "Indice de Confiance"),
                use_container_width=True,
            )
        with col_conf:
            st.markdown(f"""
            <div class='card'>
                <h3>📌 Niveau de Confiance</h3>
                <p style='font-size:1.4rem; font-weight:700; color:#1a6b3c;'>{confiance}</p>
                <p>Basé sur l'écart de score entre le favori IA et ses poursuivants.<br>
                Un fort écart indique un favori très dominant.</p>
                <p><strong>Partants analysés :</strong> {n_partants}</p>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        # ── PODIUM ──
        st.markdown("### 🥇 Top 3 Conseillé")
        top3 = pronostic.get("top3", [])
        podium_styles = ["podium-1", "podium-2", "podium-3"]
        medals = ["🥇", "🥈", "🥉"]

        cols_podium = st.columns(3)
        for i, horse in enumerate(top3[:3]):
            with cols_podium[i]:
                st.markdown(f"""
                <div class='{podium_styles[i]}'>
                    <div style='font-size:2rem; text-align:center;'>{medals[i]}</div>
                    <h3 style='text-align:center; margin:0.3rem 0;'>
                        #{horse['numero']} {horse['cheval']}
                    </h3>
                    <p style='text-align:center; font-size:1.2rem; font-weight:700; color:#1a6b3c;'>
                        Score: {horse['score_global']:.2f}/10
                    </p>
                    <p style='text-align:center; font-size:0.9rem;'>{horse.get('categorie','')}</p>
                </div>
                """, unsafe_allow_html=True)

        st.divider()

        # ── BASES ──
        col_bases, col_outsiders = st.columns(2)

        with col_bases:
            st.markdown("### 💎 2 Bases Solides")
            bases = pronostic.get("bases", [])
            for horse in bases:
                st.markdown(f"""
                <div class='card card-gold'>
                    <strong>#{horse['numero']} {horse['cheval']}</strong>
                    <span class='badge badge-green'>Score: {horse['score_global']:.2f}</span>
                </div>
                """, unsafe_allow_html=True)

        with col_outsiders:
            st.markdown("### 💡 3-5 Outsiders Intéressants")
            outsiders = pronostic.get("outsiders", [])
            for horse in outsiders:
                cote_disp = f"Cote {horse.get('cote_pmu', '?')}" if horse.get('cote_pmu', 0) > 0 else ""
                st.markdown(f"""
                <div class='card card-accent'>
                    <strong>#{horse['numero']} {horse['cheval']}</strong>
                    <span class='badge badge-orange'>Score: {horse['score_global']:.2f}</span>
                    {f"<span class='badge badge-blue'>{cote_disp}</span>" if cote_disp else ""}
                </div>
                """, unsafe_allow_html=True)

        st.divider()

        # ── ARGUMENTS ──
        st.markdown("### 💬 Analyse Argumentée du Top 5")
        arguments = pronostic.get("arguments", {})
        for horse_name, args in list(arguments.items())[:5]:
            with st.expander(f"🏇 {horse_name}"):
                for arg in args:
                    st.markdown(f"- {arg}")


    # ─────────────────────────────────────────
    # TAB 2 — CLASSEMENT
    # ─────────────────────────────────────────
    with tab2:
        st.markdown("### 📊 Classement Complet par Score IA")

        classement_data = []
        for _, row in df.sort_values("score_global", ascending=False).iterrows():
            classement_data.append({
                "Rang": int(row.get("rang_score", 0)),
                "N°": int(row.get("numero", 0)),
                "Cheval": row.get("cheval", ""),
                "Score": f"{row['score_global']:.2f}",
                "Catégorie": row.get("categorie", ""),
                "Driver": row.get("driver", ""),
                "Entraîneur": row.get("entraineur", ""),
                "% Driver": f"{row.get('reussite_driver', 0):.0f}%",
                "% Entr.": f"{row.get('reussite_entraineur', 0):.0f}%",
                "Cote PMU": row.get("cote_pmu", 0) if row.get("cote_pmu", 0) > 0 else "-",
                "Écart": int(row.get("ecart_driver", 99)) if row.get("ecart_driver", 99) < 99 else "—",
            })

        df_display = pd.DataFrame(classement_data)

        st.dataframe(
            df_display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Rang": st.column_config.NumberColumn("🥇", width="small"),
                "Score": st.column_config.TextColumn("⭐ Score"),
                "Catégorie": st.column_config.TextColumn("Statut"),
            }
        )

        # Exporter CSV
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Exporter les données (CSV)",
            data=csv,
            file_name="pronostic_hippique.csv",
            mime="text/csv",
        )


    # ─────────────────────────────────────────
    # TAB 3 — GRAPHIQUES
    # ─────────────────────────────────────────
    with tab3:
        st.markdown("### 📈 Visualisations")

        # Scores globaux
        st.plotly_chart(plot_scores_bar(df), use_container_width=True)

        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.plotly_chart(plot_radar_top3(df), use_container_width=True)
        with col_g2:
            st.plotly_chart(plot_driver_comparison(df), use_container_width=True)

        # Historique de forme
        st.plotly_chart(plot_form_history(df, top_n=5), use_container_width=True)


    # ─────────────────────────────────────────
    # TAB 4 — DONNÉES BRUTES
    # ─────────────────────────────────────────
    with tab4:
        st.markdown("### 🔍 Données Extraites et Nettoyées")

        # Qualité des données
        qual = assess_data_quality(df)
        st.markdown(f"""
        <div class='card'>
            <strong>📊 Qualité des données extraites : {qual.get('qualite', 0)}%</strong>
            <p>Nombre de partants : {qual.get('nb_chevaux', 0)}</p>
        </div>
        """, unsafe_allow_html=True)

        # Tableau données principales
        cols_afficher = [
            "numero", "cheval", "sa", "driver", "entraineur",
            "record_brut", "reussite_driver", "reussite_entraineur",
            "ecart_driver", "gains", "cote_pmu", "musique"
        ]
        cols_present = [c for c in cols_afficher if c in df.columns]
        st.dataframe(df[cols_present], use_container_width=True, hide_index=True)

        # Données brutes OCR
        if show_raw_data:
            st.markdown("#### 📝 Réponses brutes OCR")
            for i, ext in enumerate(st.session_state.raw_extractions):
                with st.expander(f"Image {i+1} — OCR brut"):
                    st.json(ext)


    # ─────────────────────────────────────────
    # TAB 5 — COMBINAISONS
    # ─────────────────────────────────────────
    with tab5:
        st.markdown("### 🎰 Combinaisons de Paris")

        trios = pronostic.get("trios", [])
        quintes = pronostic.get("quintes", [])

        col_trio, col_quinte = st.columns(2)

        with col_trio:
            st.markdown("#### 🎯 10 Combinaisons Trio")
            st.markdown("""
            <div class='card'>
                <p><small>Ces combinaisons couvrent les scénarios les plus probables
                basés sur les scores IA. La 1ère combinaison est la plus conseillée.</small></p>
            </div>
            """, unsafe_allow_html=True)

            for i, combo in enumerate(trios, 1):
                nums_str = "  —  ".join(str(n) for n in sorted(combo))
                color = "#fff7d6" if i == 1 else ("#f8f8f8" if i == 2 else "#f0f7f3")
                border = "#ffd700" if i == 1 else ("#c0c0c0" if i == 2 else "#2c9e5e")
                emoji = "🥇" if i == 1 else ("🥈" if i == 2 else "▶️")
                st.markdown(f"""
                <div style='background:{color}; border:2px solid {border}; border-radius:8px;
                            padding:0.5rem 1rem; margin:0.2rem 0; font-family:monospace;
                            font-size:1.1rem; font-weight:700;'>
                    {emoji} Trio {i} :  [ {nums_str} ]
                </div>
                """, unsafe_allow_html=True)

        with col_quinte:
            st.markdown("#### 🌟 10 Combinaisons Quinté+")
            st.markdown("""
            <div class='card'>
                <p><small>Combinaisons Quinté+ optimisées. Jouez les 3-5 premières
                pour maximiser vos chances dans le bon ordre ou désordre.</small></p>
            </div>
            """, unsafe_allow_html=True)

            for i, combo in enumerate(quintes, 1):
                nums_str = "  —  ".join(str(n) for n in sorted(combo))
                color = "#fff7d6" if i == 1 else ("#f8f8f8" if i == 2 else "#f0f7f3")
                border = "#ffd700" if i == 1 else ("#c0c0c0" if i == 2 else "#2c9e5e")
                emoji = "🥇" if i == 1 else ("🥈" if i == 2 else "▶️")
                st.markdown(f"""
                <div style='background:{color}; border:2px solid {border}; border-radius:8px;
                            padding:0.5rem 1rem; margin:0.2rem 0; font-family:monospace;
                            font-size:1.1rem; font-weight:700;'>
                    {emoji} Quinté {i} :  [ {nums_str} ]
                </div>
                """, unsafe_allow_html=True)

        st.divider()

        # Récapitulatif des bases
        st.markdown("#### 🔑 Résumé des Bases")
        bases = pronostic.get("bases", [])
        if bases:
            bases_nums = " et ".join([f"**#{b['numero']} {b['cheval']}**" for b in bases])
            st.success(f"💎 Bases recommandées : {bases_nums}")

        outsiders = pronostic.get("outsiders", [])
        if outsiders:
            out_nums = ", ".join([f"#{o['numero']}" for o in outsiders])
            st.info(f"💡 Outsiders à inclure : {out_nums}")


    # ─────────────────────────────────────────
    # TAB 6 — DÉTAIL SCORES
    # ─────────────────────────────────────────
    with tab6:
        st.markdown("### 📋 Détail des Scores par Critère")

        if show_score_detail:
            for _, row in df.sort_values("score_global", ascending=False).head(10).iterrows():
                breakdown = get_score_breakdown(row)
                horse_name = f"#{int(row['numero'])} {row['cheval']}"
                global_score = row["score_global"]

                with st.expander(f"{horse_name} — Score Global: {global_score:.2f}/10"):
                    cols = st.columns(3)
                    items = list(breakdown.items())
                    for i, (criterion, score) in enumerate(items):
                        with cols[i % 3]:
                            pct = int(score / 10 * 100)
                            color = "#1a6b3c" if score >= 7 else ("#f28a00" if score >= 4 else "#e74c3c")
                            st.markdown(f"""
                            <div style='margin:0.3rem 0;'>
                                <small><strong>{criterion}</strong></small>
                                <div style='background:#e8f5ee; border-radius:4px; height:8px; margin:3px 0;'>
                                    <div style='background:{color}; width:{pct}%; height:8px; border-radius:4px;'></div>
                                </div>
                                <small>{score:.1f}/10</small>
                            </div>
                            """, unsafe_allow_html=True)
        else:
            st.info("Activez 'Afficher détail des scores' dans la barre latérale pour voir ce panneau.")


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.divider()
st.markdown("""
<div style='text-align:center; color:#888; font-size:0.85rem; padding:1rem;'>
    🏇 <strong>PronoHippique AI</strong> — Développé avec Streamlit & IA Vision<br>
    <em>⚠️ Avertissement : Les pronostics sont fournis à titre indicatif uniquement.
    Le jeu peut créer une dépendance. Jouez de manière responsable.</em><br>
    <small>Données analysées via OCR IA — Résultats non garantis</small>
</div>
""", unsafe_allow_html=True)
