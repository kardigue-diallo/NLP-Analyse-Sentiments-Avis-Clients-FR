import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import re
from datetime import datetime

# ==========================================
# 1. CONFIGURATION DE LA PAGE
# ==========================================
st.set_page_config(
    page_title="Plateforme Décisionnelle & Explicabilité NLP",
    page_icon="🤖",
    layout="wide"
)

# Style CSS personnalisé
st.markdown("""
<style>
    .main { padding: 1rem 2rem; }
    .explain-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #0066cc;
        margin-bottom: 10px;
    }
    .receipt-box {
        background-color: #ffffff;
        border: 2px dashed #cbd5e1;
        padding: 20px;
        border-radius: 10px;
        font-family: 'Courier New', Courier, monospace;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. CHARGEMENT DES MODÈLES ET FONCTIONS
# ==========================================
@st.cache_resource
def charger_ressources():
    modele = joblib.load('model.pkl')
    vectorizer = joblib.load('tfidf.pkl')
    return modele, vectorizer

try:
    modele, vectorizer = charger_ressources()
except Exception as e:
    st.error(f"Erreur lors du chargement des fichiers 'model.pkl' ou 'tfidf.pkl' : {e}")
    st.stop()

# Liste des mots vides (Stopwords FR)
stop_words_fr = set([
    'le', 'la', 'les', 'de', 'du', 'des', 'un', 'une', 'et', 'en', 'est', 
    'que', 'qui', 'dans', 'pour', 'pas', 'sur', 'ce', 'cette', 'au', 'aux'
])

def nettoyer_texte_fr(texte):
    if not isinstance(texte, str):
        return ""
    texte = texte.lower()
    texte = re.sub(r'[^a-zàâäéèêëîïôöùûüç\s]', ' ', texte)
    texte = re.sub(r'\s+', ' ', texte).strip()
    mots = texte.split()
    mots_nettoyes = [m for m in mots if m not in stop_words_fr and len(m) > 1]
    return " ".join(mots_nettoyes)

def detecter_negations(texte):
    mots_negation = ['ne', 'pas', 'non', 'plus', 'jamais', 'rien', 'aucun', 'aucune']
    mots = re.findall(r'\b\w+\b', texte.lower())
    return [m for m in mots if m in mots_negation]

# ==========================================
# 3. BARRE LATÉRALE (SIDEBAR)
# ==========================================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/4712/4712035.png", width=80)
st.sidebar.title("⚙️ Panneau de Contrôle")
mode = st.sidebar.radio("Choisissez le mode d'analyse :", ["📝 Analyse d'un Avis Unique", "📁 Analyse par Fichier (Batch CSV)"])

st.sidebar.divider()
st.sidebar.info("💡 **Système Hybride** Combine la puissance statistique de la Régression Logistique et la précision des règles métier de Négation.")

# ==========================================
# 4. TRAITEMENT DE L'AVIS UNIQUE
# ==========================================
if mode == "📝 Analyse d'un Avis Unique":
    st.title("🎯 Analyse de Sentiment & Explicabilité Machine Learning")
    
    col_input, col_pred = st.columns([1.2, 1])
    
    with col_input:
        st.subheader("📥 Saisie de l'Avis Client")
        c_d, c_h = st.columns(2)
        date_msg = c_d.date_input("Date du message :", datetime.now())
        heure_msg = c_h.time_input("Heure du message :", datetime.now().time())
        
        avis = st.text_area(
            "Texte de l'avis client :", 
            value="Le talon est très magnifique dans la photo, vraiment c'est bon, le prix",
            height=120
        )
        btn_analyser = st.button("🚀 Analyser l'Avis", use_container_width=True, type="primary")

    if btn_analyser or avis:
        dt_client = datetime.combine(date_msg, heure_msg)
        dt_traitement = datetime.now()
        ticket_id = f"TK-{np.random.randint(100000, 999999)}"
        
        # Pretraitement
        texte_clean = nettoyer_texte_fr(avis)
        negations_trouvees = detecter_negations(avis)
        override_negatif = len(negations_trouvees) > 0
        
        # Prédiction TF-IDF + Régression Logistique
        X_vec = vectorizer.transform([texte_clean])
        proba_pos_brut = modele.predict_proba(X_vec)[0][1]
        
        # Calcul de l'explicabilité (XAI)
        feature_names = vectorizer.get_feature_names_out()
        coefs = modele.coef_[0]
        feature_index = {feat: idx for idx, feat in enumerate(feature_names)}
        
        mots_presents = texte_clean.split()
        details_mots = []
        somme_contributions = 0.0
        
        for m in mots_presents:
            if m in feature_index:
                idx = feature_index[m]
                tfidf_val = X_vec[0, idx]
                weight_val = coefs[idx]
                contrib = tfidf_val * weight_val
                somme_contributions += contrib
                details_mots.append({
                    "mot": m,
                    "tfidf": round(tfidf_val, 4),
                    "poids": round(weight_val, 4),
                    "contrib": round(contrib, 4)
                })

        with col_pred:
            st.subheader("🎯 Résultat de la Prédiction")
            if override_negatif:
                st.error("🔴 Sentiment final : NÉGATIF (Forcé par la règle de Négation)")
                st.progress(0.10)
                st.caption(f"Score IA Brut : {proba_pos_brut:.2%} | Ajustement : Négation détectée")
            elif proba_pos_brut >= 0.45:
                st.success(f"🟢 Sentiment final : POSITIF (Confiance : {proba_pos_brut:.2%})")
                st.progress(proba_pos_brut)
                st.caption(f"Note : {proba_pos_brut:.2%}")
            else:
                st.error(f"🔴 Sentiment final : NÉGATIF (Confiance : {(1 - proba_pos_brut):.2%})")
                st.progress(proba_pos_brut)
                st.caption(f"Note : {proba_pos_brut:.2%}")

        st.divider()

        # ==========================================
        # 5. DÉFINITION DES VARIABLES DU REÇU (Résolution Bug NameError)
        # ==========================================
        neg_status = f"Oui ({', '.join(negations_trouvees)})" if negations_trouvees else "Non"
        adj_status = "Appliqué (Forcé en NÉGATIF)" if override_negatif else "Aucun"
        
        if override_negatif or proba_pos_brut < 0.45:
            verdict = "NÉGATIF 🔴"
            verdict_txt = "NEGATIF [ALERT]"
            action_recommandee = "Ouverture ticket Support Prioritaire / Remplacement du Produit."
            priorite = "HAUTE 🔴"
        else:
            verdict = "POSITIF 🟢"
            verdict_txt = "POSITIF [OK]"
            action_recommandee = "Envoi de message de remerciement automatisé."
            priorite = "BASSE 🟢"

        # ==========================================
        # 6. ONGLETS DE DÉTAILS
        # ==========================================
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 1. Détail Mots & Impact Visual", 
            "🧮 2. Décomposition Mathématique (TF-IDF & Poids)", 
            "🛠️ 3. Action Métier & Réponse Auto", 
            "📄 4. Reçu & Exportation"
        ])

        # --- TAB 1 : DÉTAIL MOTS & GRAPHIQUE ---
        with tab1:
            st.markdown("### Impact Individuel des Mots (XAI)")
            st.code(f"Texte Nettoyé : {texte_clean}")
            
            col_t1, col_t2 = st.columns([1.1, 1])
            with col_t1:
                st.markdown("#### Tableau des Contributions")
                if details_mots:
                    df_mots = pd.DataFrame(details_mots)
                    st.dataframe(df_mots, use_container_width=True)
                else:
                    st.info("Aucun mot de cet avis ne figure dans le vocabulaire TF-IDF entraîné.")
            
            with col_t2:
                st.markdown("#### Graphique d'Impact par Mot")
                if details_mots:
                    df_mots_sorted = pd.DataFrame(details_mots).sort_values(by="contrib")
                    colors = ['#ef4444' if x < 0 else '#22c55e' for x in df_mots_sorted['contrib']]
                    fig = go.Figure(go.Bar(
                        x=df_mots_sorted['contrib'],
                        y=df_mots_sorted['mot'],
                        orientation='h',
                        marker_color=colors,
                        text=df_mots_sorted['contrib'],
                        textposition='auto'
                    ))
                    fig.update_layout(xaxis_title="Contribution (X × W)", yaxis_title="Mots du texte", height=300)
                    st.plotly_chart(fig, use_container_width=True)

        # --- TAB 2 : DÉCOMPOSITION MATHÉMATIQUE (LaTeX Fixé) ---
        with tab2:
            st.markdown("### 🔍 Comment sont calculées les valeurs TF-IDF et les Poids $W$ ?")
            
            c_exp1, c_exp2 = st.columns(2)
            with c_exp1:
                st.markdown("""
                <div class="explain-card">
                    <h4>1️⃣ Score TF-IDF ($X$)</h4>
                    <p>Mesure l'importance d'un mot dans le texte relatif au jeu d'apprentissage :</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.latex(r"\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t)")
                
                st.markdown("""
                - **TF (Term Frequency) :** Nombre d'occurrences du **mot entier** dans cet avis client.
                - **IDF (Inverse Document Frequency) :** Rareté du mot dans tout le jeu d'entraînement. Calculé via :
                """)
                st.latex(r"\text{IDF}(t) = \ln\left(\frac{1+N}{1+DF}\right) + 1")

            with c_exp2:
                st.markdown("""
                <div class="explain-card">
                    <h4>2️⃣ Poids du Modèle ($W$)</h4>
                    <p>Appris durant l'entraînement de la Régression Logistique :</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                - **$W > 0$ (Positif) :** Mots fortement associés au contentement client (ex: *satisfait* = $+0.8074$).
                - **$W < 0$ (Négatif) :** Mots associés à une insatisfaction (ex: *cassé* = $-0.6925$).
                - **$W \\approx 0$ (Neutre) :** Mots sans impact majeur sur la polarité.
                """)

            st.divider()
            
            if details_mots:
                mot_ex = details_mots[0]
                st.markdown(f"#### ⚠️ Exemple sur le mot : `{mot_ex['mot']}`")
                st.latex(rf"\text{{Contribution}} = \text{{TF-IDF}} \times \text{{Poids}} = {mot_ex['tfidf']} \times {mot_ex['poids']} = \mathbf{{{mot_ex['contrib']}}}")

            st.markdown("#### Calcul Global du Score $z$ et Probabilité Sigmoïde :")
            biais = float(modele.intercept_[0])
            z_total = somme_contributions + biais

            st.latex(r"z = \text{Biais} + \sum (\text{TF-IDF}_i \times \text{Poids}_i)")
            pct_brut = proba_pos_brut * 100
            st.latex(rf"P(\text{{Positif}}) = \frac{{1}}{{1 + e^{{-({z_total:.4f})}}}} = \mathbf{{{pct_brut:.2f}\%}}")

        # --- TAB 3 : ACTION MÉTIER & RÉPONSE AUTO ---
        with tab3:
            st.markdown("### 🛠️ Décision Métier & Génération de Réponse Automatique")
            
            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("Mots dans l'avis", len(avis.split()))
            col_m2.metric("Mots reconnus (vocabulaire)", len(details_mots))
            col_m3.metric("Temps de lecture estimé", f"{len(avis.split()) * 0.3:.1f}s")
            
            if "NÉGATIF" in verdict:
                st.error(f"Priorité {priorite} : {action_recommandee}")
                rep_auto = f"Objet : Assistance concernant votre commande [{ticket_id}]\nBonjour,\nNous avons bien pris en compte votre retour négatif. Un conseiller du service client traite votre demande en priorité pour résoudre ce problème au plus vite.\nCordialement,\nLe Service Client"
            else:
                st.success(f"Priorité {priorite} : {action_recommandee}")
                rep_auto = f"Objet : Merci pour votre retour ! [{ticket_id}]\nBonjour,\nUn grand merci pour votre avis positif concernant votre commande !\nToute l'équipe vous remercie de votre confiance et reste à votre entière disposition.\nExcellente journée,\nLe Service Client"

            st.text_area("✉️ Brouillon de Réponse Automatique généré :", value=rep_auto, height=150)

        # --- TAB 4 : REÇU & EXPORT ---
        with tab4:
            st.markdown("### 📄 Rapport Exécutif & Reçu d'Horodatage")
            
            recu_html = f"""
            <div class="receipt-box">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <h4>📋 REÇU D'ANALYSE NLP</h4>
                    <span style="font-weight: bold; background-color: #E2E8F0; padding: 4px 8px; border-radius: 4px;">ID Ticket : {ticket_id}</span>
                </div>
                <hr>
                <p><b>📅 Date/Heure Réception (Client) :</b> {dt_client.strftime('%d/%m/%Y à %H:%M:%S')}</p>
                <p><b>⚡ Date/Heure Traitement (Système) :</b> {dt_traitement.strftime('%d/%m/%Y à %H:%M:%S')}</p>
                <hr>
                <p><b>Texte Brut :</b> <i>"{avis}"</i></p>
                <p><b>Texte Nettoyé :</b> <code>{texte_clean}</code></p>
                <hr>
                <ul>
                    <li><b>Score IA Brut (Unigramme) :</b> {proba_pos_brut:.2%} de Positivité</li>
                    <li><b>Détection de Négation :</b> {neg_status}</li>
                    <li><b>Ajustement Hybride :</b> {adj_status}</li>
                    <li><b>Verdict Final :</b> <b>{verdict}</b></li>
                </ul>
                <hr>
                <p><b>📌 Action Recommandée :</b> {action_recommandee}</p>
            </div>
            """
            st.markdown(recu_html, unsafe_allow_html=True)
            st.divider()

            txt_recu = f"""===================================
RECU D'ANALYSE D'AVIS CLIENT ({ticket_id})
===================================
Date de depot client : {dt_client.strftime('%d/%m/%Y a %H:%M:%S')}
Date de traitement   : {dt_traitement.strftime('%d/%m/%Y a %H:%M:%S')}

Avis Brut : {avis}
Avis Nettoye : {texte_clean}

Score IA Brut : {proba_pos_brut:.2%}
Negations : {neg_status}
Verdict Final : {verdict_txt}
Action Recommandee : {action_recommandee}
==================================="""

            st.download_button(
                label="📥 Télécharger le Reçu (.txt)",
                data=txt_recu,
                file_name=f"recu_analyse_{ticket_id}.txt",
                mime="text/plain; charset=utf-8"
            )

# ==========================================
# 5. TRAITEMENT BATCH CSV
# ==========================================
else:
    st.title("📁 Analyse par Lot (Batch CSV)")
    uploaded_file = st.file_uploader("Choisissez un fichier CSV contenant une colonne 'texte' :", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if 'texte' in df.columns:
            df['texte_clean'] = df['texte'].apply(nettoyer_texte_fr)
            X_batch = vectorizer.transform(df['texte_clean'])
            df['score_positivité'] = modele.predict_proba(X_batch)[:, 1]
            df['verdict'] = df['score_positivité'].apply(lambda s: 'POSITIF' if s >= 0.45 else 'NÉGATIF')
            
            st.write("### Résultats de l'analyse :", df)
            
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Télécharger les résultats en CSV", csv, "resultats_analyse.csv", "text/csv")
        else:
            st.error("Le fichier CSV doit contenir une colonne nommée 'texte'.")