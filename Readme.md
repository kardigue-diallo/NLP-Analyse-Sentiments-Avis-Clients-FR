# Analyse de Sentiment et Routage Automatisé du Support Client (NLP)

[![Python](https://img.shields.io/badge/PYTHON-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/SCIKIT--LEARN-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/OPEN_IN-STREAMLIT-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://share.streamlit.io/)
[![License](https://img.shields.io/badge/LICENSE-MIT-green?style=for-the-badge)](LICENSE)
🌐 **Démo en ligne :** [nlp-analyse-sentiments.streamlit.app](https://nlp-analyse-sentiments.streamlit.app)

Ce projet implémente un pipeline complet de Traitement Automatique du Langage Naturel (NLP) pour la classification des avis clients en français (Positif / Négatif), l'explicabilité IA des prédictions (XAI), et le routage automatisé des tickets de support selon le niveau de confiance du modèle.

---

## Fonctionnalités Principales

* **Prétraitement Linguistique :** Nettoyage textuel avancé (expressions régulières, gestion des accents français, suppression de la ponctuation et des mots vides).
* **Vectorisation TF-IDF :** Conversion du texte brut en représentations matricielles pondérées (Unigrammes & Bigrammes).
* **Inférence Statistique & Règles Métier :** Classification par Régression Logistique couplée à un moteur de surcharge pour la détection explicite des négations (*ne, pas, jamais, aucun*).
* **Explicabilité IA (XAI) :** Décomposition exacte du score pour chaque mot ($\text{Contribution} = \text{TF-IDF} \times \text{Poids}$) et visualisations interactives Plotly.
* **Routage Métier Automatisé :**
  * **Avis Négatif (Confiance $\ge 75\%$ )** $\rightarrow$ **Priorité Haute** (Transfert support immédiat & alerte).
  * **Avis Négatif (Confiance $< 75\%$ )** $\rightarrow$ **Priorité Moyenne** (E-mail d'excuses automatique).
  * **Avis Positif** $\rightarrow$ **Priorité Basse** (Message de remerciement).
* **Analyse en Lot (Batch CSV) & Export :** Traitement simultané de fichiers CSV et génération de reçus d'analyse horodatés au format `.txt`.
* **Interface Web Streamlit :** Application interactive et intuitive pour le test d'inférence en temps réel.

---

## Décomposition Mathématique (XAI)

La probabilité finale qu'un avis soit **Positif** est calculée via la fonction sigmoïde :

$$P(Y = 1 \mid X) = \frac{1}{1 + e^{-z}} \quad \text{où} \quad z = W_0 + \sum_{i=1}^{n} (\text{TF-IDF}_i \times W_i)$$

---

## Technologies Utilisées

* **Langage :** Python 3.10+
* **NLP & Machine Learning :** `scikit-learn`, `joblib`, `numpy`, `pandas`
* **Visualisation & Interface :** `streamlit`, `plotly`

---

## Installation et Exécution en Local

### 1. Cloner le dépôt et accéder au dossier
```bash
git clone https://github.com/kardigue-diallo/NLP-Analyse-Sentiments-Avis-Clients-FR.git)
cd NLP-Analyse-Sentiments-Avis-Clients-FR
```

---

### 2. Arborescence du Projet
```text
NLP-Analyse-Sentiments-Avis-Clients-FR/
│
├── notebooks/
│   └── nlp-analyse-sentiments-avis-clients-fr.ipynb   # Notebook (EDA, Entraînement, Métriques)
│
├── app.py                                             # Application Web Streamlit
├── model.pkl                                          # Modèle de Régression Logistique entraîné
├── tfidf.pkl                                          # Vectoriseur TF-IDF ajusté
├── requirements.txt                                   # Liste des dépendances Python
└── README.md                                          # Documentation du projet
```

---

### 3. Créer et activer un environnement virtuel
```bash
# Sur Windows
python -m venv venv
venv\Scripts\activate

# Sur Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 4. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 5. Lancer l'application Streamlit
```bash
python -m streamlit run app.py
```

---

## 👤 Auteur

* **Kardigue Diallo** - [GitHub Profile](https://github.com/kardigue-diallo)
