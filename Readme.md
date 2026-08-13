# Analyse de Sentiment et Routage Automatisé du Support Client (NLP)

Ce projet implémente un pipeline complet de Traitement Automatique du Langage Naturel (NLP) pour la classification des avis clients en français (Positif / Négatif) et le routage automatisé des tickets de support selon le niveau de confiance du modèle.

---

## Fonctionnalités Principales

* **Prétraitement Linguistique :** Nettoyage textuel (regex, suppression de la ponctuation et caractères spéciaux).
* **Vectorisation TF-IDF :** Conversion du texte brut en représentations matricielles pondérées.
* **Inférence et Probabilités :** Classification par Régression Logistique avec calcul des scores de confiance.
* **Routage Métier Automatisé :** 
  * Avis Négatif (Confiance ≥ 75%) ➔ **Priorité Haute** (Transfert support immédiat).
  * Avis Négatif (Confiance < 75%) ➔ **Priorité Moyenne** (E-mail d'excuses automatique).
  * Avis Positif ➔ **Priorité Basse** (Message de remerciement).
* **Interface Web Streamlit :** Application interactive et intuitive pour le test d'inférence en temps réel.

---

## Technologies Utilisées

* **Langage :** Python 3.10+
* **NLP & Machine Learning :** `scikit-learn`, `joblib`, `numpy`, `pandas`
* **Interface Utilisateur :** `streamlit`

---

## Installation et Exécution en Local

### 1. Cloner le dépôt et accéder au dossier
```bash
git clone [https://github.com/votre-utilisateur/NLP-Analyse-Sentiments-Avis-Clients-FR.git](https://github.com/votre-utilisateur/NLP-Analyse-Sentiments-Avis-Clients-FR.git)
cd NLP-Analyse-Sentiments-Avis-Clients-FR
├── notebooks/
│   └── nlp_analyse_sentiments.ipynb  # Notebook complet (EDA, Entraînement, Métriques)
├── app.py                             # Application Web Streamlit
├── requirements.txt                   # Liste des dépendances Python
├── sentiment_model.pkl               # Modèle de Régression Logistique entraîné
├── tfidf_vectorizer.pkl               # Vectoriseur TF-IDF ajusté
└── README.md                          # Documentation du projet