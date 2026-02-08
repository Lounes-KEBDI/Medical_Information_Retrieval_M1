# 🔍 Projet de Recherche d'Information Médicale - TREC-COVID

Ce projet implémente et compare différentes approches de recherche d'information sur le dataset TREC-COVID, un corpus biomédical spécialisé sur le COVID-19.

## 📋 Structure du Projet

### 1. `download.py`
**Fonction** : Script Python pour télécharger automatiquement le dataset TREC-COVID depuis BEIR (Benchmarking IR).

- Utilise la bibliothèque `beir` pour télécharger et extraire le dataset
- Sauvegarde les données dans le dossier `data/trec-covid/trec-covid`
- Vérifie l'intégrité des données après téléchargement
- Affiche un résumé du corpus (nombre de documents, requêtes, jugements de pertinence)

**Utilisation** :
```bash
python download.py
```

---

### 2. `explore_data.ipynb`
**Fonction** : Notebook d'exploration et d'analyse du dataset TREC-COVID.

- Charge et explore la structure du corpus de documents
- Analyse les requêtes de test (50 requêtes)
- Examine les jugements de pertinence (Qrels)
- Fournit des statistiques descriptives sur les données
- Visualise la distribution des longueurs de documents, fréquences de mots, etc.

**Objectif** : Comprendre la structure et les caractéristiques du dataset avant de développer les modèles.

---

### 3. `explictation_dataset.md`
**Fonction** : Documentation détaillée sur le dataset TREC-COVID.

- Description du corpus (171,332 documents scientifiques sur le COVID-19)
- Explication des requêtes (50 requêtes de recherche)
- Description des Qrels (jugements de pertinence par des experts)
- Format des données et structure des fichiers
- Métriques d'évaluation utilisées (NDCG@10)

**Objectif** : Fournir une référence complète sur le dataset pour comprendre le contexte du projet.

---

### 4. `functions.py`
**Fonction** : Module Python contenant les fonctions généralistes réutilisables pour la recherche et l'évaluation.

**Fonctions principales** :
- `search()` : Recherche de documents pertinents pour une requête (modèle-agnostique)
- `top_k_ideal()` : Obtient le top-k idéal basé sur les Qrels (ground truth)
- `evaluate_single_query()` : Évalue une requête unique avec NDCG@k
- `evaluate_model()` : Évalue un modèle sur toutes les requêtes et retourne les statistiques

**Avantages** :
- Code réutilisable pour tous les modèles (TF-IDF, Word2Vec, BioWord2Vec, BERT, etc.)
- Évaluation cohérente et standardisée
- Facilite la comparaison entre différents modèles

---

### 5. `tf-idf.ipynb` - **BASELINE**
**Fonction** : Implémentation du modèle baseline TF-IDF (Term Frequency-Inverse Document Frequency).

**Fonctionnement** :
- Représente chaque document et requête comme un vecteur TF-IDF
- Calcule la similarité cosinus entre vecteurs de requête et documents
- Retourne les top-k documents les plus similaires

**Pourquoi baseline** :
- Méthode classique et simple de recherche d'information
- Pas d'apprentissage, uniquement basé sur les fréquences de mots
- Performance de référence pour comparer les autres modèles

**Résultats** :
- **Mean NDCG@10 : 0.5868**
- **Statut** : Baseline de référence

---

### 6. `word2vec.ipynb`
**Fonction** : Implémentation de Word2Vec générique entraîné sur le corpus TREC-COVID.

**Fonctionnement** :
- Entraîne un modèle Word2Vec sur les documents TREC-COVID
- Crée des embeddings de mots (vecteurs de 200 dimensions)
- Représente les documents par la moyenne des vecteurs de mots
- Utilise la similarité cosinus pour la recherche

**Pourquoi on pensait que ça allait fonctionner** :
- Word2Vec capture les relations sémantiques entre mots
- Les embeddings de mots permettent de capturer la similarité sémantique au-delà de la correspondance exacte
- Meilleur que TF-IDF pour comprendre le sens des mots

**Résultats** :
- **Mean NDCG@10 : 0.6099**
- **Amélioration vs TF-IDF** : +0.0231 (+3.9%)
- **Statut** : ✅ Légèrement meilleur que la baseline

**Raison de la performance modeste** :
- Word2Vec générique n'est pas spécialisé dans le domaine médical
- Vocabulaire médical spécifique mal capturé par un modèle générique
- Entraînement uniquement sur TREC-COVID (corpus limité)

---

### 7. `bioword2vec.ipynb`
**Fonction** : Implémentation de BioWord2Vec pré-entraîné sur des textes biomédicaux (PubMed, MIMIC-III).

**Fonctionnement** :
- Charge un modèle Word2Vec pré-entraîné sur 1.5M de mots biomédicaux
- Utilise les embeddings pré-entraînés pour représenter les documents
- Combine titre + texte des documents
- Représente les documents par la moyenne des vecteurs de mots BioWord2Vec

**Pourquoi on pensait que ça allait fonctionner** :
- Modèle spécialisé dans le domaine médical (vocabulaire biomédical capturé)
- Pré-entraînement sur un large corpus médical (meilleure compréhension sémantique)
- Adapté au domaine spécifique de TREC-COVID (textes médicaux)

**Résultats** :
- **Mean NDCG@10 : 0.7366**
- **Amélioration vs TF-IDF** : +0.1498 (+25.5%)
- **Amélioration vs Word2Vec** : +0.1267 (+20.8%)
- **Statut** : ✅ Meilleur modèle jusqu'à présent (avant Sentence-BERT)

**Pourquoi ça fonctionne bien** :
- Spécialisation médicale du modèle pré-entraîné
- Vocabulaire médical riche et adapté au domaine
- Bon compromis entre simplicité et performance

---

### 8. `bioword2vec_finetune.ipynb`
**Fonction** : Fine-tuning de BioWord2Vec sur le corpus TREC-COVID (approche hybride).

**Fonctionnement** :
- Charge BioWord2Vec pré-entraîné
- Crée un nouveau modèle Word2Vec avec les mêmes paramètres
- Initialise les mots communs avec les poids pré-entraînés
- Entraîne le modèle sur TREC-COVID pendant 5 époques
- Les nouveaux mots spécifiques à TREC-COVID sont appris pendant l'entraînement

**Pourquoi on pensait que ça allait fonctionner** :
- Adaptation du modèle pré-entraîné aux spécificités de TREC-COVID
- Apprentissage de nouveaux termes médicaux spécifiques au COVID-19
- Combinaison des connaissances générales (pré-entraînement) et spécifiques (fine-tuning)

**Résultats** :
- **Mean NDCG@10 : 0.6259**
- **Amélioration vs TF-IDF** : +0.0391 (+6.7%)
- **Dégradation vs BioWord2Vec** : -0.1107 (-15.0%)
- **Statut** : ❌ Moins bon que BioWord2Vec pré-entraîné

**Raison potentielle de l'échec** :
- **Overfitting** : Le modèle s'est trop spécialisé sur TREC-COVID et a perdu la généralité du pré-entraînement
- **Trop d'époques** : 5 époques peuvent être excessives pour un fine-tuning, causant une dérive des embeddings
- **Données limitées** : TREC-COVID seul n'est peut-être pas suffisant pour améliorer le modèle pré-entraîné
- **Modèle déjà optimal** : BioWord2Vec pré-entraîné était déjà bien adapté au domaine médical, le fine-tuning n'apporte pas de valeur ajoutée

---

### 9. `DAN.ipynb`
**Fonction** : Implémentation d'un Deep Averaging Network (DAN) avec architecture Bi-Encoder.

**Fonctionnement** :
- Initialise les embeddings avec BioWord2Vec pré-entraîné
- Architecture DAN : moyenne des embeddings de mots → couches cachées → embedding de document
- Architecture Bi-Encoder (siamoise) : encodeurs séparés pour requêtes et documents
- Entraînement avec TripletMarginLoss sur triplets (requête, document positif, document négatif)
- Optimise la similarité cosinus entre requêtes et documents pertinents

**Pourquoi on pensait que ça allait fonctionner** :
- Architecture neuronale profonde pour capturer des relations complexes
- Apprentissage supervisé avec les Qrels (meilleure adaptation aux données)
- Fine-tuning des embeddings pour la tâche de recherche spécifique
- Modèles de deep learning souvent meilleurs que les méthodes classiques

**Résultats** :
- **Mean NDCG@10 : 0.6630**
- **Amélioration vs TF-IDF** : +0.0762 (+13.0%)
- **Dégradation vs BioWord2Vec** : -0.0736 (-10.0%)
- **Statut** : ❌ Moins bon que BioWord2Vec pré-entraîné

**Raison potentielle de l'échec** :
- **Données d'entraînement limitées** : Seulement 50 requêtes avec Qrels, pas assez pour entraîner un modèle profond efficacement
- **Overfitting** : Le modèle s'est probablement sur-adapté aux données d'entraînement limitées
- **Architecture trop complexe** : Pour ce volume de données, un modèle simple (BioWord2Vec) peut être plus efficace
- **Hyperparamètres non optimisés** : Les paramètres (learning rate, batch size, nombre d'époques) n'ont peut-être pas été optimisés
- **Triplets mal sélectionnés** : La stratégie de sélection des triplets négatifs peut influencer significativement les performances

---

### 10. `BioBERT.ipynb`
**Fonction** : Implémentation de BioBERT (BERT spécialisé médical) pour la recherche d'information.

**Fonctionnement** :
- Charge le modèle BioBERT pré-entraîné (dmis-lab/biobert-base-cased-v1.1)
- Utilise le token [CLS] pour représenter chaque document et requête
- Crée des embeddings contextuels (768 dimensions)
- Traitement par batch optimisé pour GPU

**Pourquoi on pensait que ça allait fonctionner** :
- BERT capture le contexte bidirectionnel (meilleure compréhension sémantique)
- BioBERT spécialisé médical (vocabulaire biomédical)
- Embeddings contextuels adaptés à chaque occurrence de mot
- Modèles Transformer souvent meilleurs que Word2Vec

**Résultats** :
- **Mean NDCG@10 : 0.2886**
- **Dégradation vs TF-IDF** : -0.2982 (-50.8%)
- **Dégradation vs BioWord2Vec** : -0.4480 (-60.8%)
- **Statut** : ❌ Très mauvais résultat

**Raison potentielle de l'échec** :
- **Token [CLS] non optimisé** : Le token [CLS] de BERT n'est pas optimisé pour la recherche d'information, il est conçu pour la classification
- **Pas de fine-tuning** : BioBERT n'a pas été fine-tuné pour la tâche de recherche sémantique
- **Embeddings non normalisés** : Les embeddings bruts de BERT ne sont pas optimisés pour la similarité cosinus
- **Manque d'optimisation** : Contrairement à Sentence-BERT, BioBERT standard n'est pas entraîné sur des paires de textes similaires/dissimilaires
- **Problème d'alignement** : Les embeddings de requêtes et documents ne sont pas alignés dans l'espace sémantique pour la recherche

---

### 11. `SentenceBERT.ipynb`
**Fonction** : Implémentation de Sentence-BERT médical optimisé pour la recherche sémantique.

**Fonctionnement** :
- Charge Sentence-BERT médical pré-entraîné (pritamdeka/S-PubMedBert-MS-MARCO)
- Modèle optimisé spécifiquement pour la similarité sémantique entre textes
- Crée des embeddings de documents et requêtes en une seule ligne (`model.encode()`)
- Embeddings déjà normalisés et optimisés pour la similarité cosinus
- Pas besoin de gérer tokens, [CLS], normalisation manuelle

**Pourquoi on pensait que ça allait fonctionner** :
- Sentence-BERT entraîné spécifiquement pour la recherche sémantique (pas juste pour comprendre)
- Optimisé pour maximiser la similarité cosinus entre textes similaires
- Modèle médical spécialisé (PubMed + MS-MARCO)
- Plus simple et plus rapide que BERT standard
- Architecture optimisée pour la production

**Résultats** :
- **Mean NDCG@10 : 0.8171**
- **Amélioration vs TF-IDF** : +0.2303 (+39.2%)
- **Amélioration vs BioWord2Vec** : +0.0805 (+10.9%)
- **Statut** : ✅ **MEILLEUR MODÈLE** actuel

**Pourquoi ça fonctionne si bien** :
- **Optimisation spécifique** : Entraîné sur des paires de textes similaires/dissimilaires pour maximiser la similarité
- **Spécialisation médicale** : Modèle adapté au domaine biomédical
- **Simplicité** : Pas de fine-tuning nécessaire, modèle prêt à l'emploi
- **Embeddings normalisés** : Optimisés directement pour la similarité cosinus
- **Meilleure compréhension contextuelle** : BERT capture mieux le contexte que Word2Vec

---

## 📊 Résumé des Performances

| Modèle | Mean NDCG@10 | Amélioration vs Baseline | Statut |
|--------|--------------|--------------------------|--------|
| **TF-IDF** (baseline) | 0.5868 | - | Baseline |
| **Word2Vec** | 0.6099 | +3.9% | ✅ Légèrement meilleur |
| **BioWord2Vec** | 0.7366 | +25.5% | ✅ Très bon |
| **BioWord2Vec Fine-tuned** | 0.6259 | +6.7% | ❌ Moins bon |
| **DAN** | 0.6630 | +13.0% | ❌ Moins bon |
| **BioBERT** | 0.2886 | -50.8% | ❌ Très mauvais |
| **Sentence-BERT** | **0.8171** | **+39.2%** | ✅ **MEILLEUR** |

## 🎯 Conclusions

1. **Sentence-BERT médical** est le meilleur modèle avec un NDCG@10 de **0.8171**
2. **BioWord2Vec pré-entraîné** est un excellent compromis simplicité/performance (0.7366)
3. Le **fine-tuning** n'a pas amélioré BioWord2Vec, probablement à cause de l'overfitting
4. Les **modèles neuronaux profonds** (DAN) nécessitent plus de données pour être efficaces
5. Les **modèles spécialisés médicaux** (BioWord2Vec, Sentence-BERT) surpassent les modèles génériques
6. **BioBERT standard** obtient de très mauvais résultats (0.2886) car il n'est pas optimisé pour la recherche sémantique, contrairement à Sentence-BERT qui est spécialement entraîné pour cette tâche

## 🚀 Utilisation

1. Télécharger le dataset : `python download.py`
2. Explorer les données : Exécuter `explore_data.ipynb`
3. Exécuter les modèles : Exécuter les notebooks dans l'ordre souhaité
4. Comparer les résultats : Les résultats sont sauvegardés dans `results/`

## 📝 Notes

- Tous les modèles utilisent les fonctions généralistes de `functions.py` pour une évaluation cohérente
- Les résultats sont sauvegardés automatiquement dans `results/` au format JSON
- Le dataset TREC-COVID contient 171,332 documents et 50 requêtes de test

