# 📚 Explication du Dataset TREC-COVID et de la Tâche

## 🎯 La Tâche : Recherche d'Information (Information Retrieval)

### Le Problème à Résoudre

Imaginez qu'un médecin, un chercheur ou un patient cherche des informations sur le COVID-19. Il tape dans un moteur de recherche : **"perte de goût"**.

Le problème, c'est que les articles scientifiques utilisent des termes techniques comme **"ageusie"** ou **"dysfonctionnement olfactif"**. Un moteur de recherche classique qui cherche juste les mots exacts ne trouvera rien, car il ne comprend pas que "perte de goût" et "ageusie" signifient la même chose !

C'est ce qu'on appelle le **"Vocabulary Mismatch"** (décalage de vocabulaire) :
- Les patients utilisent des mots simples du quotidien
- Les scientifiques utilisent un jargon technique spécialisé
- Il faut comprendre le **sens** et pas juste chercher les mots exacts

### Notre Mission

Nous devons créer un système qui :
1. **Reçoit une requête** en langage naturel (ex: "perte de goût")
2. **Cherche dans une base de documents** scientifiques sur le COVID-19
3. **Retourne les documents les plus pertinents**, même s'ils n'utilisent pas exactement les mêmes mots

---

## 📦 Le Dataset TREC-COVID

### Qu'est-ce que c'est ?

TREC-COVID est un **benchmark standardisé** créé pour évaluer les systèmes de recherche d'information dans le domaine médical. Il a été créé par des chercheurs pour tester si les modèles peuvent trouver les bons articles scientifiques.

### Structure du Dataset

Le dataset contient **3 composants principaux** :

#### 1. 📄 Le Corpus (171 000 documents)

**C'est quoi ?** Une collection de 171 000 articles scientifiques sur le COVID-19.

**Format :** Chaque document contient :
- Un **identifiant unique** (_id)
- Un **titre** de l'article
- Un **texte** (résumé ou contenu de l'article)

**Exemple :**
```
Document ID: doc_12345
Titre: "Ageusie et anosmie dans les cas de COVID-19"
Texte: "Cette étude examine les cas de perte de goût et d'odorat..."
```

**Pourquoi c'est important ?** C'est notre "bibliothèque" dans laquelle nous allons chercher. Notre modèle doit parcourir ces 171 000 documents pour trouver ceux qui répondent à une requête.

---

#### 2. ❓ Les Requêtes (50 questions)

**C'est quoi ?** 50 questions en langage naturel que des utilisateurs pourraient poser.

**Format :** Chaque requête contient :
- Un **identifiant unique** (_id)
- Le **texte de la question** en langage naturel

**Exemples :**
- Requête 1: "What is the origin of COVID-19?"
- Requête 2: "What are the symptoms of COVID-19?"
- Requête 3: "How does COVID-19 spread?"

**Pourquoi c'est important ?** Ce sont les questions que notre système doit pouvoir répondre. Nous allons tester notre modèle sur ces 50 requêtes.

---

#### 3. ✅ Les Qrels (Jugements de Pertinence)

**C'est quoi ?** La "Vérité Terrain" (Ground Truth) : des experts ont regardé chaque requête et ont dit quels documents sont pertinents ou non.

**Format :** Un fichier TSV avec 3 colonnes :
- **query_id** : L'identifiant de la requête
- **doc_id** : L'identifiant du document
- **relevance** : Un score de -1 à 2 (échelle simplifiée utilisée par TREC-COVID)
  - -1 = Document non évalué ou retiré (rare)
  - 0 = Pas pertinent du tout
  - 1 = Peu pertinent
  - 2 = Pertinent

**⚠️ Note importante :** Contrairement à d'autres datasets TREC qui utilisent une échelle 0-4, TREC-COVID utilise cette échelle simplifiée (0-2) pour faciliter l'évaluation rapide pendant la pandémie. C'est normal et attendu pour ce dataset !

**Exemple :**
```
query_1  doc_12345  2  (ce document est pertinent pour la requête 1)
query_1  doc_67890  1  (ce document est peu pertinent)
query_1  doc_11111  0  (ce document n'est pas pertinent)
```

**Pourquoi c'est important ?** C'est notre **référence pour évaluer** notre modèle ! 
- Si notre modèle retourne un document avec un score de pertinence élevé (score 2), c'est bien ✅
- Si notre modèle retourne un document avec un score faible (0 ou 1), c'est moins bien ❌

---

## 🔬 Comment Évaluer Notre Modèle ?

### La Métrique : NDCG@10

**NDCG@10** signifie "Normalized Discounted Cumulative Gain at 10".

**En simple :**
- Notre modèle doit retourner les **10 meilleurs documents** pour chaque requête
- Nous comparons ces 10 documents avec les scores de pertinence des Qrels
- Plus les documents pertinents (score élevé) sont en haut de la liste, meilleur est notre score NDCG@10

**Exemple concret :**
Si pour la requête "perte de goût", notre modèle retourne :
1. Document A (score Qrel = 2) ✅ Excellent ! (pertinent)
2. Document B (score Qrel = 2) ✅ Très bien ! (pertinent)
3. Document C (score Qrel = 1) ⚠️ Moins bien... (peu pertinent)
4. Document D (score Qrel = 0) ❌ Pas bon (pas pertinent)

Nous aurons un bon score NDCG@10 car les documents pertinents sont bien classés en haut.

### Comment Évaluer sur Tout le Dataset ?

**Question importante :** On comprend comment évaluer sur une requête, mais comment faire pour **toutes les 50 requêtes** ?

**Réponse :** On calcule la métrique pour **chaque requête individuellement**, puis on fait la **moyenne** !

**Processus étape par étape :**

1. **Pour chaque requête** (il y en a 50) :
   - Notre modèle retourne les 10 meilleurs documents
   - On calcule le NDCG@10 pour cette requête spécifique
   - On obtient un score entre 0 et 1 (1 = parfait)

2. **On agrège les résultats** :
   - On additionne tous les scores NDCG@10 de chaque requête
   - On divise par le nombre de requêtes (50)
   - On obtient le **NDCG@10 moyen** sur tout le dataset

**Exemple concret :**

Supposons que nous avons testé notre modèle sur les 50 requêtes :

```
Requête 1 : NDCG@10 = 0.85
Requête 2 : NDCG@10 = 0.72
Requête 3 : NDCG@10 = 0.91
...
Requête 50 : NDCG@10 = 0.68
```

**Score final du modèle :**
```
NDCG@10 moyen = (0.85 + 0.72 + 0.91 + ... + 0.68) / 50
              = 0.78 (par exemple)
```

**Interprétation :**
- Un score de **0.78** signifie que notre modèle fonctionne bien en moyenne sur toutes les requêtes
- Plus le score est proche de **1.0**, meilleur est notre modèle
- Si le score est proche de **0**, notre modèle ne fonctionne pas bien

**Pourquoi cette approche ?**
- Certaines requêtes sont plus faciles que d'autres
- Certaines requêtes ont plus de documents pertinents que d'autres
- En faisant la moyenne, on obtient une **mesure globale** de la performance de notre modèle
- C'est la méthode standard utilisée dans la recherche d'information !

**⚠️ Note sur les scores :** Dans TREC-COVID, les scores de pertinence vont de 0 à 2 (avec parfois -1 pour les documents non évalués). Un document avec un score de 2 est considéré comme pertinent, ce qui correspondrait à un score élevé dans d'autres datasets.

**En résumé :**
1. Calculer NDCG@10 pour chaque requête → 50 scores individuels
2. Faire la moyenne de ces 50 scores → 1 score final
3. Comparer ce score final entre nos différents modèles (TF-IDF vs Word2Vec)

---

## 🎯 Notre Approche dans le Projet

### Baseline : TF-IDF

**C'est quoi ?** Un modèle classique qui compte les mots. Il cherche les documents qui contiennent le plus souvent les mots de la requête.

**Limite :** Il ne comprend pas le sens. Si nous cherchons "perte de goût" et qu'un document dit "ageusie", TF-IDF ne le trouvera pas.

### Modèle Avancé : Word2Vec (Embeddings)

**C'est quoi ?** Un modèle qui apprend la **sémantique** des mots. Il transforme les mots en vecteurs numériques qui capturent leur sens.

**Avantage :** Les mots avec un sens similaire ont des vecteurs proches. Donc "perte de goût" et "ageusie" auront des vecteurs similaires !

**Notre contribution originale :**
- Comparer un Word2Vec **pré-entraîné** (entraîné sur Google News, vocabulaire général)
- Avec un Word2Vec **entraîné spécifiquement** sur notre corpus médical COVID-19

**Hypothèse :** Le modèle entraîné sur le corpus médical devrait mieux comprendre le jargon médical et donc mieux classer les documents !

---

## 📊 Résumé en Une Phrase

**Nous devons créer un système qui trouve les bons articles scientifiques sur le COVID-19, même quand les mots utilisés dans la requête sont différents de ceux dans les articles, en utilisant des embeddings sémantiques plutôt que de simples recherches par mots-clés.**

---

## 🔄 Le Workflow Complet

1. **Charger les données** : Corpus, Requêtes, Qrels
2. **Préparer les modèles** :
   - TF-IDF (baseline)
   - Word2Vec pré-entraîné
   - Word2Vec entraîné sur le corpus
3. **Pour chaque requête** :
   - Chercher les documents les plus pertinents avec chaque modèle
   - Classer les résultats par pertinence
4. **Évaluer** :
   - Comparer les résultats avec les Qrels
   - Calculer le NDCG@10 pour chaque modèle
5. **Comparer** : Quel modèle fonctionne le mieux ?

---

## 💡 Points Clés à Retenir

✅ **Le problème** : Vocabulary Mismatch (mots différents, même sens)

✅ **La solution** : Utiliser des embeddings sémantiques (Word2Vec)

✅ **L'évaluation** : Comparer avec les Qrels (vérité terrain d'experts)

✅ **La métrique** : NDCG@10 (qualité du classement des 10 meilleurs résultats)

✅ **Notre originalité** : Comparer Word2Vec généraliste vs spécialisé médical

