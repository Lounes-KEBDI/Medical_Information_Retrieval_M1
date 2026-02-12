# 🔍 Medical Information Retrieval Project - TREC-COVID

This project implements and compares different information retrieval approaches on the TREC-COVID dataset, a biomedical corpus specialized on COVID-19.

## 📋 Project Structure

### 1. `download.py`
**Function**: Python script to automatically download the TREC-COVID dataset from BEIR (Benchmarking IR).

- Uses the `beir` library to download and extract the dataset
- Saves data in the `data/trec-covid/trec-covid` folder
- Verifies data integrity after download
- Displays a summary of the corpus (number of documents, queries, relevance judgments)

**Usage**:
```bash
python download.py
```

---

### 2. `explore_data.ipynb`
**Function**: Notebook for exploring and analyzing the TREC-COVID dataset.

- Loads and explores the document corpus structure
- Analyzes test queries (50 queries)
- Examines relevance judgments (Qrels)
- Provides descriptive statistics on the data
- Visualizes document length distributions, word frequencies, etc.

**Objective**: Understand the structure and characteristics of the dataset before developing models.

---

### 3. `explictation_dataset.md`
**Function**: Detailed documentation on the TREC-COVID dataset.

- Corpus description (171,332 scientific documents on COVID-19)
- Query explanation (50 search queries)
- Qrels description (expert relevance judgments)
- Data format and file structure
- Evaluation metrics used (NDCG@10)

**Objective**: Provide a complete reference on the dataset to understand the project context.

---

### 4. `functions.py`
**Function**: Python module containing reusable general-purpose functions for search and evaluation.

**Main functions**:
- `search()`: Search for relevant documents for a query (model-agnostic)
- `top_k_ideal()`: Get the ideal top-k based on Qrels (ground truth)
- `evaluate_single_query()`: Evaluate a single query with NDCG@k
- `evaluate_model()`: Evaluate a model on all queries and return statistics

**Advantages**:
- Reusable code for all models (TF-IDF, Word2Vec, BioWord2Vec, BERT, etc.)
- Consistent and standardized evaluation
- Facilitates comparison between different models

---

### 5. `tf-idf.ipynb` - **BASELINE**
**Function**: Implementation of the TF-IDF (Term Frequency-Inverse Document Frequency) baseline model.

**How it works**:
- Represents each document and query as a TF-IDF vector
- Computes cosine similarity between query and document vectors
- Returns the top-k most similar documents

**Why baseline**:
- Classic and simple information retrieval method
- No learning, only based on word frequencies
- Reference performance for comparing other models

**Results**:
- **Mean NDCG@10: 0.5868**
- **Status**: Reference baseline

---

### 6. `word2vec.ipynb`
**Function**: Implementation of generic Word2Vec trained on the TREC-COVID corpus.

**How it works**:
- Trains a Word2Vec model on TREC-COVID documents
- Creates word embeddings (200-dimensional vectors)
- Represents documents by averaging word vectors
- Uses cosine similarity for search

**Why we thought it would work**:
- Word2Vec captures semantic relationships between words
- Word embeddings allow capturing semantic similarity beyond exact matching
- Better than TF-IDF for understanding word meaning

**Results**:
- **Mean NDCG@10: 0.6099**
- **Improvement vs TF-IDF**: +0.0231 (+3.9%)
- **Status**: ✅ Slightly better than baseline

**Reason for modest performance**:
- Generic Word2Vec is not specialized in the medical domain
- Specific medical vocabulary poorly captured by a generic model
- Training only on TREC-COVID (limited corpus)

---

### 7. `bioword2vec.ipynb`
**Function**: Implementation of pre-trained BioWord2Vec on biomedical texts (PubMed, MIMIC-III).

**How it works**:
- Loads a Word2Vec model pre-trained on 1.5M biomedical words
- Uses pre-trained embeddings to represent documents
- Combines title + text of documents
- Represents documents by averaging BioWord2Vec word vectors

**Why we thought it would work**:
- Model specialized in the medical domain (biomedical vocabulary captured)
- Pre-training on a large medical corpus (better semantic understanding)
- Adapted to the specific domain of TREC-COVID (medical texts)

**Results**:
- **Mean NDCG@10: 0.7366**
- **Improvement vs TF-IDF**: +0.1498 (+25.5%)
- **Improvement vs Word2Vec**: +0.1267 (+20.8%)
- **Status**: ✅ Best model so far (before Sentence-BERT)

**Why it works well**:
- Medical specialization of the pre-trained model
- Rich medical vocabulary adapted to the domain
- Good trade-off between simplicity and performance

---

### 8. `bioword2vec_finetune.ipynb`
**Function**: Fine-tuning of BioWord2Vec on the TREC-COVID corpus (hybrid approach).

**How it works**:
- Loads pre-trained BioWord2Vec
- Creates a new Word2Vec model with the same parameters
- Initializes common words with pre-trained weights
- Trains the model on TREC-COVID for 5 epochs
- New words specific to TREC-COVID are learned during training

**Why we thought it would work**:
- Adaptation of the pre-trained model to TREC-COVID specificities
- Learning new medical terms specific to COVID-19
- Combination of general knowledge (pre-training) and specific knowledge (fine-tuning)

**Results**:
- **Mean NDCG@10: 0.6259**
- **Improvement vs TF-IDF**: +0.0391 (+6.7%)
- **Degradation vs BioWord2Vec**: -0.1107 (-15.0%)
- **Status**: ❌ Worse than pre-trained BioWord2Vec

**Potential reason for failure**:
- **Overfitting**: The model became too specialized on TREC-COVID and lost the generality of pre-training
- **Too many epochs**: 5 epochs may be excessive for fine-tuning, causing embedding drift
- **Limited data**: TREC-COVID alone may not be sufficient to improve the pre-trained model
- **Model already optimal**: Pre-trained BioWord2Vec was already well adapted to the medical domain, fine-tuning adds no value

---

### 9. `DAN.ipynb`
**Function**: Implementation of a Deep Averaging Network (DAN) with Bi-Encoder architecture.

**How it works**:
- Initializes embeddings with pre-trained BioWord2Vec
- DAN architecture: average of word embeddings → hidden layers → document embedding
- Bi-Encoder (siamese) architecture: separate encoders for queries and documents
- Training with TripletMarginLoss on triplets (query, positive document, negative document)
- Optimizes cosine similarity between queries and relevant documents

**Why we thought it would work**:
- Deep neural architecture to capture complex relationships
- Supervised learning with Qrels (better adaptation to data)
- Fine-tuning embeddings for the specific search task
- Deep learning models often better than classical methods

**Results**:
- **Mean NDCG@10: 0.6630**
- **Improvement vs TF-IDF**: +0.0762 (+13.0%)
- **Degradation vs BioWord2Vec**: -0.0736 (-10.0%)
- **Status**: ❌ Worse than pre-trained BioWord2Vec

**Potential reason for failure**:
- **Limited training data**: Only 50 queries with Qrels, not enough to train a deep model effectively
- **Overfitting**: The model likely overfitted to the limited training data
- **Architecture too complex**: For this data volume, a simple model (BioWord2Vec) can be more effective
- **Unoptimized hyperparameters**: Parameters (learning rate, batch size, number of epochs) may not have been optimized
- **Poorly selected triplets**: The negative triplet selection strategy can significantly influence performance

---

### 10. `BioBERT.ipynb`
**Function**: Implementation of BioBERT (medical-specialized BERT) for information retrieval.

**How it works**:
- Loads pre-trained BioBERT model (dmis-lab/biobert-base-cased-v1.1)
- Uses the [CLS] token to represent each document and query
- Creates contextual embeddings (768 dimensions)
- Batch processing optimized for GPU

**Why we thought it would work**:
- BERT captures bidirectional context (better semantic understanding)
- Medical-specialized BioBERT (biomedical vocabulary)
- Contextual embeddings adapted to each word occurrence
- Transformer models often better than Word2Vec

**Results**:
- **Mean NDCG@10: 0.2886**
- **Degradation vs TF-IDF**: -0.2982 (-50.8%)
- **Degradation vs BioWord2Vec**: -0.4480 (-60.8%)
- **Status**: ❌ Very poor result

**Potential reason for failure**:
- **[CLS] token not optimized**: BERT's [CLS] token is not optimized for information retrieval, it is designed for classification
- **No fine-tuning**: BioBERT was not fine-tuned for semantic search task
- **Unnormalized embeddings**: Raw BERT embeddings are not optimized for cosine similarity
- **Lack of optimization**: Unlike Sentence-BERT, standard BioBERT is not trained on similar/dissimilar text pairs
- **Alignment problem**: Query and document embeddings are not aligned in semantic space for search

---

### 11. `SentenceBERT.ipynb`
**Function**: Implementation of medical Sentence-BERT optimized for semantic search.

**How it works**:
- Loads pre-trained medical Sentence-BERT (pritamdeka/S-PubMedBert-MS-MARCO)
- Model optimized specifically for semantic similarity between texts
- Creates document and query embeddings in one line (`model.encode()`)
- Embeddings already normalized and optimized for cosine similarity
- No need to handle tokens, [CLS], manual normalization

**Why we thought it would work**:
- Sentence-BERT trained specifically for semantic search (not just for understanding)
- Optimized to maximize cosine similarity between similar texts
- Medical-specialized model (PubMed + MS-MARCO)
- Simpler and faster than standard BERT
- Production-optimized architecture

**Results**:
- **Mean NDCG@10: 0.8171**
- **Improvement vs TF-IDF**: +0.2303 (+39.2%)
- **Improvement vs BioWord2Vec**: +0.0805 (+10.9%)
- **Status**: ✅ **BEST MODEL** currently

**Why it works so well**:
- **Specific optimization**: Trained on similar/dissimilar text pairs to maximize similarity
- **Medical specialization**: Model adapted to the biomedical domain
- **Simplicity**: No fine-tuning needed, ready-to-use model
- **Normalized embeddings**: Directly optimized for cosine similarity
- **Better contextual understanding**: BERT captures context better than Word2Vec

---

## 📊 Performance Summary

| Model | Mean NDCG@10 | Improvement vs Baseline | Status |
|-------|--------------|-------------------------|--------|
| **TF-IDF** (baseline) | 0.5868 | - | Baseline |
| **Word2Vec** | 0.6099 | +3.9% | ✅ Slightly better |
| **BioWord2Vec** | 0.7366 | +25.5% | ✅ Very good |
| **BioWord2Vec Fine-tuned** | 0.6259 | +6.7% | ❌ Worse |
| **DAN** | 0.6630 | +13.0% | ❌ Worse |
| **BioBERT** | 0.2886 | -50.8% | ❌ Very poor |
| **Sentence-BERT** | **0.8171** | **+39.2%** | ✅ **BEST** |

## 🎯 Conclusions

1. **Medical Sentence-BERT** is the best model with an NDCG@10 of **0.8171**
2. **Pre-trained BioWord2Vec** is an excellent simplicity/performance trade-off (0.7366)
3. **Fine-tuning** did not improve BioWord2Vec, likely due to overfitting
4. **Deep neural models** (DAN) require more data to be effective
5. **Medical-specialized models** (BioWord2Vec, Sentence-BERT) outperform generic models
6. **Standard BioBERT** achieves very poor results (0.2886) because it is not optimized for semantic search, unlike Sentence-BERT which is specifically trained for this task

## 🚀 Usage

1. Download the dataset: `python download.py`
2. Explore the data: Run `explore_data.ipynb`
3. Run the models: Execute the notebooks in the desired order
4. Compare results: Results are saved in `results/`

## 📝 Notes

- All models use the general-purpose functions from `functions.py` for consistent evaluation
- Results are automatically saved in `results/` in JSON format
- The TREC-COVID dataset contains 171,332 documents and 50 test queries
