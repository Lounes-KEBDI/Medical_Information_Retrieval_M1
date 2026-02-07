"""
General-purpose functions for information retrieval evaluation.

This module provides functions to:
- Search documents using any vectorizer/model
- Get ideal top-k results from Qrels
- Evaluate model performance using scikit-learn metrics
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Callable
from sklearn.metrics import ndcg_score


def search(
    query_text: str,
    doc_vectors: Any,
    doc_ids: List[str],
    vectorizer: Any,
    top_k: int = 10,
    similarity_function: Callable = None
) -> List[Tuple[str, float]]:
    """
    Search for top-k most relevant documents for a given query.
    
    This function is model-agnostic and works with any vectorizer (TF-IDF, Word2Vec, etc.)
    
    Args:
        query_text: The query text to search for
        doc_vectors: Pre-computed document vectors (can be sparse matrix, numpy array, etc.)
        doc_ids: List of document IDs corresponding to doc_vectors (same order)
        vectorizer: Trained vectorizer/model that can transform query text to vector
        top_k: Number of top documents to return (default: 10)
        similarity_function: Optional custom similarity function. If None, uses cosine similarity.
                           Should take (query_vector, doc_vectors) and return similarities.
    
    Returns:
        List of tuples (doc_id, score) sorted by score descending, length = top_k
    """
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Transform query to vector using the provided vectorizer
    if hasattr(vectorizer, 'transform'):
        query_vector = vectorizer.transform([query_text])
    elif callable(vectorizer):
        query_vector = vectorizer([query_text])
    else:
        raise ValueError("Vectorizer must have a 'transform' method or be callable")
    
    # Calculate similarities
    if similarity_function is not None:
        similarities = similarity_function(query_vector, doc_vectors)
    else:
        # Default: cosine similarity
        similarities = cosine_similarity(query_vector, doc_vectors).flatten()
    
    # Get top-k indices
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    # Return (doc_id, score) tuples
    results = [(doc_ids[idx], float(similarities[idx])) for idx in top_indices]
    
    return results


def top_k_ideal(
    query_id: str,
    qrels: Dict[str, Dict[str, int]],
    top_k: int = 10
) -> List[Tuple[str, int]]:
    """
    Get the ideal top-k documents for a query based on Qrels (ground truth).
    
    Args:
        query_id: ID of the query
        qrels: Dictionary {query_id: {doc_id: relevance_score}}
        top_k: Number of top documents to return (default: 10)
    
    Returns:
        List of tuples (doc_id, relevance_score) sorted by relevance descending, length <= top_k
        Returns fewer than top_k if there are fewer relevant documents in Qrels
    """
    # Get all relevance judgments for this query
    query_qrels = qrels.get(query_id, {})
    
    if not query_qrels:
        return []
    
    # Sort by relevance score descending (score 2 = best, then 1, then 0)
    ideal_docs = sorted(query_qrels.items(), key=lambda x: x[1], reverse=True)
    
    # Return top-k (or fewer if not enough documents)
    return ideal_docs[:top_k]


def evaluate_single_query(
    query_id: str,
    model_results: List[Tuple[str, float]],
    qrels: Dict[str, Dict[str, int]],
    k: int = 10
) -> float:
    """
    Evaluate a single query using NDCG score from scikit-learn.
    
    Compare les résultats du modèle avec les scores réels des experts (Qrels) pour cette requête.
    
    Args:
        query_id: ID de la requête à évaluer
        model_results: Liste de tuples (doc_id, score_prédit) retournés par le modèle, triés par score décroissant
        qrels: Dictionnaire {query_id: {doc_id: score_réel}} contenant les jugements des experts
        k: Nombre de documents à considérer pour NDCG@k (default: 10)
    
    Returns:
        Score NDCG@k entre 0 et 1
    """
    # Limiter aux top-k résultats du modèle
    model_results = model_results[:k]
    
    # Récupérer les scores réels des experts pour cette requête depuis les Qrels
    query_qrels = qrels.get(query_id, {})
    
    # Construire y_score (scores prédits par le modèle) et y_true (scores réels des experts)
    # Les deux listes doivent être dans le même ordre (ordre retourné par le modèle)
    y_score = []  # Scores prédits par le modèle
    y_true = []   # Scores réels des experts pour ces mêmes documents
    
    for doc_id, predicted_score in model_results:
        # Score prédit par le modèle
        y_score.append(predicted_score)
        
        # Score réel des experts pour ce document (0 si le document n'est pas dans les Qrels)
        real_relevance_score = query_qrels.get(doc_id, 0)
        y_true.append(real_relevance_score)
    
    # Convertir en arrays numpy
    y_score = np.array(y_score)
    y_true = np.array(y_true)
    
    # Reshape pour sklearn (besoin de arrays 2D)
    y_score = y_score.reshape(1, -1)
    y_true = y_true.reshape(1, -1)
    
    # Gérer le cas où il n'y a pas de documents pertinents
    if y_true.size == 0 or y_score.size == 0 or np.sum(y_true) == 0:
        return 0.0
    
    # Calculer NDCG@k avec scikit-learn
    # Compare les scores prédits (y_score) avec les scores réels (y_true)
    ndcg = ndcg_score(y_true, y_score, k=k)
    
    return float(ndcg)


def evaluate_model(
    queries: Dict[str, str],
    all_model_results: Dict[str, List[Tuple[str, float]]],
    qrels: Dict[str, Dict[str, int]],
    k: int = 10
) -> Dict[str, Any]:
    """
    Evaluate model performance across all queries.
    
    Args:
        queries: Dictionary {query_id: query_text}
        all_model_results: Dictionary {query_id: [(doc_id, score), ...]} with results for each query
        qrels: Dictionary {query_id: {doc_id: relevance_score}}
        k: Number of top documents to consider (default: 10)
    
    Returns:
        Dictionary containing:
        - 'mean_score': Mean NDCG@k across all queries
        - 'scores': Dictionary {query_id: score} for each query
        - 'min_score': Minimum score
        - 'max_score': Maximum score
        - 'std_score': Standard deviation of scores
        - 'num_queries': Number of queries evaluated
    """
    scores = {}
    
    for query_id in queries.keys():
        # Get model results for this query
        model_results = all_model_results.get(query_id, [])
        
        # Evaluate this query
        score = evaluate_single_query(
            query_id=query_id,
            model_results=model_results,
            qrels=qrels,
            k=k
        )
        
        scores[query_id] = score
    
    # Calculate statistics
    score_values = list(scores.values())
    mean_score = np.mean(score_values)
    min_score = min(score_values) if score_values else 0.0
    max_score = max(score_values) if score_values else 0.0
    std_score = np.std(score_values) if score_values else 0.0
    
    # Display summary
    print("=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Mean NDCG@{k}: {mean_score:.4f}")
    print(f"Min NDCG@{k}: {min_score:.4f}")
    print(f"Max NDCG@{k}: {max_score:.4f}")
    print(f"Std NDCG@{k}: {std_score:.4f}")
    print(f"Number of queries: {len(queries)}")
    print("=" * 60)
    
    results = {
        'mean_score': float(mean_score),
        'scores': scores,
        'min_score': float(min_score),
        'max_score': float(max_score),
        'std_score': float(std_score),
        'num_queries': len(queries)
    }
    
    return results

