from sentence_transformers import CrossEncoder

def rerank_results(query, results, top_k=3):
    """
    Rerank search results using Cross-Encoder
    
    Args:
        query: User query string
        results: List of search results from Qdrant
        top_k: Number of top results to return
        
    Returns:
        List of top_k reranked results
    """
    model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    # Create query-document pairs
    pairs = [[query, result.payload["text"]] for result in results]
    
    # Get scores
    scores = model.predict(pairs)
    
    # Sort by score
    ranked_results = sorted(
        zip(results, scores), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    return [result for result, score in ranked_results[:top_k]]
