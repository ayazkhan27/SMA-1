"""Dense-RAG style baseline with local deterministic TF-IDF fallback.

This is not the final sentence-transformer baseline from the paper plan. It is
the CPU-safe baseline used in MVP reports when no embedding model is installed.
"""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def rank_tfidf_dense(query: str, documents: list[tuple[str, str]], k: int = 10) -> list[tuple[str, float]]:
    if not documents:
        return []
    ids = [doc_id for doc_id, _ in documents]
    texts = [text for _, text in documents]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
    matrix = vectorizer.fit_transform(texts + [query])
    sims = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
    rows = list(zip(ids, map(float, sims)))
    rows.sort(key=lambda row: (-row[1], row[0]))
    return rows[:k]


def rank_tfidf_dense_batch(queries: list[str], documents: list[tuple[str, str]], k: int = 10) -> list[list[tuple[str, float]]]:
    if not documents:
        return [[] for _ in queries]
    ids = [doc_id for doc_id, _ in documents]
    texts = [text for _, text in documents]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
    doc_matrix = vectorizer.fit_transform(texts)
    query_matrix = vectorizer.transform(queries)
    sims = query_matrix @ doc_matrix.T
    ranked: list[list[tuple[str, float]]] = []
    for row_idx in range(sims.shape[0]):
        row = sims.getrow(row_idx)
        scores = row.toarray().ravel()
        pairs = list(zip(ids, map(float, scores)))
        pairs.sort(key=lambda item: (-item[1], item[0]))
        ranked.append(pairs[:k])
    return ranked
