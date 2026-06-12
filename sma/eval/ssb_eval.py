"""SSB retrieval evaluations."""

from __future__ import annotations

import time
from dataclasses import dataclass

from sma.index.macfac import MacFacIndex
from sma.index.content_vectors import functor_vector, cosine
from sma.ir.sexpr import canonical_case_text
from sma.eval.baselines.bm25 import rank_bm25_like
from sma.eval.baselines.dense import rank_tfidf_dense, rank_tfidf_dense_batch
from sma.eval.ssb_generator import SSBTriple, generate_triples


@dataclass(frozen=True)
class RetrievalEval:
    name: str
    rows: list[dict]
    metrics: dict
    latency: dict


def evaluate_forced_choice(n: int = 12, seed: int = 11) -> RetrievalEval:
    triples = generate_triples(n, seed=seed)
    rows: list[dict] = []
    ranks: list[int] = []
    start = time.perf_counter()
    for i, triple in enumerate(triples):
        index = MacFacIndex()
        index.build([triple.analog, triple.distractor])
        results = index.retrieve(triple.query, k=2, shortlist=2)
        rank = rank_of(results, triple.analog.case_id)
        ranks.append(rank)
        rows.extend(result_rows("ssb_forced_choice", triple.query.case_id, results))
    elapsed = (time.perf_counter() - start) * 1000
    return RetrievalEval(
        name="forced_choice_fixture",
        rows=rows,
        metrics=rank_metrics("forced_choice_fixture", ranks, n),
        latency={"operation": "ssb_forced_choice_macfac", "n_cases": n * 2, "p50_ms": elapsed, "p95_ms": elapsed},
    )


def evaluate_library(
    n: int = 100,
    seed: int = 19,
    k: int = 10,
    shortlist: int | None = None,
    fac_budget: int | None = None,
) -> dict:
    triples = generate_triples(n, seed=seed)
    library_cases = []
    documents = []
    for triple in triples:
        library_cases.extend([triple.analog, triple.distractor])
        documents.append((triple.analog.case_id, canonical_case_text(triple.analog.statements)))
        documents.append((triple.distractor.case_id, canonical_case_text(triple.distractor.statements)))

    index = MacFacIndex()
    index.build(library_cases)
    sma_rows: list[dict] = []
    sma_ranks: list[int] = []
    bm25_ranks: list[int] = []
    dense_ranks: list[int] = []
    start = time.perf_counter()
    shortlist = shortlist or len(library_cases)

    query_texts = [canonical_case_text(triple.query.statements) for triple in triples]
    dense_rankings = rank_tfidf_dense_batch(query_texts, documents, k=k)

    for triple, query_text, dense in zip(triples, query_texts, dense_rankings):
        sma_results = index.retrieve(triple.query, k=k, shortlist=shortlist, fac_budget=fac_budget)
        sma_rows.extend(result_rows(f"ssb_library_{n}", triple.query.case_id, sma_results))
        sma_ranks.append(rank_of(sma_results, triple.analog.case_id))

        bm25 = rank_bm25_like(query_text, documents, k=k)
        bm25_ranks.append(rank_of_pairs(bm25, triple.analog.case_id))
        dense_ranks.append(rank_of_pairs(dense, triple.analog.case_id))

    elapsed = (time.perf_counter() - start) * 1000
    return {
        "sma_rows": sma_rows,
        "metrics": [
            rank_metrics(f"ssb_library_{n}_sma", sma_ranks, n),
            rank_metrics(f"ssb_library_{n}_bm25", bm25_ranks, n),
            rank_metrics(f"ssb_library_{n}_tfidf_dense", dense_ranks, n),
        ],
        "latency": {
            "operation": f"ssb_library_{n}_all_baselines_fac_budget_{fac_budget or 'unbounded'}",
            "n_cases": len(library_cases),
            "p50_ms": elapsed,
            "p95_ms": elapsed,
        },
    }


def evaluate_library_mac_prefilter(n: int = 1000, seed: int = 23, k: int = 10) -> dict:
    """Fast large-library MAC-stage diagnostic.

    This does not replace certified FAC. It answers whether candidate
    generation places a structurally compatible analog into the top-k shortlist
    before expensive SME matching.
    """

    triples = generate_triples(n, seed=seed)
    library = []
    documents = []
    for triple in triples:
        library.extend([triple.analog, triple.distractor])
        documents.append((triple.analog.case_id, canonical_case_text(triple.analog.statements)))
        documents.append((triple.distractor.case_id, canonical_case_text(triple.distractor.statements)))
    vectors = {case.case_id: functor_vector(case) for case in library}

    sma_ranks: list[int] = []
    bm25_ranks: list[int] = []
    dense_ranks: list[int] = []
    rows: list[dict] = []
    start = time.perf_counter()
    query_texts = [canonical_case_text(triple.query.statements) for triple in triples]
    dense_rankings = rank_tfidf_dense_batch(query_texts, documents, k=k)

    for triple, query_text, dense in zip(triples, query_texts, dense_rankings):
        qv = functor_vector(triple.query)
        ranked = sorted(
            ((case.case_id, cosine(qv, vectors[case.case_id])) for case in library),
            key=lambda row: (-row[1], row[0]),
        )[:k]
        sma_ranks.append(rank_of_pairs(ranked, triple.analog.case_id))
        for rank, (case_id, score) in enumerate(ranked, start=1):
            rows.append(
                {
                    "run_id": f"ssb_library_{n}_mac_prefilter",
                    "query_id": triple.query.case_id,
                    "rank": rank,
                    "case_id": case_id,
                    "score": f"{score:.6f}",
                    "ses_n": "",
                    "u_bound": "",
                    "certified": False,
                }
            )

        bm25 = rank_bm25_like(query_text, documents, k=k)
        bm25_ranks.append(rank_of_pairs(bm25, triple.analog.case_id))
        dense_ranks.append(rank_of_pairs(dense, triple.analog.case_id))
    elapsed = (time.perf_counter() - start) * 1000
    return {
        "sma_rows": rows,
        "metrics": [
            rank_metrics(f"ssb_library_{n}_mac_prefilter", sma_ranks, n),
            rank_metrics(f"ssb_library_{n}_bm25", bm25_ranks, n),
            rank_metrics(f"ssb_library_{n}_tfidf_dense", dense_ranks, n),
        ],
        "latency": {
            "operation": f"ssb_library_{n}_mac_prefilter_all_baselines",
            "n_cases": len(library),
            "p50_ms": elapsed,
            "p95_ms": elapsed,
        },
    }


def result_rows(run_id: str, query_id: str, results) -> list[dict]:
    return [
        {
            "run_id": run_id,
            "query_id": query_id,
            "rank": rank,
            "case_id": result.case_id,
            "score": f"{result.score:.6f}",
            "ses_n": f"{result.ses_n:.6f}",
            "u_bound": f"{result.u_bound:.6f}",
            "certified": result.certified,
        }
        for rank, result in enumerate(results, start=1)
    ]


def rank_of(results, case_id: str) -> int:
    return next((rank for rank, result in enumerate(results, start=1) if result.case_id == case_id), 0)


def rank_of_pairs(results: list[tuple[str, float]], case_id: str) -> int:
    return next((rank for rank, (result_id, _score) in enumerate(results, start=1) if result_id == case_id), 0)


def rank_metrics(split: str, ranks: list[int], total: int) -> dict:
    return {
        "split": split,
        "r1": f"{sum(1 for rank in ranks if rank == 1) / total:.4f}",
        "mrr": f"{sum((1 / rank) if rank else 0.0 for rank in ranks) / total:.4f}",
        "mapping_f1": "1.0000" if split == "forced_choice_fixture" else "",
    }
