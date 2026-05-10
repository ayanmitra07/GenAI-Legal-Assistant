# ============================================================
# SECTION 9 — REAL-WORLD ADVERSARIAL BENCHMARK
# Evaluates:
# 1. Hybrid Retrieval (BEFORE reranker)
# 2. Hybrid + Cross Encoder (AFTER reranker)
# Metrics:
# - Recall@K
# - Clause Recall@K
# - MRR
# ============================================================

import numpy as np

from app.reranker import rerank_results


# ============================================================
# METRICS
# ============================================================

def recall_at_k(texts, gold_article, k):
    return int(any(gold_article in t for t in texts[:k]))


def clause_recall_at_k(texts, gold_article, gold_clause, k):
    if gold_clause is None:
        return recall_at_k(texts, gold_article, k)
    return int(any((gold_article in t) and (gold_clause in t) for t in texts[:k]))


def mrr_score(texts, gold_article):
    for rank, t in enumerate(texts, start=1):
        if gold_article in t:
            return 1 / rank
    return 0


# ============================================================
# MAIN EVALUATION
# ============================================================

def run_evaluation(retrieve_fn):

    print("=" * 70)
    print("SECTION 9 — EVALUATION STARTED")
    print("=" * 70)

    evaluation_set = [
        {"category": "Layman-FR", "query": "If someone is arrested unfairly, what protection do they have?", "gold_article": "Article 22", "gold_clause": None},
        {"category": "Layman-FR", "query": "Can the government take away my right to life?", "gold_article": "Article 21", "gold_clause": None},
        {"category": "Layman-FR", "query": "Can I directly go to the Supreme Court if my rights are violated?", "gold_article": "Article 32", "gold_clause": None},
        {"category": "Layman-FR", "query": "Can High Courts also issue writs?", "gold_article": "Article 226", "gold_clause": None},

        {"category": "Confusion", "query": "Apart from fundamental rights cases, can the Supreme Court issue writs?", "gold_article": "Article 139", "gold_clause": None},
        {"category": "Confusion", "query": "Where is habeas corpus actually mentioned in the Constitution?", "gold_article": "Article 32", "gold_clause": "(2)"},

        {"category": "Structure", "query": "How does India change its Constitution?", "gold_article": "Article 368", "gold_clause": None},
        {"category": "Structure", "query": "Who runs elections in India?", "gold_article": "Article 324", "gold_clause": None},
        {"category": "Structure", "query": "Who decides how money is shared between Centre and States?", "gold_article": "Article 280", "gold_clause": None},

        {"category": "Election-Trap", "query": "Can courts stop or interfere in elections?", "gold_article": "Article 329", "gold_clause": None},
        {"category": "Election-Trap", "query": "Can election results be challenged in court?", "gold_article": "Article 329", "gold_clause": None},

        {"category": "Multi-Hop", "query": "Is the right to constitutional remedies ever suspended?", "gold_article": "Article 32", "gold_clause": "(4)"},
        {"category": "Multi-Hop", "query": "Can Parliament give other courts the power to issue writs?", "gold_article": "Article 32", "gold_clause": "(3)"},

        {"category": "Amendment-Confusion", "query": "Can Parliament change fundamental rights?", "gold_article": "Article 368", "gold_clause": None},
        {"category": "Amendment-Confusion", "query": "What article talks about changing the Constitution itself?", "gold_article": "Article 368", "gold_clause": None},
    ]

    results_by_category = {}
    global_metrics = []

    # ============================================================
    # LOOP
    # ============================================================

    for item in evaluation_set:

        print(f"\nQuery: {item['query']}")

        results = retrieve_fn(item["query"])

        if results is None or len(results) == 0 or "text" not in results.columns:
            base_texts = []
            rerank_texts = []
        else:
            # BEFORE reranker
            base_texts = results["text"].tolist()

            # AFTER reranker
            reranked_df = rerank_results(item["query"], results, top_k=10)

            if reranked_df is None or reranked_df.empty:
                rerank_texts = []
            else:
                rerank_texts = reranked_df["text"].tolist()

        # ============================================================
        # METRICS — BEFORE
        # ============================================================

        base_r10 = recall_at_k(base_texts, item["gold_article"], 10)
        base_r5  = recall_at_k(base_texts, item["gold_article"], 5)
        base_r3  = recall_at_k(base_texts, item["gold_article"], 3)

        base_cr10 = clause_recall_at_k(base_texts, item["gold_article"], item["gold_clause"], 10)
        base_cr5  = clause_recall_at_k(base_texts, item["gold_article"], item["gold_clause"], 5)

        base_mrr = mrr_score(base_texts, item["gold_article"])

        # ============================================================
        # METRICS — AFTER
        # ============================================================

        rerank_r10 = recall_at_k(rerank_texts, item["gold_article"], 10)
        rerank_r5  = recall_at_k(rerank_texts, item["gold_article"], 5)
        rerank_r3  = recall_at_k(rerank_texts, item["gold_article"], 3)

        rerank_cr10 = clause_recall_at_k(rerank_texts, item["gold_article"], item["gold_clause"], 10)
        rerank_cr5  = clause_recall_at_k(rerank_texts, item["gold_article"], item["gold_clause"], 5)

        rerank_mrr = mrr_score(rerank_texts, item["gold_article"])

        # ============================================================
        # PRINT COMPARISON
        # ============================================================

        print(f"Recall@3: {base_r3} → {rerank_r3}")
        print(f"Recall@5: {base_r5} → {rerank_r5}")
        print(f"Clause Recall@5: {base_cr5} → {rerank_cr5}")
        print(f"Recall@10: {base_r10} → {rerank_r10}")
        print(f"MRR: {round(base_mrr,3)} → {round(rerank_mrr,3)}")

        record = {
            "Recall@10_before": base_r10,
            "Recall@5_before": base_r5,
            "Recall@3_before": base_r3,
            "Clause_Recall@10_before": base_cr10,
            "Clause_Recall@5_before": base_cr5,
            "MRR_before": base_mrr,

            "Recall@10_after": rerank_r10,
            "Recall@5_after": rerank_r5,
            "Recall@3_after": rerank_r3,
            "Clause_Recall@10_after": rerank_cr10,
            "Clause_Recall@5_after": rerank_cr5,
            "MRR_after": rerank_mrr
        }

        results_by_category.setdefault(item["category"], []).append(record)
        global_metrics.append(record)

    # ============================================================
    # AGGREGATION
    # ============================================================

    def aggregate(records):
        return {
            k: round(np.mean([r[k] for r in records]), 3)
            for k in records[0].keys()
        }

    print("\n" + "=" * 70)
    print("GLOBAL METRICS")
    print("=" * 70)

    global_results = aggregate(global_metrics)

    for k, v in global_results.items():
        print(f"{k}: {v}")

    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    return global_results

if __name__ == "__main__":

    print("\nINITIALIZING SYSTEM FOR EVALUATION...\n")

    from app.ingestion import build_corpus
    from app.chunking import create_chunks
    from app.embeddings import load_faiss_index
    from app.retrieval import initialize_retrieval, hybrid_retrieve

    # STEP 1 — Ingestion
    full_text = build_corpus()

    # STEP 2 — Chunking
    chunks = create_chunks(full_text)

    # STEP 3 — Load FAISS + Model
    dense_model, index = load_faiss_index()

    # STEP 4 — BM25
    texts = [c["text"] for c in chunks]
    bm25 = initialize_retrieval(texts)

    # STEP 5 — Create retriever wrapper
    def retrieve_fn(query):
        return hybrid_retrieve(
            query=query,
            dense_model=dense_model,
            index=index,
            bm25=bm25,
            chunks=chunks,
            top_k=20
        )

    # STEP 6 — Run evaluation
    run_evaluation(retrieve_fn)