from sentence_transformers import CrossEncoder
import pandas as pd
import numpy as np

# ------------------------------------------------------------
# Load Cross Encoder (balanced: good + fast)
# ------------------------------------------------------------
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")


# ------------------------------------------------------------
# RERANK FUNCTION
# ------------------------------------------------------------
def rerank_results(query, retrieved_df, top_k=5):

    print("Running Cross Encoder Reranking...")

    # ------------------------------------------------------------
    # Guard Clause
    # ------------------------------------------------------------
    if retrieved_df is None or retrieved_df.empty or "text" not in retrieved_df.columns:
        print("No retrieved results to rerank.")
        return pd.DataFrame(columns=["text"])

    # ------------------------------------------------------------
    # STEP 0 — SORT BY RETRIEVAL SCORE (important)
    # ------------------------------------------------------------
    if "score" in retrieved_df.columns:
        retrieved_df = retrieved_df.sort_values(
            by="score",
            ascending=False
        ).reset_index(drop=True)

    # ------------------------------------------------------------
    # STEP 0.5 — PRESERVE TOP RESULT (CRITICAL FOR MRR)
    # ------------------------------------------------------------
    top_chunk = None
    if len(retrieved_df) > 0:
        top_chunk = retrieved_df.iloc[0].copy()

    # ------------------------------------------------------------
    # STEP 1 — SKIP RERANKING IF HIGH CONFIDENCE
    # ------------------------------------------------------------
    if "score" in retrieved_df.columns:
        try:
            scores = retrieved_df["score"].astype(float).values

            if len(scores) > 1:
                top_score = scores[0]
                second_score = scores[1]
                gap = top_score - second_score

                # Strong confidence condition
                if top_score > 0.05 and gap > 0.01:
                    print("Skipping reranking (high confidence + clear gap)")
                    return retrieved_df.head(top_k)

        except:
            pass

    # ------------------------------------------------------------
    # STEP 2 — Prepare pairs
    # ------------------------------------------------------------
    texts = [str(t) for t in retrieved_df["text"].tolist()]
    pairs = [(query, text) for text in texts]

    # ------------------------------------------------------------
    # STEP 3 — Cross Encoder scoring
    # ------------------------------------------------------------
    scores = cross_encoder.predict(
        pairs,
        batch_size=32
    )

    # ------------------------------------------------------------
    # STEP 4 — Normalize rerank scores
    # ------------------------------------------------------------
    scores = np.array(scores)

    if len(scores) > 1:
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-6)
    else:
        scores = np.array([1.0])

    # ------------------------------------------------------------
    # STEP 5 — Attach rerank scores
    # ------------------------------------------------------------
    reranked_df = retrieved_df.copy()
    reranked_df["rerank_score"] = scores

    # ------------------------------------------------------------
    # STEP 6 — HYBRID SCORING (REAL FIX)
    # ------------------------------------------------------------
    if "score" in reranked_df.columns:

        retrieval_scores = reranked_df["score"].astype(float).values

        if len(retrieval_scores) > 1:
            retrieval_scores = (
                (retrieval_scores - retrieval_scores.min()) /
                (retrieval_scores.max() - retrieval_scores.min() + 1e-6)
            )
        else:
            retrieval_scores = np.array([1.0])

        reranked_df["retrieval_score_norm"] = retrieval_scores

        # Weighted fusion
        reranked_df["final_score"] = (
            0.65 * reranked_df["rerank_score"] +
            0.35 * reranked_df["retrieval_score_norm"]
        )

    else:
        reranked_df["final_score"] = reranked_df["rerank_score"]

    # ------------------------------------------------------------
    # STEP 7 — Sort final results
    # ------------------------------------------------------------
    reranked_df = reranked_df.sort_values(
        by="final_score",
        ascending=False
    ).reset_index(drop=True)

    # ------------------------------------------------------------
    # STEP 8 — PRESERVE TOP RESULT (MRR FIX)
    # ------------------------------------------------------------
    if top_chunk is not None:
        top_text = str(top_chunk["text"])

        if reranked_df.iloc[0]["text"] != top_text:
            reranked_df = pd.concat(
                [top_chunk.to_frame().T, reranked_df]
            ).drop_duplicates(subset=["text"]).reset_index(drop=True)

    # ------------------------------------------------------------
    # STEP 9 — Return Top K
    # ------------------------------------------------------------
    return reranked_df.head(top_k)