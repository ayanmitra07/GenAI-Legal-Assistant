# ============================================================
# MAIN ENTRY — LEGAL AI SYSTEM
# ============================================================

from app.ingestion import build_corpus
from app.chunking import create_chunks
from app.embeddings import load_faiss_index
from app.retrieval import hybrid_retrieve, initialize_retrieval
from app.reranker import rerank_results
from app.reasoning import generate_answer
from app.intent import classify_intent
from app.guardrail import check_guardrails


# ============================================================
# BUILD SYSTEM (RUN ONCE)
# ============================================================

def build_system():

    print("\nBUILDING LEGAL AI SYSTEM...\n")

    # -------------------------
    # STEP 1 — INGESTION
    # -------------------------
    full_text = build_corpus()

    # -------------------------
    # STEP 2 — CHUNKING
    # -------------------------
    chunks = create_chunks(full_text)

    # -------------------------
    # STEP 3 — LOAD FAISS + MODEL
    # -------------------------
    dense_model, index = load_faiss_index()

    # -------------------------
    # STEP 4 — BM25 INIT
    # -------------------------
    texts = [c["text"] for c in chunks]
    bm25 = initialize_retrieval(texts)

    print("\nSYSTEM READY\n")

    return {
        "chunks": chunks,
        "dense_model": dense_model,
        "index": index,
        "bm25": bm25
    }


# ============================================================
# RETRIEVAL PIPELINE (WRAPPER)
# ============================================================

def retrieve_pipeline(query, system):

    return hybrid_retrieve(
        query=query,
        dense_model=system["dense_model"],
        index=system["index"],
        bm25=system["bm25"],
        chunks=system["chunks"],
        top_k=20   # 🔥 increased for better recall
    )


# ============================================================
# FULL QUERY PIPELINE
# ============================================================

def run_query(query, system):

    print("\n" + "=" * 70)
    print(f"USER QUERY: {query}")
    print("=" * 70)

    # -------------------------
    # STEP 0 — GUARDRAILS
    # -------------------------
    is_safe, message = check_guardrails(query)
    if not is_safe:
        print("Guardrail Triggered")
        return message

    # -------------------------
    # STEP 1 — INTENT
    # -------------------------
    intent = classify_intent(query)
    print("Detected Intent:", intent)

    # -------------------------
    # STEP 2 — RETRIEVAL
    # -------------------------
    results = retrieve_pipeline(query, system)

    if results is None or results.empty:
        return "No relevant results found."

    # -------------------------
    # STEP 3 — RERANKING
    # -------------------------
    print("Running Cross Encoder Reranking...")
    reranked = rerank_results(query, results, top_k=5)

    # -------------------------
    # STEP 4 — REASONING (LLM)
    # -------------------------
    answer = generate_answer(query, reranked, intent)

    return answer


# ============================================================
# EVALUATION WRAPPER
# ============================================================

def get_retriever_for_eval(system):
    """
    Returns function compatible with evaluation.py
    """

    def retrieve_fn(query):
        return hybrid_retrieve(
            query=query,
            dense_model=system["dense_model"],
            index=system["index"],
            bm25=system["bm25"],
            chunks=system["chunks"],
            top_k=20
        )

    return retrieve_fn


# ============================================================
# CHAT MODE
# ============================================================

def chat_mode(system):
    """
    Interactive chat loop
    """
    while True:
        user_query = input("\nEnter your legal query (or 'exit'): ")

        if user_query.lower() == "exit":
            print("\nExiting system...\n")
            break

        response = run_query(user_query, system)

        print("\nANSWER:\n")
        print(response)


# ============================================================
# EVALUATION MODE
# ============================================================

def eval_mode(system):
    """
    Runs evaluation explicitly
    """
    from app.evaluation import run_evaluation

    print("\n\nRUNNING EVALUATION...\n")

    retrieve_fn = get_retriever_for_eval(system)

    run_evaluation(retrieve_fn)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    # Build system once
    system = build_system()

    # -------------------------
    # MODE SELECTION
    # -------------------------
    mode = input("\nSelect mode (chat / eval): ").strip().lower()

    if mode == "eval":
        eval_mode(system)
    else:
        chat_mode(system)