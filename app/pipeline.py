from app.ingestion import build_corpus
from app.chunking import create_chunks
from app.embeddings import generate_embeddings, build_faiss_index
import traceback 


def run_pipeline():

    print("\nStarting Legal AI Data Pipeline...\n")

    try:
        # STEP 1 — INGESTION
        print("Step 1: Ingestion...")
        full_text = build_corpus()

        # STEP 2 — CHUNKING (ALREADY SAVES)
        print("Step 2: Chunking...")
        chunks = create_chunks(full_text)

        # STEP 3 — PREPARE TEXTS
        print("Step 3: Preparing texts...")
        texts = [c["text"] for c in chunks]

        # STEP 4 — EMBEDDINGS
        print("Step 4: Generating embeddings...")
        embeddings = generate_embeddings(texts)

        # STEP 5 — FAISS
        print("Step 5: Building FAISS index...")
        build_faiss_index(embeddings, texts)

        print("\n FULL PIPELINE COMPLETE\n")

    except Exception:
        print("Pipeline failed:")
        traceback.print_exc()

# ----------------------------------------
# ENTRY POINT
# ----------------------------------------
if __name__ == "__main__":
    run_pipeline()