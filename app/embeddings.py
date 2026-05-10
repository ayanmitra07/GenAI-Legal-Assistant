
import numpy as np
import faiss
import json
import pandas as pd
from sentence_transformers import SentenceTransformer #embedding model


from app.config import (
    JSON_PATH,
    EMBEDDINGS_PATH,
    FAISS_INDEX_PATH,
    EMBEDDING_MODEL_NAME
)
# ============================================================
# SECTION 3 — LOAD CORPUS (A core concept in AI/NLP)
# ============================================================
def load_chunks():
    print("="*60)
    print("SECTION 3 — LOAD CORPUS")
    print("="*60)

    # ---------------------------
    # STEP 1 — LOAD CHUNKS
    # ---------------------------
    with open(JSON_PATH, "r", encoding = "utf-8") as f:
        chunks =json.load(f)  #Convert JSON file > Python object

    constitution_df = pd.DataFrame(chunks) #Convert list of dictionaries into table format

    #Guard Clause : Protect the program from invalid input early
    if constitution_df.empty: #.empty = property of pandas DataFrame
        raise Exception("Corpus is empty - ingestion failed.") # raise: Forcefully stop the program and throw an error. Exception() is a built-in PA built-in error type in Python

    texts = constitution_df["text"].tolist() #Extract the text column from the DataFrame and convert it into a list. This will give us a list of all the chunk texts that we created in the previous steps.

    print("Total chunks loaded: ", len(texts))
    print("Sample chunk id: ", constitution_df.iloc[0]["chunk_id"])

    print("="*60)
    print("SECTION 3 COMPLETE")
    print("="*60)

    return texts



# ============================================================
# EMBEDDINGS + FAISS MODULE
# ============================================================

# ============================================================
# SECTION 4 — DENSE EMBEDDINGS (E5 BASE V2)
# ============================================================

def generate_embeddings(texts):
    print("="*60)
    print("SECTION 4 — DENSE EMBEDDINGS (E5 BASE V2)")
    print("="*60)



    #EMBEDDING_MODEL_NAME = "intfloat/e5-base-v2"
    #EMBEDDING_PATH = os.path.join(OUTPUT_DIR, "constitution_embeddings.npy")

    print("Loading embedding model: ", EMBEDDING_MODEL_NAME)

    #Download + load embedding model into memory
    dense_model = SentenceTransformer(EMBEDDING_MODEL_NAME)


    print("Total texts to embed:", len(texts))
    print("Generating embeddings...")

#ENCODING text → vectors

    clause_embeddings = dense_model.encode(
        #Prepare text before converting to vectors
        [f"passage: {t}" for t in texts], #Required E5 model: "query" for search wueries and "passage" for documents. We are aligning input format with how the embedding model was trained. models like E5 are trained with specific prefixes like "query:" and "passage:" to distinguish between queries and documents, which improves semantic retrieval performance.
        normalize_embeddings= True, #Normalize vectors (unit length). it is needed for cosine similarity search in FAISS. Normalization ensures that the length of the vector does not affect similarity calculations, allowing us to focus on the direction of the vectors in the embedding space.
        convert_to_numpy=True, #Convert the output to NumPy array format. This is useful for efficient storage and compatibility with libraries like FAISS that expect NumPy arrays.
        show_progress_bar=True,
        batch_size= 32 #Process 32 chunks at a time. Adjust based on your system's memory capacity. Larger batch sizes can speed up embedding generation but require more memory.
    )

    # Ensure FAISS compatibility
    clause_embeddings = clause_embeddings.astype("float32")

    # Save embeddings
    np.save(EMBEDDINGS_PATH, clause_embeddings) #Save the generated embeddings to a file in NumPy format. This allows us to reuse the embeddings later without having to regenerate them, which can save time and computational resources.

    print("Embedding matrix shape:", clause_embeddings.shape) #(number of chunks, embedding dimension)
    print("Saved embeddings to:", EMBEDDINGS_PATH)

    print("="*60)
    print("SECTION 4 COMPLETE")
    print("="*60)

    return clause_embeddings 

# ============================================================
# SECTION 5 — FAISS INDEX
# ============================================================
def build_faiss_index(embeddings, texts):
    """
    Builds FAISS vector index for fast similarity search

    Args:
        embeddings (np.array)
        texts (list)

    Returns:
        index (faiss index)
    """

    print("="*60)
    print("SECTION 5 — FAISS INDEX")
    print("="*60)

    #FAISS must know vector size before storing
    dimension = embeddings.shape[1] # .shape : Returns size of matrix and shape[1] gives us the embedding dimension.

    #Index: storage system, Flat: no compression (exact search), IP: Inner Product (dot product similarity). For cosine similarity, we can use inner product if the vectors are normalized.
    index = faiss.IndexFlatIP(dimension) #Create a FAISS index(vector database) 

    index.add(embeddings) #Add the clause embeddings to the FAISS index. This allows us to perform fast similarity search later when we want to retrieve relevant chunks based on a query.

    #We are saving the FAISS index (all vectors) to a file on disk so that we can load it later without having to recreate it from scratch. This is important for efficiency, especially if generating embeddings and building the index takes a long time.
    faiss.write_index(index, FAISS_INDEX_PATH)

    print("FAISS index size:", index.ntotal)

    #Validate that ALL embeddings were successfully stored in FAISS
    #index.ntotal: Number of vectors currently stored in FAISS
    #len(texts): Number of original chunks
    if index.ntotal != len(texts): #Vectors stored  vs  Original chunks
        raise Exception("FAISS index size mismatch!")

    print("="*60)
    print("SECTION 5 COMPLETE")
    print("="*60)

    return index

# ============================================================
# SECTION 6 — LOAD FAISS INDEX 
# ============================================================

def load_faiss_index():
    print("="*60)
    print("SECTION 6 — LOAD FAISS INDEX")
    print("="*60)

    # Load FAISS from disk
    index = faiss.read_index(FAISS_INDEX_PATH)

    # Load embedding model (same as used during indexing)
    dense_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print("FAISS loaded successfully")
    print("Total vectors:", index.ntotal)

    print("="*60)
    print("SECTION 6 COMPLETE")
    print("="*60)

    return dense_model, index
