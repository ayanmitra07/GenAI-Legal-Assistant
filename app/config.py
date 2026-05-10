# ============================================================
# CONFIGURATION FILE - STEP 1
#Import OS (Operating System library), used for creating folder, checking files and working with paths

import os

# ---------------------------
#Base Path
# __file__: Special Python variable. The path of the current Python file being executed
#os.path.abspath(__file__): os =Python operating system module, path = submodule for working with file paths, abspath() returns absolute path of the file
#full location from the root of the computer | C:\Users\rittw\Desktop\LEGAL_AI_ASSISTANT\config.py
#abspath(__file__) ensures Python always knows the full location.
#dirname() It removes the file name and returns the folder.
#os.path.join() Safely combine folder paths
#JSON_PATH = os.path.join()
#EMBEDDINGS_PATH This will store the embedding vectors and .npy is a NumPy file format for arrays.
#FAISS_INDEX_PATH This file will store the vector search index.
#----------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MODEL_DIR = os.path.join(BASE_DIR, "models")
JSON_PATH = os.path.join(OUTPUT_DIR, "constitution_chunks.json")
EMBEDDINGS_PATH = os.path.join(OUTPUT_DIR, "constitution_embeddings.npy")
FAISS_INDEX_PATH = os.path.join(OUTPUT_DIR, "faiss_index.index")

#Models (Single Source of Truth)
#SentenceTransformer(EMBEDDING_MODEL_NAME)>intfloat/e5-base-v2> downloads the embedding model
#CrossEncoder(CROSS_ENCODER_NAME) used for reranking retrived chunks
#"mistral-7b-instruct-v0.2.Q4_K_M.gguf" is a 7 billion neural network parameters
# ---------------------------



LLM_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
)

EMBEDDING_MODEL_NAME = "intfloat/e5-base-v2"
CROSS_ENCODER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# SECTION 2 — DATA SOURCE CONFIGURATION
# ============================================================
#REPO_URL: Where the Constitution text repository lives
#CLONE_DIR: Folder where it will be downloaded
#Here a folder named "constitution_tm" will be created in the base directory and the repository will be cloned there.
REPO_URL = "https://github.com/Constitution-of-India/tm.git"
CLONE_DIR = os.path.join(BASE_DIR,"constitution_tm")

# Ensure directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)