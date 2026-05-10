

import numpy as np
import pandas as pd
import re
from rank_bm25 import BM25Okapi
# ============================================================
# SECTION 7 — HYBRID RETRIEVAL (LEGAL-OPTIMIZED CALIBRATION)
# ============================================================

# ------------------------------------------------------------
# TOKENIZATION (CONSISTENT FOR CORPUS + QUERY)
#Corpus tokenization = Query tokenization must be same
#Preprocessing layer for search systems
# ------------------------------------------------------------

def tokenize(text):
    text = text.lower() # we converts text to lowercase. 
                        #This ensures that both the corpus and the queries are tokenized 
                        #in a consistent manner, which is crucial for accurate BM25 retrieval.
    
    #re.sub(): replace pattern in text
    #[^..]: NOT since this is inside the square brackets, 
    # r means raw string it helps python to interpret backslashes correctly. 
    # a-z means all lowercase letters, 0-9 means all digits, \s means whitespace characters. 
    # So [^a-z0-9\s] means any character that is NOT a lowercase letter, a digit, or whitespace. 
    text = re.sub(r"[^a-z0-9\s]"," ", text) #Remove everything that is NOT letters, numbers, spaces

    tokens = text.split() #Split sentence into words using space

    #Common words that don’t add meaning. Removing them can improve keyword-based retrieval performance by focusing on more meaningful words.
    stopwords = {
        "how", "can", "i", "what", "is", "the", "a",
        "an", "under", "of", "in", "to", "and",
        "on", "for", "with", "by", "at", "from"
    }

    #List Comprehension = compact way to write a loop + filter
    tokens =  [t for t in tokens if t not in stopwords] #Loop through original list, Filter condition and finally What we ADD to new list t
    
    return tokens #Send processed tokens back to caller

# ============================================================
# QUERY EXPANSION (CRITICAL FOR MRR)
# ============================================================

def expand_query(query):
    q = query.lower()
    expansions = []

    # Fundamental Rights
    if "life" in q or "liberty" in q:
        expansions.append("Article 21 right to life personal liberty")

    if "arrest" in q or "police" in q:
        expansions.append("Article 22 arrest rights magistrate detention")

    if "speech" in q or "expression" in q:
        expansions.append("Article 19 freedom speech")

    # Courts / writs
    if "supreme court" in q or "writ" in q:
        expansions.append("Article 32 constitutional remedies writs")

    if "high court" in q:
        expansions.append("Article 226 high court writ jurisdiction")

    # Elections
    if "election" in q:
        expansions.append("Article 324 election commission")

    # Constitution amendment
    if "amend" in q or "change constitution" in q:
        expansions.append("Article 368 amendment constitution")

    # Finance
    if "money" in q or "finance" in q:
        expansions.append("Article 280 finance commission")

    return query + " " + " ".join(expansions)



# ============================================================
# SECTION 6 — BM25 INDEX (built again - better version)
#Return relevance score for EACH document based on keyword matching. 
#This is a traditional information retrieval technique that can complement dense vector search 
# by providing an additional signal of relevance based on keyword overlap.
# ============================================================
def initialize_retrieval(texts):
    """
    Builds BM25 index once
    """
    print("Initializing BM25...")
    print("="*60)
    print("SECTION 6 — BM25 INDEX")
    print("="*60)

    #We convert text into tokens (words) because BM25 works with discrete tokens. 
    #Example: "Article 21 Right to life" > ["Article", "21", "Right", "to", "life"]
    #t.split(): Splits string into words based on space
    #Loop through all chunks > split each > store results
    #new: ["habeas", "corpus", "law"] 

    tokenized_corpus = [tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized_corpus) # built BM25 index (keyword search engine)

    print("BM25 index ready with documents:", len(tokenized_corpus))

    print("="*60)
    print("SECTION 6 COMPLETE")
    print("="*60)

    return bm25



# ------------------------------------------------------------
# SECTION 7 — HYBRID  RETRIEVAL STARTEGY :(DENSE-DOMINANT) (LEGAL-OPTIMIZED CALIBRATION)
# ------------------------------------------------------------

#query: user question, top_k: final results to return, rrf_k:fusion parameter, use_reformulation: whether to use cross-encoder for reformulation and reranking
def hybrid_retrieve(query, dense_model, index, bm25, chunks, top_k=20, rrf_k = 60, use_reformulation = False):

    #Rescue Effect
    #Instead of taking only top 10 or 20 results, we fetch TOP 120 candidates first because retrieval is NOT perfect.
    #Even if model ranks poorly initially We still "rescue" the correct result later during reranking. 
    #This is especially important in legal domain where precision is crucial and we cannot afford to miss relevant articles.

    # -------------------------
    # STEP 0 — QUERY EXPANSION
    # -------------------------
    expanded_query = expand_query(query)
    
    #retrieval_depth ≠ final results    
    retrieval_depth = 120

    # ------------------------------------------------------------
    # STEP 2 — DENSE SEARCH (SEMANTIC)
    # ------------------------------------------------------------

    vec = dense_model.encode(
        [f"query: {expanded_query}"],
        normalize_embeddings = True,
    ).astype("float32") #FAISS requires float32 format

    #dense_scores: similarity values
    #dense_indices: positions of matching chunks
    #Tuple (scores, indices)
    
    dense_scores, dense_indices = index.search(vec, retrieval_depth) #Search FAISS index using the query vector. This returns the similarity scores and the corresponding indices of the retrieved chunks.

    # Convert FAISS results into a ranking dictionary
    #say we have dense_indices = [[23, 78, 5, 100]], FAISS supports batch queries so it returns a list of lists. We take the first list with [0] since we are only doing one query. 
    # Then we enumerate over the indices to get their rank (position in the retrieved list). We create a dictionary where the key is the chunk index and the value is its dense rank (rank + 1 because we want to start from 1 instead of 0).
    #Ranking starts from 1
    #Dictionary comprehension {key: value for ...}
    #idx = document ID
    dense_rank = {idx: rank +1 
                  for rank, idx in enumerate(dense_indices[0])} 

    
    # ------------------------------------------------------------
    # STEP 3 — BM25 SEARCH (KEYWORD)
    # ------------------------------------------------------------

    #BM25 is NOT semantic. it is  keyword matching, term frequency (TF), inverse document frequency (IDF)
    #Score = (query word importance) × (how often it appears in doc)
    tokenized_query = tokenize(expanded_query)
    
    bm25_scores = bm25.get_scores(tokenized_query) # bm25 works only on words. bm25.get_scores() computes the BM25 relevance scores for the tokenized query against all documents in the corpus. This returns a list of scores corresponding to each document.

    #Sorting result by index
    #Get top document IDs sorted by highest BM25 score. Indices of top documents ranked by BM25 because we need document IDs to fetch content later
    #np.argsort(bm25_scores) returns indices sorted by scoring
    #[::-1] Reverse the list Ascending > Descending
    #[:retrieval_depth] Take only top 120 results because we are doing hybrid retrieval and we want to keep the candidate pool manageable for the next steps. BM25 is used as a secondary signal to complement dense retrieval, so we don't need to consider all documents, just the top ones based on keyword relevance.
    
    bm25_top = np.argsort(bm25_scores)[::-1][:retrieval_depth]


    #Ranking Dictionary (future use for RRF needs rank, not scores)
    bm25_rank = {idx: rank + 1 for rank, idx in enumerate(bm25_top)}

    
    
    # ---------------- RRF FUSION --------------------------------|Combining signals|Reciprocal Rank Fusion combines rankings from multiple systems.
    #Dense → understands meaning
    #BM25 → understands keywords
    
    #  DENSE DOMINANT CALIBRATION

    #Dense is 3× more important than BM25
    dense_weight = 1.8
    bm25_weight = 0.6

    #All unique documents from BOTH systems
    #set(dense_rank) Extracts keys from the dense_rank dictionary and creates a set of those keys. dense_rank = {23:1, 78:2} > set(dense_rank) = {23, 78}
    #set(bm25_rank) Extracts keys from the bm25_rank dictionary and creates a set of those keys. bm25_rank = {5:1, 23:2} > set(bm25_rank) = {5, 23}
    #| operator: Set union

    candidate_indices = set(dense_rank) | set(bm25_rank)

    combined_scores = {} #Empty dictionary

    for idx in candidate_indices:
        score = 0 #Initialize score for each candidate document

        #If this document was found by Dense search, give it a score based on its rank.
        #Not all documents appear in both Dense and BM25.
        #Dense score ≠ BM25 score scale so here we Convert both > rank > same scoring formula
        #rrf_k controls how fast score decreases as rank goes down. Higher rrf_k means slower decrease, which can help give more weight to lower-ranked documents. In legal domain, sometimes relevant documents might not be ranked at the very top but still important, so we want to give them a chance.

        if idx in dense_rank: #Check if this document exists in Dense results
            #dense_rank[idx]: Get rank of document
            #dense_weight * > scales the contribution of the dense model to the final score. By multiplying the reciprocal of the rank by this weight, we can adjust how much influence the dense retrieval has on the final ranking compared to BM25.
            score += dense_weight * (1 / (rrf_k + dense_rank[idx])) #RRF formula: weight * (1/(k + rank))| #Top-ranked items get higher score and Lower-ranked items get smaller score

        if idx in bm25_rank: #Check if this document exists in BM25 results
            score += bm25_weight * (1 / (rrf_k + bm25_rank[idx]))

        combined_scores[idx] = score #Store the combined score where the key is the document index and the value is the final score after fusion.

    #Take all documents > sort by final score > return top results
    #key=lambda x: combined_scores[x] For each x: Use its score as sorting criteria
    #reverse=True > DESC order , since sorting best on scores and not document id. BEST results first
    sorted_indices = sorted(
        combined_scores,
        key=lambda x: combined_scores[x],
        reverse=True
    )[:top_k] #Take only top K results

    #Take top document IDs > fetch actual data > return it in a table format
    #For each index → get the corresponding chunk
    #return Send this result back to whoever called the function\
    results = []
    for i in sorted_indices:
        chunk = chunks[i].copy()
        chunk["score"] = combined_scores[i]   # 👈 ADD THIS
        results.append(chunk)

    return pd.DataFrame(results)

