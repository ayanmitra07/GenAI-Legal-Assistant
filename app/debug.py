#dense vector retrieval (FAISS search)
query = "how can i remove prime minister"

vec = dense_model.encode(
    [f"query: {query}"],
    normalize_embeddings=True
).astype("float32")

scores, indices = index.search(vec, 10)

print("Top Dense Results:")
for i in indices[0]:
    print(chunks[i]["chunk_id"])


#BM25 (keyword-based retrieval) testing

tokenized_query = "how can i remove prime minister".split()
bm25_scores = bm25.get_scores(tokenized_query)
top = np.argsort(bm25_scores)[::-1][:10]

for i in top:
    print(chunks[i]["chunk_id"])

#Manual corpus inspection / data sanity check
for c in chunks:
    if "Prime Minister" in c["text"]:
        print(c["chunk_id"])