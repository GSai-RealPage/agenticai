import chromadb

client = chromadb.PersistentClient(path=r"c:/code/agenticai/14_advanced/04_rag/chroma_db")

documents = [
    "Machine learning is a subset of Artificial Intelligence.",
    "Python is a popular programming language.",
    "The Taj Mahal is located in Agra.",
    "Neural networks are inspired by the human brain.",
    "The capital of France is Paris.",
    "FAISS is a library for similarity search.",
    "Large Language Models are trained on huge datasets.",
    "Cricket is one of the most popular sports in India.",
    "Pandas is used for data analysis in Python.",
    "ChromaDB is a vector database.",
]
ids = [str(i) for i in range(1, len(documents) + 1)]

# ChromaDB picks the distance function per collection via "hnsw:space":
#   "l2"     - squared Euclidean distance between the two vectors (smaller = closer)
#   "cosine" - cosine distance = 1 - cosine similarity (smaller = closer)
#   "ip"     - "inner product" distance = 1 - dot product (smaller = closer)
# ChromaDB's default embedding model returns unit-length (normalized) vectors, so with
# these embeddings cosine and ip distances come out identical, and l2 relates to cosine
# similarity by l2_distance = 2 - 2*cosine_similarity: three different formulas, but on
# normalized vectors they all rank the same documents in the same order.
collections = {}
for metric in ["l2", "cosine", "ip"]:
    name = f"distance_demo_{metric}"
    try:
        client.delete_collection(name)
    except Exception:
        pass
    collection = client.create_collection(name=name, metadata={"hnsw:space": metric})
    collection.add(ids=ids, documents=documents)
    collections[metric] = collection

query = "What is artificial intelligence?"
print(f"Query: {query!r}\n")

for metric, collection in collections.items():
    results = collection.query(query_texts=[query], n_results=3)
    print(f"--- {metric.upper()} ---")
    for rank, (doc, distance) in enumerate(
        zip(results["documents"][0], results["distances"][0]), start=1
    ):
        print(f"{rank}. (distance={distance:.4f}) {doc}")
    print()
