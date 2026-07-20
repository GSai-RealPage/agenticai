# pip install faiss-cpu sentence-transformers

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# -------------------------------------------------------
# Sample Documents
# -------------------------------------------------------
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
    "ChromaDB is a vector database."
]

# -------------------------------------------------------
# Load Embedding Model
# -------------------------------------------------------
print("Loading embedding model...")

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# -------------------------------------------------------
# Generate Embeddings
# -------------------------------------------------------
print("Generating embeddings...")

embeddings = model.encode(
    documents,
    convert_to_numpy=True
).astype("float32")

# -------------------------------------------------------
# Create FAISS Index
# -------------------------------------------------------
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

print(f"Number of vectors stored = {index.ntotal}")

# -------------------------------------------------------
# Search Function
# -------------------------------------------------------
def semantic_search(query, top_k=3):

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    ).astype("float32")

    distances, indices = index.search(
        query_embedding,
        top_k
    )

    print("\nQuery :", query)
    print("-" * 60)

    for rank, (idx, distance) in enumerate(
        zip(indices[0], distances[0]),
        start=1
    ):
        print(f"{rank}. Distance = {distance:.4f}")
        print(documents[idx])
        print()

# -------------------------------------------------------
# Test Queries
# -------------------------------------------------------
semantic_search("What is artificial intelligence?")

semantic_search("Tell me about Python programming")

semantic_search("Which library performs vector similarity search?")