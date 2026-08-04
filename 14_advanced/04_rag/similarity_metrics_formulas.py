# pip install sentence-transformers numpy

import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

sentence_a = "What is artificial intelligence?"
sentence_b = "Machine learning is a subset of Artificial Intelligence."

vector_a, vector_b = model.encode([sentence_a, sentence_b], convert_to_numpy=True)

# -------------------------------------------------
# Dot product: A . B = sum(A_i * B_i)
# -------------------------------------------------
dot_product = np.dot(vector_a, vector_b)

# -------------------------------------------------
# Cosine similarity: (A . B) / (||A|| * ||B||)
# -------------------------------------------------
norm_a = np.linalg.norm(vector_a)
norm_b = np.linalg.norm(vector_b)
cosine_similarity = dot_product / (norm_a * norm_b)

# -------------------------------------------------
# L2 / Euclidean distance: sqrt(sum((A_i - B_i)^2))
# -------------------------------------------------
l2_distance = np.sqrt(np.sum((vector_a - vector_b) ** 2))

print(f"Sentence A: {sentence_a!r}")
print(f"Sentence B: {sentence_b!r}\n")

print(f"||A|| (magnitude of A) = {norm_a:.4f}")
print(f"||B|| (magnitude of B) = {norm_b:.4f}\n")

print(f"Dot product        A . B            = {dot_product:.4f}")
print(f"Cosine similarity  (A.B)/(|A||B|)   = {cosine_similarity:.4f}")
print(f"L2 distance        sqrt(sum(Ai-Bi)^2) = {l2_distance:.4f}\n")

# sentence-transformers embeddings are (near) unit length, so ||A|| and ||B|| are ~1.
# That has two consequences worth pointing out:
#   1) dot product ~= cosine similarity (dividing by ~1 * ~1 barely changes anything)
#   2) sqrt(sum((Ai-Bi)^2)) == sqrt(2 - 2*cosine_similarity) for unit vectors
# ChromaDB's "l2" space (see chromadb_distance_metrics_comparison.py) reports the
# SQUARED Euclidean distance (no sqrt) for speed, i.e. exactly "2 - 2*cosine_similarity" -
# squaring the l2_distance computed above should reproduce that same number.
print(f"l2_distance^2 (matches ChromaDB's 'l2' space) = {l2_distance ** 2:.4f}")
print(f"2 - 2*cosine_similarity                        = {2 - 2 * cosine_similarity:.4f}")
