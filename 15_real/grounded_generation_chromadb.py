# pip install chromadb openai python-dotenv

import os
import sys
import chromadb
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)
sys.stdout.reconfigure(encoding="utf-8")

DATA_DIR = os.path.join(os.path.dirname(__file__), "loan_data")
MODEL = "gpt-4o-mini"
openai_client = OpenAI()

# =====================================================================
# Same idea as grounded_generation_llamaindex.py - answer questions about
# the bank's loan policy with inline citations, so each claim in the
# answer can be traced back to an exact passage - but hand-rolled with
# plain OpenAI + ChromaDB instead of LlamaIndex's CitationQueryEngine,
# to show what that class is actually doing underneath:
#   1. Split source documents into small, numbered chunks.
#   2. Retrieve the top-k chunks for the question (semantic search).
#   3. Present them to the LLM as a numbered source list and instruct
#      it to cite [1], [2], etc. inline for every claim.
#   4. Print the chunks actually retrieved, so the citations can be
#      checked against real text, not just trusted.
#
# UNGROUNDED vs GROUNDED, side by side, on the same questions:
#   - ungrounded_answer(): the LLM answers from its own training data
#     alone, no retrieval at all - shows the hallucination risk for
#     facts (exact DTI caps, LTV bands, credit-score thresholds) that
#     are specific to THIS bank's policy and were never public training
#     data in the first place.
#   - grounded_answer(): retrieval + numbered inline citations, so the
#     answer is both accurate and independently checkable.
#
# Reuses the real bank policy + RBI documents from
# loan_eligibility_rag_reasoning.py's loan_data/, same as the
# LlamaIndex version - dense, numbered policy rules are exactly where
# precise citation matters.
# =====================================================================


def load_chunks() -> list[dict]:
    chunks = []
    for filename in ["loan_eligibility_policy.md", "rbi_guidelines.md"]:
        text = open(os.path.join(DATA_DIR, filename), encoding="utf-8").read()
        raw_sections = text.split("\n## ")
        sections = [raw_sections[0]] + [f"## {s}" for s in raw_sections[1:]]
        for i, section in enumerate(sections):
            section = section.strip()
            if section:
                chunks.append({"id": f"{filename}-{i}", "text": section, "source": filename})
    return chunks


def build_index() -> chromadb.Collection:
    collection = chromadb.Client().create_collection("loan_policy_citations")
    chunks = load_chunks()
    collection.add(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[{"source": c["source"]} for c in chunks],
    )
    return collection


POLICY_INDEX = build_index()


def retrieve(question: str, top_k: int = 5) -> list[dict]:
    results = POLICY_INDEX.query(query_texts=[question], n_results=top_k)
    return [
        {"text": doc, "source": meta["source"], "distance": dist}
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        )
    ]


def ungrounded_answer(question: str) -> str:
    response = openai_client.responses.create(
        model=MODEL,
        instructions="Answer the customer's home loan question as a bank advisor would.",
        input=question,
    )
    return response.output_text


def grounded_answer(question: str) -> tuple[str, list[dict]]:
    sources = retrieve(question)
    numbered_sources = "\n\n".join(f"[{i}] {s['text']}" for i, s in enumerate(sources, start=1))

    response = openai_client.responses.create(
        model=MODEL,
        instructions=(
            "Answer the question using ONLY the numbered sources below. Cite the source "
            "number inline in square brackets - e.g. [1] - immediately after every claim "
            "you make. If the sources don't contain the answer, say so instead of guessing."
        ),
        input=f"SOURCES:\n{numbered_sources}\n\nQuestion: {question}",
    )
    return response.output_text, sources


def ask(question: str):
    print("=" * 70)
    print(f"Q: {question}\n")

    print("--- Ungrounded (no retrieval, model's own training data) ---")
    print(ungrounded_answer(question))

    print("\n--- Grounded (retrieved chunks, inline [n] citations) ---")
    answer, sources = grounded_answer(question)
    print(answer)
    print("\nCited sources:")
    for i, s in enumerate(sources, start=1):
        preview = s["text"].replace("\n", " ")[:150]
        print(f"  [{i}] (distance={s['distance']:.3f}, {s['source']}) \"{preview}...\"")
    print()


if __name__ == "__main__":
    ask(
        "What is the maximum DTI ratio allowed for a salaried home loan applicant, "
        "and how is the maximum eligible loan amount calculated?"
    )
    ask(
        "What is the maximum loan-to-value ratio for a ₹50 lakh home loan, and what "
        "happens to an application if the applicant's credit score is below 650?"
    )
