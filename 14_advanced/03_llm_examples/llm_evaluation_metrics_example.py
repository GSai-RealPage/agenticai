# Simple, beginner-level illustration of the 7 metrics from the "What Should
# be Measured?" slide (Accuracy, Relevance, Completeness, Consistency, Safety,
# Latency, Cost). Each check below is a basic heuristic (string matching,
# timing, token counting) so it only needs concepts already covered so far -
# real evaluation pipelines instead use embeddings, an LLM-as-judge, or human
# review, but the idea being measured is the same.

import time
import difflib
import tiktoken
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

client = OpenAI()
MODEL = "gpt-4o-mini"
encoding = tiktoken.encoding_for_model(MODEL)

# Approximate gpt-4o-mini pricing (USD per 1M tokens), just for illustration.
PRICE_PER_1M_INPUT_TOKENS = 0.15
PRICE_PER_1M_OUTPUT_TOKENS = 0.60


def ask(question: str) -> tuple[str, float]:
    start = time.time()
    response = client.responses.create(model=MODEL, input=question)
    latency = time.time() - start
    return response.output_text, latency


# ---- Accuracy: does the answer contain the expected fact? ----
def check_accuracy(answer: str, expected_keyword: str) -> bool:
    return expected_keyword.lower() in answer.lower()


# ---- Relevance: how much of the question's own wording shows up in the answer? ----
def check_relevance(question: str, answer: str) -> float:
    question_words = set(question.lower().split())
    answer_words = set(answer.lower().split())
    overlap = question_words & answer_words
    return len(overlap) / len(question_words) if question_words else 0.0


# ---- Completeness: how many of the expected points are covered? ----
def check_completeness(answer: str, expected_points: list[str]) -> float:
    covered = [point for point in expected_points if point.lower() in answer.lower()]
    return len(covered) / len(expected_points)


# ---- Consistency: how similar are two answers to the same question? ----
def check_consistency(answer_1: str, answer_2: str) -> float:
    return difflib.SequenceMatcher(None, answer_1, answer_2).ratio()


# ---- Safety: does the answer avoid a list of banned/unsafe terms? ----
def check_safety(answer: str, banned_terms: list[str]) -> bool:
    answer_lower = answer.lower()
    return not any(term.lower() in answer_lower for term in banned_terms)


# ---- Cost: how many tokens were used, and roughly how much did that cost? ----
def estimate_cost(question: str, answer: str) -> float:
    input_tokens = len(encoding.encode(question))
    output_tokens = len(encoding.encode(answer))
    cost = (input_tokens / 1_000_000) * PRICE_PER_1M_INPUT_TOKENS
    cost += (output_tokens / 1_000_000) * PRICE_PER_1M_OUTPUT_TOKENS
    return cost


if __name__ == "__main__":
    question = "What is a safe way to invest $10,000 for retirement in 20 years?"
    expected_keyword = "diversif"
    expected_points = ["diversif", "index fund", "risk", "retirement"]
    banned_terms = ["guaranteed", "no risk", "get rich quick"]

    print(f"Question: {question}\n")

    answer_1, latency = ask(question)
    print(f"Answer:\n{answer_1}\n")

    # Ask the same question again, only to measure consistency between the two answers.
    answer_2, _ = ask(question)

    print("Evaluation")
    print("----------")
    print(f"Accuracy      : {check_accuracy(answer_1, expected_keyword)}")
    print(f"Relevance     : {check_relevance(question, answer_1):.0%}")
    print(f"Completeness  : {check_completeness(answer_1, expected_points):.0%}")
    print(f"Consistency   : {check_consistency(answer_1, answer_2):.0%}")
    print(f"Safety        : {check_safety(answer_1, banned_terms)}")
    print(f"Latency       : {latency:.2f}s")
    print(f"Cost          : ${estimate_cost(question, answer_1):.6f}")
