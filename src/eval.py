"""
eval.py
Runs a fixed set of test questions through the full pipeline and
saves the answers + citations to a markdown report (eval_results.md).

The test set is deliberately diverse:
  - General conceptual questions (core RAG functionality)
  - Domain-specific questions (tests retrieval precision)
  - One deliberately OUT-OF-CORPUS question (tests that the system
    correctly refuses rather than hallucinating an uncited answer —
    this is the most important trust property of a "cited" system)

Usage:
    python src/eval.py
"""

from retrieval import Retriever
from generate import answer_question

TEST_QUESTIONS = [
    "How does self-attention work?",
    "What is multi-head attention?",
    "How do vision transformers use attention mechanisms?",
    "What is the computational complexity of self-attention?",
    "How does dilated attention differ from standard local attention?",
    # Deliberately out-of-corpus — nothing in a "transformer attention"
    # paper set should answer this. A correct system says so plainly.
    "What is the capital of France?",
]

OUTPUT_FILE = "eval_results.md"


def run_eval():
    print("Loading retriever...")
    retriever = Retriever()

    report_lines = ["# Evaluation Results\n"]

    for i, question in enumerate(TEST_QUESTIONS, start=1):
        print(f"\n[{i}/{len(TEST_QUESTIONS)}] {question}")
        answer, references = answer_question(question, retriever)

        report_lines.append(f"## Q{i}: {question}\n")
        report_lines.append(f"**Answer:**\n\n{answer}\n")
        report_lines.append(f"\n**References:**\n\n```\n{references}\n```\n")
        report_lines.append("\n---\n")

        print(f"    -> {answer[:100]}...")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\nDone. Full report written to {OUTPUT_FILE}")


if __name__ == "__main__":
    run_eval()
