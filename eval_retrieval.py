"""
Evaluation script for Telecom RAG retrieval quality.

Tests 10 hand-crafted (question, expected_source_id) pairs and measures:
- Top-k recall: Was the expected source retrieved in the top 3 results?
- Average similarity score of relevant documents
- Source type distribution in results
"""

import os
from retriever import build_retriever_with_scores

os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# Test cases: (question, expected_source_type, expected_source_id, description)
TEST_CASES = [
    (
        "Why is my internet speed so slow?",
        "faq",
        "2",
        "Data speed troubleshooting - should find FAQ #2"
    ),
    (
        "My calls keep dropping",
        "faq",
        "6",
        "Call quality issue - should find FAQ #6 about dropped calls"
    ),
    (
        "How do I enable Wi-Fi calling?",
        "faq",
        "7",
        "Wi-Fi calling feature - should find FAQ #7"
    ),
    (
        "How can I activate international roaming?",
        "faq",
        "9",
        "Roaming activation - should find FAQ #9"
    ),
    (
        "SIM card not detected after restart",
        "faq",
        "20",
        "SIM detection error - should find FAQ #20"
    ),
    (
        "What's wrong with my bill this month?",
        "faq",
        "16",
        "Billing inquiry - should find FAQ #16 about higher bills"
    ),
    (
        "How do I unlock my phone for another network?",
        "faq",
        "19",
        "Phone unlock - should find FAQ #19"
    ),
    (
        "What are the international roaming charges in Europe?",
        "faq",
        "10",
        "EU roaming rates - should find FAQ #10"
    ),
    (
        "I need to report a network outage in my area",
        "faq",
        "5",
        "Network outage reporting - should find FAQ #5"
    ),
    (
        "How do I set up autopay for my bill?",
        "faq",
        "17",
        "Autopay setup - should find FAQ #17"
    ),
]

SIMILARITY_THRESHOLD = 0.5
TOP_K = 3


def evaluate():
    """Run evaluation on retrieval quality."""
    print("=" * 80)
    print("TELECOM RAG RETRIEVAL EVALUATION")
    print("=" * 80)
    print(f"Similarity Threshold: {SIMILARITY_THRESHOLD}")
    print(f"Top-K: {TOP_K}")
    print()

    retriever = build_retriever_with_scores(k_faq=5, k_tickets=5, k_guides=5)

    total_queries = len(TEST_CASES)
    top_k_recall = 0
    above_threshold_count = 0
    source_distribution = {"faq": 0, "ticket": 0, "guide": 0}
    similarity_scores = []

    for idx, (question, expected_type, expected_id, description) in enumerate(TEST_CASES, 1):
        print(f"Test {idx}/{total_queries}: {description}")
        print(f"  Question: '{question}'")
        print(f"  Expected: {expected_type} #{expected_id}")

        results = retriever(question)

        if not results:
            print(f"  ❌ No results retrieved")
            print()
            continue

        print(f"  Retrieved {len(results)} results")
        print(f"  Top {min(TOP_K, len(results))} results:")

        # Check top-k recall
        found_expected = False
        for rank, (doc, score) in enumerate(results[:TOP_K], 1):
            source_type = doc.metadata.get("source", "unknown")
            source_id = None

            if source_type == "faq":
                source_id = doc.metadata.get("faq_id")
                display_id = f"FAQ #{source_id}"
            elif source_type == "ticket":
                source_id = doc.metadata.get("ticket_id")
                display_id = f"Ticket #{source_id}"
            elif source_type == "guide":
                source_id = doc.metadata.get("page_number")
                display_id = f"Guide (p.{source_id})"
            else:
                display_id = "Unknown"

            is_match = source_type == expected_type and str(
                source_id) == expected_id
            confidence_indicator = "✓" if is_match else " "

            print(
                f"    [{rank}] {display_id} - Score: {score:.4f} {confidence_indicator}"
            )

            if score >= SIMILARITY_THRESHOLD:
                if source_type not in source_distribution:
                    source_distribution[source_type] = 0
                source_distribution[source_type] += 1
                similarity_scores.append(score)

            if is_match:
                found_expected = True
                top_k_recall += 1
                print(f"        ✓ Expected source found at rank {rank}!")

        if not found_expected:
            print(f"    ❌ Expected source NOT in top-{TOP_K}")

        # Count results above threshold
        above_threshold = sum(
            1 for _, score in results if score >= SIMILARITY_THRESHOLD)
        above_threshold_count += above_threshold

        print(
            f"  Results above threshold ({SIMILARITY_THRESHOLD}): {above_threshold}/{len(results)}")
        print()

    # Print summary statistics
    print("=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Top-{TOP_K} Recall: {top_k_recall}/{total_queries} ({top_k_recall/total_queries*100:.1f}%)")
    print(
        f"Average results above threshold: {above_threshold_count/total_queries:.1f} per query")

    if similarity_scores:
        avg_similarity = sum(similarity_scores) / len(similarity_scores)
        max_similarity = max(similarity_scores)
        min_similarity = min(similarity_scores)
        print(
            f"Average similarity of above-threshold results: {avg_similarity:.4f}")
        print(f"  Max: {max_similarity:.4f}, Min: {min_similarity:.4f}")

    print("\nSource Distribution (results above threshold):")
    for source_type, count in source_distribution.items():
        pct = count / sum(source_distribution.values()) * \
            100 if sum(source_distribution.values()) > 0 else 0
        print(f"  {source_type.title()}: {count} ({pct:.1f}%)")

    print("\n" + "=" * 80)
    print(f"INTERPRETATION:")
    if top_k_recall >= 8:
        print(f"✓ Excellent retrieval performance (top-{TOP_K} recall ≥ 80%)")
    elif top_k_recall >= 6:
        print(f"~ Good retrieval performance (top-{TOP_K} recall ≥ 60%)")
    else:
        print(f"✗ Consider tuning retrieval parameters or similarity threshold")
    print("=" * 80)


if __name__ == "__main__":
    evaluate()
