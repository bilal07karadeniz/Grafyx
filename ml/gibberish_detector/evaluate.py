"""Evaluate the trained gibberish detector.

Loads the numpy weights and runs evaluation on the test set.
Also tests edge cases and prints interactive examples.
"""

import json
import time

from features import load_vocabulary
from inference import GibberishClassifier


def load_test_data(path: str = "data/test.jsonl") -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def main():
    print("Loading model...")
    clf = GibberishClassifier("model/")

    # Evaluate on test set
    test_data = load_test_data()
    print(f"Test set: {len(test_data)} examples\n")

    correct = 0
    errors = []
    start = time.time()

    for ex in test_data:
        is_real, confidence = clf.predict(ex["query"])
        predicted_label = 1 if is_real else 0
        if predicted_label == ex["label"]:
            correct += 1
        else:
            errors.append({
                "query": ex["query"],
                "true_label": "real" if ex["label"] == 1 else "gibberish",
                "predicted": "real" if is_real else "gibberish",
                "confidence": confidence,
            })

    elapsed = time.time() - start
    accuracy = correct / len(test_data)
    print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Errors: {len(errors)} / {len(test_data)}")
    print(f"Inference time: {elapsed:.2f}s ({elapsed/len(test_data)*1000:.2f}ms per query)")

    if errors:
        print(f"\n--- First 20 misclassified ---")
        for err in errors[:20]:
            print(f"  [{err['true_label']:>9s} -> {err['predicted']:>9s}] "
                  f"(conf={err['confidence']:.3f}) {err['query']!r}")

    # Edge cases
    print("\n--- Edge Case Tests ---")
    edge_cases = [
        # Should be REAL — single words
        ("jwt", True),
        ("redis", True),
        ("user", True),
        ("send", True),
        ("config", True),
        # Should be REAL — multi-word
        ("websocket connection", True),
        ("csrf token", True),
        ("get_current_user", True),
        ("UserService", True),
        ("how to authenticate", True),
        ("database migration", True),
        ("api endpoint", True),
        ("redis cache", True),
        ("kubernetes deploy", True),
        # Should be GIBBERISH
        ("xyzzy foobar qlrmph", False),
        ("asdfghjkl qwerty", False),
        ("bbbbb nnnnn zzzzz", False),
        ("kcmxnv lqpwor zxbnm", False),
        ("aaaaaa bbbbbb", False),
        ("bopizu fexalo", False),
        ("tozameki wuvipod", False),
        # Tricky cases (should be REAL)
        ("grpc streaming", True),
        ("mqtt pub sub", True),
        ("rgba hex color", True),
        ("ssl certificate", True),
        ("pagination", True),
    ]

    edge_correct = 0
    for query, expected_real in edge_cases:
        is_real, confidence = clf.predict(query)
        status = "OK" if is_real == expected_real else "FAIL"
        if is_real == expected_real:
            edge_correct += 1
        label = "real" if is_real else "gibberish"
        print(f"  [{status:>4s}] {label:>9s} (conf={confidence:.3f}) {query!r}")

    print(f"\nEdge cases: {edge_correct}/{len(edge_cases)} correct")

    # ================================================================
    # GENERALIZATION TEST — terms NOT in training data
    # ================================================================
    print("\n--- Generalization Test (unseen terms) ---")
    unseen_cases = [
        # Real technical terms NOT in TECH_TERMS, CODE_VERBS, CODE_NOUNS, PATH_TERMS
        ("terraform", True),
        ("pulumi", True),
        ("datadog", True),
        ("grafana", True),
        ("prometheus", True),
        ("opentelemetry", True),
        ("jaeger", True),
        ("istio", True),
        ("envoy", True),
        ("consul", True),
        ("vault", True),
        ("ansible", True),
        ("packer", True),
        ("nomad", True),
        ("dagger", True),
        # Real multi-word queries with unseen terms
        ("terraform apply", True),
        ("grafana dashboard", True),
        ("prometheus metrics", True),
        ("ansible playbook", True),
        ("vault secrets", True),
        ("datadog tracing", True),
        # Unseen but clearly real function/class names
        ("get_deployment_status", True),
        ("CloudFormationStack", True),
        ("parse_helm_chart", True),
        ("validate_terraform_plan", True),
        # Clearly gibberish (also unseen patterns)
        ("zqxwvp mtkrlb nfjghd", False),
        ("qqqqwwww eeeerrrr", False),
        ("pxzlmn bvckqr", False),
        ("fghdjks wqpomx", False),
        ("mmmmm hhhhh jjjjj kkkkk", False),
    ]

    unseen_correct = 0
    for query, expected_real in unseen_cases:
        is_real, confidence = clf.predict(query)
        status = "OK" if is_real == expected_real else "FAIL"
        if is_real == expected_real:
            unseen_correct += 1
        label = "real" if is_real else "gibberish"
        print(f"  [{status:>4s}] {label:>9s} (conf={confidence:.3f}) {query!r}")

    print(f"\nGeneralization: {unseen_correct}/{len(unseen_cases)} correct "
          f"({unseen_correct/len(unseen_cases)*100:.1f}%)")

    # Re-run with production threshold (0.3)
    print("\n--- Generalization with production threshold (0.3) ---")
    prod_correct = 0
    for query, expected_real in unseen_cases:
        is_real, confidence = clf.predict(query, threshold=0.3)
        status = "OK" if is_real == expected_real else "FAIL"
        if is_real == expected_real:
            prod_correct += 1
        label = "real" if is_real else "gibberish"
        print(f"  [{status:>4s}] {label:>9s} (conf={confidence:.3f}) {query!r}")
    print(f"\nProduction threshold: {prod_correct}/{len(unseen_cases)} correct "
          f"({prod_correct/len(unseen_cases)*100:.1f}%)")


if __name__ == "__main__":
    main()
