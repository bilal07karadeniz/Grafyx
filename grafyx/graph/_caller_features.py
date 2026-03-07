"""Feature extraction for M2 Caller Disambiguator.

Extracts 25 features for each (caller_entry, target_class_name, method_name)
triple, using only data available from the caller entry dict and the graph's
pre-built indexes.  The feature vector feeds into a binary MLP that predicts
P(caller actually calls target_class.method).

Feature groups:
    0-1   Receiver-class similarity (token overlap, char bigram)
    2-6   Import / path proximity (import, package, path distance, same dir, same pkg)
    7-8   Call syntax (dot syntax, self receiver)
    9     Method uniqueness (1 / count_classes_with_method)
    10-11 Callee type flags (is_method, is_standalone)
    12    File type match
    13-14 Type resolution flags (trusted)
    15-17 Reserved (param count, arg match, decorator -- not available from index)
    18    Receiver name length (normalized)
    19    Method name commonness
    20-23 Reserved (LOC, property, classmethod, abstractmethod -- not available)
    24    Receiver is common pattern name

Dependencies:
    - numpy (always available)
    - graph instance for: _class_defined_in, _forward_import_index,
      _class_method_names
"""

import re

import numpy as np

FEATURE_COUNT = 25


def _char_bigrams(text: str) -> set[str]:
    """Extract character bigram set from text (lowercase, alpha only)."""
    t = re.sub(r'[^a-z]', '', text.lower())
    if len(t) < 2:
        return set()
    return {t[i:i + 2] for i in range(len(t) - 1)}


def _split_tokens(text: str) -> set[str]:
    """Split camelCase/snake_case identifier into lowercase token set."""
    if not text:
        return set()
    s = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', text)
    parts = re.split(r'[^a-zA-Z0-9]+', s.lower())
    return {p for p in parts if len(p) >= 2}


def extract_caller_features(
    caller_entry: dict,
    target_class_name: str,
    method_name: str,
    graph,  # CodebaseGraph instance
) -> np.ndarray:
    """Extract 25 features for P(caller actually calls target_class.method).

    Args:
        caller_entry: Dict with name, file, class?, _receivers?, _has_dot_syntax
        target_class_name: The class whose method we're checking
        method_name: The method name being called
        graph: CodebaseGraph instance for accessing indexes

    Returns:
        numpy array of shape (25,), dtype float32
    """
    vec = np.zeros(FEATURE_COUNT, dtype=np.float32)

    caller_file = caller_entry.get("file", "")
    caller_class = caller_entry.get("class", "")
    receivers = caller_entry.get("_receivers", set()) or set()
    has_dot = caller_entry.get("_has_dot_syntax", False)

    # Get class name tokens
    class_tokens = _split_tokens(target_class_name)

    # --- Feature 0: receiver_token_overlap_class_name (Jaccard) ---
    if receivers and class_tokens:
        all_recv_tokens: set[str] = set()
        for recv in receivers:
            all_recv_tokens.update(_split_tokens(recv))
        all_recv_tokens.discard("self")
        if all_recv_tokens:
            intersection = all_recv_tokens & class_tokens
            union = all_recv_tokens | class_tokens
            vec[0] = len(intersection) / len(union) if union else 0.0

    # --- Feature 1: receiver_char_bigram_sim_class_name ---
    if receivers:
        best_sim = 0.0
        class_bigrams = _char_bigrams(target_class_name)
        for recv in receivers:
            recv_clean = recv.replace("self.", "")
            if recv_clean:
                recv_bigrams = _char_bigrams(recv_clean)
                if recv_bigrams and class_bigrams:
                    inter = len(recv_bigrams & class_bigrams)
                    union_len = len(recv_bigrams | class_bigrams)
                    sim = inter / union_len if union_len else 0.0
                    best_sim = max(best_sim, sim)
        vec[1] = best_sim

    # --- Feature 2: caller_imports_callee_module (bool) ---
    target_files = getattr(graph, '_class_defined_in', {}).get(target_class_name, set())
    if caller_file and target_files:
        forward_imports = set(getattr(graph, '_forward_import_index', {}).get(caller_file, []))
        vec[2] = float(bool(forward_imports & target_files))

    # --- Feature 3: caller_imports_callee_package (bool) ---
    if caller_file and target_files:
        caller_pkg = "/".join(caller_file.replace("\\", "/").split("/")[:-1])
        for tf in target_files:
            target_pkg = "/".join(tf.replace("\\", "/").split("/")[:-1])
            if caller_pkg and target_pkg:
                for imp in getattr(graph, '_forward_import_index', {}).get(caller_file, []):
                    imp_pkg = "/".join(imp.replace("\\", "/").split("/")[:-1])
                    if imp_pkg == target_pkg:
                        vec[3] = 1.0
                        break
            if vec[3] == 1.0:
                break

    # --- Feature 4: file_path_distance (normalized shared prefix) ---
    if caller_file and target_files:
        best_dist = 0.0
        caller_parts = caller_file.replace("\\", "/").split("/")
        for tf in target_files:
            tf_parts = tf.replace("\\", "/").split("/")
            shared = 0
            for a, b in zip(caller_parts, tf_parts):
                if a == b:
                    shared += 1
                else:
                    break
            max_len = max(len(caller_parts), len(tf_parts))
            dist = shared / max_len if max_len else 0.0
            best_dist = max(best_dist, dist)
        vec[4] = best_dist

    # --- Feature 5: same_directory (bool) ---
    if caller_file and target_files:
        caller_dir = "/".join(caller_file.replace("\\", "/").split("/")[:-1])
        for tf in target_files:
            tf_dir = "/".join(tf.replace("\\", "/").split("/")[:-1])
            if caller_dir == tf_dir:
                vec[5] = 1.0
                break

    # --- Feature 6: same_top_package (bool) ---
    if caller_file and target_files:
        caller_parts = caller_file.replace("\\", "/").split("/")
        for tf in target_files:
            tf_parts = tf.replace("\\", "/").split("/")
            if (len(caller_parts) > 0 and len(tf_parts) > 0
                    and caller_parts[0] == tf_parts[0]):
                vec[6] = 1.0
                break

    # --- Feature 7: has_dot_syntax (bool) ---
    vec[7] = float(has_dot)

    # --- Feature 8: receiver_is_self (bool) ---
    if receivers:
        vec[8] = float(any(r == "self" or r.startswith("self.") for r in receivers))

    # --- Feature 9: method_uniqueness (1 / count_classes_with_method) ---
    class_method_names = getattr(graph, '_class_method_names', {})
    class_count = sum(
        1 for meths in class_method_names.values()
        if method_name in meths
    )
    vec[9] = 1.0 / max(class_count, 1)

    # --- Feature 10: callee_is_method (bool) ---
    vec[10] = float(bool(target_class_name))

    # --- Feature 11: callee_is_standalone (bool) ---
    vec[11] = float(not target_class_name)

    # --- Feature 12: same_language (bool, infer from extension) ---
    if caller_file and target_files:
        caller_ext = caller_file.rsplit(".", 1)[-1] if "." in caller_file else ""
        for tf in target_files:
            tf_ext = tf.rsplit(".", 1)[-1] if "." in tf else ""
            if caller_ext == tf_ext:
                vec[12] = 1.0
                break

    # --- Feature 13: receiver_type_known (bool) ---
    vec[13] = float(caller_entry.get("_trusted", False))

    # --- Feature 14: receiver_type_matches (bool) ---
    vec[14] = float(caller_entry.get("_trusted", False))

    # --- Features 15-17: Reserved (param count, arg match, decorator) ---
    # Not available from the index, default to 0.0

    # --- Feature 18: receiver_name_length (normalized) ---
    if receivers:
        max_len = max(len(r.replace("self.", "")) for r in receivers) if receivers else 0
        vec[18] = min(max_len / 20.0, 1.0)

    # --- Feature 19: method_name_commonness (frequency across classes) ---
    vec[19] = min(class_count / 10.0, 1.0)

    # --- Features 20-23: Reserved (LOC, property, classmethod, abstractmethod) ---
    # Not available from the index, default to 0.0

    # --- Feature 24: receiver_is_common_pattern ---
    common_receivers = {
        "db", "session", "app", "client", "conn", "cursor",
        "request", "response", "config", "logger", "cache",
        "redis", "celery", "task", "queue", "manager",
    }
    if receivers:
        recv_tokens: set[str] = set()
        for r in receivers:
            parts = r.replace("self.", "").split(".")
            recv_tokens.update(p.lower() for p in parts if p and p != "self")
        vec[24] = float(bool(recv_tokens & common_receivers))

    return vec
