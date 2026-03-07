"""25-feature extraction for Caller Disambiguator.

Given a (call_site, candidate_callee) pair, extracts 25 features
that help decide whether the call resolves to this particular callee.

Features:
   0: receiver_token_overlap_class_name  (Jaccard of receiver vs class name)
   1: receiver_char_bigram_sim_class_name
   2: caller_imports_callee_module       (bool)
   3: caller_imports_callee_package      (bool)
   4: file_path_distance                 (normalized)
   5: same_directory                     (bool)
   6: same_top_package                   (bool)
   7: has_dot_syntax                     (bool)
   8: receiver_is_self                   (bool)
   9: method_uniqueness                  (1 / count_of_methods_with_same_name)
  10: callee_is_method                   (bool)
  11: callee_is_standalone               (bool)
  12: same_language                      (bool)
  13: receiver_type_known                (bool)
  14: receiver_type_matches              (bool)
  15: callee_param_count                 (normalized)
  16: arg_count_matches_params           (bool)
  17: callee_has_decorator               (bool)
  18: receiver_name_length               (normalized)
  19: method_name_commonness             (frequency across codebase)
  20: caller_complexity                  (LOC normalized)
  21: callee_is_property                 (bool)
  22: callee_is_classmethod              (bool)
  23: callee_is_abstractmethod           (bool)
  24: receiver_is_common_pattern         (frequency of receiver pattern)
"""

import re
import numpy as np

FEATURE_COUNT = 25

# Common receiver patterns (e.g., self, cls, request, db, app)
COMMON_RECEIVERS = frozenset({
    "self", "cls", "request", "req", "response", "res",
    "db", "session", "app", "client", "conn", "cursor",
    "logger", "log", "config", "settings", "cache",
    "manager", "handler", "factory", "builder",
    "context", "ctx", "env",
})


def _split_tokens(text: str) -> list[str]:
    if not text:
        return []
    s = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', text)
    parts = re.split(r'[^a-zA-Z0-9]+', s.lower())
    return [p for p in parts if len(p) >= 2]


def _char_bigrams(text: str) -> set[str]:
    t = re.sub(r'[^a-z]', '', text.lower())
    if len(t) < 2:
        return set()
    return {t[i:i + 2] for i in range(len(t) - 1)}


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def _path_components(path: str) -> list[str]:
    """Split a file path into normalized components."""
    return [c for c in path.replace("\\", "/").split("/") if c]


def _path_distance(path_a: str, path_b: str) -> float:
    """Compute normalized distance between two file paths.

    Returns 0.0 for same file, 1.0 for completely different paths.
    Uses the edit distance on path components.
    """
    parts_a = _path_components(path_a)
    parts_b = _path_components(path_b)
    if not parts_a or not parts_b:
        return 1.0

    # Find common prefix length
    common = 0
    for a, b in zip(parts_a, parts_b):
        if a == b:
            common += 1
        else:
            break

    max_depth = max(len(parts_a), len(parts_b))
    if max_depth == 0:
        return 0.0
    return 1.0 - (common / max_depth)


def extract_features(
    # Call-site info
    receiver_text: str,
    method_name: str,
    caller_file: str,
    caller_imports: list[str],
    arg_count: int,
    has_dot_syntax: bool,
    caller_loc: int = 0,
    # Candidate callee info
    callee_name: str,
    callee_class_name: str,
    callee_file: str,
    callee_module: str,
    callee_package: str,
    callee_param_count: int,
    callee_is_method: bool,
    callee_decorators: list[str],
    callee_language: str = "python",
    caller_language: str = "python",
    # Disambiguation context
    method_count_with_same_name: int = 1,
    method_name_frequency: float = 0.0,
    receiver_type_annotation: str = "",
) -> np.ndarray:
    """Extract 25 features for a (call_site, candidate_callee) pair."""
    vec = np.zeros(FEATURE_COUNT, dtype=np.float32)

    receiver_lower = receiver_text.lower().strip() if receiver_text else ""
    receiver_tokens = set(_split_tokens(receiver_text)) if receiver_text else set()
    class_name_tokens = set(_split_tokens(callee_class_name)) if callee_class_name else set()

    # 0: receiver_token_overlap_class_name (Jaccard)
    vec[0] = _jaccard(receiver_tokens, class_name_tokens)

    # 1: receiver_char_bigram_sim_class_name
    if receiver_text and callee_class_name:
        r_bg = _char_bigrams(receiver_text)
        c_bg = _char_bigrams(callee_class_name)
        if r_bg and c_bg:
            vec[1] = len(r_bg & c_bg) / len(r_bg | c_bg)

    # 2: caller_imports_callee_module (bool)
    caller_imports_lower = [imp.lower() for imp in caller_imports]
    callee_module_lower = callee_module.lower() if callee_module else ""
    if callee_module_lower:
        vec[2] = 1.0 if any(
            callee_module_lower in imp or imp.endswith(callee_module_lower)
            for imp in caller_imports_lower
        ) else 0.0

    # 3: caller_imports_callee_package (bool)
    callee_package_lower = callee_package.lower() if callee_package else ""
    if callee_package_lower:
        vec[3] = 1.0 if any(
            callee_package_lower in imp for imp in caller_imports_lower
        ) else 0.0

    # 4: file_path_distance (normalized)
    vec[4] = _path_distance(caller_file, callee_file)

    # 5: same_directory (bool)
    caller_dir = "/".join(_path_components(caller_file)[:-1])
    callee_dir = "/".join(_path_components(callee_file)[:-1])
    vec[5] = 1.0 if caller_dir and caller_dir == callee_dir else 0.0

    # 6: same_top_package (bool)
    caller_parts = _path_components(caller_file)
    callee_parts = _path_components(callee_file)
    if caller_parts and callee_parts:
        vec[6] = 1.0 if caller_parts[0] == callee_parts[0] else 0.0

    # 7: has_dot_syntax (bool)
    vec[7] = 1.0 if has_dot_syntax else 0.0

    # 8: receiver_is_self (bool)
    vec[8] = 1.0 if receiver_lower in ("self", "cls") else 0.0

    # 9: method_uniqueness (1 / count_of_methods_with_same_name)
    vec[9] = 1.0 / max(1, method_count_with_same_name)

    # 10: callee_is_method (bool)
    vec[10] = 1.0 if callee_is_method else 0.0

    # 11: callee_is_standalone (bool)
    vec[11] = 1.0 if not callee_is_method else 0.0

    # 12: same_language (bool)
    vec[12] = 1.0 if caller_language == callee_language else 0.0

    # 13: receiver_type_known (bool)
    type_known = bool(receiver_type_annotation and receiver_type_annotation.strip())
    vec[13] = 1.0 if type_known else 0.0

    # 14: receiver_type_matches (bool)
    if type_known and callee_class_name:
        type_tokens = set(_split_tokens(receiver_type_annotation))
        class_tokens = set(_split_tokens(callee_class_name))
        vec[14] = 1.0 if (type_tokens & class_tokens) else 0.0

    # 15: callee_param_count (normalized)
    vec[15] = min(1.0, callee_param_count / 10.0)

    # 16: arg_count_matches_params (bool)
    # Allow +/- 1 for self/cls and optional params
    vec[16] = 1.0 if abs(arg_count - callee_param_count) <= 1 else 0.0

    # 17: callee_has_decorator (bool)
    vec[17] = 1.0 if callee_decorators else 0.0

    # 18: receiver_name_length (normalized)
    vec[18] = min(1.0, len(receiver_lower) / 20.0) if receiver_lower else 0.0

    # 19: method_name_commonness (frequency across codebase)
    vec[19] = min(1.0, method_name_frequency)

    # 20: caller_complexity (LOC normalized)
    vec[20] = min(1.0, caller_loc / 200.0)

    # 21: callee_is_property (bool)
    decorator_names = {d.lower().strip("@") for d in callee_decorators}
    vec[21] = 1.0 if "property" in decorator_names else 0.0

    # 22: callee_is_classmethod (bool)
    vec[22] = 1.0 if "classmethod" in decorator_names else 0.0

    # 23: callee_is_abstractmethod (bool)
    vec[23] = 1.0 if (
        "abstractmethod" in decorator_names
        or "abc.abstractmethod" in decorator_names
    ) else 0.0

    # 24: receiver_is_common_pattern (frequency of receiver pattern)
    if receiver_lower:
        # Exact match in common patterns
        if receiver_lower in COMMON_RECEIVERS:
            vec[24] = 1.0
        # Partial match (e.g., "self.cache" -> "self" is common)
        elif receiver_lower.split(".")[0] in COMMON_RECEIVERS:
            vec[24] = 0.5
        else:
            vec[24] = 0.0

    return vec
