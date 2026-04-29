"""Comprehensive generalization stress test for the gibberish detector.

Tests 500+ terms NOT present in any training word list.
This is the real test of whether the model generalizes.
"""

import time
from inference import GibberishClassifier


# ============================================================
# UNSEEN REAL TERMS — none of these appear in training data
# ============================================================

# Common English words (general vocabulary)
UNSEEN_ENGLISH = [
    "apple", "banana", "orange", "grape", "cherry", "lemon", "mango",
    "table", "chair", "window", "door", "floor", "ceiling", "wall",
    "water", "river", "ocean", "mountain", "forest", "desert", "island",
    "house", "garden", "kitchen", "bedroom", "bathroom", "garage",
    "school", "teacher", "student", "lesson", "homework", "exam",
    "doctor", "nurse", "hospital", "medicine", "patient", "surgery",
    "market", "price", "discount", "purchase", "receipt", "refund",
    "travel", "flight", "hotel", "ticket", "passport", "luggage",
    "weather", "temperature", "humidity", "pressure", "forecast",
    "music", "guitar", "piano", "violin", "drums", "singer",
    "movie", "actor", "director", "script", "scene", "camera",
    "animal", "elephant", "dolphin", "penguin", "tiger", "eagle",
    "planet", "galaxy", "asteroid", "comet", "nebula", "orbit",
    "engine", "battery", "circuit", "sensor", "motor", "piston",
    "fabric", "cotton", "leather", "silk", "wool", "velvet",
    "diamond", "silver", "copper", "bronze", "titanium", "platinum",
    "oxygen", "carbon", "nitrogen", "hydrogen", "helium", "calcium",
    "muscle", "skeleton", "protein", "vitamin", "mineral", "enzyme",
    "novel", "chapter", "author", "editor", "publisher", "reader",
    "bridge", "tunnel", "highway", "railway", "airport", "harbor",
    "budget", "salary", "pension", "mortgage", "interest", "dividend",
    "election", "candidate", "parliament", "congress", "senator", "mayor",
    "freedom", "justice", "equality", "democracy", "republic", "sovereign",
    "culture", "language", "dialect", "accent", "grammar", "alphabet",
    "energy", "voltage", "current", "resistance", "frequency", "wavelength",
    "gravity", "velocity", "momentum", "friction", "density", "volume",
]

# Tech tools/libraries NOT in training word lists
UNSEEN_TECH = [
    "bun", "deno", "zig", "nim", "odin", "vlang", "gleam", "roc",
    "htmx", "alpine", "petite", "astro", "remix", "gatsby", "hugo",
    "strapi", "directus", "payload", "sanity", "contentful",
    "turso", "neon", "planetscale", "xata", "fauna", "surreal",
    "upstash", "dragonfly", "valkey", "garnet", "keydb",
    "temporal", "inngest", "trigger", "defer", "quirrel",
    "clerk", "lucia", "authjs", "passport", "devise",
    "resend", "plunk", "loops", "novu", "courier",
    "inngest", "hono", "elysia", "nitro", "vinxi",
    "bento", "posthog", "amplitude", "mixpanel", "heap",
    "langchain", "llamaindex", "haystack", "semantic",
    "ollama", "groq", "together", "replicate", "modal",
    "wandb", "mlflow", "neptune", "comet", "clearml",
    "polars", "dask", "vaex", "modin", "cudf",
    "streamlit", "gradio", "panel", "solara", "marimo",
    "pydantic", "attrs", "msgspec", "cattrs", "mashumaro",
    "httpx", "aiohttp", "treq", "urllib", "requests",
    "sqlmodel", "tortoise", "peewee", "pony", "dataset",
    "celery", "huey", "dramatiq", "taskiq", "arq",
    "locust", "vegeta", "wrk", "hey", "bombardier",
    "snyk", "trivy", "grype", "clair", "anchore",
    "prowler", "checkov", "tfsec", "kics", "semgrep",
]

# Multi-word real queries with unseen terms
UNSEEN_MULTIWORD = [
    "apple payment integration",
    "hotel booking system",
    "weather forecast api",
    "music player widget",
    "elephant migration pattern",
    "budget calculator tool",
    "election results dashboard",
    "gravity simulation engine",
    "protein folding algorithm",
    "diamond inventory tracker",
    "bridge construction plan",
    "salary negotiation helper",
    "language translation service",
    "battery charging monitor",
    "oxygen level sensor",
    "htmx form submission",
    "polars dataframe filter",
    "streamlit sidebar layout",
    "pydantic model validation",
    "langchain agent executor",
    "ollama model inference",
    "temporal workflow retry",
    "posthog event tracking",
    "clerk user authentication",
    "turso database migration",
    "bun test runner",
    "deno deploy function",
    "astro page routing",
    "remix loader action",
    "hono middleware handler",
]

# Function/class names with unseen terms (snake_case / CamelCase)
UNSEEN_IDENTIFIERS = [
    "get_weather_forecast",
    "calculate_mortgage_rate",
    "parse_election_results",
    "validate_passport_number",
    "render_galaxy_map",
    "BatteryMonitor",
    "DiamondInventory",
    "ProteinFolder",
    "WeatherService",
    "BudgetCalculator",
    "GravitySimulator",
    "OxygenSensor",
    "LanguageDetector",
    "SalaryReport",
    "BridgeDesigner",
    "HotelBookingManager",
    "MusicPlayerController",
    "ElectionTracker",
    "TicketReservation",
    "CurrencyConverter",
]

# Short unseen terms (3-5 chars) — hardest for the model
UNSEEN_SHORT = [
    "elm", "lua", "php", "sql", "css", "jsx", "tsx", "vue",
    "pod", "gem", "pip", "npm", "apt", "yum", "rpm",
    "vim", "sed", "awk", "tar", "zip", "ssh", "ftp",
    "bus", "hub", "lab", "map", "log", "tag", "tab",
    "row", "col", "div", "nav", "img", "svg", "xml",
    "max", "min", "sum", "avg", "len", "str", "int",
    "bit", "hex", "bin", "oct", "nil", "nan", "inf",
    "err", "msg", "req", "res", "ctx", "env", "cfg",
]

# ============================================================
# UNSEEN GIBBERISH — various types the model should catch
# ============================================================

import random
import string
random.seed(99)  # Different seed from training

def gen_keyboard_mash(n):
    rows = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
    results = []
    for _ in range(n):
        row = random.choice(rows)
        words = []
        for _ in range(random.randint(2, 4)):
            words.append("".join(random.choices(row, k=random.randint(4, 8))))
        results.append(" ".join(words))
    return results

def gen_consonant_soup(n):
    consonants = "bcdfghjklmnpqrstvwxyz"
    results = []
    for _ in range(n):
        words = []
        for _ in range(random.randint(2, 4)):
            words.append("".join(random.choices(consonants, k=random.randint(3, 7))))
        results.append(" ".join(words))
    return results

def gen_repeated_chars(n):
    results = []
    for _ in range(n):
        words = []
        for _ in range(random.randint(2, 4)):
            c = random.choice(string.ascii_lowercase)
            words.append(c * random.randint(3, 8))
        results.append(" ".join(words))
    return results

def gen_random_strings(n):
    results = []
    for _ in range(n):
        words = []
        for _ in range(random.randint(2, 5)):
            words.append("".join(random.choices(string.ascii_lowercase, k=random.randint(3, 8))))
        results.append(" ".join(words))
    return results

def gen_pronounceable_gibberish(n):
    """CV-alternating nonsense — the hardest type."""
    consonants = "bcdfghjklmnpqrstvwxyz"
    vowels = "aeiou"
    results = []
    for _ in range(n):
        words = []
        for _ in range(random.randint(1, 3)):
            length = random.randint(4, 10)
            word = "".join(
                random.choice(vowels) if i % 2 else random.choice(consonants)
                for i in range(length)
            )
            words.append(word)
        results.append(" ".join(words))
    return results

def gen_fake_identifiers(n):
    """Fake snake_case and CamelCase gibberish."""
    consonants = "bcdfghjklmnpqrstvwxyz"
    vowels = "aeiou"
    results = []
    for _ in range(n):
        def nonsense(min_l=3, max_l=7):
            l = random.randint(min_l, max_l)
            return "".join(
                random.choice(vowels) if i % 2 else random.choice(consonants)
                for i in range(l)
            )
        if random.random() < 0.5:
            results.append(f"{nonsense()}_{nonsense()}")
        else:
            results.append(f"{nonsense().capitalize()}{nonsense().capitalize()}")
    return results

# Generate gibberish
GIBBERISH_KEYBOARD = gen_keyboard_mash(40)
GIBBERISH_CONSONANT = gen_consonant_soup(40)
GIBBERISH_REPEATED = gen_repeated_chars(30)
GIBBERISH_RANDOM = gen_random_strings(40)
GIBBERISH_PRONOUNCEABLE = gen_pronounceable_gibberish(50)
GIBBERISH_FAKE_IDS = gen_fake_identifiers(30)

ALL_GIBBERISH = (GIBBERISH_KEYBOARD + GIBBERISH_CONSONANT + GIBBERISH_REPEATED +
                 GIBBERISH_RANDOM + GIBBERISH_PRONOUNCEABLE + GIBBERISH_FAKE_IDS)


def run_category(clf, name, cases, expected_real, threshold=0.3, min_length=6):
    """Run a category of tests and return stats."""
    correct = 0
    failures = []
    for query in cases:
        is_real, conf = clf.predict(query, threshold=threshold, min_length=min_length)
        if is_real == expected_real:
            correct += 1
        else:
            failures.append((query, conf))

    pct = correct / len(cases) * 100 if cases else 0
    print(f"  {name:40s}  {correct:3d}/{len(cases):3d}  ({pct:5.1f}%)")
    if failures:
        # Show up to 5 failures
        for q, c in failures[:5]:
            label = "real" if expected_real else "gibberish"
            actual = "gibberish" if expected_real else "real"
            print(f"    FAIL: {q!r:40s}  conf={c:.3f}  (expected {label}, got {actual})")
        if len(failures) > 5:
            print(f"    ... and {len(failures) - 5} more failures")
    return correct, len(cases), failures


def main():
    print("Loading model...")
    clf = GibberishClassifier("model/")

    threshold = 0.3
    min_length = 6  # Short queries bypass model

    print(f"\nStress Test — threshold={threshold}, min_length={min_length}")
    print(f"  (queries < {min_length} chars bypass model → classified as real)")
    print(f"{'Category':40s}  {'Score':>9s}")
    print("-" * 60)

    total_correct = 0
    total_count = 0
    all_failures = []

    # Real term categories
    for name, cases in [
        ("English words (unseen)", UNSEEN_ENGLISH),
        ("Tech tools (unseen)", UNSEEN_TECH),
        ("Multi-word queries (unseen)", UNSEEN_MULTIWORD),
        ("Identifiers (unseen)", UNSEEN_IDENTIFIERS),
        ("Short terms 3-5 chars (unseen)", UNSEEN_SHORT),
    ]:
        c, n, f = run_category(clf, name, cases, expected_real=True,
                               threshold=threshold, min_length=min_length)
        total_correct += c
        total_count += n
        all_failures.extend([(q, conf, "real") for q, conf in f])

    # Gibberish categories
    for name, cases in [
        ("Keyboard mashing", GIBBERISH_KEYBOARD),
        ("Consonant soup", GIBBERISH_CONSONANT),
        ("Repeated characters", GIBBERISH_REPEATED),
        ("Random strings", GIBBERISH_RANDOM),
        ("Pronounceable gibberish", GIBBERISH_PRONOUNCEABLE),
        ("Fake identifiers", GIBBERISH_FAKE_IDS),
    ]:
        c, n, f = run_category(clf, name, cases, expected_real=False,
                               threshold=threshold, min_length=min_length)
        total_correct += c
        total_count += n
        all_failures.extend([(q, conf, "gibberish") for q, conf in f])

    print("-" * 60)
    total_pct = total_correct / total_count * 100
    print(f"  {'TOTAL':40s}  {total_correct:3d}/{total_count:3d}  ({total_pct:5.1f}%)")

    # Real vs gibberish breakdown
    real_cases = (UNSEEN_ENGLISH + UNSEEN_TECH + UNSEEN_MULTIWORD +
                  UNSEEN_IDENTIFIERS + UNSEEN_SHORT)
    gib_cases = ALL_GIBBERISH
    real_correct = sum(1 for q in real_cases
                       if clf.predict(q, threshold=threshold, min_length=min_length)[0])
    gib_correct = sum(1 for q in gib_cases
                      if not clf.predict(q, threshold=threshold, min_length=min_length)[0])

    print(f"\n  Real correctly classified:      {real_correct}/{len(real_cases)} "
          f"({real_correct/len(real_cases)*100:.1f}%)")
    print(f"  Gibberish correctly classified:  {gib_correct}/{len(gib_cases)} "
          f"({gib_correct/len(gib_cases)*100:.1f}%)")
    print(f"  Total failures: {len(all_failures)}")

    # Speed test
    print("\n--- Speed Test ---")
    all_queries = real_cases + gib_cases
    start = time.time()
    for _ in range(3):
        for q in all_queries:
            clf.predict(q, threshold=threshold)
    elapsed = time.time() - start
    total_preds = len(all_queries) * 3
    print(f"  {total_preds} predictions in {elapsed:.2f}s")
    print(f"  {elapsed/total_preds*1000:.2f}ms per prediction")


if __name__ == "__main__":
    main()
