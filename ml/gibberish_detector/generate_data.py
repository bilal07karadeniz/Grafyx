"""Generate training data for the gibberish detector.

Produces a balanced dataset of real code search queries and gibberish strings.
No external dependencies — uses only Python stdlib.

Output: data/train.jsonl, data/val.jsonl, data/test.jsonl
"""

import json
import os
import random
import string
import hashlib

random.seed(42)

# Load extracted English words from Python stdlib docstrings
_ENGLISH_WORDS_PATH = os.path.join(os.path.dirname(__file__), "data", "english_words.json")
if os.path.exists(_ENGLISH_WORDS_PATH):
    with open(_ENGLISH_WORDS_PATH) as _f:
        STDLIB_ENGLISH = json.load(_f)
else:
    STDLIB_ENGLISH = []

# ============================================================
# Real query components — extracted from common code patterns
# ============================================================

# Common programming terms (verbs)
CODE_VERBS = [
    "get", "set", "create", "delete", "update", "find", "search", "fetch",
    "send", "receive", "parse", "validate", "authenticate", "authorize",
    "connect", "disconnect", "initialize", "configure", "handle", "process",
    "transform", "convert", "serialize", "deserialize", "encode", "decode",
    "encrypt", "decrypt", "compress", "decompress", "upload", "download",
    "render", "display", "format", "filter", "sort", "merge", "split",
    "join", "map", "reduce", "aggregate", "calculate", "compute", "generate",
    "build", "compile", "deploy", "install", "register", "subscribe",
    "publish", "emit", "dispatch", "invoke", "execute", "run", "start",
    "stop", "pause", "resume", "retry", "rollback", "commit", "save",
    "load", "read", "write", "open", "close", "lock", "unlock", "sync",
    "cache", "flush", "clear", "reset", "refresh", "reload", "migrate",
    "export", "import", "inject", "resolve", "bind", "attach", "detach",
    "mount", "unmount", "wrap", "unwrap", "clone", "copy", "move",
]

# Common programming nouns
CODE_NOUNS = [
    "user", "account", "session", "token", "password", "email", "message",
    "notification", "event", "handler", "listener", "callback", "middleware",
    "router", "controller", "service", "repository", "factory", "builder",
    "adapter", "proxy", "decorator", "observer", "strategy", "command",
    "queue", "stack", "cache", "buffer", "pool", "stream", "channel",
    "socket", "connection", "request", "response", "header", "body",
    "payload", "config", "settings", "options", "params", "args",
    "database", "table", "column", "row", "record", "model", "schema",
    "migration", "query", "transaction", "cursor", "index", "key",
    "file", "directory", "path", "url", "endpoint", "route", "api",
    "server", "client", "worker", "thread", "process", "task", "job",
    "logger", "error", "exception", "warning", "debug", "trace",
    "test", "mock", "fixture", "assertion", "spec", "suite",
    "component", "module", "package", "plugin", "extension", "hook",
    "template", "layout", "view", "page", "form", "input", "button",
    "widget", "dialog", "modal", "menu", "toolbar", "sidebar",
    "image", "video", "audio", "document", "asset", "resource",
    "permission", "role", "policy", "rule", "filter", "validator",
    "parser", "formatter", "converter", "mapper", "reducer",
    "provider", "consumer", "producer", "subscriber", "publisher",
    "scheduler", "timer", "cron", "webhook", "websocket", "graphql",
    "authentication", "authorization", "encryption", "compression",
    "serialization", "deserialization", "pagination", "throttling",
    "middleware", "interceptor", "guard", "pipe", "resolver",
]

# Technical acronyms and abbreviations (look weird but are real)
TECH_TERMS = [
    "jwt", "oauth", "csrf", "cors", "ssl", "tls", "tcp", "udp", "http",
    "https", "grpc", "mqtt", "amqp", "smtp", "imap", "dns", "cdn",
    "api", "sdk", "cli", "gui", "ide", "orm", "sql", "nosql", "redis",
    "kafka", "rabbitmq", "celery", "nginx", "docker", "kubernetes",
    "aws", "gcp", "azure", "s3", "ec2", "lambda", "iam", "vpc",
    "html", "css", "dom", "svg", "json", "xml", "yaml", "toml", "csv",
    "utf", "ascii", "base64", "sha", "md5", "aes", "rsa", "hmac",
    "sse", "rtc", "webrtc", "webgl", "wasm", "pwa", "spa", "ssr",
    "crud", "rest", "soap", "rpc", "pub", "sub", "fifo", "lifo",
    "regex", "glob", "xpath", "jmespath", "jsonpath",
    "async", "await", "promise", "observable", "rxjs", "saga",
    "redux", "vuex", "pinia", "mobx", "zustand", "recoil",
    "react", "angular", "vue", "svelte", "nextjs", "nuxt",
    "fastapi", "django", "flask", "express", "nestjs", "spring",
    "pytest", "jest", "mocha", "cypress", "playwright", "selenium",
    "numpy", "pandas", "scipy", "sklearn", "tensorflow", "pytorch",
    "qdrant", "pinecone", "weaviate", "milvus", "chromadb",
    "postgresql", "mysql", "mongodb", "sqlite", "cassandra",
    "elasticsearch", "opensearch", "solr", "lucene",
    "webpack", "vite", "rollup", "esbuild", "babel", "typescript",
    "eslint", "prettier", "ruff", "mypy", "pylint", "flake8",
    "git", "github", "gitlab", "bitbucket", "jira", "confluence",
    "rgba", "rgb", "hex", "hsl", "hsla", "cmyk", "dpi", "fps",
    "gpu", "cpu", "ram", "ssd", "nvme", "pci", "usb", "hdmi",
]

# Common English words that appear in code/documentation — teaches the model
# general English character patterns so it generalizes to unseen words.
COMMON_ENGLISH = [
    # General nouns
    "name", "value", "type", "list", "item", "data", "text", "number",
    "string", "object", "class", "method", "function", "variable", "constant",
    "result", "output", "status", "state", "action", "context", "scope",
    "level", "count", "size", "length", "width", "height", "depth",
    "color", "label", "title", "description", "content", "version",
    "group", "category", "section", "block", "chunk", "batch", "range",
    "limit", "offset", "margin", "padding", "border", "radius",
    "source", "target", "origin", "destination", "location", "position",
    "parent", "child", "sibling", "ancestor", "descendant", "root",
    "node", "edge", "vertex", "graph", "tree", "leaf", "branch",
    "link", "chain", "sequence", "array", "matrix", "vector", "tensor",
    "signal", "noise", "pattern", "sample", "feature", "weight", "bias",
    "score", "rank", "priority", "order", "sort", "search",
    # General verbs
    "add", "remove", "insert", "append", "prepend", "push", "pop",
    "check", "verify", "test", "assert", "expect", "match", "compare",
    "enable", "disable", "toggle", "switch", "select", "deselect",
    "show", "hide", "expand", "collapse", "zoom", "pan", "scroll",
    "submit", "cancel", "confirm", "reject", "approve", "deny",
    "assign", "allocate", "release", "acquire", "dispose", "destroy",
    "wait", "sleep", "wake", "notify", "alert", "warn", "inform",
    # Adjectives/modifiers
    "active", "inactive", "enabled", "disabled", "visible", "hidden",
    "valid", "invalid", "required", "optional", "default", "custom",
    "primary", "secondary", "public", "private", "protected", "internal",
    "global", "local", "static", "dynamic", "abstract", "concrete",
    "virtual", "remote", "native", "foreign", "external", "embedded",
    "secure", "trusted", "verified", "certified", "signed", "unsigned",
    "async", "sync", "parallel", "sequential", "concurrent", "atomic",
    "mutable", "immutable", "readonly", "writable", "volatile",
    "temporary", "permanent", "persistent", "transient", "ephemeral",
    # Domain words
    "customer", "product", "inventory", "payment", "invoice", "receipt",
    "shipping", "delivery", "tracking", "address", "contact", "profile",
    "dashboard", "analytics", "report", "metric", "insight", "trend",
    "campaign", "audience", "segment", "funnel", "conversion",
    "calendar", "schedule", "booking", "reservation", "appointment",
    "notification", "reminder", "subscription", "newsletter",
    "workflow", "pipeline", "stage", "step", "milestone", "checkpoint",
    "review", "feedback", "comment", "annotation", "tag", "label",
    "folder", "archive", "backup", "restore", "snapshot", "replica",
    "cluster", "shard", "partition", "replica", "failover", "recovery",
    "monitor", "health", "status", "uptime", "latency", "throughput",
    "bandwidth", "capacity", "quota", "usage", "billing", "cost",
    # More tech brand names / tools (for generalization)
    "terraform", "pulumi", "ansible", "puppet", "chef", "vagrant",
    "consul", "vault", "nomad", "packer", "boundary", "waypoint",
    "grafana", "prometheus", "datadog", "splunk", "kibana", "logstash",
    "jaeger", "zipkin", "tempo", "loki", "cortex", "thanos",
    "istio", "envoy", "linkerd", "traefik", "haproxy", "caddy",
    "argocd", "flux", "spinnaker", "tekton", "jenkins", "drone",
    "harbor", "quay", "nexus", "artifactory", "sonatype",
    "dagger", "bazel", "gradle", "maven", "cargo", "poetry",
    "airflow", "prefect", "dagster", "luigi", "kedro", "metaflow",
    "spark", "flink", "beam", "storm", "samza", "pulsar",
    "superset", "redash", "tableau", "looker", "metabase", "dbt",
    "sentry", "bugsnag", "rollbar", "raygun", "honeybadger",
    "stripe", "paypal", "braintree", "adyen", "mollie",
    "twilio", "sendgrid", "mailgun", "postmark", "mailchimp",
    "auth0", "okta", "keycloak", "cognito", "firebase",
    "supabase", "hasura", "prisma", "drizzle", "typeorm",
    "tailwind", "bootstrap", "material", "chakra", "radix",
    "storybook", "chromatic", "figma", "sketch", "zeplin",
    "vercel", "netlify", "heroku", "render", "railway",
    "cloudflare", "fastly", "akamai", "bunny", "stackpath",
    "lambda", "fargate", "aurora", "dynamo", "kinesis",
    "pubsub", "bigtable", "spanner", "dataflow", "vertex",
    "cosmos", "synapse", "purview", "sentinel", "defender",
]

# File/path related terms
PATH_TERMS = [
    "src", "lib", "app", "core", "utils", "helpers", "common", "shared",
    "services", "models", "views", "controllers", "routes", "api",
    "components", "pages", "layouts", "hooks", "context", "store",
    "config", "settings", "constants", "types", "interfaces",
    "tests", "specs", "fixtures", "mocks", "stubs",
    "migrations", "seeds", "scripts", "tools", "bin",
    "static", "public", "assets", "media", "uploads",
    "templates", "emails", "notifications", "jobs", "tasks",
    "middleware", "guards", "interceptors", "pipes", "filters",
    "decorators", "validators", "serializers", "transformers",
]


def generate_real_queries(n: int) -> list[str]:
    """Generate realistic code search queries."""
    queries = []

    all_words = CODE_NOUNS + CODE_VERBS + TECH_TERMS + PATH_TERMS + COMMON_ENGLISH
    # English words from stdlib docs — teaches general English patterns
    all_english = all_words + STDLIB_ENGLISH
    patterns = [
        # Single terms — mix of code-specific and general English
        lambda: random.choice(all_words),
        lambda: random.choice(all_english),
        lambda: random.choice(all_english),
        lambda: random.choice(all_english),
        # Two-word combinations
        lambda: f"{random.choice(CODE_VERBS)} {random.choice(CODE_NOUNS)}",
        lambda: f"{random.choice(CODE_NOUNS)} {random.choice(CODE_NOUNS)}",
        lambda: f"{random.choice(CODE_NOUNS)} {random.choice(CODE_VERBS)}",
        # Three-word combinations
        lambda: f"{random.choice(CODE_VERBS)} {random.choice(CODE_NOUNS)} {random.choice(CODE_NOUNS)}",
        lambda: f"{random.choice(CODE_NOUNS)} {random.choice(CODE_VERBS)} {random.choice(CODE_NOUNS)}",
        # With tech terms
        lambda: f"{random.choice(TECH_TERMS)} {random.choice(CODE_NOUNS)}",
        lambda: f"{random.choice(CODE_NOUNS)} {random.choice(TECH_TERMS)}",
        lambda: f"{random.choice(CODE_VERBS)} {random.choice(TECH_TERMS)} {random.choice(CODE_NOUNS)}",
        # snake_case style (as if searching for a function name)
        lambda: f"{random.choice(CODE_VERBS)}_{random.choice(CODE_NOUNS)}",
        lambda: f"{random.choice(CODE_VERBS)}_{random.choice(CODE_NOUNS)}_{random.choice(CODE_NOUNS)}",
        # CamelCase style (as if searching for a class name)
        lambda: f"{random.choice(CODE_NOUNS).capitalize()}{random.choice(CODE_NOUNS).capitalize()}",
        lambda: f"{random.choice(CODE_NOUNS).capitalize()}{random.choice(CODE_VERBS).capitalize()}",
        # Natural language style
        lambda: f"how to {random.choice(CODE_VERBS)} {random.choice(CODE_NOUNS)}",
        lambda: f"where is {random.choice(CODE_NOUNS)} {random.choice(CODE_VERBS)}",
        lambda: f"{random.choice(CODE_NOUNS)} {random.choice(CODE_NOUNS)} {random.choice(CODE_VERBS)}ing",
        # Path-like queries
        lambda: f"{random.choice(PATH_TERMS)}/{random.choice(CODE_NOUNS)}",
        lambda: f"{random.choice(PATH_TERMS)} {random.choice(CODE_NOUNS)} {random.choice(CODE_VERBS)}",
        # With filler words (realistic queries include these)
        lambda: f"the {random.choice(CODE_NOUNS)} that {random.choice(CODE_VERBS)}s",
        lambda: f"{random.choice(CODE_NOUNS)} for {random.choice(CODE_NOUNS)} {random.choice(CODE_VERBS)}",
    ]

    while len(queries) < n:
        pattern = random.choice(patterns)
        q = pattern()
        queries.append(q)

    return queries


def generate_gibberish(n: int) -> list[str]:
    """Generate gibberish strings that look nothing like real queries."""
    queries = []

    # Consonant clusters that don't appear in English/code
    nonsense_syllables = [
        "xyzzy", "qlrmph", "zbnkwt", "fghjkl", "bvnmqw", "zcxvbn",
        "qwrtp", "dfghjk", "xlpqz", "mnbvcx", "pqrst", "wxyzt",
        "krmbl", "gnrph", "tskdf", "zxqwv", "blrft", "grmpf",
        "snkrd", "thwmp", "krznl", "plxft", "brngt", "dwqxz",
    ]

    def random_gibberish_word(min_len=3, max_len=10):
        """Generate a single gibberish word."""
        length = random.randint(min_len, max_len)
        # Mix of consonant-heavy patterns
        chars = []
        consonants = "bcdfghjklmnpqrstvwxyz"
        vowels = "aeiou"
        for i in range(length):
            if random.random() < 0.3:
                chars.append(random.choice(vowels))
            else:
                chars.append(random.choice(consonants))
        return "".join(chars)

    generators = [
        # Pure random characters
        lambda: " ".join(random_gibberish_word() for _ in range(random.randint(2, 5))),
        # Known nonsense syllables
        lambda: " ".join(random.sample(nonsense_syllables, random.randint(2, 4))),
        # Keyboard mashing patterns
        lambda: " ".join("".join(random.choices("asdfghjkl", k=random.randint(4, 8)))
                         for _ in range(random.randint(2, 4))),
        lambda: " ".join("".join(random.choices("qwertyuiop", k=random.randint(4, 8)))
                         for _ in range(random.randint(2, 4))),
        # Random lowercase strings
        lambda: " ".join("".join(random.choices(string.ascii_lowercase, k=random.randint(3, 8)))
                         for _ in range(random.randint(2, 5))),
        # Repeated characters
        lambda: " ".join(c * random.randint(3, 7)
                         for c in random.sample(string.ascii_lowercase, random.randint(2, 4))),
        # Reversed real words (still gibberish)
        lambda: " ".join(random.choice(CODE_NOUNS)[::-1] for _ in range(random.randint(2, 4))),
        # Random mixed with numbers
        lambda: " ".join("".join(random.choices(string.ascii_lowercase + string.digits, k=random.randint(4, 8)))
                         for _ in range(random.randint(2, 4))),
        # Single long gibberish string
        lambda: random_gibberish_word(15, 30),
        # Mixed gibberish syllables
        lambda: "".join(random.choice(nonsense_syllables) for _ in range(random.randint(2, 4))),
    ]

    while len(queries) < n:
        gen = random.choice(generators)
        q = gen()
        queries.append(q)

    return queries


def generate_hard_negatives(n: int) -> list[str]:
    """Generate tricky gibberish that might fool a simple model.

    These look *plausible* but contain NO real code words — pure gibberish
    that happens to be pronounceable or have real-looking structure.

    IMPORTANT: Never mix real code words with gibberish. A query like
    "user xkzptf session" looks like a real query with a typo, and labeling
    it as gibberish confuses the model.
    """
    queries = []

    def random_pronounceable(min_len=4, max_len=8):
        """Generate a pronounceable nonsense word (alternating consonant-vowel)."""
        consonants = "bcdfghjklmnpqrstvwxyz"
        vowels = "aeiou"
        length = random.randint(min_len, max_len)
        return "".join(random.choice(vowels) if i % 2 else random.choice(consonants)
                       for i in range(length))

    generators = [
        # Pronounceable gibberish words (look like real words but aren't)
        lambda: " ".join(random_pronounceable() for _ in range(random.randint(2, 4))),
        # Gibberish with real suffixes (no real root words)
        lambda: f"{random_pronounceable()}tion {random_pronounceable()}ment",
        lambda: f"{random_pronounceable()}ing {random_pronounceable()}er",
        lambda: f"{random_pronounceable()}able {random_pronounceable()}ize",
        # Fake snake_case (pronounceable but meaningless)
        lambda: f"{random_pronounceable()}_{random_pronounceable()}",
        lambda: f"{random_pronounceable()}_{random_pronounceable()}_{random_pronounceable()}",
        # Fake CamelCase
        lambda: f"{random_pronounceable().capitalize()}{random_pronounceable().capitalize()}",
        # Random 3-letter combos that look like acronyms but aren't
        lambda: " ".join("".join(random.choices(string.ascii_lowercase, k=3))
                         for _ in range(random.randint(3, 5))),
        # Single pronounceable gibberish word (short)
        lambda: random_pronounceable(3, 6),
        # Longer pronounceable gibberish
        lambda: random_pronounceable(10, 18),
    ]

    while len(queries) < n:
        gen = random.choice(generators)
        q = gen()
        queries.append(q)

    return queries


def split_data(examples: list[dict], train_ratio=0.8, val_ratio=0.1) -> tuple:
    """Split data into train/val/test sets deterministically."""
    random.shuffle(examples)
    n = len(examples)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return examples[:train_end], examples[train_end:val_end], examples[val_end:]


def main():
    print("Generating training data for gibberish detector...")

    # Generate balanced dataset
    n_per_class = 50000

    real_queries = generate_real_queries(n_per_class)
    gibberish = generate_gibberish(int(n_per_class * 0.7))  # 70% easy gibberish
    hard_negatives = generate_hard_negatives(int(n_per_class * 0.3))  # 30% hard cases

    # Build labeled examples
    examples = []
    for q in real_queries:
        examples.append({"query": q, "label": 1})  # 1 = real
    for q in gibberish + hard_negatives:
        examples.append({"query": q, "label": 0})  # 0 = gibberish

    # Deduplicate
    seen = set()
    unique = []
    for ex in examples:
        key = ex["query"].lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(ex)
    examples = unique

    # Add explicit single-word terms AFTER dedup. Ensures every known term
    # appears in training with enough frequency to learn its pattern.
    all_term_lists = [TECH_TERMS, CODE_VERBS, CODE_NOUNS, PATH_TERMS, COMMON_ENGLISH,
                      STDLIB_ENGLISH]
    seen_terms = set()
    for term_list in all_term_lists:
        for term in term_list:
            if term.lower() not in seen_terms:
                seen_terms.add(term.lower())
                for _ in range(2):
                    examples.append({"query": term, "label": 1})

    print(f"Total examples (with explicit terms): {len(examples)}")

    # Split
    train, val, test = split_data(examples)
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    # Save
    os.makedirs("data", exist_ok=True)
    for split_name, split_data_list in [("train", train), ("val", val), ("test", test)]:
        path = f"data/{split_name}.jsonl"
        with open(path, "w") as f:
            for ex in split_data_list:
                f.write(json.dumps(ex) + "\n")
        print(f"Saved {path} ({len(split_data_list)} examples)")

    # Print some examples
    print("\n--- Sample REAL queries ---")
    for ex in random.sample([e for e in test if e["label"] == 1], 10):
        print(f"  {ex['query']}")
    print("\n--- Sample GIBBERISH queries ---")
    for ex in random.sample([e for e in test if e["label"] == 0], 10):
        print(f"  {ex['query']}")


if __name__ == "__main__":
    main()
