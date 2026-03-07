"""Tests for enhanced search: source token index, graph expansion, embeddings."""

from unittest.mock import MagicMock, patch
from grafyx.search import CodeSearcher, _tokenize_source, EmbeddingSearcher


def _make_mock_graph_with_source():
    """Create a mock graph that supports iter_functions_with_source()."""
    graph = MagicMock()

    # Standard function dicts (for get_all_functions / get_all_classes / get_all_files)
    graph.get_all_functions.return_value = [
        {
            "name": "send_event",
            "signature": "def send_event(data: dict)",
            "file": "/project/streaming.py",
            "language": "python",
            "line": 10,
            "docstring": "Send SSE event to client",
        },
        {
            "name": "create_access_token",
            "signature": "def create_access_token(user_id: str) -> str",
            "file": "/project/auth/jwt.py",
            "language": "python",
            "line": 15,
            "docstring": "Create a JWT access token",
        },
        {
            "name": "login",
            "signature": "def login(email: str, password: str)",
            "file": "/project/routes/auth.py",
            "language": "python",
            "line": 20,
            "docstring": "Handle user login",
        },
        {
            "name": "list_users",
            "signature": "def list_users()",
            "file": "/project/routes/admin.py",
            "language": "python",
            "line": 5,
            "docstring": "List all users",
        },
        {
            "name": "process_data",
            "signature": "def process_data(data: list)",
            "file": "/project/main.py",
            "language": "python",
            "line": 1,
            "docstring": "Process incoming data",
        },
    ]
    graph.get_all_classes.return_value = []
    graph.get_all_files.return_value = [
        {"path": "/project/streaming.py", "function_count": 3, "class_count": 0,
         "import_count": 2, "language": "python"},
        {"path": "/project/auth/jwt.py", "function_count": 2, "class_count": 0,
         "import_count": 1, "language": "python"},
        {"path": "/project/routes/auth.py", "function_count": 3, "class_count": 0,
         "import_count": 3, "language": "python"},
        {"path": "/project/routes/admin.py", "function_count": 2, "class_count": 0,
         "import_count": 2, "language": "python"},
        {"path": "/project/main.py", "function_count": 1, "class_count": 0,
         "import_count": 0, "language": "python"},
    ]

    # Source code for iter_functions_with_source()
    # send_event contains SSE-related terms in source but NOT in name/docstring
    graph.iter_functions_with_source.return_value = [
        (
            "send_event",
            "/project/streaming.py",
            '''def send_event(data: dict):
    """Send SSE event to client"""
    response = StreamingResponse(
        content_type="text/event-stream"
    )
    event_data = json.dumps(data)
    return f"data: {event_data}\\n\\n"
''',
            "",
        ),
        (
            "create_access_token",
            "/project/auth/jwt.py",
            '''def create_access_token(user_id: str) -> str:
    """Create a JWT access token"""
    payload = {"sub": user_id, "exp": datetime.utcnow() + timedelta(hours=1)}
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")
''',
            "",
        ),
        (
            "login",
            "/project/routes/auth.py",
            '''def login(email: str, password: str):
    """Handle user login"""
    user = authenticate(email, password)
    token = create_access_token(user.id)
    return {"access_token": token}
''',
            "",
        ),
        (
            "list_users",
            "/project/routes/admin.py",
            '''def list_users():
    """List all users"""
    return db.query(User).all()
''',
            "",
        ),
        (
            "process_data",
            "/project/main.py",
            '''def process_data(data: list):
    """Process incoming data"""
    return [transform(d) for d in data]
''',
            "",
        ),
    ]

    # Caller index: login and list_users call create_access_token
    graph._caller_index = {
        "create_access_token": [
            {"name": "login", "file": "/project/routes/auth.py"},
            {"name": "list_users", "file": "/project/routes/admin.py"},
        ],
        "send_event": [
            {"name": "stream_chat", "file": "/project/api/chat.py"},
        ],
    }

    # Import index: routes/auth.py imports auth/jwt.py
    graph.get_importers.side_effect = lambda path: {
        "/project/auth/jwt.py": ["/project/routes/auth.py", "/project/routes/admin.py"],
        "/project/streaming.py": ["/project/api/chat.py", "/project/api/events.py"],
    }.get(path, [])

    graph._import_index = {
        "/project/auth/jwt.py": ["/project/routes/auth.py", "/project/routes/admin.py"],
        "/project/streaming.py": ["/project/api/chat.py", "/project/api/events.py"],
    }

    return graph


# ── Test _tokenize_source ──


class TestTokenizeSource:
    def test_extracts_identifiers(self):
        source = "def process_data(items):\n    return transform(items)"
        tokens = _tokenize_source(source)
        assert "process" in tokens
        assert "transform" in tokens

    def test_extracts_string_contents(self):
        source = '''response = Response(content_type="text/event-stream")'''
        tokens = _tokenize_source(source)
        assert "text" in tokens
        assert "event" in tokens
        assert "stream" in tokens

    def test_filters_code_stopwords(self):
        source = "def foo():\n    return self.value if True else None"
        tokens = _tokenize_source(source)
        assert "self" not in tokens
        assert "true" not in tokens
        assert "none" not in tokens

    def test_filters_short_tokens(self):
        source = "x = ab + cd"
        tokens = _tokenize_source(source)
        # All tokens are < 3 chars, should be empty
        assert len(tokens) == 0

    def test_splits_camel_case(self):
        source = "result = StreamingResponse()"
        tokens = _tokenize_source(source)
        assert "streaming" in tokens
        assert "response" in tokens

    def test_empty_source(self):
        assert _tokenize_source("") == []
        assert _tokenize_source(None) == []


# ── Test source token index ──


class TestSourceTokenIndex:
    def test_source_index_built_lazily(self):
        graph = _make_mock_graph_with_source()
        searcher = CodeSearcher(graph)
        assert searcher._source_index is None
        searcher._ensure_source_index()
        assert searcher._source_index is not None

    def test_source_index_finds_sse_terms(self):
        graph = _make_mock_graph_with_source()
        searcher = CodeSearcher(graph)
        searcher._ensure_source_index()
        # "event" should be in the index, pointing to send_event
        assert "event" in searcher._source_index
        # Find the symbol index for send_event
        event_indices = searcher._source_index["event"]
        symbol_names = [searcher._source_symbols[i][0] for i in event_indices]
        assert "send_event" in symbol_names

    def test_source_search_finds_streaming_functions(self):
        graph = _make_mock_graph_with_source()
        searcher = CodeSearcher(graph)
        token_weights = {"streaming": 1.0, "sse": 1.0, "event": 1.0}
        results = searcher._source_search(["streaming", "event"], token_weights)
        names = [r[0] for r in results]
        assert "send_event" in names

    def test_source_search_returns_empty_for_unknown_terms(self):
        graph = _make_mock_graph_with_source()
        searcher = CodeSearcher(graph)
        token_weights = {"xyznonexistent": 1.0}
        results = searcher._source_search(["xyznonexistent"], token_weights)
        assert results == []


# ── Test graph expansion ──


class TestGraphExpansion:
    def test_caller_expansion_finds_consumers(self):
        """Searching for 'JWT token' should find create_access_token AND its callers."""
        graph = _make_mock_graph_with_source()
        searcher = CodeSearcher(graph)
        results = searcher.search("JWT access token", max_results=10)
        names = [r["name"] for r in results]
        # Should find the definition
        assert "create_access_token" in names
        # Should also find callers via graph expansion
        assert "login" in names or any("calls create_access_token" in r.get("context", "") for r in results)

    def test_import_expansion_finds_importing_files(self):
        """search_files for 'JWT token' should include files that import jwt.py."""
        graph = _make_mock_graph_with_source()
        searcher = CodeSearcher(graph)
        results = searcher.search_files("JWT access token", max_results=10)
        files = [r["file"] for r in results]
        # auth/jwt.py should be found directly
        assert any("jwt.py" in f for f in files)
        # routes/auth.py should appear via import expansion
        assert any("routes/auth.py" in f for f in files) or any("routes/admin.py" in f for f in files)

    def test_source_fallback_enables_expansion_chain(self):
        """Query 'SSE streaming' should find send_event (source) then its callers."""
        graph = _make_mock_graph_with_source()
        searcher = CodeSearcher(graph)
        results = searcher.search("streaming event source", max_results=10)
        names = [r["name"] for r in results]
        files = [r["file_path"] for r in results]
        # send_event found via source tokens
        assert "send_event" in names
        # Its caller should appear via graph expansion
        assert "stream_chat" in names or any("chat.py" in f for f in files)


# ── Test search_files with fallbacks ──


class TestSearchFilesEnhanced:
    def test_source_fallback_in_search_files(self):
        """search_files should find streaming.py via source tokens."""
        graph = _make_mock_graph_with_source()
        searcher = CodeSearcher(graph)
        results = searcher.search_files("SSE event streaming", max_results=10)
        files = [r["file"] for r in results]
        assert any("streaming.py" in f for f in files)

    def test_import_expansion_in_search_files(self):
        """search_files should find files that import high-scoring files."""
        graph = _make_mock_graph_with_source()
        searcher = CodeSearcher(graph)
        results = searcher.search_files("JWT token authentication", max_results=10)
        files = [r["file"] for r in results]
        # Should include jwt.py or its importers
        has_jwt = any("jwt.py" in f for f in files)
        has_importers = any("routes" in f for f in files)
        assert has_jwt or has_importers


# ── Test EmbeddingSearcher class ──


class TestEmbeddingSearcher:
    def test_available_without_fastembed(self):
        """EmbeddingSearcher.available() should return False without fastembed."""
        # fastembed is not installed in test environment
        with patch("grafyx.search._embeddings._HAS_EMBEDDINGS", False):
            assert not EmbeddingSearcher.available()

    def test_embedding_search_returns_empty_when_not_ready(self):
        """Embedding search should return empty when not built."""
        graph = _make_mock_graph_with_source()
        searcher = CodeSearcher(graph)
        searcher._embedding_searcher = None
        results = searcher._embedding_search("test query")
        assert results == []

    def test_embedding_init_skipped_without_fastembed(self):
        """_ensure_embeddings should not crash without fastembed."""
        graph = _make_mock_graph_with_source()
        searcher = CodeSearcher(graph)
        with patch("grafyx.search._embeddings._HAS_EMBEDDINGS", False):
            searcher._ensure_embeddings()
            assert searcher._embedding_searcher is None


# ── Test that existing functionality is preserved ──


class TestExistingSearchPreserved:
    def test_exact_name_match_still_works(self):
        graph = _make_mock_graph_with_source()
        searcher = CodeSearcher(graph)
        results = searcher.search("process_data")
        names = [r["name"] for r in results]
        assert "process_data" in names

    def test_docstring_match_still_works(self):
        graph = _make_mock_graph_with_source()
        searcher = CodeSearcher(graph)
        results = searcher.search("handle login")
        names = [r["name"] for r in results]
        assert "login" in names

    def test_search_files_still_works(self):
        graph = _make_mock_graph_with_source()
        searcher = CodeSearcher(graph)
        results = searcher.search_files("data processing")
        assert len(results) > 0
        assert "file" in results[0]


# ── Test source token blending in primary scoring ──


def _make_mock_graph_with_jwt_functions():
    """Create a mock graph with JWT-related functions for scoring tests."""
    graph = MagicMock()

    graph.get_all_functions.return_value = [
        {
            "name": "create_access_token",
            "signature": "def create_access_token(user_id: str) -> str",
            "file": "/project/auth/jwt.py",
            "language": "python",
            "line": 15,
            "docstring": "Create a JWT access token",
        },
        {
            "name": "get_current_user",
            "signature": "def get_current_user(token: str) -> User",
            "file": "/project/auth/deps.py",
            "language": "python",
            "line": 10,
            "docstring": "Extract current user from request",
        },
        {
            "name": "list_users",
            "signature": "def list_users()",
            "file": "/project/routes/admin.py",
            "language": "python",
            "line": 5,
            "docstring": "List all users",
        },
    ]
    graph.get_all_classes.return_value = []
    graph.get_all_files.return_value = [
        {"path": "/project/auth/jwt.py", "function_count": 2, "class_count": 0,
         "import_count": 1, "language": "python"},
        {"path": "/project/auth/deps.py", "function_count": 1, "class_count": 0,
         "import_count": 2, "language": "python"},
        {"path": "/project/routes/admin.py", "function_count": 2, "class_count": 0,
         "import_count": 2, "language": "python"},
    ]

    # get_current_user calls jwt.decode() — key source tokens
    graph.iter_functions_with_source.return_value = [
        (
            "create_access_token",
            "/project/auth/jwt.py",
            '''def create_access_token(user_id: str) -> str:
    """Create a JWT access token"""
    payload = {"sub": user_id, "exp": datetime.utcnow() + timedelta(hours=1)}
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")
''',
            "",
        ),
        (
            "get_current_user",
            "/project/auth/deps.py",
            '''def get_current_user(token: str) -> User:
    """Extract current user from request"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=401)
        user = db.query(User).filter(User.id == user_id).first()
        return user
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid authentication")
''',
            "",
        ),
        (
            "list_users",
            "/project/routes/admin.py",
            '''def list_users():
    """List all users"""
    return db.query(User).all()
''',
            "",
        ),
    ]

    graph._caller_index = {}
    graph.get_importers.side_effect = lambda path: []
    graph._import_index = {}

    return graph


class TestSourceTokenBlending:
    """Test that source token signals are blended into primary scoring."""

    def test_jwt_query_finds_get_current_user(self):
        """'JWT authentication token verification' should find get_current_user
        via source tokens (jwt, decode, authentication) even though its name
        doesn't contain 'jwt'."""
        graph = _make_mock_graph_with_jwt_functions()
        searcher = CodeSearcher(graph)
        results = searcher.search("JWT authentication token verification")

        names = [r["name"] for r in results]
        # Both functions should appear (create_access_token by name, get_current_user by source)
        assert "create_access_token" in names
        assert "get_current_user" in names

    def test_source_boosted_score_reasonable(self):
        """Source-boosted results should have meaningful scores, not near-zero."""
        graph = _make_mock_graph_with_jwt_functions()
        searcher = CodeSearcher(graph)
        results = searcher.search("jwt decode token")

        for r in results:
            if r["name"] == "get_current_user":
                # Source blending should give a meaningful score
                assert r["score"] > 0.2
                break


class TestDualEngineSearch:
    """Test that dual-engine (embedding + keyword) max-score works correctly."""

    def test_embedding_upgrade_replaces_keyword_score(self):
        """When embedding score > keyword score, the result should be upgraded."""
        graph = _make_mock_graph_with_jwt_functions()
        searcher = CodeSearcher(graph)

        # Mock embedding searcher returning high score for get_current_user
        mock_emb = MagicMock()
        mock_emb._ready = True
        mock_emb.search.return_value = [
            ("get_current_user", "/project/auth/deps.py", "", 0.75),
        ]
        mock_emb._file_ready = False
        searcher._embedding_searcher = mock_emb
        searcher._embedding_init_done = True

        results = searcher.search("JWT authentication token verification")

        # get_current_user should appear with embedding-boosted score
        for r in results:
            if r["name"] == "get_current_user":
                # With embedding score 0.75, should be at least 0.7
                assert r["score"] >= 0.7
                break
        else:
            # Must be found
            assert False, "get_current_user not found in results"

    def test_embedding_adds_new_results(self):
        """Embedding-only hits should be added to results."""
        graph = _make_mock_graph_with_jwt_functions()
        searcher = CodeSearcher(graph)

        # Mock embedding returning a result not in keyword results
        mock_emb = MagicMock()
        mock_emb._ready = True
        mock_emb.search.return_value = [
            ("mysterious_auth_function", "/project/auth/mystery.py", "", 0.60),
        ]
        mock_emb._file_ready = False
        searcher._embedding_searcher = mock_emb
        searcher._embedding_init_done = True

        results = searcher.search("JWT authentication", max_results=15)

        names = [r["name"] for r in results]
        assert "mysterious_auth_function" in names

    def test_embedding_no_score_cap(self):
        """Embedding scores should NOT be capped at 0.75 anymore."""
        graph = _make_mock_graph_with_jwt_functions()
        searcher = CodeSearcher(graph)

        # Mock embedding with score > 0.75
        mock_emb = MagicMock()
        mock_emb._ready = True
        mock_emb.search.return_value = [
            ("perfect_match", "/project/auth/exact.py", "", 0.85),
        ]
        mock_emb._file_ready = False
        searcher._embedding_searcher = mock_emb
        searcher._embedding_init_done = True

        results = searcher.search("JWT authentication", max_results=15)

        for r in results:
            if r["name"] == "perfect_match":
                # Should be 0.85, NOT capped at 0.75
                assert r["score"] >= 0.80
                break

    def test_search_files_dual_engine(self):
        """search_files should use file-level embeddings when available."""
        graph = _make_mock_graph_with_jwt_functions()
        searcher = CodeSearcher(graph)

        # Mock embedding searcher with file-level search
        mock_emb = MagicMock()
        mock_emb._ready = True
        mock_emb.search.return_value = [
            ("get_current_user", "/project/auth/deps.py", "", 0.70),
        ]
        mock_emb._file_ready = True
        mock_emb.search_files.return_value = [
            ("/project/auth/deps.py", 0.80),
        ]
        searcher._embedding_searcher = mock_emb
        searcher._embedding_init_done = True

        results = searcher.search_files("JWT authentication", max_results=10)

        files = [r["file"] for r in results]
        assert any("deps.py" in f for f in files)


class TestConfidenceFlag:
    """Test that low_confidence flag is set for weak matches."""

    def test_nonsense_query_gets_low_confidence(self):
        """A completely unrelated query should produce low_confidence results."""
        graph = _make_mock_graph_with_jwt_functions()
        searcher = CodeSearcher(graph)
        # Use truly nonsensical tokens that can't match any name, docstring,
        # path, or source token in the mock data.
        results = searcher.search("zxqvbn wkrpfl ghtmnz")

        if results:
            # All results should have low_confidence if best score < 0.55
            assert results[0].get("low_confidence", False) is True

    def test_good_query_no_low_confidence(self):
        """A strong name match should NOT have low_confidence."""
        graph = _make_mock_graph_with_jwt_functions()
        searcher = CodeSearcher(graph)
        results = searcher.search("create_access_token")

        assert len(results) > 0
        # Exact name match should score high — no low_confidence
        top = results[0]
        assert top["name"] == "create_access_token"
        assert "low_confidence" not in top or top["low_confidence"] is False
