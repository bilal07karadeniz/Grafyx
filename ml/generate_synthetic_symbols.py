"""Generate a large, diverse synthetic all_symbols.json for ML training.

Since graph-sitter is not available for symbol extraction from real repos,
this script generates ~50K realistic symbols covering many coding domains
and patterns. The output matches the format expected by all generate_data.py
scripts (relevance_ranker_v2, caller_disambiguator, source_token_filter,
symbol_importance).

Domains covered: web frameworks, ORMs, auth, caching, task queues, CLI tools,
data processing, ML/AI, file I/O, networking, testing, DevOps, messaging,
payments, search, monitoring, security, serialization, and more.

Usage:
    python ml/generate_synthetic_symbols.py
"""

import json
import random
import textwrap
from pathlib import Path

OUTPUT = Path(__file__).parent / "all_symbols.json"

# ── Domain definitions ───────────────────────────────────────────────
# Each domain: (package_name, files, class_defs, function_defs)
# class_defs: [(name, base_classes, methods, docstring)]
# function_defs: [(name, params, docstring, decorators, calls)]

DOMAINS = []


def _domain(name, files, classes, functions):
    DOMAINS.append((name, files, classes, functions))


# ── Web Framework ────────────────────────────────────────────────────
_domain("web_framework", [
    "app/main.py", "app/routes/auth.py", "app/routes/users.py",
    "app/routes/products.py", "app/routes/orders.py", "app/routes/admin.py",
    "app/middleware/cors.py", "app/middleware/auth.py", "app/middleware/rate_limit.py",
    "app/middleware/logging.py", "app/config.py", "app/__init__.py",
    "app/dependencies.py", "app/exceptions.py", "app/schemas.py",
], [
    ("Application", ["BaseApplication"], ["configure", "run", "shutdown", "add_middleware", "add_route", "get_router"],
     "Main web application class with routing and middleware support."),
    ("Router", ["BaseRouter"], ["add_route", "get", "post", "put", "delete", "include_router", "url_for"],
     "URL router that maps paths to handler functions."),
    ("Request", ["BaseRequest"], ["json", "form", "query_params", "headers", "cookies", "url", "method", "path_params"],
     "HTTP request object wrapping ASGI scope."),
    ("Response", ["BaseResponse"], ["set_cookie", "delete_cookie", "set_header", "json", "html", "redirect", "stream"],
     "HTTP response builder."),
    ("Middleware", ["ABC"], ["process_request", "process_response", "dispatch"],
     "Abstract base for HTTP middleware."),
    ("CORSMiddleware", ["Middleware"], ["process_request", "process_response", "is_allowed_origin"],
     "Cross-Origin Resource Sharing middleware."),
    ("AuthMiddleware", ["Middleware"], ["process_request", "authenticate", "get_token"],
     "Authentication middleware for JWT/OAuth."),
    ("RateLimiter", ["Middleware"], ["process_request", "check_rate", "increment_counter", "get_client_id"],
     "Rate limiting middleware using token bucket."),
    ("WebSocket", [], ["accept", "send", "receive", "close", "send_json", "receive_json"],
     "WebSocket connection handler."),
    ("StaticFiles", [], ["serve", "get_path", "check_exists"],
     "Static file serving handler."),
], [
    ("create_app", ["config"], "Create and configure the web application.", [""], ["Application", "configure", "add_middleware"]),
    ("register_routes", ["app", "router"], "Register all route handlers.", [], ["add_route", "include_router"]),
    ("get_current_user", ["request"], "Extract current user from request.", ["Depends"], ["get_token", "verify_token"]),
    ("login", ["username", "password"], "Authenticate user and return JWT.", ["router.post"], ["authenticate", "create_token"]),
    ("logout", ["request"], "Invalidate current session.", ["router.post"], ["revoke_token", "clear_session"]),
    ("register_user", ["user_data"], "Register a new user account.", ["router.post"], ["validate", "create_user", "send_email"]),
    ("get_users", ["skip", "limit"], "List all users with pagination.", ["router.get"], ["query", "paginate"]),
    ("get_user_by_id", ["user_id"], "Retrieve a single user by ID.", ["router.get"], ["query", "get_or_404"]),
    ("update_user", ["user_id", "user_data"], "Update user profile.", ["router.put"], ["validate", "update", "commit"]),
    ("delete_user", ["user_id"], "Delete a user account.", ["router.delete"], ["delete", "commit"]),
    ("create_product", ["product_data"], "Create a new product.", ["router.post"], ["validate", "create", "commit"]),
    ("list_products", ["category", "page"], "List products with filtering.", ["router.get"], ["query", "filter", "paginate"]),
    ("process_order", ["order_data"], "Process a new order.", ["router.post"], ["validate", "create_order", "process_payment"]),
    ("get_order_status", ["order_id"], "Get current order status.", ["router.get"], ["query", "serialize"]),
    ("health_check", [], "Application health check endpoint.", ["router.get"], []),
    ("startup_event", [], "Run on application startup.", ["app.on_event"], ["init_db", "init_cache"]),
    ("shutdown_event", [], "Run on application shutdown.", ["app.on_event"], ["close_db", "close_cache"]),
    ("error_handler", ["request", "exc"], "Global error handler.", [], ["log_error", "json_response"]),
    ("validate_request", ["schema", "data"], "Validate request data against schema.", [], ["validate", "raise_validation_error"]),
    ("serialize_response", ["data", "schema"], "Serialize response data.", [], ["dump", "jsonify"]),
])

# ── Database / ORM ───────────────────────────────────────────────────
_domain("database", [
    "db/engine.py", "db/session.py", "db/models/user.py", "db/models/product.py",
    "db/models/order.py", "db/models/base.py", "db/migrations/001_initial.py",
    "db/repositories/user_repo.py", "db/repositories/product_repo.py",
    "db/repositories/base.py", "db/__init__.py", "db/utils.py",
], [
    ("DatabaseEngine", [], ["connect", "disconnect", "execute", "fetch_one", "fetch_all", "begin_transaction", "commit", "rollback"],
     "Database connection engine with connection pooling."),
    ("Session", [], ["query", "add", "delete", "commit", "rollback", "flush", "close", "refresh", "expire", "merge"],
     "Database session for unit-of-work pattern."),
    ("BaseModel", ["DeclarativeBase"], ["save", "delete", "update", "to_dict", "from_dict"],
     "Base model class for all ORM models."),
    ("UserModel", ["BaseModel"], ["check_password", "set_password", "generate_token", "get_permissions"],
     "User database model with authentication."),
    ("ProductModel", ["BaseModel"], ["get_price", "apply_discount", "check_stock", "reserve_stock"],
     "Product model with inventory management."),
    ("OrderModel", ["BaseModel"], ["calculate_total", "apply_coupon", "update_status", "get_items"],
     "Order model with line items."),
    ("BaseRepository", ["ABC"], ["get_by_id", "get_all", "create", "update", "delete", "find_by", "count", "exists"],
     "Abstract repository pattern base class."),
    ("UserRepository", ["BaseRepository"], ["get_by_email", "get_by_username", "search_users", "get_active_users"],
     "User-specific repository with custom queries."),
    ("ConnectionPool", [], ["acquire", "release", "close_all", "get_stats", "resize"],
     "Database connection pool manager."),
    ("QueryBuilder", [], ["select", "where", "join", "order_by", "limit", "offset", "group_by", "having", "build"],
     "Fluent SQL query builder."),
    ("Migration", ["BaseMigration"], ["upgrade", "downgrade", "get_revision"],
     "Database migration with up/down operations."),
    ("Transaction", [], ["begin", "commit", "rollback", "savepoint", "release_savepoint"],
     "Transaction context manager."),
], [
    ("get_engine", ["database_url"], "Create database engine from URL.", [], ["create_engine", "configure"]),
    ("get_session", [], "Get current database session.", ["contextmanager"], ["Session", "close"]),
    ("init_db", ["engine"], "Initialize database schema.", [], ["create_all", "run_migrations"]),
    ("run_migrations", ["engine", "target"], "Run pending database migrations.", [], ["get_pending", "upgrade"]),
    ("create_tables", ["engine"], "Create all database tables.", [], ["create_all"]),
    ("drop_tables", ["engine"], "Drop all database tables.", [], ["drop_all"]),
    ("seed_data", ["session"], "Seed initial data into database.", [], ["add", "commit"]),
    ("execute_raw_query", ["session", "sql", "params"], "Execute raw SQL query safely.", [], ["execute", "fetchall"]),
    ("paginate_query", ["query", "page", "per_page"], "Apply pagination to a query.", [], ["offset", "limit", "all"]),
    ("apply_filters", ["query", "filters"], "Apply dynamic filters to query.", [], ["filter", "where"]),
    ("bulk_insert", ["session", "model_class", "records"], "Bulk insert multiple records.", [], ["add_all", "commit"]),
    ("soft_delete", ["session", "record"], "Soft delete a record (set deleted_at).", [], ["update", "commit"]),
])

# ── Auth / Security ──────────────────────────────────────────────────
_domain("auth", [
    "auth/jwt.py", "auth/oauth.py", "auth/permissions.py", "auth/passwords.py",
    "auth/session.py", "auth/__init__.py", "auth/providers/google.py",
    "auth/providers/github.py", "auth/mfa.py", "auth/crypto.py",
], [
    ("JWTManager", [], ["create_token", "verify_token", "refresh_token", "decode_token", "revoke_token", "get_claims"],
     "JWT token management with refresh flow."),
    ("OAuthProvider", ["ABC"], ["authorize_url", "exchange_code", "get_user_info", "refresh_access_token"],
     "Abstract OAuth2 provider."),
    ("GoogleOAuth", ["OAuthProvider"], ["authorize_url", "exchange_code", "get_user_info"],
     "Google OAuth2 implementation."),
    ("PermissionManager", [], ["check_permission", "grant_permission", "revoke_permission", "get_roles", "has_role"],
     "Role-based access control manager."),
    ("PasswordHasher", [], ["hash_password", "verify_password", "needs_rehash", "generate_salt"],
     "Secure password hashing with bcrypt/argon2."),
    ("SessionManager", [], ["create_session", "get_session", "delete_session", "extend_session", "list_sessions"],
     "Server-side session management."),
    ("MFAManager", [], ["generate_secret", "verify_totp", "generate_backup_codes", "enable_mfa", "disable_mfa"],
     "Multi-factor authentication with TOTP."),
    ("Encryptor", [], ["encrypt", "decrypt", "generate_key", "rotate_key"],
     "Symmetric encryption for sensitive data."),
], [
    ("create_access_token", ["user_id", "scopes"], "Create a JWT access token.", [], ["encode", "set_expiry"]),
    ("verify_access_token", ["token"], "Verify and decode a JWT token.", [], ["decode", "check_expiry"]),
    ("hash_password", ["password"], "Hash a password using bcrypt.", [], ["generate_salt", "hashpw"]),
    ("verify_password", ["password", "hashed"], "Verify a password against hash.", [], ["checkpw"]),
    ("check_permissions", ["user", "resource", "action"], "Check if user has permission.", [], ["get_roles", "check_role_permission"]),
    ("require_auth", ["func"], "Decorator requiring authentication.", ["wraps"], ["verify_token", "get_current_user"]),
    ("require_role", ["role"], "Decorator requiring a specific role.", ["wraps"], ["get_current_user", "has_role"]),
    ("generate_api_key", [], "Generate a secure API key.", [], ["secrets.token_urlsafe"]),
    ("validate_api_key", ["key"], "Validate an API key.", [], ["query", "compare_digest"]),
    ("encrypt_field", ["value", "key"], "Encrypt a database field value.", [], ["encrypt", "b64encode"]),
    ("decrypt_field", ["encrypted", "key"], "Decrypt a database field value.", [], ["b64decode", "decrypt"]),
    ("sanitize_input", ["value"], "Sanitize user input to prevent XSS.", [], ["escape", "strip"]),
])

# ── Caching ──────────────────────────────────────────────────────────
_domain("caching", [
    "cache/backend.py", "cache/redis_backend.py", "cache/memory_backend.py",
    "cache/decorators.py", "cache/__init__.py", "cache/serializer.py",
    "cache/invalidation.py",
], [
    ("CacheBackend", ["ABC"], ["get", "set", "delete", "exists", "clear", "get_many", "set_many", "ttl"],
     "Abstract cache backend interface."),
    ("RedisCache", ["CacheBackend"], ["get", "set", "delete", "exists", "clear", "pipeline", "publish", "subscribe"],
     "Redis-backed cache implementation."),
    ("MemoryCache", ["CacheBackend"], ["get", "set", "delete", "exists", "clear", "get_stats"],
     "In-memory LRU cache implementation."),
    ("CacheSerializer", [], ["serialize", "deserialize", "compress", "decompress"],
     "Cache value serializer with optional compression."),
    ("CacheInvalidator", [], ["invalidate_pattern", "invalidate_tags", "register_dependency", "cascade_invalidate"],
     "Smart cache invalidation with tag support."),
], [
    ("cached", ["ttl", "key_prefix"], "Decorator for caching function results.", ["wraps"], ["get", "set", "make_key"]),
    ("cache_page", ["ttl"], "Decorator for caching page responses.", ["wraps"], ["get", "set"]),
    ("invalidate_cache", ["pattern"], "Invalidate cache entries matching pattern.", [], ["delete", "scan"]),
    ("warm_cache", ["keys"], "Pre-populate cache with frequently accessed data.", [], ["get", "set", "query"]),
    ("get_cache_stats", [], "Get cache hit/miss statistics.", [], ["info", "dbsize"]),
    ("make_cache_key", ["prefix", "args", "kwargs"], "Generate a deterministic cache key.", [], ["hashlib.md5", "json.dumps"]),
])

# ── Task Queue ───────────────────────────────────────────────────────
_domain("task_queue", [
    "tasks/worker.py", "tasks/scheduler.py", "tasks/handlers/email.py",
    "tasks/handlers/export.py", "tasks/handlers/notification.py",
    "tasks/__init__.py", "tasks/retry.py", "tasks/monitoring.py",
], [
    ("TaskWorker", [], ["start", "stop", "process_task", "acknowledge", "reject", "get_status"],
     "Background task worker process."),
    ("TaskScheduler", [], ["schedule", "cancel", "reschedule", "get_pending", "run_due"],
     "Cron-like task scheduler."),
    ("TaskQueue", [], ["enqueue", "dequeue", "peek", "size", "purge", "get_stats"],
     "Message queue for async task processing."),
    ("RetryPolicy", [], ["should_retry", "get_delay", "increment_attempts", "is_exhausted"],
     "Exponential backoff retry policy."),
    ("TaskResult", [], ["get", "ready", "failed", "revoke", "forget"],
     "Handle to an async task result."),
], [
    ("send_email_task", ["to", "subject", "body"], "Send email asynchronously.", ["task"], ["send_email", "log"]),
    ("process_export_task", ["export_id", "format"], "Export data to file asynchronously.", ["task"], ["query", "serialize", "upload"]),
    ("send_notification", ["user_id", "message"], "Send push notification.", ["task"], ["get_device_tokens", "push"]),
    ("cleanup_expired", [], "Clean up expired records.", ["periodic_task"], ["query", "delete", "commit"]),
    ("retry_failed_tasks", [], "Retry tasks that failed.", [], ["get_failed", "requeue"]),
    ("aggregate_metrics", [], "Aggregate task performance metrics.", ["periodic_task"], ["query", "compute", "store"]),
    ("schedule_task", ["task_name", "args", "eta"], "Schedule a task for later execution.", [], ["enqueue", "set_eta"]),
    ("cancel_task", ["task_id"], "Cancel a pending task.", [], ["get_task", "revoke"]),
])

# ── CLI Tools ────────────────────────────────────────────────────────
_domain("cli", [
    "cli/main.py", "cli/commands/init.py", "cli/commands/build.py",
    "cli/commands/deploy.py", "cli/commands/test.py", "cli/commands/lint.py",
    "cli/utils.py", "cli/__init__.py", "cli/config.py",
], [
    ("CLI", ["BaseCommand"], ["run", "parse_args", "execute", "show_help", "show_version"],
     "Main CLI application entry point."),
    ("InitCommand", ["BaseCommand"], ["execute", "validate_args", "create_project", "init_config"],
     "Initialize a new project."),
    ("BuildCommand", ["BaseCommand"], ["execute", "compile", "bundle", "optimize", "report"],
     "Build project artifacts."),
    ("DeployCommand", ["BaseCommand"], ["execute", "validate_environment", "push", "rollback", "verify"],
     "Deploy to production/staging."),
    ("ProgressBar", [], ["update", "finish", "set_description", "set_postfix"],
     "Terminal progress bar display."),
], [
    ("main", [], "CLI entry point.", [], ["CLI", "run", "parse_args"]),
    ("init_project", ["name", "template"], "Initialize a new project from template.", [], ["create_dir", "copy_template", "init_git"]),
    ("build_project", ["config"], "Build the project.", [], ["compile", "bundle", "write_manifest"]),
    ("deploy", ["target", "version"], "Deploy to target environment.", [], ["validate", "push", "verify"]),
    ("run_tests", ["pattern", "verbose"], "Run test suite.", [], ["discover", "run", "report"]),
    ("lint_code", ["paths", "fix"], "Lint source code.", [], ["parse", "check_rules", "report"]),
    ("format_code", ["paths"], "Format source code.", [], ["parse", "reformat", "write"]),
    ("generate_docs", ["output_dir"], "Generate documentation.", [], ["parse_modules", "render", "write"]),
    ("check_dependencies", [], "Check for outdated dependencies.", [], ["read_lockfile", "fetch_latest", "compare"]),
    ("create_migration", ["name"], "Create a new migration file.", [], ["get_current_revision", "generate_template"]),
    ("print_table", ["headers", "rows"], "Print a formatted table to terminal.", [], ["format_row", "print"]),
    ("confirm_action", ["message"], "Ask user for confirmation.", [], ["input", "lower"]),
    ("load_config", ["path"], "Load configuration from file.", [], ["read", "parse", "validate"]),
    ("save_config", ["config", "path"], "Save configuration to file.", [], ["serialize", "write"]),
])

# ── Data Processing ──────────────────────────────────────────────────
_domain("data_processing", [
    "data/pipeline.py", "data/transformers.py", "data/loaders.py",
    "data/validators.py", "data/exporters.py", "data/__init__.py",
    "data/schemas.py", "data/aggregations.py",
], [
    ("DataPipeline", [], ["add_step", "run", "validate", "get_stats", "reset"],
     "Composable data processing pipeline."),
    ("CSVLoader", ["BaseLoader"], ["load", "stream", "get_schema", "validate_headers"],
     "CSV file data loader."),
    ("JSONLoader", ["BaseLoader"], ["load", "stream", "validate_schema"],
     "JSON file data loader."),
    ("DataTransformer", ["ABC"], ["transform", "inverse_transform", "fit", "fit_transform"],
     "Abstract data transformer."),
    ("Normalizer", ["DataTransformer"], ["transform", "inverse_transform", "fit"],
     "Numeric value normalizer."),
    ("DataValidator", [], ["validate", "get_errors", "add_rule", "validate_field"],
     "Schema-based data validator."),
    ("DataExporter", ["ABC"], ["export", "stream_export", "get_format"],
     "Abstract data exporter."),
    ("AggregationEngine", [], ["group_by", "sum", "avg", "count", "min", "max", "percentile"],
     "Data aggregation engine."),
], [
    ("load_csv", ["path", "delimiter"], "Load data from CSV file.", [], ["open", "reader", "parse_row"]),
    ("load_json", ["path"], "Load data from JSON file.", [], ["open", "json.load"]),
    ("transform_data", ["data", "transformers"], "Apply transformations to data.", [], ["transform", "validate"]),
    ("validate_schema", ["data", "schema"], "Validate data against schema.", [], ["check_type", "check_required", "check_format"]),
    ("export_to_csv", ["data", "path"], "Export data to CSV file.", [], ["open", "writer", "writerow"]),
    ("export_to_parquet", ["data", "path"], "Export data to Parquet format.", [], ["to_parquet"]),
    ("merge_datasets", ["datasets", "key"], "Merge multiple datasets on key.", [], ["join", "deduplicate"]),
    ("filter_records", ["data", "predicate"], "Filter records by predicate.", [], ["filter"]),
    ("compute_statistics", ["data", "columns"], "Compute descriptive statistics.", [], ["mean", "std", "median", "quantile"]),
    ("detect_outliers", ["data", "column", "threshold"], "Detect outliers using IQR method.", [], ["quantile", "filter"]),
    ("normalize_column", ["data", "column"], "Normalize a numeric column to [0,1].", [], ["min", "max", "apply"]),
    ("encode_categorical", ["data", "column"], "One-hot encode a categorical column.", [], ["get_dummies", "concat"]),
    ("parse_date", ["value", "format"], "Parse a date string.", [], ["strptime"]),
    ("clean_text", ["text"], "Clean and normalize text data.", [], ["lower", "strip", "re.sub"]),
    ("split_dataset", ["data", "ratio"], "Split dataset into train/test.", [], ["shuffle", "slice"]),
])

# ── ML / AI ──────────────────────────────────────────────────────────
_domain("ml", [
    "ml/model.py", "ml/training.py", "ml/evaluation.py", "ml/features.py",
    "ml/preprocessing.py", "ml/__init__.py", "ml/inference.py",
    "ml/hyperparameters.py", "ml/registry.py",
], [
    ("BaseModel", ["ABC"], ["fit", "predict", "evaluate", "save", "load", "get_params"],
     "Abstract machine learning model."),
    ("NeuralNetwork", ["BaseModel"], ["forward", "backward", "fit", "predict", "compile"],
     "Neural network model."),
    ("FeatureExtractor", [], ["extract", "fit", "transform", "get_feature_names"],
     "Feature extraction pipeline."),
    ("ModelTrainer", [], ["train", "validate", "early_stopping", "save_checkpoint", "load_checkpoint"],
     "Model training orchestrator."),
    ("ModelEvaluator", [], ["evaluate", "confusion_matrix", "classification_report", "roc_curve"],
     "Model evaluation utilities."),
    ("HyperparameterTuner", [], ["search", "get_best", "get_results", "cross_validate"],
     "Hyperparameter optimization."),
    ("ModelRegistry", [], ["register", "get_model", "list_models", "delete_model", "get_latest"],
     "Model versioning and registry."),
    ("DataAugmenter", [], ["augment", "random_crop", "random_flip", "add_noise"],
     "Data augmentation for training."),
], [
    ("train_model", ["model", "data", "epochs"], "Train a machine learning model.", [], ["fit", "validate", "save_checkpoint"]),
    ("evaluate_model", ["model", "test_data"], "Evaluate model on test data.", [], ["predict", "compute_metrics"]),
    ("extract_features", ["data", "config"], "Extract features from raw data.", [], ["transform", "normalize"]),
    ("preprocess_data", ["data"], "Preprocess raw data for training.", [], ["clean", "normalize", "split"]),
    ("grid_search", ["model", "param_grid", "data"], "Grid search for hyperparameters.", [], ["cross_validate", "get_best"]),
    ("predict_batch", ["model", "inputs"], "Run batch prediction.", [], ["preprocess", "forward", "postprocess"]),
    ("save_model", ["model", "path"], "Save model weights and config.", [], ["state_dict", "json.dump", "savez"]),
    ("load_model", ["path"], "Load model from disk.", [], ["json.load", "load", "from_pretrained"]),
    ("compute_metrics", ["y_true", "y_pred"], "Compute classification metrics.", [], ["accuracy_score", "f1_score", "precision_score"]),
    ("cross_validate", ["model", "data", "k_folds"], "K-fold cross validation.", [], ["split", "fit", "evaluate"]),
    ("learning_rate_schedule", ["optimizer", "epoch"], "Adjust learning rate.", [], ["step", "get_lr"]),
    ("compute_loss", ["predictions", "targets"], "Compute loss function.", [], ["binary_cross_entropy", "mean"]),
])

# ── Networking / HTTP Client ─────────────────────────────────────────
_domain("networking", [
    "http/client.py", "http/retry.py", "http/auth.py", "http/__init__.py",
    "http/interceptors.py", "http/models.py",
], [
    ("HTTPClient", [], ["get", "post", "put", "delete", "patch", "head", "request", "close"],
     "HTTP client with connection pooling."),
    ("AsyncHTTPClient", ["HTTPClient"], ["get", "post", "put", "delete", "request", "close"],
     "Async HTTP client."),
    ("RetryHandler", [], ["execute_with_retry", "should_retry", "get_backoff_delay"],
     "HTTP request retry handler."),
    ("RequestInterceptor", ["ABC"], ["before_request", "after_response", "on_error"],
     "Request/response interceptor."),
    ("ConnectionPool", [], ["acquire", "release", "close_all", "get_stats"],
     "HTTP connection pool."),
    ("APIClient", ["HTTPClient"], ["authenticate", "get_resource", "create_resource", "update_resource"],
     "REST API client base class."),
], [
    ("fetch_url", ["url", "headers"], "Fetch content from URL.", [], ["get", "raise_for_status"]),
    ("post_json", ["url", "data"], "POST JSON data to URL.", [], ["post", "json"]),
    ("download_file", ["url", "dest"], "Download file from URL.", [], ["get", "stream", "write"]),
    ("upload_file", ["url", "file_path"], "Upload file to URL.", [], ["open", "post", "multipart"]),
    ("create_session", ["base_url", "headers"], "Create a configured HTTP session.", [], ["Session", "mount"]),
    ("parse_response", ["response"], "Parse HTTP response.", [], ["json", "raise_for_status"]),
    ("build_url", ["base", "path", "params"], "Build URL with query parameters.", [], ["urljoin", "urlencode"]),
    ("retry_request", ["func", "max_retries"], "Retry HTTP request with backoff.", [], ["sleep", "execute"]),
])

# ── File I/O ─────────────────────────────────────────────────────────
_domain("file_io", [
    "storage/local.py", "storage/s3.py", "storage/gcs.py", "storage/__init__.py",
    "storage/base.py", "storage/utils.py",
], [
    ("StorageBackend", ["ABC"], ["read", "write", "delete", "exists", "list_files", "get_metadata"],
     "Abstract file storage backend."),
    ("LocalStorage", ["StorageBackend"], ["read", "write", "delete", "exists", "list_files", "move"],
     "Local filesystem storage."),
    ("S3Storage", ["StorageBackend"], ["read", "write", "delete", "exists", "list_files", "generate_presigned_url"],
     "AWS S3 file storage."),
    ("FileWatcher", [], ["watch", "stop", "on_created", "on_modified", "on_deleted"],
     "File system change watcher."),
], [
    ("read_file", ["path"], "Read file contents.", [], ["open", "read"]),
    ("write_file", ["path", "content"], "Write content to file.", [], ["open", "write"]),
    ("read_json", ["path"], "Read and parse JSON file.", [], ["open", "json.load"]),
    ("write_json", ["path", "data"], "Write data as JSON to file.", [], ["open", "json.dump"]),
    ("read_yaml", ["path"], "Read and parse YAML file.", [], ["open", "yaml.safe_load"]),
    ("copy_file", ["src", "dest"], "Copy file from source to destination.", [], ["shutil.copy2"]),
    ("move_file", ["src", "dest"], "Move file from source to destination.", [], ["shutil.move"]),
    ("delete_file", ["path"], "Delete a file.", [], ["os.remove"]),
    ("list_directory", ["path", "pattern"], "List files in directory matching pattern.", [], ["glob", "iterdir"]),
    ("ensure_directory", ["path"], "Create directory if it doesn't exist.", [], ["mkdir", "makedirs"]),
    ("get_file_hash", ["path", "algorithm"], "Compute hash of file contents.", [], ["open", "hashlib.new", "hexdigest"]),
    ("compress_file", ["path", "output"], "Compress file with gzip.", [], ["open", "gzip.open", "copyfileobj"]),
    ("extract_archive", ["path", "dest"], "Extract zip/tar archive.", [], ["zipfile.extractall"]),
    ("get_mime_type", ["path"], "Detect MIME type of file.", [], ["mimetypes.guess_type"]),
    ("tail_file", ["path", "lines"], "Read last N lines of file.", [], ["open", "seek", "readlines"]),
])

# ── Testing ──────────────────────────────────────────────────────────
_domain("testing", [
    "tests/test_auth.py", "tests/test_users.py", "tests/test_database.py",
    "tests/test_api.py", "tests/test_cache.py", "tests/conftest.py",
    "tests/factories.py", "tests/fixtures.py", "tests/helpers.py",
], [
    ("TestClient", [], ["get", "post", "put", "delete", "login", "logout"],
     "Test HTTP client for integration tests."),
    ("UserFactory", ["BaseFactory"], ["create", "create_batch", "build"],
     "Factory for creating test users."),
    ("MockDatabase", [], ["query", "execute", "reset", "seed"],
     "Mock database for unit tests."),
], [
    ("test_login_success", ["client", "user"], "Test successful login.", ["pytest.mark.asyncio"], ["post", "assert_equal"]),
    ("test_login_invalid_password", ["client", "user"], "Test login with wrong password.", ["pytest.mark.asyncio"], ["post", "assert_equal"]),
    ("test_register_user", ["client"], "Test user registration.", ["pytest.mark.asyncio"], ["post", "assert_equal"]),
    ("test_get_users_pagination", ["client", "users"], "Test user list pagination.", [], ["get", "assert_equal"]),
    ("test_create_product", ["client", "admin"], "Test product creation.", [], ["post", "assert_equal"]),
    ("test_order_workflow", ["client", "user", "product"], "Test full order workflow.", [], ["post", "get", "assert_equal"]),
    ("test_cache_hit", ["cache"], "Test cache hit scenario.", [], ["set", "get", "assert_equal"]),
    ("test_cache_expiry", ["cache"], "Test cache TTL expiry.", [], ["set", "sleep", "get", "assert_none"]),
    ("test_database_rollback", ["session"], "Test transaction rollback.", [], ["begin", "add", "rollback", "query"]),
    ("test_permission_denied", ["client", "user"], "Test access denied for unauthorized user.", [], ["get", "assert_equal"]),
    ("test_rate_limiting", ["client"], "Test rate limiting middleware.", [], ["get", "assert_equal"]),
    ("test_input_validation", ["client"], "Test request validation.", [], ["post", "assert_equal"]),
    ("create_test_user", ["session"], "Create a test user fixture.", ["pytest.fixture"], ["UserFactory", "create"]),
    ("create_test_product", ["session"], "Create a test product fixture.", ["pytest.fixture"], ["ProductFactory", "create"]),
    ("mock_redis", [], "Mock Redis connection fixture.", ["pytest.fixture"], ["MagicMock"]),
    ("assert_json_response", ["response", "expected"], "Assert JSON response body.", [], ["json", "assert_equal"]),
])

# ── Monitoring / Logging ─────────────────────────────────────────────
_domain("monitoring", [
    "monitoring/metrics.py", "monitoring/tracing.py", "monitoring/health.py",
    "monitoring/alerts.py", "monitoring/__init__.py", "monitoring/logging.py",
], [
    ("MetricsCollector", [], ["counter", "gauge", "histogram", "summary", "export"],
     "Prometheus-style metrics collector."),
    ("Tracer", [], ["start_span", "end_span", "set_tag", "log_event", "get_trace_id"],
     "Distributed tracing."),
    ("HealthChecker", [], ["check", "register_check", "get_status", "is_healthy"],
     "Application health checker."),
    ("AlertManager", [], ["send_alert", "resolve_alert", "silence", "get_active"],
     "Alert management."),
    ("Logger", [], ["debug", "info", "warning", "error", "critical", "exception"],
     "Structured logger."),
], [
    ("setup_logging", ["level", "format"], "Configure application logging.", [], ["basicConfig", "getLogger"]),
    ("log_request", ["request", "response", "duration"], "Log HTTP request/response.", [], ["info", "format"]),
    ("record_metric", ["name", "value", "tags"], "Record a metric value.", [], ["counter", "labels", "inc"]),
    ("start_trace", ["name"], "Start a new trace span.", [], ["start_span", "set_tag"]),
    ("end_trace", ["span"], "End a trace span.", [], ["end_span"]),
    ("check_health", [], "Run all health checks.", [], ["check", "aggregate"]),
    ("send_alert", ["severity", "message"], "Send an alert notification.", [], ["post", "log"]),
    ("get_system_metrics", [], "Collect system resource metrics.", [], ["cpu_percent", "memory_info", "disk_usage"]),
    ("format_log_entry", ["level", "message", "context"], "Format a structured log entry.", [], ["json.dumps", "timestamp"]),
    ("rotate_logs", ["path", "max_size"], "Rotate log files.", [], ["stat", "rename", "open"]),
])

# ── Messaging / Events ───────────────────────────────────────────────
_domain("messaging", [
    "events/bus.py", "events/handlers.py", "events/models.py",
    "events/__init__.py", "events/middleware.py",
], [
    ("EventBus", [], ["publish", "subscribe", "unsubscribe", "get_handlers", "process"],
     "Publish-subscribe event bus."),
    ("EventHandler", ["ABC"], ["handle", "can_handle", "get_event_types"],
     "Abstract event handler."),
    ("Event", [], ["serialize", "deserialize", "get_type", "get_timestamp"],
     "Base event class."),
    ("MessageQueue", [], ["send", "receive", "acknowledge", "reject", "get_stats"],
     "Message queue adapter."),
    ("EventStore", [], ["append", "get_events", "get_by_aggregate", "replay"],
     "Event sourcing store."),
], [
    ("publish_event", ["event_type", "data"], "Publish an event to the bus.", [], ["create_event", "publish"]),
    ("handle_user_created", ["event"], "Handle user creation event.", [], ["send_welcome_email", "setup_profile"]),
    ("handle_order_placed", ["event"], "Handle order placement event.", [], ["process_payment", "send_confirmation"]),
    ("handle_payment_completed", ["event"], "Handle payment completion.", [], ["update_order_status", "send_receipt"]),
    ("replay_events", ["aggregate_id", "from_version"], "Replay events for an aggregate.", [], ["get_events", "apply"]),
    ("create_event", ["event_type", "data", "metadata"], "Create a new event instance.", [], ["Event", "serialize"]),
    ("dispatch_event", ["event"], "Dispatch event to registered handlers.", [], ["get_handlers", "handle"]),
    ("register_handler", ["event_type", "handler"], "Register an event handler.", [], ["subscribe"]),
])

# ── Payment Processing ───────────────────────────────────────────────
_domain("payments", [
    "payments/processor.py", "payments/stripe.py", "payments/models.py",
    "payments/__init__.py", "payments/webhooks.py", "payments/refunds.py",
], [
    ("PaymentProcessor", ["ABC"], ["charge", "refund", "capture", "void", "get_transaction"],
     "Abstract payment processor."),
    ("StripePayment", ["PaymentProcessor"], ["charge", "refund", "create_customer", "create_subscription"],
     "Stripe payment integration."),
    ("PaymentIntent", [], ["create", "confirm", "cancel", "get_status"],
     "Payment intent model."),
    ("Subscription", [], ["create", "cancel", "update", "pause", "resume", "get_invoices"],
     "Recurring subscription model."),
], [
    ("process_payment", ["amount", "currency", "source"], "Process a payment charge.", [], ["create_intent", "confirm"]),
    ("refund_payment", ["transaction_id", "amount"], "Issue a payment refund.", [], ["get_transaction", "refund"]),
    ("create_subscription", ["customer_id", "plan_id"], "Create a recurring subscription.", [], ["create_customer", "subscribe"]),
    ("handle_webhook", ["event_type", "data"], "Handle payment webhook event.", [], ["verify_signature", "dispatch"]),
    ("calculate_tax", ["amount", "country", "state"], "Calculate tax for an amount.", [], ["get_rate", "multiply"]),
    ("apply_coupon", ["order", "coupon_code"], "Apply a discount coupon.", [], ["validate_coupon", "calculate_discount"]),
    ("generate_invoice", ["order"], "Generate an invoice for an order.", [], ["get_items", "calculate_total", "render"]),
    ("verify_payment_signature", ["payload", "signature"], "Verify webhook signature.", [], ["hmac.compare_digest"]),
])

# ── Search / Indexing ────────────────────────────────────────────────
_domain("search", [
    "search/indexer.py", "search/searcher.py", "search/tokenizer.py",
    "search/__init__.py", "search/ranking.py", "search/filters.py",
], [
    ("SearchIndexer", [], ["index", "delete", "update", "bulk_index", "optimize"],
     "Full-text search indexer."),
    ("SearchEngine", [], ["search", "suggest", "facet", "aggregate", "scroll"],
     "Search engine with ranking."),
    ("Tokenizer", [], ["tokenize", "stem", "remove_stopwords", "normalize"],
     "Text tokenizer for search."),
    ("RankingFunction", ["ABC"], ["score", "boost", "decay"],
     "Abstract ranking/scoring function."),
    ("SearchFilter", [], ["apply", "combine", "negate"],
     "Search result filter."),
], [
    ("index_document", ["doc_id", "content", "metadata"], "Index a document for search.", [], ["tokenize", "store", "update_index"]),
    ("search_documents", ["query", "filters", "limit"], "Search indexed documents.", [], ["parse_query", "execute", "rank"]),
    ("build_search_index", ["documents"], "Build search index from documents.", [], ["tokenize", "store_all"]),
    ("suggest_completions", ["prefix", "limit"], "Auto-complete suggestions.", [], ["prefix_search", "rank"]),
    ("compute_relevance_score", ["query", "document"], "Compute relevance score.", [], ["tfidf", "bm25"]),
    ("highlight_matches", ["query", "text"], "Highlight matching terms in text.", [], ["tokenize", "wrap_matches"]),
    ("faceted_search", ["query", "facets"], "Search with faceted navigation.", [], ["search", "aggregate"]),
    ("reindex_all", [], "Rebuild the entire search index.", [], ["get_all_documents", "bulk_index"]),
    ("tokenize_query", ["query"], "Tokenize and normalize a search query.", [], ["split", "stem", "remove_stopwords"]),
])

# ── Serialization ────────────────────────────────────────────────────
_domain("serialization", [
    "serializers/base.py", "serializers/json.py", "serializers/xml.py",
    "serializers/__init__.py", "serializers/schema.py",
], [
    ("Serializer", ["ABC"], ["serialize", "deserialize", "validate"],
     "Abstract serializer."),
    ("JSONSerializer", ["Serializer"], ["serialize", "deserialize", "validate"],
     "JSON serialization."),
    ("XMLSerializer", ["Serializer"], ["serialize", "deserialize", "validate"],
     "XML serialization."),
    ("SchemaValidator", [], ["validate", "add_field", "get_errors", "is_valid"],
     "Schema validation for serialization."),
    ("ModelSerializer", ["Serializer"], ["serialize", "deserialize", "get_fields", "validate"],
     "ORM model serializer."),
], [
    ("serialize_model", ["instance", "fields"], "Serialize a model instance.", [], ["get_fields", "to_dict"]),
    ("deserialize_model", ["data", "model_class"], "Deserialize data into model.", [], ["validate", "from_dict"]),
    ("validate_data", ["data", "schema"], "Validate data against schema.", [], ["check_required", "check_types"]),
    ("serialize_list", ["instances", "fields"], "Serialize a list of instances.", [], ["serialize_model"]),
    ("to_json", ["data"], "Convert data to JSON string.", [], ["json.dumps"]),
    ("from_json", ["json_str"], "Parse JSON string to data.", [], ["json.loads"]),
    ("to_xml", ["data", "root_tag"], "Convert data to XML string.", [], ["Element", "tostring"]),
    ("deep_merge", ["base", "override"], "Deep merge two dictionaries.", [], ["isinstance", "update"]),
])

# ── Configuration ────────────────────────────────────────────────────
_domain("config", [
    "config/settings.py", "config/loader.py", "config/validators.py",
    "config/__init__.py", "config/env.py",
], [
    ("Settings", ["BaseSettings"], ["get", "set", "validate", "reload", "to_dict"],
     "Application settings manager."),
    ("ConfigLoader", [], ["load_file", "load_env", "merge", "validate"],
     "Configuration loader from multiple sources."),
    ("EnvParser", [], ["parse", "get_bool", "get_int", "get_list", "get_secret"],
     "Environment variable parser."),
], [
    ("load_settings", ["env"], "Load settings for environment.", [], ["load_file", "load_env", "validate"]),
    ("get_setting", ["key", "default"], "Get a configuration value.", [], ["get", "environ"]),
    ("validate_config", ["config"], "Validate configuration values.", [], ["check_required", "check_types"]),
    ("merge_configs", ["base", "override"], "Merge configuration dicts.", [], ["deep_merge"]),
    ("get_database_url", ["env"], "Get database URL for environment.", [], ["get_setting", "format"]),
    ("get_redis_url", ["env"], "Get Redis URL for environment.", [], ["get_setting", "format"]),
    ("parse_env_file", ["path"], "Parse .env file into dict.", [], ["open", "split", "strip"]),
])

# ── Scheduling / Cron ────────────────────────────────────────────────
_domain("scheduling", [
    "scheduler/core.py", "scheduler/jobs.py", "scheduler/triggers.py",
    "scheduler/__init__.py",
], [
    ("Scheduler", [], ["add_job", "remove_job", "start", "stop", "pause", "resume", "get_jobs"],
     "Job scheduler with cron and interval triggers."),
    ("Job", [], ["execute", "pause", "resume", "reschedule", "get_next_run"],
     "Scheduled job."),
    ("CronTrigger", [], ["get_next_fire_time", "matches"],
     "Cron expression trigger."),
    ("IntervalTrigger", [], ["get_next_fire_time"],
     "Fixed interval trigger."),
], [
    ("schedule_job", ["func", "trigger", "args"], "Schedule a job.", [], ["add_job"]),
    ("parse_cron_expression", ["expression"], "Parse a cron expression.", [], ["split", "validate"]),
    ("run_scheduled_jobs", [], "Run all due jobs.", [], ["get_due_jobs", "execute"]),
    ("cancel_job", ["job_id"], "Cancel a scheduled job.", [], ["remove_job"]),
])

# ── Validation ───────────────────────────────────────────────────────
_domain("validation", [
    "validators/core.py", "validators/types.py", "validators/strings.py",
    "validators/__init__.py",
], [
    ("Validator", [], ["validate", "add_rule", "get_errors", "is_valid"],
     "Composable data validator."),
    ("StringValidator", ["Validator"], ["min_length", "max_length", "matches", "email", "url"],
     "String validation rules."),
    ("NumberValidator", ["Validator"], ["min_value", "max_value", "positive", "integer"],
     "Numeric validation rules."),
    ("SchemaValidator", ["Validator"], ["required", "optional", "nested", "list_of"],
     "Nested schema validation."),
], [
    ("validate_email", ["email"], "Validate email address format.", [], ["re.match"]),
    ("validate_url", ["url"], "Validate URL format.", [], ["urlparse", "check_scheme"]),
    ("validate_phone", ["phone"], "Validate phone number format.", [], ["re.match", "strip"]),
    ("validate_password_strength", ["password"], "Check password meets requirements.", [], ["len", "re.search"]),
    ("validate_date_range", ["start", "end"], "Validate a date range.", [], ["parse", "compare"]),
    ("validate_json_schema", ["data", "schema"], "Validate against JSON schema.", [], ["jsonschema.validate"]),
    ("sanitize_html", ["html"], "Sanitize HTML removing dangerous tags.", [], ["bleach.clean"]),
    ("validate_uuid", ["value"], "Validate UUID format.", [], ["UUID"]),
])

# ── Utility / Helpers ────────────────────────────────────────────────
_domain("utils", [
    "utils/strings.py", "utils/dates.py", "utils/collections.py",
    "utils/math.py", "utils/__init__.py", "utils/decorators.py",
    "utils/retry.py",
], [], [
    ("slugify", ["text"], "Convert text to URL-safe slug.", [], ["re.sub", "lower", "strip"]),
    ("truncate", ["text", "max_length"], "Truncate text with ellipsis.", [], ["len"]),
    ("camel_to_snake", ["name"], "Convert camelCase to snake_case.", [], ["re.sub", "lower"]),
    ("snake_to_camel", ["name"], "Convert snake_case to camelCase.", [], ["split", "capitalize", "join"]),
    ("pluralize", ["word"], "Pluralize an English word.", [], ["endswith"]),
    ("format_bytes", ["size"], "Format bytes to human readable string.", [], ["log", "format"]),
    ("format_duration", ["seconds"], "Format seconds to human readable duration.", [], ["divmod", "format"]),
    ("parse_iso_date", ["date_str"], "Parse ISO 8601 date string.", [], ["fromisoformat"]),
    ("format_iso_date", ["dt"], "Format datetime as ISO 8601.", [], ["isoformat"]),
    ("now_utc", [], "Get current UTC datetime.", [], ["datetime.utcnow"]),
    ("chunk_list", ["lst", "size"], "Split list into chunks of given size.", [], ["range", "slice"]),
    ("flatten", ["nested_list"], "Flatten nested list.", [], ["isinstance", "extend"]),
    ("deep_get", ["obj", "path", "default"], "Get nested dict value by dot path.", [], ["split", "get"]),
    ("deep_set", ["obj", "path", "value"], "Set nested dict value by dot path.", [], ["split", "setdefault"]),
    ("memoize", ["func"], "Memoization decorator.", ["wraps"], ["lru_cache"]),
    ("retry", ["max_retries", "delay"], "Retry decorator with backoff.", ["wraps"], ["sleep", "log"]),
    ("timer", ["func"], "Timing decorator.", ["wraps"], ["time.time", "log"]),
    ("deprecated", ["message"], "Mark function as deprecated.", ["wraps"], ["warnings.warn"]),
    ("singleton", ["cls"], "Singleton decorator.", [], ["getattr", "setattr"]),
    ("debounce", ["wait"], "Debounce function calls.", ["wraps"], ["Timer", "cancel"]),
    ("throttle", ["rate"], "Throttle function calls.", ["wraps"], ["time.time"]),
    ("safe_divide", ["a", "b", "default"], "Divide with zero safety.", [], []),
    ("clamp", ["value", "min_val", "max_val"], "Clamp value to range.", [], ["min", "max"]),
    ("levenshtein_distance", ["s1", "s2"], "Compute edit distance between strings.", [], ["min"]),
    ("generate_uuid", [], "Generate a UUID4 string.", [], ["uuid4", "str"]),
    ("hash_string", ["text", "algorithm"], "Hash a string.", [], ["hashlib.new", "hexdigest"]),
    ("base64_encode", ["data"], "Base64 encode data.", [], ["b64encode", "decode"]),
    ("base64_decode", ["data"], "Base64 decode data.", [], ["b64decode"]),
])


# ── Symbol generation ────────────────────────────────────────────────


def _generate_source_for_function(name, params, docstring, calls, class_name=None):
    """Generate realistic source code for a function."""
    indent = "    " if class_name else ""
    param_list = ", ".join(params) if params else ""
    if class_name and param_list:
        param_list = "self, " + param_list
    elif class_name:
        param_list = "self"

    lines = [f"{indent}def {name}({param_list}):"]
    if docstring:
        lines.append(f'{indent}    """{docstring}"""')

    # Add some realistic body
    if calls:
        for call in calls[:3]:
            if "." in call:
                lines.append(f"{indent}    result = {call}()")
            else:
                lines.append(f"{indent}    result = self.{call}() " if class_name else f"{indent}    result = {call}()")
        lines.append(f"{indent}    return result")
    else:
        lines.append(f"{indent}    pass")

    return "\n".join(lines)


def _generate_source_for_class(cls_name, bases, methods, docstring):
    """Generate realistic source code for a class."""
    base_str = ", ".join(bases) if bases else ""
    lines = [f"class {cls_name}({base_str}):" if base_str else f"class {cls_name}:"]
    if docstring:
        lines.append(f'    """{docstring}"""')
    lines.append("")
    for method in methods[:5]:
        lines.append(f"    def {method}(self):")
        lines.append(f"        pass")
        lines.append("")
    return "\n".join(lines)


def generate_symbols():
    """Generate all synthetic symbols."""
    rng = random.Random(42)
    symbols = []

    # Fake repo names for diversity
    repos = [
        "python/fastapi-clone", "python/flask-clone", "python/django-clone",
        "python/celery-clone", "python/sqlalchemy-clone", "python/pydantic-clone",
        "python/httpx-clone", "python/rich-clone", "python/pytest-clone",
        "python/click-clone", "python/starlette-clone", "python/aiohttp-clone",
        "python/langchain-clone", "python/requests-clone", "python/attrs-clone",
        "python/myproject-web", "python/myproject-api", "python/myproject-cli",
        "python/myproject-data", "python/myproject-ml",
    ]

    for domain_name, files, class_defs, func_defs in DOMAINS:
        repo = rng.choice(repos)

        # Generate class symbols
        for cls_name, bases, methods, docstring in class_defs:
            file = rng.choice(files)
            source = _generate_source_for_class(cls_name, bases, methods, docstring)
            symbols.append({
                "type": "class",
                "name": cls_name,
                "file": file,
                "docstring": docstring,
                "source": source,
                "methods": methods,
                "base_classes": bases,
                "language": "python",
                "repo": repo,
            })

            # Generate method symbols for each class
            for method_name in methods:
                method_doc = f"{method_name.replace('_', ' ').capitalize()} for {cls_name}."
                method_source = _generate_source_for_function(
                    method_name, [], method_doc, [], class_name=cls_name
                )
                symbols.append({
                    "type": "function",
                    "name": method_name,
                    "file": file,
                    "docstring": method_doc,
                    "source": method_source,
                    "params": ["self"],
                    "decorators": [],
                    "class_name": cls_name,
                    "calls": [],
                    "language": "python",
                    "repo": repo,
                })

        # Generate standalone function symbols
        for func_name, params, docstring, decorators, calls in func_defs:
            file = rng.choice(files)
            source = _generate_source_for_function(func_name, params, docstring, calls)
            symbols.append({
                "type": "function",
                "name": func_name,
                "file": file,
                "docstring": docstring,
                "source": source,
                "params": params,
                "decorators": decorators,
                "class_name": None,
                "calls": calls,
                "language": "python",
                "repo": repo,
            })

    # ── Add dunders (important for training dunder negatives) ─────
    dunder_names = [
        "__init__", "__repr__", "__str__", "__eq__", "__hash__",
        "__len__", "__getitem__", "__setitem__", "__delitem__",
        "__iter__", "__next__", "__contains__", "__enter__", "__exit__",
        "__call__", "__getattr__", "__setattr__", "__del__",
        "__lt__", "__le__", "__gt__", "__ge__", "__ne__",
        "__add__", "__sub__", "__mul__", "__truediv__",
        "__bool__", "__int__", "__float__", "__bytes__",
    ]

    for dunder in dunder_names:
        for domain_name, files, class_defs, _ in DOMAINS:
            if not class_defs:
                continue
            repo = rng.choice(repos)
            cls = rng.choice(class_defs)
            cls_name = cls[0]
            file = rng.choice(files)
            source = _generate_source_for_function(dunder, [], f"Magic method {dunder}.", [], class_name=cls_name)
            symbols.append({
                "type": "function",
                "name": dunder,
                "file": file,
                "docstring": f"Magic method {dunder} for {cls_name}.",
                "source": source,
                "params": ["self"],
                "decorators": [],
                "class_name": cls_name,
                "calls": [],
                "language": "python",
                "repo": repo,
            })

    # ── Add __init__.py file functions (important for init file negatives) ──
    init_funcs = [
        "register_plugins", "setup_logging", "configure_app", "init_extensions",
        "create_default_config", "register_blueprints", "setup_middleware",
        "load_models", "register_signals", "init_celery",
    ]
    for func_name in init_funcs:
        for domain_name, files, _, _ in DOMAINS:
            init_files = [f for f in files if f.endswith("__init__.py")]
            if not init_files:
                continue
            repo = rng.choice(repos)
            file = rng.choice(init_files)
            symbols.append({
                "type": "function",
                "name": func_name,
                "file": file,
                "docstring": f"Module initialization: {func_name.replace('_', ' ')}.",
                "source": f"def {func_name}():\n    pass\n",
                "params": [],
                "decorators": [],
                "class_name": None,
                "calls": [],
                "language": "python",
                "repo": repo,
            })

    # ── Duplicate common method names across classes (for disambiguation) ──
    common_methods = ["execute", "get", "set", "run", "process", "handle", "validate", "create", "update", "delete",
                      "save", "load", "close", "open", "start", "stop", "connect", "send", "receive", "parse"]
    extra_classes = [
        ("DatabaseExecutor", "db/executor.py"), ("ToolExecutor", "tools/executor.py"),
        ("TaskExecutor", "tasks/executor.py"), ("CommandExecutor", "cli/executor.py"),
        ("QueryRunner", "db/runner.py"), ("TestRunner", "tests/runner.py"),
        ("JobRunner", "scheduler/runner.py"), ("ScriptRunner", "scripts/runner.py"),
        ("RequestHandler", "http/handler.py"), ("EventHandler", "events/handler.py"),
        ("ErrorHandler", "errors/handler.py"), ("WebhookHandler", "webhooks/handler.py"),
        ("CacheManager", "cache/manager.py"), ("SessionManager", "auth/manager.py"),
        ("ConnectionManager", "db/manager.py"), ("ResourceManager", "resources/manager.py"),
    ]
    for cls_name, file_path in extra_classes:
        repo = rng.choice(repos)
        methods_for_cls = rng.sample(common_methods, min(8, len(common_methods)))
        source = _generate_source_for_class(cls_name, [], methods_for_cls, f"{cls_name} implementation.")
        symbols.append({
            "type": "class",
            "name": cls_name,
            "file": file_path,
            "docstring": f"{cls_name} implementation.",
            "source": source,
            "methods": methods_for_cls,
            "base_classes": [],
            "language": "python",
            "repo": repo,
        })
        for method in methods_for_cls:
            method_source = _generate_source_for_function(method, [], f"{method} for {cls_name}.", [], class_name=cls_name)
            symbols.append({
                "type": "function",
                "name": method,
                "file": file_path,
                "docstring": f"{method.replace('_', ' ').capitalize()} for {cls_name}.",
                "source": method_source,
                "params": ["self"],
                "decorators": [],
                "class_name": cls_name,
                "calls": [],
                "language": "python",
                "repo": repo,
            })

    # ── Add more diverse standalone functions ─────────────────────
    extra_verbs = ["fetch", "calculate", "render", "compile", "dispatch",
                   "broadcast", "subscribe", "unsubscribe", "emit", "trigger",
                   "initialize", "terminate", "allocate", "deallocate", "map",
                   "reduce", "transform", "convert", "migrate", "synchronize"]
    extra_nouns = ["record", "batch", "stream", "channel", "buffer",
                   "partition", "segment", "fragment", "chunk", "block",
                   "pipeline", "workflow", "process", "thread", "coroutine",
                   "callback", "listener", "observer", "subscriber", "publisher"]

    for _ in range(2000):
        verb = rng.choice(extra_verbs)
        noun = rng.choice(extra_nouns)
        name = f"{verb}_{noun}"
        domain_idx = rng.randrange(len(DOMAINS))
        _, files, _, _ = DOMAINS[domain_idx]
        file = rng.choice(files)
        repo = rng.choice(repos)
        params = rng.sample(["data", "config", "options", "timeout", "retry", "callback", "context"], rng.randint(0, 3))
        doc = f"{verb.capitalize()} the {noun}."
        source = _generate_source_for_function(name, params, doc, [])
        symbols.append({
            "type": "function",
            "name": name,
            "file": file,
            "docstring": doc,
            "source": source,
            "params": params,
            "decorators": [],
            "class_name": None,
            "calls": [],
            "language": "python",
            "repo": repo,
        })

    # ── Add TypeScript-style symbols for cross-language coverage ──
    ts_symbols = [
        ("createApp", "src/app.ts", "Create the application."),
        ("useAuth", "src/hooks/useAuth.ts", "Authentication hook."),
        ("useRouter", "src/hooks/useRouter.ts", "Router hook."),
        ("useState", "src/hooks/useState.ts", "State management hook."),
        ("fetchData", "src/api/client.ts", "Fetch data from API."),
        ("renderComponent", "src/renderer.ts", "Render a component."),
        ("handleSubmit", "src/forms/handler.ts", "Handle form submission."),
        ("validateForm", "src/forms/validation.ts", "Validate form data."),
        ("formatCurrency", "src/utils/format.ts", "Format currency value."),
        ("parseQueryString", "src/utils/url.ts", "Parse URL query string."),
    ]
    for name, file, doc in ts_symbols:
        symbols.append({
            "type": "function",
            "name": name,
            "file": file,
            "docstring": doc,
            "source": f"export function {name}() {{\n  // {doc}\n}}\n",
            "params": [],
            "decorators": [],
            "class_name": None,
            "calls": [],
            "language": "typescript",
            "repo": "typescript/myapp",
        })

    print(f"Generated {len(symbols)} symbols")
    funcs = sum(1 for s in symbols if s["type"] == "function")
    classes = sum(1 for s in symbols if s["type"] == "class")
    dunders = sum(1 for s in symbols if s["type"] == "function" and s["name"].startswith("__") and s["name"].endswith("__"))
    methods = sum(1 for s in symbols if s["type"] == "function" and s.get("class_name"))
    print(f"  Functions: {funcs} ({methods} methods, {dunders} dunders)")
    print(f"  Classes: {classes}")

    return symbols


if __name__ == "__main__":
    symbols = generate_symbols()
    OUTPUT.write_text(json.dumps(symbols, indent=2))
    print(f"\nSaved to {OUTPUT} ({OUTPUT.stat().st_size:,} bytes)")
