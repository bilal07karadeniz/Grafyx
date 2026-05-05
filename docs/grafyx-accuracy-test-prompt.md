# Grafyx Accuracy Test Prompt

Paste the content below (between the --- lines) into a Claude Code session on your real project.

> **Tip:** This is 25 tests across many tools. On large projects, context can fill up. If you're running into context pressure, split into three sessions (tests 1–8, 9–17, 18–25) and paste the rubric below at the top of each. The final report can be assembled from all three.

---

You have access to Grafyx MCP tools. I need you to run a systematic accuracy audit of every tool. For EACH test below, call the tool, then rate accuracy on the 5-bucket scale (defined below) and note any issues.

## IMPORTANT RULES

- **DO NOT MODIFY ANY CODE. This is a READ-ONLY audit. Do not edit, create, or delete any files. Only call Grafyx MCP tools and read files to verify results.**
- **If a tool result looks suspicious, your job is to verify with the `Read` tool against the actual source and report. You are NOT to "fix" Grafyx, write a workaround, or modify this test prompt. Stop and report rather than improvise.**
- After each tool call, tell me: what it returned, whether the result is CORRECT, and the rating using the scale below.
- To verify results, READ the actual source files and compare against what Grafyx reported. Don't just trust the output — cross-check it.
- If something is wrong or missing, explain SPECIFICALLY what's wrong with the symbol name, file path, and what you expected.
- Don't skip any test (unless its "if your project uses X" gate is unmet — then write N/A and move on).
- Write your findings in a structured report at the end.

## Rating Scale (use these exact values, not arbitrary 1–100 numbers)

| Score | Meaning |
|---|---|
| **5** | Perfect — nothing wrong |
| **4** | Minor issue (≤1 false positive/negative, easy workaround) |
| **3** | Useful but materially incomplete (multiple things missing or wrong, but the answer was still helpful) |
| **2** | Mostly wrong — gives misleading info more than half the time |
| **1** | Broken / refuses to work / completely incorrect |
| **N/A** | Test gate not applicable to this project (e.g., FastAPI test on a Django project) |

## Known Limitations — Do NOT Grade These Down As Bugs

Some patterns are *documented* architectural limits of static analysis,
not regressions. If the audit hits one of these, mark the affected test
**N/A** or call it out under "Known limitations encountered" rather than
scoring 1–3.

- **Decorator-on-instance dispatch** — `@router.get("/x")` is NOT
  attributed as a caller of `APIRouter.get`. Resolving this would
  require type-flow analysis. Same for `@app.delete`, `@router.patch`,
  etc. Test 7/8/17 may surface this.
- **Same-file function-level dependencies** — `get_dependency_graph`
  reports *file-level* imports only. A function calling a sibling helper
  in the same file does NOT appear in `depended_on_by`. Test 15.
- **Celery `.delay()` / `.apply_async()`** — task dispatch through the
  Celery registry is not detected. Test 17.
- **Pipecat FrameProcessor methods** — `process_frame`, `push_frame`,
  etc. show 0 callers due to dynamic pipeline dispatch. Test 16/17.
- **TypeScript object-literal methods** — `const api = { fn: () => ... }`
  may show 0 functions in skeleton/module counts (graph-sitter parser
  limitation, partially worked around in v0.2.2 via Pass 8 but not
  100%). Test 1/3/4.
- **`from pkg import C` where `C` is a class re-exported via
  `__init__.py`** — `imported_by` may undercount by 1–2 files vs.
  direct grep for lazy-loader `__init__.py` patterns. Test 5.

---

## TEST 1: Project Skeleton

Call `get_project_skeleton` with default settings.

- Does the file/function/class count look correct?
- Is the directory structure accurate?
- Are any important directories missing?
- Rating: ?/5

## TEST 2: Project Skeleton (signatures detail)

Call `get_project_skeleton` with `detail="signatures"` and `include_hints=true`.

- Are the hints useful? Would a fresh agent who knows nothing about this codebase actually follow them productively?
- Rating: ?/5

## TEST 3: Module Context

Call `get_module_context` on your MAIN backend package (e.g., "app" or "backend" or "api" — whatever the core Python package is).

- Does it list all files in that directory?
- Are function/class counts per file accurate?
- Are internal import relationships correct?
- Rating: ?/5

## TEST 4: Module Context (frontend)

**If your project has a frontend (TypeScript / JavaScript / React / Vue):** Call `get_module_context` on a frontend directory (e.g., "src/components" or "app/components"). Does it handle TS/React files? Are object-literal methods (`const api = { foo: () => ... }`) detected?

**If your project has no frontend:** N/A.

- Rating: ?/5 or N/A

## TEST 5: File Context

Pick an IMPORTANT backend file (a core service, main router, or auth module) and call `get_file_context` on it.

- Are all functions listed?
- Are all classes listed with correct methods?
- Are imports correct?
- Is `imported_by` accurate? (Check: are files that import this file actually listed? Pick 3 files Grafyx says import this one and verify with `Read`.)
- Rating: ?/5

## TEST 6: Function Context — Simple

Pick a well-known utility function (something called from many places) and call `get_function_context`.

- Are the outgoing calls correct?
- Are the incoming callers correct? Check 3-4 of them manually with `Read` — do they actually call this function?
- Are there MISSING callers you know about but aren't listed?
- Rating: ?/5

## TEST 7: Function Context — Ambiguous Name

Pick a method name that exists in MULTIPLE classes (e.g., "execute", "process", "handle", "create", "get", "update", "validate"). Call `get_function_context` with just the method name.

- Does it show a disambiguation list?
- If you pick one, are the callers CORRECT for that specific class's method? (Not callers of a different class's method with the same name?)
- This tests the M2 caller disambiguator — getting this wrong is a known concern, please report concretely.
- Rating: ?/5

## TEST 8: Function Context — FastAPI Route Handler

**If your project uses FastAPI:** Pick a route handler decorated with `@router.get/post/etc`. Call `get_function_context` on it. Does it show the decorator? Are `Depends()` injected functions listed as dependencies or calls?

**If your project doesn't use FastAPI:** Pick any decorated function (Django view, Celery task, click command, pytest fixture, etc.) and check whether the decorator appears in the response.

- Rating: ?/5 or N/A

## TEST 9: Class Context

Pick your most important class (main service, core model, or base class) and call `get_class_context`.

- Are all methods listed?
- Are base classes correct?
- Is `cross_file_usages` accurate? Check 3-4 files — do they actually use this class?
- Are there files that use this class but are NOT listed?
- Rating: ?/5

## TEST 10: Class Context — Inheritance

Pick a class that has subclasses. Call `get_class_context`, then call `get_subclasses` on the base class.

- Does `get_class_context` show the correct base classes?
- Does `get_subclasses` find ALL subclasses?
- **If the project has multiple classes with the same name** (e.g., two
  `Base`, two `Config`, two `Manager`), call `get_subclasses` without
  `file_path=` first — the response should set `ambiguous: true` and
  list `candidates`. Then call again with `file_path=<one of them>` and
  verify only the subclasses of THAT specific class are returned. (New
  in v0.2.2; absence of disambiguation = bug.)
- Rating: ?/5

## TEST 11: Search — Exact Match

Call `find_related_code` with a query that exactly matches a known function name (e.g., "authenticate_user" or whatever exists).

- Is that function the #1 result?
- Is the score high (>0.7)?
- Rating: ?/5

## TEST 12: Search — Semantic/Conceptual ⭐ HEADLINE SEARCH TEST

**This test directly measures the fastembed-backed semantic encoder
(introduced in v0.2.0, hardened in v0.2.2).** The synthetic benchmark
in `docs/benchmarks/0.2.0/` reported nDCG@10 = 0.787 on FastAPI +
Django. If real-project queries score materially worse, that's a
finding worth filing as a bug.

Call `find_related_code` with a CONCEPTUAL query that doesn't match any symbol name exactly. Examples:

- "handle user login" (if your auth functions are named differently)
- "send notification to user"
- "process payment"
- "validate input data"
- "retry with exponential backoff"

Pick something where you KNOW which functions should match but the query words don't appear in the function name.

- Did it find the right functions?
- Were there false positives (irrelevant results)?
- Rating: ?/5

## TEST 13: Search — Gibberish

Call `find_related_code` with `"xyzzy foobar qlrmph"`.

- Does it return empty or low-confidence results (with `degraded` or `low_confidence` flags)?
- Or does it return irrelevant results with high scores? (BAD — that means the gibberish detector failed.)
- Rating: ?/5

## TEST 14: Search — Cross-Language

**If your project has both backend AND frontend code:** Call `find_related_files` with a query about a feature that spans both (e.g., "user authentication", "chat messages", "file upload"). Does it find BOTH Python backend files AND TypeScript/React frontend files?

**If your project is single-language:** N/A.

- Rating: ?/5 or N/A

## TEST 15: Dependency Graph

Pick a core function and call `get_dependency_graph` with `depth=2`.

- Are forward dependencies correct? (What this function depends on)
- Are reverse dependencies correct? (What depends on this function)
- Are any listed dependencies actually EXTERNAL packages incorrectly shown as local? (e.g., sqlalchemy, langchain, pydantic shown as local deps — known failure mode worth checking)
- Rating: ?/5

## TEST 16: Call Graph

Pick a function that you know has a deep call chain and call `get_call_graph` with `depth=3`.

- Is the outgoing call tree correct?
- Is the incoming caller tree correct?
- Are there phantom calls (listed but don't actually happen)?
- Are there missing calls you know about?
- Rating: ?/5

## TEST 17: Call Graph — Dynamic Dispatch

**If your project uses ANY framework that dispatches dynamically** (LangChain chains, Pipecat pipelines, Celery `.delay()`, plugin systems, callback registries, observer patterns):

Pick a function that's called via the dynamic dispatch path. Call `get_call_graph`.

- Does Grafyx detect the connection? Or does the dynamic dispatch make it miss?
- This is a known limitation — Celery `.delay()` callers are explicitly not detected. Please report what specifically gets missed in your stack.

**If your project doesn't have dynamic-dispatch frameworks:** N/A.

- Rating: ?/5 or N/A

## TEST 18: Unused Symbols — Functions

Call `get_unused_symbols` with `symbol_type="functions"`, `include_tests=false`, `max_results=30`.

- Go through the top 10 results. For each one: is it TRULY unused, or is it a false positive?
- Count: X truly unused / 10 checked = X0% precision
- Rating: ?/5

## TEST 19: Unused Symbols — Classes

Call `get_unused_symbols` with `symbol_type="classes"`, `include_tests=false`.

- Check top 5. Are they truly unused?
- Are any FastAPI models, Pydantic schemas, SQLAlchemy models, or LangChain tools falsely flagged as unused? (These often look unused statically because they're discovered via reflection / import side effects.)
- Rating: ?/5

## TEST 20: Conventions

Call `get_conventions`.

- Are the detected conventions accurate?
- Does the naming convention match what you actually use?
- Are the counts/percentages believable?
- Rating: ?/5

## TEST 21: Navigation Flow Test

Simulate a real exploration workflow:

1. Call `get_project_skeleton` with `include_hints=true`
2. Follow the top hint suggestion
3. From that result, follow its top hint
4. Continue for 3-4 hops

- Did the hints lead you to interesting/important code?
- Or did they lead to dead ends or unimportant code?
- Rate the OVERALL navigation flow: ?/5

## TEST 22: Refresh After Edit

This tests the watchdog / incremental-refresh path nothing else covers.

1. Note the result of `get_function_context` on a specific function (just observe, don't change anything yet).
2. **Pretend** the function was edited (you don't actually need to edit it — just describe what change you'd make hypothetically; we don't want to modify code).
3. Call `refresh_graph` and confirm it completes.
4. Re-call `get_function_context` on the same function — does the response come back equivalent? Any errors? Did the refresh take a reasonable time (<10s for small repos, <60s for huge ones)?

- Rating: ?/5

## TEST 23: Token Budget / Response Size

Pick a complex tool response. Either:

- `get_file_context` on a large file (>1000 lines) with `detail="full"`, OR
- `get_class_context` on a class with many methods + cross_file_usages, OR
- `get_call_graph` with `depth=3` on a hub function.

- Is the response under ~25K tokens (i.e., usable inside a single Claude turn)?
- If truncation kicks in, does it preserve the most important info, or does it cut off the headline answer?
- Rating: ?/5

## TEST 24: Degraded Mode

Test the v0.2.0 fallback behavior:

1. If the embedding extra is installed: call `find_related_code` with any query and check the response. It should have `model: {model: "jina-v2", ...}` and NOT have `degraded: true`.
2. **Optional:** start a separate Grafyx process with `GRAFYX_ENCODER=invalid_xyz` (or simulate by reading the response when the encoder hasn't built yet). Call `find_related_code`. Does the response include `degraded: true` and an `action_hint` field pointing at the embeddings extra?

- Rating: ?/5

---

## FINAL REPORT

After all tests, create a summary table:

| Test # | Tool | Rating /5 | Key Issues |
|--------|------|-----------|------------|
| 1 | get_project_skeleton | ?/5 | ... |
| 2 | skeleton + hints | ?/5 | ... |
| 3 | module context (backend) | ?/5 | ... |
| 4 | module context (frontend) | ?/5 or N/A | ... |
| 5 | file context | ?/5 | ... |
| 6 | function context (simple) | ?/5 | ... |
| 7 | function context (ambiguous) | ?/5 | ... |
| 8 | function context (decorated) | ?/5 or N/A | ... |
| 9 | class context | ?/5 | ... |
| 10 | class context (inheritance) | ?/5 | ... |
| 11 | search (exact match) | ?/5 | ... |
| 12 | search (semantic) ⭐ | ?/5 | ... |
| 13 | search (gibberish) | ?/5 | ... |
| 14 | search (cross-language) | ?/5 or N/A | ... |
| 15 | dependency graph | ?/5 | ... |
| 16 | call graph | ?/5 | ... |
| 17 | call graph (dynamic dispatch) | ?/5 or N/A | ... |
| 18 | unused symbols (functions) | ?/5 | ... |
| 19 | unused symbols (classes) | ?/5 | ... |
| 20 | conventions | ?/5 | ... |
| 21 | navigation flow (hints) | ?/5 | ... |
| 22 | refresh after edit | ?/5 | ... |
| 23 | token budget | ?/5 | ... |
| 24 | degraded mode | ?/5 | ... |

Then answer these questions:

1. Which tool was MOST accurate? (highest rating)
2. Which tool was LEAST accurate? (lowest rating)
3. What was the #1 most common type of error? (missing callers? false unused? bad search results? wrong dependencies?)
4. Were there any tools that were actively MISLEADING (gave confident but wrong information)?
5. For search specifically: what kinds of queries worked well vs poorly?
6. For caller/dependency tracking: what patterns were missed most? (DI? callbacks? dynamic dispatch? decorators?)
7. Overall, if you had to pick ONE thing to fix that would help the most, what would it be?
8. List every specific false positive and false negative you found, with the exact symbol name, file path, and why it was wrong.
9. **Search calibration:** the published synthetic benchmark in
   `docs/benchmarks/0.2.0/summary.md` reports nDCG@10 = 0.787 for the
   default `jina-v2` encoder. Compare your real-project semantic search
   quality (TEST 12). Does it match the synthetic benchmark roughly, or
   is it materially different? If different, why might that be — query
   style (terse vs docstring-like), codebase characteristics, encoder
   mismatch?
10. **Known-limitations encountered:** which of the documented
    limitations from the section at the top of this prompt did the
    audit hit? List them so we can distinguish "static-analysis ceiling"
    from "actual bug". Tests that surfaced one of these should be
    excluded from the overall accuracy rating.

Rate the overall Grafyx accuracy: ?/5

(When computing the overall rating, **exclude tests that scored
low purely due to documented limitations**. Score the rest. A repo that
heavily uses one of the limitation patterns will have several N/A or
excluded tests — that's expected.)
