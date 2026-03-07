# Grafyx Accuracy Test Prompt

Paste the content below (between the --- lines) into a Claude Code session on your real project.

---

You have access to Grafyx MCP tools. I need you to run a systematic accuracy audit of every tool. For EACH test below, call the tool, then rate accuracy 1-100 and note any issues.

## IMPORTANT RULES
- **DO NOT MODIFY ANY CODE. This is a READ-ONLY audit. Do not edit, create, or delete any files. Only call Grafyx MCP tools and read files to verify results.**
- After each tool call, tell me: what it returned, whether the result is CORRECT, and rate 1-100 (1=completely wrong, 50=half right, 100=perfect)
- To verify results, READ the actual source files and compare against what Grafyx reported. Don't just trust the output — cross-check it.
- If something is wrong or missing, explain SPECIFICALLY what's wrong
- Don't skip any test. Run them all in order.
- Write your findings in a structured report at the end.

---

## TEST 1: Project Skeleton
Call `get_project_skeleton` with default settings.
- Does the file/function/class count look correct?
- Is the directory structure accurate?
- Are any important directories missing?
- Rate: ?/100

## TEST 2: Project Skeleton (signatures detail)
Call `get_project_skeleton` with `detail="signatures"` and `include_hints=true`.
- Are the hints useful? Would you actually follow them?
- Rate hints quality: ?/100

## TEST 3: Module Context
Call `get_module_context` on your MAIN backend package (e.g., "app" or "backend" or "api" — whatever the core Python package is).
- Does it list all files in that directory?
- Are function/class counts per file accurate?
- Are internal import relationships correct?
- Rate: ?/100

## TEST 4: Module Context (frontend)
Call `get_module_context` on a frontend directory (e.g., "src/components" or "app/components").
- Does it handle TypeScript/React files?
- Rate: ?/100

## TEST 5: File Context
Pick an IMPORTANT backend file (a core service, main router, or auth module) and call `get_file_context` on it.
- Are all functions listed?
- Are all classes listed with correct methods?
- Are imports correct?
- Is `imported_by` accurate? (Check: are files that import this file actually listed?)
- Rate: ?/100

## TEST 6: Function Context — Simple
Pick a well-known utility function (something called from many places) and call `get_function_context`.
- Are the outgoing calls correct?
- Are the incoming callers correct? Check 3-4 of them manually — do they actually call this function?
- Are there MISSING callers you know about but aren't listed?
- Rate: ?/100

## TEST 7: Function Context — Ambiguous Name
Pick a method name that exists in MULTIPLE classes (e.g., "execute", "process", "handle", "create", "get", "update", "validate").
Call `get_function_context` with just the method name.
- Does it show a disambiguation list?
- If you pick one, are the callers CORRECT for that specific class's method? (Not callers of a different class's method with the same name?)
- Rate: ?/100

## TEST 8: Function Context — FastAPI Route Handler
Pick a FastAPI route handler decorated with @router.get/post/etc.
Call `get_function_context` on it.
- Does it show the decorator?
- Are Depends() injected functions listed as dependencies or calls?
- Rate: ?/100

## TEST 9: Class Context
Pick your most important class (main service, core model, or base class) and call `get_class_context`.
- Are all methods listed?
- Are base classes correct?
- Is `cross_file_usages` accurate? Check 3-4 files — do they actually use this class?
- Are there files that use this class but are NOT listed?
- Rate: ?/100

## TEST 10: Class Context — Inheritance
Pick a class that has subclasses. Call `get_class_context`.
- Does it show the correct base classes?
- Now call `get_subclasses` on the base class — are all subclasses found?
- Rate: ?/100

## TEST 11: Search — Exact Match
Call `find_related_code` with a query that exactly matches a known function name (e.g., "authenticate_user" or whatever exists).
- Is that function the #1 result?
- Is the score high (>0.7)?
- Rate: ?/100

## TEST 12: Search — Semantic/Conceptual
Call `find_related_code` with a CONCEPTUAL query that doesn't match any symbol name exactly. Examples:
- "handle user login" (if your auth functions are named differently)
- "send notification to user"
- "process payment"
- "validate input data"
Pick something where you KNOW which functions should match but the query words don't appear in the function name.
- Did it find the right functions?
- Were there false positives (irrelevant results)?
- Rate: ?/100

## TEST 13: Search — Gibberish
Call `find_related_code` with "xyzzy foobar qlrmph".
- Does it return empty or low-confidence results?
- Or does it return irrelevant results with high scores? (BAD)
- Rate: ?/100

## TEST 14: Search — Cross-Language
Call `find_related_files` with a query about a feature that spans frontend AND backend (e.g., "user authentication", "chat messages", "file upload").
- Does it find BOTH Python backend files AND TypeScript/React frontend files?
- Rate: ?/100

## TEST 15: Dependency Graph
Pick a core function and call `get_dependency_graph` with depth=2.
- Are forward dependencies correct? (What this function depends on)
- Are reverse dependencies correct? (What depends on this function)
- Are any listed dependencies actually EXTERNAL packages incorrectly shown as local? (e.g., sqlalchemy, langchain, pydantic shown as local deps)
- Rate: ?/100

## TEST 16: Call Graph
Pick a function that you know has a deep call chain and call `get_call_graph` with depth=3.
- Is the outgoing call tree correct?
- Is the incoming caller tree correct?
- Are there phantom calls (listed but don't actually happen)?
- Are there missing calls you know about?
- Rate: ?/100

## TEST 17: Call Graph — LangChain/Pipecat
Pick a LangChain chain, tool, or Pipecat pipeline function. Call `get_call_graph`.
- Does it detect the chain/pipeline connections?
- Or does the dynamic dispatch nature of LangChain/Pipecat make it miss connections?
- Rate: ?/100

## TEST 18: Unused Symbols
Call `get_unused_symbols` with `symbol_type="functions"`, `include_tests=false`, `max_results=30`.
- Go through the top 10 results. For each one: is it TRULY unused, or is it a false positive?
- Count: X truly unused / 10 checked = X0% precision
- Rate: ?/100

## TEST 19: Unused Symbols — Classes
Call `get_unused_symbols` with `symbol_type="classes"`, `include_tests=false`.
- Check top 5. Are they truly unused?
- Are any FastAPI models, Pydantic schemas, or LangChain tools falsely flagged?
- Rate: ?/100

## TEST 20: Conventions
Call `get_conventions`.
- Are the detected conventions accurate?
- Does the naming convention match what you actually use?
- Are the counts/percentages believable?
- Rate: ?/100

## TEST 21: Navigation Flow Test
Simulate a real exploration workflow:
1. Call `get_project_skeleton` with `include_hints=true`
2. Follow the top hint suggestion
3. From that result, follow its top hint
4. Continue for 3-4 hops
- Did the hints lead you to interesting/important code?
- Or did they lead to dead ends or unimportant code?
- Rate the OVERALL navigation flow: ?/100

## TEST 22: Circular Dependencies
Call `get_dependency_graph` on something you suspect has circular imports.
- Does it detect the cycle?
- Or does it miss it?
- Rate: ?/100

---

## FINAL REPORT

After all tests, create a summary table:

| Test # | Tool | Rating /100 | Key Issues |
|--------|------|-------------|------------|
| 1 | get_project_skeleton | ?/100 | ... |
| 2 | skeleton + hints | ?/100 | ... |
| 3 | module context (backend) | ?/100 | ... |
| 4 | module context (frontend) | ?/100 | ... |
| 5 | file context | ?/100 | ... |
| 6 | function context (simple) | ?/100 | ... |
| 7 | function context (ambiguous) | ?/100 | ... |
| 8 | function context (FastAPI) | ?/100 | ... |
| 9 | class context | ?/100 | ... |
| 10 | class context (inheritance) | ?/100 | ... |
| 11 | search (exact match) | ?/100 | ... |
| 12 | search (semantic) | ?/100 | ... |
| 13 | search (gibberish) | ?/100 | ... |
| 14 | search (cross-language) | ?/100 | ... |
| 15 | dependency graph | ?/100 | ... |
| 16 | call graph | ?/100 | ... |
| 17 | call graph (LangChain/Pipecat) | ?/100 | ... |
| 18 | unused symbols (functions) | ?/100 | ... |
| 19 | unused symbols (classes) | ?/100 | ... |
| 20 | conventions | ?/100 | ... |
| 21 | navigation flow (hints) | ?/100 | ... |
| 22 | circular dependencies | ?/100 | ... |

Then answer these questions:
1. Which tool was MOST accurate? (highest rating)
2. Which tool was LEAST accurate? (lowest rating)
3. What was the #1 most common type of error? (missing callers? false unused? bad search results? wrong dependencies?)
4. Were there any tools that were actively MISLEADING (gave confident but wrong information)?
5. For search specifically: what kinds of queries worked well vs poorly?
6. For caller/dependency tracking: what patterns were missed most? (DI? callbacks? dynamic dispatch? decorators?)
7. Overall, if you had to pick ONE thing to fix that would help the most, what would it be?
8. List every specific false positive and false negative you found, with the exact symbol name and why it was wrong.

Rate the overall Grafyx accuracy: ?/100
