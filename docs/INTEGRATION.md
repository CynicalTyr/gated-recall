# Integration patterns

This kernel is the **inject policy** in front of a store you already have.
It does not open SQLite, call an embedding API, or start an MCP server.

```
Your chat worker
    │
    ▼
should_retrieve?  ──no──►  empty string (greeting / isolated / review)
    │ yes
    ▼
identity_allowlist  ──empty──►  empty string (deny, do not search)
    │ named ids
    ▼
your FTS list + your vector list  →  RRF on rowid / message_id / chunk_id
    │
    ▼
cap chars → append to the user turn (primary lane only)

Failover to a small local model
    │
    ▼
strip_recall_blocks / prepare_user_turn(..., lane=fallback)
```

Do **not** put `hybrid_search` in the system prompt. Do **not** add an MCP
`memory_search` tool that runs on every turn. That is always-on RAG with
tool tokens on top.

## Map libraries onto this kernel

| Library (live docs) | What they optimize | What you do instead |
| ------------------- | ------------------ | ------------------- |
| LangChain `create_agent` + `InMemorySaver` (`thread_id`) | Persist **thread history** | History is not recall. Call `build_inject` *before* `invoke` on recall-shaped turns only. |
| LangChain `SummarizationMiddleware` (`libs/langchain_v1/langchain/agents/middleware/summarization.py`) | Compress old **messages** near the token ceiling | Summarizing history does not strip a recall block before a smaller fallback model. Use `Lane.FALLBACK`. |
| LangChain NVIDIA RAG chain (`retriever \| prompt \| llm`) | Always retrieve then generate | Keep the retriever. Gate it with `should_retrieve` so “hey” never hits it. |
| Mem0 `client.search(..., filters={user_id})` (`docs/api-reference/memory/search-memories.mdx`) | Hybrid store + required entity id (400 if missing) | Mem0 still searches when *you* call it. Wrap the call in `build_inject` so greetings never call `search`. Empty `user_ids` here is deny without a round trip. |
| sqlite-vec NBC notebook RRF (`examples/nbc-headlines/3_search.ipynb`) | Fuse FTS + vectors **in SQL** on `article_id` | Keep that SQL. Pass the ranked lists into `reciprocal_rank_fusion` only if you fuse in Python. The kernel still decides *whether* rows enter the prompt. |
| CynicVec | ANN **cache** + shrink-guard save | Pair: cache miss fail-opens to FTS. Identity still fail-closes here. |

## Write order (do not invert)

**gate → allowlist → search → fuse → cap → append.**

If you search first and then ask the gate, you already paid latency and you
already risked logging snippets. `build_inject` does not call your lambdas
on greetings, empty ids, fallback, or evaluator.

```python
from gated_recall import Lane, RecallMonitor, build_inject, prepare_user_turn

def hybrid_messages(query: str, user_ids: tuple[str, ...]) -> list:
    return my_fts_plus_vectors(query, allowed_users=user_ids)

mon = RecallMonitor()
block = build_inject(
    user_text,
    lane=Lane.PRIMARY,
    user_ids=(canonical_user,),
    canonical_user=canonical_user,
    operator_ids=tuple(
        x.strip() for x in os.environ.get("GATED_RECALL_OPERATORS", "").split(",") if x.strip()
    ),
    max_chars=int(os.environ.get("GATED_RECALL_MAX_CHARS", "2000")),
    hybrid_messages=hybrid_messages,
    hybrid_documents=hybrid_documents if operator else None,
    monitor=mon,
)
print(mon.as_log(), file=sys.stderr)  # ids and counts, never snippets
user_turn = f"{user_text}\n{block}" if block else user_text
```

On failover:

```python
user_turn = prepare_user_turn(user_turn, user_text, lane=Lane.FALLBACK)
# now call the small model
```

## Env names (names only)

Copy `.env.example` to `.env` in *your* worker. This library does not
read them.

| Name | Role |
| ---- | ---- |
| `GATED_RECALL_OPERATORS` | Comma-separated ids allowed to search **documents** |
| `GATED_RECALL_MAX_CHARS` | Cap on the injected block (example **2000**) |

Never commit `.env`. Never paste live values into issues.

## Harness (inspect only)

Keep retrieval in **your daemon**. Cursor / Claude Desktop / Codex / other
IDEs already ReAct. Optional inspect: last `RecallMonitor.as_log()`. If you
wrap anything in MCP, logs go to **stderr** so stdout stays a JSON-RPC pipe.
Restart the harness after editing that wrapper.

Paste-ready policy is in [`START_HERE.md`](../START_HERE.md) §5.

## Fail-open vs fail-closed

| Layer | Stance |
| ----- | ------ |
| Missing vector **cache** (CynicVec / FAISS file) | **Fail-open** → FTS / SQL |
| Empty `user_ids` | **Fail-closed** → no search |
| `Lane.EVALUATOR` | **Fail-closed** → no hybrid |
| `Lane.FALLBACK` | **Fail-closed** on inject; **strip** leftover blocks |

## Pairing

- [cynicvec](https://github.com/CynicalTyr/cynicvec) — ANN is a cache. This kernel is **when** hits enter the prompt.
- [gated-rewoo](https://github.com/CynicalTyr/gated-rewoo) — planner skip. Memory inject is a different gate.
- `agent-review-envelope` — generator ≠ evaluator. Do not hybrid-RAG the reviewer.
