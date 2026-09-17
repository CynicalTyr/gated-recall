# Advanced: always-on RAG vs gated 0-or-capped inject

This guide is for people who already ran [`START_HERE.md`](../START_HERE.md)
and want the design that keeps showing up in production: **why “hey” still
pays for retrieval**, how a small-model fallback inherits a large-window
recall block, why Reciprocal Rank Fusion on snippet prefixes splits one
message, and how empty identity filters leak tenants.

Search terms this document is meant to answer: *RAG always inject greetings*,
*hybrid RRF snippet prefix*, *identity empty filter RAG*, *strip RAG on
fallback small model*, *evaluator same retrieved hits nodding loop*,
*token budget always-on top-k agent memory*.

---

## 1. The failure that looks like a smarter agent

A chat worker embeds the user turn, takes top-5, and prepends them on every
request — including “hey”, “thanks”, and “ok”. The demo looks informed. The
invoice and the KV cache grow while the user said nothing to recall.

That path has four hidden properties:

1. **The retrieve succeeded.** A greeting still has nearest neighbors. Success
   is the bug.
2. **Failover copies the block.** The large-window call already stuffed
   `=== SEMANTIC MEMORY RECALL` into the user turn. The 8k emergency model
   inherits it.
3. **Snippet-prefix fusion lies.** Two paraphrases of one `message_id` become
   two hits. Two different messages that share a header become one.
4. **Empty filter looks like “search everyone.”** Guests see operator
   documents. Similarity is not permission.

Per the Cynical0n3 NotebookLM (`systems` + `deep-thought`): always-on top-k
is the token-budget blowout and “lost in the middle”; the writer must not
also accept its own review; ANN caches may fail-open to lexical search;
**identity must fail-closed**. A competent engineer still violates this by
`similarity_search` in the system prompt and by pasting the same hits into
the evaluator.

The correct primitive is **0 or a capped block**, not “top-5 on every turn.”

---

## 2. Quantified incident (lab shape, no live host)

A companion loop handled **200** short turns in a day (“hey”, “thanks”,
status pings) and **12** recall-shaped questions. Always-on RAG used
**top-5** hits at ~**400 characters** each (~**2,000 characters**,
roughly **500–700 BPE tokens**) on every turn.

| Step | Without this kernel | With this kernel |
| ---- | ------------------- | ---------------- |
| 200 greetings | 200 × ~2,000 chars retrieved and stuffed | **0** retrieval calls, **0** chars |
| 12 recall questions | 12 × ~2,000 chars | 12 × capped block (same cap, only when needed) |
| Day total inject | ~**424,000** characters into prompts | ~**24,000** characters |
| Cloud → 8k local failover mid-turn | Small model sees the leftover 2k block + the question | `strip_recall_blocks` leaves the question |
| Guest, forgotten `user_ids` | Search returns every tenant’s rows | Empty allowlist → **no search** |
| Evaluator / systems review | Same hits as the writer (nodding loop) | `Lane.EVALUATOR` → `""` |

A 2,000-character block on a four-token “hey” is hundreds of tokens the
model does not need. At typical hosted rates that is still cents; at
**context-window and KV-cache** cost on a small GPU it is the difference
between a snappy local fallback and an OOM. Track **skip rate** (should be
high) and injected chars per turn (should be ~0 except recall-shaped). Do
**not** optimize “recall hit rate” until greetings retrieve again.

---

## 3. Fuse on ids, not `snippet[:100]`

sqlite-vec’s NBC headlines notebook already fuses FTS and vectors in SQL
on `article_id` with RRF `k=60`. That is the right **join key** for news
rows. Agent **messages** paraphrase themselves. If you fuse in Python on
the first 100 characters of the snippet, one decision becomes two hits and
two decisions that share a prefix become one.

```python
from gated_recall import Hit, reciprocal_rank_fusion

fused = reciprocal_rank_fusion(vector_hits, fts_hits, k=60, top_k=5)
# Hit.message_id / rowid / chunk_id is the key. Snippet prefix is last resort.
```

---

## 4. What `strip_recall_blocks` is *not*

GitHub `search_code` for `strip_recall_blocks` hits other products. Read
them before assuming this kernel is a clone:

| File | What that strip is for |
| ---- | ---------------------- |
| `agentic-in/elephant-agent` `packages/models/ephemeral_injection.py` | Remove “Current-turn recall support:” so a **tool-loop retry** can rebuild a prefix-cache-friendly suffix. Not a greeting gate, not empty-deny, not small-model failover. |
| `12ziyad/universal-memory-engine` `packages/_kernel/py/_kernel.py` | Remove `<itsuki-recalled-context-v1>` so stored memory is not **re-captured as user speech**. Hosted/graph memory SDK, not a drop-in inject policy. |

This kernel strips `=== SEMANTIC MEMORY RECALL` / `=== DOCUMENT MEMORY RECALL`
**because the primary lane already appended them** and the next call is a
smaller window. Same English name, different contract.

---

## How this stands out

Researched with Context7 (`libraryId=/websites/langchain` `SummarizationMiddleware`
+ `InMemorySaver` `thread_id`; NVIDIA `retriever | prompt | llm` always
retrieves; `libraryId=/mem0ai/mem0` `search` requires an entity id in
`filters` else HTTP 400, default `top_k=10`; `libraryId=/asg017/sqlite-vec`
`examples/nbc-headlines/3_search.ipynb` RRF on `article_id`, `rrf_k=60`)
and GitHub-MCP (`langchain-ai/langchain` file
`libs/langchain_v1/langchain/agents/middleware/summarization.py`;
`mem0ai/mem0` file `docs/api-reference/memory/search-memories.mdx`;
`asg017/sqlite-vec` file `examples/nbc-headlines/3_search.ipynb`;
`should_retrieve_semantic language:python` → **zero** hits;
`Lane.FALLBACK` + `SEMANTIC MEMORY RECALL` → **zero** hits;
`gated-recall in:name` → **zero** repos). DeepWiki on
`langchain-ai/langchain` describes retrievers and middleware that compress
**history**, not a retrieve-when gate. DeepWiki on `mem0ai/mem0` describes
a memory **layer** (extract, hybrid retrieve, graph) the caller still
invokes. Sibling kernels `cynicvec` and `gated-rewoo` are ANN persist and
planner skip — not inject policy.

| Obvious alternative | What they optimize | What they miss | This kernel |
| ------------------- | ------------------ | -------------- | ----------- |
| LangChain RAG chain / retriever in the prompt | Nearest docs on **every** invoke | Greetings still retrieve; failover keeps the block | Gate + strip on `Lane.FALLBACK` |
| `SummarizationMiddleware` (`summarization.py`) | Shrink old **messages** near the ceiling | Does not decide *whether* hybrid hits enter; does not skip the evaluator | 0-or-capped inject; evaluator returns `""` |
| Mem0 `search` (`search-memories.mdx`) | Hybrid store; **requires** an entity filter | You still call search on “hey”; `OR` user+agent widens scope; not a small-model strip | Empty `user_ids` never calls your lambda; greetings never call it |
| sqlite-vec NBC RRF (`3_search.ipynb`) | Fuse FTS + vectors in SQL | Does not own the agent prompt | You keep SQL; this owns **when** rows are formatted into the turn |
| elephant-agent `ephemeral_injection.py` | Prefix-cache-safe suffix + tool-loop cache | Full agent runtime; strip is retry hygiene | Stdlib policy; strip is **failover** |
| Itsuki `_kernel.py` | Marker wrap + echo-defence + secret scrub | Hosted memory engine | No HTTP; you bring the store |
| `cynicvec` (sibling) | Shrink-guard ANN **cache** | Cache miss ≠ “do not inject” | Fail-open cache, fail-closed identity |
| `gated-rewoo` (sibling) | Skip planner on short queries | Planner ≠ memory | Same family: the **gate** is the product |

**Non-obvious / high-leverage:** retrieval success on a greeting is still a
failure. Empty identity is deny, not global search — copying ANN fail-open
onto the allowlist is how guests see operator documents. RRF key is
`rowid` / `message_id` / `chunk_id`. The other fuse is
`strip_recall_blocks` on fallback so an 8k local model does not inherit a
2k block from the large window.

**Mental model to replace:** adopters think “RAG *is* memory, so more hits
are more intelligence.” The governing model is **0 or a capped block**.
LangChain persists thread history; Mem0 stores and searches when asked;
sqlite-vec ranks rows. None of them skip “hey”, strip on small-model
failover, and refuse the evaluator the writer’s hits as one function.

**Incentive:** always-inject looks smarter in a 3-turn demo and needs one
code path. After a cloud timeout, that path is how the emergency model
OOMs or hallucinates from leftover hits.

**Second-order effect:** once copied, teams will optimize “recall hit rate”
and widen `_RECALL_INTENT` until greetings match. That metric is the
bypass. Count skip rate and injected chars per turn. Do **not** add
`always=True`. Do **not** add an MCP tool that bypasses the gate.

---

## 5. Measuring use

| Signal | Healthy |
| ------ | ------- |
| Greeting / “thanks” inject chars | **0** |
| Empty `user_ids` hybrid lambda calls | **0** |
| Evaluator hybrid chars | **0** |
| Fallback still contains `SEMANTIC MEMORY RECALL` | **Zero** |
| Recall-shaped inject | Capped (`GATED_RECALL_MAX_CHARS`, default 2000) |
| Monitor log | ids and counts, **not** snippets |

---

## 6. Short comparison (same facts, operator table)

See **How this stands out** above for library/file evidence. In one
line: this is not a vector database and not a hosted memory API. It is a
stdlib kernel you drop into the chat worker you already have. You still
own FTS, embeddings, and the query tool. The product is **when** those
rows may spend tokens.
