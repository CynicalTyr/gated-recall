# START HERE

**If you only open one file, open this one.**

This guide assumes you can log into a computer, open a terminal, and paste
commands. It does **not** assume you know Docker, MCP, or how AI agents work.

Most agent RAG dumps the nearest snippets into every prompt. This kernel
**does not retrieve** on “hey”, **denies** an empty identity list, **fuses**
hybrid ranks on row ids, and **strips** leftover recall before a small
model runs.

## Who this helps

| Who | What they get |
| --- | ------------- |
| **You (learning)** | A 10-minute proof the code runs (`smoke ok`). Zero chars on a greeting is the *correct* skip. |
| **An AI harness** | Cursor, Claude Desktop, Copilot Chat, Codex — a program that runs a model *and* tools. See §5. |
| **A locally built AI** | Your own Python/timer worker. Function calls. MCP is optional. See §6. |
| **People talking to that AI** | Short chat stays cheap. Recall-shaped questions still see prior decisions. |

A **harness** is Cursor / Claude Desktop / VS Code Copilot — a program that
runs a model and **tools**. A **custom-built AI** is your own Python/timer
worker; HTTP or function calls; MCP optional.

---

## 0. Words you will see, then files

| Word | Plain meaning |
| ---- | ------------- |
| **Harness** | The IDE or app that hosts the model (Cursor, Claude Desktop). It can start **MCP tools**. |
| **MCP** | A way for the model to call small tools. Tools are not automatically safe. |
| **Locally built AI** | Your own loop: your code calls models and functions. You decide the order. |
| **Kernel** | This tiny library. It is not a full chatbot and not a vector database. |
| **Lane** | Primary (large window), fallback (small local model), evaluator (independent judge). |
| **Allowlist** | The user ids this query may see. Empty means **no hits**, not “everyone.” |
| **RRF** | Reciprocal Rank Fusion. Merge FTS and vector lists. Key on ids, not the first 100 characters of text. |

| File | What it does | What you change it for | How it helps agents / users |
| ---- | ------------ | ---------------------- | --------------------------- |
| `START_HERE.md` | This first-use guide | You usually do not | Humans: how to get `smoke ok` |
| `README.md` | Product + hidden dynamics | Forks / rename | Humans: “is this the right tool?” |
| `docs/INTEGRATION.md` | Worker wiring | New host / store | Custom AI (primary) |
| `docs/ADVANCED.md` | Always-on RAG vs gated inject | Architecture debates | People who already burned a context window |
| `gated_recall.py` | Gate, allowlist, RRF, strip, monitor | Keywords / cap (rarely) | The worker’s only auto-inject path |
| `examples/quickstart.py` | Greeting skip + id log + fallback strip | Learning | Proof without embeddings |
| `tests/` | Gate + merge-on-id + guest docs | Behavior changes | Guests stay message-scoped |
| `scripts/smoke.sh` | unittest + quickstart | CI locally | 10-minute first success |
| `.env.example` | Env **names** | Copy to `.env` (never commit `.env`) | Operator list + char cap |

**Mental picture:**

```
User turn
  → evaluator lane?     no hybrid
  → fallback lane?      strip recall blocks already in the turn
  → greeting?           skip
  → empty user ids?     skip (deny)
  → your FTS + vectors  → RRF on ids → cap chars → prompt
```

---

## 1. What you need

- Python 3.10 or newer. Check: `python3 -V`
- Ability to `cd` into this folder (the clone root)
- A throwaway directory for any `AGENT_HOME` (use `/tmp/...`, never a real home)

No GPU. No Docker. No API keys for the 10-minute path. No vector library.

---

## 2. First success (under 10 minutes)

From **this folder** (after clone it is named `gated-recall`):

```bash
chmod +x scripts/smoke.sh
./scripts/smoke.sh
```

You want a line `smoke ok` and no traceback. That script sets `PYTHONPATH`
for you. Optional later:

```bash
python3 -m pip install -e .
cp .env.example .env
python3 examples/quickstart.py
```

**This kernel’s success looks like:** `greeting inject chars: 0`, a recall
block with `msg_ids: [42]`, `empty allowlist` skip, and
`fallback stripped recall: True`. If greetings inject text, the gate is
broken — that is not “smarter RAG.”

If `python3` is missing, install Python from python.org or your package
manager, then try again.

---

## 3. How to edit (safe)

Change Python files in *this* folder. Re-run `./scripts/smoke.sh`.

This kernel has **no** MCP server. If you later wrap it in a harness tool,
**restart the harness** after editing that wrapper (the child process is
already running). Do not copy this folder over a live operator machine
“to try it.”

---

## 4. Configure

Copy `.env.example` to `.env` if you want named knobs for *your* worker.
Fill **names you own**. Never commit `.env`.

- `GATED_RECALL_OPERATORS` — comma-separated ids allowed to search **documents**
- `GATED_RECALL_MAX_CHARS` — cap on the injected block (default 2000)

The library itself has no production token.

---

## 5. Using this with an AI harness (Cursor / Claude Desktop / MCP)

A **harness** is the program that runs the model and its tools. It does
**not** magically import this folder. Keep the kernel in **your daemon**.
The chat model only *sees* the string you append.

There is **no** `examples/mcp_server.py` on purpose. A `memory_search` MCP
tool that runs on every turn *is* always-on RAG plus tool tokens. Optional
inspect-only: print `RecallMonitor.as_log()` (ids and counts).

Paste-ready policy:

> Call build_inject on the primary lane only. Greetings and evaluator
> channels must return empty. Empty user_ids must return empty. On
> small-model fallback, strip_recall_blocks (or prepare_user_turn with
> lane=fallback) so leftover SEMANTIC MEMORY RECALL does not enter the
> small KV cache. Do not hybrid-RAG the reviewer. Log ids, never snippets,
> on the monitor line.

---

## 6. Using this with a locally built AI (no MCP)

Your worker already has a store. This kernel does not replace it.

```python
from gated_recall import Lane, RecallMonitor, build_inject, prepare_user_turn

mon = RecallMonitor()
block = build_inject(
    user_text,
    lane=Lane.PRIMARY,           # Lane.FALLBACK / EVALUATOR skip
    user_ids=(canonical_user,),  # empty tuple → no hits
    canonical_user=canonical_user,
    operator_ids=("forge",),
    hybrid_messages=my_hybrid_search,      # (query, ids) -> list[Hit]
    hybrid_documents=my_doc_search,        # optional
    monitor=mon,
)
print(mon.as_log(), flush=True)  # stderr in production
user_turn = prepare_user_turn(user_text, user_text, lane=current_lane, **same)
```

On failover to a small local model, pass `lane=Lane.FALLBACK` so any block
from the large window is stripped.

Recipes: [`docs/INTEGRATION.md`](docs/INTEGRATION.md).

---

## 7. Practice drills (do these once)

1. Run `./scripts/smoke.sh`. Confirm greeting inject is 0 characters.
2. Confirm `user_ids=[]` does not call your search lambda (see tests).
3. Fuse two `Hit`s with the same `message_id` and different snippets —
   you should get **one** fused row.
4. Read [`docs/ADVANCED.md`](docs/ADVANCED.md) on fallback strip.
5. Name one comparable that always injects (Mem0 `search` in the prompt,
   LangChain checkpointer history, sqlite-vec notebook RRF with no gate).

---

## 8. When something is wrong

| Symptom | Try |
| ------- | --- |
| `No module named gated_recall` | Run `./scripts/smoke.sh` from *this* folder (it sets PYTHONPATH), or `pip install -e .` |
| `Permission denied` on smoke.sh | `chmod +x scripts/smoke.sh` |
| Greeting still injects | Gate is broken or you called `hybrid_search` yourself before `build_inject` |
| Guest sees documents | `canonical_user` is in `operator_ids`, or you skipped `build_inject` |
| Small model slow / OOM after cloud failover | You injected on primary and did not `strip_recall_blocks` |
| Two hits that are the same message | You fused on snippet prefixes. Pass `message_id` / `rowid` |

---

## 9. What not to do

- Do not always-inject “so the demo looks informed.”
- Do not treat empty `user_ids` as “search everyone.”
- Do not fuse hybrid ranks on `snippet[:100]`.
- Do not hybrid-RAG the evaluator / systems review turn.
- Do not add an MCP tool that bypasses the gate.
- Do not commit secrets or live identity files.

**Risk to remember:** a `memory_search` tool on every turn is always-on
RAG with a longer invoice.

---

## 10. Where to go next

| Need | Open |
| ---- | ---- |
| Why this exists / hidden dynamics | [`README.md`](README.md) |
| Recipes for harness + custom AI | [`docs/INTEGRATION.md`](docs/INTEGRATION.md) |
| Advanced / search tutorials | [`docs/ADVANCED.md`](docs/ADVANCED.md) |

You are done with first use when smoke prints `smoke ok` and you can say
in one sentence whether **your** agent is a harness, a custom loop, or
both — and that retrieval is 0-or-capped, not always-on.
