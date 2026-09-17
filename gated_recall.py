"""Gated hybrid recall: inject when needed, never on small-model fallback.

Stdlib only. You supply ranked lists (FTS, vectors, or both). This kernel
decides *whether* they enter the prompt, *who* they may cover, and *how*
they fuse. It is not a vector database.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable, Sequence

RECALL_MARKER = "=== SEMANTIC MEMORY RECALL"
DOCUMENT_MARKER = "=== DOCUMENT MEMORY RECALL"
LESSONS_MARKER = "=== RETRIEVED LESSONS (data, not proof) ==="

_RECALL_INTENT = (
    "what did we",
    "remember when",
    "last time",
    "previously",
    "earlier today",
    "what broke",
    "you said",
    "we decided",
)

_SHORT_CHATTER = (
    "hey",
    "hi",
    "hello",
    "thanks",
    "thank you",
    "thx",
    "ok",
    "okay",
    "yo",
    "sup",
    "gm",
    "cheers",
    "cool",
    "nice",
)


class Lane(str, Enum):
    """Where the next model call sits."""

    PRIMARY = "primary"
    FALLBACK = "fallback"
    EVALUATOR = "evaluator"


@dataclass(frozen=True)
class Hit:
    """One retrieved row. Stable ids beat snippet prefixes for fusion."""

    snippet: str
    rowid: int | None = None
    message_id: int | None = None
    chunk_id: int | None = None
    user_id: str = ""
    timestamp: str = ""
    filename: str = ""
    rank: int = 0


@dataclass
class RecallMonitor:
    """Last inject, for an agent or operator to inspect — not a verdict."""

    skipped: str = ""
    lane: str = ""
    chars: int = 0
    msg_ids: list[Any] = field(default_factory=list)
    doc_ids: list[Any] = field(default_factory=list)

    def as_log(self) -> str:
        if self.skipped:
            return f"[recall] skip={self.skipped} lane={self.lane}"
        return (
            f"[recall] lane={self.lane} chars={self.chars} "
            f"msg_ids={self.msg_ids} doc_ids={self.doc_ids}"
        )


def identity_allowlist(user_ids: Sequence[str] | None) -> tuple[str, ...]:
    """Fail closed: missing or empty ids retrieve nothing.

    A forgotten filter that returns every tenant is how guests see
    operator documents. Empty is deny, not 'search the world.'
    """
    if not user_ids:
        return ()
    out = tuple(u.strip() for u in user_ids if u and str(u).strip())
    return out


def should_retrieve(
    text: str,
    *,
    channel: str = "",
    isolated: bool = False,
    is_system_task: bool = False,
) -> bool:
    """True only for recall-shaped turns on a companion channel.

    Greetings, isolated personas, system tasks, and evaluator channels
    skip. The model still has a bounded lexical tail if *you* inject one;
    this gate is for *retrieved passages*, not the last 80 lines.
    """
    if not text or not str(text).strip():
        return False
    if isolated or is_system_task:
        return False
    ch = (channel or "").strip().lower()
    if ch in {"systems_review", "evaluator", "review"}:
        return False
    lower = str(text).lower()
    if any(p in lower for p in _RECALL_INTENT):
        return True
    t = lower.strip().rstrip("!.?,")
    if not t:
        return False
    if len(t) <= 48:
        for prefix in _SHORT_CHATTER:
            if t == prefix or t.startswith(prefix + " ") or t.startswith(prefix + "!"):
                return False
    return False


def rrf_key(hit: Hit | dict) -> tuple[str, Any]:
    """Stable fusion key: integer ids first, snippet prefix only as fallback."""
    if isinstance(hit, Hit):
        for field_name in ("rowid", "message_id", "chunk_id"):
            val = getattr(hit, field_name)
            if val is not None:
                return (field_name, val)
        return ("snippet", (hit.snippet or "")[:100])
    for field_name in ("rowid", "message_id", "chunk_id"):
        if field_name in hit and hit[field_name] is not None:
            return (field_name, hit[field_name])
    return ("snippet", (hit.get("snippet") or "")[:100])


def reciprocal_rank_fusion(
    semantic: Sequence[Hit | dict],
    lexical: Sequence[Hit | dict],
    *,
    k: int = 60,
    top_k: int = 5,
) -> list[Hit | dict]:
    """Merge two ranked lists with RRF. Same id from FTS and vectors is one hit.

    Fusing on snippet[:100] splits one message that paraphrases itself and
    merges two messages that share a prefix. Ids are the product.
    """
    scores: dict[tuple[str, Any], float] = {}
    data: dict[tuple[str, Any], Hit | dict] = {}
    for rank, row in enumerate(semantic):
        key = rrf_key(row)
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        data.setdefault(key, row)
    for rank, row in enumerate(lexical):
        key = rrf_key(row)
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        data.setdefault(key, row)
    ordered = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [data[key] for key in ordered[:top_k]]


def format_recall(hits: Sequence[Hit | dict], query: str) -> str:
    if not hits:
        return ""
    lines = [f'{RECALL_MARKER} ({len(hits)} matches for: "{query[:80]}") ===']
    for i, raw in enumerate(hits, 1):
        if isinstance(raw, Hit):
            lines.append(
                f"[{i}] [{raw.timestamp}] [{raw.user_id}] {raw.snippet}"
            )
        else:
            lines.append(
                f"[{i}] [{raw.get('timestamp', '')}] [{raw.get('user_id', '')}] "
                f"{raw.get('snippet', '')}"
            )
    return "\n".join(lines) + "\n"


def format_documents(hits: Sequence[Hit | dict], query: str) -> str:
    if not hits:
        return ""
    lines = [f'{DOCUMENT_MARKER} ({len(hits)} matches for: "{query[:80]}") ===']
    for i, raw in enumerate(hits, 1):
        if isinstance(raw, Hit):
            name = raw.filename or "document"
            lines.append(
                f"[{i}] [{raw.timestamp}] [{raw.user_id}] [{name}] {raw.snippet}"
            )
        else:
            name = raw.get("filename") or "document"
            lines.append(
                f"[{i}] [{raw.get('timestamp', '')}] [{raw.get('user_id', '')}] "
                f"[{name}] {raw.get('snippet', '')}"
            )
    return "\n".join(lines) + "\n"


def strip_recall_blocks(user_turn: str) -> str:
    """Remove baked-in recall from a turn before a small-model fallback.

    If you retrieved on the large window then failover, the block is already
    in the user message. Leaving it there fills an 8k KV cache with hits
    the small model cannot use.
    """
    if not user_turn:
        return ""
    out: list[str] = []
    skipping = False
    for line in user_turn.splitlines(keepends=True):
        stripped = line.lstrip()
        if stripped.startswith(RECALL_MARKER) or stripped.startswith(DOCUMENT_MARKER):
            skipping = True
            continue
        if skipping:
            if stripped.startswith("===") and not stripped.startswith(RECALL_MARKER):
                skipping = False
                out.append(line)
            continue
        out.append(line)
    return "".join(out).rstrip() + ("\n" if user_turn.endswith("\n") else "")


def _hit_id(row: Hit | dict) -> Any:
    if isinstance(row, Hit):
        return row.message_id or row.rowid or row.chunk_id
    return row.get("message_id") or row.get("rowid") or row.get("chunk_id")


def _doc_id(row: Hit | dict) -> Any:
    if isinstance(row, Hit):
        return row.chunk_id or row.rowid
    return row.get("chunk_id") or row.get("rowid")


def build_inject(
    prompt: str,
    *,
    lane: Lane | str = Lane.PRIMARY,
    channel: str = "",
    isolated: bool = False,
    is_system_task: bool = False,
    user_ids: Sequence[str] | None = None,
    operator_ids: Sequence[str] = (),
    canonical_user: str = "",
    max_chars: int = 2000,
    hybrid_messages: Callable[[str, tuple[str, ...]], Sequence[Hit | dict]] | None = None,
    hybrid_documents: Callable[[str, tuple[str, ...]], Sequence[Hit | dict]] | None = None,
    monitor: RecallMonitor | None = None,
) -> str:
    """Return a recall block or empty string. Never raises on empty store.

    ``hybrid_messages`` / ``hybrid_documents`` are *your* search callables.
    This function does not talk to a database.
    """
    mon = monitor if monitor is not None else RecallMonitor()
    lane_v = Lane(lane) if not isinstance(lane, Lane) else lane
    mon.lane = lane_v.value
    mon.skipped = ""
    mon.chars = 0
    mon.msg_ids = []
    mon.doc_ids = []

    if lane_v == Lane.EVALUATOR:
        mon.skipped = "evaluator"
        return ""
    if lane_v == Lane.FALLBACK:
        mon.skipped = "fallback"
        return ""

    if not should_retrieve(
        prompt,
        channel=channel,
        isolated=isolated,
        is_system_task=is_system_task,
    ):
        mon.skipped = "gate"
        return ""

    allow = identity_allowlist(user_ids)
    if not allow:
        mon.skipped = "empty_allowlist"
        return ""

    parts: list[str] = []
    msg_hits: Sequence[Hit | dict] = ()
    if hybrid_messages is not None:
        msg_hits = hybrid_messages(prompt, allow) or ()
        block = format_recall(msg_hits, prompt)
        if block:
            parts.append(block)
    doc_hits: Sequence[Hit | dict] = ()
    op = {u.strip().lower() for u in operator_ids if u.strip()}
    if (
        hybrid_documents is not None
        and canonical_user.strip().lower() in op
    ):
        doc_hits = hybrid_documents(prompt, allow) or ()
        block = format_documents(doc_hits, prompt)
        if block:
            parts.append(block)

    if not parts:
        mon.skipped = "no_hits"
        return ""
    ctx = "\n".join(parts)
    if len(ctx) > max_chars:
        ctx = ctx[:max_chars] + "\n[... truncated for recall cap ...]\n"
    mon.chars = len(ctx)
    mon.msg_ids = [_hit_id(h) for h in msg_hits if _hit_id(h) is not None]
    mon.doc_ids = [_doc_id(h) for h in doc_hits if _doc_id(h) is not None]
    return ctx


def prepare_user_turn(
    user_turn: str,
    prompt: str,
    **kwargs: Any,
) -> str:
    """Primary: maybe append inject. Fallback: strip any existing recall block."""
    lane = kwargs.get("lane", Lane.PRIMARY)
    lane_v = Lane(lane) if not isinstance(lane, Lane) else lane
    if lane_v == Lane.FALLBACK:
        return strip_recall_blocks(user_turn)
    extra = build_inject(prompt, **kwargs)
    if not extra:
        return user_turn
    if extra in user_turn:
        return user_turn
    sep = "" if user_turn.endswith("\n") else "\n"
    return f"{user_turn}{sep}{extra}"
