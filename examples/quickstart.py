#!/usr/bin/env python3
"""10-minute proof: greetings skip, recall injects, fallback strips, empty ids deny."""
from __future__ import annotations

from gated_recall import Hit, Lane, RecallMonitor, build_inject, prepare_user_turn

STORE = [
    Hit(
        snippet="we moved DNS to a local resolver last Tuesday",
        message_id=42,
        user_id="forge",
        timestamp="2026-01-06",
    )
]


def search_messages(query: str, user_ids: tuple[str, ...]) -> list[Hit]:
    if "forge" not in user_ids:
        return []
    return list(STORE)


def main() -> None:
    mon = RecallMonitor()
    greeting = build_inject(
        "hey",
        user_ids=["forge"],
        hybrid_messages=search_messages,
        monitor=mon,
    )
    print("greeting inject chars:", len(greeting), "skip:", mon.skipped)
    assert greeting == ""
    assert mon.skipped == "gate"

    recall = build_inject(
        "what did we decide about DNS",
        user_ids=["forge"],
        hybrid_messages=search_messages,
        monitor=mon,
    )
    print("recall inject chars:", len(recall), "msg_ids:", mon.msg_ids)
    print(mon.as_log())
    assert "SEMANTIC MEMORY RECALL" in recall
    assert "42" in mon.as_log()
    assert "local resolver" not in mon.as_log()  # ids only, no snippet dump

    empty = build_inject(
        "what did we decide about DNS",
        user_ids=[],
        hybrid_messages=search_messages,
        monitor=mon,
    )
    print("empty allowlist chars:", len(empty), "skip:", mon.skipped)
    assert empty == ""

    stuffed = "what broke?\n" + recall
    stripped = prepare_user_turn(stuffed, "what broke?", lane=Lane.FALLBACK)
    print("fallback stripped recall:", "SEMANTIC MEMORY RECALL" not in stripped)
    assert "SEMANTIC MEMORY RECALL" not in stripped
    print("quickstart ok")


if __name__ == "__main__":
    main()
