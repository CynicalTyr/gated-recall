"""Offline tests for gated hybrid recall. No network, no live host."""
from __future__ import annotations

import unittest

from gated_recall import (
    Hit,
    Lane,
    RecallMonitor,
    build_inject,
    identity_allowlist,
    prepare_user_turn,
    reciprocal_rank_fusion,
    rrf_key,
    should_retrieve,
    strip_recall_blocks,
)


class GateTests(unittest.TestCase):
    def test_greeting_skips(self) -> None:
        self.assertFalse(should_retrieve("hey"))
        self.assertFalse(should_retrieve("thanks!"))

    def test_recall_shaped_passes(self) -> None:
        self.assertTrue(should_retrieve("what did we decide about DNS"))
        self.assertTrue(should_retrieve("you said the NAS was unmounted"))

    def test_evaluator_channel_skips(self) -> None:
        self.assertFalse(
            should_retrieve("what did we decide", channel="systems_review")
        )

    def test_isolated_skips(self) -> None:
        self.assertFalse(should_retrieve("what did we decide", isolated=True))


class AllowlistTests(unittest.TestCase):
    def test_empty_is_deny(self) -> None:
        self.assertEqual(identity_allowlist(None), ())
        self.assertEqual(identity_allowlist([]), ())
        self.assertEqual(identity_allowlist(["", "  "]), ())

    def test_named_ids(self) -> None:
        self.assertEqual(identity_allowlist(["forge", "operator"]), ("forge", "operator"))


class RrfTests(unittest.TestCase):
    def test_same_id_merges(self) -> None:
        sem = [Hit(snippet="alpha prefix different", message_id=42, rank=0)]
        lex = [Hit(snippet="beta prefix different", message_id=42, rank=0)]
        fused = reciprocal_rank_fusion(sem, lex, top_k=5)
        self.assertEqual(len(fused), 1)
        self.assertEqual(rrf_key(fused[0]), ("message_id", 42))

    def test_different_ids_stay_two(self) -> None:
        sem = [Hit(snippet="same snippet prefixxxxxx", message_id=1)]
        lex = [Hit(snippet="same snippet prefixxxxxx", message_id=2)]
        fused = reciprocal_rank_fusion(sem, lex, top_k=5)
        self.assertEqual(len(fused), 2)


class StripTests(unittest.TestCase):
    def test_strips_recall_keeps_question(self) -> None:
        turn = (
            "what broke last night?\n"
            "=== SEMANTIC MEMORY RECALL (1 matches for: \"what broke\") ===\n"
            "[1] [t] [forge] the NAS unmounted\n"
        )
        out = strip_recall_blocks(turn)
        self.assertIn("what broke last night?", out)
        self.assertNotIn("SEMANTIC MEMORY RECALL", out)
        self.assertNotIn("NAS unmounted", out)


class InjectTests(unittest.TestCase):
    def setUp(self) -> None:
        self.msg = [
            Hit(snippet="we fixed nginx", message_id=42, user_id="forge", timestamp="2026-01-01")
        ]
        self.doc = [
            Hit(
                snippet="deploy notes",
                chunk_id=99,
                filename="notes.txt",
                user_id="forge",
                timestamp="2026-01-02",
            )
        ]

    def test_primary_recall_injects_messages(self) -> None:
        mon = RecallMonitor()
        out = build_inject(
            "what did we decide",
            lane=Lane.PRIMARY,
            user_ids=["forge"],
            hybrid_messages=lambda q, ids: self.msg,
            monitor=mon,
        )
        self.assertIn("SEMANTIC MEMORY RECALL", out)
        self.assertIn("we fixed nginx", out)
        self.assertEqual(mon.msg_ids, [42])
        self.assertEqual(mon.skipped, "")

    def test_greeting_no_hybrid_call(self) -> None:
        calls = {"n": 0}

        def hybrid(q: str, ids: tuple[str, ...]):
            calls["n"] += 1
            return self.msg

        out = build_inject(
            "hey",
            lane=Lane.PRIMARY,
            user_ids=["forge"],
            hybrid_messages=hybrid,
        )
        self.assertEqual(out, "")
        self.assertEqual(calls["n"], 0)

    def test_fallback_never_injects(self) -> None:
        out = build_inject(
            "what did we decide",
            lane=Lane.FALLBACK,
            user_ids=["forge"],
            hybrid_messages=lambda q, ids: self.msg,
        )
        self.assertEqual(out, "")

    def test_evaluator_never_injects(self) -> None:
        out = build_inject(
            "what did we decide",
            lane=Lane.EVALUATOR,
            user_ids=["forge"],
            hybrid_messages=lambda q, ids: self.msg,
        )
        self.assertEqual(out, "")

    def test_empty_allowlist_no_search(self) -> None:
        calls = {"n": 0}

        def hybrid(q: str, ids: tuple[str, ...]):
            calls["n"] += 1
            return self.msg

        out = build_inject(
            "what did we decide",
            user_ids=[],
            hybrid_messages=hybrid,
        )
        self.assertEqual(out, "")
        self.assertEqual(calls["n"], 0)

    def test_guest_skips_documents(self) -> None:
        doc_calls = {"n": 0}

        def docs(q: str, ids: tuple[str, ...]):
            doc_calls["n"] += 1
            return self.doc

        out = build_inject(
            "what did we decide",
            user_ids=["guest"],
            canonical_user="guest",
            operator_ids=("forge",),
            hybrid_messages=lambda q, ids: self.msg,
            hybrid_documents=docs,
        )
        self.assertIn("nginx", out)
        self.assertNotIn("notes.txt", out)
        self.assertEqual(doc_calls["n"], 0)

    def test_operator_gets_documents(self) -> None:
        out = build_inject(
            "what did we decide",
            user_ids=["forge"],
            canonical_user="forge",
            operator_ids=("forge",),
            hybrid_messages=lambda q, ids: self.msg,
            hybrid_documents=lambda q, ids: self.doc,
        )
        self.assertIn("notes.txt", out)

    def test_prepare_fallback_strips(self) -> None:
        stuffed = (
            "what broke?\n"
            "=== SEMANTIC MEMORY RECALL (1 matches for: \"x\") ===\n"
            "[1] secret tenant row\n"
        )
        out = prepare_user_turn(stuffed, "what broke?", lane=Lane.FALLBACK)
        self.assertNotIn("secret tenant row", out)
        self.assertIn("what broke?", out)

    def test_char_cap(self) -> None:
        long_hit = [Hit(snippet="x" * 5000, message_id=1)]
        out = build_inject(
            "what did we decide",
            user_ids=["forge"],
            max_chars=80,
            hybrid_messages=lambda q, ids: long_hit,
        )
        self.assertIn("truncated for recall cap", out)
        self.assertLessEqual(len(out), 80 + 80)


if __name__ == "__main__":
    unittest.main()
