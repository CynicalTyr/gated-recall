# Security

- Never open issues that paste live API keys, cookies, or `.env` values.
- This project is a **library**, not a hosted memory service.
- An empty identity allowlist must return **no hits**. If you find a path
  that searches every tenant when `user_ids` is omitted, file a redacted
  repro. That is a privacy bug, not a convenience feature.
- Do not ask maintainers to always-inject RAG “for demo quality.” That is
  how a small-model fallback inherits a 2k recall block and OOMs.

Fail-open on a missing vector **cache** (see `cynicvec`) must not be copied
onto this allowlist. Identity is fail-closed.
