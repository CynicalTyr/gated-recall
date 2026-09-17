# Contributing

1. Keep the public tree free of secrets, LAN IPs, and live operator paths.
2. Add or extend a test under `tests/` for behavior changes.
3. Run:

```bash
python3 -m unittest discover -s tests -q
```

4. Do not expand this kernel into a vector database. You still bring FTS
   and/or ANN. This package stays the **inject policy**: gate, identity
   empty-deny, id-keyed RRF, fallback strip, evaluator skip.

Issues: one problem per ticket. Feature ideas: say who it helps and the
60-second demo that would prove it.
