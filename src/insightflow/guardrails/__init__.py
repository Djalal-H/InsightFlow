"""Prompt-injection guardrails: layered controls, none of which is a boundary by itself.

- `text`: remove invisible and tag characters that can hide instructions.
- `screen`: a heuristic screen that names suspicious patterns (a signal for logs and policy).
- `spotlight`: mark untrusted text (passages, web pages, tool output) in prompts as data.
- `output`: check the final answer (system-prompt canary, exfiltration links, secrets, size).
- `classifier`: an optional hosted content-safety model.

The architectural controls live elsewhere: consequential tools need a scope and a human
approval (`agents/nodes/approve.py`), and the rate limiter bounds repeated attempts.
"""
