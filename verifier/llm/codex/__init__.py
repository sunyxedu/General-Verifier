"""Verifier seats on the Codex CLI.

``runtime`` is the Codex runtime; the modules it composes each own one part
of a call: ``command`` (instruction and command line), ``policy`` (the
seat's CLI boundary), ``paths`` (operator-bound directories), ``images``
(image inputs), ``process`` (the launched process group), ``attempts``
(the attempts of one call), ``failures`` (transient-failure
classification), ``stream`` (event-stream parsing), ``usage`` (the usage
ledger), ``diagnostics`` (content-free stream statistics) and
``service_tier``.
"""
