# Log and Output Index

Root-level runtime and status logs were pruned to keep the repository reproducible and avoid bundling transient environment data.
Only two curated examples remain under `docs/examples/` for reference:

- `docs/examples/DebugProdFinal.log` – DepthAI/OAK-D boot warnings and token-gated model notice from a failed production start.
- `docs/examples/status_debug.txt` – Systemd status excerpt illustrating the ContinuonBrain startup service output format.

All other `Debug*.log`, `server*.log`, `start_log*.txt`, `status*.txt`, and similar run artifacts should be ignored going forward to
prevent noisy commits.
