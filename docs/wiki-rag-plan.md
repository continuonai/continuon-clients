# Wiki RAG Plan (Pi5 → Continuon Brain runtime)

Scope: how to use an offline Wikipedia slice as a retrieval source for the Pi5 runtime (no pretraining), feed it to the planner, and keep costs/offline posture intact.

## Goals
- Offline-first: serve answers from local wiki shards and episodic memory by default.
- Bounded: controllable disk/memory footprint with explicit shard manifests.
- Tool-logged: retrieval/tool calls land in RLDS `step_metadata` for replay and cloud distillation.
- Safe: remote API only as a last resort, budget-gated by environment variables.

## Shard format (edge)
- Layout (under `/opt/continuonos/brain/memory/wiki` by default):
  - `manifest.json` — list of shards, each with `id`, `kind` (`wiki` or `episodic`), `dim`, `size`, `embedding_file`, `metadata_file`, `license`, optional `domain`/`language`.
  - `shard-id/embeddings.npy` — float32 matrix `[num_docs, dim]`, L2-normalized rows.
  - `shard-id/metadata.jsonl` — one JSON per line: `{"id": "...", "title": "...", "url": "...", "text": "...", "license": "CC-BY-SA-3.0", "domain": "household"}`.
- Manifest example:
```json
{
  "version": "2025-12-11",
  "shards": [
    {
      "id": "wiki-simple-household-v1",
      "kind": "wiki",
      "dim": 384,
      "size": 120000,
      "embedding_file": "wiki-simple-household-v1/embeddings.npy",
      "metadata_file": "wiki-simple-household-v1/metadata.jsonl",
      "license": "CC-BY-SA-3.0",
      "language": "en",
      "domain": "household"
    }
  ]
}
```

## Build pipeline (off-device)
1) Filter wiki: pick domains (e.g., household items/tools/procedures) or Simple English. Source: [wikimedia/wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia).
2) Chunk: 512–1024 tokens per chunk; drop references/boilerplate.
3) Embed: small encoder (<=384–512 dim) that can run on HAT/CPU for live queries; L2-normalize.
4) Write shard: `embeddings.npy` + `metadata.jsonl`; compute manifest entry with `size`, `dim`, `license`.
5) Sign & ship: copy shard dir + `manifest.json` to `/opt/continuonos/brain/memory/wiki`; keep checksums if available.

## On-device usage
- Loader: `continuonbrain.services.librarian.Librarian` consumes `manifest.json`, lazy-loads shards, and scores cosine similarity.
- Inputs: caller supplies an embedding vector for the query (use the same encoder as the shard). If the HAT encoder is unavailable, fall back to CPU.
- Retrieval contract: returns a list of `{score, metadata}`; caller logs selections into RLDS `step_metadata.retrievals`.
- Episodic memory: store the same format under `/opt/continuonos/brain/memory/episodic` with `kind: "episodic"`. Librarian can load both.

## Planner integration
- Prompt slots: (vision detection + pose) + (top-k wiki facts) + (top-k episodic recalls) + (goal/context) + (safety summary).
- Fallback order: local SLM → local wiki/episodic RAG → remote API (budget-gated). Remote calls log tool traces.
- Safety: RAG cannot issue motor commands; it only supplies facts/context to the planner. Fast-loop safety clamps remain authoritative.

## Remote “unknown-answer” agent (optional)
- Env gates:
  - `ALLOW_REMOTE_AGENT=1`
  - `MAX_API_CENTS_PER_DAY` (integer; required to enable)
- Behavior: only call remote after local SLM + RAG fail or confidence < threshold; log invocation + cost estimate into `step_metadata.tool_calls`.

## Recommended default paths (edge)
- Wiki manifest: `/opt/continuonos/brain/memory/wiki/manifest.json`
- Episodic manifest (optional): `/opt/continuonos/brain/memory/episodic/manifest.json`
- Cache budget: keep in-RAM cache ≤64–128 MB for small Pi runs; stream metadata from disk when possible.

