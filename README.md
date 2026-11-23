# fde-04-permission-rag

**Forward Deployed Engineering — Stage 4 of 12: Permission-Filtered RAG & GraphRAG**

Retrieval that checks permissions before it ranks: exact tenant isolation, document ACLs, hybrid search, a citation on every sentence, multi-hop graph paths, and an honest 'I don't know'.

**Live demo:** https://fde-04-permission-rag.braylo777.workers.dev — the Rust core runs in your browser as WebAssembly. No server, no tracking.

## Why this matters

The fastest way to lose an enterprise customer is a chatbot that quotes the CEO's salary to an intern, or another customer's data to a tenant. Filtering after retrieval is not enough: a hidden document can still shape scores, snippets and counts. This repo applies tenant and ACL checks before anything is ranked, grounds every answer sentence in a cited chunk, and abstains when the evidence is weak.

## What it does

- Pre-retrieval filter: exact tenant match, user / group ACL, explicit deny, tenant-wide flag that never crosses tenants, tag filters
- Hybrid search: sentence-aware chunking, BM25, hashed character-trigram vectors, Reciprocal Rank Fusion
- Extractive answers with a [doc#chunk] citation per sentence and a verbatim grounding check
- Abstention when term coverage or lexical evidence is below a threshold
- GraphRAG: shortest multi-hop path over triples; each edge inherits its source document's ACL
- Leak test: proves no out-of-scope document is returned and lists what the filter blocked

## What it does not do (yet)

- Hashed n-gram vectors stand in for a real embedding model; swap search::vector for your embeddings
- No LLM generation step: answers are extractive, which is what makes the grounding check exact
- No persistent index or vector DB; documents are passed in per request
- Triple extraction from text (entity / relation NER) is not included; triples are inputs

## Operations

Every operation takes JSON in and gives JSON out. The CLI, the tests and the demo page all use the same entry point.

| Operation | What it does |
|---|---|
| `search` | Bo (support, tenant acme) searches for hotel limits. The exec and Globex policies match the words but are filtered out before ranking: visible_docs shows only what Bo may see. |
| `answer` | Each sentence ends with its source. Change the principal's groups to ["exec"] and ask about the CEO hotel limit to see the answer change with access. |
| `graph_query` | How is Acme connected to Dana Park? The path uses two permitted documents. Try goal Globex: that edge comes from another tenant, so no path is returned. |
| `leak_test` | Runs queries as Bo. leaked must be empty; blocked_by_filter shows the documents that would have ranked without the permission filter. |

## Run it

You need Rust (stable). No Node, no npm.

```bash
git clone https://github.com/braylo777/fde-04-permission-rag
cd fde-04-permission-rag
cargo test
cargo run -- ops
```

Try the first demo example from the command line:

```bash
python3 -c 'import json;print(json.dumps(json.load(open("web/examples.json"))["ops"][0]["input"]))' | cargo run -q -- search
```

## Build and deploy the demo

```bash
./build.sh    # tests, then compiles the core to web/app.wasm
./deploy.sh   # serves web/ as a Cloudflare Workers static-assets site (needs a Cloudflare login)
```

`web/` is a plain static site (`index.html`, `app.js`, `app.wasm`). It has no build step, so any static host works.

## Layout

```
src/lib.rs        operation list and router
src/api.rs        JSON entry point + the 3 WASM exports (alloc, dealloc, run)
src/main.rs       CLI
src/*.rs          one module per capability, each with unit tests
tests/            integration tests (every demo example must pass)
web/              demo page; examples.json drives both the page and this README
```

## Design choices

- **Rust, few dependencies.** Only `serde`, `serde_json`, `sha2` and `hmac`, pinned to exact versions with a committed `Cargo.lock`.
- **Deterministic.** Time, randomness and keys are inputs, so every result can be replayed in a test or an audit.
- **Fail closed.** Unknown input, missing fields or an unmatched policy gives an error or a denial, never a silent pass.

## Part of a series

This is one of 12 repos that work through a public Forward Deployed Engineering roadmap ([source post](https://x.com/suraj_sharma14/status/2124827808535048566)). All 12: [github.com/braylo777?q=fde-](https://github.com/braylo777?tab=repositories&q=fde-).

## License

MIT — see [LICENSE](LICENSE). Built by [Braylo SI](https://braylo.si).
