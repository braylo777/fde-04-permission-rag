//! Stage 4 — Permission-Filtered RAG & GraphRAG.
//!
//! * `acl`    — who may see a document: exact tenant match, then user / group ACL, then
//!   metadata filters. Applied *before* ranking, so a hidden document can never influence
//!   scores, snippets, counts or answers.
//! * `search` — chunking, BM25, hashed n-gram vectors, and Reciprocal Rank Fusion (hybrid).
//! * `answer` — extractive answer with a citation on every sentence, plus abstention when
//!   the evidence is weak.
//! * `graph`  — multi-hop traversal over (subject, relation, object) triples; each edge
//!   inherits the ACL of the document it came from.
//! * `leak_test` — run queries as one principal and prove nothing from another tenant or
//!   outside the ACL comes back.

mod api;
pub mod acl;
pub mod answer;
pub mod graph;
pub mod search;

pub use api::dispatch;
use serde::Deserialize;
use serde_json::{json, Value};

pub const OPS: &[&str] = &["search", "answer", "graph_query", "leak_test"];

#[derive(Debug, Clone, Deserialize)]
pub struct Doc {
    pub id: String,
    pub tenant: String,
    pub title: String,
    pub text: String,
    #[serde(default)]
    pub acl: acl::Acl,
    #[serde(default)]
    pub tags: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Principal {
    pub user: String,
    pub tenant: String,
    #[serde(default)]
    pub groups: Vec<String>,
}

pub(crate) fn parse<T: serde::de::DeserializeOwned>(v: &Value, k: &str) -> Result<T, String> {
    serde_json::from_value(v.get(k).cloned().ok_or(format!("missing field `{k}`"))?).map_err(|e| format!("`{k}`: {e}"))
}

pub fn handle(op: &str, i: &Value) -> Result<Value, String> {
    match op {
        "search" => {
            let docs: Vec<Doc> = parse(i, "docs")?;
            let who: Principal = parse(i, "principal")?;
            let q = i["query"].as_str().ok_or("missing `query`")?;
            let k = i["k"].as_u64().unwrap_or(5) as usize;
            let tags: Vec<String> = i["require_tags"].as_array().map(|a| a.iter().filter_map(|t| t.as_str().map(String::from)).collect()).unwrap_or_default();
            let visible = acl::visible(&docs, &who, &tags);
            let hits = search::hybrid(&visible, q, k);
            Ok(json!({"visible_docs": visible.len(), "hits": hits}))
        }
        "answer" => answer::answer_op(i),
        "graph_query" => graph::query_op(i),
        "leak_test" => acl::leak_test_op(i),
        _ => Err(format!("unknown op `{op}`; known: {}", OPS.join(", "))),
    }
}
