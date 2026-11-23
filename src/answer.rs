//! Grounded, extractive answers with abstention.
//!
//! The answer is built only from sentences in permitted chunks. Each sentence carries a
//! `[doc_id#chunk]` citation, and a final check confirms every cited sentence is a verbatim
//! substring of its chunk. If the best evidence is weak (low term coverage or no lexical
//! match), the system abstains instead of guessing.

use crate::{acl, parse, search, Doc, Principal};
use serde_json::{json, Value};

pub fn answer_op(i: &Value) -> Result<Value, String> {
    let docs: Vec<Doc> = parse(i, "docs")?;
    let who: Principal = parse(i, "principal")?;
    let q = i["query"].as_str().ok_or("missing `query`")?;
    let min_cov = i["min_term_coverage"].as_f64().unwrap_or(0.5);
    let max_sent = i["max_sentences"].as_u64().unwrap_or(2) as usize;
    let visible = acl::visible(&docs, &who, &[]);
    let hits = search::hybrid(&visible, q, 4);
    let best_cov = hits
        .iter()
        .map(|h| h["term_coverage"].as_f64().unwrap_or(0.0))
        .fold(0.0, f64::max);
    let best_bm25 = hits
        .iter()
        .map(|h| h["bm25"].as_f64().unwrap_or(0.0))
        .fold(0.0, f64::max);
    if hits.is_empty() || best_cov < min_cov || best_bm25 <= 0.0 {
        return Ok(json!({
            "abstained": true,
            "answer": "I can't answer that from the documents you have access to.",
            "reason": format!("best term coverage {:.2} below {min_cov:.2} or no lexical match", best_cov),
            "citations": []
        }));
    }
    // Score each sentence of the top chunks by query-term overlap; keep the best ones.
    let qt = search::tokens(q);
    let mut cands: Vec<(f64, String, String, String)> = Vec::new();
    for h in &hits {
        let text = h["text"].as_str().unwrap_or("");
        let cite = format!("{}#{}", h["doc_id"].as_str().unwrap_or(""), h["chunk"]);
        for s in search::sentences(text) {
            let st = search::tokens(&s);
            let overlap = qt.iter().filter(|t| st.contains(t)).count() as f64;
            if overlap > 0.0 {
                cands.push((
                    overlap + h["rrf"].as_f64().unwrap_or(0.0),
                    s,
                    cite.clone(),
                    text.to_string(),
                ));
            }
        }
    }
    cands.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    let picked: Vec<_> = cands.into_iter().take(max_sent).collect();
    let grounded = picked
        .iter()
        .all(|(_, s, _, chunk)| chunk.contains(s.as_str()));
    let answer = picked
        .iter()
        .map(|(_, s, c, _)| format!("{s} [{c}]"))
        .collect::<Vec<_>>()
        .join(" ");
    let citations: Vec<Value> = picked
        .iter()
        .map(|(_, s, c, _)| json!({"cite": c, "quote": s}))
        .collect();
    Ok(json!({"abstained": false, "answer": answer, "citations": citations, "grounded": grounded}))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn corpus() -> Value {
        json!([
            {"id": "pol-1", "tenant": "acme", "title": "Travel policy", "acl": {"tenant_wide": true},
             "text": "Economy class is required for flights under 6 hours. Hotel limit is 180 USD per night."},
            {"id": "hr-9", "tenant": "acme", "title": "Exec comp", "acl": {"groups": ["exec"]},
             "text": "The CEO hotel limit is 900 USD per night."}
        ])
    }
    fn ask(q: &str, groups: Value) -> Value {
        answer_op(&json!({"docs": corpus(), "principal": {"user": "bo", "tenant": "acme", "groups": groups}, "query": q})).unwrap()
    }
    #[test]
    fn cites_and_is_grounded() {
        let r = ask("hotel limit per night", json!([]));
        assert_eq!(r["abstained"], false);
        assert_eq!(r["grounded"], true);
        assert!(r["answer"].as_str().unwrap().contains("[pol-1#0]"));
        assert!(!r["answer"].as_str().unwrap().contains("900"));
    }
    #[test]
    fn abstains_on_unknown_topic() {
        assert_eq!(ask("parental leave weeks", json!([]))["abstained"], true);
    }
    #[test]
    fn exec_sees_exec_doc() {
        assert!(ask("CEO hotel limit", json!(["exec"]))["answer"]
            .as_str()
            .unwrap()
            .contains("900"));
    }
}
