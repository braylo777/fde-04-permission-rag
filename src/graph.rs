//! GraphRAG-style multi-hop traversal over (subject, relation, object) triples.
//! Every triple names its source document; a triple is usable only if the principal can see
//! that document. Breadth-first search returns the shortest permitted path, with citations.

use crate::{acl, parse, Doc, Principal};
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet, VecDeque};

#[derive(Debug, Clone, Deserialize)]
pub struct Triple {
    pub s: String,
    pub r: String,
    pub o: String,
    pub doc: String,
}

pub fn paths(
    triples: &[Triple],
    start: &str,
    goal: Option<&str>,
    max_hops: usize,
) -> Vec<Vec<Triple>> {
    let norm = |x: &str| x.to_lowercase();
    let mut adj: HashMap<String, Vec<(&Triple, String)>> = HashMap::new();
    for t in triples {
        adj.entry(norm(&t.s)).or_default().push((t, norm(&t.o)));
        adj.entry(norm(&t.o)).or_default().push((t, norm(&t.s))); // traverse both directions
    }
    let mut out = Vec::new();
    let mut q: VecDeque<(String, Vec<Triple>)> = VecDeque::from([(norm(start), vec![])]);
    let mut seen: HashSet<String> = HashSet::from([norm(start)]);
    while let Some((node, path)) = q.pop_front() {
        if path.len() >= max_hops {
            continue;
        }
        for (t, next) in adj.get(&node).into_iter().flatten() {
            if seen.contains(next) {
                continue;
            }
            seen.insert(next.clone());
            let mut p = path.clone();
            p.push((*t).clone());
            match goal {
                Some(g) if norm(g) == *next => return vec![p],
                Some(_) => {}
                None => out.push(p.clone()),
            }
            q.push_back((next.clone(), p));
        }
    }
    out
}

pub fn query_op(i: &Value) -> Result<Value, String> {
    let docs: Vec<Doc> = parse(i, "docs")?;
    let who: Principal = parse(i, "principal")?;
    let triples: Vec<Triple> = parse(i, "triples")?;
    let start = i["start"].as_str().ok_or("missing `start`")?;
    let goal = i["goal"].as_str();
    let hops = i["max_hops"].as_u64().unwrap_or(3) as usize;
    let ok_docs: HashSet<String> = acl::visible(&docs, &who, &[])
        .into_iter()
        .map(|d| d.id)
        .collect();
    let permitted: Vec<Triple> = triples
        .iter()
        .filter(|t| ok_docs.contains(&t.doc))
        .cloned()
        .collect();
    let hidden = triples.len() - permitted.len();
    let found = paths(&permitted, start, goal, hops);
    let fmt: Vec<Value> = found
        .iter()
        .map(|p| {
            json!({
                "hops": p.len(),
                "path": p.iter().map(|t| format!("{} -[{}]-> {}", t.s, t.r, t.o)).collect::<Vec<_>>(),
                "citations": p.iter().map(|t| t.doc.clone()).collect::<Vec<_>>()
            })
        })
        .collect();
    let abstain = goal.is_some() && fmt.is_empty();
    Ok(
        json!({"paths": fmt, "edges_hidden_by_acl": hidden, "abstained": abstain,
        "note": if abstain { "no permitted path within max_hops" } else { "" }}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    fn t(s: &str, r: &str, o: &str, d: &str) -> Triple {
        Triple {
            s: s.into(),
            r: r.into(),
            o: o.into(),
            doc: d.into(),
        }
    }
    #[test]
    fn two_hop_path() {
        let ts = vec![
            t("Acme", "acquired", "Bolt Ltd", "d1"),
            t("Bolt Ltd", "supplies", "Contoso", "d2"),
        ];
        let p = paths(&ts, "acme", Some("contoso"), 3);
        assert_eq!(p[0].len(), 2);
    }
    #[test]
    fn hop_limit() {
        let ts = vec![
            t("A", "x", "B", "d"),
            t("B", "x", "C", "d"),
            t("C", "x", "D", "d"),
        ];
        assert!(paths(&ts, "A", Some("D"), 2).is_empty());
        assert_eq!(paths(&ts, "A", Some("D"), 3)[0].len(), 3);
    }
    #[test]
    fn acl_cuts_the_path() {
        let i = json!({
            "docs": [{"id": "d1", "tenant": "t", "title": "", "text": "", "acl": {"tenant_wide": true}},
                     {"id": "d2", "tenant": "t", "title": "", "text": "", "acl": {"groups": ["legal"]}}],
            "principal": {"user": "u", "tenant": "t"},
            "triples": [{"s": "A", "r": "x", "o": "B", "doc": "d1"}, {"s": "B", "r": "y", "o": "C", "doc": "d2"}],
            "start": "A", "goal": "C"});
        let r = query_op(&i).unwrap();
        assert_eq!(r["abstained"], true);
        assert_eq!(r["edges_hidden_by_acl"], 1);
    }
}
