//! Hybrid retrieval: BM25 (exact terms) + hashed character-trigram vectors (fuzzy match),
//! fused with Reciprocal Rank Fusion. The vector side is a stand-in for an embedding model:
//! swap `vector` for real embeddings and nothing else changes.

use crate::Doc;
use serde_json::{json, Value};
use std::collections::HashMap;

pub const STOP: &[&str] = &[
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "is", "are", "was", "what",
    "who", "how", "does", "do", "with", "by", "at", "it", "this", "that", "be", "our", "we",
    "which", "from", "as", "can", "i",
];

pub fn tokens(s: &str) -> Vec<String> {
    s.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty() && !STOP.contains(t))
        .map(|t| {
            t.strip_suffix('s')
                .filter(|r| r.len() > 3)
                .unwrap_or(t)
                .to_string()
        })
        .collect()
}

#[derive(Debug, Clone)]
pub struct Chunk {
    pub doc_id: String,
    pub title: String,
    pub idx: usize,
    pub text: String,
}

/// Split on sentence ends, then pack sentences into chunks of at most `max_words`.
pub fn chunk(d: &Doc, max_words: usize) -> Vec<Chunk> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for s in sentences(&d.text) {
        if !cur.is_empty()
            && cur.split_whitespace().count() + s.split_whitespace().count() > max_words
        {
            out.push(std::mem::take(&mut cur));
        }
        if !cur.is_empty() {
            cur.push(' ');
        }
        cur.push_str(&s);
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out.into_iter()
        .enumerate()
        .map(|(idx, text)| Chunk {
            doc_id: d.id.clone(),
            title: d.title.clone(),
            idx,
            text,
        })
        .collect()
}

pub fn sentences(t: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let chars: Vec<char> = t.chars().collect();
    for (i, &c) in chars.iter().enumerate() {
        cur.push(c);
        let end = matches!(c, '.' | '?' | '!')
            && chars.get(i + 1).map(|n| n.is_whitespace()).unwrap_or(true);
        let decimal = c == '.'
            && chars
                .get(i + 1)
                .map(|n| n.is_ascii_digit())
                .unwrap_or(false);
        if end && !decimal {
            out.push(cur.trim().to_string());
            cur.clear();
        }
    }
    if !cur.trim().is_empty() {
        out.push(cur.trim().to_string());
    }
    out
}

pub fn bm25(chunks: &[Chunk], q: &[String]) -> Vec<f64> {
    let (k1, b) = (1.2, 0.75);
    let toks: Vec<Vec<String>> = chunks
        .iter()
        .map(|c| tokens(&format!("{} {}", c.title, c.text)))
        .collect();
    let n = chunks.len() as f64;
    let avg = toks.iter().map(Vec::len).sum::<usize>() as f64 / n.max(1.0);
    let mut df: HashMap<&str, f64> = HashMap::new();
    for t in &toks {
        let mut seen: Vec<&str> = t.iter().map(String::as_str).collect();
        seen.sort();
        seen.dedup();
        for s in seen {
            *df.entry(s).or_default() += 1.0;
        }
    }
    toks.iter()
        .map(|t| {
            q.iter()
                .map(|term| {
                    let f = t.iter().filter(|x| *x == term).count() as f64;
                    if f == 0.0 {
                        return 0.0;
                    }
                    let d = df.get(term.as_str()).copied().unwrap_or(0.0);
                    let idf = ((n - d + 0.5) / (d + 0.5) + 1.0).ln();
                    idf * f * (k1 + 1.0) / (f + k1 * (1.0 - b + b * t.len() as f64 / avg))
                })
                .sum()
        })
        .collect()
}

const DIM: usize = 512;

fn fnv(s: &str) -> u64 {
    s.bytes().fold(0xcbf29ce484222325u64, |h, b| {
        (h ^ b as u64).wrapping_mul(0x100000001b3)
    })
}

pub fn vector(s: &str) -> Vec<f64> {
    let mut v = vec![0.0; DIM];
    for t in tokens(s) {
        let padded = format!("#{t}#");
        let cs: Vec<char> = padded.chars().collect();
        for w in cs.windows(3) {
            let g: String = w.iter().collect();
            v[(fnv(&g) % DIM as u64) as usize] += 1.0;
        }
    }
    let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 0.0 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
    v
}

pub fn cosine(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn ranks(scores: &[f64]) -> Vec<Option<usize>> {
    let mut idx: Vec<usize> = (0..scores.len()).filter(|&i| scores[i] > 0.0).collect();
    idx.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap().then(a.cmp(&b)));
    let mut r = vec![None; scores.len()];
    for (rank, i) in idx.into_iter().enumerate() {
        r[i] = Some(rank);
    }
    r
}

/// Hybrid search over already-permitted docs. RRF with k = 60.
pub fn hybrid(docs: &[Doc], query: &str, k: usize) -> Vec<Value> {
    let chunks: Vec<Chunk> = docs.iter().flat_map(|d| chunk(d, 60)).collect();
    if chunks.is_empty() {
        return vec![];
    }
    let q = tokens(query);
    let b = bm25(&chunks, &q);
    let qv = vector(query);
    let v: Vec<f64> = chunks
        .iter()
        .map(|c| cosine(&qv, &vector(&format!("{} {}", c.title, c.text))))
        .collect();
    let (rb, rv) = (ranks(&b), ranks(&v));
    let mut fused: Vec<(usize, f64)> = (0..chunks.len())
        .map(|i| {
            let s = rb[i].map(|r| 1.0 / (60.0 + r as f64 + 1.0)).unwrap_or(0.0)
                + rv[i].map(|r| 1.0 / (60.0 + r as f64 + 1.0)).unwrap_or(0.0);
            (i, s)
        })
        .filter(|(_, s)| *s > 0.0)
        .collect();
    fused.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
    let qset: Vec<&String> = q.iter().collect();
    fused
        .into_iter()
        .take(k)
        .map(|(i, s)| {
            let c = &chunks[i];
            let ct = tokens(&format!("{} {}", c.title, c.text));
            let covered = qset.iter().filter(|t| ct.contains(t)).count();
            json!({
                "doc_id": c.doc_id, "chunk": c.idx, "title": c.title, "text": c.text,
                "rrf": (s * 10000.0).round() / 10000.0,
                "bm25": (b[i] * 1000.0).round() / 1000.0,
                "cosine": (v[i] * 1000.0).round() / 1000.0,
                "term_coverage": if qset.is_empty() { 0.0 } else { covered as f64 / qset.len() as f64 }
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::acl::Acl;
    fn d(id: &str, text: &str) -> Doc {
        Doc {
            id: id.into(),
            tenant: "t".into(),
            title: id.into(),
            text: text.into(),
            acl: Acl::default(),
            tags: vec![],
        }
    }
    #[test]
    fn sentence_split_keeps_decimals() {
        assert_eq!(
            sentences("Rate is 4.5 percent. Next one!"),
            vec!["Rate is 4.5 percent.", "Next one!"]
        );
    }
    #[test]
    fn bm25_prefers_matching_doc() {
        let docs = vec![
            d("a", "The refund policy allows 30 days."),
            d("b", "Shipping takes five days."),
        ];
        let h = hybrid(&docs, "refund policy", 2);
        assert_eq!(h[0]["doc_id"], "a");
    }
    #[test]
    fn vector_side_catches_morphology() {
        let docs = vec![
            d("a", "Reimbursement requests need receipts."),
            d("b", "Office hours are nine to five."),
        ];
        let h = hybrid(&docs, "reimbursements", 2);
        assert_eq!(h[0]["doc_id"], "a");
    }
    #[test]
    fn chunking_respects_limit() {
        let text = "One two three. Four five six. Seven eight nine.";
        let c = chunk(&d("a", text), 6);
        assert_eq!(c.len(), 2);
    }
    #[test]
    fn empty_corpus() {
        assert!(hybrid(&[], "x", 3).is_empty());
    }
}
