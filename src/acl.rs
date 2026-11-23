//! Document-level access control. Deny by default.

use crate::{parse, search, Doc, Principal};
use serde::Deserialize;
use serde_json::{json, Value};

#[derive(Debug, Clone, Default, Deserialize)]
pub struct Acl {
    #[serde(default)]
    pub users: Vec<String>,
    #[serde(default)]
    pub groups: Vec<String>,
    /// Visible to everyone *in the same tenant*. Never crosses tenants.
    #[serde(default)]
    pub tenant_wide: bool,
    #[serde(default)]
    pub deny_users: Vec<String>,
}

pub fn can_see(d: &Doc, p: &Principal) -> bool {
    if d.tenant != p.tenant || d.tenant.is_empty() {
        return false; // tenant isolation comes first and is exact
    }
    if d.acl.deny_users.contains(&p.user) {
        return false; // explicit deny beats any allow
    }
    d.acl.tenant_wide
        || d.acl.users.contains(&p.user)
        || d.acl.groups.iter().any(|g| p.groups.contains(g))
}

pub fn visible(docs: &[Doc], p: &Principal, require_tags: &[String]) -> Vec<Doc> {
    docs.iter()
        .filter(|d| can_see(d, p) && require_tags.iter().all(|t| d.tags.contains(t)))
        .cloned()
        .collect()
}

/// For each query: every returned chunk must be visible to the principal.
/// Also reports which forbidden documents *would* have ranked if filtering were off.
pub fn leak_test_op(i: &Value) -> Result<Value, String> {
    let docs: Vec<Doc> = parse(i, "docs")?;
    let who: Principal = parse(i, "principal")?;
    let queries: Vec<String> = parse(i, "queries")?;
    let allowed = visible(&docs, &who, &[]);
    let mut results = Vec::new();
    let mut leaks = 0;
    for q in &queries {
        let filtered = search::hybrid(&allowed, q, 5);
        let leaked: Vec<String> = filtered
            .iter()
            .filter_map(|h| h["doc_id"].as_str())
            .filter(|id| !docs.iter().any(|d| d.id == *id && can_see(d, &who)))
            .map(String::from)
            .collect();
        leaks += leaked.len();
        let unfiltered = search::hybrid(&docs, q, 5);
        let would_leak: Vec<String> = unfiltered
            .iter()
            .filter(|h| h["bm25"].as_f64().unwrap_or(0.0) > 0.0).filter_map(|h| h["doc_id"].as_str())
            .filter(|id| !allowed.iter().any(|d| d.id == *id))
            .map(String::from)
            .collect();
        results.push(json!({"query": q, "leaked": leaked, "blocked_by_filter": would_leak}));
    }
    Ok(json!({"pass": leaks == 0, "leaks": leaks, "results": results}))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn doc(id: &str, tenant: &str, acl: Acl) -> Doc {
        Doc {
            id: id.into(),
            tenant: tenant.into(),
            title: id.into(),
            text: "salary bands".into(),
            acl,
            tags: vec![],
        }
    }
    fn alice() -> Principal {
        Principal {
            user: "alice".into(),
            tenant: "acme".into(),
            groups: vec!["hr".into()],
        }
    }
    #[test]
    fn tenant_wide_never_crosses_tenants() {
        assert!(!can_see(
            &doc(
                "x",
                "globex",
                Acl {
                    tenant_wide: true,
                    ..Default::default()
                }
            ),
            &alice()
        ));
        assert!(can_see(
            &doc(
                "x",
                "acme",
                Acl {
                    tenant_wide: true,
                    ..Default::default()
                }
            ),
            &alice()
        ));
    }
    #[test]
    fn group_allow_and_explicit_deny() {
        let a = Acl {
            groups: vec!["hr".into()],
            ..Default::default()
        };
        assert!(can_see(&doc("x", "acme", a.clone()), &alice()));
        let d = Acl {
            deny_users: vec!["alice".into()],
            ..a
        };
        assert!(!can_see(&doc("x", "acme", d), &alice()));
    }
    #[test]
    fn empty_acl_denies() {
        assert!(!can_see(&doc("x", "acme", Acl::default()), &alice()));
    }
    #[test]
    fn same_user_name_other_tenant_denied() {
        let a = Acl {
            users: vec!["alice".into()],
            ..Default::default()
        };
        assert!(!can_see(&doc("x", "globex", a), &alice()));
    }
}
