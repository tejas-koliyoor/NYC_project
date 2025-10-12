# PII policy (project)

PII in source: none directly; do not join emails/phones in raw table.
Access: role‑based (data-team r/w, analytics read)
Retention: raw mirrors 24 months; feature snapshots 12 months; model artifacts 24 months;
Right to erasure: delete by primary key across stores within 30 days.
Security: encryption at rest/in transit; secrets in vault; no PII in logs/metrics.
