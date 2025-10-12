# Runbook — inference returns constant or zero scores

Severity: P2
Triage (10 min)
1) GET /health — up? model_loaded?
2) Check logs (last 60m) for load/NaN errors
3) Send known‑good /predict payload — check variance
4) Verify model file presence & timestamp
5) Verify data freshness vs SLA; dummy features present?

Mitigation
- Roll back to previous image tag
- If features broken: disable feature view; fall back to baseline

Comms
- Update #ml-ops, open incident ticket, assign owner

Post‑incident
- Root cause, timeline, impact; add tests/monitors
