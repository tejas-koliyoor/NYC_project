. Common Failure Modes & Fix Procedures
4.1 API Container CrashLoopBackOff

Symptom:

kubectl get pods
# shows CrashLoopBackOff


Likely Causes:

Model file missing or corrupted

Python dependencies mismatch

Code exceptions on startup

Steps:

Check logs:

kubectl logs <pod>


Verify model existence:

/app/artifacts/model.joblib


Rebuild image:

docker build -t nyc-taxi-api:latest .
kind load docker-image nyc-taxi-api:latest --name nyc-taxi
kubectl rollout restart deployment/nyc-taxi-api

4.2 /predict returns 500 (Inference failed)

Causes:

Incoming request schema invalid

Missing required features

Pandas/scikit-learn mismatch

Steps:

Check API logs:

kubectl logs <pod>


Validate input using:

GET /docs

4.3 High Latency or Timeout

Causes:

Model taking too long to load in each request

High traffic causing queueing

Steps:

Check metrics:

/metrics → prediction_request_latency_seconds


Scale out:

kubectl scale deployment/nyc-taxi-api --replicas=4


Optimize preprocessing/model if persistent.

4.4 Prometheus Shows No Metrics

Causes:

Wrong scrape path

/metrics endpoint down

Steps:

Check:

curl http://localhost:8000/metrics


Validate Prometheus config.

Restart Prometheus container:

docker compose restart prometheus

5. Deployment Procedures
5.1 Local (Kind)
kind create cluster --name nyc-taxi
docker build -t nyc-taxi-api:latest .
kind load docker-image nyc-taxi-api:latest --name nyc-taxi
kubectl apply -f k8s/deployment.yaml

5.2 Rollout Restart
kubectl rollout restart deployment/nyc-taxi-api

5.3 Rollback
kubectl rollout undo deployment/nyc-taxi-api

6. Monitoring & Dashboards

Metrics:

prediction_requests_total

prediction_errors_total

prediction_request_latency_seconds

Grafana panels:

API latency histogram

Error rate

Pod CPU/Memory usage

7. On-Call Contact

Primary: Tejas KV

Secondary: (future teammate)

8. Useful One-Liners
kubectl get pods -o wide
kubectl logs -f deployment/nyc-taxi-api
kubectl top pods
kubectl describe deployment nyc-taxi-api

9. Change Log

Day 13: Runbook created.


---

# ✅ **Day 14: Blameless Postmortem Template**  
> Goes under: `runbooks/postmortem_template.md`

### **`runbooks/postmortem_template.md`**

```markdown
# Postmortem Report (Blameless Template)

**Incident ID:**  
**Date of Incident:**  
**Author:**  
**Severity (SEV-1/2/3/4):**  
**Status:** Draft / Final

---

## 1. Summary

A short, 3–4 sentence overview of what happened, the impact, and resolution.

---

## 2. Impact

- Who was affected?  
- What functionality broke?  
- How long did the incident last?  
- Quantify impact if possible.

---

## 3. Timeline (UTC)

| Time (UTC) | Event |
|------------|--------|
| 00:00 | First alert triggered |
| 00:02 | Engineer acknowledged |
| 00:05 | Investigation begins |
| ... | ... |
| 00:25 | Issue resolved |

---

## 4. Root Cause Analysis (RCA)

### 4.1 What actually happened?
Step-by-step factual reconstruction.

### 4.2 Why did it happen?
(5 Whys recommended)

---

## 5. Detection

- How was the issue detected?  
- Were alerts triggered?  
- Detection gaps?

---

## 6. Resolution

Exactly how the issue was fixed:
1. Commands run  
2. Rollouts, restarts  
3. Reconfigs applied

---

## 7. Lessons Learned

- What went well?
- What went poorly?
- What surprised us?

---

## 8. Action Items (with Owners)

| Action | Owner | Priority | ETA |
|--------|--------|----------|-----|
| Add alert for high latency | Tejas | High | 1 week |
| Improve unit test coverage | Tejas | Medium | 2 weeks |
| Add liveness probe for model load | Tejas | Medium | Next deploy |

---

## 9. Preventing Recurrence

- Architecture changes  
- CI/CD changes  
- Monitoring improvements  
- Runbook updates  

---

## 10. Appendix

- Logs  
- Screenshots  
- Metrics graphs  
- Related PRs/issues  
