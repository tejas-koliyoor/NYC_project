# NYC Taxi API — Production Runbook
_Last updated: {{03-12-2025}}_
_Owner: Tejas KV_
_Service: nyc-taxi-api (FastAPI + scikit-learn model)_

---

## 1. Service Overview

The **NYC Taxi API** provides:
- `POST /predict` → returns predicted trip duration (in minutes).
- `GET /health` → service health indicator.
- `GET /metrics` → Prometheus metrics (requests, latency, errors).

The model is a scikit-learn pipeline trained on NYC Taxi trip data.  
Deployment environment: Docker + Kubernetes (kind/minikube or cloud).

---
User → FastAPI Service → Model Pipeline → Response
↘
Prometheus /metrics → Grafana Dashboard


Kubernetes:
- **Deployment** with 2 replicas
- **Service** exposing port 8000
- **Readiness probe:** `/health`
- **Liveness probe:** `/health`
- **Metrics:** Prometheus scrape on `/metrics`

---

## 3. Health Checks

### 3.1 Quick Health Check

```bash
curl http://localhost:8000/health

Healthy response:

{"status": "ok", "model_loaded": true, "version": "v1"}

### 3.2 Kubernetes Status
kubectl get pods
kubectl describe pod <pod-name>
kubectl logs <pod-name>



