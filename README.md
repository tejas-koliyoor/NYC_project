# NYC Taxi ML — production-grade junior project

End-to-end, reproducible, deployable, monitored ML project using **NYC TLC Yellow Taxi** data.
Focus: demonstrate **data contracts → features → MLflow → FastAPI → Docker → Prometheus → CI → runbooks → governance**.

## Quickstart
```bash
pip install -r requirements.txt
# (Later) python -m src.train data/nyc_taxi_sample.csv
uvicorn api.main:app --reload
```

Artifacts map to the 30‑day sprint:
- `data_contracts/` — day 2
- `src/` + `tests/` — days 3–4
- MLflow logging — day 5
- `features/` registry (simulated) — day 6
- `experiments/` — day 7
- `api/`, `docker/`, `k8s/` — days 8–11
- `monitoring/` — day 10
- `.github/workflows/` — day 12
- `runbooks/` — days 13–14
- `governance/` — day 15
- `model_card.md`, demo assets — days 16–18
