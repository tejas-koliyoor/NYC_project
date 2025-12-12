# 🗂️ NYC Taxi Trip Duration Model — Model Card  
_Version: v1 • Last Updated: 2025-11_

## 1. Model Overview
This model predicts **trip duration (in minutes)** for NYC Yellow Taxi rides.  
It is designed for skill building purpose in the 30-Day ML Sprint.

**Model type:** Random Forest Regressor  
**Input:** A single taxi trip’s metadata (distance, pickup time, locations…)  
**Output:** Estimated trip duration in minutes  
**Serving:** FastAPI + Docker + Prometheus + Kubernetes (optional)

---

## 2. Intended Use
### ✔ What this model is meant for
- Demonstrating a full MLOps pipeline  
- Local inference demos  
- Teaching ML deployment, monitoring, and validation  
- Building a functional end-to-end microservice

### ✖ What this model is *not* meant for
- Real-time commercial prediction  
- Regulatory decision-making  
- High-accuracy ETA prediction in production  
- Any use involving safety-critical operations  

---

## 3. Training Data
Dataset: **NYC Yellow Taxi (March 2025 sample)**  
Rows: ~10,000  
Features used:

| Feature         | Type        | Description            |
|-----------------|-------------|------------------------|
| trip_distance   | numeric     | Trip distance in miles |
| fare_amount     | numeric     | Fare paid              |
| passenger_count | numeric     | Number of passengers   |
| pickup_hour     | numeric     | Hour of day (0–23)     |
| pickup_dow      | numeric     | Day of week (0–6)      |
| is_weekend      | numeric     | 0/1 derived feature    |
| PULocationID    | categorical | Pickup zone ID         |
| DOLocationID    | categorical | Dropoff zone ID        |
| payment_type    | categorical | Cash/credit/other      |
| RatecodeID      | categorical | Rate code              |

⚠️ PII (pickup timestamps) were removed.  
⚠️ High-cardinality features were bucketized or one-hot encoded.

---

## 4. Model Performance
Validation set size: 20% holdout

**Metric:** Mean Absolute Error (MAE)  
**Validation MAE:** ~2.3 minutes

Interpretation:
- Predictions are usually within ±2–3 minutes of actual duration.
- Suitable for demos, not production ETA systems.

---

## 5. Ethical Considerations
- No personal data used.  
- No demographic or sensitive attributes.  
- No risk for discrimination or harm.  
- Should not be used to make financial or legal decisions.

---

## 6. Limitations
- Small training sample — accuracy is limited.
- No historical traffic features (weather, holidays).
- Not optimized for latency or scaling.
- Categorical location IDs treated as arbitrary categories; may not generalize.

---

## 7. Risks
- Should not be used for commercial dispatching or dynamic pricing.
- Could perform poorly during unusual events (storms, road closures).
- Cold-start zones with few examples may be inaccurately predicted.

---

## 8. Maintenance & Versioning
Artifacts stored under `artifacts/`:

- `model.pkl` — serialized RandomForest pipeline  
- `feature_names.json` — schema for inference  
- `model_card.md` — model documentation  
- API version returned in `/health` route  

Model update guidelines:
- Retrain monthly if used in production  
- Check monitoring dashboards for drift  
- Update model card after every new release  

---

## 9. Contact
Maintainer: **Anoop Janardhanan Nair**  
Course: Research in Computer & Systems Engineering, TU Ilmenau  
Repository: GitHub → NYC_Project  

