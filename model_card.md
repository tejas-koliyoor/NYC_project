# Model Card — NYC Taxi Trip Outcome (prototype)

Owner: ml-data-team (tejaskoliyoorvaranashi@gmail.com)
Purpose: Predict a  high‑fare or long‑trip flag  as a placeholder for demo.
Data: nyc_taxi_yellow_tripdata (see contract)
Algorithm: TBD baseline (sklearn)
Monitoring: latency, request rate, score drift; input null rate
Privacy: no direct PII; IDs hashed if introduced

MODEL CARD — NYC Taxi Trip Duration Model
1. Model Details

Model Type: Regression (predicting trip duration in minutes)
Architecture: scikit-learn Pipeline → Preprocessing (imputer, scaler, one-hot encoder) → RandomForestRegressor
Framework: Python 3.11, scikit-learn 1.x
Artifact Paths:

Model file: models/model.pkl

Feature names: models/feature_names.json

Author: Anoop Janardhanan Nair
Created: 2025
Repository: https://github.com/tejas-kv/NYC_Project

2. Intended Use
Primary Use

This model predicts NYC taxi trip duration given basic ride attributes, e.g.:

pickup/dropoff location

passenger count

fare details

pickup hour and day distance

It is intended for:

ML pipeline demos

MLOps education

Practicing containerization, monitoring, CI/CD, Kubernetes

Out-of-Scope Use

This model must NOT be used for:

Real-time taxi dispatch decisions

Pricing

Any fairness-sensitive or legally binding use

High-stakes predictions (medical, financial, legal)

3. Training Data

The model uses a subset of the NYC Yellow Taxi public dataset (open data).

Data Source:
https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page

Preprocessing Steps

Removed extreme outliers (e.g., duration < 1 min or > 120 min)

Derived hour-of-day, day-of-week, and weekend boolean

Imputed missing continuous values with median

Imputed missing categorical values with most frequent

One-hot encoded categorical variables

4. Evaluation

Metrics:

MAE (Mean Absolute Error) on validation split

Typical value: ~6–8 minutes MAE (varies by month)

Train/Test Split:

80% training

20% validation

5. Ethical Considerations
⚠️ Bias & Fairness

Taxi trip duration can correlate with:

neighborhood congestion

traffic patterns

socioeconomic geography

The model inherits biases present in the NYC dataset.
It does not contain demographic attributes (age, race, gender), but location can be correlated with these indirectly.

⚠️ Privacy

This repository includes a PII Scrubbing Layer:

removes names, phone numbers, emails, IDs

removes GPS coordinates

hashes location IDs if needed

retention policy: input data is never logged; prediction logs are anonymized

6. Limitations

Does not model traffic accidents or sudden weather events

Accuracy decreases for very short (< 0.5 miles) or long (> 20 miles) trips

Model is static and not updated over time

RandomForestRegressor is not the most interpretable model

This is a teaching demo — not production-grade

## Input Features

### Numerical Features
- `trip_distance`
- `fare_amount`
- `passenger_count`
- `pickup_hour`
- `pickup_dow`
- `is_weekend`

### Categorical Features
- `PULocationID`
- `DOLocationID`
- `payment_type`
- `RatecodeID`
- `store_and_fwd_flag`

All preprocessing (scaling and encoding) is included **inside the pipeline**.

---

## Model Architecture
- Algorithm: `RandomForestRegressor`
- Number of trees: 200
- Preprocessing:
  - `StandardScaler` for numerical features
  - `OneHotEncoder(handle_unknown="ignore")` for categorical features


---

## Evaluation
- Train/validation split: 80% / 20%
- Metric: Mean Absolute Error (MAE)
- Typical performance:  
**MAE ≈ X–Y minutes** (depends on dataset slice)

---

## Limitations
- Does not account for:
- Traffic conditions
- Weather
- Road closures or events
- Performance may degrade for rare routes or extreme trip lengths
- Assumes NYC taxi schema compatibility

---

## Ethical Considerations
- No personal identifiers are used
- Location IDs may indirectly reflect socio-economic patterns
- Predictions should not be used for discriminatory or punitive decisions

---

## Data Privacy
- No PII is stored or logged
- Input records are processed in-memory only
- Example payloads contain synthetic or anonymized data

---

## Versioning
- Model artifact: `artifacts/model.joblib`
- Pipeline versioned via Git
- Reproducible via Docker

---

## Contact
Maintained as part of an academic ML systems project.

