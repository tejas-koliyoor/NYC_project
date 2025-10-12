# Data contract — nyc_taxi_yellow_tripdata

External source: NYC TLC Yellow Taxi Trip Records (Trip Data page + Data Dictionary)
Internal steward: TEJAS KOLIYOOR VARANASHI (ml‑data‑team)  • Slack: #ml-data • Email: tejaskoliyoorvaranashi@gmail.com

Refresh & SLA
- Mirror cadence: monthly (previous month)
- SLA: latest month available by the **5th, 12:00 CET**
- Backfills: allowed; any backfill triggers re‑materialization of downstream features

Schema (initial subset)
- tpep_pickup_datetime: TIMESTAMP (UTC), not null
- tpep_dropoff_datetime: TIMESTAMP (UTC), not null
- PULocationID: INT, not null
- DOLocationID: INT, not null
- passenger_count: INT, nullable -> treat as unknown
- trip_distance: FLOAT miles, nullable; must be ≥0 and <200 else flag
- fare_amount: DECIMAL USD, nullable; must be ≥0 else flag
- payment_type: INT code, nullable -> set -1 (unknown)
- store_and_fwd_flag: CHAR(1) {'Y','N'}, nullable -> default 'N'

Business semantics
- Standardize to UTC in storage; analytics in CET
- Derived duration = dropoff - pickup (1 min ≤ duration ≤ 12 hr else flag)
- Zero fare and ultra‑short trips retained but flagged (`is_anomaly`)

Quality checks
- Freshness: at least one record for latest mirrored month post‑SLA
- Joinability: PU/DO IDs must exist in zone map
- Range checks as above; anomaly rate reported daily

Access & governance
- Readers: analytics_ro, ml_ro • Writer: ingestion_svc
- Logs/metrics contain no PII
- Retention: monthly files 24 months; aggregates indefinite

Change management
- PR + version bump; notify #data-contracts
