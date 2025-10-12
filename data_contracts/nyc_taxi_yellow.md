## Data contract — raw_ext.nyc_taxi_yellow_tripdata (mirror)

External source: NYC TLC Yellow Taxi Trip Records — Trip Data page + official Yellow Data Dictionary


Internal steward: TEJAS KOLIYOOR VARANASHI (ml-data-team) • Slack #ml-data • Email tejaskoliyoorvaranashi@gmail.com

1.Refresh & SLA


Cadence: Monthly mirror of the prior month (as TLC publishes).

SLA: Latest month must be present in warehouse by the 5th of each month, 12:00 CET (late if missed).

Backfills: Allowed; any backfill must trigger re-materialization of downstream features and a changelog entry.


2.Global semantics

Timezone: All timestamps stored in UTC; analytics may render in CET.

Units: Fares in USD; trip_distance in miles.

Primary key (best-effort): (VendorID, tpep_pickup_datetime, PULocationID, _ingest_rowid).

3.Schema 


For each: Type • Nulls • Rules/Range/Codes • Handling on bad values • Example.



VendorID — INT • not null • codes {1 CMT, 2 Curb, 6 Myle, 7 Helix} • if unknown → set -1, is_anomaly=1 • ex: 2

tpep_pickup_datetime — TIMESTAMP(UTC) • not null • must be ≤ dropoff and within file’s month±31d • invalid → drop row (log) • ex: 2025-03-01T08:12:34Z

tpep_dropoff_datetime — TIMESTAMP(UTC) • not null • duration must be 1–720 min • else is_anomaly=1 • ex: 2025-03-01T08:33:10Z

PULocationID — INT • not null • must join to Taxi Zone map • non-joinable → drop row • ex: 142

DOLocationID — INT • not null • must join to Taxi Zone map • non-joinable → drop row • ex: 236

passenger_count — INT • nullable→keep as NULL • valid 0–8; <0 → set NULL, is_anomaly=1 • ex: 1

trip_distance — FLOAT miles • nullable • must be ≥0 and <200; <0 → NULL+flag; ≥200 → cap 200 + flag • ex: 3.2

RatecodeID — INT • nullable → default 99 (unknown) • codes {1 Standard, 2 JFK, 3 Newark, 4 Nassau/Westchester, 5 Negotiated, 6 Group, 99 Unknown} • ex: 1

store_and_fwd_flag — CHAR(1) • nullable→default N • allowed {Y,N} • other → N, is_anomaly=1 • ex: N

payment_type — INT • nullable→-1 • codes {0 Flex Fare, 1 Credit, 2 Cash, 3 No charge, 4 Dispute, 5 Unknown, 6 Voided} • else -1+flag • ex: 1

fare_amount — DECIMAL(10,2) USD • nullable • must be ≥0; <0 → NULL, is_anomaly=1 • ex: 12.50

extra — DECIMAL(10,2) USD • nullable • ≥0 else NULL+flag • ex: 0.50

mta_tax — DECIMAL(10,2) USD • nullable • ≥0 else NULL+flag • ex: 0.50

tip_amount — DECIMAL(10,2) USD • nullable • ≥0; note: cash tips not captured → zeros expected • ex: 2.00

tolls_amount — DECIMAL(10,2) USD • nullable • ≥0 else flag • ex: 0.00

improvement_surcharge — DECIMAL(10,2) USD • nullable • ≥0 else flag • ex: 0.30

congestion_surcharge — DECIMAL(10,2) USD • nullable • ≥0 else flag • ex: 2.75

total_amount — DECIMAL(10,2) USD • nullable • ≥0 (excludes cash tips); negative → is_anomaly=1 • ex: 18.05
