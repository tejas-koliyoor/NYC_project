# Monitoring metrics spec

Service
- requests_total{endpoint} [Counter]
- request_latency_seconds{endpoint} [Histogram]
- http_5xx_total [Counter]

Model proxy quality
- prediction_rate [Gauge]
- avg_score [Gauge] + score_histogram [Histogram]
- input_null_rate{feature} [Gauge]
- daily drift test vs training dist; alert on JS divergence > 0.1

Alerts
- p95 latency > 300ms (5m)
- 5xx > 1% (5m)
- score distribution shift >10% day‑over‑day
