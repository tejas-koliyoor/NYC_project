# NYC Project — Data Retention & Privacy Policy
_Last updated: {{03-12-2025}}_
_Owner: Tejas KV_

## 1. Purpose

This policy defines:
- What data is collected
- How long it is stored
- How it is protected
- When and how it is deleted

It follows common best practices, GDPR guidelines, and data minimization principles.

---

## 2. Data We Collect

### 2.1 Input data (via `/predict`)
Raw input records may include:
- trip metadata (distance, fare, hour, locations)
- categorical features (payment type, rate code)
- no required PII fields

### 2.2 System metadata
- Request timestamps
- Aggregated metrics (Prometheus)
- Error counters
- Latency histograms

### 2.3 What is **NOT** collected
The system explicitly does NOT store:
- Names
- Emails
- Phone numbers
- Addresses
- IDs (passport, license, SSN)
- Any sensitive user profile data

PII is automatically removed server-side using `src/privacy.py`.

---

## 3. Data Retention Rules

### 3.1 Online storage (in-memory)

- Incoming request data exists **only in memory** during inference.
- It is NOT logged.
- It is NOT persisted.
- It is discarded immediately after response.

### 3.2 Logs

- Application logs must **never include raw input rows**.
- Errors may include model stack traces, but never user data.
- Logs are retained for **7 days** for debugging, then deleted.

### 3.3 Metrics (Prometheus)

Prometheus retains **aggregated** counters only:
- No raw samples
- No user-identifiable data

Retention: **15 days** (default Prometheus retention).

### 3.4 Model Artifacts

- `model.joblib` is not user data.
- Stored indefinitely unless replaced in a new release.

---

## 4. Data Minimization Strategy

- Drop PII fields immediately.
- Mask email/phone patterns.
- Accept only required model features.
- Reject unknown fields during schema validation (FastAPI/Pydantic).

---

## 5. GDPR Considerations

Under GDPR:
- The API is a **data processor**, not a controller.
- No personal data leaves the client’s environment.
- No user profiling occurs.
- No IDs or machine identifiers are retained.
- Requests cannot be re-associated with individuals.

---

## 6. Deletion Procedures

### Immediate deletion (built-in)
All request data is deleted automatically after processing.

### Manual deletion
Not applicable — no stored user data.

---

## 7. Ownership & Review

Responsible Owner: **Tejas KV**  
Review Cycle: **Every 90 days**  
Next Review: {{DATE + 90 DAYS}}
