## 🚕 NYC Taxi Trip Duration Prediction — Production ML Service

---

### 📌 Problem
- Taxi trip duration in NYC varies significantly by time, location, and traffic  
- Inaccurate estimates impact passenger ETAs, fleet utilization, and pricing decisions  
- Rule-based or manual estimation does not scale to city-level operations  

---

### 🧠 Approach
- Trained a regression model on historical NYC Taxi trip data  
- Built an API-first inference service using FastAPI  
- Enforced schema validation to block invalid inputs  
- Reused the same preprocessing pipeline at training and inference (no train–serve skew)  
- Deployed as a Dockerized service for reproducibility and consistency  

---

### 📊 Metric
- Target: Trip duration (minutes)  
- Evaluated using regression metrics (MAE, RMSE) on held-out data  
- Focused on stable, reliable predictions rather than overfitting for benchmark scores  

---

### 💼 Business Impact
- Enables real-time ETA estimation for passengers  
- Supports driver allocation and fleet planning  
- Helps detect abnormal or inefficient trips  
- Architecture reusable for other prediction services (ETA, demand, churn, fraud)  

---

### 🚀 Key Takeaway
**Transforms real-world NYC taxi data into a reliable, production-ready machine-learning prediction service.**

---
### Workflow
<img width="940" height="246" alt="image" src="https://github.com/user-attachments/assets/5501b9cb-344e-4c24-9de7-8961095df9bf" />


## ⚡ Quickstart

```bash
# Create virtual environment 
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start API
uvicorn app.main:app --reload
# Open
http://localhost:8000

