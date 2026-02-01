# Loan Default Risk Modeling with SHAP Explainability

This project builds an end-to-end **credit default risk model** using XGBoost
and provides **model explainability** using SHAP, deployed as a Streamlit web app.

## 🔍 Key Features
- Binary classification (Default vs Non-default)
- XGBoost model
- SHAP global & local explainability
- Streamlit dashboard for visualization
- Artifact-based explainability (production-friendly)

## 📊 Explainability Outputs
- Global SHAP summary plot
- Feature importance table
- SHAP dependence plots
- Default-class specific explanations
- Individual prediction force plots

## 🚀 How to Run the App

```bash
pip install -r requirements.txt
streamlit run app.py
📁 Project Structure

notebooks/ → EDA, training, SHAP generation

artifacts/ → saved explainability outputs

app.py → Streamlit dashboard

📸 Dashboard Preview

🧠 Tools & Libraries

Python

XGBoost

SHAP

Pandas

Streamlit

Matplotlib

📌 Notes

SHAP artifacts are generated offline and loaded by the UI.
This mirrors production explainability workflows.