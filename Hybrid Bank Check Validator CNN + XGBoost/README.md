# Hybrid Bank Check Validator (CNN + XGBoost) 🏦🔍

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10%2B-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.12%2B-red)

> **Combines deep learning (CNN) and traditional ML (XGBoost) to detect fraudulent checks with 95%+ accuracy.**

## 🚀 Quick Start
Run in 1 minute:
```bash
git clone https://github.com/your-repo/bank-check-validator.git
cd bank-check-validator
pip install -r requirements.txt
streamlit run app.py



## 📁 Project Structure
bank-check-validator/
├── model/               # Pretrained models
│   ├── cnn_features.h5
│   ├── xgboost_model.pkl
│   └── scaler.pkl
├── data/                # Sample checks
│   ├── valid/           # Real checks
│   └── invalid/         # Fake checks
├── app.py               # Web interface
├── train_model.py       # Model training
└── check_validator.py   # Core AI logic


# How to Use
# Web Interface (Recommended)

streamlit run app.py

# command line:
python predict_check.py path/to/check.jpg

# Example Output:

✅ VALID (97.5% confidence)
MICR: Present | Edges: 0.82 | Blur: 120.5