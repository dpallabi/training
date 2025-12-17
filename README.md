# 🧠 Multi-Objective Diabetes Risk Classification & Optimization Framework

## 📌 Overview
This project presents an **end-to-end machine learning pipeline for diabetes risk prediction**, combining **classical ML models**, **advanced hyperparameter optimization**, and **multi-objective decision analysis**.

The framework jointly optimizes **predictive performance**, **clinical reliability**, and **computational efficiency** using a **hybrid SMAC + NSGA-II optimization strategy**, making it suitable for **real-world medical decision-support systems**.

---

## 📁 Dataset
- **Dataset:** Pima Indians Diabetes Dataset  
- **Source:** Kaggle  
- **Samples:** 768 patients  
- **Target Variable:** Diabetes Outcome (0 = Non-diabetic, 1 = Diabetic)

Preprocessing includes:
- Median imputation for invalid zero values
- Feature scaling (StandardScaler)
- Class imbalance handling using **SMOTE**

---

## ⚙️ Machine Learning Models
The following classifiers are evaluated and optimized:

- **Support Vector Machine (SVM)**
- **Random Forest (RF)**
- **XGBoost (XGB)**
- **Logistic Regression (LR)**

Each model is tuned over a **model-specific hyperparameter space**.

---

## 🎯 Multi-Objective Optimization Goals
All objectives are treated as **minimization problems**:

1. **1 − Accuracy** (maximize diagnostic accuracy)
2. **1 − Abnormal Recall** (maximize detection of diabetic patients)
3. **False Negative Rate (FNR)** (minimize missed diagnoses)
4. **Normalized Training Time**
5. **Normalized Testing Time**

This ensures balanced trade-offs between **clinical safety** and **deployment efficiency**.

---

## 🔁 Optimization Strategy

### Phase 1: SMAC + ParEGO (Exploration)
- Broad hyperparameter search
- Efficient exploration of large configuration spaces
- Generates high-quality candidate solutions

### Phase 2: NSGA-II (Refinement)
- Evolutionary multi-objective optimization
- Refines Pareto-optimal solutions
- Preserves diversity across competing objectives

---

## 📊 Key Visualizations & Results

### Phase 1: Pareto Front Analysis

<img width="4909" height="2077" alt="multi_pareto_analysis" src="https://github.com/user-attachments/assets/06efdd29-840b-41e4-aba6-1ce7d666d97b" />

### Phase 2: Tableau Interactive Dashboards

<img width="1920" height="1080" alt="Screenshot 2025-12-17 093205" src="https://github.com/user-attachments/assets/ff4ecbf5-b899-4c06-b856-c811bda53dc1" />

<img width="1920" height="1080" alt="Screenshot 2025-12-17 093226" src="https://github.com/user-attachments/assets/bc9b8f00-8ff3-4b99-99e6-83c7b2cfab43" />

---

## 🧠 Key Takeaways
- Multi-objective optimization provides **safer and more deployable models** than single-metric tuning
- XGBoost and Random Forest consistently dominate Pareto fronts
- Explicit FNR minimization significantly improves clinical reliability
- Hybrid SMAC + NSGA-II balances exploration and refinement effectively

---

## 📌 Applications
- Clinical decision-support systems
- Risk-aware medical ML deployment
- Multi-objective optimization research
- Explainable and efficient healthcare AI
