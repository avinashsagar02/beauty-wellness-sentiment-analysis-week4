# Beauty & Wellness Customer Sentiment & Feedback Analysis

[![Internship Track](https://img.shields.io/badge/Internship-Data%20Science%20with%20Python-blue.svg)](https://github.com)
[![Deliverable](https://img.shields.io/badge/Deliverable-Week%204%20Task-green.svg)](https://github.com)
[![Python Version](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://python.org)

## 📌 Project Overview
This repository contains the end-to-end data processing, exploratory data analysis, feature engineering, and natural language preprocessing architecture for evaluating consumer sentiment, product efficacy, and feedback patterns in the **Beauty and Personal Care E-Commerce sector**.

While aggregate star ratings (1 to 5 stars) provide broad indicators of customer satisfaction, they fail to isolate underlying defect drivers—such as chemical allergies, formulation breakouts, and dispensing pump malfunctions. This project implements an automated, reproducible Natural Language Processing (NLP) framework to extract actionable intelligence from unstructured consumer reviews.

---

## 🎯 Week 4 Objectives: EDA & Feature Engineering
- **Multi-Dimensional EDA:** Executing univariate distributions, bivariate correlation analysis, and cross-cohort demographic evaluations using Seaborn, Matplotlib, and SciPy.
- **Novel Feature Engineering:**
  - `adverse_reaction_flag`: Binary indicator mapping formulation risks and dermal irritation (`burn`, `breakout`, `rash`, `cystic`).
  - `packaging_defect_flag`: Isolates mechanical failures (`pump`, `broke`, `leak`) from formula complaints.
  - `skin_category_interaction`: Composite categorical variable pairing biological user characteristics with formula categories (`Sensitive_Serum`, `Dry_Moisturizer`).
  - `price_expectation_ratio`: Logarithmic interaction metric evaluating consumer price sensitivity and review verbosity.
- **Hypothesis Testing via Inferential Statistics:** Mann-Whitney U and Chi-Square statistical validation testing review verbosity across rating polarities.
- **Hypothetical Business Decision Scenario:** A practical brand decision-making framework translating engineered defect spikes into cosmetic chemistry reformulations and packaging optimizations.

---

## 📂 Repository Structure
```text
├── README.md                    # Project documentation, EDA framework & feature engineering
├── feature_engineering.py       # Week 4: Multi-dimensional feature engineering & statistical tests
├── data_cleaning_pipeline.py    # Week 3: Production 5-gate cleaning & preprocessing pipeline
├── eda_script.py                # Week 2: Initial exploratory data analysis & N-gram modeling
└── requirements.txt             # Project environment dependencies
```

---

## 🚀 Execution Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/avinashsagar02/beauty-wellness-sentiment-analysis.git
cd beauty-wellness-sentiment-analysis
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Week 4 Feature Engineering Pipeline
```bash
python feature_engineering.py
```

---

## 🛠️ Tech Stack & Tooling Justification
- **Python 3.10+**: Core runtime.
- **Pandas & NumPy**: Tabular feature transformations and vector operations.
- **SciPy (`scipy.stats`)**: Inferential statistics (Mann-Whitney U, Kruskal-Wallis, Chi-Square).
- **Scikit-Learn**: MinMaxScaler and feature encoding.
- **Matplotlib & Seaborn**: Multi-dimensional distribution visualizations and correlation matrices.
