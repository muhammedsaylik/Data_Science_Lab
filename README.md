<h1 align='center'>Data Science Portfolio</h1>         

<p align="center">
  <a href="https://github.com/muhammedsaylik"><img alt="Profile" src="https://img.shields.io/badge/Portfolio-Data%20Science-blue?style=for-the-badge&logo=databricks"/></a>
  <img alt="MIT License" src="https://img.shields.io/badge/License-MIT-green?style=flat-square" />
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python" />
  <img alt="Notebooks" src="https://img.shields.io/badge/Jupyter-Notebooks-orange?style=flat-square&logo=jupyter" />
  <img alt="Top Language" src="https://img.shields.io/github/languages/top/muhammedsaylik/Data_Science_Portfolio?style=flat-square" />
</p>

---

## About ✨
> This repository showcases my data science journey, ranging from deep statistical theories and implementing algorithms from scratch to deploying end-to-end machine learning and time series forecasting pipelines. All projects are strictly organized following clean code principles, professional English naming conventions, and industry-standard folder architecture.

---

## Repository Structure & Navigation 🔭
- [01_EDA](./01_EDA) — Exploratory Data Analysis, comprehensive reporting, and multivariate statistical analyses (`multivariate_anova`).
- [02_Visualization](./02_Visualization) — Advanced data storytelling and visualization frameworks using Matplotlib and Seaborn.
- [03_Feature_Engineering](./03_Feature_Engineering) — Feature engineering techniques, scaling methods, Mutual Information scoring, and robust model validation strategies.
- [04_Machine_Learning](./04_Machine_Learning) — Regression and Classification workflows with comprehensive metric optimization.
- [05_Time_Series](./05_Time_Series) — Financial econometrics, volatility modeling, backtesting, and production-ready Streamlit forecasting applications.
- [06_Algorithms_from_Scratch](./06_Algorithms_from_Scratch) — Pure Python/NumPy implementations of Linear Algebra, mathematical optimization methods, and Game Theory algorithms.
- [07_Statistical_Learning](./07_Statistical_Learning) — Foundational data structures exercises and library practices.
- [utils](./utils) — Reusable custom scripts for evaluation metrics, modeling workflows, and visualization helpers.

---

## Featured Projects ⭐

### 1. Algorithms from Scratch (Mathematical & Statistical Foundations)
Implementing the core mathematical theories behind popular machine learning techniques completely from scratch without external high-level libraries. Includes Linear Algebra (Gram-Schmidt, PCA Eigendecomposition), Optimization (Bisection method, Newton Method, KKT Conditions), and Game Theory (Q-Learning, Genetic Algorithms).
* 📂 **Directory:** [06_Algorithms_from_Scratch](./06_Algorithms_from_Scratch)

### 2. BTC Financial Econometrics & Machine Learning
A rigorous research project combining financial econometrics and time series forecasting on Bitcoin price movements. Features advanced feature engineering, tailored XGBoost modeling, programmatic backtesting, and model interpretability via Explainable AI (XAI).
* 📂 **Directory:** [btc_financial_econometrics](./05_Time_Series/btc_financial_econometrics)

### 3. Feature Engineering & Validation Pipeline
An end-to-end framework applied to hotel booking demand data, focusing heavily on robust data preprocessing, structured feature generation, Mutual Information scoring, and leak-free validation architectures to ensure dependable model evaluation.
* 📂 **Directory:** [03_Feature_Engineering](./03_Feature_Engineering)

---

## Tech Stack 🔧
- **Data Manipulation & Analysis:** Pandas, NumPy, Scikit-Learn
- **Modeling & Time Series:** XGBoost, Statsmodels, Arch (Volatility Modeling)
- **Visualization & UI:** Matplotlib, Seaborn, Streamlit
- **Engineering & Testing:** Python Scripts (`src/`), PyTest, Git

---

## Quick Start (Local Setup) 🚀
```bash
# Clone the repository
git clone [https://github.com/muhammedsaylik/Data_Science_Portfolio.git](https://github.com/muhammedsaylik/Data_Science_Portfolio.git)
cd Data_Science_Portfolio

# Create and activate a virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Mac/Linux:
source .venv/bin/activate

# Install the required dependencies
pip install pandas numpy scikit-learn matplotlib seaborn jupyter streamlit xgboost statsmodels pytest

# Launch Jupyter Notebook
jupyter notebook

