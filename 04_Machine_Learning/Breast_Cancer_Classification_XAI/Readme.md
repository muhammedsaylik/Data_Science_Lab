# Breast Cancer Classification and Explainable AI

## Project Overview

This project focuses on the classification of breast cancer tumors using machine learning models and Explainable Artificial Intelligence (XAI) techniques.

The Breast Cancer Wisconsin dataset was used to build and evaluate predictive models without extensive data preprocessing in order to examine the inherent predictive power of the dataset.

## Models

* Random Forest Classifier
* XGBoost Classifier

## Model Performance

### Random Forest

* Accuracy: **96.49%**
* ROC-AUC: **0.9953**

### XGBoost

* Accuracy: **96.49%**
* ROC-AUC: **0.9898**

## Explainability

SHAP (SHapley Additive exPlanations) was used to interpret model predictions.

The analysis revealed that features related to cell nucleus size and irregularity, such as:

* `worst area`
* `worst concave points`
* `worst concavity`
* `mean concave points`

had the greatest influence on model decisions.

In addition to global feature importance, local explanations were generated for individual observations using SHAP Waterfall plots to understand why the model made specific predictions.

## Conclusion

Both Random Forest and XGBoost achieved excellent predictive performance on the Breast Cancer Wisconsin dataset. Random Forest showed a slightly higher ROC-AUC score, while both models produced identical classification accuracy.

The project demonstrates that highly interpretable machine learning models can achieve excellent performance on medical classification tasks and that Explainable AI techniques provide valuable insights into the decision-making process of predictive models.
