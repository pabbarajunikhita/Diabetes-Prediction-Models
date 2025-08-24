# Diabetes-Prediction-Models
**Author:** Nikhita Pabbaraju  
**Course:** Stat 4770 – Python for Data Science  

Analysis of diabetes risk factors and predictive modeling using logistic regression, random forest, and KNN.

## Project Overview
This independent project explores **predicting diabetes onset** using machine learning on the Pima Indians Diabetes dataset. The goal is to identify key predictors and build models that accurately classify patients, demonstrating skills in Python, data analysis, and machine learning.

## Dataset
- **Source:** [Kaggle – Pima Indians Diabetes](https://www.kaggle.com/datasets/nikhilnarasimhan3264/pima-indians-diabetes)  
- **Samples:** 768 female patients, age ≥ 21  
- **Features:** 8 clinical variables (Glucose, BMI, Blood Pressure, Insulin, Age, etc.)  
- **Target:** `Outcome` (1 = diabetes, 0 = no diabetes)  
- **Preprocessing:** Missing/implausible values imputed; dataset split 70/30 for training/testing  

## Methods
- **Exploratory Data Analysis (EDA):** Correlation analysis, distributions, heatmaps  
- **Machine Learning Models:**  
  - Logistic Regression  
  - Random Forest Classifier  
  - K-Nearest Neighbors (KNN)  
- **Evaluation Metrics:** Accuracy, Precision, Recall (Sensitivity), Specificity, F1 Score, ROC-AUC  

## Results

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|---------|-----------|--------|----------|---------|
| Logistic Regression | 73.16% | 71.19% | 48.28% | 0.575 | 0.821 |
| Random Forest | 73.59% | 72.41% | 48.28% | 0.579 | 0.807 |
| KNN | 70.99% | 63.89% | 52.87% | 0.578 | 0.759 |

**Key Insight:** Random Forest achieved the best overall balance of performance metrics, while Glucose and BMI were the strongest predictors.

## Tools & Technologies
- **Language:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn  
- **Skills Demonstrated:** Data cleaning, EDA, feature analysis, model building, evaluation, visualization  
