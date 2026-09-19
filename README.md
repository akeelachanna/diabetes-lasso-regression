# Diabetes Progression Prediction Using Lasso Regression

Predicting diabetes disease progression using Lasso Regression, with automatic feature selection to identify key health indicators.

## Dataset
- Built-in `load_diabetes()` dataset from scikit-learn
- 442 patients with 10 health features (age, sex, BMI, blood pressure, and 6 blood test measurements)
- Target: Disease progression after 1 year

## Project Workflow
1. Loaded and explored the dataset (EDA with statistics and correlation analysis)
2. Visualized target distribution and feature correlations
3. Trained Lasso Regression with multiple alpha values for tuning
4. Selected best alpha based on R² score and feature count
5. Compared Lasso with Linear Regression and Ridge Regression
6. Visualized feature coefficients (which features were kept vs eliminated)

## Key Insights
- Lasso eliminated irrelevant features while maintaining good prediction accuracy
- Identified key health indicators (BMI, blood pressure, and specific blood test measurements)
- Lasso provided simpler, more interpretable model compared to Linear Regression

## Tools Used
- Python (pandas, numpy, scikit-learn)
- Matplotlib, Seaborn
- Google Colab

## Files
- `diabetes-lasso-regression.ipynb` — Complete notebook with code and outputs
- `lasso_coefficients.csv` — Feature coefficients from the Lasso model
