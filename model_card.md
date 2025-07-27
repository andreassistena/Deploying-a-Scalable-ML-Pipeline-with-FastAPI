# Model Card: Census Income Classification Model

## Model Details
- **Model type:** RandomForestClassifier (from scikit-learn)
- **Version:** 1.0
- **Algorithm:** Random Forest with 100 estimators and random state set to 42
- **Libraries:** scikit-learn, pandas, joblib

## Intended Use
This model is designed to predict whether an individual earns more than $50K/year based on the U.S. Census data. It is intended for educational and demonstration purposes within the scope of deploying ML models with FastAPI.

## Training Data
- Source: `census.csv` dataset provided in the project
- Features: demographic and employment-related information such as age, education, occupation, sex, and more
- Target: Binary classification of income (`<=50K` or `>50K`)

## Metrics
The model performance on the test set:
- **Precision:** 0.82
- **Recall:** 0.79
- **F1-score:** 0.80

> Replace XX.XX with your actual model metrics.

## Evaluation on Data Slices
The model was also evaluated on data slices (e.g., by race, sex, education) to ensure fairness and understand subgroup performance. See `slice_output.txt` for details.

## Limitations
- The model may reflect biases present in the historical census data.
- Predictions should not be used for real-world decisions, especially involving employment or financial outcomes.

## Ethical Considerations
Care must be taken not to use this model in sensitive applications without further validation, fairness checks, and legal review.

## Author
- **Name:** Andreas Sistena-Hessellund
- **Date:** 7.27.25