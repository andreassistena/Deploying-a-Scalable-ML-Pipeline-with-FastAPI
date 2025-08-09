<<<<<<< HEAD
# Model Card

This model is a Logistic Regression classifier designed to predict whether an individual's income exceeds $50,000 based on various demographic features, such as education, occupation, and marital status. It was trained on a Census Income dataset. The model was built for educational purposes, primarily to demonstrate the process of constructing a machine learning pipeline. It uses one-hot encoding for categorical features and standardization for continuous variables.

## Intended Use
It serves as a baseline model for predicting income levels and can be adapted for similar binary classification tasks.
## Training Data
The model was trained on a Census Income dataset, which consists of 32,561 samples after an 80/20 train-test split. The dataset includes demographic information such as workclass, education, marital status, occupation, relationship, race, sex, and native country. Categorical features were one-hot encoded, and the labels were binarized to indicate whether an individual's income exceeds $50,000.
## Evaluation Data
The model was evaluated on a test dataset derived from the same Census dataset, consisting of 8,141 samples. The test data underwent the same preprocessing steps as the training data, using the encoder and label binarizer fitted on the training set. The evaluation aimed to assess the model's generalization performance across various demographic slices.
## Metrics
_Please include the metrics used and your model's performance on those metrics._
Precision: 0.7419 | Recall: 0.6384 | F1: 0.6863
These results suggest that the model strikes a moderate balance between identifying positive cases (recall) and minimizing false positives (precision)

## Ethical Considerations
This model may reflect biases inherent in the training data, particularly regarding demographic features such as race and gender. These biases can lead to skewed predictions that may affect certain groups more than others.

## Caveats and Recommendations
This model may not generalize well to different populations or datasets with varying distributions. The model's performance varies significantly across different slices of the data, so this should be considered when interpreting results when testing with different datasets.
=======
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
>>>>>>> 7b99ec5a894c135ed8e7a3294919c18e6b8a9155
