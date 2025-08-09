from sklearn.metrics import fbeta_score, precision_score, recall_score
from ml.data import process_data
# TODO: add necessary import
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import joblib


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train, cv=None):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    # Train and return a model
    if cv is None:
        cv = StratifiedKFold(n_splits=5)

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
    }
    clf = GridSearchCV(
        RandomForestClassifier(random_state=42), param_grid, cv=cv
    )
    clf.fit(X_train, y_train)
    return clf.best_estimator_


def compute_model_metrics(y, preds):

    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : sklearn.base.BaseEstimator
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    # Run model inferences and return the predictions
    return model.predict(X)


def save_model(model, path):
    """ Serializes model to a file.

    Inputs
    ------
    model
        Trained machine learning model or OneHotEncoder.
    path : str
        Path to save pickle file.
    """
    # Save a model
    joblib.dump(model, path)


def load_model(path):
    """ Loads pickle file from `path` and returns it."""
    # Load a model
    return joblib.load(path)


def performance_on_categorical_slice(
    data, column_name, slice_value,
    categorical_features, label, encoder, lb, model
):

    # Computes the metrics on a slice of the data
    data_slice = data[data[column_name] == slice_value]

    X_slice, y_slice, _, _ = process_data(
        # your code here
        # use training = False
        data_slice,
        categorical_features=categorical_features,
        label=label,
        encoder=encoder,
        lb=lb,
        training=False
    )
    preds = inference(model, X_slice)
    precision, recall, fbeta = compute_model_metrics(y_slice, preds)

    log_message = (
        f"Precision: {precision:.4f} | "
        f"Recall: {recall:.4f} | "
        f"F1: {fbeta:.4f}\n"
        f"{column_name}: {slice_value}, Count: {len(data_slice)}\n"
    )

    with open('slice_output.txt', 'a') as f:
        f.write(log_message)

    print(log_message)

    return precision, recall, fbeta
