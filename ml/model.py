from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from ml.data import process_data
import joblib


def train_model(X_train, y_train):
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
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Computes model precision, recall, and F1.

    Inputs
    ------
    y : np.array
        True labels.
    preds : np.array
        Predicted labels.

    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """
    Run model inference and return predictions.

    Inputs
    ------
    model : classifier
        Trained ML model.
    X : np.array
        Data for prediction.

    Returns
    -------
    preds : np.array
        Predicted labels.
    """
    return model.predict(X)


def save_model(model, path):
    """
    Serializes model to a file.

    Inputs
    ------
    model : classifier or encoder
        Trained ML model or preprocessor.
    path : str
        Path to save pickle file.
    """
    joblib.dump(model, path)


def load_model(path):
    """
    Loads pickle file from `path` and returns the object.

    Inputs
    ------
    path : str
        Path to .pkl file.

    Returns
    -------
    model : classifier or encoder
        Loaded object from file.
    """
    return joblib.load(path)


def performance_on_categorical_slice(
    data,
    column_name,
    slice_value,
    categorical_features,
    label,
    encoder,
    lb,
    model
):
    """
    Computes performance metrics for a specific slice of the data.

    Inputs
    ------
    data : pd.DataFrame
    column_name : str
        The name of the categorical column to slice on.
    slice_value : str
        The value of the column to filter by.
    categorical_features : list
    label : str
    encoder : OneHotEncoder
    lb : LabelBinarizer
    model : classifier

    Returns
    -------
    precision, recall, fbeta : float
    """
    slice_df = data[data[column_name] == slice_value]
    X_slice, y_slice, _, _ = process_data(
        slice_df,
        categorical_features=categorical_features,
        label=label,
        training=False,
        encoder=encoder,
        lb=lb
    )
    preds = inference(model, X_slice)
    precision, recall, fbeta = compute_model_metrics(y_slice, preds)
    return precision, recall, fbeta
