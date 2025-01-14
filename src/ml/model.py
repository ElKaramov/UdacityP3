from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import pandas as pd
import logging

logging.basicConfig(filename='LogFile.log',
                    level=logging.INFO,
                    filemode='w',
                    format='%(name)s - %(levelname)s - %(message)s')


# Optional: implement hyperparameter tuning.
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

    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    logging.info('Searching for the best hyperparameter for LogisticRegressio')
    clf = GridSearchCV(estimator=LogisticRegression(penalty='l2'), cv=3, param_grid=param_grid,verbose =2)

    clf.fit(X_train, y_train)
    logging.info("********* Best parameters found ***********")
    logging.info("BEST PARAMS: {}".format(clf.best_params_))

    return clf


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
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
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)

    return preds

def compute_slices(df, feature, y, preds):
    """
    Compute the performance on slices for a given categorical feature
    a slice corresponds to one value option of the categorical feature analyzed
    ------
    df: 
        test dataframe pre-processed with features as column used for slices
    feature:
        feature on which to perform the slices
    y : np.array
        corresponding known labels, binarized.
    preds : np.array
        Predicted labels, binarized

    Returns
    ------
    Dataframe with
        n_samples: integer - number of data samples in the slice
        precision : float
        recall : float
        fbeta : float
    row corresponding to each of the unique values taken by the feature (slice)
    """    
    slice_options = df[feature].unique().tolist()
    perf_df = pd.DataFrame(index=slice_options, 
                            columns=['feature','n_samples','precision', 'recall', 'fbeta'])
    for option in slice_options:
        slice_mask_bool = df[feature]==option

        slice_y = y[slice_mask_bool]
        slice_preds = preds[slice_mask_bool]
        precision, recall, fbeta = compute_model_metrics(slice_y, slice_preds)
        
        perf_df.at[option, 'feature'] = feature
        perf_df.at[option, 'n_samples'] = len(slice_y)
        perf_df.at[option, 'precision'] = precision
        perf_df.at[option, 'recall'] = recall
        perf_df.at[option, 'fbeta'] = fbeta

    # reorder columns in performance dataframe
    perf_df.reset_index(names='feature value', inplace=True)
    colList = list(perf_df.columns)
    colList[0], colList[1] =  colList[1], colList[0]
    perf_df = perf_df[colList]

    return perf_df