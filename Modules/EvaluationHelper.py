import numpy as np
from sklearn.model_selection import cross_val_score

def cross_validate(model, x, y, cv, method_names):
    """Performs cross validation on a regression model and returns the mean of the scores for each method name.
    
    model: the regression model to evaluate
    x: the features to evaluate on
    y: the labels to evaluate on
    cv: the number of cross validation folds to use
    method_names: a list of method names to evaluate on (e.g. ["r2", "neg_mean_absolute_error"])
    
    returns: a dictionary of method names and their corresponding mean scores
    """
    metric_dict = {}
    for name in method_names:
        metric_dict[name] = np.mean(cross_val_score(estimator=model,X=x, y=y,cv=cv, scoring=name))
    
    return metric_dict

def evaluate_preds(y_true,y_preds, eval_methods):
    """
    Performs evaludation comparison on y_true vs y_pred labels on a classification model.
    y_true: the true labels
    y_preds: the predicted labels
    eval_methods: a dictionary of method names and their corresponding evaluation functions (e.g. {"R Squared" : r2_score, "Mean Absolute Error (MAE)" : mean_absolute_error})

    returns: a dictionary of method names and their corresponding scores"""
    metric_dict = {}
    for name, method in eval_methods.items():
        metric_dict[name] = method(y_true, y_preds)
    
    return metric_dict