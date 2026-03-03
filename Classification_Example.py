"""
Link to dataset used: https://archive.ics.uci.edu/dataset/45/heart+disease

Create data dictionary:

  1. #3  (age)       
  2. #4  (sex)       
  3. #9  (cp)        
  4. #10 (trestbps)  
  5. #12 (chol)      
  6. #16 (fbs)       
  7. #19 (restecg)   
  8. #32 (thalach)   
  9. #38 (exang)     
  10. #40 (oldpeak)   
  11. #41 (slope)     
  12. #44 (ca)        
  13. #51 (thal)
  14. #58 (num)       (the predicted attribute)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, classification_report 
from Modules.EvaluationHelper import cross_validate
from Modules.ModelExportHelper import export_model_to_file 

# Create function to fit and score different models
def fit_and_score(x_train, x_test, y_train, y_test):
    np.random.seed(42)

    models = {
        "RandomForestClassifier" : RandomForestClassifier(),
        "LogisticRegression" : LogisticRegression(),
        "KNeighborsClassifier" : KNeighborsClassifier()
    }

    results={}
    for name, model in models.items():
        model.fit(x_train,y_train)
        results[name] = model.score(x_test,y_test)
    return results

def tune_hyper_params_KNN(x_train, x_test, y_train, y_test):
    train_scores = []
    test_scores = []

    # Create a list of different values for n neighbors
    neighbors = range(1,21)
    # Setup KNN instance
    knn = KNeighborsClassifier()

    # Loop through different n_neighbots
    for i in neighbors:
        knn.set_params(n_neighbors=i)

        # Fit the algorithm
        knn.fit(x_train,y_train)
        train_scores.append(knn.score(x_train,y_train))
        test_scores.append(knn.score(x_test,y_test))

    print(f"KneighborsClassifer turned Scores: ")
    print(f"-> Average score on training data: {np.mean(train_scores)}")
    print(f"-> Average score on test data: {np.mean(test_scores)}")

def cv_tuner(tuning_method, x_train, y_train, param_distributions, cv=5, verbose=True, n_iter=20, cv_method_name=None):
    np.random.seed(42)

    if cv_method_name == "RandomizedSearchCV":
        cv_method = RandomizedSearchCV(tuning_method, param_distributions=param_distributions, cv=cv, verbose=verbose, n_iter=n_iter)
    elif cv_method_name == "GridSearchCV":
        cv_method = GridSearchCV(tuning_method, param_grid=param_distributions, cv=cv, verbose=verbose)
    else:
        raise Exception("Must specify Cross-Validation method in function call.")
    
    cv_method.fit(x_train,y_train)

    print(f"Best score for {tuning_method} after {cv_method_name} tuning: {cv_method.best_score_}")
    print(f"Best parameter combination for {tuning_method} after {cv_method_name} tuning: {cv_method.best_params_}")

    return cv_method

def classification_example():
    df = pd.read_csv("heart-disease.csv")

    np.random.seed(42)

    # Split into x/y
    x = df.drop("target",axis=1)
    y = df.target

    # Split into train/test dataset
    x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.2)

    # Fit and Score different models
    base_model_scores = fit_and_score(x_train,x_test,y_train,y_test)
    print(f"Base Model Scores: {base_model_scores}")

    # ---------------------------------------- TUNING KNN (Manually) ----------------------------------------------------
    tune_hyper_params_KNN(x_train, x_test, y_train, y_test)

    # ---------------------------------------- Hyperparameter tuning with RandomizedSearchCV ------------------------------

    # Create a hyperparam grid for LogisticRegression
    log_reg_grid = {
        "C" : np.logspace(-4,4,20),
        "solver" : ["liblinear"]
    }  

    # Create a hyperparam grid for RandomForestClassifier
    rf_grid = {
        "n_estimators" : [100,200,300,500],
        "max_depth" : [None, 3,5,10],
        "min_samples_split" : np.array([2,4,6]),
        "min_samples_leaf" : np.array([2,4,6])
    }

    # RandomizedSearchCV Tuning for LogisticRegression
    rs_log_reg_model = cv_tuner(LogisticRegression(), x_train, y_train, param_distributions=log_reg_grid, cv=5, verbose=True, n_iter=20, cv_method_name="RandomizedSearchCV")

    # RandomizedSearchCV Tuning for RandomForestClassifier
    rs_rand_forest_model = cv_tuner(RandomForestClassifier(), x_train, y_train, param_distributions=rf_grid, cv=5, verbose=True, n_iter=20, cv_method_name="RandomizedSearchCV")

    # ---------------------------------------- Hyperparameter tuning with GridSearchCV ------------------------------

    # RandomizedSearchCV Tuning for LogisticRegression
    gs_log_reg_model = cv_tuner(LogisticRegression(), x_train, y_train, param_distributions=log_reg_grid, cv=5, verbose=True, n_iter=20, cv_method_name="GridSearchCV")

    # RandomizedSearchCV Tuning for RandomForestClassifier
    gs_rand_forest_model = cv_tuner(RandomForestClassifier(), x_train, y_train, param_distributions=rf_grid, cv=5, verbose=True, n_iter=20, cv_method_name="GridSearchCV")

    # ------------------------------------------ Evaluation ----------------------------------------------------------
    # From here on we will only be using GridSearchCV tuned version of LogisticRegression Model
    # Confusion Matrix
    y_preds = gs_log_reg_model.predict(x_test)
    cm_df = pd.DataFrame(confusion_matrix(y_test,y_preds))
    print(f"Confusion Matrix:")
    print(cm_df)

    # Classification Report
    print("Classification Report")
    print(classification_report(y_test,y_preds))

    # Cross-Validation evaluation scores
    cv_method_list = ["accuracy","precision", "recall", "f1"]
    
    cv_metrics_dict = cross_validate(gs_log_reg_model, x, y, 5, cv_method_list)

    print(f"Cross-Validation evaluation scores for GridSearchCV tuned LogisticRegression:")
    print(cv_metrics_dict)

    # ------------------------------------------ Feature Importance ----------------------------------------------------------
    # Creating and training an instance of LogisticRegression with best params returned from GridSearchCV version
    clf = LogisticRegression(C=0.23357214690901212, solver="liblinear")
    clf.fit(x_train,y_train)

    # Match coef's to feature columns
    feature_dict = dict(zip(df.columns, list(clf.coef_[0])))

    print(f"Feature Importance (Feature to Coefficient mapping): {feature_dict}")
    print(feature_dict)

    # ------------------------------------------ Save Model Using Pickle ----------------------------------------------------------
    export_model_to_file(gs_log_reg_model, "GridSearchCV_LogisticRegression")

if __name__ == "__main__":
    classification_example()