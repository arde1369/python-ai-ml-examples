from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import numpy as np

from Evaluation.EvaluationHelper import cross_validate, evaluate_preds

def create_column_transformer():
    # Define different features and transformer pipeline
    categorical_features = ["Make", "Colour"]
    categorical_transformer = Pipeline(steps = [
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")) # Encodes the non-numeric values into numerics
        ])

    door_feature = ["Doors"]
    door_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value=4)),
        ])

    numeric_feature = ["Odometer (KM)"]
    numeric_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
        ])
    
    return ColumnTransformer(transformers= [
                                    ("cat", categorical_transformer, categorical_features),
                                    ("door", door_transformer, door_feature),
                                    ("num", numeric_transformer, numeric_feature)
                                    ])

def regression_example():
    # Setup Random seed
    np.random.seed(42)

    # Import data and drop rows with missing labels
    data = pd.read_csv("car-sales-extended-missing-data.csv")
    data.dropna(subset="Price", inplace=True)

    # Setup the preprocessing steps (Fill missing values and then convert to numbers)
    preprocessor = create_column_transformer()

    # Create a preprocessing and modeling pipeline
    model = Pipeline(steps=[("preprocessor", preprocessor),
                            ("model", RandomForestRegressor())  
                        ])
    # create x/y
    x = data.drop("Price", axis=1)
    y = data["Price"]

    x_t= preprocessor.fit_transform(x)

    # Split into train and test
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

    # Fit and score the model
    model.fit(x_train, y_train)
    model.score(x_test,y_test)

    # Evaluate baseline model
    baseline_preds = model.predict(x_test)
    baseline_metrics = evaluate_preds(y_test,baseline_preds, 
         {
            "R Squared" : r2_score,
            "Mean Squared Error (MSE)" : mean_squared_error,
            "Mean Absolute Error (MAE)" : mean_absolute_error
        })

    # Cross validate
    cross_validated_metrics_base = cross_validate(RandomForestRegressor(),x_t,y,5,["r2","neg_mean_squared_error","neg_mean_absolute_error"])

    # Tune the model
    pipe_grid = {
    "preprocessor__num__imputer__strategy" :  ["mean","median"],
    "model__n_estimators" : [100,1000],
    "model__max_depth" : [None,5],
    "model__max_features" : ["log2","sqrt"],
    "model__min_samples_split" : [2,4]
    }

    tuned_model = GridSearchCV(model,param_grid=pipe_grid, cv=5, verbose=2)
    tuned_model.fit(x_train, y_train)

    # Evaluate tuned model
    tuned_preds = tuned_model.predict(x_test)

    tuned_model_metrics = evaluate_preds(y_test,tuned_preds, {
            "R Squared" : r2_score,
            "Mean Squared Error (MSE)" : mean_squared_error,
            "Mean Absolute Error (MAE)" : mean_absolute_error
        })

    # Cross validate
    cross_validated_metrics_tuned = cross_validate(RandomForestRegressor(),x_t,y,5,["r2","neg_mean_squared_error","neg_mean_absolute_error"])

    metrics_dict_base = {
        "Baseline Metrics" : baseline_metrics,
        "Tuned Metrics" : tuned_model_metrics
    }

    metrics_dict_tuned = {
        "CV Baseline Metrics" : cross_validated_metrics_base,
        "CV Tuned Metrics" : cross_validated_metrics_tuned
    }

    print(metrics_dict_base)
    print("---------------------------")
    print(metrics_dict_tuned)

if __name__ == "__main__":
    regression_example()

    