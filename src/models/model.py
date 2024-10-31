import logging

import pandas as pd
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score, f1_score
import mlflow.catboost
from mlflow.models import infer_signature
import joblib

data = pd.read_csv('../../data/processed/processed_train.csv')
data.drop(['target_reg'], axis=1, inplace=True)

target = 'target_class'
features = [col for col in data.columns if col != target]

X = data[features]
y = data[target]

if y.dtype == 'object' or y.dtype.name == 'category':
    y = y.astype('category').cat.codes

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

param_grid = {
    'iterations': [150],
    'depth': [2],
    'learning_rate': [0.01],
    'l2_leaf_reg': [1, 3],
    'border_count': [64]
}

cat_model = CatBoostClassifier(
    loss_function='MultiClass',
    eval_metric='Accuracy',
    verbose=0,
    random_seed=42
)

grid_search = GridSearchCV(
    estimator=cat_model,
    param_grid=param_grid,
    scoring='accuracy',
    cv=3,
    verbose=1,
    n_jobs=-1
)

tracking_uri=os.getenv('MLFLOW_TRACKING_URI')
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("CatBoost_Multiclass_Classification_Experiment")

with mlflow.start_run():
    grid_search.fit(
        X_train, y_train,
        cat_features=categorical_features,
        eval_set=(X_test, y_test),
        verbose=0
    )

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    y_pred = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')

    signature = infer_signature(X_train, best_model.predict(X_train))

    logging.info(f"Лучшая точность (Accuracy): {accuracy}")
    logging.info(f"F1 Macro: {f1_macro}")
    logging.info(f"F1 Weighted: {f1_weighted}")

    mlflow.log_params(best_params)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_macro", f1_macro)
    mlflow.log_metric("f1_weighted", f1_weighted)

    mlflow.catboost.log_model(cb_model=best_model,
                              artifact_path="catboost_model",
                              signature=signature,
                              input_example=X_train)

    best_model.save_model('catboost_model.cbm')
    logging.info("Модель сохранена в файл 'catboost_model.cbm'")

    joblib.dump(best_model, 'catboost_model.joblib')
    logging.info("Модель также сохранена с помощью joblib в файл 'catboost_model.joblib'")
