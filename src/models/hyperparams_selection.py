import argparse
import json
from typing import Tuple, Dict, Any
import logging
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score
from sklearn.model_selection import train_test_split
import Optuna

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hyperparameters selection")

    parser.add_argument("-d", "--data_path", type=str, required=True, help="PATH to prepared data")
    parser.add_argument("-o", "--output_path", type=str, required=True, help="Output path for model_params")
    parser.add_argument("-m", "--metrics", type=str, required=True, help="Metrics to use")
    return parser.parse_args()


def load_train_data(data_path: str) -> pd.DataFrame:
    logging.info("Loading training data")
    data = pd.read_csv(data_path)
    data.drop(['target_reg'], axis=1, inplace=True)
    logging.info("Train data loaded")
    return data


def split_data(data: pd.DataFrame, target_col: str, test_size=0.2, random_state=42):
    feature_cols = [col for col in data.columns if col != target_col]
    features = data[feature_cols]
    target = data[target_col]

    if target.dtype == 'object' or target.dtype.name == 'category':
        target = target.astype('category').cat.codes

    return train_test_split(
        features, target, test_size=test_size, random_state=random_state, stratify=target
    )


def hyperparams_selection(features_train: pd.DataFrame,
                          features_test: pd.DataFrame,
                          target_train: pd.DataFrame,
                          target_test: pd.DataFrame,
                          param_grid: Dict) -> Tuple[Any, Any]:

    categorical_features = features_train.select_dtypes(include=['object', 'category']).columns.tolist()

    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', param_grid['iterations'][0], param_grid['iterations'][1]),
            'depth': trial.suggest_int('depth', param_grid['depth'][0], param_grid['depth'][1]),
            'learning_rate': trial.suggest_float('learning_rate', param_grid['learning_rate'][0],
                                                 param_grid['learning_rate'][1], log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', param_grid['l2_leaf_reg'][0],
                                               param_grid['l2_leaf_reg'][1], log=True),
            'border_count': trial.suggest_int('border_count', param_grid['border_count'][0],
                                              param_grid['border_count'][1]),
            'random_seed': 42
        }

        model = CatBoostClassifier(
            loss_function='MultiClass',
            eval_metric='MultiClass',
            custom_metric='Recall',
            verbose=0,
            train_dir=None,
            **params
        )

        model.fit(
            features_train, target_train,
            cat_features=categorical_features,
            eval_set=(features_test, target_test),
            verbose=0
        )

        preds = model.predict(features_test)
        recall = recall_score(target_test, preds, average='macro')

        return recall

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50, n_jobs=-1)

    best_params = study.best_params
    best_params['random_seed'] = 42

    best_model = CatBoostClassifier(
        loss_function='MultiClass',
        eval_metric='MultiClass',
        custom_metric='Recall',
        verbose=0,
        train_dir=None,
        **best_params
    )

    best_model.fit(
        features_train, target_train,
        cat_features=categorical_features,
        eval_set=(features_test, target_test),
        verbose=0
    )

    return best_params, best_model


def save_params(path: str, params: Dict):
    params_json = json.dumps(params, indent=4, ensure_ascii=False)
    with open(path, "w") as f:
        f.write(params_json)


def get_metrics(model, features_test, target_test):
    target_pred = model.predict(features_test)
    target_pred_prob = model.predict_proba(features_test)

    accuracy = accuracy_score(target_test, target_pred)
    f1_macro = f1_score(target_test, target_pred, average='macro')
    f1_weighted = f1_score(target_test, target_pred, average='weighted')
    roc_auc = roc_auc_score(target_test, target_pred_prob, multi_class="ovr", average="macro")
    recall = recall_score(target_test, target_pred, average="macro")

    metrics = {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "roc_auc": roc_auc,
        "recall": recall
    }

    return metrics


def save_metrics(path: str, metrics):
    metrics_json = json.dumps(metrics, indent=4, ensure_ascii=False)
    with open(path, "w") as f:
        f.write(metrics_json)


def model_save(path: str, model: CatBoostClassifier):
    model.save_model(path)


def main():
    target_column = 'target_class'
    param_grid = {
        'iterations': [100, 150],
        'depth': [2],
        'learning_rate': [0.01, 0.05],
        'l2_leaf_reg': [3],
        'border_count': [64]
    }

    args = parse_args()
    train_data = load_train_data(args.data_path)
    features_train, features_test, target_train, target_test = split_data(train_data, target_column)
    _, best_model = hyperparams_selection(features_train, features_test, target_train, target_test, param_grid)
    save_params(args.output_path, best_model.get_params())
    metrics = get_metrics(best_model, features_test, target_test)
    save_metrics(args.metrics, metrics)
    model_save('models/model.cbm', best_model)


if __name__ == '__main__':
    main()