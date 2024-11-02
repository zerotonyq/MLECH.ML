import argparse
import json
import logging

from catboost import CatBoostClassifier
from hyperparams_selection import load_train_data

def parse_args():
    parser = argparse.ArgumentParser(description="Train classification model on best params")
    parser.add_argument("-d", "--dataset", type=str, required=True, help="Dataset path")
    parser.add_argument("-p", "--params", type=str, required=True, help="Model parameters")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output model path")

    return parser.parse_args()


def load_params(params_path):
    with open(params_path, "r", encoding='utf-8') as f:
        params = json.load(f)

    logging.info("Params loaded")
    return params


def model_fit(params, features, target):
    model = CatBoostClassifier(**params)
    logging.info("Model initialized")
    categorical_features = features.select_dtypes(include=['object', 'category']).columns.tolist()
    model.fit(features, target, cat_features=categorical_features)
    logging.info('Model fitted')
    return model


def model_save(model, path):
    model.save_model(path)
    logging.info(f'Model saved to file: {path}')

def main():
    args = parse_args()

    data = load_train_data(args.dataset)
    target = data['target_class'].copy(deep=True)
    data.drop(['target_class'], axis=1, inplace=True)
    params = load_params(args.params)
    model = model_fit(params, data, target)
    model_save(model, args.output)


if __name__ == "__main__":
    main()