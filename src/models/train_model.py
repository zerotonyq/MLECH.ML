import argparse
import json
import logging

from catboost import CatBoostClassifier
from hyperparams_selection import load_train_data

def parse_args():
    """
    Parses command-line arguments for the script.

    Arguments:
        -d, --dataset (str): Path to the dataset.
        -p, --params (str): Path to the JSON file with model parameters.
        -o, --output (str): Path to save the trained model.

    Returns:
        argparse.Namespace: An object containing the parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Train classification model on best params")
    parser.add_argument("-d", "--dataset", type=str, required=True, help="Dataset path")
    parser.add_argument("-p", "--params", type=str, required=True, help="Model parameters")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output model path")

    return parser.parse_args()


def load_params(params_path):
    """
    Loads model parameters from a JSON file.

    Args:
        params_path (str): Path to the JSON file with model parameters.

    Returns:
        dict: A dictionary of model parameters.

    Raises:
        FileNotFoundError: If the file is not found.
        JSONDecodeError: If the file contents are not valid JSON.
    """
    with open(params_path, "r", encoding='utf-8') as f:
        params = json.load(f)

    logging.info("Params loaded")
    return params


def model_fit(params, features, target):
    """
    Initializes and trains a CatBoost classifier with the specified parameters.

    Args:
        params (dict): Parameters for initializing the CatBoostClassifier.
        features (pd.DataFrame): Features for training the model.
        target (pd.Series): Target variable (class labels).

    Returns:
        CatBoostClassifier: The trained CatBoost model.

    Note:
        Identifies categorical features in `features` and passes them to the model for optimal training.
    """
    model = CatBoostClassifier(**params)
    logging.info("Model initialized")
    categorical_features = features.select_dtypes(include=['object', 'category']).columns.tolist()
    model.fit(features, target, cat_features=categorical_features)
    logging.info('Model fitted')
    return model


def model_save(model, path):
    """
    Saves the trained model to the specified file.

    Args:
        model (CatBoostClassifier): The trained CatBoost model.
        path (str): Path to save the model.

    Returns:
        None

    Note:
        Uses the `save_model` method of CatBoostClassifier to save the model.
    """
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