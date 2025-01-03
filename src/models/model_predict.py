from catboost import CatBoostClassifier
import pandas as pd
import argparse

def parse_args():
    """
    Parses command-line arguments for the script.

    Arguments:
        -m, --model_path (str): Path to the trained model file.
        -d, --data_path (str): Path to the prepared data for prediction.
        -o, --output_path (str): Path to save the predictions.
        -p, --probabilities (flag): If specified, predicts probabilities instead of class labels.

    Returns:
        argparse.Namespace: An object containing the parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Launch catboost model and predict")

    parser.add_argument("-m", "--model_path", type=str, required=True, help="PATH to model file")
    parser.add_argument("-d", "--data_path", type=str, required=True, help="PATH to prepared data")
    parser.add_argument("-o", "--output_path", type=str, required=True, help="Output path")

    parser.add_argument("-p", "--probabilities", action="store_true", help="Flag for predicting probabilities")

    return parser.parse_args()


def load_model(model_path):
    """
    Loads a pre-trained CatBoost model from the specified path.

    Args:
        model_path (str): Path to the model file.

    Returns:
        CatBoostClassifier: Loaded CatBoost model.

    Raises:
        CatBoostError: If the model cannot be loaded from the provided path.
    """
    mdl = CatBoostClassifier()
    return mdl.load_model(model_path)


def pred(mdl, dt):
    """
    Predicts class labels for the provided data using the loaded model.

    Args:
        mdl (CatBoostClassifier): The loaded CatBoost model.
        dt (pd.DataFrame): Data for prediction.

    Returns:
        np.ndarray: Predicted class labels.
    """
    return mdl.predict(dt)


def pred_proba(mdl, dt):
    """
    Predicts class probabilities for the provided data using the loaded model.

    Args:
        mdl (CatBoostClassifier): The loaded CatBoost model.
        dt (pd.DataFrame): Data for prediction.

    Returns:
        np.ndarray: Predicted class probabilities.
    """
    return mdl.predict_proba(dt)


def main():
    args = parse_args()
    model = load_model(args.model_path)
    data = pd.read_csv(args.data_path)

    if 'target_class' in data.columns:
        data.drop(['target_class'], axis=1, inplace=True)

    if 'target_reg' in data.columns:
        data.drop(['target_reg'], axis=1, inplace=True)

    if args.probabilities:
        prediction = pred_proba(model, data)
    else:
        prediction = pred(model, data)

    output_df = pd.DataFrame(prediction, columns=["prediction"])
    output_df.to_csv(args.output_path, index=False)
    print(f"Predictions saved into file: {args.output_path}")

if __name__ == '__main__':
    main()
