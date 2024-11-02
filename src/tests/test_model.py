import pandas as pd
import argparse

from sklearn.metrics import accuracy_score, f1_score

ACCURACY_THRESHOLD = 0.9
F1_THRESHOLD = 0.8

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--predictions', type=str, required=True)
    parser.add_argument('-t', '--test', type=str, required=True)
    return parser.parse_args()


def test_model(pred_path, real_path):
    predicted = pd.read_csv(pred_path)
    real = pd.read_csv(real_path)

    accuracy = accuracy_score(real['target_class'], predicted)
    f1_macro = f1_score(real['target_class'], predicted, average='macro')

    assert accuracy >= ACCURACY_THRESHOLD, f"Accuracy {accuracy} ниже порога {ACCURACY_THRESHOLD}"
    assert f1_macro >= F1_THRESHOLD, f"F1 {f1_macro} ниже порога {F1_THRESHOLD}"

    print("Tests passed!")
    print(f"Accuracy: {accuracy}")
    print(f"F1: {f1_macro}")


if __name__ == "__main__":
    args = parse_args()
    test_model(args.predictions, args.test)
