import argparse
import pickle

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


def load_df(path):
    with open(path, "rb") as f:
        data = pickle.load(f)

    # Support both formats:
    # 1) df directly
    # 2) {"df": df, "embeddings": ...}
    if isinstance(data, dict):
        return data["df"]

    return data


def compute_metrics(df):

    y_true = df["true_label"]
    y_pred = df["pred_label"]

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")

    print("\nDataset size:", len(df))
    print("\nAccuracy:", round(acc, 4))
    print("Macro F1:", round(f1_macro, 4))

    print("\nClassification report:\n")
    print(classification_report(y_true, y_pred))

    return y_true, y_pred


def plot_confusion_matrix(y_true, y_pred):

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6,5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues"
    )

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    plt.tight_layout()
    plt.show()


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--confusion", action="store_true")

    args = parser.parse_args()

    df = load_df(args.input)

    y_true, y_pred = compute_metrics(df)

    if args.confusion:
        plot_confusion_matrix(y_true, y_pred)


if __name__ == "__main__":
    main()
