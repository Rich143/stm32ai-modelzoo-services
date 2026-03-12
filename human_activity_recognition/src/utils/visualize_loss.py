import argparse
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_df(path):
    with open(path, "rb") as f:
        df = pickle.load(f)
    return df


def add_columns(df):
    df["correct"] = df["true_label"] == df["pred_label"]
    return df


def plot_loss_distribution(df):
    plt.figure(figsize=(6,4))
    sns.histplot(df["loss"], bins=50)
    plt.title("Loss distribution")
    plt.xlabel("Cross entropy loss")
    plt.ylabel("Count")
    plt.tight_layout()


def plot_loss_vs_confidence(df):
    plt.figure(figsize=(6,4))

    sns.scatterplot(
        data=df,
        x="confidence",
        y="loss",
        hue="correct",
        alpha=0.6
    )

    plt.title("Loss vs prediction confidence")
    plt.tight_layout()


def plot_loss_by_class(df):
    plt.figure(figsize=(6,4))

    sns.boxplot(
        data=df,
        x="true_label",
        y="loss"
    )

    plt.title("Loss distribution by true class")
    plt.tight_layout()


def print_top_samples(df, n=20):
    print("\nTop high-loss samples:\n")
    print(
        df.head(n)[
            ["sample_id","true_label","pred_label","confidence","loss"]
        ]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Pickle dataframe file")
    args = parser.parse_args()

    df = load_df(args.input)
    df = add_columns(df)

    print(f"Loaded {len(df)} samples")

    print_top_samples(df)

    plot_loss_distribution(df)
    plot_loss_vs_confidence(df)
    plot_loss_by_class(df)

    plt.show()


if __name__ == "__main__":
    main()

