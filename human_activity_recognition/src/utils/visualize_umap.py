import argparse
import pickle
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(path):
    with open(path, "rb") as f:
        data = pickle.load(f)

    return data["df"]


def plot_by_label(df):

    plt.figure(figsize=(7,6))

    sns.scatterplot(
        data=df,
        x="umap_x",
        y="umap_y",
        hue="true_label",
        palette="tab10",
        s=15,
        alpha=0.7
    )

    plt.title("UMAP colored by TRUE label")
    plt.legend(bbox_to_anchor=(1.05,1))
    plt.tight_layout()


def plot_by_prediction(df):

    plt.figure(figsize=(7,6))

    sns.scatterplot(
        data=df,
        x="umap_x",
        y="umap_y",
        hue="pred_label",
        palette="tab10",
        s=15,
        alpha=0.7
    )

    plt.title("UMAP colored by PREDICTED label")
    plt.legend(bbox_to_anchor=(1.05,1))
    plt.tight_layout()


def plot_by_loss(df):

    plt.figure(figsize=(7,6))

    sc = plt.scatter(
        df["umap_x"],
        df["umap_y"],
        c=df["loss"],
        cmap="magma",
        s=15,
        alpha=0.8
    )

    plt.colorbar(sc, label="Loss")

    plt.title("UMAP colored by loss")
    plt.tight_layout()


def plot_errors(df):

    plt.figure(figsize=(7,6))

    correct = df[df["correct"]]
    wrong = df[~df["correct"]]

    plt.scatter(
        correct["umap_x"],
        correct["umap_y"],
        s=10,
        # alpha=0.4,
        label="correct"
    )

    plt.scatter(
        wrong["umap_x"],
        wrong["umap_y"],
        s=10,
        color="red",
        label="wrong"
    )

    plt.legend()
    plt.title("Prediction errors")
    plt.tight_layout()


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)

    args = parser.parse_args()

    df = load_data(args.input)

    print("Samples:", len(df))
    print("Top loss samples:")
    print(df.sort_values("loss", ascending=False).head(20))

    plot_by_label(df)
    plot_by_prediction(df)
    plot_by_loss(df)
    plot_errors(df)

    plt.show()


if __name__ == "__main__":
    main()

