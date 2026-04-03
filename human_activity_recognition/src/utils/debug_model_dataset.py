import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import umap
import pickle

def get_embedding_umap(
    model,
    X,
    Y,
    batch_size=256,
    penultimate_layer=None,
    output_file="embedding_umap.pkl",
):

    # If layer name not specified, assume second-last layer
    if penultimate_layer is None:
        penultimate_layer = model.layers[-2].name

    embedding_model = tf.keras.Model(
        inputs=model.input,
        outputs=model.get_layer(penultimate_layer).output
    )

    loss_fn = tf.keras.losses.CategoricalCrossentropy(reduction="none")

    preds = model.predict(X, batch_size=batch_size, verbose=0)

    losses = loss_fn(Y, preds).numpy()

    embeddings = embedding_model.predict(X, batch_size=batch_size, verbose=0)

    true_classes = np.argmax(Y, axis=1)
    pred_classes = np.argmax(preds, axis=1)
    confidences = np.max(preds, axis=1)

    print("Running UMAP...")

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=30,
        min_dist=0.1,
        metric="euclidean",
        random_state=42,
        verbose=True
    )

    embedding_2d = reducer.fit_transform(embeddings)

    df = pd.DataFrame({
        "sample_id": np.arange(len(X)),
        "true_label": true_classes,
        "pred_label": pred_classes,
        "confidence": confidences,
        "loss": losses,
        "umap_x": embedding_2d[:,0],
        "umap_y": embedding_2d[:,1],
    })

    df["correct"] = df["true_label"] == df["pred_label"]

    data = {
        "df": df,
        "embeddings": embeddings
    }

    with open(output_file, "wb") as f:
        pickle.dump(data, f)

    print(f"Saved to {output_file}")

    return df

def get_sample_losses(model, X, Y, batch_size=256):

    loss_fn = tf.keras.losses.CategoricalCrossentropy(
        reduction="none"
    )

    preds = model.predict(X, batch_size=batch_size, verbose=0)

    losses = loss_fn(Y, preds).numpy()

    true_classes = np.argmax(Y, axis=1)
    pred_classes = np.argmax(preds, axis=1)
    confidences = np.max(preds, axis=1)

    df = pd.DataFrame({
        "sample_id": np.arange(len(X)),
        "true_label": true_classes,
        "pred_label": pred_classes,
        "confidence": confidences,
        "loss": losses,
    })

    df_sorted = df.sort_values("loss", ascending=False)

    return df_sorted

def load_pickle_data(path):
    with open(path, "rb") as f:
        data = pickle.load(f)

    return data["df"]


def get_n_highest_loss_samples(df_sorted, top_k=50):
    return df_sorted.head(top_k)

# def get_high_loss_samples_numpy(model, X, Y, top_k=50, batch_size=256):

    # loss_fn = tf.keras.losses.CategoricalCrossentropy(
        # reduction="none"
    # )

    # preds = model.predict(X, batch_size=batch_size, verbose=0)

    # losses = loss_fn(Y, preds).numpy()

    # true_classes = np.argmax(Y, axis=1)
    # pred_classes = np.argmax(preds, axis=1)
    # confidences = np.max(preds, axis=1)

    # df = pd.DataFrame({
        # "sample_id": np.arange(len(X)),
        # "true_label": true_classes,
        # "pred_label": pred_classes,
        # "confidence": confidences,
        # "loss": losses,
    # })

    # df_sorted = df.sort_values("loss", ascending=False)

    # return df_sorted.head(top_k), df

def load_dataset(dataset_path):
    data = np.load(dataset_path)

    X = data["X"]
    Y = data["Y"]

    X = X.astype(np.float32)
    Y = Y.astype(np.float32)

    print(f"Dataset size: {len(X)} samples")

    return X, Y

def load_model(model_path):
    model = tf.keras.models.load_model(model_path, compile=True)

    return model

import numpy as np
import matplotlib.pyplot as plt


def plot_accel_samples(
    df,
    X,
    n=5,
    mode="top_loss",   # "top_loss", "random", "index"
    indices=None,
    figsize=(12, 3)
):
    """
    Plot accelerometer samples from dataset.

    Args:
        df: DataFrame with sample_id, true_label, pred_label, confidence, loss
        X: numpy array (N, 48, 3, 1)
        n: number of samples to plot
        mode:
            - "top_loss": highest loss samples
            - "random": random samples
            - "index": use provided indices
        indices: list of indices (used if mode="index")
    """

    if mode == "top_loss":
        selected = df.sort_values("loss", ascending=False).head(n)

    elif mode == "random":
        selected = df.sample(n)

    elif mode == "index":
        assert indices is not None, "Provide indices for mode='index'"
        selected = df.iloc[indices]

    else:
        raise ValueError("Invalid mode")

    selected = selected.reset_index(drop=True)

    fig, axes = plt.subplots(n, 1, figsize=(figsize[0], figsize[1] * n))

    if n == 1:
        axes = [axes]

    for i, row in selected.iterrows():

        sample_id = int(row["sample_id"])
        sample = X[sample_id].squeeze()  # (48,3)

        ax = axes[i]

        ax.plot(sample[:,0], label="x")
        ax.plot(sample[:,1], label="y")
        ax.plot(sample[:,2], label="z")

        title = (
            f"ID: {sample_id} | "
            f"true: {row['true_label']} | "
            f"pred: {row['pred_label']} | "
            f"conf: {row['confidence']:.2f} | "
            f"loss: {row['loss']:.3f}"
        )

        ax.set_title(title)
        ax.legend(loc="upper right")

    plt.tight_layout()
    plt.show()

def main():

    parser = argparse.ArgumentParser(
        description="Find high-loss samples"
    )

    parser.add_argument(
        "--pickle-dataframe",
        required=True,
        help="Path to pickle dataframe file"
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to dataset file"
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Number of highest loss samples to print"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=256
    )

    args = parser.parse_args()

    df = load_pickle_data(args.pickle_dataframe)

    df_sorted = df.sort_values("loss", ascending=False)

    dataset = load_dataset(args.dataset)
    X = dataset[0]
    Y = dataset[1]

    top_loss_samples = get_n_highest_loss_samples(
        df_sorted,
        top_k=args.top_k
    )

    print("\nTop high-loss samples:\n")
    print(top_loss_samples.to_string(index=False))

    plot_accel_samples(df, X, n=5, mode="top_loss")


if __name__ == "__main__":
    main()
