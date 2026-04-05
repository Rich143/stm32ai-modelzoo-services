import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import umap
import pickle
from cleanlab import Datalab

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
        "embeddings": list(embeddings),
        "preds": list(preds),
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

def get_hard_vs_easy_samples(
    df,
    target_class,
    n=5,
    loss_threshold=0.5,
    random_state=42,
    hard_mode="top_loss"  # "top_loss", "misclassified", "high_conf_error"
):
    """
    Select hard and easy samples for a given class.

    Args:
        df: dataframe with sample_id, true_label, pred_label, confidence, loss
        target_class: class to inspect
        n: number of samples per group
        loss_threshold: max loss for easy samples
        random_state: for reproducibility
        hard_mode:
            - "top_loss": highest loss samples
            - "misclassified": only wrong predictions
            - "high_conf_error": wrong + high confidence

    Returns:
        hard_df: dataframe of hard samples
        easy_df: dataframe of easy samples
    """

    class_df = df[df.true_label == target_class]

    # --- HARD samples ---
    if hard_mode == "top_loss":
        hard = class_df.sort_values("loss", ascending=False)

    elif hard_mode == "misclassified":
        hard = class_df[class_df.pred_label != target_class] \
            .sort_values("loss", ascending=False)

    elif hard_mode == "high_conf_error":
        hard = class_df[
            (class_df.pred_label != target_class) &
            (class_df.confidence > 0.9)
        ].sort_values("loss", ascending=False)

    else:
        raise ValueError("Invalid hard_mode")

    hard = hard.head(n)

    # --- EASY samples ---
    easy_pool = class_df[
        (class_df.pred_label == target_class) &
        (class_df.loss < loss_threshold)
    ]

    if len(easy_pool) < n:
        print(f"Warning: only {len(easy_pool)} easy samples available")
        easy = easy_pool
    else:
        easy = easy_pool.sample(n, random_state=random_state)

    return hard.reset_index(drop=True), easy.reset_index(drop=True)


def plot_time_domain_comparison(hard_df, easy_df, X):
    n = max(len(hard_df), len(easy_df))

    fig, axes = plt.subplots(n, 2, figsize=(14, 3*n))

    if n == 1:
        axes = [axes]

    for i in range(n):

        # ----- HARD -----
        if i < len(hard_df):
            row = hard_df.iloc[i]
            sample = X[int(row.sample_id)].squeeze()

            ax = axes[i][0]
            ax.plot(sample[:,0], label="x")
            ax.plot(sample[:,1], label="y")
            ax.plot(sample[:,2], label="z")

            ax.set_title(
                f"HARD | pred={row.pred_label} | conf={row.confidence:.2f} | loss={row.loss:.3f}"
            )
            ax.legend()

        else:
            axes[i][0].axis("off")

        # ----- EASY -----
        if i < len(easy_df):
            row = easy_df.iloc[i]
            sample = X[int(row.sample_id)].squeeze()

            ax = axes[i][1]
            ax.plot(sample[:,0], label="x")
            ax.plot(sample[:,1], label="y")
            ax.plot(sample[:,2], label="z")

            ax.set_title(
                f"EASY | pred={row.pred_label} | conf={row.confidence:.2f} | loss={row.loss:.3f}"
            )
            ax.legend()

        else:
            axes[i][1].axis("off")

    plt.suptitle("Time Domain: Hard vs Easy")
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.01)

def plot_fft_comparison(hard_df, easy_df, X, fs=1.0, pad_factor=4):
    n = max(len(hard_df), len(easy_df))

    fig, axes = plt.subplots(n, 2, figsize=(14, 3*n))

    if n == 1:
        axes = [axes]

    for i in range(n):

        def compute_fft(sample):
            N = sample.shape[0]
            N_pad = N * pad_factor

            freqs = np.fft.rfftfreq(N_pad, d=1/fs)
            fft_vals = np.abs(np.fft.rfft(sample, n=N_pad, axis=0))

            return freqs, fft_vals

        if i < len(hard_df):
            row = hard_df.iloc[i]
            sample = X[int(row.sample_id)].squeeze()

            freqs, fft_vals = compute_fft(sample)

            ax = axes[i][0]
            for j, axis_name in enumerate(["x","y","z"]):
                ax.plot(freqs, fft_vals[:,j], label=axis_name)

            ax.set_title(
                f"HARD | pred={row.pred_label} | conf={row.confidence:.2f} | loss={row.loss:.3f}"
            )
            ax.set_xlabel("Frequency")
            ax.legend()

        else:
            axes[i][0].axis("off")

        if i < len(easy_df):
            row = easy_df.iloc[i]
            sample = X[int(row.sample_id)].squeeze()

            freqs, fft_vals = compute_fft(sample)

            ax = axes[i][1]
            for j, axis_name in enumerate(["x","y","z"]):
                ax.plot(freqs, fft_vals[:,j], label=axis_name)

            ax.set_title(
                f"EASY | pred={row.pred_label} | conf={row.confidence:.2f} | loss={row.loss:.3f}"
            )
            ax.set_xlabel("Frequency")
            ax.legend()

        else:
            axes[i][1].axis("off")

    plt.suptitle("FFT (Zero-Padded): Hard vs Easy")
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.01)



from scipy.signal import stft


def plot_stft_comparison(hard_df, easy_df, X, fs=1.0, nperseg=16, pad_factor=4):
    n = max(len(hard_df), len(easy_df))

    fig, axes = plt.subplots(n, 2, figsize=(14, 3*n))

    if n == 1:
        axes = [axes]

    for i in range(n):

        def compute_stft(sample):
            magnitude = np.linalg.norm(sample, axis=1)

            nfft = nperseg * pad_factor

            f, t, Zxx = stft(
                magnitude,
                fs=fs,
                nperseg=nperseg,
                nfft=nfft
            )

            return f, t, np.abs(Zxx)

        if i < len(hard_df):
            row = hard_df.iloc[i]
            sample = X[int(row.sample_id)].squeeze()

            f, t, Z = compute_stft(sample)

            ax = axes[i][0]
            im = ax.pcolormesh(t, f, Z, shading='gouraud')
            ax.set_title(
                f"HARD | pred={row.pred_label} | conf={row.confidence:.2f} | loss={row.loss:.3f}"
            )
            ax.set_ylabel("Freq")

        else:
            axes[i][0].axis("off")

        if i < len(easy_df):
            row = easy_df.iloc[i]
            sample = X[int(row.sample_id)].squeeze()

            f, t, Z = compute_stft(sample)

            ax = axes[i][1]
            im = ax.pcolormesh(t, f, Z, shading='gouraud')
            ax.set_title(
                f"EASY | pred={row.pred_label} | conf={row.confidence:.2f} | loss={row.loss:.3f}"
            )

        else:
            axes[i][1].axis("off")

    plt.suptitle("STFT (Zero-Padded): Hard vs Easy")
    plt.tight_layout()
    plt.show()

def run_cleanlab(df: pd.DataFrame):
    preds = np.stack(df["preds"].values)        # shape (N, num_classes)
    embeddings = np.stack(df["embeddings"].values)  # shape (N, embedding_dim)
    true_labels = df["true_label"].values       # shape (N,)

    lab = Datalab(data=df, label_name="true_label")

    lab.find_issues(
        pred_probs=preds,
        features=embeddings
    )

    lab.report()


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

    # hard_df, easy_df = get_hard_vs_easy_samples(df, target_class=1, n=5)

    # plot_time_domain_comparison(hard_df, easy_df, X)
    # plot_fft_comparison(hard_df, easy_df, X, fs=50)
    # plot_stft_comparison(hard_df, easy_df, X, fs=50)

    run_cleanlab(df)

if __name__ == "__main__":
    main()
