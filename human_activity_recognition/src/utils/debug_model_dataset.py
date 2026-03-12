import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

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
        random_state=42
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

def main():

    parser = argparse.ArgumentParser(
        description="Find high-loss samples for a trained Keras model"
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Path to .keras model file"
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to dataset .npz file containing X and Y arrays"
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

    print(f"Loading model from: {args.model}")
    model = load_model(args.model)

    print(f"Loading dataset from: {args.dataset}")
    X, Y = load_dataset(args.dataset)

    top_samples, full_df = get_high_loss_samples_numpy(
        model,
        X,
        Y,
        top_k=args.top_k,
        batch_size=args.batch_size
    )

    print("\nTop high-loss samples:\n")
    print(top_samples.to_string(index=False))


if __name__ == "__main__":
    main()
