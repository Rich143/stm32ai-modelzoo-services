import mlflow
import pandas as pd
import matplotlib.pyplot as plt

# --- user inputs ---
experiment_name = "HAR_CNN_PAMAP2"
sweep_id = "092B91BE-517C-45F3-B7E1-E7E98FDD3330"
tracking_uri = "http://127.0.0.1:5000"

plot_metrics_fig_1 = ["float_acc_test_set"]

plot_metrics_fig_2 = ["test_set/f1_weighted_avg",
                      "test_set/precision_weighted_avg",
                      "test_set/recall_weighted_avg"]

plot_metrics_fig_3 = ["test_set/cm/cycling-cycling",
                      "test_set/cm/running-running",
                      "test_set/cm/walking-walking",
                      "test_set/cm/stationary-stationary"]

plot_metrics_fig_4 = ["test_set/support/cycling",
                      "test_set/support/running",
                      "test_set/support/walking",
                      "test_set/support/stationary"]

plot_metrics_fig_5 = ["test_set/f1/cycling",
                      "test_set/f1/running",
                      "test_set/f1/walking",
                      "test_set/f1/stationary"]

plot_metrics_fig_6 = ["test_set/precision/cycling",
                      "test_set/precision/running",
                      "test_set/precision/walking",
                      "test_set/precision/stationary"]

plot_metrics_fig_7 = ["test_set/recall/cycling",
                      "test_set/recall/running",
                      "test_set/recall/walking",
                      "test_set/recall/stationary"]

# --------------------

def get_parent_ids(group, input_length):
    # include the parent run id for this input length
    parent_ids = group["tags.mlflow.parentRunId"].dropna().unique()
    if len(parent_ids) == 1:
        parent_id = parent_ids[0]  # one parent per input_length
    else:
        print("Multiple parents for input_length {}".format(input_length))
        for id in parent_ids:
            print("Parent run_name: {}".format(get_run_name(client, id)))
        raise ValueError(f"Multiple parents for input_length {input_length}")

    return parent_id

def get_root_id(client, parent_id):
    parent = client.get_run(parent_id)

    root_id = parent.data.tags.get("mlflow.parentRunId")
    if root_id is None:
        raise ValueError(f"Parent run {parent_id} has no root run")

    return root_id

def get_run_name(client, run_id):
    return client.get_run(run_id).data.tags.get("mlflow.runName")

mlflow.set_tracking_uri(tracking_uri)

client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment_by_name(experiment_name)
experiment_id = experiment.experiment_id

# Fetch all runs
runs = mlflow.search_runs(
    experiment_ids=[experiment_id],
    filter_string=f"tags.sweep_id = '{sweep_id}'",
)

runs = runs[runs["params.seed"].notna()]

# Convert params
runs["params.input_length"] = runs["params.input_length"].astype(float)

# Get numeric metrics
metric_cols = [c for c in runs.columns if c.startswith("metrics.")]
metric_names = [c.replace("metrics.", "") for c in metric_cols]

root_run_id = None

# Group by input_length and compute mean/std for each metric
summary = []
for input_length, group in runs.groupby("params.input_length"):
    data = {"input_length": input_length}

    parent_id = get_parent_ids(group, input_length)
    data["parent_run_id"] = parent_id

    root_id = get_root_id(client,parent_id)
    data["root_run_id"] = root_id
    root_run_id = root_id

    for m in metric_names:
        vals = group[f"metrics.{m}"].dropna()
        data[f"{m}_mean"] = vals.mean()
        data[f"{m}_std"] = vals.std()

    summary.append(data)

summary_df = pd.DataFrame(summary).sort_values("input_length")

# --- plot multiple metrics on the same graph ---

def plot_figure(summary_df, plot_metrics, title, root_run_id = None, save=False):
    plt.figure(figsize=(20,20))  # create a single figure

    for metric in plot_metrics:
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"

        plt.errorbar(
            summary_df["input_length"],
            summary_df[mean_col],
            yerr=summary_df[std_col],
            fmt='-o',
            capsize=4,
            label=metric.title()
        )

    plt.xlabel("Input Length")
    plt.ylabel("Metric Value")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()

    if save:
        filename = f"mlflow/plots/{title}.png".replace(" ", "_")

        plt.savefig(filename)

        if root_run_id is not None:
            client.log_artifact(root_run_id, filename)
        else:
            raise ValueError("root_run_id is None, can't log artifact")
    else:
        plt.show()

    plt.close()


plot_figure(summary_df, plot_metrics_fig_1, "Accuracy vs Input Length (±1 std)", root_run_id=root_run_id, save=True)
plot_figure(summary_df, plot_metrics_fig_2, "F1, precision, recall vs Input Length (±1 std)", root_run_id=root_run_id, save=True)
plot_figure(summary_df, plot_metrics_fig_3, "CM vs Input Length (±1 std)", root_run_id=root_run_id, save=True)
plot_figure(summary_df, plot_metrics_fig_4, "Support vs Input Length (±1 std)", root_run_id=root_run_id, save=True)
plot_figure(summary_df, plot_metrics_fig_5, "F1 (by class) vs Input Length (±1 std)", root_run_id=root_run_id, save=True)
plot_figure(summary_df, plot_metrics_fig_6, "Precision (by class) vs Input Length (±1 std)", root_run_id=root_run_id, save=True)
plot_figure(summary_df, plot_metrics_fig_7, "Recall (by class) vs Input Length (±1 std)", root_run_id=root_run_id, save=True)


for _, row in summary_df.iterrows():
    parent_run_id = row["parent_run_id"]
    parent_run_name = get_run_name(client, parent_run_id)

    # start a nested run under the parent (optional, or just log to parent)
    # here we log directly to parent
    for metric in metric_names:
        mean_val = row[f"{metric}_mean"]
        std_val = row[f"{metric}_std"]

        # log as separate metrics for clarity
        client.log_metric(parent_run_id, f"{metric}_mean", mean_val)
        client.log_metric(parent_run_id, f"{metric}_std", std_val)
