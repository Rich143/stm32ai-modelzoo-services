import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="Plot MLflow sweep results.")

    parser.add_argument(
        "--experiment_name",
        type=str,
        default="HAR_CNN_PAMAP2",
        help="Name of the MLflow experiment."
    )

    parser.add_argument(
        "--sweep_id",
        type=str,
        default="HAR_CNN_PAMAP2_251019_153022_dd5a65",
        help="Sweep ID used as a tag filter."
    )

    parser.add_argument(
        "--tracking_uri",
        type=str,
        default="http://127.0.0.1:5000",
        help="MLflow tracking server URI."
    )

    parser.add_argument(
        "--sweep_parameter",
        type=str,
        default="input_length",
        help="Sweep parameter used to group runs."
    )

    return parser.parse_args()

# --- user inputs ---
args = parse_args()
experiment_name = args.experiment_name
sweep_id = args.sweep_id
tracking_uri = args.tracking_uri
sweep_parameter_name = args.sweep_parameter

metric_templates = {
    "fig_1": {
        "metric_list": ["float_acc_{ds}_set"],
        "figure_name": "Accuracy vs {sweep_param} (±1 std)"
    },
    "fig_2": {
        "metric_list": [
            "{ds}_set/f1_weighted_avg",
            "{ds}_set/precision_weighted_avg",
            "{ds}_set/recall_weighted_avg",
        ],
        "figure_name": "F1, precision, recall vs {sweep_param} (±1 std)"
    },
    "fig_3": {
        "metric_list": [
            "{ds}_set/cm/cycling-cycling",
            "{ds}_set/cm/running-running",
            "{ds}_set/cm/walking-walking",
            "{ds}_set/cm/stationary-stationary",
        ],
        "figure_name": "CM vs {sweep_param} (±1 std)"
    },
    "fig_4": {
        "metric_list": [
            "{ds}_set/support/cycling",
            "{ds}_set/support/running",
            "{ds}_set/support/walking",
            "{ds}_set/support/stationary",
        ],
        "figure_name": "Support vs {sweep_param} (±1 std)"
    },
    "fig_5": {
        "metric_list": [
            "{ds}_set/f1/cycling",
            "{ds}_set/f1/running",
            "{ds}_set/f1/walking",
            "{ds}_set/f1/stationary",
        ],
        "figure_name": "F1 (by class) vs {sweep_param} (±1 std)"
    },
    "fig_6": {
        "metric_list": [
            "{ds}_set/precision/cycling",
            "{ds}_set/precision/running",
            "{ds}_set/precision/walking",
            "{ds}_set/precision/stationary",
        ],
        "figure_name": "Precision (by class) vs {sweep_param} (±1 std)"
    },
    "fig_7": {
        "metric_list": [
            "{ds}_set/recall/cycling",
            "{ds}_set/recall/running",
            "{ds}_set/recall/walking",
            "{ds}_set/recall/stationary",
        ],
        "figure_name": "Recall (by class) vs {sweep_param} (±1 std)"
    },
}

# --------------------

def get_parent_ids(group, sweep_parameter_name, sweep_parameter_value):
    # include the parent run id for this input length
    parent_ids = group["tags.mlflow.parentRunId"].dropna().unique()
    if len(parent_ids) == 1:
        parent_id = parent_ids[0]  # one parent per input_length
    else:
        print(f"Multiple parents for {sweep_parameter_name} = {sweep_parameter_value}")
        for id in parent_ids:
            print("Parent run_name: {}".format(get_run_name(client, id)))
        raise ValueError(f"Multiple parents for {sweep_parameter_name} = {sweep_parameter_value}")

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

if len(runs) == 0:
    print(f"No runs found for experiment {experiment_name} and sweep {sweep_id}")
    sys.exit(1)

truthy = ["true", "1", "yes", "t"]

# Filter out non-worker runs
runs = runs[runs["tags.worker_run"].str.lower().isin(truthy)]

# Convert params
runs[f"params.{sweep_parameter_name}"] = runs[f"params.{sweep_parameter_name}"].astype(float)

# Get numeric metrics
metric_cols = [c for c in runs.columns if c.startswith("metrics.")]
metric_names = [c.replace("metrics.", "") for c in metric_cols]

root_run_id = None

# Group by sweep param and compute mean/std for each metric
summary = []
for sweep_parameter_value, group in runs.groupby(f"params.{sweep_parameter_name}"):
    data = {sweep_parameter_name: sweep_parameter_value}

    parent_id = get_parent_ids(group,
                               sweep_parameter_name,
                               sweep_parameter_value)
    data["parent_run_id"] = parent_id

    root_id = get_root_id(client,parent_id)
    data["root_run_id"] = root_id
    root_run_id = root_id

    for m in metric_names:
        vals = group[f"metrics.{m}"].dropna()
        data[f"{m}_mean"] = vals.mean()
        data[f"{m}_std"] = vals.std()

    summary.append(data)

summary_df = pd.DataFrame(summary).sort_values(sweep_parameter_name)

# --- plot multiple metrics on the same graph ---

def plot_figure(summary_df, plot_metrics, title, root_run_id = None, save=False):
    plt.figure(figsize=(20,20))  # create a single figure

    for metric in plot_metrics:
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"

        plt.errorbar(
            summary_df[sweep_parameter_name],
            summary_df[mean_col],
            yerr=summary_df[std_col],
            fmt='-o',
            capsize=4,
            label=metric.title()
        )

    plt.xlabel(sweep_parameter_name)
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

def plot_all_metrics(summary_df, root_run_id, sweep_parameter_name="param", dataset="validation", save=False):
    ds = dataset.lower()

    # Human-readable name for titles
    ds_name = "Test Set" if ds == "test" else "Validation Set"

    for fig_id, cfg in metric_templates.items():

        # Expand metric templates like "{ds}_set/f1/cycling"
        metric_list = [m.format(ds=ds) for m in cfg["metric_list"]]

        # Insert sweep parameter name and append dataset
        fig_title = (
            cfg["figure_name"].format(sweep_param=sweep_parameter_name)
            + f" ({ds_name})"
        )

        # Call your actual plot function
        plot_figure(
            summary_df,
            metric_list,
            fig_title,
            root_run_id=root_run_id,
            save=save,
        )

plot_all_metrics(summary_df,
                 root_run_id,
                 dataset="validation",
                 sweep_parameter_name=sweep_parameter_name,
                 save=True)
plot_all_metrics(summary_df,
                 root_run_id,
                 dataset="test",
                 sweep_parameter_name=sweep_parameter_name,
                 save=True)

for _, row in summary_df.iterrows():
    parent_run_id = row["parent_run_id"]
    parent_run_name = get_run_name(client, parent_run_id)

    # Log metrics to parent
    for metric in metric_names:
        mean_val = row[f"{metric}_mean"]
        std_val = row[f"{metric}_std"]

        # log as separate metrics for clarity
        client.log_metric(parent_run_id, f"{metric}_mean", mean_val)
        client.log_metric(parent_run_id, f"{metric}_std", std_val)
