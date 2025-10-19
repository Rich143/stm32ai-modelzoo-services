import mlflow
import pandas as pd

experiment_name = "HAR_CNN_PAMAP2"
sweep_id = "092B91BE-517C-45F3-B7E1-E7E98FDD3330"
tracking_uri = "http://127.0.0.1:5000"

mlflow.set_tracking_uri(tracking_uri)

client = mlflow.tracking.MlflowClient()
exp = client.get_experiment_by_name(experiment_name)
experiment_id = exp.experiment_id

# --- Step 1: get parent runs for this sweep ---
parent_runs = mlflow.search_runs(
    experiment_ids=[experiment_id],
    filter_string=f"tags.sweep_id = '{sweep_id}'",
    max_results=1000
)

parent_ids = parent_runs["run_id"].tolist()

print(parent_runs)

for parent_name in parent_runs["tags.mlflow.runName"].tolist():
    print("Parent run is {}".format(parent_name))
    # print("Parent run is {}".format(parent["run_name"]))

# --- Step 2: get child runs whose parentRunId is in parent_ids ---
all_runs = mlflow.search_runs(
    experiment_ids=[experiment_id],
    filter_string="",
    max_results=5000
)

# keep only runs whose parentRunId is in parent_ids
child_runs = all_runs[all_runs["tags.mlflow.parentRunId"].isin(parent_ids)]

first_level_ids = child_runs["run_id"].tolist()

print("First level children")
for idx, row in child_runs.iterrows():
    print("run_name: {}".format(row["tags.mlflow.runName"]))
    print("Setting sweep id to {}".format(sweep_id))
    client.set_tag(row["run_id"], "sweep_id", sweep_id)
# for run in child_runs["tags.mlflow.runName"].tolist():
    # print("run_name: {}".format(run))


# --- Step 3: Get child runs one level deeper (individual runs per seed)
second_level_children = all_runs[all_runs["tags.mlflow.parentRunId"].isin(first_level_ids)]

print("Second level children")
for idx, row in second_level_children.iterrows():
    print("run_name: {}".format(row["tags.mlflow.runName"]))
    print("Setting sweep id to {}".format(sweep_id))
    client.set_tag(row["run_id"], "sweep_id", sweep_id)
# for run in second_level_children["tags.mlflow.runName"].tolist():
    # print("run_name: {}".format(run))
