import argparse
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import optuna

def plot_3d_pareto_plotly(
    study,
    highlight_best_f1=True,
    log_scale_params=False,
    log_scale_maccs=False,
):
    """
    Interactive 3D Plotly visualization of Optuna multi-objective study.

    Assumes objective returns:
        [f1_score (maximize), num_params (minimize), maccs (minimize)]
    """

    # Get completed trials
    all_trials = [
        t for t in study.trials
        if t.values is not None and t.state.name == "COMPLETE"
    ]

    if len(all_trials) == 0:
        print("No completed trials found.")
        return
    else:
        print(f"Found {len(all_trials)} completed trials.")

    pareto_trials = study.best_trials

    if len(pareto_trials) == 0:
        print("No pareto front found.")
        return
    else:
        print(f"Found {len(pareto_trials)} pareto front trials.")

    # Convert to numpy arrays
    all_vals = np.array([t.values for t in all_trials])
    pareto_vals = np.array([t.values for t in pareto_trials])

    # Axis transforms
    def maybe_log(x, use_log):
        return np.log10(x) if use_log else x

    all_f1 = all_vals[:, 0]
    all_params = maybe_log(all_vals[:, 1], log_scale_params)
    all_maccs = maybe_log(all_vals[:, 2], log_scale_maccs)

    pareto_f1 = pareto_vals[:, 0]
    pareto_params = maybe_log(pareto_vals[:, 1], log_scale_params)
    pareto_maccs = maybe_log(pareto_vals[:, 2], log_scale_maccs)

    fig = go.Figure()

    # --- All trials ---
    fig.add_trace(
        go.Scatter3d(
            x=all_f1,
            y=all_params,
            z=all_maccs,
            mode="markers",
            marker=dict(
                size=4,
                opacity=0.3,
            ),
            name="All Trials",
            hovertemplate=(
                "F1: %{x}<br>"
                "Params: %{customdata[0]}<br>"
                "MACCs: %{customdata[1]}<extra></extra>"
            ),
            customdata=np.stack(
                [all_vals[:, 1], all_vals[:, 2]],
                axis=-1
            )
        )
    )

    # --- Pareto front ---
    fig.add_trace(
        go.Scatter3d(
            x=pareto_f1,
            y=pareto_params,
            z=pareto_maccs,
            mode="markers",
            marker=dict(
                size=8,
                symbol="diamond",
            ),
            name="Pareto Front",
            hovertemplate=(
                "F1: %{x}<br>"
                "Params: %{customdata[0]}<br>"
                "MACCs: %{customdata[1]}<extra></extra>"
            ),
            customdata=np.stack(
                [pareto_vals[:, 1], pareto_vals[:, 2]],
                axis=-1
            )
        )
    )

    # --- Highlight best F1 ---
    if highlight_best_f1:
        best_trial = max(all_trials, key=lambda t: t.values[0])
        bf1, bparams, bmaccs = best_trial.values

        fig.add_trace(
            go.Scatter3d(
                x=[bf1],
                y=[maybe_log(bparams, log_scale_params)],
                z=[maybe_log(bmaccs, log_scale_maccs)],
                mode="markers",
                marker=dict(
                    size=12,
                    symbol="cross",
                ),
                name="Best F1",
            )
        )

    fig.update_layout(
        title="Interactive 3D Pareto Front (NSGA-II)",
        scene=dict(
            xaxis_title="F1 Score (maximize)",
            yaxis_title="Num Params" + (" (log10)" if log_scale_params else ""),
            zaxis_title="MACCs" + (" (log10)" if log_scale_maccs else ""),
        ),
        margin=dict(l=0, r=0, b=0, t=50),
    )

    fig.show()

def plot_2d_pareto_plotly(
    study,
    highlight_best_f1=True,
    log_scale_params=False,
    log_scale_maccs=False,
):
    """
    Side-by-side 2D Pareto visualization for a multi-objective Optuna study.

    Assumes objective returns:
        [f1_score (maximize), num_params (minimize), maccs (minimize)]

    Produces one figure with:
        - F1 vs Params
        - F1 vs MACCs
    """

    # -------------------------------------------------
    # Internal Helpers
    # -------------------------------------------------

    def _get_trials():
        all_trials = [
            t for t in study.trials
            if t.values is not None and t.state.name == "COMPLETE"
        ]

        if not all_trials:
            print("No completed trials found.")
            return None, None

        pareto_trials = study.best_trials
        if not pareto_trials:
            print("No pareto front found.")
            return None, None

        print(f"Found {len(all_trials)} completed trials.")
        print(f"Found {len(pareto_trials)} pareto front trials.")

        return all_trials, pareto_trials

    def _maybe_log(x, use_log):
        return np.log10(x) if use_log else x

    def _extract_arrays(all_trials, pareto_trials):
        all_vals = np.array([t.values for t in all_trials])
        pareto_vals = np.array([t.values for t in pareto_trials])

        data = {
            "all_f1": all_vals[:, 0],
            "all_params": _maybe_log(all_vals[:, 1], log_scale_params),
            "all_maccs": _maybe_log(all_vals[:, 2], log_scale_maccs),
            "pareto_f1": pareto_vals[:, 0],
            "pareto_params": _maybe_log(pareto_vals[:, 1], log_scale_params),
            "pareto_maccs": _maybe_log(pareto_vals[:, 2], log_scale_maccs),
            "all_raw": all_vals,
            "pareto_raw": pareto_vals,
        }

        if highlight_best_f1:
            best_trial = max(all_trials, key=lambda t: t.values[0])
            bf1, bparams, bmaccs = best_trial.values
            data["best"] = (
                bf1,
                _maybe_log(bparams, log_scale_params),
                _maybe_log(bmaccs, log_scale_maccs),
            )

        return data

    def _add_scatter_layer(
        fig,
        x,
        y,
        raw_vals,
        name,
        row,
        col,
        size=6,
        opacity=1.0,
        symbol="circle",
    ):
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                marker=dict(size=size, opacity=opacity, symbol=symbol),
                name=name,
                hovertemplate=(
                    "F1: %{y}<br>"
                    "Params: %{customdata[0]}<br>"
                    "MACCs: %{customdata[1]}<extra></extra>"
                ),
                customdata=np.stack(
                    [raw_vals[:, 1], raw_vals[:, 2]],
                    axis=-1,
                ),
            ),
            row=row,
            col=col,
        )

    # -------------------------------------------------
    # Main Logic
    # -------------------------------------------------

    all_trials, pareto_trials = _get_trials()
    if all_trials is None:
        return

    data = _extract_arrays(all_trials, pareto_trials)

    # Create subplots
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "F1 vs Num Params",
            "F1 vs MACCs",
        ),
        horizontal_spacing=0.12,
    )

    # -------------------------------------------------
    # Left Plot: F1 vs Params
    # -------------------------------------------------

    _add_scatter_layer(
        fig,
        data["all_params"],
        data["all_f1"],
        data["all_raw"],
        "All Trials",
        row=1,
        col=1,
        opacity=0.3,
    )

    _add_scatter_layer(
        fig,
        data["pareto_params"],
        data["pareto_f1"],
        data["pareto_raw"],
        "Pareto Front",
        row=1,
        col=1,
        size=10,
        symbol="diamond",
    )

    if highlight_best_f1:
        bf1, bparams, _ = data["best"]
        fig.add_trace(
            go.Scatter(
                x=[bparams],
                y=[bf1],
                mode="markers",
                marker=dict(size=14, symbol="cross"),
                name="Best F1",
            ),
            row=1,
            col=1,
        )

    # -------------------------------------------------
    # Right Plot: F1 vs MACCs
    # -------------------------------------------------

    _add_scatter_layer(
        fig,
        data["all_maccs"],
        data["all_f1"],
        data["all_raw"],
        "All Trials",
        row=1,
        col=2,
        opacity=0.3,
    )

    _add_scatter_layer(
        fig,
        data["pareto_maccs"],
        data["pareto_f1"],
        data["pareto_raw"],
        "Pareto Front",
        row=1,
        col=2,
        size=10,
        symbol="diamond",
    )

    if highlight_best_f1:
        bf1, _, bmaccs = data["best"]
        fig.add_trace(
            go.Scatter(
                x=[bmaccs],
                y=[bf1],
                mode="markers",
                marker=dict(size=14, symbol="cross"),
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    # -------------------------------------------------
    # Layout
    # -------------------------------------------------

    fig.update_xaxes(
        title_text="Num Params" + (" (log10)" if log_scale_params else ""),
        row=1,
        col=1,
    )
    fig.update_xaxes(
        title_text="MACCs" + (" (log10)" if log_scale_maccs else ""),
        row=1,
        col=2,
    )
    fig.update_yaxes(
        title_text="F1 Score (maximize)",
        row=1,
        col=1,
    )

    fig.update_layout(
        title="2D Pareto Front (NSGA-II)",
        height=500,
        width=1100,
    )

    fig.show()


def print_pareto_hyperparameters(study):
    """
    Prints hyperparameters for all Pareto front trials in a clean table format,
    sorted by F1 score (highest first).

    Assumes multi-objective study with:
        [F1 (maximize), Params (minimize), MACCs (minimize)]
    """

    pareto_trials = study.best_trials

    if not pareto_trials:
        print("No Pareto front trials found.")
        return

    # -------------------------------------------------
    # Sort by F1 descending
    # -------------------------------------------------
    pareto_trials_sorted = sorted(
        pareto_trials,
        key=lambda t: t.values[0],  # F1
        reverse=True,
    )

    print(f"\nFound {len(pareto_trials_sorted)} Pareto front trials.")
    print("Sorted by F1 score (descending).\n")

    for i, trial in enumerate(pareto_trials_sorted, start=1):

        f1, num_params, maccs = trial.values

        print("=" * 70)
        print(f"Pareto Trial {i}")
        print(f"Trial Number: {trial.number}")
        print("-" * 70)
        print(f"F1 Score : {f1:.6f}")
        print(f"# Params : {num_params}")
        print(f"MACCs    : {maccs}")
        print("-" * 70)

        params = trial.params

        if not params:
            print("No hyperparameters recorded.")
            continue

        # Sort parameters alphabetically for easier comparison
        sorted_items = sorted(params.items(), key=lambda x: x[0])

        # Compute column width dynamically
        max_name_len = max(len(name) for name, _ in sorted_items)

        # Header
        print(f"{'Parameter'.ljust(max_name_len)} | Value")
        print(f"{'-' * max_name_len}-+{'-' * 20}")

        # Rows
        for name, value in sorted_items:
            print(f"{name.ljust(max_name_len)} | {value}")

        print("\n")


# =====================================================
# Main Entry Point
# =====================================================

def main():
    parser = argparse.ArgumentParser(
        description="Load Optuna study and analyze Pareto front."
    )

    parser.add_argument(
        "--study-name",
        type=str,
        required=True,
        help="Name of the Optuna study",
    )

    parser.add_argument(
        "--storage",
        type=str,
        required=True,
        help="Database storage URL (e.g., sqlite:///example.db)",
    )

    # Optional behaviors
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot 2D Pareto front.",
    )

    parser.add_argument(
        "--print-table",
        action="store_true",
        help="Print hyperparameter tables for Pareto trials.",
    )

    parser.add_argument("--log-params", action="store_true")
    parser.add_argument("--log-maccs", action="store_true")

    args = parser.parse_args()

    print("Loading study...")
    study = optuna.load_study(
        study_name=args.study_name,
        storage=args.storage,
    )
    print("Study loaded.")

    # If neither plot nor print specified → default to plotting
    if not args.plot and not args.print_table:
        args.plot = True

    # -------------------------------------------------
    # Execute Selected Actions
    # -------------------------------------------------

    if args.print_table:
        print_pareto_hyperparameters(study)

    if args.plot:
        plot_2d_pareto_plotly(
            study,
            log_scale_params=args.log_params,
            log_scale_maccs=args.log_maccs,
        )


if __name__ == "__main__":
    main()
