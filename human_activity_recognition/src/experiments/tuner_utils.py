import numpy as np
import plotly.graph_objects as go


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

    pareto_trials = study.best_trials

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
                    symbol="star",
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
