import argparse
import pickle
import plotly.express as px
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.signal import stft

st.set_page_config(layout="wide")

# Central lookup: name (lowercase) → ID
_ACTIVITY_NAME_TO_GLOBAL_ACTIVITY_ID = {
    'stationary': 0,
    'walking': 1,
    'running': 2,
    'cycling': 3,
}

# Reverse: ID → name
_GLOBAL_ACTIVITY_ID_TO_NAME = {v: k for k, v in _ACTIVITY_NAME_TO_GLOBAL_ACTIVITY_ID.items()}


def global_activity_id_to_name(id: int) -> str:
    """
    Convert an activity ID to its name.
    Raises KeyError if the ID is not found.
    """
    if id not in _GLOBAL_ACTIVITY_ID_TO_NAME:
        raise KeyError(f"Activity ID not found: {id}")
    return _GLOBAL_ACTIVITY_ID_TO_NAME[id]


def load_data(path):
    with open(path, "rb") as f:
        data = pickle.load(f)

    return data["df"]

def plot_umap_by_true_label(df, highlight_ids=None, title=None):
    opactity = 0.7

    if title is None:
        title = "UMAP colored by TRUE label"

    if highlight_ids:
        opactity = 0.1

    fig = px.scatter(
        df,
        x="umap_x",
        y="umap_y",
        color="true_label_name",
        title=title,
        opacity=opactity,
        color_discrete_sequence=px.colors.qualitative.Set1,  # distinct colors
        hover_data={
            "sample_id": True,
            "true_label_name": True,
            "pred_label_name": True,
            "confidence": True,
            "loss": True
        },
        custom_data=["sample_id"],
    )

    fig.update_traces(marker=dict(size=5))

    # ----------------------------
    # Highlight selected samples
    # ----------------------------
    if highlight_ids and len(highlight_ids) > 0:
        df_highlight = df[df["sample_id"].isin(highlight_ids)]
        df_highlight["selected"] = True

        fig.add_scatter(
            x=df_highlight["umap_x"],
            y=df_highlight["umap_y"],
            mode="markers",
            marker=dict(
                size=20,
                color="black",   # highlight color
                symbol="circle-open",
                line=dict(width=2)
            ),
            name="Selected",
            showlegend=True,
            customdata=df_highlight[[
                "sample_id",
                "true_label_name",
                "pred_label_name",
                "confidence",
                "loss",
                "umap_x",
                "umap_y",
                "selected",
            ]].values,
            hovertemplate=(
                "sample_id: %{customdata[0]}<br>"
                "true_label: %{customdata[1]}<br>"
                "pred_label: %{customdata[2]}<br>"
                "confidence: %{customdata[3]:.3f}<br>"
                "loss: %{customdata[4]:.3f}<br>"
                "umap: %{customdata[5]:.3f}, %{customdata[6]:.3f}<br>"  # umap_x, umap_y
                "selected: %{customdata[7]}<br>"
                "<extra></extra>"
            ),
        )

    fig.update_layout(
        legend=dict(
            x=1.02,
            y=1,
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        autosize=True
    )

    event = st.plotly_chart(fig, use_container_width=True, on_select="rerun", key=title)

    # Extract sample_id only
    if event:
        sample_ids = [pt["customdata"][0] for pt in event["selection"]["points"]]
    else:
        sample_ids = []

    return sample_ids

import plotly.graph_objects as go

def compute_global_ranges(selected_samples, X, fs=50, pad_factor=4, nperseg=16):
    time_min, time_max = np.inf, -np.inf
    fft_min, fft_max = np.inf, -np.inf
    stft_min, stft_max = np.inf, -np.inf

    for sample_id in selected_samples:
        sample = X[int(sample_id)].squeeze()

        # ---- TIME ----
        time_min = min(time_min, sample.min())
        time_max = max(time_max, sample.max())

        # ---- FFT ----
        N = sample.shape[0]
        N_pad = N * pad_factor
        fft_vals = np.abs(np.fft.rfft(sample, n=N_pad, axis=0))
        fft_vals = 20 * np.log10(fft_vals + 1e-8)

        fft_min = min(fft_min, fft_vals.min())
        fft_max = max(fft_max, np.percentile(fft_vals, 100))

        # ---- STFT ----
        magnitude = np.linalg.norm(sample, axis=1)
        _, _, Zxx = stft(magnitude, fs=fs, nperseg=nperseg, nfft=nperseg*pad_factor)
        Z = np.abs(Zxx)
        Z_log = 20 * np.log10(Z + 1e-8)

        stft_min = min(stft_min, Z_log.min())
        stft_max = max(stft_max, np.percentile(Z_log, 100))

    return {
        "time": (time_min, time_max * 1.1),
        "fft": (fft_min, fft_max * 1.1),
        "stft": (stft_min, stft_max * 1.5)
    }

def plot_sample_time_fft_stft(sample, ranges, fs=50, nperseg=16, pad_factor=4):
    """
    Plot time domain, FFT, and STFT for a single sample (shape: [time, 3])
    """
    time = np.arange(sample.shape[0]) / fs

    # ---------------- Time Domain ----------------
    fig_time = go.Figure()
    for i, axis_name in enumerate(["x","y","z"]):
        fig_time.add_trace(go.Scatter(y=sample[:,i], x=time, mode='lines', name=axis_name))
    fig_time.update_layout(
        title=f"Time Domain",
        xaxis_title="Time",
        yaxis_title="Acceleration"
    )
    # fig_time.update_yaxes(range=ranges["time"], type="log")

    # ---------------- FFT ----------------
    N = sample.shape[0]
    N_pad = N * pad_factor
    freqs = np.fft.rfftfreq(N_pad, d=1/fs)
    fft_vals = np.abs(np.fft.rfft(sample, n=N_pad, axis=0))
    fft_vals = 20 * np.log10(fft_vals + 1e-8)

    fig_fft = go.Figure()
    for i, axis_name in enumerate(["x","y","z"]):
        fig_fft.add_trace(go.Scatter(y=fft_vals[:,i], x=freqs, mode='lines', name=axis_name))
    fig_fft.update_layout(
        title=f"FFT",
        xaxis_title="Frequency",
        yaxis_title="Amplitude (dB)"
    )
    fig_fft.update_yaxes(range=ranges["fft"])

    # ---------------- STFT ----------------
    magnitude = np.linalg.norm(sample, axis=1)
    f, t, Zxx = stft(magnitude, fs=fs, nperseg=nperseg, nfft=nperseg*pad_factor)

    Z = np.abs(Zxx)
    Z_log = 20 * np.log10(Z + 1e-8)

    fig_stft = go.Figure(data=go.Heatmap(
        z=Z_log,
        x=t,
        y=f,
        colorscale='Viridis',
        zmin=ranges["stft"][0],
        zmax=ranges["stft"][1],
    ))
    fig_stft.update_layout(
        title=f"STFT (dB Scale)",
        xaxis_title="Time",
        yaxis_title="Frequency"
    )

    return fig_time, fig_fft, fig_stft

# ----------------------------
# Load data (with caching)
# ----------------------------
@st.cache_data
def load_dataframe_cached(path):
    with path as f:
        data = pickle.load(f)

    df = data["df"] if "df" in data else data

    df["correct"] = df["true_label"] == df["pred_label"]

    df["true_label_name"] = df["true_label"].apply(global_activity_id_to_name)
    df["pred_label_name"] = df["pred_label"].apply(global_activity_id_to_name)

    return df

@st.cache_data
def load_dataset_cached(dataset_path):
    data = np.load(dataset_path)

    X = data["X"]
    Y = data["Y"]

    X = X.astype(np.float32)
    Y = Y.astype(np.float32)

    print(f"Dataset size: {len(X)} samples")

    return X, Y




###
# App Main
###

st.title("UMAP Dataset Explorer")

if st.button("Clear selected samples"):
    st.session_state.plot_selected_samples = []

# ----------------------------
# Sidebar
# ----------------------------
dataframe_file = st.sidebar.file_uploader(
    "Choose a dataframe file (.pkl)",
    type=["pkl"]
)

if dataframe_file is None:
    st.warning("Please upload a dataframe file to continue.")
    st.stop()  # Stop the app until a file is uploaded

dataset_file = st.sidebar.file_uploader(
    "Choose a dataset file (.npz)",
    type=["npz"]
)

if dataset_file is None:
    st.warning("Please upload a dataset file to continue.")
    st.stop()  # Stop the app until a file is uploaded

df = load_dataframe_cached(dataframe_file)

dataset = load_dataset_cached(dataset_file)
X = dataset[0]
Y = dataset[1]

# # ----------------------------
# # Info (replaces print)
# # ----------------------------
# st.write(f"**Samples:** {len(df)}")

st.subheader("Top loss samples")
top_loss_df = df.sort_values("loss", ascending=False).head(20)
top_loss_df = top_loss_df[["sample_id", "true_label_name", "pred_label_name", "confidence", "loss", "umap_x", "umap_y", "preds"]]
event = st.dataframe(top_loss_df, on_select="rerun")
df_samples = []

if event:
    for row in event.selection["rows"]:
        sample_id = top_loss_df.iloc[row]["sample_id"]
        df_samples.append(sample_id)

st.write(f"DF Samples {df_samples}")

# ----------------------------
# Plot
# ----------------------------
if "plot_selected_samples" not in st.session_state:
    st.session_state.plot_selected_samples = []

st.subheader("Selection Plot")
title = "UMAP colored by TRUE label"
sample_ids = plot_umap_by_true_label(df, title=title)

if sample_ids:
    sample_id = sample_ids[0]  # Only handle first click
    st.write(f"Clicked sample_id: {sample_id}")
    st.session_state.plot_selected_samples.append(sample_id)
else:
    sample_id = 0
    st.write("No sample selected")

st.subheader("Highlight Plot")
highlight_ids = st.session_state.plot_selected_samples + df_samples
title = "UMAP colored by TRUE label (highlighting selected samples)"
plot_umap_by_true_label(df, highlight_ids=highlight_ids, title=title)


def display_sample_plot(sample_id, sample_ranges):
    row = df[df.sample_id == sample_id].iloc[0]
    sample = X[int(sample_id)].squeeze()

    st.markdown(f"### Sample {sample_id} | {row.true_label_name} \n #### pred={row.pred_label_name} | conf={row.confidence:.2f} | loss={row.loss:.3f}")

    fig_time, fig_fft, fig_stft = plot_sample_time_fft_stft(
        sample, ranges=sample_ranges
    )

    st.plotly_chart(fig_time, use_container_width=True, key=f"sample_time_{sample_id}")
    st.plotly_chart(fig_fft, use_container_width=True, key=f"sample_fft_{sample_id}")
    st.plotly_chart(fig_stft, use_container_width=True, key=f"sample_stft_{sample_id}")


# ----------------- Selected samples -----------------
st.subheader("Selected Samples")
left_col, middle_col, right_col = st.columns(3)

all_selected_samples = df_samples + st.session_state.plot_selected_samples

sample_ranges = compute_global_ranges(all_selected_samples, X)

with left_col:
    if len(all_selected_samples) > 0:
        display_sample_plot(all_selected_samples[0], sample_ranges)

with middle_col:
    if len(all_selected_samples) > 1:
        display_sample_plot(all_selected_samples[1], sample_ranges)

with right_col:
    if len(all_selected_samples) > 2:
        for sample_id in all_selected_samples[2:]:
            display_sample_plot(sample_id, sample_ranges)
