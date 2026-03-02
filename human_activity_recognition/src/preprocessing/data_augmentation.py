import pandas as pd
import tensorflow as tf
import numpy as np
from typing import Tuple, Optional

from dataclasses import dataclass

@dataclass
class NoiseConfig:
    noise_std: float

@dataclass
class AmplitudeScaleConfig:
    min_scale: float
    max_scale: float

@dataclass
class RotationConfig:
    max_roll_deg: float
    max_pitch_deg: float
    max_yaw_deg: float

class AugmentationConfig:
    def __init__(self,
                 seed: int,
                 noise_cfg: Optional[NoiseConfig] = None,
                 amplitude_scale_cfg: Optional[AmplitudeScaleConfig] = None,
                 rotation_cfg: Optional[RotationConfig] = None,
                ):
        self.epoch = tf.Variable(0,
                                 dtype=tf.int64,
                                 trainable=False)
        self.seed = seed

        self.noise_cfg = noise_cfg
        self.amplitude_scale_cfg = amplitude_scale_cfg
        self.rotation_cfg = rotation_cfg


def get_stateless_augmentation_seed(cfg: AugmentationConfig, batch_idx) -> int:
    base_seed = tf.constant( [cfg.seed, 0], dtype=tf.int32)

    # seed = fold(global_seed, epoch)
    seed1 = tf.random.experimental.stateless_fold_in(base_seed, cfg.epoch)

    # seed = fold(seed1, batch_idx)
    final_seed = tf.random.experimental.stateless_fold_in(seed1, batch_idx)

    return final_seed

def add_noise(batch: Tuple[tf.Tensor, tf.Tensor],
              batch_seed: int,
              noise_cfg: NoiseConfig):
    x, y = batch

    noise = tf.random.stateless_normal(
        shape=tf.shape(x),
        seed=batch_seed,
        stddev=noise_cfg.noise_std,
        dtype=x.dtype,
    )
    return x + noise, y

def apply_amplitude_scaling(batch: Tuple[tf.Tensor, tf.Tensor],
                            batch_seed: int,
                            scale_cfg: AmplitudeScaleConfig
                           ) -> Tuple[tf.Tensor, tf.Tensor]:

    x, y = batch
    batch_size = tf.shape(x)[0]

    # ---- Sample one scale per sample ----
    scales = tf.random.stateless_uniform(
        shape=(batch_size,),
        seed=batch_seed,
        minval=scale_cfg.min_scale,
        maxval=scale_cfg.max_scale,
        dtype=x.dtype
    )

    # Reshape for broadcasting across (48, 3, 1)
    scales = tf.reshape(scales, (batch_size, 1, 1, 1))

    # ---- Apply scaling ----
    scaled = x * scales  # broadcast multiply

    return scaled, y

def apply_full_rotation(batch: Tuple[tf.Tensor, tf.Tensor],
                        batch_seed: int,
                        rotation_cfg: RotationConfig
                       ) -> Tuple[tf.Tensor, tf.Tensor]:

    x, y = batch
    batch_size = tf.shape(x)[0]

    # Convert degrees → radians
    max_roll  = rotation_cfg.max_roll_deg  * np.pi / 180.0
    max_pitch = rotation_cfg.max_pitch_deg * np.pi / 180.0
    max_yaw   = rotation_cfg.max_yaw_deg   * np.pi / 180.0

    # ---- Sample each axis independently ----
    roll = tf.random.stateless_uniform(
        shape=(batch_size,),
        seed=batch_seed,
        minval=-max_roll,
        maxval=max_roll,
        dtype=x.dtype
    )

    pitch = tf.random.stateless_uniform(
        shape=(batch_size,),
        seed=batch_seed + 1,
        minval=-max_pitch,
        maxval=max_pitch,
        dtype=x.dtype
    )

    yaw = tf.random.stateless_uniform(
        shape=(batch_size,),
        seed=batch_seed + 2,
        minval=-max_yaw,
        maxval=max_yaw,
        dtype=x.dtype
    )

    # ---- Trig ----
    cr, sr = tf.cos(roll), tf.sin(roll)
    cp, sp = tf.cos(pitch), tf.sin(pitch)
    cy, sy = tf.cos(yaw), tf.sin(yaw)

    # ---- Build rotation matrix R = Rz · Ry · Rx ----
    r00 = cy * cp
    r01 = cy * sp * sr - sy * cr
    r02 = cy * sp * cr + sy * sr

    r10 = sy * cp
    r11 = sy * sp * sr + cy * cr
    r12 = sy * sp * cr - cy * sr

    r20 = -sp
    r21 = cp * sr
    r22 = cp * cr

    R = tf.stack([
        tf.stack([r00, r01, r02], axis=1),
        tf.stack([r10, r11, r12], axis=1),
        tf.stack([r20, r21, r22], axis=1),
    ], axis=1)  # (B,3,3)

    # ---- Apply rotation ----
    x_squeezed = tf.squeeze(x, axis=-1)  # (B,48,3)
    rotated = tf.matmul(x_squeezed, R)   # row-vector convention
    rotated = tf.expand_dims(rotated, axis=-1)

    return rotated, y

def generate_apply_augmentation(cfg: AugmentationConfig):
    def apply_augmentation(batch_idx: int,
                           batch: Tuple[tf.Tensor, tf.Tensor]
                          ) -> Tuple[tf.Tensor, tf.Tensor]:

        batch_seed = get_stateless_augmentation_seed(cfg, batch_idx)
        x,y = batch

        if cfg.noise_cfg is not None:
            noise_seed = tf.random.experimental.stateless_fold_in(
                batch_seed, 1
            )

            x, y = add_noise((x, y),
                             noise_seed,
                             cfg.noise_cfg)

        if cfg.amplitude_scale_cfg is not None:
            amplitude_seed = tf.random.experimental.stateless_fold_in(
                batch_seed, 2
            )

            x, y = apply_amplitude_scaling((x, y),
                                           amplitude_seed,
                                           cfg.amplitude_scale_cfg)

        if cfg.rotation_cfg is not None:
            rotation_seed = tf.random.experimental.stateless_fold_in(
                batch_seed, 3
            )

            x, y = apply_full_rotation((x, y),
                                       rotation_seed,
                                       cfg.rotation_cfg)


        return x, y

    return apply_augmentation

if __name__ == "__main__":

    import plotly.graph_objects as go

    def animate_original_vs_rotated(original: tf.Tensor,
                                    rotated: tf.Tensor,
                                    title="Original vs Rotated"):

        # Convert to numpy
        if isinstance(original, tf.Tensor):
            original = original.numpy()
        if isinstance(rotated, tf.Tensor):
            rotated = rotated.numpy()

        original = np.squeeze(original, axis=-1)  # (T,3)
        rotated = np.squeeze(rotated, axis=-1)    # (T,3)
        timesteps = original.shape[0]

        frames = []
        slider_steps = []

        for t in range(timesteps):
            trace_orig = go.Scatter3d(
                x=[0, original[t, 0]],
                y=[0, original[t, 1]],
                z=[0, original[t, 2]],
                mode='lines+markers',
                line=dict(width=6, color='blue'),
                marker=dict(size=4, color='blue'),
                name='Original'
            )

            trace_rot = go.Scatter3d(
                x=[0, rotated[t, 0]],
                y=[0, rotated[t, 1]],
                z=[0, rotated[t, 2]],
                mode='lines+markers',
                line=dict(width=6, color='red'),
                marker=dict(size=4, color='red'),
                name='Rotated'
            )

            frames.append(go.Frame(data=[trace_orig, trace_rot], name=str(t)))

            slider_steps.append(
                dict(
                    method="animate",
                    args=[[str(t)],
                          {"mode": "immediate",
                           "frame": {"duration": 0, "redraw": True},
                           "transition": {"duration": 0}}],
                    label=str(t)
                )
            )

        # Initial frame
        trace_orig_init = go.Scatter3d(
            x=[0, original[0, 0]],
            y=[0, original[0, 1]],
            z=[0, original[0, 2]],
            mode='lines+markers',
            line=dict(width=6, color='blue'),
            marker=dict(size=4, color='blue'),
            name='Original'
        )

        trace_rot_init = go.Scatter3d(
            x=[0, rotated[0, 0]],
            y=[0, rotated[0, 1]],
            z=[0, rotated[0, 2]],
            mode='lines+markers',
            line=dict(width=6, color='red'),
            marker=dict(size=4, color='red'),
            name='Rotated'
        )

        fig = go.Figure(
            data=[trace_orig_init, trace_rot_init],
            frames=frames
        )

        fig.update_layout(
            title=title,
            scene=dict(
                xaxis=dict(range=[-6, 6]),
                yaxis=dict(range=[-6, 6]),
                zaxis=dict(range=[-6, 6]),
                aspectmode='cube'
            ),
            updatemenus=[dict(
                type="buttons",
                direction="left",
                x=0.1,
                y=0,
                showactive=False,
                buttons=[
                    dict(label="Play",
                         method="animate",
                         args=[None,
                               {"frame": {"duration": 200, "redraw": True},
                                "fromcurrent": True}]),
                    dict(label="Pause",
                         method="animate",
                         args=[[None],
                               {"frame": {"duration": 0, "redraw": False},
                                "mode": "immediate"}])
                ]
            )],
            sliders=[dict(
                active=0,
                currentvalue={"prefix": "Timestep: "},
                pad={"t": 50},
                steps=slider_steps
            )]
        )

        fig.show()    # ---- Create dummy dataset ----

    B = 4  # batch size
    window_len = 48
    x_dummy = np.random.rand(B, window_len, 3, 1).astype(np.float32) * 4
    # const_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # (3,)
    # const_vec = const_vec.reshape(1, 1, 3, 1)                # (1,1,3,1)

    # x_dummy = np.tile(const_vec, (B, window_len, 1, 1))      # (B,48,3,1)

    y_dummy = np.array([0,1,2,3], dtype=np.int64)

    ds = tf.data.Dataset.from_tensor_slices((x_dummy, y_dummy))
    ds = ds.batch(B).enumerate()

    # ---- Apply yaw rotation and visualize ----
    noise_cfg = NoiseConfig(noise_std=0.125)
    scaling_cfg = AmplitudeScaleConfig(min_scale=0.8, max_scale=1.2)
    rotation_cfg = RotationConfig(max_roll_deg=5.0,
                                  max_pitch_deg=5.0,
                                  max_yaw_deg=15.0)

    cfg = AugmentationConfig(seed=42,
                             noise_cfg=noise_cfg,
                             amplitude_scale_cfg=scaling_cfg,
                             rotation_cfg=rotation_cfg)

    aug_fn = generate_apply_augmentation(cfg)

    for batch_idx, batch in ds.take(1):
        x_batch, y_batch = batch

        x_aug, _ = aug_fn(batch_idx, batch)
        # batch_seed = get_stateless_noise_seed(cfg,
                                              # batch_idx,
                                              # 3)

        # x_aug, _ = apply_full_rotation(batch,
                                       # batch_seed,
                                       # max_roll_deg=5.0,
                                       # max_pitch_deg=5.0,
                                       # max_yaw_deg=15.0)

        # x_aug, _ = apply_amplitude_scaling(batch,
                                              # batch_seed,
                                              # min_scale=0.8,
                                              # max_scale=1.2)

        # x_aug, _ = add_noise(batch,
                             # batch_seed,
                             # noise_std=0.125)

        for sample_idx in range(B):
            animate_original_vs_rotated(x_batch[sample_idx], x_aug[sample_idx],
                                        title=f"Sample {sample_idx} Original vs Rotated")

