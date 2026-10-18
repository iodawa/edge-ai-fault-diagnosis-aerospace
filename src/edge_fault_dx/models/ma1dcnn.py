"""MA1DCNN architecture -- src/edge_fault_dx/models/ma1dcnn.py.

Moved here from training/train.py, where it originally lived inline per this
project's "every script runs standalone, nothing imported from elsewhere in
the codebase" discipline (so each script could be self-check-verified in
isolation). That discipline served the proving-out phase; now that trial13 is
confirmed (0.8401 macro-F1, four independent runs -- see train.py's own
module docstring), the architecture belongs in its own importable module,
matching this repo's scaffold (src/edge_fault_dx/models/, separate from
training/, compression/, evaluation/, deployment/).

Nothing about the architecture itself changed in this move -- every class and
function below is byte-for-byte identical to what train.py defined inline.
train.py now imports build_ma1dcnn (and MA1DCNN) from here instead of
defining them, and re-exports both so `from train import build_ma1dcnn` --
which threshold_sweep.py and false_alarm_review.py both rely on -- keeps
working unchanged.

Per this repo's own scaffold comment (see models/ma1dcnn.py's prior TODO
stub): architecture per Wang et al. (2020) -- full bibliographic details
(journal/venue, volume, pages) still unverified; see the praxis write-up's
References section for that open item.
"""
from __future__ import annotations

import torch
import torch.nn as nn

CAM_REDUCTION_RATIO = 4  # repeated from ma1dcnn.ipynb -- needed to construct MA1DCNN below


# =============================================================================
# Model -- repeated verbatim from ma1dcnn.ipynb / training/train.py
# =============================================================================

class Conv1dSamePadding(nn.Module):
    """1D convolution that keeps the sequence length unchanged (stride=1, odd kernel)."""

    def __init__(self, in_channels, out_channels, kernel_size, conv_dropout=0.0, use_batchnorm=False):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f"Conv1dSamePadding needs an odd kernel_size, got {kernel_size}")
        padding = (kernel_size - 1) // 2  # same-padding formula: P = (K - 1) / 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=padding)
        # Identity (zero params/buffers) when disabled (the default); placed BEFORE
        # activation -- normalizes the raw conv output, not an already-ReLU-clipped one.
        self.bn = nn.BatchNorm1d(out_channels) if use_batchnorm else nn.Identity()
        self.activation = nn.ReLU()
        self.conv_dropout = nn.Dropout(p=conv_dropout)  # p=0.0 default is a no-op, existing configs unaffected

    def forward(self, x):
        return self.conv_dropout(self.activation(self.bn(self.conv(x))))  # self.bn is a no-op when use_batchnorm=False


class ChannelAttentionModule(nn.Module):
    """CAM: one learned attention weight per CHANNEL, shared across all timesteps."""

    def __init__(self, n_channels, reduction_ratio=CAM_REDUCTION_RATIO):
        super().__init__()
        hidden = max(1, n_channels // reduction_ratio)
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excite = nn.Sequential(
            nn.Linear(n_channels, hidden), nn.ReLU(),
            nn.Linear(hidden, n_channels), nn.Sigmoid(),
        )

    def forward(self, x):
        batch, channels, length = x.shape
        squeezed = self.squeeze(x).view(batch, channels)
        channel_weights = self.excite(squeezed).view(batch, channels, 1)
        return x * channel_weights


class ElementwiseAttentionModule(nn.Module):
    """EAM: one learned attention weight per TIMESTEP, shared across all channels."""

    def __init__(self, n_channels):
        super().__init__()
        self.score = nn.Conv1d(in_channels=n_channels, out_channels=1, kernel_size=1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        attention = self.activation(self.score(x))
        return x * attention


class MA1DCNN(nn.Module):
    """3 conv blocks -> CAM -> EAM -> global average pooling -> FC head -> 5 logits."""

    def __init__(self, n_input_channels, channel_1, channel_2, channel_3,
                 kernel_size, fc_units, dropout, n_outputs,
                 cam_reduction_ratio=CAM_REDUCTION_RATIO, conv_dropout=0.0, use_batchnorm=False):
        super().__init__()
        # conv_dropout/use_batchnorm default to off -- BEFORE_TUNING_CONFIG/AFTER_TUNING_CONFIG/
        # BEFORE_TUNING_CONFIG_CONV_DROPOUT build unaffected models. The two are independent,
        # composable levers -- use_batchnorm=True layers ON TOP of conv_dropout, never replaces it.
        self.conv1 = Conv1dSamePadding(n_input_channels, channel_1, kernel_size, conv_dropout, use_batchnorm)
        self.conv2 = Conv1dSamePadding(channel_1, channel_2, kernel_size, conv_dropout, use_batchnorm)
        self.conv3 = Conv1dSamePadding(channel_2, channel_3, kernel_size, conv_dropout, use_batchnorm)
        self.cam = ChannelAttentionModule(channel_3, reduction_ratio=cam_reduction_ratio)
        self.eam = ElementwiseAttentionModule(channel_3)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(channel_3, fc_units)
        self.fc_activation = nn.ReLU()
        self.output_layer = nn.Linear(fc_units, n_outputs)

    def forward(self, x):
        # x: (batch, n_input_channels, L) -- a batch of windowed, normalized sensor sequences.
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.cam(x)
        x = self.eam(x)
        x = self.global_avg_pool(x)
        x = x.view(x.shape[0], -1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.fc_activation(x)
        return self.output_layer(x)  # RAW logits -- BCEWithLogitsLoss expects these, not sigmoid(logits)

    def predict_proba(self, x):
        """sigmoid(forward(x)) -- for inference/thresholding, never inside a training loop's backward pass."""
        with torch.no_grad():
            return torch.sigmoid(self.forward(x))


def build_ma1dcnn(config):
    """Construct an MA1DCNN from a plain hyperparameter dict (e.g. BEFORE_TUNING_CONFIG)."""
    return MA1DCNN(
        n_input_channels=config["n_input_channels"], channel_1=config["channel_1"],
        channel_2=config["channel_2"], channel_3=config["channel_3"],
        kernel_size=config["kernel_size"], fc_units=config["fc_units"],
        dropout=config["dropout"], n_outputs=config["n_outputs"],
        conv_dropout=config.get("conv_dropout", 0.0),      # .get() -- NOT config[...] -- for backward compatibility
        use_batchnorm=config.get("use_batchnorm", False),
    )
