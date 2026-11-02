"""StandardCNN -- the architecture-only comparator H1 needs --
src/edge_fault_dx/models/standard_cnn.py.

Built for H1 (Phase 5): MA1DCNN with its ChannelAttentionModule and
ElementwiseAttentionModule removed and nothing else changed -- same conv
stack, same channel sizes, same kernel size, same FC head, same training
config (TRIAL_13_MODEL_CONFIG / TRIAL_13_TRAIN_CONFIG, unchanged). This is
the actual design constraint H1's "architecture only" independent variable
imposes (h1_completion_plan.docx Section 3): any other difference -- a wider
conv stack, a different optimizer, more training epochs -- would confound
the comparison H1 asks for.

Extracted from h1_standard_cnn_and_noise_robustness.ipynb Section 1 (real
results in h1_notebook_code_walkthrough.docx Sections 4-4.2 and
praxis_writeup.docx Section 4.5.1), following this repo's own
"notebook-first-then-extract" convention -- the same path models/ma1dcnn.py
and training/train.py already took.

Conv1dSamePadding is imported from .ma1dcnn unchanged, not redefined -- it
was already architecture-agnostic (it only knows in_channels/out_channels/
kernel_size, nothing MA1DCNN-specific), so there is no reason to duplicate
it here.

Real, confirmed results (praxis_writeup.docx Section 4.5.1):

    layer group        | MA1DCNN params | StandardCNN params | difference
    conv1 + conv2 + conv3 | 55,904       | 55,904              | identical
    cam + eam              | 8,481       | 0 (removed)         | the entire diff
    fc + output_layer      | 2,149       | 2,149               | identical
    total                  | 66,534      | 58,053              | 12.7% smaller

A real build of StandardCNN(**TRIAL_13_MODEL_CONFIG) came out at exactly
58,053 parameters -- matching this hand count to the digit, run as a
self-check before any training. Trained once with the same hyperparameters,
early-stopping rule, and seed as MA1DCNN's own trial13 run, StandardCNN
reached a best validation macro-F1 of 0.831868 (0.8319), setting its own
retention bar at 0.781868 (baseline minus 5 points) -- StandardCNN's own
confirmed baseline, not MA1DCNN's 0.7940 bar.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .ma1dcnn import Conv1dSamePadding


class StandardCNN(nn.Module):
    """3 conv blocks -> global average pooling -> FC head -> 5 logits.

    Identical to MA1DCNN (models/ma1dcnn.py) except it has no
    ChannelAttentionModule and no ElementwiseAttentionModule -- conv3's
    output goes straight into global_avg_pool. This is the whole point: H1
    isolates "architecture only" as its independent variable, so every
    other choice (channel sizes, kernel_size, fc_units, dropout, training
    config) is copied from TRIAL_13_MODEL_CONFIG / TRIAL_13_TRAIN_CONFIG
    unchanged.
    """

    def __init__(self, n_input_channels, channel_1, channel_2, channel_3,
                 kernel_size, fc_units, dropout, n_outputs,
                 conv_dropout=0.0, use_batchnorm=False):
        super().__init__()
        self.conv1 = Conv1dSamePadding(n_input_channels, channel_1, kernel_size, conv_dropout, use_batchnorm)
        self.conv2 = Conv1dSamePadding(channel_1, channel_2, kernel_size, conv_dropout, use_batchnorm)
        self.conv3 = Conv1dSamePadding(channel_2, channel_3, kernel_size, conv_dropout, use_batchnorm)
        # no self.cam, no self.eam -- this is the entire architectural difference
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
        # (MA1DCNN inserts self.cam(x) then self.eam(x) here -- the only diff)
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


def build_standard_cnn(config):
    """Construct a StandardCNN from a plain hyperparameter dict.

    Same config dict shape as build_ma1dcnn() -- TRIAL_13_MODEL_CONFIG works
    unchanged as input to both, which is exactly what keeps the comparison
    fair (h1_completion_plan.docx Section 3).
    """
    return StandardCNN(
        n_input_channels=config["n_input_channels"], channel_1=config["channel_1"],
        channel_2=config["channel_2"], channel_3=config["channel_3"],
        kernel_size=config["kernel_size"], fc_units=config["fc_units"],
        dropout=config["dropout"], n_outputs=config["n_outputs"],
        conv_dropout=config.get("conv_dropout", 0.0),
        use_batchnorm=config.get("use_batchnorm", False),
    )
