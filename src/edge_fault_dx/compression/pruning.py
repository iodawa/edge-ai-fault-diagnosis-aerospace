"""Structured channel pruning (Phase 4, Step 2) and fine-tune recovery
(Phase 4, Step 3).

Extracted from 04_compression_pruning_quant.ipynb and documented function-
by-function in pruning_code_explained.docx (Sections 3-9). This replaces
the one-line TODO stub previously at this path.

Reused unchanged from elsewhere in this project (imported, not redefined
here): MA1DCNN / build_ma1dcnn (models/ma1dcnn.py); train_one_epoch,
evaluate (training/train.py).

Real, measured results at the three pruning ratios this project used
(pruning_code_explained.docx Sections 6 and 9.5), against the real
trial13 checkpoint (val_f1_macro = 0.8440, retention bar = 0.7940):

    ratio | channel_1/2/3  | params (cut)    | recovered val_f1_macro
    20%   | 26 / 51 / 102  | 43,522 (-34.6%) | 0.8436  PASSED (+5.0 pts)
    40%   | 19 / 38 / 77   | 25,467 (-61.7%) | 0.8445  PASSED (+5.1 pts)
    60%   | 13 / 26 / 51   | 12,486 (-81.2%) | 0.8254  PASSED (+3.1 pts)

NOTE ON PROVENANCE: the lines inside _topk_keep_indices,
compute_pruned_channel_counts, the three weight-copy blocks in
prune_ma1dcnn, and the core loop of recover_pruned_model are copied
verbatim from pruning_code_explained.docx. Full function signatures,
imports, and bias-copy lines (the walkthrough showed weight-copying only,
to illustrate the index-slicing trick -- bias copying follows the same
one-dimensional slice and is required for the "lossless at ratio=0.0"
self-check in Section 7.1 to actually pass) were completed to make this a
runnable module. Check this against the notebook's literal source before
treating it as byte-identical.
"""

from __future__ import annotations

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from ..models.ma1dcnn import build_ma1dcnn
from ..training.train import evaluate, train_one_epoch

# ---------------------------------------------------------------------------
# Step 3 recovery budget -- judgment calls named explicitly, not defaulted
# (pruning_code_explained.docx Section 9.2).
# ---------------------------------------------------------------------------
RECOVERY_N_EPOCHS = 20   # below run_training()'s own 30-epoch cap -- recovery
                         # starts from trial13's learned weights, not scratch
RECOVERY_PATIENCE = 5    # same patience run_training() itself uses


# ---------------------------------------------------------------------------
# Section 3 -- channel importance
# ---------------------------------------------------------------------------
def _l1_channel_importance(conv_weight: torch.Tensor) -> torch.Tensor:
    """L1-norm importance of each output channel of a conv weight tensor,
    shape (out_channels, in_channels, kernel_size). A channel whose filter
    weights are all close to zero contributes almost nothing to that
    layer's output, so it is the safest one to remove.
    """
    return conv_weight.detach().abs().sum(dim=(1, 2))   # sums dims 1,2; leaves dim 0


def _topk_keep_indices(importance: torch.Tensor, n_keep: int) -> torch.Tensor:
    """The indices of the n_keep highest-importance channels, re-sorted
    ascending by index so every downstream slice stays in the original
    channel order.
    """
    n_keep = max(1, min(n_keep, importance.shape[0]))
    topk = torch.topk(importance, k=n_keep).indices   # the n_keep LARGEST importance scores
    return torch.sort(topk).values                     # re-sorted ascending by index


# ---------------------------------------------------------------------------
# Section 4 -- how many channels survive
# ---------------------------------------------------------------------------
def compute_pruned_channel_counts(channel_1, channel_2, channel_3, prune_ratio):
    """The same keep_frac = 1 - prune_ratio is applied to all three conv
    blocks independently -- "prune 20%" means each of channel_1, channel_2,
    channel_3 independently keeps round(old * 0.8) channels, NOT that 20%
    of the total parameter budget is removed (that is a larger number --
    see Section 6's parameter-count math; conv2/conv3/CAM scale roughly
    quadratically in keep_frac, not linearly). max(1, ...) guarantees a
    layer is never pruned to 0 channels.
    """
    keep_frac = 1.0 - prune_ratio
    new_c1 = max(1, round(channel_1 * keep_frac))
    new_c2 = max(1, round(channel_2 * keep_frac))
    new_c3 = max(1, round(channel_3 * keep_frac))
    return new_c1, new_c2, new_c3


# ---------------------------------------------------------------------------
# Section 5 -- the full cascading rebuild
# ---------------------------------------------------------------------------
def prune_ma1dcnn(model, config: dict, channel_1: int, channel_2: int, channel_3: int):
    """Structurally prune MA1DCNN's conv trunk to the given channel counts
    and rebuild every dependent layer so shapes stay consistent. This is
    the core of Phase 4 Step 2 -- it does NOT fine-tune anything;
    recover_pruned_model() below is the separate step that does that.

    Dependent layers, per pruning_code_explained.docx Section 2's table:
        conv1             -- output side only (-> channel_1)
        conv2             -- both sides (channel_1 in, channel_2 out)
        conv3             -- both sides (channel_2 in, channel_3 out)
        cam.excite[0]/[2] -- both sides, AND the hidden bottleneck width
                             itself is re-derived from channel_3 (Section 5.2)
        eam.score         -- input side only (channel_3)
        fc                -- input side only (channel_3)
        output_layer      -- never touched -- no dimension depends on a
                             pruned channel count

    At prune_ratio = 0.0 (channel_1/2/3 unchanged), the returned model is
    numerically IDENTICAL to `model` on a real forward pass
    (torch.allclose(..., atol=1e-5)) -- this is pruning_code_explained.docx
    Section 7.1's strongest self-check, and the one that actually proves
    every index above is wired to the right place, not just the right shape.
    """
    pruned_config = dict(config)
    pruned_config.update(channel_1=channel_1, channel_2=channel_2, channel_3=channel_3)
    pruned_model = build_ma1dcnn(pruned_config)

    keep1 = _topk_keep_indices(_l1_channel_importance(model.conv1.conv.weight), channel_1)
    keep2 = _topk_keep_indices(_l1_channel_importance(model.conv2.conv.weight), channel_2)
    keep3 = _topk_keep_indices(_l1_channel_importance(model.conv3.conv.weight), channel_3)

    with torch.no_grad():
        # --- 5.1 conv1 -> conv2 -> conv3 --------------------------------
        # conv1: in_channels unchanged (input is never pruned), out_channels -> keep1
        pruned_model.conv1.conv.weight.copy_(model.conv1.conv.weight[keep1, :, :])
        pruned_model.conv1.conv.bias.copy_(model.conv1.conv.bias[keep1])

        # conv2: in_channels -> keep1 (conv1's surviving outputs), out_channels -> keep2
        pruned_model.conv2.conv.weight.copy_(model.conv2.conv.weight[keep2, :, :][:, keep1, :])
        pruned_model.conv2.conv.bias.copy_(model.conv2.conv.bias[keep2])

        # conv3: in_channels -> keep2, out_channels -> keep3
        pruned_model.conv3.conv.weight.copy_(model.conv3.conv.weight[keep3, :, :][:, keep2, :])
        pruned_model.conv3.conv.bias.copy_(model.conv3.conv.bias[keep3])

        # --- 5.2 CAM: the one non-obvious dependency --------------------
        # hidden = channel_3 // CAM_REDUCTION_RATIO is DERIVED from channel_3,
        # so pruning channel_3 also shrinks the hidden dimension itself, which
        # needs its own separate importance ranking, not a blind truncation.
        old_excite_in, old_excite_out = model.cam.excite[0], model.cam.excite[2]
        new_excite_in, new_excite_out = pruned_model.cam.excite[0], pruned_model.cam.excite[2]
        new_hidden = new_excite_in.weight.shape[0]   # already sized correctly from new_c3

        # Step 1: slice the channel_3 dimension on excite[0]'s INPUT columns to keep3
        excite_in_input_pruned = old_excite_in.weight[:, keep3]               # (old_hidden, new_c3)

        # Step 2: rank the OLD hidden rows by L1-norm (using the already-sliced
        #         weights), and keep only the new_hidden best ones
        hidden_importance = excite_in_input_pruned.detach().abs().sum(dim=1)  # (old_hidden,)
        keep_hidden = _topk_keep_indices(hidden_importance, new_hidden)

        new_excite_in.weight.copy_(excite_in_input_pruned[keep_hidden, :])    # (new_hidden, new_c3)
        new_excite_in.bias.copy_(old_excite_in.bias[keep_hidden])
        new_excite_out.weight.copy_(old_excite_out.weight[keep3, :][:, keep_hidden])  # (new_c3, new_hidden)
        new_excite_out.bias.copy_(old_excite_out.bias[keep3])

        # --- 5.3 EAM, FC, output_layer -----------------------------------
        # EAM: score is Conv1d(channel_3, 1, kernel_size=1) -- in_channels -> keep3, out stays 1
        pruned_model.eam.score.weight.copy_(model.eam.score.weight[:, keep3, :])
        pruned_model.eam.score.bias.copy_(model.eam.score.bias)

        # FC: Linear(channel_3, fc_units) -- in_features (columns) -> keep3, fc_units unchanged
        pruned_model.fc.weight.copy_(model.fc.weight[:, keep3])
        pruned_model.fc.bias.copy_(model.fc.bias)

        # output_layer: Linear(fc_units, n_outputs) -- untouched, neither dimension ever pruned
        pruned_model.output_layer.weight.copy_(model.output_layer.weight)
        pruned_model.output_layer.bias.copy_(model.output_layer.bias)

    return pruned_model


# ---------------------------------------------------------------------------
# Section 9 -- fine-tune / recover each pruned variant (Phase 4, Step 3)
# ---------------------------------------------------------------------------
def recover_pruned_model(
    model,
    train_config: dict,
    train_loader,
    val_loader,
    device,
    n_epochs: int = RECOVERY_N_EPOCHS,
    patience: int = RECOVERY_PATIENCE,
):
    """Fine-tune/recover a freshly-pruned model's accuracy (Phase 4, Step 3).

    Pruning only deletes channels -- it retrains nothing, so a pruned
    model's accuracy immediately after prune_ma1dcnn() is just whatever
    survived having 34.6% / 61.7% / 81.2% of its parameters removed out
    from under it. This loop is self-contained new code: the same
    early-stopping / best-checkpoint-by-val_f1_macro structure as train.py's
    run_training(), MINUS the build_ma1dcnn() call at the top -- training
    continues directly from the pruned model's own weights (in place)
    rather than a fresh random initialization. train.py is not edited to
    add this; train_one_epoch/evaluate are imported unchanged
    (pruning_code_explained.docx Section 9.1).

    Trial13's own confirmed hyperparameters are reused unchanged for the
    recovery run, not retuned: learning_rate = 0.0004689002238683733,
    CosineAnnealingWarmRestarts(T_0=10, T_mult=2),
    weight_decay = 0.002164379191774169, threshold = 0.4214610774407138
    (Section 9.2) -- pass these via `train_config`.
    """
    model = model.to(device)
    optimizer = Adam(
        model.parameters(),
        lr=train_config["learning_rate"],
        weight_decay=train_config["weight_decay"],
    )
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    best_f1 = -1.0
    best_state = None
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, n_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device=device)
        val_metrics = evaluate(model, val_loader, device=device, threshold=train_config["threshold"])
        scheduler.step()
        history.append({"epoch": epoch, "train_loss": train_loss, **val_metrics})

        if val_metrics["f1_macro"] > best_f1:      # strictly greater -- a tie does not count
            best_f1 = val_metrics["f1_macro"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break                               # early stopping

    model.load_state_dict(best_state)   # the BEST epoch, not the last
    return model, history
