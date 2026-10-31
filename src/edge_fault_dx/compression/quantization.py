"""Post-training INT8 quantization: static PTQ (Phase 4, Step 5 attempt 1),
QAT fallback (Step 5 attempt 2), and the support code Step 6's quantized-
model benchmark needed (parameter counting, checkpoint-reload repair).

Extracted from 04_compression_pruning_quant.ipynb and documented function-
by-function in pruning_code_explained.docx (Sections 11-12). This replaces
the one-line TODO stub previously at this path.

Scope: only the convolutional trunk (conv1, conv2, conv3) is ever
quantized -- never the whole MA1DCNN. ChannelAttentionModule and
ElementwiseAttentionModule both perform elementwise multiplies that
PyTorch's eager-mode quantization cannot handle without rewriting those
classes through torch.ao.nn.quantized.FloatFunctional, which would mean
editing MA1DCNN itself -- violating this project's "architecture repeated
verbatim, never modified" convention (Section 11.1).

Real, measured results (Sections 11.2-11.4; retention bar = 0.7940,
i.e. baseline 0.8440 minus 5 points):

    ratio | FP32 recovered | PTQ macro-F1       | QAT macro-F1
    20%   | 0.8436          | 0.7768 BELOW BAR   | 0.8385 PASSED (+0.0445)
    40%   | 0.8445          | 0.7475 BELOW BAR   | 0.8306 PASSED (+0.0366)
    60%   | 0.8254          | 0.7485 BELOW BAR   | 0.8202 PASSED (+0.0262)

PTQ missed the retention bar at all three ratios, consistently -- a real
property of static PTQ on this model (it never lets the model compensate
for INT8 rounding during training), not a bug. QAT was built in direct
response and cleared the bar at all three. The QAT checkpoints
(ma1dcnn_trial13_prunedXX_quantized_qat.pt) are what Step 6 and everything
downstream should measure, not the PTQ ones (Section 11.4).

NOTE ON PROVENANCE: QuantizedConvTrunk, the PTQ prepare/calibrate/convert
block, the QAT prepare_qat/freeze/convert block, count_params_any(), and
repair_missing_module_dicts() (with its _REQUIRED_MODULE_DICTS tuple) are
copied verbatim from pruning_code_explained.docx. Full function signatures,
the training loop around the QAT fine-tune (the doc shows the structural
difference from PTQ -- prepare_qat() instead of prepare(), the
cam/eam/fc/output_layer freeze -- but elides the loop body with "..." since
it is the same best-epoch/early-stopping pattern as recover_pruned_model()
in pruning.py), and imports were completed to make this a runnable module.
resweep_threshold() is described only as "ported from threshold_sweep.py's
own sweep() logic" (Section 11.2) -- that file does not yet exist anywhere
in src/edge_fault_dx/ (checked against the live repo tree), so the stub
below documents the real re-swept threshold values this project found but
cannot import a sweep() that was never extracted into src/. Check all of
this against the notebook's literal source before treating it as
byte-identical.
"""

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn
from torch.optim import Adam

from ..training.train import evaluate, train_one_epoch

# ---------------------------------------------------------------------------
# Section 11.3 -- QAT fine-tune budget, named explicitly
# ---------------------------------------------------------------------------
QAT_N_EPOCHS = 8
QAT_PATIENCE = 3
QAT_LR_SCALE = 0.1   # trial13's own learning_rate x this -- small enough not
                     # to undo recovery's already-confirmed weights, large
                     # enough for the trunk to adapt within QAT_N_EPOCHS


# ---------------------------------------------------------------------------
# Section 11.1 -- the quantized wrapper
# ---------------------------------------------------------------------------
class QuantizedConvTrunk(nn.Module):
    """Holds conv1/conv2/conv3 behind a QuantStub/DeQuantStub pair; reuses
    cam/eam/global_avg_pool/dropout/fc/fc_activation/output_layer BY
    REFERENCE, unquantized, from the original recovered model. conv3 alone
    is 61.8% of the baseline's total parameters, so quantizing just the
    trunk still captures most of the compression opportunity despite
    leaving everything downstream at full FP32 precision.
    """

    def __init__(self, model):
        super().__init__()
        self.quant = torch.quantization.QuantStub()
        self.conv1, self.conv2, self.conv3 = model.conv1, model.conv2, model.conv3
        self.dequant = torch.quantization.DeQuantStub()
        self.cam, self.eam, self.fc = model.cam, model.eam, model.fc   # unquantized, by reference
        self.global_avg_pool = model.global_avg_pool
        self.dropout = model.dropout
        self.fc_activation = model.fc_activation
        self.output_layer = model.output_layer

    def forward(self, x):
        x = self.quant(x)                            # FP32 -> INT8
        x = self.conv3(self.conv2(self.conv1(x)))     # INT8 straight through the trunk
        x = self.dequant(x)                           # INT8 -> FP32, before CAM's elementwise multiply
        return self.output_layer(self.fc_activation(self.fc(self.dropout(
            self.global_avg_pool(self.eam(self.cam(x))).view(x.shape[0], -1)))))


# ---------------------------------------------------------------------------
# Section 11.2 -- static PTQ (Step 5, attempt 1)
# ---------------------------------------------------------------------------
def quantize_model_ptq(model, calib_loader, n_calib_batches: int = 200):
    """Static post-training quantization of MA1DCNN's conv trunk. One
    calibration pass, no gradients: assigns fbgemm's default per-channel
    qconfig to the trunk only (qconfig=None on everything downstream, so
    prepare()/convert() leave those modules untouched), runs real batches
    from the training split through it in eval() mode so each observer can
    record the min/max range it actually sees, then converts -- swapping
    each observed FP32 module for its real INT8 equivalent using those
    recorded ranges. No weight is ever updated by a gradient in this path.

    Real result, all three pruning ratios: missed the 0.7940 retention bar
    (0.7768 / 0.7475 / 0.7485) -- see module docstring.
    """
    wrapped = QuantizedConvTrunk(model)
    wrapped.eval()
    wrapped.qconfig = torch.quantization.get_default_qconfig("fbgemm")
    torch.quantization.prepare(wrapped, inplace=True)      # inserts observers

    with torch.no_grad():
        for i, (X, _) in enumerate(calib_loader):
            if i >= n_calib_batches:
                break
            wrapped(X)                                      # observers watch, output discarded

    torch.quantization.convert(wrapped, inplace=True)       # the actual FP32 -> INT8 swap
    return wrapped


# ---------------------------------------------------------------------------
# Section 11.3 -- QAT fallback (Step 5, attempt 2), triggered by PTQ's result
# ---------------------------------------------------------------------------
def quantize_model_qat(
    model,
    train_config: dict,
    train_loader,
    val_loader,
    device,
    n_epochs: int = QAT_N_EPOCHS,
    patience: int = QAT_PATIENCE,
):
    """Differs from quantize_model_ptq in exactly one structural way:
    instead of a single calibration pass with no gradients, it calls
    prepare_qat() (not prepare()), which inserts FakeQuantize modules into
    the trunk's forward pass -- they still round activations/weights to
    what INT8 would represent, but the backward pass, via PyTorch's
    straight-through gradient estimator, can now flow through that
    rounding and update conv1/conv2/conv3's real FP32 weights to be robust
    to it. cam/eam/fc/output_layer are frozen for the whole fine-tune --
    same scope boundary as PTQ, keeping the experiment "teach the trunk to
    tolerate quantization" rather than a second, different full-model
    fine-tune. Reuses train_one_epoch/evaluate from train.py unchanged,
    same convention as recover_pruned_model() in pruning.py.

    Real result, all three pruning ratios: cleared the 0.7940 retention
    bar (0.8385 / 0.8306 / 0.8202), recovering 6.2-8.3 points of macro-F1
    over PTQ at every ratio -- see module docstring.

    NOTE: train_one_epoch()'s and evaluate()'s real signatures (confirmed
    against train.ipynb) are (model, loader, optimizer, loss_fn, device)
    and (model, loader, loss_fn, device, threshold) respectively -- both
    take loss_fn as a required argument. An earlier version of this
    function omitted it; fixed here to BCEWithLogitsLoss(), matching
    run_training()'s own loss_fn in train.ipynb Section 9.
    """
    wrapped = QuantizedConvTrunk(model)
    wrapped.train()
    wrapped.qconfig = torch.quantization.get_default_qat_qconfig("fbgemm")   # QAT qconfig, not static
    torch.quantization.prepare_qat(wrapped, inplace=True)   # inserts FakeQuantize, not plain observers

    for name in ("cam", "eam", "fc", "output_layer"):
        for p in getattr(wrapped, name).parameters():
            p.requires_grad_(False)                          # only the trunk is fine-tuned

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = Adam(
        (p for p in wrapped.parameters() if p.requires_grad),
        lr=train_config["learning_rate"] * QAT_LR_SCALE,
    )

    best_f1 = -1.0
    best_state = None
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, n_epochs + 1):                     # up to 8 epochs, patience=3
        train_one_epoch(wrapped, train_loader, optimizer, loss_fn, device)
        val_metrics = evaluate(wrapped, val_loader, loss_fn, device, threshold=train_config["threshold"])
        history.append({"epoch": epoch, **val_metrics})

        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            best_state = {k: v.clone() for k, v in wrapped.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    wrapped.load_state_dict(best_state)
    wrapped.eval()
    torch.quantization.convert(wrapped, inplace=True)        # same final step as PTQ
    return wrapped, history


# ---------------------------------------------------------------------------
# Section 2 (phase4_architecture.docx) / 11.2-11.3 -- re-sweep the operating
# threshold against each compressed model's own validation output
# ---------------------------------------------------------------------------
def resweep_threshold(model, val_loader, device, thresholds=None):
    """trial13's threshold=0.4215 was tuned against the UNCOMPRESSED
    model's own probability calibration. Pruning and quantization can
    shift that calibration, so the threshold must be re-swept against each
    compressed model's own validation output, not inherited unchanged
    (phase4_architecture.docx Section 2).

    pruning_code_explained.docx Section 11.2 describes this as "ported
    from threshold_sweep.py's own sweep() logic, re-pointed at the
    compressed model's predictions". That module lives at
    src/edge_fault_dx/training/threshold_sweep.py -- it exists locally but,
    like this file until now, had not yet been pushed to GitHub as of this
    extraction; push it alongside pruning.py/quantization.py or this import
    will fail. Real re-swept thresholds this project found: PTQ landed all
    three ratios at 0.400; QAT landed 20% and 40% at 0.425, and 60% at 0.400.
    """
    from ..training.threshold_sweep import sweep
    return sweep(model, val_loader, device=device, thresholds=thresholds)


# ---------------------------------------------------------------------------
# Section 12.1 -- counting parameters on a model with quantized submodules
# ---------------------------------------------------------------------------
def count_params_any(model) -> int:
    """A naive sum(p.numel() for p in model.parameters()) silently drops a
    quantized Conv1d's weight entirely: PyTorch's quantized Conv1d does
    not store its INT8 weight as an ordinary nn.Parameter -- it packs
    weight and bias together into an opaque PackedParams object accessed
    via ._weight_bias(), which .parameters() never iterates into. Run
    against a QAT-quantized checkpoint, the naive count silently reports
    only cam/eam/fc/output_layer's FP32 parameters and quietly treats the
    entire INT8 trunk as zero -- not an error, just a wrong, too-small
    number with no warning attached. This walks every submodule explicitly
    instead of trusting .parameters().

    Caught and fixed before the real Step 6 numbers were produced: the fix
    was verified by comparing against the pre-quantization pruned model's
    own already-confirmed parameter count (43,522 / 25,467 / 12,486) --
    quantization changes a weight's dtype, never how many weights the
    model has, and with this function the two counts match exactly
    (pruning_code_explained.docx Section 12.1).
    """
    total = 0
    for name, m in model.named_modules():
        if hasattr(m, "_weight_bias"):                 # quantized Conv1d / Linear
            w, b = m._weight_bias()
            total += w.numel() + (b.numel() if b is not None else 0)
        elif list(m.named_parameters(recurse=False)):   # ordinary FP32 modules
            total += sum(p.numel() for p in m.parameters(recurse=False))
    return total


# ---------------------------------------------------------------------------
# Section 12.2 -- cross-process checkpoint-reload repair
# ---------------------------------------------------------------------------
_REQUIRED_MODULE_DICTS = (
    "_parameters", "_buffers", "_non_persistent_buffers_set",
    "_backward_hooks", "_is_full_backward_hook", "_forward_hooks",
    "_forward_hooks_with_kwargs", "_forward_hooks_always_called",
    "_forward_pre_hooks", "_forward_pre_hooks_with_kwargs",
    "_backward_pre_hooks", "_state_dict_hooks", "_state_dict_pre_hooks",
    "_load_state_dict_pre_hooks", "_load_state_dict_post_hooks", "_modules",
)


def repair_missing_module_dicts(model):
    """Loading a QAT-quantized checkpoint back from disk in a genuinely
    separate Python process (matching a Colab runtime restarting overnight
    between saving a QAT checkpoint and reloading it) reliably corrupts
    the quantized Conv1d submodules' internal nn.Module bookkeeping:
    torch.save(qat_model, path) followed by torch.load(path) in a fresh
    process loses roughly 16 of nn.Module's internal dict attributes on
    each quantized Conv1d, while leaving the actual trained INT8 weights
    untouched. The first forward() call then fails with
    AttributeError: 'Conv1d' object has no attribute '_backward_hooks'.

    This was confirmed as a reliable, reproducing bug (not an occasional
    fluke) by a true two-process test, and the repair itself was verified
    to change nothing about what the model computes: a reference input was
    pushed through the model before pickling, and again after reload +
    repair, and the two outputs matched bit-for-bit via torch.equal()
    (pruning_code_explained.docx Section 12.2). Every attribute restored
    here is empty bookkeeping machinery nn.Module expects to exist, never
    a tensor with learned values in it.
    """
    for name, m in model.named_modules():
        for attr in _REQUIRED_MODULE_DICTS:
            if attr not in m.__dict__:
                object.__setattr__(m, attr, OrderedDict())
    return model
