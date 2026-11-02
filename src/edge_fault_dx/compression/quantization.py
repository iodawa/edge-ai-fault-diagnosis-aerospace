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

StandardCNN ADDITIONS (Phase 5, H1): quantize_standard_cnn_ptq() and
quantize_standard_cnn_qat(), plus the QuantizedStandardCNNTrunk wrapper
they build on. Extracted verbatim from
h1_standard_cnn_and_noise_robustness.ipynb Sections 4-4.1 (real results in
h1_notebook_code_walkthrough.docx Sections 8-9 and praxis_writeup.docx
Section 4.5.1). The wrapper is simpler than QuantizedConvTrunk because
StandardCNN has no cam/eam to carry through unquantized -- conv3's
dequantized output goes straight into global_avg_pool. Only fc/
output_layer are frozen during the QAT fine-tune (StandardCNN's own
version of "only the trunk is fine-tuned"), matching the scope boundary
quantize_model_qat() already uses for MA1DCNN.

THREE REAL BUGS, found only by actually running this (notebook Section
4.1 -- all three fixes are in quantize_standard_cnn_ptq()/
quantize_standard_cnn_qat() below, not just documented):

1. QuantizedStandardCNNTrunk reuses conv1/conv2/conv3 by REFERENCE
   (self.conv1 = model.conv1, not a copy). torch.quantization.convert()
   therefore mutates those same submodules on the CALLER's own `model`
   object, not just on the wrapper -- silently replacing the caller's
   FP32 conv trunk with INT8 modules. If the caller later reused that
   same `model` for QAT after PTQ missed the retention bar, QAT wrapped
   an already-quantized trunk with no ordinary trainable parameters left
   and Adam(...) raised "optimizer got an empty parameter list". Fixed
   by deep-copying `model` before wrapping, so PTQ and QAT each get
   their own independent, still-FP32 copy of the recovered checkpoint.
2. Eager-mode PyTorch quantization (the "fbgemm" backend used here) only
   has CPU kernels for its quantized ops -- there is no CUDA
   implementation of quantized::conv1d / quantized::linear. On a CUDA
   runtime, wrapping a CUDA model crashed with a device-mismatch error
   the moment calibration batches (plain CPU tensors from the
   DataLoader) hit it; moving the batches to CUDA instead would not have
   fixed it, since convert() would then try to build quantized ops on a
   CUDA module, which fbgemm does not support at all. Fixed by always
   quantizing on CPU here, regardless of what device the rest of the
   notebook trains/benchmarks FP32 models on.
3. torch.quantization.prepare()/prepare_qat() recursively propagate
   wrapped.qconfig to every child module that does not set its own --
   including fc/output_layer, even though they are meant to stay
   unquantized "by reference". Left unset, convert() silently turned
   them into real quantized nn.Linear, and forward() then fed them an
   FP32 tensor right after dequant(), crashing with NotImplementedError:
   Could not run 'quantized::linear' with arguments from the 'CPU'
   backend. Fixed by explicitly setting qconfig=None on everything after
   dequant() (global_avg_pool, dropout, fc, fc_activation, output_layer)
   before calling prepare()/prepare_qat().

Real, measured results (StandardCNN's own retention bar = 0.781868,
i.e. its own confirmed baseline 0.831868 minus 5 points -- NOT MA1DCNN's
0.7940 bar; praxis_writeup.docx Section 4.5.1):

    ratio | FP32 recovered | PTQ macro-F1        | QAT macro-F1
    20%   | 0.834464        | 0.790884 PASSED     | 0.834483 PASSED (forced,
          |                 | (bar 0.781868)      | Section 6.1 -- see below)
    40%   | 0.841175        | 0.778157 BELOW BAR  | 0.831526 PASSED
    60%   | 0.808305        | 0.722852 BELOW BAR  | 0.802401 PASSED

Unlike MA1DCNN, StandardCNN's 20% PTQ checkpoint actually cleared its own
bar (0.790884 > 0.781868) -- PTQ was not a uniform failure mode across
architectures, only at 40%/60% here. QAT at 20% was run anyway (notebook
Section 6.1), not because PTQ needed a fallback at that ratio, but because
H1's controlled comparison (Section 8.1) needed both architectures on the
SAME compression method (QAT) at the SAME ratio (20%) to isolate
"architecture only" as the one varying factor -- comparing MA1DCNN-QAT
against StandardCNN-PTQ would have confounded architecture with
compression method, which is exactly what the notebook's Section 8
confounded-vs-controlled split exists to show.

HONEST GAP, carried over unchanged from the MA1DCNN path above:
resweep_threshold() was never wired into the compression loop that
produced any of these StandardCNN checkpoints. Every verdict in both
tables above used the fixed 0.4215 threshold tuned against MA1DCNN's own
uncompressed probability calibration -- not a threshold re-swept per
compressed StandardCNN variant, the way three of MA1DCNN's own checkpoints
were (PTQ at 0.400 for all ratios; QAT at 0.425 for 20%/40%, 0.400 for
60%). This was a deliberate scope decision documented in
h1_completion_plan.docx Section 6 (re-sweeping was left for a later pass,
not silently skipped), not an oversight discovered after the fact -- but
it means every StandardCNN number above is, strictly, a lower bound: a
re-swept threshold could only move these figures up, never down.
"""

from __future__ import annotations

import copy
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

    NOT YET WIRED IN for StandardCNN (see module docstring's "HONEST GAP"
    note) -- every StandardCNN quantization verdict in this module still
    used the fixed 0.4215 threshold, not a StandardCNN-specific re-sweep.
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

    Reused unchanged for StandardCNN -- it walks named_modules() generically
    and has no MA1DCNN-specific assumption, so the same function that
    verified 43,522/25,467/12,486 for MA1DCNN also verified StandardCNN's
    own quantized checkpoints' parameter counts (38,192 / 22,367 / 11,147,
    before quantization -- quantization does not change these either).
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


# =============================================================================
# StandardCNN additions (Phase 5, H1) -- h1_standard_cnn_and_noise_robustness
# .ipynb Sections 5-6; real results in h1_notebook_code_walkthrough.docx
# Sections 8-9 and praxis_writeup.docx Section 4.5.1. See module docstring's
# "StandardCNN ADDITIONS" section for the full real-numbers table and the
# honest resweep_threshold-not-wired-in caveat that applies to every
# function below.
# =============================================================================
class QuantizedStandardCNNTrunk(nn.Module):
    """StandardCNN's version of QuantizedConvTrunk. Simpler than the
    MA1DCNN wrapper above because there is no cam/eam to carry through
    unquantized -- conv3's dequantized output goes straight into
    global_avg_pool, matching StandardCNN.forward()'s own structure
    (models/standard_cnn.py) with conv1/conv2/conv3 now behind a
    QuantStub/DeQuantStub pair instead of running in plain FP32.
    fc/fc_activation/output_layer/dropout/global_avg_pool are reused BY
    REFERENCE, unquantized, from the original recovered StandardCNN --
    same "only the trunk is quantized" scope boundary as MA1DCNN's own
    QuantizedConvTrunk.
    """

    def __init__(self, model):
        super().__init__()
        self.quant = torch.quantization.QuantStub()
        self.conv1, self.conv2, self.conv3 = model.conv1, model.conv2, model.conv3
        self.dequant = torch.quantization.DeQuantStub()
        self.global_avg_pool = model.global_avg_pool
        self.dropout = model.dropout
        self.fc = model.fc
        self.fc_activation = model.fc_activation
        self.output_layer = model.output_layer

    def forward(self, x):
        x = self.quant(x)
        x = self.conv3(self.conv2(self.conv1(x)))
        x = self.dequant(x)                       # INT8 -> FP32 before global_avg_pool
        return self.output_layer(self.fc_activation(self.fc(self.dropout(
            self.global_avg_pool(x).view(x.shape[0], -1)))))


def quantize_standard_cnn_ptq(model, calib_loader, n_calib_batches: int = 200):
    """StandardCNN counterpart to quantize_model_ptq() above -- identical
    procedure, wrapped in QuantizedStandardCNNTrunk instead of
    QuantizedConvTrunk since StandardCNN has no cam/eam to carry through
    unquantized.

    Real result, StandardCNN's own 0.781868 retention bar
    (praxis_writeup.docx Section 4.5.1):
        20% -> 0.790884  PASSED
        40% -> 0.778157  BELOW BAR
        60% -> 0.722852  BELOW BAR
    Note this differs from MA1DCNN's pattern -- MA1DCNN's PTQ missed its
    own bar at all three ratios, but StandardCNN's PTQ actually cleared
    its bar at 20%. See module docstring's "HONEST GAP" note: the fixed
    0.4215 threshold was used for this verdict, not a re-swept one.

    See module docstring's "THREE REAL BUGS" note for why this deep-copies
    `model` and forces CPU rather than wrapping it directly on `device`.
    """
    wrapped = QuantizedStandardCNNTrunk(copy.deepcopy(model).to("cpu"))
    wrapped.eval()
    wrapped.qconfig = torch.quantization.get_default_qconfig("fbgemm")
    # Bug 3 (module docstring): prepare() would otherwise propagate
    # wrapped.qconfig to fc/output_layer too, despite the class docstring
    # saying they're "unquantized, by reference". Opt them out for real.
    for name in ("global_avg_pool", "dropout", "fc", "fc_activation", "output_layer"):
        getattr(wrapped, name).qconfig = None
    torch.quantization.prepare(wrapped, inplace=True)      # inserts observers

    with torch.no_grad():
        for i, (X, _) in enumerate(calib_loader):
            if i >= n_calib_batches:
                break
            wrapped(X.to("cpu"))                            # observers watch, output discarded

    torch.quantization.convert(wrapped, inplace=True)       # the actual FP32 -> INT8 swap
    return wrapped


def quantize_standard_cnn_qat(
    model,
    train_config: dict,
    train_loader,
    val_loader,
    device,
    n_epochs: int = QAT_N_EPOCHS,
    patience: int = QAT_PATIENCE,
):
    """StandardCNN counterpart to quantize_model_qat() above -- identical
    prepare_qat()/FakeQuantize/fine-tune/convert procedure. Only fc and
    output_layer are frozen during the trunk fine-tune (one fewer module
    than MA1DCNN's version, since StandardCNN has no cam/eam).

    Real result, StandardCNN's own 0.781868 retention bar:
        40% -> 0.831526  PASSED
        60% -> 0.802401  PASSED
        20% -> 0.834483  PASSED (forced run, notebook Section 6.1 -- PTQ
                                 had already cleared the bar at 20%, but
                                 H1's controlled comparison, Section 8.1,
                                 needed a StandardCNN-QAT checkpoint at 20%
                                 to compare against MA1DCNN-QAT at 20% on
                                 the SAME compression method, isolating
                                 architecture as the one varying factor)
    Params at 20% after pruning (unchanged by quantization, verified via
    count_params_any()): 38,192.

    Same deepcopy and CPU-forcing fixes as quantize_standard_cnn_ptq()
    above, same reasons (module docstring's "THREE REAL BUGS" note): this
    still ends in torch.quantization.convert(), which needs fbgemm's
    CPU-only quantized ops, so the fine-tune leading up to it has to
    happen on CPU too. `device` is kept in the signature for call-site
    compatibility with quantize_model_qat() but intentionally NOT used to
    place anything here -- QAT always runs on CPU, independent of
    whatever device the rest of the notebook trains FP32 models on.

    NOTE: the notebook's own version of this fine-tune loop reuses a
    module-level `loss_fn` global rather than constructing one locally --
    a notebook-only convenience that does not survive extraction into an
    importable module. BCEWithLogitsLoss() is instantiated locally here
    instead, matching quantize_model_qat()'s own existing convention
    above (and run_training_like()'s loss_fn in the notebook is itself
    just nn.BCEWithLogitsLoss(), so this changes no behavior).
    """
    wrapped = QuantizedStandardCNNTrunk(copy.deepcopy(model).to("cpu"))
    wrapped.train()
    wrapped.qconfig = torch.quantization.get_default_qat_qconfig("fbgemm")   # QAT qconfig, not static
    # Bug 3 (module docstring), same fix as quantize_standard_cnn_ptq() above.
    for name in ("global_avg_pool", "dropout", "fc", "fc_activation", "output_layer"):
        getattr(wrapped, name).qconfig = None
    torch.quantization.prepare_qat(wrapped, inplace=True)   # inserts FakeQuantize, not plain observers

    for name in ("fc", "output_layer"):
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
        # "cpu" here, not `device` -- see the CPU-forcing note above.
        train_one_epoch(wrapped, train_loader, optimizer, loss_fn, "cpu")
        val_metrics = evaluate(wrapped, val_loader, loss_fn, "cpu", threshold=train_config["threshold"])
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
