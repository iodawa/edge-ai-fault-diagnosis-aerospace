"""export_onnx.py -- src/edge_fault_dx/deployment/

Export one E-O MA1DCNN checkpoint to ONNX, check the export against PyTorch,
and list the operators the exported graph uses (the check_ops step that
decides the ONNX Runtime attention build flags).

Three checkpoint kinds, matching how each file was actually saved:

  baseline   ma1dcnn_trial13.pt                       plain state_dict
  recovered  ma1dcnn_trial13_prunedXX_recovered.pt    dict: state_dict,
                                                      pruned_config, history, ratio
  qat        ma1dcnn_trial13_prunedXX_quantized_qat.pt whole pickled
                                                      QuantizedConvTrunk object

Run this where those files load (Colab or the workstation), NOT on the
Jetson. Only the resulting .onnx files go to the Jetson.

Design choices, all from the Step 7 plan:
  * opset 17, fixed and named, not left to the installed default.
  * dynamo=False, pinned: the new default exporter fails on the quantized
    checkpoint; the legacy exporter handles it.
  * FP32 exports (baseline, recovered): max absolute logit difference
    against PyTorch via ONNX Runtime CPU must be < 1e-4 (check_parity).
  * QAT exports are NOT bit-identical to PyTorch's INT8 kernels (rounding
    differs on a few values), so a max-absolute-difference limit is the
    wrong test. They are accepted on decisions instead (check_decision_parity):
    the PyTorch and ONNX models must give the same macro-F1 on the real
    validation windows, flip under 0.1% of decisions, and have a mean logit
    difference under 0.01 (the last guards against a graph that is wrong
    everywhere but happens to land on the same decisions). The limits were
    set after the first real run (observed: 66-81 flips of 875,885 = under
    0.01%, identical F1 to four decimals, mean difference 0.0018-0.0021),
    not before it.
"""
from __future__ import annotations

import argparse
import collections
from pathlib import Path

import numpy as np
import torch

OPSET = 17
N_INPUT_CHANNELS = 28   # train.py DEFAULT_N_INPUT_CHANNELS
WINDOW_LENGTH = 50      # windowing.py WINDOW_LENGTH
KINDS = ("baseline", "recovered", "qat")
ATOL = {"baseline": 1e-4, "recovered": 1e-4}   # FP32 kinds only; QAT uses check_decision_parity
FLIP_FRAC_MAX = 1e-3   # at most 0.1% of thresholded decisions may differ
F1_GAP_MAX = 1e-3      # macro-F1 (PyTorch vs ONNX) must agree within this
MEAN_ABS_MAX = 1e-2    # mean logit difference limit (observed 0.0018-0.0021)
# QAT operating thresholds, from the comments in compression/quantization.py;
# using them, PyTorch reproduces the recorded Step 6 macro-F1 values
# (0.8385 / 0.8306 / 0.8202) exactly, which corroborates them.
QAT_THRESHOLDS = {20: 0.425, 40: 0.425, 60: 0.400}

# Operator types / domains whose presence means the attention build flags
# must NOT be turned off (see the ONNX Runtime build plan).
ATTENTION_OPS = {"Attention", "MultiHeadAttention", "GroupQueryAttention"}


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_model(path, kind):
    """Return an eval-mode PyTorch model for the given checkpoint file."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    from edge_fault_dx.models.ma1dcnn import build_ma1dcnn

    if kind == "baseline":
        from edge_fault_dx.training.train import TRIAL_13_MODEL_CONFIG
        state = torch.load(path, map_location="cpu")
        model = build_ma1dcnn(TRIAL_13_MODEL_CONFIG)
        model.load_state_dict(state)
    elif kind == "recovered":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model = build_ma1dcnn(ckpt["pruned_config"])
        model.load_state_dict(ckpt["state_dict"])
    else:  # qat: whole pickled object; needs the repo classes importable
        from edge_fault_dx.compression.quantization import (  # noqa: F401
            QuantizedConvTrunk, repair_missing_module_dicts)
        from edge_fault_dx.models.ma1dcnn import (  # noqa: F401
            ChannelAttentionModule, Conv1dSamePadding,
            ElementwiseAttentionModule, MA1DCNN)
        import __main__
        # The checkpoint recorded these classes as living in __main__.
        for cls in (QuantizedConvTrunk, MA1DCNN, Conv1dSamePadding,
                    ChannelAttentionModule, ElementwiseAttentionModule):
            if not hasattr(__main__, cls.__name__):
                setattr(__main__, cls.__name__, cls)
        model = torch.load(path, map_location="cpu", weights_only=False)
        model = repair_missing_module_dicts(model)
    model.eval()
    return model


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------
def export_onnx(model, out_path, opset=OPSET):
    model.eval()
    dummy = torch.randn(1, N_INPUT_CHANNELS, WINDOW_LENGTH)
    torch.onnx.export(
        model, dummy, str(out_path), opset_version=opset,
        input_names=["window"], output_names=["logits"],
        dynamic_axes={"window": {0: "batch"}, "logits": {0: "batch"}},
        do_constant_folding=True,
        dynamo=False,  # required for the quantized checkpoint
    )
    return out_path


def _batches(source):
    """Yield float32 input tensors from a DataLoader ((x, y) pairs) or an
    iterable of tensors / arrays."""
    for item in source:
        x = item[0] if isinstance(item, (tuple, list)) else item
        yield torch.as_tensor(x, dtype=torch.float32)


def check_parity(model, onnx_path, windows, atol=1e-4, n_batches=20,
                 strict=True):
    """Push the same windows through PyTorch and ONNX Runtime (CPU) and
    refuse to trust the export if the largest absolute difference in the
    logits reaches atol. `windows` is a val DataLoader or any iterable of
    (batch, 28, 50) arrays. Returns max abs diff."""
    import onnxruntime as ort
    model.eval()
    sess = ort.InferenceSession(str(onnx_path),
                                providers=["CPUExecutionProvider"])
    max_diff, n_seen = 0.0, 0
    with torch.no_grad():
        for i, x in enumerate(_batches(windows)):
            if i >= n_batches:
                break
            torch_out = model(x).numpy()
            onnx_out = sess.run(["logits"], {"window": x.numpy()})[0]
            max_diff = max(max_diff, float(np.abs(torch_out - onnx_out).max()))
            n_seen += 1
    if n_seen == 0:
        raise ValueError("check_parity received no batches")
    if not max_diff < atol:
        if not strict:
            print(f"parity (informational) max abs diff {max_diff:.3e} "
                  f">= {atol} over {n_seen} batches")
            return max_diff
        raise AssertionError(
            f"ONNX export parity FAILED: max abs diff {max_diff:.3e} >= "
            f"{atol} over {n_seen} batches -- do not trust this ONNX graph.")
    print(f"parity OK -- max abs diff {max_diff:.2e} < {atol} "
          f"over {n_seen} batches")
    return max_diff


def check_decision_parity(model, onnx_path, loader, threshold,
                          flip_frac_max=FLIP_FRAC_MAX, f1_gap_max=F1_GAP_MAX,
                          mean_abs_max=MEAN_ABS_MAX):
    """Accept a QAT export on decisions, not on raw logit differences.
    Runs the WHOLE loader (real validation windows and labels) through
    PyTorch and ONNX Runtime CPU and requires (a) the fraction of
    thresholded decisions that differ <= flip_frac_max, (b) macro-F1 from
    the two models within f1_gap_max, and (c) the mean absolute logit
    difference <= mean_abs_max. Prints max / mean / 99.9th-percentile
    logit differences for the record. Returns a dict of the numbers."""
    import onnxruntime as ort
    from edge_fault_dx.training.train import compute_metrics
    model.eval()
    sess = ort.InferenceSession(str(onnx_path),
                                providers=["CPUExecutionProvider"])
    tl, ol, ys = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = torch.as_tensor(x, dtype=torch.float32)
            tl.append(model(x))
            ol.append(torch.from_numpy(
                sess.run(["logits"], {"window": x.numpy()})[0]))
            ys.append(torch.as_tensor(y))
    if not tl:
        raise ValueError("check_decision_parity received no batches")
    tl, ol, ys = torch.cat(tl), torch.cat(ol), torch.cat(ys)
    d = (tl - ol).abs()
    flat = d.flatten()
    if flat.numel() > 10_000_000:      # torch.quantile input-size limit
        flat = flat[torch.randperm(flat.numel())[:10_000_000]]
    flips = int(((torch.sigmoid(tl) >= threshold)
                 != (torch.sigmoid(ol) >= threshold)).sum().item())
    n = tl.numel()
    f1_t = float(compute_metrics(ys, tl, threshold)["f1_macro"])
    f1_o = float(compute_metrics(ys, ol, threshold)["f1_macro"])
    res = dict(max_abs=float(d.max()), mean_abs=float(d.mean()),
               p999=float(torch.quantile(flat, 0.999)), flips=flips, n=n,
               flip_frac=flips / n, f1_torch=f1_t, f1_onnx=f1_o)
    print(f"  threshold {threshold}: max abs diff {res['max_abs']:.4f}, "
          f"mean {res['mean_abs']:.6f}, 99.9th pct {res['p999']:.4f}")
    print(f"  decision flips: {flips} of {n} ({100 * res['flip_frac']:.4f}%)")
    print(f"  macro-F1 PyTorch {f1_t:.4f}  ONNX {f1_o:.4f}")
    if (res["flip_frac"] > flip_frac_max or abs(f1_t - f1_o) > f1_gap_max
            or res["mean_abs"] > mean_abs_max):
        raise AssertionError(
            "QAT decision parity FAILED -- do not trust this ONNX graph: "
            f"flip fraction {res['flip_frac']:.5f} (limit {flip_frac_max}), "
            f"F1 gap {abs(f1_t - f1_o):.5f} (limit {f1_gap_max}), "
            f"mean abs diff {res['mean_abs']:.5f} (limit {mean_abs_max}).")
    print("decision parity OK")
    return res


def verify(model, onnx_path, kind, loader, threshold=None):
    """One call per exported file: the right check for its kind."""
    if kind == "qat":
        if threshold is None:
            raise ValueError("kind='qat' needs the operating threshold")
        return check_decision_parity(model, onnx_path, loader, threshold)
    return check_parity(model, onnx_path, loader, atol=ATOL[kind])


# --------------------------------------------------------------------------
# check_ops: which operators does the exported graph use?
# --------------------------------------------------------------------------
def list_ops(onnx_path):
    import onnx
    m = onnx.load(str(onnx_path))
    counts = collections.Counter(n.op_type for n in m.graph.node)
    domains = sorted({o.domain or "ai.onnx" for o in m.opset_import})
    node_domains = sorted({n.domain or "ai.onnx" for n in m.graph.node})
    flagged = sorted(set(counts) & ATTENTION_OPS)
    contrib = "com.microsoft" in domains or "com.microsoft" in node_domains
    print("op counts:", dict(counts))
    print("opset domains:", domains, " node domains:", node_domains)
    if flagged or contrib:
        print("WARNING: attention or com.microsoft ops present "
              f"({flagged}, contrib={contrib}) -- do NOT turn the ONNX "
              "Runtime attention build flags off.")
    else:
        print("no Attention/MultiHeadAttention ops and no com.microsoft "
              "domain in this graph.")
    return counts, domains, flagged, contrib


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--kind", required=True, choices=KINDS)
    ap.add_argument("--out", required=True, help="output .onnx path")
    ap.add_argument("--val-npy", help="(N,28,50) float32 .npy of REAL "
                    "validation windows for the parity check")
    a = ap.parse_args()

    model = load_model(a.checkpoint, a.kind)
    export_onnx(model, a.out)
    print(f"exported {a.kind} model -> {a.out}")

    if a.val_npy:
        arr = np.load(a.val_npy).astype(np.float32)
        wins = [arr[i:i + 256] for i in range(0, len(arr), 256)]
        label = "REAL validation windows"
    else:
        g = np.random.default_rng(0)
        wins = [g.standard_normal((64, N_INPUT_CHANNELS, WINDOW_LENGTH))
                .astype(np.float32) for _ in range(5)]
        label = "SYNTHETIC random windows only (pass --val-npy for real)"
    print("parity data:", label)
    if a.kind == "qat":
        print("QAT: max-diff below is informational only; accept a QAT "
              "export with check_decision_parity on real validation data.")
        check_parity(model, a.out, wins, atol=1e-2, strict=False)
    else:
        check_parity(model, a.out, wins, atol=ATOL[a.kind])
    list_ops(a.out)


if __name__ == "__main__":
    main()
