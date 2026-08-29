# Data Dictionary

## Files in use

This project trains on three N-CMAPSS sub-datasets, each representing a distinct engine fault mode, combined into one training set:

| File | Fault mode | Dev units | Dev rows | Test rows | Total rows |
|---|---|---|---|---|---|
| N-CMAPSS_DS01-005.h5 | Fault mode 1 (single-component) | 6 | 4,906,636 | 2,735,232 | 7,641,868 |
| N-CMAPSS_DS03-012.h5 | Fault mode 2 (dual-component) | 9 | 5,571,277 | 4,251,560 | 9,822,837 |
| N-CMAPSS_DS08a-009.h5 | Fault mode 7 (multi-component) | 9 | 4,885,389 | 3,722,997 | 8,608,386 |
| **Combined** | | **24** | **15,363,302** | **10,709,789** | **26,073,091** |

All three files share an identical HDF5 key structure and column layout, confirmed directly in `notebooks/01_eda.ipynb`.

## N-CMAPSS field groups (verified against the real files)

### W — Flight-condition descriptors (4 columns)

| Field | Description |
|---|---|
| alt | Altitude |
| Mach | Flight Mach number |
| TRA | Throttle-resolver angle |
| T2 | Fan-inlet temperature |

### X_s — Physical sensor measurements (14 columns)

| Field | Description |
|---|---|
| T24 | LPC outlet temperature |
| T30 | HPC outlet temperature |
| T48 | HPT outlet temperature |
| T50 | LPT outlet temperature |
| P15 | Bypass-duct pressure |
| P2 | Fan-inlet pressure |
| P21 | LPC outlet pressure |
| P24 | LPC outlet pressure (duct) |
| Ps30 | HPC outlet static pressure |
| P40 | Burner exit pressure |
| P50 | LPT outlet pressure |
| Nf | Fan speed |
| Nc | Core speed |
| Wf | Fuel flow |

Units (°R for temperatures, psia for pressures, rpm for speeds) follow standard N-CMAPSS convention — confirm exact units against NASA's documentation bundled with the original download before final write-up.

### X_v — Virtual/derived sensor measurements (14 columns)

Confirmed present, 14 columns, float64. **Column names not yet decoded.** Run the same decode step already used for `W_var`/`X_s_var`:
```python
xv_names = [n.decode() if isinstance(n, bytes) else n for n in f['X_v_var'][:]]
```
Update this section once that output is captured.

### T — Engine health parameters (10 columns)

Confirmed present, 10 columns, float64. **Column names not yet decoded** — same treatment needed for `T_var`. Per general N-CMAPSS documentation this group typically holds flow/efficiency degradation modifiers for the five rotating sub-components, but this should be confirmed from `T_var` directly rather than assumed.

### A — Auxiliary / identifying data (4 columns)

| Field | Description |
|---|---|
| unit | Engine unit ID — numbering restarts per file, not global (see per-file unit counts above) |
| cycle | Flight cycle index within that unit's run-to-failure trajectory |
| Fc | Flight class |
| hs | Health state: 1 = healthy, 0 = degraded |

### Y — Remaining Useful Life target (1 column)

Single integer column: cycles remaining until failure. No `_var` name list — a single value needs no column labels.

## Derived training label (built in `preprocessing.py` — not native to the files)

None of the three files contains a native multi-class fault label. The training label is derived as:

- `hs == 1` → class 0 (healthy), pooled across all three files
- `hs == 0` in `N-CMAPSS_DS01-005.h5` → class 1
- `hs == 0` in `N-CMAPSS_DS03-012.h5` → class 2
- `hs == 0` in `N-CMAPSS_DS08a-009.h5` → class 3

Pooling healthy samples across files forces the model to learn genuine degradation signatures rather than per-file artifacts — see `deliverables/multiclass_fault_classification_plan.docx` for the full reasoning.

## Jetson-Bench (self-generated) fields

| Field | Description | Unit |
|---|---|---|
| sample_id | Index into the N-CMAPSS test set | — |
| model_variant | `baseline_fp32` \| `compressed_int8` \| pruning ratio variant | — |
| latency_ms | End-to-end single-sample inference latency (incl. preprocessing) | ms |
| power_w | INA3221-logged instantaneous power draw | W |
| energy_j | Energy per inference (power × latency) | J |
| timestamp | Wall-clock time of the measurement | ISO 8601 |
