# Data Dictionary

## N-CMAPSS field groups

### W — Flight-condition descriptors

| Field | Description | Unit |
|---|---|---|
| Alt | Altitude | ft |
| Mach | Flight Mach number | — |
| TRA | Throttle-resolver angle | % |
| T2 | Fan-inlet temperature | °C |

### Xs — Sensor measurements

| Field | Description | Unit |
|---|---|---|
| Nf | Fan speed | rpm |
| Nc | Core speed | rpm |
| T24 | Station temperature (LPC outlet) | °R |
| T30 | Station temperature (HPC outlet) | °R |
| T50 | Station temperature (LPT outlet) | °R |
| Ps30 | Static pressure (HPC outlet) | psia |
| P40 | Static pressure | psia |

### Labels

| Field | Description |
|---|---|
| Fault_Label | Fault class: healthy, `fan_fail`, `lpc_fail`, `hpc_fail`, `hpt_fail`, `lpt_fail` |
| RUL | Remaining Useful Life | cycles |
| Unit | Engine unit ID (1 of 100) |
| Cycle | Flight cycle index within the unit's run-to-failure trajectory |

Fill in exact units/ranges/dtypes as EDA (`notebooks/01_eda.ipynb`)
confirms them against the DS03 HDF5 schema.

## Jetson-Bench (self-generated) fields

| Field | Description | Unit |
|---|---|---|
| sample_id | Index into the N-CMAPSS test set | — |
| model_variant | `baseline_fp32` \| `compressed_int8` \| pruning ratio variant | — |
| latency_ms | End-to-end single-sample inference latency (incl. preprocessing) | ms |
| power_w | INA3221-logged instantaneous power draw | W |
| energy_j | Energy per inference (power × latency) | J |
| timestamp | Wall-clock time of the measurement | ISO 8601 |
