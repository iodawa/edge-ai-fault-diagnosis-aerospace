 # Data Dictionary
## Files in use
This project trains on nine N-CMAPSS sub-datasets, following the same data scope used in Akindoju (2025) — the closest available prior work using this same architecture and dataset. Fault-mode assignment per file follows Akindoju's documented fault-code-to-component mapping (Table 4.1), rather than the informal "Fault mode N" numbering used in an earlier draft of this table.
| File | Fault mode (component affected) | Dev units | Dev rows | Test rows | Total rows |
|---|---|---|---|---|---|
| N-CMAPSS_DS01-005.h5 | HPT failure | 6 | 4,906,636 | 2,735,232 | 7,641,868 |
| N-CMAPSS_DS02-006.h5 | HPT failure | 6 | 5,263,447 | 1,253,743 | 6,517,190 |
| N-CMAPSS_DS03-012.h5 | HPT + LPT failures | 9 | 5,571,277 | 4,251,560 | 9,822,837 |
| N-CMAPSS_DS04.h5 | Fan failure | 6 | 6,377,452 | 3,602,561 | 9,980,013 |
| N-CMAPSS_DS05.h5 | HPC failure | 6 | 4,350,606 | 2,562,046 | 6,912,652 |
| N-CMAPSS_DS06.h5 | HPC + LPC failures | 6 | 4,257,209 | 2,522,447 | 6,779,656 |
| N-CMAPSS_DS07.h5 | LPT failure | 6 | 4,350,176 | 2,869,786 | 7,219,962 |
| N-CMAPSS_DS08a-009.h5 | Multiple (fan/LPC/HPC/HPT/LPT) — catastrophic | 9 | 4,885,389 | 3,722,997 | 8,608,386 |
| N-CMAPSS_DS08c-008.h5 | Multiple (fan/LPC/HPC/HPT/LPT) — catastrophic | 6 | 4,299,918 | 2,117,819 | 6,417,737 |
| **Combined** | | **60** | **44,262,110** | **25,638,191** | **69,900,301** |
All nine files share an identical HDF5 key structure and column layout, confirmed directly in `notebooks/01_eda.ipynb`. A tenth file, `N-CMAPSS_DS08d-010.h5`, was extracted from NASA's download but found to be truncated at the source — its own internal HDF5 metadata expects 32 bytes more than the archive actually contains, confirmed via a checksum test against the archive's own record — and was excluded from the dataset. See `deliverables/ds08d_finding_and_sources.docx` for the full investigation. Fault-mode assignment source: Akindoju, T. M. (2025), Table 4.1.
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
| Field | Description |
|---|---|
| T40 | Total temperature at burner outlet |
| P30 | Total pressure at HPC outlet |
| P45 | Total pressure at HPT outlet |
| W21 | Fan flow |
| W22 | Flow out of LPC |
| W25 | Flow into HPC |
| W31 | HPT coolant bleed |
| W32 | HPT coolant bleed |
| W48 | Flow out of HPT |
| W50 | Flow out of LPT |
| SmFan | Fan stall margin |
| SmLPC | LPC stall margin |
| SmHPC | HPC stall margin |
| phi | Ratio of fuel flow to Ps30 |
Confirmed via `notebooks/01_eda.ipynb` by decoding `X_v_var` directly from `N-CMAPSS_DS03-012.h5`. Descriptions cross-checked against Chao, Kulkarni, Goebel & Fink (2020), Table 3 — this file's 14 columns are a subset of that paper's full 18-column virtual-sensor list (missing `epr`, `NRf`, `NRc`, `PCNfR`).
### T — Engine health parameters (10 columns)
| Field | Description |
|---|---|
| fan_eff_mod | Fan efficiency modifier |
| fan_flow_mod | Fan flow modifier |
| LPC_eff_mod | LPC efficiency modifier |
| LPC_flow_mod | LPC flow modifier |
| HPC_eff_mod | HPC efficiency modifier |
| HPC_flow_mod | HPC flow modifier |
| HPT_eff_mod | HPT efficiency modifier |
| HPT_flow_mod | HPT flow modifier |
| LPT_eff_mod | LPT efficiency modifier |
| LPT_flow_mod | LPT flow modifier |
Confirmed via `notebooks/01_eda.ipynb` by decoding `T_var` directly from `N-CMAPSS_DS03-012.h5` — exact match to Chao, Kulkarni, Goebel & Fink (2020), Table 4.
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
