"""Wrap tegrastats/jtop to log per-inference power draw (Phase 6).

Writes CSV records matching the Jetson-Bench schema in
docs/data-dictionary.md (sample_id, model_variant, latency_ms, power_w,
energy_j, timestamp) to results/tables/jetson_bench/.

TODO: implement using jtop (https://github.com/rbonghi/jetson_stats) with
a 10 Hz power-sampling loop, idle-power subtracted, spot-checked against
an external USB power meter per the committee Q&A validation approach.
"""
