"""Edge-Optimized MA1DCNN for real-time aerospace fault diagnosis.

Package layout mirrors the six-phase praxis methodology
(docs/praxis/06-methodology-graphical-model.md):

    data          -> Phase 1-2 (collect, preprocess, augment)
    models        -> Phase 3 (MA1DCNN / E-O MA1DCNN architectures)
    training      -> Phase 3 (training loop, Optuna search)
    compression   -> Phase 4 (pruning, INT8 quantization)
    evaluation    -> Phase 5 (metrics, noise-robustness)
    deployment    -> Phase 6 (ONNX export, TensorRT build, inference server)
"""
