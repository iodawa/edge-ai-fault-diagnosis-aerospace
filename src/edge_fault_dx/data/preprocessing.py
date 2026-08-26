"""Clean, normalize, augment, and split N-CMAPSS data (Phase 2).

TODO:
- Normalize each sensor channel to zero mean / unit variance.
- Inject Gaussian and impulse noise (synthetic-to-real gap, per
  Liao et al. 2024's -6dB noise-robustness precedent).
- Perform a time-ordered train/validation/test split per engine unit.
"""
