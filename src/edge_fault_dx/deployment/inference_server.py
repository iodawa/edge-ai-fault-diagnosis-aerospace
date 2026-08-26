"""Lightweight inference wrapper with a fixed I/O contract (Phase 6).

Sensor streams in -> fault-class / RUL / power-draw out. Intended to run
containerized on JetPack (see deployment/Dockerfile) and be triggered
from the onboard sensor bus.
"""
