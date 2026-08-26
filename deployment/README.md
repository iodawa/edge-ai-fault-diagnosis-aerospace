# Deployment

- `Dockerfile` / `docker-compose.yml` — containerize the inference server
  for JetPack (per the committee Q&A: "containerized via Docker on
  JetPack and triggered automatically from the onboard sensor bus").
- `jetson/tegrastats_logger.py` — wraps `tegrastats`/`jtop` to log
  per-inference power draw (INA3221) into `results/tables/jetson_bench/`.

JetPack + TensorRT are installed on the Jetson Nano itself, not via
`environment.yml` — follow NVIDIA's JetPack SDK install instructions for
the Jetson Nano before running anything in this folder.
