# Edge-Optimized MA1DCNN for Real-Time Aerospace Fault Diagnosis

**Independent Research Project — Isse Odawa**

An edge-optimized, attention-based 1D convolutional neural network
(E-O MA1DCNN) for real-time aircraft engine fault classification on
resource-constrained edge hardware (NVIDIA Jetson Nano).

## Thesis

> An edge-optimized MA1DCNN predictive model is required to accelerate
> real-time aircraft engine fault diagnosis, preventing catastrophic
> failures and reducing unscheduled maintenance costs.

Current fault-diagnosis models are too computationally complex to run
on-device, which delays fault identification and drives unscheduled
maintenance costs (priced at $78/minute by Gerdes et al., 2016). This
project closes the algorithm-to-hardware gap by compressing a
multi-head-attention 1D-CNN (Optuna-tuned hyperparameters, mixed-precision
training, structured pruning, INT8 quantization via TensorRT) and
validating it on a Jetson Nano against a 98.16% workstation-accuracy
baseline.

Full problem framing, hypotheses, and methodology live in
[`docs/praxis/`](docs/praxis/) — see the index below.

## Repository structure

```
edge-ai-fault-diagnosis-aerospace/
├── README.md                   ← you are here
├── LICENSE
├── CITATION.cff
├── environment.yml / requirements.txt
├── docs/
│   ├── praxis/                 ← the praxis proposal itself, one file per section
│   ├── data-dictionary.md      ← N-CMAPSS + Jetson-Bench field definitions
│   ├── model-card.md           ← model details, intended use, metrics, limitations
│   ├── glossary.md             ← terms & acronyms
│   └── architecture.md         ← system/pipeline diagram & design notes
├── data/
│   ├── raw/                    ← N-CMAPSS .h5 (gitignored, see data/README.md)
│   ├── interim/                ← cleaned/augmented intermediates (gitignored)
│   └── processed/              ← train/val/test splits (gitignored)
├── notebooks/                  ← exploratory work, numbered by pipeline stage
├── src/edge_fault_dx/
│   ├── data/                   ← loaders, preprocessing, augmentation
│   ├── models/                 ← MA1DCNN, E-O MA1DCNN architectures
│   ├── training/                ← training loop, Optuna search
│   ├── compression/            ← pruning, INT8 quantization
│   ├── evaluation/             ← accuracy/macro-precision/latency/energy metrics
│   └── deployment/             ← ONNX export, TensorRT build, inference server
├── configs/                    ← YAML configs per experiment (baseline, optuna, deploy)
├── scripts/                    ← thin CLI entry points that wire configs → src/
├── deployment/                 ← Dockerfile, docker-compose, Jetson power-logging
├── results/                    ← figures, tables, generated reports (gitignored, sample kept)
├── tests/                      ← unit tests for data/model/metric code
└── .github/workflows/          ← CI (lint + tests on push)
```

The `docs/praxis/` folder is deliberately structured to mirror the praxis
proposal itself, so every reviewed claim (problem statement,
thesis, research questions, hypotheses, methodology, data sources) has a
single canonical, version-controlled home instead of living only in a
slide deck.

## Praxis documentation index

| # | Document | Contents |
|---|---|---|
| 0 | [Scope of Work](docs/praxis/00-scope-of-work.md) | Objective, novelty, methodology summary, deliverables |
| 1 | [Problem Statement](docs/praxis/01-problem-statement.md) | Issue, "so what," problem statement, industry context |
| 2 | [Thesis Statement](docs/praxis/02-thesis-statement.md) | Research product, format, scope, inputs/outputs |
| 3 | [Research Questions](docs/praxis/03-research-questions.md) | RQ1–RQ3 |
| 4 | [Hypotheses](docs/praxis/04-hypotheses.md) | H1–H3, variables, testability |
| 5 | [Annotated Bibliography](docs/praxis/05-annotated-bibliography.md) | 6 sources: summary, methodology, evaluation, relevance |
| 6 | [Methodology / Graphical Model](docs/praxis/06-methodology-graphical-model.md) | 6-phase research pipeline (what/why/how) |
| 7 | [Data Sources](docs/praxis/07-data-sources.md) | N-CMAPSS + Jetson-Bench source records, hypothesis-to-data alignment |
| 8 | [Committee Q&A Log](docs/praxis/08-committee-qa-log.md) | Running record of advisor/committee questions and responses |

## Quickstart

```bash
git clone <this-repo-url>
cd edge-ai-fault-diagnosis-aerospace
conda env create -f environment.yml   # or: pip install -r requirements.txt
conda activate edge-fault-dx
```

Data setup: N-CMAPSS is not committed to the repo (see
[`data/README.md`](data/README.md) for the NASA PCoE download link and
expected layout under `data/raw/`).

```bash
# example pipeline stages once data is in place
python scripts/run_baseline.py --config configs/baseline.yaml
python scripts/run_compression.py --config configs/compression.yaml
python scripts/benchmark_jetson.py --config configs/jetson_deploy.yaml
```

## Targets this repo is measured against

| Metric | Target | Baseline |
|---|---|---|
| Accuracy | ≥ 93.25% (within 5% of baseline) | 98.16% (workstation, FP32) |
| Inference latency (Jetson Nano) | ≥ 25% reduction vs. FP32 | — |
| Energy per inference | ≥ 20% reduction | — |
| Power draw | ≥ 3% reduction | — |

See [`docs/praxis/04-hypotheses.md`](docs/praxis/04-hypotheses.md) for full
testability criteria.

## Citing this work

See [`CITATION.cff`](CITATION.cff).

## License

See [`LICENSE`](LICENSE) — replace the placeholder with the license your
program requires (many independent-research repos use an
all-rights-reserved or CC BY-NC placeholder until publication clears).
