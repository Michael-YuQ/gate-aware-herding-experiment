# Server Guide

This guide is for running the current repository on a Linux server.

## What Works Right Now

The repository currently supports:

- the square-enclosure B1 benchmark
- the core 2D simulator
- the CBF/QP baseline controller
- the B1 batch runner

The following parts are still stubs and should not be treated as runnable yet:

- `exp/herding/controllers/coverage.py`
- `exp/herding/controllers/naive_push.py`
- `exp/herding/controllers/closed_formation.py`
- `exp/herding/metrics/heading.py`
- `exp/herding/scripts/run_b2.py`
- `exp/herding/scripts/run_analysis.py`

## Requirements

- Linux server
- Python 3.12
- Git
- CPU only

CUDA is not required.

## Clone

```bash
git clone <REPO_URL>
cd experiment/exp
```

Replace `<REPO_URL>` with the GitHub repository URL.

## Setup

Use the included bootstrap script:

```bash
bash setup_env.sh
source .venv/bin/activate
```

Manual setup is also fine:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Smoke Test

Run a short check first:

```bash
python herding/scripts/run_b1.py \
  --controller cbf_qp \
  --n_seeds 1 \
  --target_model repulsion_sum \
  --n_workers 1 \
  --max_time 1 \
  --T_hold 0.2
```

## Normal Run

```bash
python herding/scripts/run_b1.py \
  --controller cbf_qp \
  --n_seeds 3 \
  --target_model repulsion_sum \
  --n_workers 1
```

## Outputs

Results are written to:

- `exp/herding/results/cbf_qp_b1_raw.json`
- `exp/herding/results/cbf_qp_b1.json`

## Background Run

For a longer run:

```bash
nohup python herding/scripts/run_b1.py \
  --controller cbf_qp \
  --n_seeds 3 \
  --target_model repulsion_sum \
  --n_workers 1 \
  > run_b1.log 2>&1 &
```

To watch logs:

```bash
tail -f run_b1.log
```

## Updating the Server Copy

```bash
cd /path/to/experiment
git pull
cd exp
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Then rerun the desired experiment command.
