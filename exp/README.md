# Gate-aware Angular-Coverage Herding with Slower Robot Swarms

A 2D multi-robot herding simulation study around square enclosures with 1-4 entrances.

## Environment

- Python 3.12
- CPU-only
- No CUDA dependency

## Setup

Linux or macOS:

```bash
bash setup_env.sh
source .venv/bin/activate
```

Manual setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## What Works Right Now

- `herding/sim/`: square enclosure world, target models, robot model, collision helpers
- `herding/controllers/cbf_qp.py`: CBF/QP baseline
- `herding/metrics/evaluation.py`: episode rollout and aggregation helpers
- `herding/scripts/run_b1.py`: B1 batch runner

## Not Implemented Yet

- `herding/controllers/coverage.py`
- `herding/controllers/naive_push.py`
- `herding/controllers/closed_formation.py`
- `herding/metrics/heading.py`
- `herding/scripts/run_b2.py`
- `herding/scripts/run_analysis.py`

## Run B1

Quick smoke test:

```bash
python herding/scripts/run_b1.py --controller cbf_qp --n_seeds 1 --target_model repulsion_sum --n_workers 1 --max_time 1 --T_hold 0.2
```

Longer run:

```bash
python herding/scripts/run_b1.py --controller cbf_qp --n_seeds 3 --target_model repulsion_sum --n_workers 1
```

## Files You Will Care About

- Experiment runner: `herding/scripts/run_b1.py`
- Baseline controller: `herding/controllers/cbf_qp.py`
- Proposal: `../idea/proposal.md`
- Task plan: `task_plan.json`

## Output

- Raw and aggregated experiment outputs are written to `herding/results/`
- Generated figures should go to `herding/figures/`
