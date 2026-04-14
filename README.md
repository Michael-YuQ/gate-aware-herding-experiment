# Gate-aware Angular-Coverage Herding

This repository contains a 2D multi-robot herding simulation project for the paper idea
"Gate-aware Angular-Coverage Herding with Slower Robot Swarms in Square Enclosures with 1-4 Entrances".

The experiment code lives under `exp/`.

## Quick Start

```bash
git clone <your-repo-url>
cd experiment/exp
bash setup_env.sh
source .venv/bin/activate
python herding/scripts/run_b1.py --controller cbf_qp --n_seeds 1 --target_model repulsion_sum --n_workers 1
```

## Current Status

- Implemented and runnable:
  - core 2D square-enclosure simulator
  - target and robot dynamics
  - CBF/QP baseline controller
  - B1 experiment runner for the square-enclosure benchmark
- Present as stubs and not yet implemented:
  - `herding/controllers/coverage.py`
  - `herding/controllers/naive_push.py`
  - `herding/controllers/closed_formation.py`
  - `herding/metrics/heading.py`
  - `herding/scripts/run_b2.py`
  - `herding/scripts/run_analysis.py`

## Experiment Plan

- Proposal document: `idea/proposal.md`
- Task breakdown: `exp/task_plan.json`

## Notes

- CPU-only project. CUDA is not required.
- Generated results under `exp/herding/results/` are ignored by git.
