# Gate-aware Angular-Coverage Herding with Slower Robot Swarms in Square Enclosures with 1–4 Entrances

A 2D multi-robot herding simulation study comparing an entrance-aware heading-space coverage controller against CBF/QP, naive-push, and closed-formation baselines.

## Environment

- Python 3.12, CPU-only (no GPU dependencies)
- Virtual environment at `.venv/`

**Activate:**
```bash
source .venv/bin/activate
```

**Key packages:**
- `numpy 2.4.4` — array operations
- `scipy 1.17.0` — optimization, Wilcoxon rank-sum, bootstrap CIs
- `cvxpy 1.8.2` — QP solver for the CBF/QP baseline
- `matplotlib 3.10.8` — trajectory and result plots
- `seaborn 0.13.2` — statistical plot styling
- `pandas 2.3.3` — result aggregation and tabulation

## Project Structure

```
herding/
├── sim/            # 2D simulation engine (world, target, robot, collision)
├── controllers/    # CBF/QP baseline, naive push, closed-formation, coverage controller
├── metrics/        # Heading-space coverage metric (Cov(t)) and evaluation
├── configs/        # Experiment configuration files (YAML or Python dicts)
├── scripts/        # Experiment runners (run_b1.py, run_b2.py, run_analysis.py)
├── results/        # Raw results (JSON/CSV per configuration)
├── figures/        # Generated plots and visualizations
└── utils/          # Shared helpers (logging, seeding, config loading)
```

## Running Experiments

```bash
source .venv/bin/activate

# Benchmark B1: square enclosure
python herding/scripts/run_b1.py

# Benchmark B2: narrow-passage
python herding/scripts/run_b2.py

# Ablation, sensitivity, and diagnostic analysis
python herding/scripts/run_analysis.py
```

## Results

Raw results are written to `results/` (JSON/CSV). Figures are saved to `figures/`. Experiment summaries are in `EXPERIMENT_RESULTS/`.

## Task Progress

See `task_plan.json` for the full task list and completion status.
