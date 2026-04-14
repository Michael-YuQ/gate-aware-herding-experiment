# Benchmark B1 experiment runner: square enclosure with varying entrance configs, speeds, robot counts
# Usage: python herding/scripts/run_b1.py [--controller cbf_qp] [--n_seeds 3] [--target_model all]

import argparse
import json
import os
import sys
import time
import itertools
import multiprocessing as mp
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from herding.sim.world import World
from herding.sim.robot import HolonomicRobot
from herding.sim.target import RepulsionSumTarget, MyopicAvoidanceTarget
from herding.metrics.evaluation import run_episode, compute_bootstrap_ci
from herding.controllers.cbf_qp import CBFQPController

import numpy as np


TARGET_CLASSES = {
    'repulsion_sum': RepulsionSumTarget,
    'myopic': MyopicAvoidanceTarget,
}

CONTROLLER_CLASSES = {
    'cbf_qp': CBFQPController,
}


def _sample_initial_positions(world, n_robots, rng):
    L = world.L
    ring_radius = 1.5 * L
    angle_target = rng.uniform(0, 2 * np.pi)
    target_pos = np.array([ring_radius * np.cos(angle_target),
                            ring_radius * np.sin(angle_target)])
    r_ring = 0.7 * L
    angles = np.linspace(0, 2 * np.pi, n_robots, endpoint=False)
    angles = angles + rng.uniform(0, 2 * np.pi)
    robot_positions = [np.array([r_ring * np.cos(a), r_ring * np.sin(a)]) for a in angles]
    return target_pos, robot_positions


def run_single_episode(args):
    """Worker function for multiprocessing."""
    (L, entrance_config, w_frac, v_ratio, n_robots, target_model_name, seed,
     controller_name, max_time, dt, T_hold) = args

    w = w_frac * L
    v_target = 1.0
    v_robot = v_ratio * v_target

    world = World(L=L, entrance_config=entrance_config, w=w)
    controller_class = CONTROLLER_CLASSES[controller_name]
    target_class = TARGET_CLASSES[target_model_name]

    rng = np.random.default_rng(seed)
    target_pos, robot_positions = _sample_initial_positions(world, n_robots, rng)

    robots = [HolonomicRobot(rpos, v_robot, dt=dt) for rpos in robot_positions]
    target = target_class(target_pos, v_target=v_target, dt=dt, sigma_noise=0.05)
    controller = controller_class(world=world, v_robot=v_robot)

    ep = run_episode(world, controller, target, robots, max_time=max_time, dt=dt, T_hold=T_hold)

    return {
        'L': L,
        'entrance_config': entrance_config,
        'w_frac': w_frac,
        'v_ratio': v_ratio,
        'n_robots': n_robots,
        'target_model': target_model_name,
        'seed': seed,
        'success': ep['success'],
        'time_to_entry': ep['time_to_entry'],
    }


def aggregate_results(raw_results):
    """Group by config key and compute success_rate, median_tte, and CIs."""
    from collections import defaultdict
    groups = defaultdict(list)
    for r in raw_results:
        key = (r['L'], r['entrance_config'], r['w_frac'], r['v_ratio'],
               r['n_robots'], r['target_model'])
        groups[key].append(r)

    aggregated = []
    for key, episodes in groups.items():
        L, entrance_config, w_frac, v_ratio, n_robots, target_model = key
        successes = [e['success'] for e in episodes]
        ttes = [e['time_to_entry'] for e in episodes if e['success'] and e['time_to_entry'] is not None]
        success_rate = float(np.mean(successes))
        ci_lo, ci_hi = compute_bootstrap_ci(successes)
        median_tte = float(np.median(ttes)) if ttes else None
        aggregated.append({
            'L': L,
            'entrance_config': entrance_config,
            'w_frac': w_frac,
            'v_ratio': v_ratio,
            'n_robots': n_robots,
            'target_model': target_model,
            'n_seeds': len(episodes),
            'n_success': int(np.sum(successes)),
            'success_rate': success_rate,
            'ci_lo': ci_lo,
            'ci_hi': ci_hi,
            'median_tte': median_tte,
        })
    return aggregated


def main():
    parser = argparse.ArgumentParser(description='Run B1 benchmark')
    parser.add_argument('--controller', default='cbf_qp')
    parser.add_argument('--n_seeds', type=int, default=3)
    parser.add_argument('--target_model', default='all',
                        help='repulsion_sum | myopic | all')
    parser.add_argument('--max_time', type=float, default=300.0)
    parser.add_argument('--dt', type=float, default=0.02)
    parser.add_argument('--T_hold', type=float, default=10.0)
    parser.add_argument('--n_workers', type=int, default=None,
                        help='Number of parallel workers (default: cpu_count)')
    parser.add_argument('--output_dir', default=None)
    args = parser.parse_args()

    L_list = [2, 4]
    entrance_configs = ['1', '2_opposite', '4']
    w_fracs = [0.3, 0.5]
    v_ratios = [0.5, 0.7, 0.9]
    n_robots_list = [2, 3, 4, 5, 6, 8, 10]

    if args.target_model == 'all':
        target_models = ['repulsion_sum', 'myopic']
    else:
        target_models = [args.target_model]

    seeds = list(range(args.n_seeds))

    tasks = list(itertools.product(
        L_list, entrance_configs, w_fracs, v_ratios, n_robots_list, target_models, seeds
    ))

    task_args = [
        (L, ec, wf, vr, nm, tm, seed,
         args.controller, args.max_time, args.dt, args.T_hold)
        for L, ec, wf, vr, nm, tm, seed in tasks
    ]

    total = len(task_args)
    print(f"Total episodes to run: {total}")

    if args.output_dir is not None:
        out_dir = Path(args.output_dir)
    else:
        out_dir = ROOT / 'herding' / 'results'
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / f'{args.controller}_b1_raw.json'
    agg_path = out_dir / f'{args.controller}_b1.json'

    existing_raw = []
    if raw_path.exists():
        with open(raw_path) as f:
            existing_raw = json.load(f)
        print(f"Resuming: found {len(existing_raw)} existing results.")

    existing_keys = set()
    for r in existing_raw:
        k = (r['L'], r['entrance_config'], r['w_frac'], r['v_ratio'],
             r['n_robots'], r['target_model'], r['seed'])
        existing_keys.add(k)

    pending = [
        ta for ta in task_args
        if (ta[0], ta[1], ta[2], ta[3], ta[4], ta[5], ta[6]) not in existing_keys
    ]
    print(f"Pending episodes: {len(pending)} (skipping {len(task_args) - len(pending)} completed)")

    n_workers = args.n_workers or max(1, mp.cpu_count())
    print(f"Using {n_workers} workers")

    raw_results = list(existing_raw)
    start = time.time()
    batch_size = max(1, min(200, len(pending)))

    with mp.Pool(processes=n_workers) as pool:
        done = 0
        for i in range(0, len(pending), batch_size):
            batch = pending[i:i + batch_size]
            batch_results = pool.map(run_single_episode, batch)
            raw_results.extend(batch_results)
            done += len(batch_results)
            elapsed = time.time() - start
            rate = done / elapsed if elapsed > 0 else 0
            eta = (len(pending) - done) / rate if rate > 0 else 0
            print(f"Progress: {done}/{len(pending)} | Elapsed: {elapsed:.0f}s | ETA: {eta:.0f}s")
            with open(raw_path, 'w') as f:
                json.dump(raw_results, f)

    aggregated = aggregate_results(raw_results)
    output = {
        'controller': args.controller,
        'benchmark': 'B1',
        'n_seeds': args.n_seeds,
        'dt': args.dt,
        'max_time': args.max_time,
        'T_hold': args.T_hold,
        'configurations': aggregated,
    }
    with open(agg_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nDone. Results saved to {agg_path}")
    n_cfgs = len(aggregated)
    n_success_cfgs = sum(1 for c in aggregated if c['success_rate'] > 0)
    avg_sr = float(np.mean([c['success_rate'] for c in aggregated]))
    print(f"Configs: {n_cfgs} | Configs with any success: {n_success_cfgs} | Mean success rate: {avg_sr:.3f}")


if __name__ == '__main__':
    main()
