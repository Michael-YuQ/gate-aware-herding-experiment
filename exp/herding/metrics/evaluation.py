# Evaluation utilities: success rate, time-to-entry, minimum robots needed, diagnostic metric validity

import numpy as np
from scipy import stats


def run_episode(world, controller, target_model, robots, max_time=300, dt=0.02, T_hold=10.0):
    """
    Main simulation loop.
    Returns dict with keys: success, time_to_entry, trajectory_target, trajectory_robots, cov_series.
    controller.compute(robot_positions, target_pos) -> list of (dx, dy) commands.
    """
    from ..sim.robot import apply_elastic_separation

    max_steps = int(max_time / dt)
    hold_steps_required = int(T_hold / dt)

    trajectory_target = []
    trajectory_robots = []
    cov_series = []

    consecutive_inside = 0
    time_to_entry = None
    success = False

    rng = np.random.default_rng()

    for step in range(max_steps):
        t = step * dt
        robot_positions = [r.pos.copy() for r in robots]

        trajectory_target.append(target_model.pos.copy())
        trajectory_robots.append(robot_positions)

        cmds = controller.compute(robot_positions, target_model.pos)

        for i, robot in enumerate(robots):
            robot.step(cmds[i])

        apply_elastic_separation(robots)

        target_model.step([r.pos for r in robots], world, rng)

        if world.is_inside(target_model.pos):
            if consecutive_inside == 0:
                time_to_entry = t
            consecutive_inside += 1
            if consecutive_inside >= hold_steps_required:
                success = True
                break
        else:
            consecutive_inside = 0
            time_to_entry = None

        cov_series.append(None)

    return {
        'success': success,
        'time_to_entry': time_to_entry if success else None,
        'trajectory_target': trajectory_target,
        'trajectory_robots': trajectory_robots,
        'cov_series': cov_series,
    }


def _sample_initial_positions(world, n_robots, rng):
    """
    Sample target on ring of radius 1.5L around enclosure center.
    Sample robot positions uniformly on ring of radius 0.7L around enclosure center.
    """
    L = world.L
    ring_radius = 1.5 * L
    angle_target = rng.uniform(0, 2 * np.pi)
    target_pos = np.array([ring_radius * np.cos(angle_target),
                            ring_radius * np.sin(angle_target)])

    robot_positions = []
    angles = np.linspace(0, 2 * np.pi, n_robots, endpoint=False)
    angles = angles + rng.uniform(0, 2 * np.pi)
    r_ring = 0.7 * L
    for angle in angles:
        rpos = np.array([r_ring * np.cos(angle), r_ring * np.sin(angle)])
        robot_positions.append(rpos)

    return target_pos, robot_positions


def evaluate_config(controller_class, config_dict, n_seeds=3):
    """
    Run multiple episodes with different seeds.
    config_dict keys: world, controller_kwargs, target_class, target_kwargs,
                      v_robot, n_robots, dt, max_time, T_hold
    """
    from ..sim.robot import HolonomicRobot

    results = []
    world = config_dict['world']
    n_robots = config_dict['n_robots']
    v_robot = config_dict['v_robot']
    dt = config_dict.get('dt', 0.02)
    max_time = config_dict.get('max_time', 300)
    T_hold = config_dict.get('T_hold', 10.0)
    target_class = config_dict['target_class']
    target_kwargs = config_dict.get('target_kwargs', {})
    controller_kwargs = config_dict.get('controller_kwargs', {})

    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        target_pos, robot_positions = _sample_initial_positions(world, n_robots, rng)

        robots = [HolonomicRobot(rpos, v_robot, dt=dt) for rpos in robot_positions]
        target = target_class(target_pos, **target_kwargs)
        controller = controller_class(world=world, v_robot=v_robot, **controller_kwargs)

        ep = run_episode(world, controller, target, robots,
                         max_time=max_time, dt=dt, T_hold=T_hold)
        results.append(ep)

    successes = [r['success'] for r in results]
    ttes = [r['time_to_entry'] for r in results if r['success'] and r['time_to_entry'] is not None]

    success_rate = float(np.mean(successes))
    median_tte = float(np.median(ttes)) if ttes else None
    ci_lo, ci_hi = compute_bootstrap_ci(successes)

    return {
        'success_rate': success_rate,
        'ci_lo': ci_lo,
        'ci_hi': ci_hi,
        'median_tte': median_tte,
        'n_success': int(np.sum(successes)),
        'n_seeds': n_seeds,
        'per_episode': results,
    }


def compute_bootstrap_ci(values, n_bootstrap=10000, ci=0.95):
    """Bootstrap confidence interval for the mean of values (e.g., success rate)."""
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(42)
    boot_means = np.array([
        np.mean(rng.choice(values, size=len(values), replace=True))
        for _ in range(n_bootstrap)
    ])
    alpha = (1.0 - ci) / 2.0
    lo = float(np.percentile(boot_means, 100 * alpha))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha)))
    return lo, hi


def compute_statistics(results_a, results_b):
    """Wilcoxon rank-sum test on time-to-entry distributions."""
    ttes_a = [r['time_to_entry'] for r in results_a if r.get('success') and r.get('time_to_entry') is not None]
    ttes_b = [r['time_to_entry'] for r in results_b if r.get('success') and r.get('time_to_entry') is not None]
    if len(ttes_a) < 2 or len(ttes_b) < 2:
        return {'statistic': None, 'pvalue': None, 'note': 'insufficient data'}
    stat, pvalue = stats.ranksums(ttes_a, ttes_b)
    return {'statistic': float(stat), 'pvalue': float(pvalue)}
