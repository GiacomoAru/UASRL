# === Standard library ===
import math
import os
import pickle
import time
from typing import Tuple
import sys
import contextlib

# === Third-party libraries ===
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize

# === Optimization / Control ===
import osqp
import scipy.sparse as sp

# === Plotting ===
import matplotlib.pyplot as plt

import re

####################################################################################################
####################################################################################################

#   ╔═════════╗
#   ║   CBF   ║
#   ╚═════════╝


@contextlib.contextmanager
def suppress_osqp_output():
    """
    Context manager to suppress OSQP solver output.

    Temporarily redirects `stdout` and `stderr` to `/dev/null` in order to 
    silence logs produced by the OSQP solver. Once the context is exited, 
    the original streams are restored.

    Yields
    ------
    None
        Control is returned to the enclosed code block, with OSQP output suppressed.
    """

    # Redirect stdout and stderr to /dev/null (silence solver logs)
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield  # run the code block inside context
        finally:
            # Restore original stdout and stderr
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def cbf_velocity_filter_qp(
    v_cmd: float,
    omega_cmd: float,
    ray_distances: np.ndarray,
    ray_angles: np.ndarray,
    d_safe: float = 0.5,
    alpha: float = 5.0,
    d_safe_threshold_mult: float = 3.0,
    debug: bool = False
) -> tuple[float, float]:
    """
    Apply a Control Barrier Function (CBF) quadratic program filter 
    to enforce safety constraints on velocity commands.

    Given nominal linear and angular velocities, along with LIDAR 
    ray distances and angles, this function formulates and solves 
    a quadratic program (QP) that minimally modifies the commands 
    to ensure obstacles remain outside a safe distance.

    Parameters
    ----------
    v_cmd : float
        Nominal forward (linear) velocity command.
    omega_cmd : float
        Nominal angular velocity command.
    ray_distances : np.ndarray
        Array of LIDAR distances for each ray.
    ray_angles : np.ndarray
        Array of angles (in radians) corresponding to each LIDAR ray.
    d_safe : float, optional
        Minimum safe distance from obstacles (default is 0.5).
    alpha : float, optional
        CBF relaxation parameter controlling constraint aggressiveness (default is 5.0).
    d_safe_threshold_mult : float, optional
        Multiplier on `d_safe` that defines the maximum distance at which 
        obstacles are considered in the constraints (default is 3.0).
    debug : bool, optional
        If True, OSQP solver messages are shown and debug information is printed (default is False).

    Returns
    -------
    tuple of float
        A tuple (v_safe, omega_safe) representing the filtered forward 
        and angular velocities that satisfy the CBF constraints.
    """

    # Robot assumed at origin in its local frame
    robot_state = np.array([0.0, 0.0, 0.0])  
    nominal_u = np.array([v_cmd, omega_cmd])

    # Convert lidar polar coordinates to Cartesian
    obstacles = np.column_stack((
        ray_distances * np.cos(ray_angles),
        ray_distances * np.sin(ray_angles)
    ))

    max_considered_distance = d_safe * d_safe_threshold_mult
    A_list, b_list = [], []

    for obs in obstacles:
        delta = robot_state[:2] - obs
        dist = np.linalg.norm(delta)

        # Skip obstacles too far to be relevant
        if dist > max_considered_distance:
            continue

        # Barrier function h = distance^2 - d_safe^2
        h = dist**2 - d_safe**2
        x, y, theta = robot_state
        v_nom, omega_nom = nominal_u

        # Derivatives of h wrt control inputs
        dh_dv = 2 * (delta[0] * np.cos(theta) + delta[1] * np.sin(theta))
        dh_domega = 2 * (delta[0] * -np.sin(theta) + delta[1] * np.cos(theta))

        # Inequality constraint
        a_i = -np.array([dh_dv, dh_domega])
        b_i = alpha * h + 2 * (
            delta[0] * v_nom * np.cos(theta)
            + delta[1] * v_nom * np.sin(theta)
            + np.dot(delta, [-np.sin(theta), np.cos(theta)]) * omega_nom
        )

        A_list.append(a_i)
        b_list.append(b_i)

    # If no relevant constraints, return original commands
    if not A_list:
        return v_cmd, omega_cmd

    # Build QP matrices
    A = sp.csc_matrix(np.vstack(A_list))
    l = -np.inf * np.ones_like(b_list)
    u = np.array(b_list)

    # Add constraint: v >= 0 (no backward motion)
    A_vel = sp.csc_matrix([[1.0, 0.0]])
    l_vel = np.array([0.0])
    u_vel = np.array([np.inf])

    # Stack together
    A_full = sp.vstack([A, A_vel])
    l_full = np.hstack([l, l_vel])
    u_full = np.hstack([u, u_vel])

    # Quadratic cost: minimize deviation from nominal command
    P = sp.csc_matrix(np.eye(2) * 2.0)
    q = np.zeros(2)

    # Solve QP
    prob = osqp.OSQP()
    prob.setup(P=P, q=q, A=A_full, l=l_full, u=u_full, verbose=debug, polish=True)

    if debug:
        res = prob.solve()
    else:
        with suppress_osqp_output():
            res = prob.solve()

    if res.info.status != 'solved':
        if debug:
            print("OSQP failed:", res.info.status)
        return v_cmd, omega_cmd

    dv, domega = res.x
    return v_cmd + dv, omega_cmd + domega

def CBF_from_obs(ray_obs, action, env_info, 
                 d_safe, alpha, d_safe_mul,
                 precomputed_angles_rad=None):
    """
    Apply a Control Barrier Function (CBF) filter to policy actions 
    using ray-based observations.

    This function takes normalized policy outputs (forward and angular velocities) 
    along with ray sensor observations, converts them into physical velocities, 
    and applies a CBF quadratic program filter to enforce safety constraints. 
    The filtered actions are then normalized back to the policy’s action space.

    Parameters
    ----------
    ray_obs : array-like
        Normalized ray sensor observations (values in [0, 1]).
    action : array-like
        Policy network outputs (normalized forward and angular velocities).
    env_info : object
        Environment information object containing sensor and agent settings.
        Must include:
            - ray_sensor_settings['rays_per_direction'] (int)
            - ray_sensor_settings['max_ray_degrees'] (float)
            - ray_sensor_settings['ray_length'] (float)
            - agent_settings['max_movement_speed'] (float)
            - agent_settings['max_turn_speed'] (float)
    d_safe : float
        Minimum safe distance from obstacles.
    alpha : float
        CBF relaxation parameter controlling constraint aggressiveness.
    d_safe_mul : float
        Multiplier on `d_safe` defining the maximum distance considered 
        for constraints.
    precomputed_angles_rad : np.ndarray, optional
        Precomputed ray angles in radians. If None, they are generated 
        from environment settings (default is None).

    Returns
    -------
    np.ndarray
        A 2-element array [v_safe_norm, omega_safe_norm], representing 
        the safe normalized forward and angular velocities (in range [-1, 1]).
    """

    # Precompute ray angles once if not already provided
    if precomputed_angles_rad is None:
        precomputed_angles_rad = generate_angles_rad(
            env_info.settings['ray_sensor_settings']['rays_per_direction'],
            env_info.settings['ray_sensor_settings']['max_ray_degrees']
        )
        
    # Convert normalized ray observations into distances
    ray_distances = [x * env_info.settings['ray_sensor_settings']['ray_length'] for x in ray_obs]

    # Policy network outputs are normalized velocities (not accelerations)
    nn_v_front = action[0] * env_info.settings['agent_settings']['max_movement_speed']
    nn_v_ang = np.radians(action[1] * env_info.settings['agent_settings']['max_turn_speed'])

    # Apply Control Barrier Function via QP filter
    v_safe, omega_safe = cbf_velocity_filter_qp(
        nn_v_front, nn_v_ang,
        ray_distances, precomputed_angles_rad,
        d_safe=d_safe,
        alpha=alpha,
        d_safe_threshold_mult=d_safe_mul,
        debug=False
    )

    # Normalize outputs back to [-1, 1] (compatible with policy space)
    v_safe_norm = v_safe / env_info.settings['agent_settings']['max_movement_speed']
    omega_safe_norm = np.degrees(omega_safe) / env_info.settings['agent_settings']['max_turn_speed']

    return np.array([v_safe_norm, omega_safe_norm])


def generate_angles_rad(k, n):
    """
    Generate evenly spaced ray angles in radians for a ray sensor.

    The function creates a symmetric set of ray angles centered at 0, 
    covering the range [-n, n] degrees. The total number of rays is 
    `2 * k + 1`, where `k` is the number of rays per side.

    Parameters
    ----------
    k : int
        Number of rays per side. The total number of rays will be `2 * k + 1`.
    n : float
        Maximum angle in degrees (positive side). The rays will cover 
        from +n to -n.

    Returns
    -------
    list of float
        List of ray angles in radians, ordered from left (+n) to right (-n).
    """

    if k == 1:
        # Only one ray → straight ahead
        return [0.0]
    
    # k = number of rays per side → total = 2*k + 1 (including center ray)
    total_rays = k * 2 + 1
    
    # Angular step (degrees) between consecutive rays
    step = 2 * n / (total_rays - 1)
    
    # Generate angles from left (+n) to right (-n)
    angoli_gradi = [n - i * step for i in range(total_rays)]
    
    # Convert to radians
    angoli_radianti = [math.radians(a) for a in angoli_gradi]
    
    return angoli_radianti



####################################################################################################
####################################################################################################

#   ╔═════════════════════╗
#   ║   Data Menagement   ║
#   ╚═════════════════════╝


scalar_keys = [
    'total_reward',
    'total_length',
    'total_collisions',
    'total_success',
    'mean_u_e',
    'std_u_e',
    'uf_activations_tot',
    'cbf_activations_tot',
    'uf_when_cbf',
    'cbf_when_uf',
    'steps',
    'inner_steps_mean'
]

list_keys = [
    'u_e',
    'uf_activation',
    'cbf_activation_avg',
    'cbf_mean_change',
    'dist_goal',
    'angle_goal',
    'dist_ema',
    'angle_ema',
    'f_velocity',
    'l_velocity',
    'r_velocity',
    'f_action',
    'r_action'
] + [f'ray_{i}' for i in range(17)]


def extract_stats(episode, msg, CONFIG_DICT):
    """
    Extract and compute statistics from a single episode of agent interaction.

    This function processes episode step data and environment summary messages 
    to compute aggregated statistics about rewards, collisions, success, 
    uncertainty filter (UF) activity, Control Barrier Function (CBF) corrections, 
    velocities, goals, and ray sensor readings.

    Parameters
    ----------
    episode : list of dict
        Sequence of step dictionaries. Each step contains:
            - 'u_e' : float
                Uncertainty estimation value.
            - 'uf_activation' : bool
                Whether the uncertainty filter was activated.
            - 'inner_steps' : list of (float, int)
                Sub-steps with CBF correction magnitude and activation flag.
            - 'state' : array-like
                State vector, last 7 entries include velocities and goal info.
            - 'ray' : list
                Ray sensor observations (distances).
            - 'action' : array-like
                Executed action [forward, rotation].
    msg : dict
        Episode-level information containing:
            - 'reward' : float
            - 'length' : int
            - 'collisions' : int
            - 'success' : bool
    CONFIG_DICT : dict
        Configuration dictionary (not directly used, but included for consistency).

    Returns
    -------
    dict
        Dictionary of aggregated statistics, including:
            - total_reward : float
            - total_length : int
            - total_collisions : int
            - total_success : int
            - mean_u_e : float
            - std_u_e : float
            - uf_activations_tot : int
            - cbf_activations_tot : int
            - uf_when_cbf : int
            - cbf_when_uf : int
            - u_e, uf_activation, cbf_activation_avg, cbf_mean_change : list
            - dist_goal, angle_goal, dist_ema, angle_ema : list
            - f_velocity, l_velocity, r_velocity : list
            - f_action, r_action : list
            - ray_i : list
                For each ray index (0 to 16).
            - steps : int
            - inner_steps_mean : float
    """

    # Initialize stats container
    ret = {
        'total_reward': 0,
        'total_length': 0,
        'total_collisions': 0,
        'total_success': 0,

        'mean_u_e': 0,
        'std_u_e': 0,
        
        'uf_activations_tot': 0,       # total UF activations
        'cbf_activations_tot': 0,      # total CBF activations
        'uf_when_cbf': 0,              # UF triggered when CBF active
        'cbf_when_uf': 0,              # CBF triggered when UF active
        
        'u_e': [],
        'uf_activation': [],
        'cbf_activation_avg': [],
        'cbf_mean_change': [],         # average magnitude of CBF corrections
        
        'dist_goal': [],
        'angle_goal': [],
        'dist_ema': [],
        'angle_ema': [],
        
        'f_velocity': [],
        'l_velocity': [],
        'r_velocity': [],
        
        'f_action': [],
        'r_action': [],

        'steps': 0,
        'inner_steps_mean': 0
    }
    
    # Preallocate lists for each ray sensor
    for i in range(17):  # 17 rays
        ret[f'ray_{i}'] = []
        
    # Episode-level info from environment message
    ret['total_reward'] = msg['reward']
    ret['total_length'] = msg['length']
    ret['total_collisions'] = msg['collisions']
    ret['total_success'] = int(msg['success'])
    ret['steps'] = len(episode)
    
    # Step-by-step processing
    for step in episode:
        if len(step['inner_steps']) == 0:
            print(step)  # debug: unexpected empty inner_steps
        
        # Uncertainty filter
        ret['u_e'].append(step['u_e'])
        ret['uf_activation'].append(step['uf_activation'])
        
        # Aggregate CBF activity across inner steps
        cbf_act_avg = 0
        cbf_mean_change = 0
        for in_s in step['inner_steps']:
            cbf_act_avg += in_s[1]      # activation flag
            cbf_mean_change += in_s[0]  # correction magnitude
        
        ret['cbf_activations_tot'] += cbf_act_avg
        if step['uf_activation']:
            ret['cbf_when_uf'] += cbf_act_avg
        if cbf_act_avg > 0:
            ret['uf_when_cbf'] += step['uf_activation']
            
        # Store mean correction and activation ratio
        ret['cbf_mean_change'].append(cbf_mean_change / cbf_act_avg if cbf_act_avg > 0 else 0)
        ret['cbf_activation_avg'].append(cbf_act_avg / len(step['inner_steps']))
        
        # Track average number of inner steps
        ret['inner_steps_mean'] += len(step['inner_steps'])
        
        # Parse state vector (last 7 features are velocities + goal info)
        state = step['state'][-7:]
        ret['f_velocity'].append(state[0])
        ret['l_velocity'].append(state[1])
        ret['r_velocity'].append(state[2])
        
        ret['dist_goal'].append(state[3])
        ret['angle_goal'].append(state[4])
        
        ret['dist_ema'].append(state[5])
        ret['angle_ema'].append(state[6])

        # Store ray distances from last frame
        for i, r in enumerate(step['ray'][-1]):
            ret[f'ray_{i}'].append(r)
            
        # Store executed actions
        ret['f_action'].append(step['action'][0])
        ret['r_action'].append(step['action'][1])
        
    # Aggregate statistics
    u_e = np.array(ret['u_e'])
    ret['mean_u_e'] = u_e.mean()
    ret['std_u_e'] = u_e.std()
    ret['uf_activations_tot'] = np.array(ret['uf_activation']).sum()
    
    # Normalize mean inner steps
    ret['inner_steps_mean'] /= len(episode)
    return ret


def save_stats(stats, env_info, config_dict,
               test_name, RESULTS_DIR="./results", duration=None):
    """
    Save detailed statistics and summaries from multiple episodes.

    This function stores raw statistics, environment information, and configuration 
    in a pickle file. It also builds aggregate summaries (overall, successes only, 
    failures only) and updates CSV files that track results across multiple tests.

    Parameters
    ----------
    stats : list of dict
        List of episode-level statistics, typically produced by `extract_stats`.
    env_info : object
        Environment information object with configuration and settings.
    config_dict : dict
        Configuration dictionary used for the experiment.
    test_name : str
        Name of the test run, used as filename and for experiment identification.
    RESULTS_DIR : str, optional
        Directory where results are saved (default is "./results").
    duration : float or None, optional
        Duration of the experiment in seconds. If None, defaults to -1 in summaries.

    Outputs
    -------
    {RESULTS_DIR}/{test_name}.pkl : pickle
        File containing a dictionary with 'stats', 'env_info', and 'config_dict'.
    all_tests.csv : CSV
        Aggregated statistics for all test runs.
    all_success.csv : CSV
        Aggregated statistics for successful episodes only.
    all_failures.csv : CSV
        Aggregated statistics for failed episodes only.

    Notes
    -----
    The summary files contain mean and standard deviation of:
        - total_reward
        - total_success
        - total_length
        - total_collisions
        - UF activation percentages
        - CBF activation percentages
        - true positive / false positive / true negative / false negative rates
    """

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Save raw stats object (consistent with base version)
    to_save = {
        'stats': stats,
        'env_info': env_info,
        'config_dict': config_dict
    }
    with open(os.path.join(RESULTS_DIR, f"{test_name}.pkl"), "wb") as f:
        pickle.dump(to_save, f)

    # Helper to extract number from test_name using regex
    def extract_number(pattern, default=-1):
        m = re.search(pattern, test_name)
        return int(float(m.group(1))) if m else default

    percentile = extract_number(r'_(\d+(?:\.\d+)?)pctl')
    cbf_config = extract_number(r'_cbf(\d+(?:\.\d+)?)_')

    # Build summary statistics from a list of episodes
    def build_summary(episodes):
        if not episodes:
            return None

        def avg_and_std(key):
            vals = [ep_stats[key] for ep_stats in episodes
                    if key in ep_stats and np.isscalar(ep_stats[key]) and np.issubdtype(type(ep_stats[key]), np.number)]
            if vals:
                return float(np.mean(vals)), float(np.std(vals))
            return 0.0, 0.0

        total_reward_mean, total_reward_std = avg_and_std("total_reward")
        total_success_mean, total_success_std = avg_and_std("total_success")
        total_length_mean, total_length_std = avg_and_std("total_length")
        total_collisions_mean, total_collisions_std = avg_and_std("total_collisions")

        def mean_std_over_episodes(fn):
            vals = [fn(x) for x in episodes]
            return float(np.mean(vals)), float(np.std(vals))

        uf_mean, uf_std = mean_std_over_episodes(lambda x: np.array(x['uf_activation']).mean())
        cbf_mean, cbf_std = mean_std_over_episodes(lambda x: np.array(x['cbf_activation_avg']).mean())

        u_e_thr = config_dict['uncertainty_filter']['threshold']
        tp_mean, tp_std = mean_std_over_episodes(lambda x: ((np.array(x['u_e']) > u_e_thr) & (np.array(x['cbf_activation_avg']) > 0)).mean())
        fp_mean, fp_std = mean_std_over_episodes(lambda x: ((np.array(x['u_e']) > u_e_thr) & (np.array(x['cbf_activation_avg']) == 0)).mean())
        tn_mean, tn_std = mean_std_over_episodes(lambda x: ((np.array(x['u_e']) < u_e_thr) & (np.array(x['cbf_activation_avg']) == 0)).mean())
        fn_mean, fn_std = mean_std_over_episodes(lambda x: ((np.array(x['u_e']) < u_e_thr) & (np.array(x['cbf_activation_avg']) > 0)).mean())

        ep_count = len(episodes)
        
        return {
            "test_name": test_name,
            "percentile": percentile,
            "cbf_config": cbf_config,
            "episode_count": ep_count,

            "total_reward_mean": total_reward_mean,
            "total_reward_std": total_reward_std,

            "total_success_mean": total_success_mean,
            "total_success_std": total_success_std,

            "total_length_mean": total_length_mean,
            "total_length_std": total_length_std,

            "total_collisions_mean": total_collisions_mean,
            "total_collisions_std": total_collisions_std,

            "uf_activations_perc_mean": uf_mean,
            "uf_activations_perc_std": uf_std,

            "cbf_activations_perc_mean": cbf_mean,
            "cbf_activations_perc_std": cbf_std,

            "true_positive_mean": tp_mean,
            "true_positive_std": tp_std,

            "false_positive_mean": fp_mean,
            "false_positive_std": fp_std,

            "true_negative_mean": tn_mean,
            "true_negative_std": tn_std,

            "false_negative_mean": fn_mean,
            "false_negative_std": fn_std,

            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration": duration if duration is not None else -1
        }

    summary_all = build_summary(stats)
    summary_success = build_summary([ep for ep in stats if ep.get("total_success", 0) > 0])
    summary_fail = build_summary([ep for ep in stats if ep.get("total_success", 0) == 0])

    # Update CSV files with new results
    def update_csv(summary, filename):
        if summary is None:
            summary = {
                "test_name": test_name,
                "percentile": percentile,
                "cbf_config": cbf_config,
                "episode_count": 0,
                "total_reward_mean": np.nan,
                "total_reward_std": np.nan,
                "total_success_mean": np.nan,
                "total_success_std": np.nan,
                "total_length_mean": np.nan,
                "total_length_std": np.nan,
                "total_collisions_mean": np.nan,
                "total_collisions_std": np.nan,
                "uf_activations_perc_mean": np.nan,
                "uf_activations_perc_std": np.nan,
                "cbf_activations_perc_mean": np.nan,
                "cbf_activations_perc_std": np.nan,
                "true_positive_mean": np.nan,
                "true_positive_std": np.nan,
                "false_positive_mean": np.nan,
                "false_positive_std": np.nan,
                "true_negative_mean": np.nan,
                "true_negative_std": np.nan,
                "false_negative_mean": np.nan,
                "false_negative_std": np.nan,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "duration": duration if duration is not None else -1
            }
            
        csv_path = os.path.join(RESULTS_DIR, filename)
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df = pd.concat([df, pd.DataFrame([summary])], ignore_index=True)
        else:
            df = pd.DataFrame([summary])
        df.to_csv(csv_path, index=False)

    update_csv(summary_all, "all_tests.csv")
    update_csv(summary_success, "all_success.csv")
    update_csv(summary_fail, "all_failures.csv")

def load_stats(test_name, RESULTS_DIR="./results"):
    """
    Load previously saved statistics from a pickle file.

    Parameters
    ----------
    test_name : str
        Name of the test run, used to identify the pickle file.
    RESULTS_DIR : str, optional
        Directory where results are stored (default is "./results").

    Returns
    -------
    dict
        Dictionary containing the saved data with keys:
            - 'stats' : list of dict
                Episode-level statistics.
            - 'env_info' : object
                Environment information object.
            - 'config_dict' : dict
                Experiment configuration dictionary.
    """

    # Load previously saved stats object from pickle file
    
    with open(os.path.join(RESULTS_DIR, f"{test_name}.pkl"), "rb") as f:
        stats = pickle.load(f)
    return stats


def load_global_stats(RESULTS_DIR="./results"):
    """
    Load aggregated global test statistics from CSV.

    This function reads the "all_tests.csv" file, containing summaries 
    of all experiments, and returns a sorted DataFrame. If the file 
    does not exist or is empty, returns None.

    Parameters
    ----------
    RESULTS_DIR : str, optional
        Directory where results are stored (default is "./results").

    Returns
    -------
    pandas.DataFrame or None
        DataFrame with global test statistics, sorted by timestamp. 
        Returns None if the file does not exist or is empty.
    """

    csv_path = os.path.join(RESULTS_DIR, "all_tests.csv")

    # Check if file exists
    if not os.path.exists(csv_path):
        print("No global test file found.")
        return None

    # Load CSV, ignore commented lines
    df = pd.read_csv(csv_path, comment="#")

    if df.empty:
        print("Test file is empty.")
        return None

    # Sort by timestamp in ascending order
    df_sorted = df.sort_values("timestamp")

    return df_sorted



####################################################################################################
####################################################################################################

#   ╔════════════════════════╗
#   ║   Data Visualization   ║
#   ╚════════════════════════╝


def print_stats_light(all_stats, tot_episodes):

    # Numero di episodi considerati (dati disponibili)
    n = len(all_stats)
    if n == 0:
        print(print_text + "No stats available.")
        return

    print_text = f"[ep {n}/{tot_episodes}] "
    # Funzione per media per una chiave, solo valori numerici scalari
    def avg(key):
        vals = [ep_stats[key] for ep_stats in all_stats
                if key in ep_stats and np.isscalar(ep_stats[key]) and np.issubdtype(type(ep_stats[key]), np.number)]
        return sum(vals) / len(vals) if vals else 0


    print_text += (
        f"R: {avg('total_reward'):.2f} | "
        f"S: {avg('total_success'):.4f} | "
        f"C: {avg('total_collisions'):.2f} | "
        f"L: {avg('total_length'):.1f} | "
        f"UF: {avg('uf_activations_tot'):.2f} | "
        f"CBF: {avg('cbf_activations_tot'):.2f} | "
        f"UF->CBF: {avg('uf_when_cbf'):.2f} | "
        f"CBF->UF: {avg('cbf_when_uf'):.2f}"
    )
    print(print_text)

def get_stats_summary(all_stats, filter_fn= lambda x: True):
    filtered = [ep for ep in all_stats if filter_fn(ep)]
    n = len(filtered)
    if n == 0:
        print("No episodes matching filter.")
        return pd.DataFrame()  # ritorna DataFrame vuoto

    # Calcola le statistiche tabellari scalar
    rows = []
    for key in scalar_keys:
        vals = [ep[key] for ep in filtered
                if key in ep and np.isscalar(ep[key]) and np.issubdtype(type(ep[key]), np.number)]
        
        if not vals:
            # Valori mancanti, salta o metti NaN
            rows.append({
                'stat': key,
                'min': np.nan, '25%': np.nan, '50%': np.nan,
                '75%': np.nan, 'max': np.nan, 'mean': np.nan, 'std': np.nan
            })
            continue

        arr = np.array(vals)
        rows.append({
            'stat': key,
            'min': arr.min(),
            '25%': np.percentile(arr, 25),
            '50%': np.percentile(arr, 50),
            '75%': np.percentile(arr, 75),
            'max': arr.max(),
            'mean': arr.mean(),
            'std': arr.std()
        })

    df = pd.DataFrame(rows).set_index('stat')
    print(f"Stats summary on {n} episodes")
    return df


def plot_stats_old(
    all_stats,
    test_name,
    
    keys=None,
    filter_fn=lambda x: True,
    plot_type='joint',
    smooth_time=False,
    
    log_scale=False,
    figsize=(6, 4)
):
    
    filtered = [ep for ep in all_stats if filter_fn(ep)]
    n = len(filtered)
    if n == 0:
        print("No episodes matching filter.")
        return

    def get_scalar_values(key):
        vals = [ep[key] for ep in filtered if key in ep and not isinstance(ep[key], list)]
        return vals

    def get_aggregated_list_values(key):
        values = []
        for ep in filtered:
            if key in ep and isinstance(ep[key], list):
                values.extend(ep[key])
        return values

    def get_time_series(key):
        values_by_time = []
        for ep in filtered:
            if key in ep and isinstance(ep[key], list):
                values_by_time.append(ep[key])
        return values_by_time

    # === OPZIONE HELP ===
    if keys is None:
        scalar_keys = set()
        list_keys = set()
        empty_or_mixed = set()

        for ep in filtered:
            for k, v in ep.items():
                if isinstance(v, list):
                    list_keys.add(k)
                elif isinstance(v, (int, float, np.number)):
                    scalar_keys.add(k)
                else:
                    empty_or_mixed.add(k)

        print(f"\nAvailable keys (from {n} episodes):\n")

        print("🔢 Scalar values (per episode):")
        print(", ".join(sorted(scalar_keys)) or "  None")

        print("\n📈 List values (per step):")
        print(", ".join(sorted(list_keys)) or "  None")

        if empty_or_mixed:
            print("\n⚠️ Other / non-standard values:")
            print(", ".join(sorted(empty_or_mixed)))

        print("\nYou can use:\n  - keys='some_variable'         → histogram of a single variable\n"
              "  - keys=('var1', 'var2')        → scatter plot or hexbin of two variables\n"
              "  - keys=('some_list', 'time')   → plot value over time\n")
        return
    
    if isinstance(keys, str) and keys.lower() == "all":
        
        list_keys = list(filtered[0].keys())
        for key in list_keys:
            if isinstance(filtered[0].get(key, None), list):
                values = get_aggregated_list_values(key)
            else:
                values = get_scalar_values(key)

            if len(values) == 0:
                print(f"{key}: no data")
                continue

            plt.figure(figsize=figsize)
            sns.histplot(values, bins=30, kde=True, color='steelblue')
            plt.title(f"{test_name} - {key} ({n} ep.)")
            plt.xlabel(key)
            plt.ylabel("Frequency")
            plt.tight_layout()
            plt.show()
            
        return

    if isinstance(keys, str):
        list_keys = list(filtered[0].keys())
        if keys in list_keys:
            if isinstance(filtered[0].get(keys, None), list):
                values = get_aggregated_list_values(keys)
            else:
                values = get_scalar_values(keys)

            if len(values) == 0:
                print(f"{keys}: no data")
                return

            values = np.array(values)
            mean = np.mean(values)
            median = np.median(values)
            p25 = np.percentile(values, 25)
            p75 = np.percentile(values, 75)

            base_color = 'steelblue'
            line_color = '#FF7F50'
            
            plt.figure(figsize=figsize)
            sns.histplot(values, bins=30, kde=True, color=base_color)

            # Linee statistiche con stesso colore, tratteggi diversi
            plt.axvline(mean, color=line_color, linestyle='--', label=f'Mean: {mean:.2f}')
            plt.axvline(median, color=line_color, linestyle='-', label=f'Median: {median:.2f}')
            plt.axvline(p25, color=line_color, linestyle=':', label=f'25th: {p25:.2f}')
            plt.axvline(p75, color=line_color, linestyle=':', label=f'75th: {p75:.2f}')

            plt.title(f"{test_name} - {keys} ({n} ep.)")
            plt.xlabel(keys)
            plt.ylabel("Frequency")
            plt.legend()
            plt.tight_layout()
            plt.show()
            return
    
    # Variabile nel tempo (time series)
    if isinstance(keys, (tuple, list)) and len(keys) == 2 and keys[1] == 'time':
        key = keys[0]
        values_by_time = get_time_series(key)
        if not values_by_time:
            print(f"No time series data for key: {key}")
            return

        max_len = max(len(seq) for seq in values_by_time)
        padded = np.full((len(values_by_time), max_len), np.nan)
        for i, seq in enumerate(values_by_time):
            padded[i, :len(seq)] = seq

        time = np.arange(1, max_len + 1)
        mean = np.nanmean(padded, axis=0)
        std = np.nanstd(padded, axis=0)

        plt.figure(figsize=figsize)
        plt.plot(time, mean, label=f"Mean {key}", color='steelblue')
        plt.fill_between(time, mean - std, mean + std, alpha=0.2, color='steelblue', label="±1 std")
        if smooth_time:
            smoothed = gaussian_filter1d(mean, sigma=2)
            plt.plot(time, smoothed, linestyle='--', color='orangered', label="Smoothed")
        
        avg_len = np.mean([len(seq) for seq in values_by_time])
        plt.axvline(avg_len, color='gray', linestyle='--', linewidth=1.5, label=f"Avg episode end ({avg_len:.1f})")

        plt.xlabel("Time step")
        plt.ylabel(key)
        plt.title(f"{test_name} - {key} over time ({n} ep.)")

        if log_scale:
            plt.yscale('log')

        plt.legend()
        plt.tight_layout()
        plt.show()

        return

    # Confronto tra 2 variabili
    if isinstance(keys, (tuple, list)) and len(keys) == 2:
        key1, key2 = keys
        x_vals = []
        y_vals = []

        for ep in filtered:
            v1 = ep.get(key1, [])
            v2 = ep.get(key2, [])

            if isinstance(v1, list) and isinstance(v2, list):
                l = min(len(v1), len(v2))
                if l > 0:
                    x_vals.extend(v1[:l])
                    y_vals.extend(v2[:l])
            elif not isinstance(v1, list) and not isinstance(v2, list):
                x_vals.append(v1)
                y_vals.append(v2)
            else:
                continue

        if len(x_vals) == 0:
            print(f"No paired data for '{key1}' and '{key2}'")
            return

        if plot_type == 'hexbin':
            plt.figure(figsize=figsize)
            # Usare bins automatici ma con logaritmo dei counts
            hb = plt.hexbin(x_vals, y_vals, gridsize=40, cmap='Blues', bins='log')
            plt.colorbar(hb, label='Log Counts')
            plt.xlabel(key1)
            plt.ylabel(key2)
            plt.title(f"{test_name} - {key1} & {key2} ({n} ep.)")
            plt.tight_layout()
            plt.show()
            return
        else:
            # scatter con seaborn jointplot
            df = pd.DataFrame({key1: x_vals, key2: y_vals})
            g = sns.jointplot(data=df, x=key1, y=key2, kind='scatter', alpha=0.3, height=6)
            plt.suptitle(f"{test_name} - {key1} & {key2} ({n} ep.)", y=1.02)

            # Non mettiamo log in scatter perché potrebbe confondere (a meno che non vuoi aggiungere extra)
            plt.show()
            return

    print("Invalid input for keys parameter.")
    
def plot_stats(
    all_stats,
    plot_title='test_name',
    keys=None,
    
    filter_fn=lambda x: True,
    plot_type='hexbin',
    smooth_time=False,
    log_scale=False,
    figsize=(6, 4),
    ax=None,
    max_points=10000,
    random_seed=99,
    remove_outliers=True,  # <--- nuovo parametro
    outlier_quantile=0.01,   # <--- soglia di rimozione
    
    avg=False,
    v_line = None,
    h_line = None,
    norm=False
):
    
    filtered = [ep for ep in all_stats if filter_fn(ep)]
    n = len(filtered)
    if n == 0:
        print("No episodes matching filter.")
        return

    def get_scalar_values(key):
        return [ep[key] for ep in filtered if key in ep and not isinstance(ep[key], list)]

    def get_aggregated_list_values(key):
        values = []
        for ep in filtered:
            if key in ep and isinstance(ep[key], list):
                values.extend(ep[key])
        return values

    def get_time_series(key):
        values_by_time = []
        for ep in filtered:
            if key in ep and isinstance(ep[key], list):
                values_by_time.append(ep[key])
        return values_by_time

    def remove_outlier_bounds(x, y=None, q=0.01):
        if y is None:
            low, high = np.quantile(x, [q, 1 - q])
            return x[(x >= low) & (x <= high)]
        else:
            x = np.array(x)
            y = np.array(y)
            mask_x = (x >= np.quantile(x, q)) & (x <= np.quantile(x, 1 - q))
            mask_y = (y >= np.quantile(y, q)) & (y <= np.quantile(y, 1 - q))
            mask = mask_x & mask_y
            return x[mask], y[mask]


    if keys is None:
        scalar_keys = set()
        list_keys = set()
        other_keys = set()

        for ep in filtered:
            for k, v in ep.items():
                if isinstance(v, list):
                    list_keys.add(k)
                elif isinstance(v, (int, float, np.number)):
                    scalar_keys.add(k)
                else:
                    other_keys.add(k)

        print(f"\nAvailable keys (from {n} episodes):\n")

        print("• Scalar values (per episode, one number per ep):")
        print("  " + (", ".join(sorted(scalar_keys)) if scalar_keys else "None"))

        print("\n• List values (per step, one list per ep):")
        print("  " + (", ".join(sorted(list_keys)) if list_keys else "None"))

        if other_keys:
            print("\n• Other / non-standard values (ignored by default):")
            print("  " + ", ".join(sorted(other_keys)))

        print("\n=== Usage Examples ===")
        print("  # 1) Single variable → histogram")
        print("  keys = 'some_scalar'             → histogram of scalar values (per episode)")
        print("  keys = 'some_list'               → histogram of concatenated step values")

        print("\n  # 2) Scatter / Hexbin / KDE plots")
        print("  keys = ('var1', 'var2')           → compare two scalar or list variables")
        print("    e.g. keys=('score', 'duration')")

        print("\n  # 3) Time series (mean ± std across episodes)")
        print("  keys = ('some_list', 'time')      → plot variable over time steps")
        print("    e.g. keys=('reward_per_step', 'time')")

        print("\n  # 4) Ray aggregation (for ray_0 ... ray_16 lists)")
        print("  keys = ('var', 'ray_mean')        → aggregate over all rays per step")
        print("    Available aggregations: ray_mean, ray_min, ray_max, ray_std, ray_sum")

        print("\n=== Optional arguments ===")
        print("  plot_type:    'hexbin' (default), 'scatter', 'kde'")
        print("  smooth_time:  True → smooth time-series mean with Gaussian filter")
        print("  log_scale:    True → log scale for y-axis (and hexbin counts)")
        print("  max_points:   Limit points for scatter/KDE to avoid slowdown (default 10000)")
        print("  random_seed:  Seed for subsampling points")
        print("  remove_outliers: True → remove extreme values")
        print(f"  outlier_quantile: Quantile cutoff for outlier removal (default {outlier_quantile})")
        print("  figsize:      Tuple for figure size, e.g. (6,4)")

        print("\nTip: Scalar = one value per episode, List = one value per step.\n")
        return


    # === Istogramma ===
    if isinstance(keys, str):
        if isinstance(filtered[0].get(keys, None), list):
            values = get_aggregated_list_values(keys)
        else:
            values = get_scalar_values(keys)

        if len(values) == 0:
            print(f"{keys}: no data")
            return

        values = np.array(values)
        if remove_outliers:
            values = remove_outlier_bounds(values, q=outlier_quantile)

        mean = np.mean(values)
        median = np.median(values)
        p25 = np.percentile(values, 25)
        p75 = np.percentile(values, 75)

        base_color = 'steelblue'
        line_color = '#FF7F50'

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        sns.histplot(values, bins=30, kde=True, color=base_color, ax=ax)
        ax.axvline(mean, color=line_color, linestyle='--', label=f'Mean: {mean:.2f}')
        ax.axvline(median, color=line_color, linestyle='-', label=f'Median: {median:.2f}')
        ax.axvline(p25, color=line_color, linestyle=':', label=f'25th: {p25:.2f}')
        ax.axvline(p75, color=line_color, linestyle=':', label=f'75th: {p75:.2f}')

        ax.set_title(plot_title)
        ax.set_xlabel(keys)
        ax.set_ylabel("Frequency")
        ax.legend()
        return

    # === Time series ===
    if isinstance(keys, (tuple, list)) and len(keys) == 2 and keys[1] == 'time':
        key = keys[0]
        values_by_time = get_time_series(key)
        if not values_by_time:
            print(f"No time series data for key: {key}")
            return

        max_len = max(len(seq) for seq in values_by_time)
        padded = np.full((len(values_by_time), max_len), np.nan)
        for i, seq in enumerate(values_by_time):
            padded[i, :len(seq)] = seq

        time = np.arange(1, max_len + 1)
        mean = np.nanmean(padded, axis=0)
        std = np.nanstd(padded, axis=0)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        ax.plot(time, mean, label=f"Mean {key}", color='steelblue')
        ax.fill_between(time, mean - std, mean + std, alpha=0.2, color='steelblue', label="±1 std")
        if smooth_time:
            smoothed = gaussian_filter1d(mean, sigma=2)
            ax.plot(time, smoothed, linestyle='--', color='orangered', label="Smoothed")

        if avg:
            avg_len = np.mean([len(seq) for seq in values_by_time])
            ax.axvline(avg_len, color='gray', linestyle='--', linewidth=1.5, label=f"Avg episode end ({avg_len:.1f})")

        ax.set_xlabel("Time step")
        ax.set_ylabel(key)
        ax.set_title(plot_title)

        if log_scale:
            ax.set_yscale('log')

        ax.legend()
        return

    # === Scatter / Hexbin / Density ===
    if isinstance(keys, (tuple, list)) and len(keys) == 2:
        key1, key2 = keys
        x_vals = []
        y_vals = []

        for ep in filtered:
            v1 = ep.get(key1, [])

            if isinstance(key2, str) and key2.startswith('ray_'):
                aggregation = key2.rsplit('_', 1)[-1]
                ray_keys = [f'ray_{i}' for i in range(17)]
                values_lists = [ep.get(k, []) for k in ray_keys]
                values_lists = [v for v in values_lists if isinstance(v, list)]
                if not values_lists:
                    continue
                l = min(len(v) for v in values_lists)
                if l == 0:
                    continue
                stacked = np.stack([v[:l] for v in values_lists], axis=0)
                agg_map = {
                    'mean': np.mean,
                    'avg': np.mean,
                    'min': np.min,
                    'max': np.max,
                    'std': np.std,
                    'sum': np.sum,
                }
                v2 = agg_map.get(aggregation, np.mean)(stacked, axis=0).tolist()
            else:
                v2 = ep.get(key2, [])

            if isinstance(v1, list) and isinstance(v2, list):
                l = min(len(v1), len(v2))
                if l > 0:
                    x_vals.extend(v1[:l])
                    y_vals.extend(v2[:l])
            elif not isinstance(v1, list) and not isinstance(v2, list):
                x_vals.append(v1)
                y_vals.append(v2)

        if len(x_vals) == 0:
            print(f"No paired data for '{key1}' and '{key2}'")
            return

        x_vals = np.array(x_vals)
        y_vals = np.array(y_vals)
        if remove_outliers:
            x_vals, y_vals = remove_outlier_bounds(x_vals, y_vals, q=outlier_quantile)

        # Normalizzazione opzionale rispetto all'asse y
        if norm:
            # qui ridistribuiamo i dati rispetto a y
            # per ogni valore di y normalizziamo la densità degli x
            y_bins = np.unique(y_vals)
            new_x = []
            new_y = []
            for val in y_bins:
                mask = y_vals == val
                if mask.any():
                    # normalizza le frequenze di x dentro questo valore di y
                    counts, edges = np.histogram(x_vals[mask], bins=40, density=True)
                    centers = (edges[:-1] + edges[1:]) / 2
                    for c, cnt in zip(centers, counts):
                        new_x.extend([c] * int(cnt * 100))  # replica proporzionale
                        new_y.extend([val] * int(cnt * 100))
            if new_x and new_y:
                x_vals = np.array(new_x)
                y_vals = np.array(new_y)


        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        if plot_type == 'hexbin':
            im = ax.hexbin(x_vals, y_vals, gridsize=40, cmap='viridis',
                           bins='log' if log_scale else None)
            fig = ax.get_figure()
            fig.colorbar(im, ax=ax, label='Log Counts' if log_scale else 'Counts')

        elif plot_type == 'scatter':
            if len(x_vals) > max_points:
                np.random.seed(random_seed)
                idx = np.random.choice(len(x_vals), size=max_points, replace=False)
                x_vals = x_vals[idx]
                y_vals = y_vals[idx]
            
            from scipy.stats import gaussian_kde
            xy = np.vstack([x_vals, y_vals])
            z = gaussian_kde(xy)(xy)
            idx = z.argsort()
            x_vals, y_vals, z = x_vals[idx], y_vals[idx], z[idx]
            sc = ax.scatter(x_vals, y_vals, c=z, s=20, cmap='viridis')
            fig = ax.get_figure()
            fig.colorbar(sc, ax=ax, label='Density')

        elif plot_type == 'kde':
            
            if len(x_vals) > max_points:
                np.random.seed(random_seed)
                idx = np.random.choice(len(x_vals), size=max_points, replace=False)
                x_vals = x_vals[idx]
                y_vals = y_vals[idx]
                
            df = pd.DataFrame({key1: x_vals, key2: y_vals})
            sns.kdeplot(data=df, x=key1, y=key2, fill=True, cmap='viridis', ax=ax, cbar=True)
            # ax.scatter(x_vals, y_vals, alpha=0.3, s=20, c='black')

        else:
            print(f"Unsupported plot_type: {plot_type}")
            return

        ax.set_xlabel(key1)
        ax.set_ylabel(key2)
        ax.set_title(plot_title)
        
        if v_line is not None:
            ax.axvline(v_line, color="red", linestyle="--", linewidth=1)

        if h_line is not None:
            ax.axhline(h_line, color="red", linestyle="--", linewidth=1)

        return

    print("Invalid input for keys parameter.")

def plot_rays(episode, ray_keys=None, ray_angles=None, step=None, polar=True):
    """
    Visualizza i raggi LiDAR di un episodio in un determinato step (default: ultimo).

    Args:
        episode: dizionario contenente i raggi (ray_1, ..., ray_n) come liste nel tempo.
        ray_keys: lista delle chiavi dei raggi. Default: ray_1 ... ray_17.
        ray_angles: angoli relativi di ciascun raggio, in radianti.
        step: quale timestep visualizzare. Se None, prende l'ultimo disponibile.
        polar: se True, grafico polare; altrimenti cartesiano.
    """
    if ray_keys is None:
        ray_keys = [f"ray_{i}" for i in range(0, 17)]
    
    if ray_angles is None:
        # Angoli equispaziati tra -135° e 135°
        ray_angles = np.linspace(-3*np.pi/4, 3*np.pi/4, len(ray_keys))
    
    ray_values = []
    for k in ray_keys:
        if k not in episode or not isinstance(episode[k], list):
            print(episode[k])
            print(f"Missing or invalid ray data: {k}")
            return
        values = episode[k]
        if step is None:
            ray_values.append(values[-1])
        elif 0 <= step < len(values):
            ray_values.append(values[step])
        else:
            print(f"Invalid step index: {step}")
            return

    ray_values = np.array(ray_values)
    angles = np.array(ray_angles)

    if polar:
        plt.figure(figsize=(6, 6))
        ax = plt.subplot(111, polar=True)
        ax.plot(angles, ray_values, marker='o', color='steelblue', label='Ray distances')
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        # ax.set_title("LiDAR scan (polar)")
        ax.grid(True)
    else:
        x = ray_values * np.cos(angles)
        y = ray_values * np.sin(angles)
        plt.figure(figsize=(6, 6))
        plt.plot(x, y, 'o-', color='steelblue', label='Ray endpoints')
        plt.scatter(0, 0, color='black', marker='x', label='Robot')
        plt.axis('equal')
        # plt.title("LiDAR scan (cartesian)")
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.grid(True)
        plt.legend()

    plt.tight_layout()
    plt.show()

def load_multiple(save_dir, global_stats=None):
    if global_stats is None:
        global_stats = load_global_stats(save_dir)
        
    ret = {}
    for name in global_stats['test_name']:
        ret[name] = load_stats(name, save_dir)
    return ret

def plot_multiple(stats_list, figsize=(6,3), cols=1, **args):
    n = math.ceil(len(stats_list) / cols)
    print()
    # Creo una figura con n subplot verticali (puoi cambiare layout se vuoi)
    fig, axs = plt.subplots(n, cols, figsize=(figsize[0]*cols, figsize[1]*n), squeeze=False)
    axs = axs.flatten()  # per sicurezza se n=1

    for i, key in enumerate(stats_list):
        plot_stats(stats_list[key]['stats'], key.rsplit('_', 1)[0], ax=axs[i], **args)

    plt.tight_layout()
    plt.show()

def plot_all(stats, figsize=(6,3), cols=2, **args):
    
    all_keys = scalar_keys + list_keys
    n = len(all_keys) // cols
    
    # Creo una figura con n subplot verticali (puoi cambiare layout se vuoi)
    fig, axs = plt.subplots(n, cols, figsize=(figsize[0]*cols, figsize[1]*n), squeeze=False)
    axs = axs.flatten()  # per sicurezza se n=1

    for i, key in enumerate(all_keys):
        plot_stats(stats['stats'], 'name', key, ax=axs[i], **args)

    plt.tight_layout()
    plt.show()