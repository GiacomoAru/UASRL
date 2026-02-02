import contextlib
import math
import os
import pickle
import re
import sys
import time
from typing import Tuple

import numpy as np
import osqp
import pandas as pd
import scipy.sparse as sp


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

    # Quadratic cost: minimize deviation from nominal command
    P = sp.csc_matrix(np.eye(2) * 2.0)
    q = np.zeros(2)

    # Solve QP
    prob = osqp.OSQP()
    prob.setup(P=P, q=q, A=A, l=l, u=u, verbose=debug, polish=True)

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