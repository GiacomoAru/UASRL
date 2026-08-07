# === Standard library ===
import argparse
import json
import math
import os
import random
import time
import uuid
from distutils.util import strtobool
from functools import reduce
from typing import Tuple
from pprint import pprint
import csv

# === Third-party libraries ===
import numpy as np
import yaml

# === PyTorch ===
import torch
import torch.nn as nn
import torch.nn.functional as F

# === Unity ML-Agents ===
from mlagents_envs.side_channel.side_channel import (
    SideChannel,
    IncomingMessage,
    OutgoingMessage,
)

import osqp
import scipy.sparse as sp
import contextlib
import sys

def load_models(actor=None,
                qf_ensemble=None,
                qf_ensemble_target=None,
                save_path=None,
                suffix='',
                DEVICE='cpu'):

    if save_path is None:
        raise ValueError("save_path must be provided")
    
    # ===== Actor =====

    # Definiamo i possibili nomi dei file in ordine di preferenza
    files_to_try = [f'actor{suffix}.pth', f'agent{suffix}.pth']
    loaded = False

    if actor is not None:
        for file_name in files_to_try:
            full_path = os.path.join(save_path, file_name)
            
            if os.path.exists(full_path):
                print(f"Loading weights from {full_path}...")
                actor.load_state_dict(
                    torch.load(full_path, map_location=DEVICE)
                )
                loaded = True
                break  # Interrompe il ciclo una volta trovato il file
                
        if not loaded:
            attempted_paths = [os.path.join(save_path, name) for name in files_to_try]
            raise FileNotFoundError(
                "No actor weights found. Tried: " + ", ".join(attempted_paths)
            )

    # ===== Q ensemble =====
    if qf_ensemble is not None:
        for i, qf in enumerate(qf_ensemble):
            qf.load_state_dict(
                torch.load(
                    os.path.join(save_path, f'qf{i+1}{suffix}.pth'),
                    map_location=DEVICE
                )
            )

    # ===== Q target ensemble =====
    if qf_ensemble_target is not None:
        for i, qf in enumerate(qf_ensemble_target):
            qf.load_state_dict(
                torch.load(
                    os.path.join(save_path, f'qf{i+1}_target{suffix}.pth'),
                    map_location=DEVICE
                )
            )


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
    sensor_offset: float = 0.10,
    max_movement_speed: float | None = None,
    max_turn_speed: float | None = None,
    debug: bool = False
) -> tuple[float, float]:
    """
    Apply a Control Barrier Function (CBF) quadratic program filter
    to enforce safety constraints on velocity commands.

    Given nominal linear and angular velocities, along with LIDAR
    ray distances and angles, this function formulates and solves
    a quadratic program (QP) that minimally modifies the commands
    to ensure obstacles remain outside a safe distance.

    Matches the offset-control-point barrier in the rebuttal: rays are taken
    relative to a control point `ell` ahead of the robot center (coincident
    with the range-sensor origin), h_i = x_i^2 + y_i^2 - d_safe^2, and
    d/dt h_i = -2*x_i*v - 2*ell*y_i*omega, which is what makes omega enter
    the barrier at first order (ell=0 recovers the center-distance barrier,
    where steering only enters at higher order).

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
    sensor_offset : float, optional
        Longitudinal offset `ell` (meters) of the control point / ray-sensor
        origin ahead of the robot center (default 0.10, the simulation value
        used in the rebuttal; the physical platform uses its own extrinsics).
    debug : bool, optional
        If True, OSQP solver messages are shown and debug information is printed (default is False).

    Returns
    -------
    tuple of float
        A tuple (v_safe, omega_safe) representing the filtered forward
        and angular velocities that satisfy the CBF constraints.
    """

    ray_distances = np.asarray(ray_distances, dtype=np.float64).reshape(-1)
    ray_angles = np.asarray(ray_angles, dtype=np.float64).reshape(-1)

    if ray_distances.shape != ray_angles.shape:
        raise ValueError(
            "ray_distances and ray_angles must contain the same number of elements"
        )
    if not np.all(np.isfinite(ray_distances)) or not np.all(np.isfinite(ray_angles)):
        raise ValueError("ray distances and angles must be finite")
    if d_safe <= 0 or alpha <= 0 or d_safe_threshold_mult <= 0:
        raise ValueError("d_safe, alpha and d_safe_threshold_mult must be positive")
    if sensor_offset < 0:
        raise ValueError("sensor_offset must be nonnegative")
    if max_movement_speed is not None and max_movement_speed <= 0:
        raise ValueError("max_movement_speed must be positive")
    if max_turn_speed is not None and max_turn_speed <= 0:
        raise ValueError("max_turn_speed must be positive")

    max_v = np.inf if max_movement_speed is None else float(max_movement_speed)
    max_omega = np.inf if max_turn_speed is None else float(max_turn_speed)
    nominal_u = np.array([
        np.clip(float(v_cmd), 0.0, max_v),
        np.clip(float(omega_cmd), -max_omega, max_omega),
    ])

    # Rays are given in the sensor/control-point frame already (the control
    # point c is ell ahead of the robot center and coincident with the
    # range-sensor origin), so ray Cartesian coordinates are (x_i, y_i) as-is.
    obstacles = np.column_stack((
        ray_distances * np.cos(ray_angles),
        ray_distances * np.sin(ray_angles)
    ))

    max_considered_distance = d_safe * d_safe_threshold_mult
    A_list, b_list = [], []

    for x_i, y_i in obstacles:
        dist = math.hypot(x_i, y_i)

        # Skip obstacles too far to be relevant
        if dist > max_considered_distance:
            continue

        # Barrier h_i = x_i^2 + y_i^2 - d_safe^2. For the control point
        # offset ell ahead of the robot, d/dt h_i = -2*x_i*v - 2*ell*y_i*omega,
        # so the standard condition d/dt h_i + alpha*h_i >= 0 becomes:
        h = x_i**2 + y_i**2 - d_safe**2
        a_i = np.array([2.0 * x_i, 2.0 * sensor_offset * y_i])
        b_i = alpha * h

        A_list.append(a_i)
        b_list.append(b_i)

    # If no relevant constraints, return original commands
    if not A_list:
        return float(nominal_u[0]), float(nominal_u[1])

    # Build QP matrices
    A = sp.csc_matrix(np.vstack(A_list))
    l = -np.inf * np.ones_like(b_list)
    u = np.array(b_list)

    # Physical limits are constraints on the final controls, not on a
    # correction delta. The previous implementation constrained delta_v >= 0,
    # which made every required deceleration infeasible.
    A_controls = sp.csc_matrix(np.eye(2))
    l_controls = np.array([0.0, -max_omega])
    u_controls = np.array([max_v, max_omega])

    # Stack together
    A_full = sp.vstack([A, A_controls], format="csc")
    l_full = np.hstack([l, l_controls])
    u_full = np.hstack([u, u_controls])

    # Quadratic cost: minimize ||D^-1(u - u_nominal)||^2, D=diag(v_max, omega_max),
    # so deviations on the two actuators are weighted by their own limits
    # rather than compared in raw m/s vs rad/s units.
    v_scale = max_v if np.isfinite(max_v) else 1.0
    omega_scale = max_omega if np.isfinite(max_omega) else 1.0
    d_inv_sq = np.array([1.0 / v_scale**2, 1.0 / omega_scale**2])
    P = sp.csc_matrix(np.diag(2.0 * d_inv_sq))
    q = -2.0 * d_inv_sq * nominal_u

    # Solve QP
    prob = osqp.OSQP()
    prob.setup(P=P, q=q, A=A_full, l=l_full, u=u_full, verbose=debug, polishing=True)

    if debug:
        res = prob.solve()
    else:
        with suppress_osqp_output():
            res = prob.solve()

    if not res.info.status.lower().startswith('solved'):
        if debug:
            print("OSQP failed:", res.info.status)
        return 0.0, float(nominal_u[1])

    v_safe, omega_safe = res.x
    return float(v_safe), float(omega_safe)

def CBF_from_obs(ray_obs, action,

                 ray_original_lenght,
                 max_movement_speed,
                 max_turn_speed,

                 d_safe, alpha, d_safe_mul,

                 precomputed_angles_rad,
                 sensor_offset=0.10):
 
    if ray_original_lenght <= 0 or max_movement_speed <= 0 or max_turn_speed <= 0:
        raise ValueError("ray length and movement limits must be positive")

    ray_obs = np.asarray(ray_obs, dtype=np.float64).reshape(-1)
    action = np.asarray(action, dtype=np.float64).reshape(-1)
    if action.size != 2:
        raise ValueError("action must contain forward and angular commands")

    # ML-Agents ray hit fractions are normalized in [0, 1].
    ray_distances = np.clip(ray_obs, 0.0, 1.0) * ray_original_lenght

    # Policy network outputs are normalized velocities (not accelerations)
    nn_v_front = max(0.0, action[0] * max_movement_speed) # the controller chop away negative values
    nn_v_ang = np.radians(action[1] * max_turn_speed)
    max_turn_speed_rad = np.radians(max_turn_speed)

    # Apply Control Barrier Function via QP filter
    v_safe, omega_safe = cbf_velocity_filter_qp(
        nn_v_front, nn_v_ang,
        
        ray_distances,
        precomputed_angles_rad,
        
        d_safe=d_safe,
        alpha=alpha,
        d_safe_threshold_mult=d_safe_mul,
        sensor_offset=sensor_offset,
        max_movement_speed=max_movement_speed,
        max_turn_speed=max_turn_speed_rad,
        debug=False
    )

    # Normalize outputs back to [-1, 1] (compatible with policy space)
    v_safe_norm = np.clip(v_safe / max_movement_speed, 0.0, 1.0)
    omega_safe_norm = np.clip(np.degrees(omega_safe) / max_turn_speed, -1.0, 1.0)

    return np.array([v_safe_norm, omega_safe_norm])


def ecbf_terms_unicycle(obs_xy, d_safe, k1, k2, v_nom):
    """
    Calcola i termini del vincolo lineare HOCBF per un modello uniciclo.
    Vincolo forma: A_v * v + A_omega * omega <= b
    """
    x_o, y_o = obs_xy
    
    dist2 = x_o**2 + y_o**2
    h = dist2 - d_safe**2
    
    coeff_v = -(2.0 * v_nom - 2.0 * k1 * x_o)
    coeff_w = -(-2.0 * v_nom * y_o)
    b_val   = k2 * h
    
    return coeff_v, coeff_w, b_val

def ecbf_velocity_filter_qp(
    v_cmd: float,
    omega_cmd: float,
    ray_distances: np.ndarray,
    ray_angles: np.ndarray,
    d_safe: float = 0.3,
    d_safe_mul: float = 3.0,
    k1: float = 2.0,
    k2: float = 2.0,
    max_v: float = 1.0,
    debug: bool = False
) -> tuple[float, float]:
    
    # 1. Converti coordinate polari in Cartesiane (frame robot)
    # Assicuriamoci che siano array numpy appiattiti
    dists = np.array(ray_distances).flatten()
    angles = np.array(ray_angles).flatten()
    
    obs_x = dists * np.cos(angles)
    obs_y = dists * np.sin(angles)
    
    obstacles = np.column_stack((obs_x, obs_y))

    A_list, b_list = [], []

    # Se la velocità nominale è troppo bassa, la linearizzazione fallisce (divisione per zero o gradienti nulli)
    # Usiamo un piccolo epsilon o la v_cmd stessa
    v_linearization = max(v_cmd, 0.01)

    for obs in obstacles:
        # Filtra ostacoli troppo lontani per risparmiare calcoli (opzionale ma consigliato)
        if np.linalg.norm(obs) > d_safe * d_safe_mul: 
            continue

        Av, Aw, b_i = ecbf_terms_unicycle(obs, d_safe, k1, k2, v_linearization)

        # A_i @ [v, omega] <= b_i
        A_list.append([Av, Aw])
        b_list.append(b_i)

    # --- Costruzione QP ---
    # Variabili decisionali: x = [v, omega] (Velocità Assolute)
    # Obiettivo: minimizzare (v - v_cmd)^2 + (omega - w_cmd)^2
    # Espanso: v^2 - 2*v*v_cmd + omega^2 - 2*omega*w_cmd
    # Forma standard OSQP: 1/2 x'Px + q'x
    
    # P = Matrice diagonale 2x2 (moltiplicata per 2 per cancellare 1/2 del solver)
    P = sp.csc_matrix(np.eye(2) * 2.0)
    
    # q = vettore lineare [-2*v_cmd, -2*w_cmd]
    q = np.array([-2.0 * v_cmd, -2.0 * omega_cmd])

    # --- Vincoli ---
    # 1. Vincoli CBF (dinamici)
    if A_list:
        A_cbf = np.array(A_list)
        u_cbf = np.array(b_list)
        l_cbf = -np.inf * np.ones_like(u_cbf) # Limite inferiore -inf (è disuguaglianza <=)
    else:
        # Se non ci sono ostacoli, ritorniamo i comandi originali
        return v_cmd, omega_cmd

    # 2. Vincolo: v >= 0 (niente retromarcia se non voluta)
    # 1*v + 0*omega >= 0  ->  -1*v <= 0 se usiamo <=
    # Ma OSQP usa l <= Ax <= u.
    
    # Aggiungiamo vincoli sui limiti fisici delle variabili
    # 0 <= v <= max_v
    # -inf <= omega <= inf (o limiti fisici se ne hai)
    
    # Costruiamo la matrice A completa: [A_cbf; I]
    # Dove I è la matrice identità per vincolare direttamente le variabili
    A_vars = np.eye(2)
    l_vars = np.array([0.0, -np.inf])       # v >= 0
    u_vars = np.array([max_v, np.inf])      # v <= max_v
    
    # Stack finale
    A_full = sp.vstack([sp.csc_matrix(A_cbf), sp.csc_matrix(A_vars)])
    l_full = np.hstack([l_cbf, l_vars])
    u_full = np.hstack([u_cbf, u_vars])

    # Risoluzione
    prob = osqp.OSQP()
    prob.setup(P=P, q=q, A=A_full, l=l_full, u=u_full, verbose=debug, polishing=True)
    
    res = prob.solve()

    if res.info.status != 'solved':
        if debug: print(f"QP Fallito: {res.info.status}")
        # Fallback: frena
        return 0.0, omega_cmd

    v_safe, omega_safe = res.x
    return v_safe, omega_safe

def ECBF_from_obs(ray_obs, action,
                  ray_original_lenght, 
                  max_movement_speed,
                  max_turn_speed,
                  d_safe, alpha, d_safe_mul,
                  precomputed_angles_rad):
    """
    Wrapper per adattare input/output normalizzati al filtro QP fisico.
    """
        
    # Converti osservazioni normalizzate in distanze fisiche
    ray_distances = np.array([x * ray_original_lenght for x in ray_obs])

    # Denormalizza l'azione del policy network
    nn_v_front = action[0] * max_movement_speed
    nn_v_ang = action[1] * max_turn_speed # Rimuovi np.radians se max_turn_speed è già in rad/s

    # Mappatura parametri:
    # alpha viene usato come guadagno per k1 e k2
    k1_gain = alpha
    k2_gain = alpha
    
    # Applica il filtro QP
    v_safe, omega_safe = ecbf_velocity_filter_qp(
        v_cmd=nn_v_front, 
        omega_cmd=nn_v_ang,
        ray_distances=ray_distances,
        ray_angles=precomputed_angles_rad,
        d_safe=d_safe,
        k1=k1_gain,
        k2=k2_gain,
        max_v=max_movement_speed,
        debug=False
    )

    # Normalizza di nuovo per ritornare nello spazio azioni [-1, 1]
    v_safe_norm = np.clip(v_safe / max_movement_speed, -1.0, 1.0)
    omega_safe_norm = np.clip(omega_safe / max_turn_speed, -1.0, 1.0)

    return np.array([v_safe_norm, omega_safe_norm])



def save_stats_to_csv(info_dict, stats_dict, filepath):
    """
    Salva media e deviazione standard contenute in stats_dict in un file CSV.
    Structure: chiave -> chiave_mean, chiave_std
    """
    
    # 1. Costruiamo la riga appiattendo i dati
    row = {}
    for key, stats_obj in stats_dict.items():
        # stats_obj è un'istanza della tua nuova classe RunningStats
        # Creiamo due colonne per ogni statistica
        row[f"{key}_mean"] = stats_obj.mean
        row[f"{key}_std"] = stats_obj.std_dev
    for key in info_dict:
        row[key] = info_dict[key]
        
    file_exists = os.path.isfile(filepath)

    with open(filepath, mode='a', newline='') as f:
        # fieldnames viene derivato automaticamente dalle chiavi che abbiamo appena creato
        writer = csv.DictWriter(f, fieldnames=row.keys())
        
        # Se il file è nuovo, scrivi l'header (che conterrà i nomi con _mean e _std)
        if not file_exists:
            writer.writeheader()
        
        # Scrivi la riga
        writer.writerow(row)
