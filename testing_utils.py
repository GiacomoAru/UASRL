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


import os
import torch

def load_models(actor=None,
                qf_ensemble=None,
                qf_ensemble_target=None,
                save_path=None,
                suffix='',
                DEVICE='cpu'):

    if save_path is None:
        raise ValueError("save_path must be provided")
    
    # ===== Actor =====
    if actor is not None:
        actor.load_state_dict(
            torch.load(
                os.path.join(save_path, f'actor{suffix}.pth'),
                map_location=DEVICE
            )
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