# === Standard library ===
import argparse
import json
import os
import random
import time
import uuid
from distutils.util import strtobool
from functools import reduce
from typing import Tuple
from pprint import pprint

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
import yaml
import argparse
import os

def load_models(actor, qf_ensemble, qf_ensemble_target, save_path, suffix=''):
    actor.load_state_dict(
        torch.load(os.path.join(save_path, f'actor{suffix}.pth'))
    )

    for i, qf in enumerate(qf_ensemble):
        qf.load_state_dict(
            torch.load(os.path.join(save_path, f'qf{i+1}{suffix}.pth'))
        )

    for i, qft in enumerate(qf_ensemble_target):
        qft.load_state_dict(
            torch.load(os.path.join(save_path, f'qf{i+1}_target{suffix}.pth'))
        )
