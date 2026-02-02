# === Standard library ===
import argparse
import json
import math
import os
import random
import uuid
from distutils.util import strtobool
from functools import reduce
from typing import Tuple

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

# === Optimization / Control ===
import osqp
import scipy.sparse as sp
from scipy.optimize import minimize


####################################################################################################
####################################################################################################

#   ╔══════════════════════╗
#   ║   Training Classes   ║
#   ╚══════════════════════╝

class DenseSoftQNetwork(nn.Module):
    # raycast observation is a 2d tensor of shape (stack, ray)
    # state observation is a 1d tensor containing the remaining state information
    # action space is a 1d tensor containing the action information
    
    def __init__(self, 
                 raycast_observation_shape: Tuple[int, int], state_observation_size: int, 
                 action_size: int,
                 dense_layer: list[int]):
        
        super().__init__()
        
        self.input_dim = reduce(
                        lambda x, y: x * y, raycast_observation_shape
                    ) + state_observation_size + action_size
        self.output_dim = 1
        
        dense_layer = [self.input_dim] + dense_layer + [self.output_dim]
        self.layers =  nn.ModuleList()
        for i, layer in enumerate(dense_layer[:-1]):
            self.layers.append(nn.Linear(layer, dense_layer[i + 1]))
        

    def forward(self, raycast_obs, state_obs, action):
        
        x = torch.cat([raycast_obs.flatten(start_dim=1), state_obs, action], 1)
        
        # for each layer, first the layer and then the activation function
        for layer in self.layers[:-1]:
            x = layer(x)
            x = F.relu(x)
            
        # the last layer does not have an activation function
        x = self.layers[-1](x)
        
        return x

LOG_STD_MAX = 3
LOG_STD_MIN = -6
class DenseActor(nn.Module):
    
    def __init__(self, 
                 raycast_observation_shape: Tuple[int, int], state_observation_size: int, 
                 action_size: int, action_space_min_value: int, action_space_max_value: int,
                 dense_layer: list[int]):
        
        super().__init__()
        
        self.input_dim = reduce(
                        lambda x, y: x * y, raycast_observation_shape
                    ) + state_observation_size
        
        self.output_dim = action_size
        
        
        dense_layer = [self.input_dim] + dense_layer
        self.layers = nn.ModuleList()
        for i, layer in enumerate(dense_layer[:-1]):
            self.layers.append(nn.Linear(layer, dense_layer[i + 1]))
        self.mean_layer = nn.Linear(dense_layer[-1], self.output_dim)
        self.logstd_layer = nn.Linear(dense_layer[-1], self.output_dim)   
                
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((action_space_max_value - action_space_min_value) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((action_space_max_value + action_space_min_value) / 2.0, dtype=torch.float32)
        )

    def forward(self, raycast_obs, state_obs):
        
        x = torch.cat([raycast_obs.flatten(start_dim=1), state_obs], 1)
        
        # for each layer, first the layer and then the activation function
        for layer in self.layers:
            x = layer(x)
            x = F.relu(x)

        mean = self.mean_layer(x)
        log_std = self.logstd_layer(x)
        
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, raycast_obs, state_obs, variance_scale=1.0):
        mean, log_std = self(raycast_obs, state_obs)
        
        if variance_scale > 0:
            std = log_std.exp()
            std = std * variance_scale  # ← Riduci la deviazione standard con il coefficiente v
            
            normal = torch.distributions.Normal(mean, std)
            
            x_t = normal.rsample()
            y_t = torch.tanh(x_t)
            action = y_t * self.action_scale + self.action_bias
            
            log_prob = normal.log_prob(x_t)
            
            # Correzione del log_prob per la trasformazione tanh
            log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
            log_prob = log_prob.sum(1, keepdim=True)

            mean_action = torch.tanh(mean) * self.action_scale + self.action_bias
            
            return action, log_prob, mean_action, log_std
        
        else:
            
            return mean, 1.0, mean, log_std


# Create the StringLogChannel class
class CustomChannel(SideChannel):

    settings = {}
    msg_queue = []
    settings_token = "SETTINGS"
    data_token = "DATA"
    
    def __init__(self) -> None:
        super().__init__(uuid.UUID("621f0a70-4f87-11ea-a6bf-784f4387d1f7"))

    def on_message_received(self, msg: IncomingMessage) -> None:
        
        received = msg.read_string()
        
        if received.startswith(self.settings_token):
            received = json.loads(received.removeprefix(self.settings_token))
            el = received['obj_name']
            del received['obj_name']
            
            self.settings[el] = received
        elif received.startswith(self.data_token):
            received = json.loads(received.removeprefix(self.data_token))
            self.msg_queue.append(received)
        else:
            print("TOKEN MESSAGGIO NON RICONOSCIUTO")
            
class DebugSideChannel(SideChannel):
    """
    Side channel per inviare/ricevere messaggi di debug
    """

    def __init__(self):
        super().__init__(uuid.UUID("abcdefab-1234-5678-9abc-def012345678"))

    def send_agent_action_debug(self, forward_speed: float, angular_speed: float, cbf: bool = False, uf: bool = False) -> None:
        data = {
            "forward": float(forward_speed),
            "rotate": float(angular_speed),
            "cbf_activation": bool(cbf),
            "uf_activation": bool(uf)
        }

        json_str = json.dumps(data)
        msg = OutgoingMessage()
        msg.write_string(json_str)
        self.queue_message_to_send(msg)
    
    def on_message_received(self, msg: IncomingMessage) -> None:
        print('MESSAGGIO INATTESO')


# VAE

class StableVAE(nn.Module):
    def __init__(self, input_dim, 
                 encoder_layers=[256], 
                 latent_dim=32, 
                 activation='relu', 
                 dropout=None, 
                 stable_variant=True):
        super().__init__()

        # Supported activation functions
        activations = {
            'relu': F.relu,
            'gelu': F.gelu,
            'lrelu': F.leaky_relu,
            'elu': F.elu,
        }

        # Raise an error if the selected activation is not supported
        if activation not in activations:
            raise ValueError(f"Activation '{activation}' not supported. Choose from {list(activations.keys())}")
        
        self.activation_fn = activations[activation]
        self.dropout = dropout
        self.stable_variant = stable_variant

        # Encoder: sequence of linear layers
        self.encoder = nn.ModuleList()
        dims = [input_dim] + encoder_layers
        for i in range(len(dims) - 1):
            self.encoder.append(nn.Linear(dims[i], dims[i + 1]))
        
        # Separate output layers for mean and log-variance of latent space
        self.encoder_mean = nn.Linear(dims[-1], latent_dim)
        self.encoder_logvar = nn.Linear(dims[-1], latent_dim)

        # Decoder: symmetric architecture with reversed encoder layers
        dims = [latent_dim] + list(reversed(encoder_layers)) + [input_dim]
        self.decoder = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.decoder.append(nn.Linear(dims[i], dims[i + 1]))

        # Dropout layers (optional)
        if dropout is not None:
            self.encoder_dropout = nn.Dropout(dropout)
            self.decoder_dropout = nn.Dropout(dropout)

    def encode(self, x):
        # Pass through encoder layers with activation
        for i, layer in enumerate(self.encoder):
            x = self.activation_fn(layer(x))
            # Apply dropout only after the first encoder layer
            if self.dropout is not None and i == 0:
                x = self.encoder_dropout(x)

        mean = self.encoder_mean(x)
        logvar = self.encoder_logvar(x)

        # Optional variance stabilization
        if self.stable_variant:
            logvar = torch.tanh(logvar)
            logvar = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (logvar + 1)
        
        return mean, logvar

    def reparameterize(self, mean, logvar):
        # Standard reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z):
        # Pass through decoder layers with activation
        for layer in self.decoder[:-1]:
            z = self.activation_fn(layer(z))

        # Apply dropout only at the last decoder layer
        if self.dropout is not None:
            z = self.decoder_dropout(z)

        return self.decoder[-1](z)

    def forward(self, x):
        # Full forward pass: encode -> reparameterize -> decode
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        recon = self.decode(z)
        return recon, mean, logvar
  

####################################################################################################
####################################################################################################

#   ╔════════════════════╗
#   ║   Training Utils   ║
#   ╚════════════════════╝

def get_initial_action(agent_id, previous_movements, alpha=0.8, noise_std_init=0.3, noise_decay=0.99, min_noise_std=0.05, step=0):
    """
    Genera un'azione di movimento con esplorazione migliorata.

    Params:
    - agent_id: identificatore agente
    - previous_movements: dict {agent_id: (speed, steer)}
    - alpha: peso per smoothing tra azione precedente e nuova
    - noise_std_init: deviazione standard iniziale del rumore gaussiano
    - noise_decay: fattore di decadenza del rumore per step temporali
    - min_noise_std: rumore minimo
    - step: numero di passo temporale (per decadenza rumore)

    Return:
    - np.array [speed, steer]
    """

    # Calcola rumore decaduto
    noise_std = max(noise_std_init * (noise_decay ** step), min_noise_std)

    # Se non abbiamo precedente, inizializziamo vicino a zero (velocità bassa, sterzo neutro)
    if agent_id not in previous_movements:
        base_speed = np.random.uniform(0.0, 0.5)
        base_steer = np.random.uniform(-0.2, 0.2)
    else:
        base_speed, base_steer = previous_movements[agent_id]

    # Aggiungi rumore gaussiano per esplorazione
    noisy_speed = base_speed + np.random.normal(0, noise_std)
    noisy_steer = base_steer + np.random.normal(0, noise_std)

    # Smoothing verso rumore + base
    speed = alpha * base_speed + (1 - alpha) * noisy_speed
    steer = alpha * base_steer + (1 - alpha) * noisy_steer

    # Clipping per sicurezza
    speed = np.clip(speed, 0.0, 1.0)     # solo avanti (modifica se vuoi retromarcia)
    steer = np.clip(steer, -1.0, 1.0)

    # Aggiorna stato precedente
    previous_movements[agent_id] = (speed, steer)

    return np.array([speed, steer])
    
####################################################################################################
####################################################################################################

#   ╔═══════════════════╗
#   ║   Parsing Utils   ║
#   ╚═══════════════════╝
    
def _load_config(config_path):
    """
    Load a YAML configuration file and return its contents as a dictionary.

    Args:
        config_path (str): The file path to the YAML configuration file.

    Returns:
        dict: The contents of the YAML file as a dictionary. If the file is empty,
              an empty dictionary is returned.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config or {}

def parse_args_from_file(config_path):

    config = _load_config(config_path)

    parser = argparse.ArgumentParser()

    # Utility to get values from config with fallback to default
    def get(key, default):
        return config.get(key, default)
    
    parser.add_argument("--exp-name", type=str, default=get("exp-name", os.path.basename(__file__).rstrip(".py")))
    parser.add_argument("--env-id", type=str, default=get("env-id", "Environment-ID"))
    
    parser.add_argument('--q-ensemble-n', type=int, default=get("q-ensemble-n", 2))
    parser.add_argument('--bootstrap-batch-proportion', type=float, default=get("bootstrap-batch-proportion", 0.8))
    
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(str(x))), default=get("torch-deterministic", True))
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(str(x))), default=get("cuda", True))
    
    parser.add_argument("--loss-log-interval", type=int, default=get("loss-log-interval", 100))
    parser.add_argument("--metrics-log-interval", type=int, default=get("metrics-log-interval", 300))
    parser.add_argument("--metrics-smoothing", type=int, default=get("metrics-smoothing", 0.95))
    
    parser.add_argument("--q-network-layers", type=int, nargs='+', default=get("q-network-layers", [64, 64]),
                        help="Hidden layers for Q network as list of ints")
    parser.add_argument("--actor-network-layers", type=int, nargs='+', default=get("actor-network-layers", [64, 64]),
                        help="Hidden layers for Actor network as list of ints")
    
    parser.add_argument("--total-timesteps", type=int, default=get("total-timesteps", 1000000))
    parser.add_argument("--buffer-size", type=int, default=get("buffer-size", int(1e6)))
    parser.add_argument("--update-per-step", type=int, default=get("update-per-step", 1))
    
    parser.add_argument("--gamma", type=float, default=get("gamma", 0.99))
    parser.add_argument("--tau", type=float, default=get("tau", 0.005))
    parser.add_argument("--batch-size", type=int, default=get("batch-size", 256))
    parser.add_argument("--learning-starts", type=int, default=get("learning-starts", int(5e3)))
    parser.add_argument("--policy-lr", type=float, default=get("policy-lr", 3e-4))
    parser.add_argument("--q-lr", type=float, default=get("q-lr", 1e-3))
    
    parser.add_argument("--policy-frequency", type=int, default=get("policy-frequency", 2))
    parser.add_argument("--target-network-frequency", type=int, default=get("target-network-frequency", 1))
    parser.add_argument("--noise-clip", type=float, default=get("noise-clip", 0.5))
    
    parser.add_argument("--alpha", type=float, default=get("alpha", 0.2))
    parser.add_argument("--autotune", type=lambda x: bool(strtobool(str(x))), default=get("autotune", True))
    parser.add_argument("--alpha-lr", type=float, default=get("alpha-lr", 1e-4))
    
    
    return parser.parse_args([])

####################################################################################################
####################################################################################################

#   ╔═══════════════════╗
#   ║   Testing Utils   ║
#   ╚═══════════════════╝

        
####################################################################################################
####################################################################################################

#   ╔═══════════════════════╗
#   ║   Data Manipolation   ║
#   ╚═══════════════════════╝

def organize_observations_for_conv(flat_observations, num_stacks, num_rays_per_dir, num_tags, 
                                   remove_last = False):

    features_per_ray = num_tags + 2
    total_rays = num_rays_per_dir * 2 + 1 # Il primo e l'ultimo sono uguali se i gradi sono 180
    
    # Reshape (stack, ray, feature)
    reshaped = np.array(flat_observations) 
    reshaped = reshaped[[i for i in range(features_per_ray - 1, num_stacks*total_rays*features_per_ray, features_per_ray)]]
    reshaped = reshaped.reshape((num_stacks, total_rays))
    if remove_last:
        reshaped = reshaped[:,:-1] # remove the last ray
    
    return reshaped

def collect_data_after_step(environment, 
                            env_info):
    
    RAY_STACK = env_info.settings['ray_sensor_settings']['observation_stacks']
    RAY_PER_DIRECTION = env_info.settings['ray_sensor_settings']['rays_per_direction']
    DELETE_LAST_RAY = env_info.settings['ray_sensor_settings']['ignore_last_ray']
    BEHAVOIR_NAME = env_info.settings['behavior_parameters_settings']['behavior_name']
                
    decision_steps, terminal_steps = environment.get_steps(BEHAVOIR_NAME)
    
    obs = {}
    
    for id in decision_steps:
        decision_step = decision_steps[id]
        # ray_obs, state_obs, reward, action, done
        obs[id] = [organize_observations_for_conv(decision_step.obs[0], RAY_STACK, RAY_PER_DIRECTION, 1, DELETE_LAST_RAY),
                   decision_step.obs[1],
                   decision_step.reward,
                   None,
                   0]
        
    for id in terminal_steps:
        terminal_step = terminal_steps[id]
        # ray_obs, state_obs, reward, action, done
        obs[id] = [organize_observations_for_conv(terminal_step.obs[0], RAY_STACK, RAY_PER_DIRECTION, 1, DELETE_LAST_RAY),
                   terminal_step.obs[1],
                   terminal_step.reward,
                   None,
                   1]
        
    return obs

####################################################################################################
####################################################################################################

#   ╔═════════╗
#   ║   CBF   ║
#   ╚═════════╝




from scipy.optimize import minimize
import numpy as np


def cbf_velocity_filter(
    v_cmd, omega_cmd,
    ray_distances, ray_angles,
    d_safe=0.5,
    alpha=5,
    d_safe_threshold_mult=3,
    debug=False
):
    x = 0
    y = 0
    theta = 0
    
    # Calcolo in coordinate robot-centriche (il robot è sempre in [0, 0, 0])
    obs_local = []
    for dist, ang_rel in zip(ray_distances, ray_angles):
        px = dist * np.cos(ang_rel)
        py = dist * np.sin(ang_rel)
        obs_local.append(np.array([px, py]))

    obs_local = np.array(obs_local)

    if debug:
        print("=== cbf_velocity_filter ===")
        print(f"Nominal command: v = {v_cmd:.3f} m/s, ω = {omega_cmd:.3f} rad/s")
        print(f"Robot pos: x = {x}, y = {y}, θ = {theta}")
        print("Detected obstacles (local frame):")
        for i, (ox, oy) in enumerate(obs_local):
            print(f"  [{i}] x = {ox:.2f}, y = {oy:.2f}, dist = {np.linalg.norm([ox, oy]):.2f} m")

    def barrier(state, obs, d_safe):
        px, py, _ = state
        return (px - obs[0]) ** 2 + (py - obs[1]) ** 2 - d_safe ** 2

    def cbf_constraint(delta_u, state, u_nominal, obs):
        px, py, theta = state
        v_nom, omega_nom = u_nominal
        dv, domega = delta_u

        v = v_nom + dv
        omega = omega_nom + domega

        delta_p = np.array([px - obs[0], py - obs[1]])

        dot_h = 2 * (
            delta_p[0] * v * np.cos(theta) +
            delta_p[1] * v * np.sin(theta) +
            np.dot(delta_p, np.array([-np.sin(theta), np.cos(theta)])) * omega
        )
        h_val = barrier(state, obs, d_safe)
        return dot_h + alpha * h_val

    state = np.array([x, y, theta])
    u_nominal = np.array([v_cmd, omega_cmd])

    d_focus = d_safe * d_safe_threshold_mult

    constraints = []
    count_active = 0
    for obs in obs_local:
        distance = np.linalg.norm(obs[:2] - np.array([x, y]))
        if distance < d_focus:
            obs_copy = obs.copy()
            constraints.append({
                'type': 'ineq',
                'fun': lambda delta_u, s=state, u=u_nominal, o=obs_copy: cbf_constraint(delta_u, s, u, o)
            })
            count_active += 1

    def objective(delta_u):
        return delta_u[0] ** 2 + delta_u[1] ** 2

    res = minimize(objective, [0.0, 0.0], constraints=constraints, method='SLSQP')

    v_safe, omega_safe = v_cmd, omega_cmd
    if res.success:
        dv, domega = res.x
        v_safe += dv
        omega_safe += domega
    else:
        if debug:
            print("⚠️ Optimization failed:", res.message)
        dv, domega = 0.0, 0.0  # fallback

    min_h = min([barrier(state, o, d_safe) for o in obs_local])
    
    if debug:
        print(f"CBF adjusted command: v = {v_safe:.3f}, ω = {omega_safe:.3f}")
        print(f"Active constraints: {count_active}")
        print(f"Min barrier h: {min_h:.4f}")
        print("===============================")

    return v_safe, omega_safe

def cbf_velocity_filter_qp(
    v_cmd, omega_cmd,
    ray_distances, ray_angles,
    d_safe=0.5,
    alpha=5,
    d_safe_threshold_mult=3,
    debug=False
):
    x, y, theta = 0.0, 0.0, 0.0
    state = np.array([x, y, theta])
    u_nominal = np.array([v_cmd, omega_cmd])
    
    obs_local = np.array([
        [d * np.cos(a), d * np.sin(a)]
        for d, a in zip(ray_distances, ray_angles)
    ])

    d_focus = d_safe * d_safe_threshold_mult
    A_list, b_list = [], []

    for obs in obs_local:
        delta_p = np.array([x - obs[0], y - obs[1]])
        distance = np.linalg.norm(delta_p)
        if distance > d_focus:
            continue

        h = distance**2 - d_safe**2
        px, py, theta = state
        v_nom, omega_nom = u_nominal

        # Derivata della funzione barriera
        d_dot_h_dv = 2 * (
            delta_p[0] * np.cos(theta) +
            delta_p[1] * np.sin(theta)
        )
        d_dot_h_domega = 2 * (
            delta_p[0] * (-np.sin(theta)) +
            delta_p[1] * (np.cos(theta))
        )

        a_i = np.array([-d_dot_h_dv, -d_dot_h_domega])
        b_i = alpha * h + 2 * (
            delta_p[0] * v_nom * np.cos(theta) +
            delta_p[1] * v_nom * np.sin(theta) +
            np.dot(delta_p, np.array([-np.sin(theta), np.cos(theta)])) * omega_nom
        )

        A_list.append(a_i)
        b_list.append(b_i)

    if len(A_list) == 0:
        return v_cmd, omega_cmd

    A = sp.csc_matrix(np.vstack(A_list))
    l = -1e20 * np.ones_like(b_list)  # no lower bounds
    u = np.array(b_list)

    P = sp.csc_matrix(np.array([[2.0, 0.0], [0.0, 2.0]]))  # H matrix
    q = np.zeros(2)  # f vector

    prob = osqp.OSQP()
    prob.setup(P=P, q=q, A=A, l=l, u=u, verbose=debug, polish=True)
    res = prob.solve()

    if res.info.status != 'solved':
        if debug:
            print("⚠️ OSQP failed:", res.info.status)
        return v_cmd, omega_cmd

    dv, domega = res.x
    return v_cmd + dv, omega_cmd + domega


def genera_angoli_radianti(k, n):
    if k == 1:
        return [0.0]
    
    # k è il numero di raggi per lato, quindi il totale è il doppio più quello centrale
    total_rays = k * 2 + 1
    
    step = 2 * n / (total_rays - 1)
    # Genera gli angoli da sinistra a destra (come prima)
    angoli_gradi = [n - i * step for i in range(total_rays)]
    # Converti in radianti
    angoli_radianti = [math.radians(a) for a in angoli_gradi]
    
    return angoli_radianti


def CBF_from_obs(ray_obs, action, env_info, 
                 d_safe, alpha, d_safe_mul,
                 angoli_radianti_precalcolati = None):
    
    if angoli_radianti_precalcolati is None:
        # posso farlo una sola volta
        angoli_radianti_precalcolati = genera_angoli_radianti(
                env_info.settings['ray_sensor_settings']['rays_per_direction'],
                env_info.settings['ray_sensor_settings']['max_ray_degrees']
            )
        
    ray_distances = [x * env_info.settings['ray_sensor_settings']['ray_length'] for x in ray_obs]

    # rete restituisce direttamente le VELOCITÀ (non accelerazioni)
    nn_v_front = action[0] * env_info.settings['agent_settings']['max_movement_speed']
    nn_v_ang = np.radians(action[1] * env_info.settings['agent_settings']['max_turn_speed'])

    # applica la CBF
    v_safe, omega_safe = cbf_velocity_filter_qp(
        nn_v_front, nn_v_ang,
        ray_distances, angoli_radianti_precalcolati,
        
        d_safe=d_safe,
        alpha=alpha,
        d_safe_threshold_mult=d_safe_mul,
        
        debug=False
    )

    # normalizza per restituire valori tra -1 e 1 (compatibili con policy)
    v_safe_norm = v_safe / env_info.settings['agent_settings']['max_movement_speed']
    omega_safe_norm = np.degrees(omega_safe) / env_info.settings['agent_settings']['max_turn_speed']

    return np.array([v_safe_norm, omega_safe_norm])


####################################################################################################
####################################################################################################

#   ╔═════════════════╗
#   ║   funny Utils   ║
#   ╚═════════════════╝

adjectives = [
    "quirky", "wobbly", "spunky", "zany", "goofy",
    "bubbly", "jolly", "snazzy", "whimsical", "dizzy",
    "wacky", "jumpy", "bouncy", "loopy", "nutty",
    "silly", "bizarre", "zesty", "peppy", "dapper",
    "fizzy", "fuzzy", "jazzy", "nerdy", "plucky",
    "smelly", "tipsy", "twisty", "wiggly", "zippy",
    "cheeky", "clumsy", "daffy", "dreamy", "fickle",
    "giddy", "hoppy", "itchy", "jaunty", "kooky",
    "lanky", "merry", "nifty", "perky", "quirky",
    "rusty", "snappy", "tipsy", "uptight", "vivid",
    "wonky", "yappy", "zesty", "airy", "blithe",
    "cuddly", "dandy", "eerie", "feisty", "glitzy",
    "hazy", "jumpy", "kooky", "loony", "mushy",
    "naughty", "oddball", "peppy", "quirky", "racy",
    "snazzy", "ticklish", "upbeat", "vapid", "wimpy",
    "yummy", "zany", "stinky"
]

nouns = [
    "penguin", "marshmallow", "pogo", "doodle", "pickle",
    "banana", "wombat", "noodle", "taco", "bubbles",
    "meerkat", "gizmo", "moose", "pudding", "zebra",
    "muffin", "nugget", "poptart", "dolphin", "goblin",
    "jellybean", "kiwi", "llama", "mango", "narwhal",
    "octopus", "pancake", "quokka", "raccoon", "sloth",
    "tofu", "unicorn", "vortex", "walrus", "yeti",
    "zombie", "zeppelin", "pickle", "yodel", "amoonguss",
    "beagle", "cupcake", "dingo", "earwig", "flamingo",
    "gazelle", "hippo", "iguana", "jackal", "kangaroo",
    "lemur", "mongoose", "narwhal", "otter", "parrot",
    "quail", "rabbit", "squid", "tapir", "urchin",
    "vulture", "wombat", "xerus", "yak", "zebra",
    "apple", "bubble", "cactus", "daisy", "ember",
    "feather", "gumdrop", "honey", "iceberg", "jelly",
    "kiwi", "lollipop", "mushroom", "nectar", "oyster",
    "pepper", "quiche", "rosebud", "sundae", "tulip",
    "umbrella", "velvet", "willow", "xylophone", "yarn",
    "zeppelin"
]

def generate_funny_name():
    adj = random.choice(adjectives)
    noun = random.choice(nouns)
    return f"{adj}_{noun}"
