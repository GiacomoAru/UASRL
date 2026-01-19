# === Standard library ===
import argparse
import json
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


####################################################################################################
####################################################################################################

#   ╔══════════════════════╗
#   ║   Training Classes   ║
#   ╚══════════════════════╝


class DenseSoftQNetwork(nn.Module):
    """
    Dense neural network for approximating a Soft Q-function.

    This network estimates the expected return of a state-action pair by 
    combining raycast-based observations, additional state features, and 
    action vectors. The inputs are processed through a configurable stack 
    of fully connected layers.

    Parameters
    ----------
    raycast_observation_shape : Tuple[int, int]
        Shape of the raycast observation (number of rays, features per ray).
    state_observation_size : int
        Size of the additional state observation vector.
    action_size : int
        Dimension of the action space.
    dense_layer : list[int]
        Sizes of the hidden dense layers.

    Methods
    -------
    __init__(raycast_observation_shape, state_observation_size, action_size, dense_layer)
        Build the network architecture, computing the input dimension and 
        creating the sequence of dense layers.

    forward(raycast_obs, state_obs, action)
        Run a forward pass through the network. The raycast observations 
        are flattened and concatenated with state features and the action. 
        The resulting tensor is propagated through the dense layers with 
        ReLU activations, except for the final layer, which outputs a scalar 
        Q-value for each input sample.
    """

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

class DenseActor(nn.Module):
    """
    Dense neural network actor module for policy approximation.

    This network maps a combination of raycast-based observations and 
    additional state features into an action distribution. The mean 
    and log standard deviation of the distribution are learned, and 
    actions are sampled with optional variance scaling. A rescaling 
    mechanism ensures actions are mapped to the target action space.

    Parameters
    ----------
    raycast_observation_shape : Tuple[int, int]
        Shape of the raycast observation (number of rays, features per ray).
    state_observation_size : int
        Size of the additional state observation vector.
    action_size : int
        Dimension of the action space (output size).
    action_space_min_value : int
        Minimum value for each action dimension.
    action_space_max_value : int
        Maximum value for each action dimension.
    dense_layer : list[int]
        Sizes of the hidden dense layers.

    Attributes
    ----------
    LOG_STD_MAX : int
        Maximum log standard deviation allowed for the action distribution.
    LOG_STD_MIN : int
        Minimum log standard deviation allowed for the action distribution.
    input_dim : int
        Total input dimension after flattening and concatenation of inputs.
    output_dim : int
        Dimension of the action space (number of actions).
    layers : nn.ModuleList
        Sequence of dense hidden layers.
    mean_layer : nn.Linear
        Final linear layer producing the mean of the action distribution.
    logstd_layer : nn.Linear
        Final linear layer producing the log standard deviation of the distribution.
    action_scale : torch.Tensor
        Scaling factor for mapping actions to the target range.
    action_bias : torch.Tensor
        Bias term for mapping actions to the target range.

    Methods
    -------
    forward(raycast_obs, state_obs)
        Runs a forward pass through the network. Returns the mean and log 
        standard deviation of the action distribution given the inputs.
    get_action(raycast_obs, state_obs, variance_scale=1.0)
        Samples an action from the policy given observations. Returns the 
        sampled action, its log probability, the mean action, and the log 
        standard deviation. If variance_scale is 0, the output is a 
        deterministic action.
    """

    
    LOG_STD_MAX = 3
    LOG_STD_MIN = -6

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
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, raycast_obs, state_obs, variance_scale=1.0):
        mean, log_std = self(raycast_obs, state_obs)
        
        if variance_scale > 0:
            std = log_std.exp()
            std = std * variance_scale  # redcuce standard deviation with coefficient v
            
            normal = torch.distributions.Normal(mean, std)
            
            x_t = normal.rsample()
            y_t = torch.tanh(x_t)
            action = y_t * self.action_scale + self.action_bias
            
            log_prob = normal.log_prob(x_t)
            
            # log_prob correction for the tanh function
            log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
            log_prob = log_prob.sum(1, keepdim=True)

            mean_action = torch.tanh(mean) * self.action_scale + self.action_bias
            
            return action, log_prob, mean_action, log_std
        
        else:
            mean_action = torch.tanh(mean) * self.action_scale + self.action_bias
            return mean_action, -torch.inf, mean_action, -torch.inf



class CustomChannel(SideChannel):
    """
    Custom side channel for communication between Unity ML-Agents Toolkit and Python.

    This channel is designed to handle end-episode and configuration messages exchanged
    between Unity environments and Python code. It distinguishes between settings 
    messages and data messages, storing them appropriately.

    Attributes
    ----------
    settings : dict
        Dictionary of configuration settings received from Unity, keyed by object name.
    msg_queue : list
        Queue of data messages received from Unity.
    settings_token : str
        Token prefix used to identify settings messages ("SETTINGS").
    data_token : str
        Token prefix used to identify data messages ("DATA").

    Methods
    -------
    __init__()
        Initializes the side channel with a fixed UUID.
    on_message_received(msg: IncomingMessage)
        Handles incoming messages from Unity. If the message starts with the settings 
        token, updates the settings dictionary. If it starts with the data token, appends 
        the message to the queue. Otherwise, prints a warning for unrecognized tokens.
    reset()
        Clears the data message queue.
    """

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
    
    def reset(self):
        self.msg_queue = []
            
class DebugSideChannel(SideChannel):
    """
    Custom side channel for debugging agent behavior.

    This channel sends detailed information about the agent’s actions, 
    policy outputs, Control Barrier Function (CBF) activations, uncertainty 
    filtering (UF), and related values from Python to the Unity environment. 
    It is primarily intended for monitoring and debugging Safe RL experiments.

    Methods
    -------
    __init__()
        Initializes the DebugSideChannel with a fixed UUID.
    send_agent_action_debug(forward_speed, angular_speed, policy_forward=0.0, 
                            policy_rotate=0.0, cbf=False, cbf_forward=0.0, 
                            cbf_rotate=0.0, uf=False, uf_threshold=0.0, 
                            uncertainty_value=0.0)
        Sends a JSON-encoded message to Unity containing the agent’s action 
        and debug information. Includes policy outputs, CBF corrections, 
        UF status, thresholds, and uncertainty values.
    on_message_received(msg: IncomingMessage)
        Handles unexpected incoming messages from Unity. Prints a warning 
        if such a message is received.
    """


    def __init__(self):
        super().__init__(uuid.UUID("abcdefab-1234-5678-9abc-def012345678"))
    
    def send_agent_action_debug(
        self,
        forward_speed: float,
        angular_speed: float,
        
        policy_forward: float = 0.0,
        policy_rotate: float = 0.0,
        
        cbf: bool = False,
        cbf_forward: float = 0.0,
        cbf_rotate: float = 0.0,
        uf: bool = False,
        uf_threshold: float = 0.0,
        uncertainty_value: float = 0.0,
    ) -> None:
        data = {
            "forward": float(forward_speed),
            "rotate": float(angular_speed),
            
            "policy_forward_action": float(policy_forward),
            "policy_rotate_action": float(policy_rotate),

            "cbf_activation": bool(cbf),
            "cbf_forward_action": float(cbf_forward),
            "cbf_rotate_action": float(cbf_rotate),

            "uf_activation": bool(uf),
            "uf_threshold": float(uf_threshold),
            "uncertainty_value": float(uncertainty_value),
        }

        json_str = json.dumps(data)
        msg = OutgoingMessage()
        msg.write_string(json_str)
        self.queue_message_to_send(msg)

    
    def on_message_received(self, msg: IncomingMessage) -> None:
        print('MESSAGGIO INATTESO')



####################################################################################################
####################################################################################################

#   ╔════════════════════╗
#   ║   Training Utils   ║
#   ╚════════════════════╝


def get_initial_action(agent_id, previous_movements, alpha=0.8, noise_std_init=0.3, 
                       noise_decay=0.99, min_noise_std=0.05, step=0):
    """
    Generate an initial action for an agent with noise and smoothing.

    This function computes a forward and steering action for an agent, 
    introducing Gaussian noise and temporal smoothing with past actions. 
    The action is clipped to valid ranges and stored in the `previous_movements` 
    dictionary for continuity across steps.

    Parameters
    ----------
    agent_id : hashable
        Unique identifier for the agent.
    previous_movements : dict
        Dictionary mapping agent IDs to their last (speed, steer) actions.
    alpha : float, optional
        Smoothing factor for blending base and noisy actions (default is 0.8).
    noise_std_init : float, optional
        Initial standard deviation of the Gaussian noise (default is 0.3).
    noise_decay : float, optional
        Multiplicative decay applied to the noise standard deviation at each step (default is 0.99).
    min_noise_std : float, optional
        Minimum noise standard deviation to prevent vanishing randomness (default is 0.05).
    step : int, optional
        Current time step, used to apply noise decay (default is 0).

    Returns
    -------
    np.ndarray
        A 2-element array [speed, steer], where speed is in [0.0, 1.0] and 
        steer is in [-1.0, 1.0].
    """


    # compute noise
    noise_std = max(noise_std_init * (noise_decay ** step), min_noise_std)

    # if it's the first action, generate forward and rotational action
    if agent_id not in previous_movements:
        base_speed = np.random.uniform(0.0, 0.5)
        base_steer = np.random.uniform(-0.2, 0.2)
    else:
        base_speed, base_steer = previous_movements[agent_id]

    # add gaussian noise for diversify the action
    noisy_speed = base_speed + np.random.normal(0, noise_std)
    noisy_steer = base_steer + np.random.normal(0, noise_std)

    # smoothing with previous actions
    speed = alpha * base_speed + (1 - alpha) * noisy_speed
    steer = alpha * base_steer + (1 - alpha) * noisy_steer

    # clipping to avoid wrong values
    speed = np.clip(speed, 0.0, 1.0)     # only forward action are allowed
    steer = np.clip(steer, -1.0, 1.0)

    # keep track of the computed action
    previous_movements[agent_id] = (speed, steer)

    return np.array([speed, steer])
   
   
   
####################################################################################################
####################################################################################################

#   ╔═══════════════════╗
#   ║   Parsing Utils   ║
#   ╚═══════════════════╝
    
    
def _load_config(config_path):
    """
    Load a YAML configuration file.

    This function reads a YAML file from the specified path and parses it 
    into a Python dictionary. If the file is empty, an empty dictionary is returned.

    Parameters
    ----------
    config_path : str or Path
        Path to the YAML configuration file.

    Returns
    -------
    dict
        Dictionary containing the configuration parameters. Returns an 
        empty dictionary if the file is empty.
    """

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config or {}

def parse_args_from_file(config_path):
    """
    Parse training arguments from a YAML configuration file.

    This function loads parameters from a YAML file and uses them to 
    populate an argparse parser with defaults. If a parameter is not 
    specified in the configuration, a fallback default value is used.

    Parameters
    ----------
    config_path : str or Path
        Path to the YAML configuration file.

    Returns
    -------
    argparse.Namespace
        Parsed arguments containing all hyperparameters and settings for training.

    Configuration Parameters
    ------------------------
    exp-name : str, default=filename of the script
        Name of the experiment.
    env-id : str, default="Environment-ID"
        Identifier of the environment.
    q-ensemble-n : int, default=2
        Number of Q-networks in the ensemble.
    bootstrap-batch-proportion : float, default=0.8
        Proportion of the batch used for bootstrapping.
    torch-deterministic : bool, default=True
        Whether to enforce deterministic operations in PyTorch.
    cuda : bool, default=True
        Whether to enable CUDA if available.
    loss-log-interval : int, default=100
        Interval (in steps) for logging losses.
    metrics-log-interval : int, default=300
        Interval (in steps) for logging metrics.
    metrics-smoothing : float, default=0.95
        Exponential smoothing factor for metrics.
    q-network-layers : list of int, default=[64, 64]
        Hidden layer sizes for the Q-network.
    actor-network-layers : list of int, default=[64, 64]
        Hidden layer sizes for the Actor network.
    total-timesteps : int, default=1_000_000
        Total number of training timesteps.
    buffer-size : int, default=1e6
        Replay buffer capacity.
    update-per-step : int, default=1
        Number of updates per environment step.
    gamma : float, default=0.99
        Discount factor for rewards.
    tau : float, default=0.005
        Soft update coefficient for target networks.
    batch-size : int, default=256
        Training batch size.
    learning-starts : int, default=5000
        Number of steps before learning starts.
    policy-lr : float, default=3e-4
        Learning rate for the policy network.
    q-lr : float, default=1e-3
        Learning rate for the Q-networks.
    policy-frequency : int, default=2
        Frequency of policy network updates.
    target-network-frequency : int, default=1
        Frequency of target network updates.
    noise-clip : float, default=0.5
        Clipping value for target policy smoothing noise.
    alpha : float, default=0.2
        Entropy regularization coefficient.
    autotune : bool, default=True
        Whether to enable automatic tuning of alpha.
    alpha-lr : float, default=1e-4
        Learning rate for alpha when autotuning is enabled.
    """

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

#   ╔═══════════════════════╗
#   ║   Data Manipolation   ║
#   ╚═══════════════════════╝


def organize_observations_for_conv(flat_observations, num_stacks, num_rays_per_dir, num_tags, 
                                   remove_last=False):
    """
    Organize flat ray-based observations into a 2D structure for convolutional processing.

    This function reshapes a flat list of observations into a matrix 
    where rows correspond to stacked frames and columns correspond to rays. 
    Only a subset of features per ray is selected (skipping the first `num_tags+1` features). 
    Optionally, the last ray can be removed.

    Parameters
    ----------
    flat_observations : array-like
        Flattened list or array of observations.
    num_stacks : int
        Number of stacked frames.
    num_rays_per_dir : int
        Number of rays per direction. The total number of rays is 
        computed as `2 * num_rays_per_dir + 1`.
    num_tags : int
        Number of categorical or tag features associated with each ray.
    remove_last : bool, optional
        If True, removes the last ray from the output (default is False).

    Returns
    -------
    np.ndarray
        Array of shape (num_stacks, total_rays) if `remove_last=False`, 
        otherwise shape (num_stacks, total_rays - 1).
    """

    features_per_ray = num_tags + 2
    total_rays = num_rays_per_dir * 2 + 1 # Il primo e l'ultimo sono uguali se i gradi sono 180
    
    # Reshape (stack, ray, feature)
    reshaped = np.array(flat_observations) 
    reshaped = reshaped[[i for i in range(features_per_ray - 1, num_stacks*total_rays*features_per_ray, features_per_ray)]]
    reshaped = reshaped.reshape((num_stacks, total_rays))
    if remove_last:
        reshaped = reshaped[:,:-1] # remove the last ray
    
    return reshaped

def collect_data_after_step(environment, env_info):
    """
    Collect observations, rewards, and termination flags from the environment after one step.

    This function retrieves decision and terminal steps from a Unity ML-Agents 
    environment, extracts ray-based and state observations, and organizes 
    them into a dictionary keyed by agent ID. Ray observations are reshaped 
    for convolutional processing. Rewards, actions, and done flags are also included.

    Parameters
    ----------
    environment : UnityEnvironment
        The Unity ML-Agents environment instance.
    env_info : object
        Environment information object containing sensor and behavior settings. 
        It must provide the following fields in `settings`:
            - ray_sensor_settings['observation_stacks'] (int)
            - ray_sensor_settings['rays_per_direction'] (int)
            - ray_sensor_settings['ignore_last_ray'] (bool)
            - behavior_parameters_settings['behavior_name'] (str)

    Returns
    -------
    dict
        Dictionary mapping agent IDs to lists of collected data in the format:
        [ray_obs, state_obs, reward, action, done], where
            ray_obs : np.ndarray
                Ray-based observation organized for convolutional input.
            state_obs : array-like
                Additional state observation.
            reward : float
                Reward received by the agent.
            action : None
                Placeholder for the action taken (filled later in the pipeline).
            done : int
                0 if the agent is active, 1 if the agent is in a terminal state.
    """

    RAY_STACK = env_info.settings['ray_sensor_settings']['observation_stacks']
    RAY_PER_DIRECTION = env_info.settings['ray_sensor_settings']['rays_per_direction']
    DELETE_LAST_RAY = env_info.settings['ray_sensor_settings']['ignore_last_ray']
    BEHAVOIR_NAME = env_info.settings['behavior_parameters_settings']['behavior_name']
                
    decision_steps, terminal_steps = environment.get_steps(BEHAVOIR_NAME)
    
    obs = {}
    
    for id in decision_steps:
        decision_step = decision_steps[id]
        # ray_obs, state_obs, reward, action, done
        obs[id] = [organize_observations_for_conv(decision_step.obs[0], RAY_STACK, RAY_PER_DIRECTION, 2, DELETE_LAST_RAY),
                   decision_step.obs[1],
                   decision_step.reward,
                   None,
                   0]
        
    for id in terminal_steps:
        terminal_step = terminal_steps[id]
        # ray_obs, state_obs, reward, action, done
        obs[id] = [organize_observations_for_conv(terminal_step.obs[0], RAY_STACK, RAY_PER_DIRECTION, 2, DELETE_LAST_RAY),
                   terminal_step.obs[1],
                   terminal_step.reward,
                   None,
                   1]
        
    return obs



####################################################################################################
####################################################################################################

#   ╔═════════════════╗
#   ║   Funny Utils   ║
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
    """
    Generate a random name by combining an adjective and a noun.

    This function selects a random adjective and a random noun from 
    predefined lists and concatenates them with an underscore.

    Returns
    -------
    str
        A string in the format "adjective_noun".
    """

    adj = random.choice(adjectives)
    noun = random.choice(nouns)
    return f"{adj}_{noun}"
