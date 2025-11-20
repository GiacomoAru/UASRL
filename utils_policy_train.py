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
import yaml
import argparse
import os


####################################################################################################
####################################################################################################

#   ╔══════════════════════╗
#   ║   Training Classes   ║
#   ╚══════════════════════╝

####################################################################################################
####################################################################################################


class DenseSoftQNetwork(nn.Module):
    
    def __init__(self, 
                 input_dim: int,
                 dense_layer: list[int]):
        
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = 1
        
        dense_layer = [self.input_dim] + dense_layer + [self.output_dim]
        self.layers =  nn.ModuleList()
        for i, layer in enumerate(dense_layer[:-1]):
            self.layers.append(nn.Linear(layer, dense_layer[i + 1]))
        

    def forward(self, x):
        
        # for each layer, first the layer and then the activation function
        for layer in self.layers[:-1]:
            x = layer(x)
            x = F.relu(x)
            
        # the last layer does not have an activation function
        x = self.layers[-1](x)
        
        return x

class DenseActor(nn.Module):
    
    LOG_STD_MAX = 4
    LOG_STD_MIN = -8

    def __init__(self, 
                 input_dim: int,
                 output_dim: int,
                 
                 output_space_max_value: float,
                 output_space_min_value: float,
                 dense_layer: list[int]):
        
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        
        dense_layer = [self.input_dim] + dense_layer
        self.layers = nn.ModuleList()
        for i, layer in enumerate(dense_layer[:-1]):
            self.layers.append(nn.Linear(layer, dense_layer[i + 1]))
        self.mean_layer = nn.Linear(dense_layer[-1], self.output_dim)
        self.logstd_layer = nn.Linear(dense_layer[-1], self.output_dim)   
                
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((output_space_max_value - output_space_min_value) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((output_space_max_value + output_space_min_value) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        
        # for each layer, first the layer and then the activation function
        for layer in self.layers:
            x = layer(x)
            x = F.relu(x)

        mean = self.mean_layer(x)
        log_std = self.logstd_layer(x)
        
        log_std = torch.tanh(log_std)
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x, variance_scale=1.0):
        mean, log_std = self(x)
        
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
    
    settings = {}
    start_msg_queue = []
    stop_msg_queue = []
    
    START_EPISODE_TOKEN = "01"
    END_EPISODE_TOKEN = "02"
    DATA_TOKEN = "03"
    DEBUG_TOKEN = "04"
    SEPARATOR_TOKEN = '|'
    
    def __init__(self) -> None:
        super().__init__(uuid.UUID("621f0a70-4f87-11ea-a6bf-784f4387d1f7"))

    def on_message_received(self, msg: IncomingMessage) -> None:
        
        received = msg.read_string()
        splitted = received.split(self.SEPARATOR_TOKEN)
        
        if splitted[0] == self.DATA_TOKEN:
            data = json.loads(splitted[1])
            el = data['obj_name']
            del data['obj_name']
            
            self.settings[el] = data
        elif splitted[0] == self.START_EPISODE_TOKEN:
            data = json.loads(splitted[1])
            self.start_msg_queue.append(data)
        elif splitted[0] == self.END_EPISODE_TOKEN:
            data = json.loads(splitted[1])
            self.start_msg_queue.append(data)   
        else:
            print("TOKEN MESSAGGIO NON IMPLEMENTATO")
    
    def send_agent_action_debug(
        self,
        agent_id: int,
        
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
        msg.write_string(f'{self.DEBUG_TOKEN}{self.SEPARATOR_TOKEN}{agent_id}{self.SEPARATOR_TOKEN}{msg}')
        self.queue_message_to_send(msg)
    
    def clear_queue(self):
        self.start_msg_queue = []
        self.stop_msg_queue = []



####################################################################################################
####################################################################################################

#   ╔════════════════════╗
#   ║   Training Utils   ║
#   ╚════════════════════╝

####################################################################################################
####################################################################################################

def get_initial_action(agent_id, 
                       alpha=0.9, 
                       noise_mean=0.0, 
                       noise_std=0.1):
    
    # -----------------------------------------------------------
    # 1. CREAZIONE MEMORIA PERSISTENTE (Solo la prima volta)
    # -----------------------------------------------------------
    # Controlliamo se la funzione ha già l'attributo "memory".
    # Se non ce l'ha, lo creiamo come un dizionario vuoto.
    if not hasattr(get_initial_action, "memory"):
        get_initial_action.memory = {}
    
    # Creiamo un alias più corto per comodità
    previous_movements = get_initial_action.memory

    # -----------------------------------------------------------
    # 2. LOGICA DI AGGIORNAMENTO
    # -----------------------------------------------------------
    
    # Se è la prima volta per questo agente
    if agent_id not in previous_movements:
        actual_speed = np.random.uniform(0.0, 1.0)
        actual_steer = np.random.uniform(-1.0, 1.0)
        
        target_speed = np.random.uniform(0.0, 1.0)
        target_steer = np.random.uniform(-1.0, 1.0)
    else:
        # Recuperiamo lo stato precedente
        actual_speed, actual_steer, target_speed, target_steer = previous_movements[agent_id]

    # Se abbiamo raggiunto il target (o siamo molto vicini), ne scegliamo uno nuovo
    if abs(target_speed - actual_speed) < 0.01:
        target_speed = np.random.uniform(0.0, 1.0)
        
    if abs(target_steer - actual_steer) < 0.01:
        target_steer = np.random.uniform(-1.0, 1.0) # Corretto: range da -1 a 1
    
    # Avviciniamo il valore attuale al target (smoothing)
    actual_speed = alpha * actual_speed + (1 - alpha) * target_speed
    actual_steer = alpha * actual_steer + (1 - alpha) * target_steer
    
    # -----------------------------------------------------------
    # 3. RUMORE E OUTPUT
    # -----------------------------------------------------------
    
    # Aggiungiamo rumore gaussiano per diversificare
    noisy_speed = actual_speed + np.random.normal(noise_mean, noise_std)
    noisy_steer = actual_steer + np.random.normal(noise_mean, noise_std)

    # Clipping per assicurarsi che i valori siano validi per i motori
    # Usiamo noisy_speed/steer qui, non 'speed' che non era definito
    final_speed = np.clip(noisy_speed, 0.0, 1.0)
    final_steer = np.clip(noisy_steer, -1.0, 1.0)

    # Salviamo lo stato "pulito" (senza rumore) per il prossimo passo
    previous_movements[agent_id] = (actual_speed, actual_steer, target_speed, target_steer)

    return np.array([final_speed, final_steer])
   
   
   
####################################################################################################
####################################################################################################

#   ╔═══════════════════╗
#   ║   Parsing Utils   ║
#   ╚═══════════════════╝
    
####################################################################################################
####################################################################################################

def parse_config(config_path = './config/train.yaml'):
    """
    Carica un file YAML e restituisce un oggetto Namespace con tutti i parametri.
    Non c'è bisogno di definire i nomi manualmente.
    """
    
    # 1. Controlla se il file esiste
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"File di configurazione non trovato: {config_path}")

    # 2. Carica il dizionario dal YAML
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f) or {}

    # 3. Imposta alcuni default intelligenti se mancano nel file (opzionale)
    if "exp_name" not in config_dict:
        config_dict["exp_name"] = os.path.basename(__file__).rstrip(".py")

    # 4. MAGIA: Convertiamo il dizionario direttamente in argparse.Namespace
    # L'operatore ** esplode il dizionario in argomenti chiave=valore
    args = argparse.Namespace(**config_dict)

    return args

####################################################################################################
####################################################################################################

#   ╔═══════════════════════╗
#   ║   Data Manipolation   ║
#   ╚═══════════════════════╝


def organize_observations_linear(flat_observations, num_tags):
    features_per_ray = num_tags + 2
    # obs_arr = np.array(flat_observations)
    selected = flat_observations[features_per_ray - 1::features_per_ray]
    return selected


def collect_data_after_step(environment, BEHAVIOUR_NAME):

    decision_steps, terminal_steps = environment.get_steps(BEHAVIOUR_NAME)
    
    obs = {}
    
    for id in decision_steps:
        decision_step = decision_steps[id]
        # agent_id, obs, reward, action, done
        state = np.concatenate([organize_observations_linear(decision_step.obs[0], 2),
                                decision_step.obs[1:]])
        obs[id] = [decision_step.obs[1][0],
                   state,
                   decision_step.reward,
                   None,
                   0]
    
    for id in terminal_step:
        terminal_step = terminal_steps[id]
        # agent_id, obs, reward, action, done
        state = np.concatenate([organize_observations_linear(terminal_steps.obs[0], 2),
                                terminal_steps.obs[1:]])
        obs[id] = [terminal_steps.obs[1][0],
                   state,
                   terminal_steps.reward,
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
