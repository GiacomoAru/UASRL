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


####################################################################################################
####################################################################################################

#   ╔══════════════════════╗
#   ║   Training Classes   ║
#   ╚══════════════════════╝

####################################################################################################
####################################################################################################

# Funzione per inizializzare i pesi in modo stabile
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.orthogonal_(m.weight, gain=torch.nn.init.calculate_gain('relu'))
        torch.nn.init.constant_(m.bias, 0)


class DenseSoftQNetwork(nn.Module):
    
    def __init__(self, 
                 obs_dim: int,
                 action_dim: int,
                 dense_layer: list[int]):
        
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.output_dim = 1
        
        dense_layer = [self.obs_dim + self.action_dim] + dense_layer + [self.output_dim]
        self.layers =  nn.ModuleList()
        for i, layer in enumerate(dense_layer[:-1]):
            self.layers.append(nn.Linear(layer, dense_layer[i + 1]))
        
        self.apply(weights_init_)
        
        self.register_buffer('obs_mean', torch.zeros(self.obs_dim))
        self.register_buffer('obs_std', torch.ones(self.obs_dim))
        
    def set_normalization_params(self, mean, std):
        self.obs_mean.copy_(mean)
        self.obs_std.copy_(std)
        
    def forward(self, x, a):
        x = (x - self.obs_mean) / (self.obs_std + 1e-6)
        # x = torch.clamp(x, -5.0, 5.0)
        
        x = torch.cat([x, a], 1)
        
        # for each layer, first the layer and then the activation function
        for layer in self.layers[:-1]:
            x = layer(x)
            x = F.relu(x)
            
        # the last layer does not have an activation function
        x = self.layers[-1](x)
        
        return x

class DenseActor(nn.Module):
    
    LOG_STD_MAX = 2
    LOG_STD_MIN = -10

    def __init__(self, 
                 input_dim: int,
                 output_dim: int,
                 output_space_min_value: float,
                 output_space_max_value: float,
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
            "action_scale", 
            torch.tensor((output_space_max_value - output_space_min_value) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((output_space_max_value + output_space_min_value) / 2.0, dtype=torch.float32)
        )

        self.register_buffer('input_mean', torch.zeros(input_dim))
        self.register_buffer('input_std', torch.ones(input_dim))
        
        # --- FIX 1: Inizializzazione Pesi Ortogonale ---
        self.apply(weights_init_)

    def set_normalization_params(self, mean, std):
        self.input_mean.copy_(mean)
        self.input_std.copy_(std)
        
    def forward(self, x):
        x = (x - self.input_mean) / (self.input_std + 1e-6)
        x = torch.clamp(x, -5.0, 5.0) # 5 deviazioni standard 
        
        for layer in self.layers:
            x = layer(x)
            x = F.relu(x)

        mean = self.mean_layer(x)
        log_std = self.logstd_layer(x)
        
        log_std = torch.tanh(log_std)
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1) # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x, std_scale=1.0):

        mean, log_std = self(x)
            
        if std_scale > 0:
            std = log_std.exp()
            std = std * std_scale 
            
            normal = torch.distributions.Normal(mean, std)
            
            x_t = normal.rsample()
            y_t = torch.tanh(x_t)
            action = y_t * self.action_scale + self.action_bias
            
            log_prob = normal.log_prob(x_t)
            log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
            log_prob = log_prob.sum(1, keepdim=True)

            mean_action = torch.tanh(mean) * self.action_scale + self.action_bias
            
            return action, log_prob, mean_action
        
        else:
            mean_action = torch.tanh(mean) * self.action_scale + self.action_bias
            return mean_action, -torch.inf, mean_action
        
class CustomChannel(SideChannel):
    
    START_EPISODE_TOKEN = "01"
    END_EPISODE_TOKEN = "02"
    DATA_TOKEN = "03"
    DEBUG_TOKEN = "04"
    SEPARATOR_TOKEN = '|'
    
    def __init__(self) -> None:
        super().__init__(uuid.UUID("621f0a70-4f87-11ea-a6bf-784f4387d1f7"))
        
        # Inizializza le code qui dentro per ogni singola istanza!
        self.start_msg_queue = []
        self.stop_msg_queue = []
    

    def on_message_received(self, msg: IncomingMessage) -> None:
        
        received = msg.read_string()
        splitted = received.split(self.SEPARATOR_TOKEN)

        if splitted[0] == self.START_EPISODE_TOKEN:
            data = json.loads(splitted[1])
            self.start_msg_queue.append(data)
        elif splitted[0] == self.END_EPISODE_TOKEN:
            data = json.loads(splitted[1])
            self.stop_msg_queue.append(data)   
        else:
            print(f"TOKEN MESSAGGIO NON IMPLEMENTATO: {received}")
    
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
        # Devi usare json_str
        msg.write_string(f'{self.DEBUG_TOKEN}{self.SEPARATOR_TOKEN}{agent_id}{self.SEPARATOR_TOKEN}{json_str}')
        self.queue_message_to_send(msg)
    
    def clear_queue(self):
        self.start_msg_queue = []
        self.stop_msg_queue = []



class RunningMean:
    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def update(self, value, n=1):
        """Aggiorna la media con un nuovo valore o batch di valori"""
        self.sum += value * n
        self.count += n

    def mean(self):
        if self.count == 0:
            return 0.0
        return self.sum / self.count

    def reset(self):
        self.sum = 0.0
        self.count = 0
        
class DenseStackedObservations:
    def __init__(self, stack_size, observation_shape, n_agents=1):
        self.stack_size = stack_size
        self.observation_shape = observation_shape
        self.n_agents = n_agents
        
        # stack e empty flags
        self.stack = np.zeros((n_agents, observation_shape * stack_size), dtype=np.float32)
        self.empty = np.ones((n_agents,), dtype=bool)

    def reset(self, observation_ids):
        ids = np.atleast_1d(np.array(observation_ids, dtype=np.int32))
        # reset empty flags
        self.empty[ids] = True

    def add_observation(self, observation_stack, observation_ids):
        ids = np.atleast_1d(np.array(observation_ids, dtype=np.int32))
        obs = np.atleast_2d(np.array(observation_stack, dtype=np.float32))

        if len(ids) > self.n_agents:
            raise ValueError("ID Number > n_agents")
        
        # first observation
        mask_init = self.empty[ids]
        if np.any(mask_init):
            ids_to_init = ids[mask_init]
            obs_to_init = obs[mask_init]
            
            # create first stack by tiling the first observation
            self.stack[ids_to_init] = np.tile(obs_to_init, (1, self.stack_size))
            self.empty[ids_to_init] = False
        
        # subsequent observations
        mask_update = ~mask_init
        if np.any(mask_update):
            ids_to_update = ids[mask_update]
            obs_to_update = obs[mask_update]
            
            # rolling the stack to the left
            self.stack[ids_to_update] = np.roll(self.stack[ids_to_update], -self.observation_shape, axis=1)
            # appending the new observation at the end
            self.stack[ids_to_update, -self.observation_shape:] = obs_to_update
        
    def get_stacked_observations(self, observation_ids):
        ids = np.atleast_1d(np.array(observation_ids, dtype=np.int32))
        # return the obs stack
        return self.stack[ids].copy()
           
####################################################################################################
####################################################################################################

#   ╔═══════════════════╗
#   ║   Parsing Utils   ║
#   ╚═══════════════════╝
    
####################################################################################################
####################################################################################################

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def parse_config_file(config_path = './config/train.yaml'):

    # 1. Controlla se il file esiste
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"File di configurazione non trovato: {config_path}")

    # 2. Carica il dizionario dal YAML
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f) or {}
        
    return config_dict

def parse_args():
    
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("config_path", type=str, help="main config file path")

    initial_args, remaining_argv = pre_parser.parse_known_args()
    CONFIG_PATH = initial_args.config_path
    print(f'Config path: {CONFIG_PATH}')

    file_config_dict = parse_config_file(CONFIG_PATH)
    parser = argparse.ArgumentParser(description="Training Script")
    parser.add_argument("config_path", type=str)

    for key, value in file_config_dict.items():
        # Gestione specifica per i booleani
        if isinstance(value, bool):
            parser.add_argument(f"--{key}", type=str2bool, default=value)
        
        # Gestione per None (assumiamo stringa o saltiamo)
        elif value is None:
            parser.add_argument(f"--{key}", type=str, default=value)
            
        # Gestione per tutti gli altri tipi (int, float, str)
        else:
            parser.add_argument(f"--{key}", type=type(value), default=value)
        
    args = parser.parse_args()
    return args

####################################################################################################
####################################################################################################

#   ╔════════════════════╗
#   ║   Training Utils   ║
#   ╚════════════════════╝

####################################################################################################
####################################################################################################

def save_models(actor, qf_ensemble, qf_ensemble_target, save_path, suffix=''):
    torch.save(actor.state_dict(), os.path.join(save_path, f'actor{suffix}.pth'))
    for i, qf in enumerate(qf_ensemble):
        torch.save(qf.state_dict(), os.path.join(save_path, f'qf{i+1}{suffix}.pth'))
    for i, qft in enumerate(qf_ensemble_target):
        torch.save(qft.state_dict(), os.path.join(save_path, f'qf{i+1}_target{suffix}.pth'))
        
def modify_config_for_curriculum(step, total_step, obs_config):
    if 'og_initial_fill_percentage' not in obs_config:
        obs_config['og_initial_fill_percentage'] = obs_config['initial_fill_percentage']
    
    obs_config['initial_fill_percentage'] = 0.15 + (obs_config['og_initial_fill_percentage'] - 0.15)*(step/total_step)
    print(f'Curriculum step {step}/{total_step}:\n\tsetting initial_fill_percentage to {obs_config["initial_fill_percentage"]}')
    
def get_initial_action_batch(agent_ids, 
                             alpha=0.8, 
                             target_accuracy=0.05,
                             noise_mean=0.0, 
                             noise_std=0.1):
    
    # -----------------------------------------------------------
    # 1. GESTIONE MEMORIA E SETUP
    # -----------------------------------------------------------
    if not hasattr(get_initial_action_batch, "memory"):
        get_initial_action_batch.memory = {}
    
    memory = get_initial_action_batch.memory
    num_agents = len(agent_ids)
    if num_agents < 1:
        return np.zeros((0, 2), dtype=np.float32)
    
    # Creiamo una matrice temporanea per contenere lo stato di QUESTA batch
    # Shape: (num_agents, 4) -> [act_spd, act_str, tgt_spd, tgt_str]
    batch_state = np.zeros((num_agents, 4), dtype=np.float32)
    
    # Carichiamo i dati dalla memoria (o inizializziamo se nuovi)
    # Nota: Questo ciclo è necessario perché gli ID potrebbero non essere sequenziali,
    # ma è veloce perché fa solo copie di dati.
    for i, agent_id in enumerate(agent_ids):
        if agent_id in memory:
            batch_state[i] = memory[agent_id]
        else:
            # Inizializzazione casuale per nuovi agenti (tutte e 4 le variabili tra -1 e 1)
            batch_state[i] = np.random.uniform(-1.0, 1.0, 4)

    # Separiamo le viste per comodità (sono riferimenti, non copie!)
    # actuals: colonne 0 e 1 (Speed, Steer)
    # targets: colonne 2 e 3 (TargetSpeed, TargetSteer)
    actuals = batch_state[:, 0:2] 
    targets = batch_state[:, 2:4] 

    # -----------------------------------------------------------
    # 2. LOGICA VETTORIALIZZATA (Il cuore della batch)
    # -----------------------------------------------------------
    
    # A. Controllo raggiungimento target
    # Creiamo una maschera booleana: True dove la differenza è piccola
    reached_target_mask = np.abs(targets - actuals) < target_accuracy
    
    # Generiamo nuovi target casuali per TUTTI (è più veloce che farne pochi)
    new_random_targets = np.random.uniform(-1.0, 1.0, (num_agents, 2))
    
    # Usiamo np.where per aggiornare SOLO dove la maschera è True
    # Se reached_target_mask è True -> prendi new_random_targets
    # Altrimenti -> mantieni il vecchio targets
    targets[:] = np.where(reached_target_mask, new_random_targets, targets)
    
    # B. Smoothing (Avvicinamento al target)
    # Formula: new = alpha * old + (1 - alpha) * target
    actuals[:] = alpha * actuals + (1 - alpha) * targets
    
    # -----------------------------------------------------------
    # 3. SALVATAGGIO E OUTPUT
    # -----------------------------------------------------------
    
    # Salviamo lo stato aggiornato ("pulito") nella memoria persistente
    # Dobbiamo rimettere i dati nel dizionario per il prossimo step
    for i, agent_id in enumerate(agent_ids):
        memory[agent_id] = batch_state[i]
        
    # C. Aggiunta Rumore e Clipping (per l'output finale)
    noise = np.random.normal(noise_mean, noise_std, (num_agents, 2))
    
    final_actions = actuals + noise
    final_actions = np.clip(final_actions, -1.0, 1.0)
    
    return final_actions
   
def apply_unity_settings(channel, config_dict, label_prefix=''):

    for key, value in config_dict.items():
        # Unity accetta SOLO float su questo canale.
        # Convertiamo bool e int, ignoriamo stringhe/liste.
        if isinstance(value, bool):
            channel.set_float_parameter(label_prefix + key, 1.0 if value else 0.0)
        elif isinstance(value, (int, float)):
            channel.set_float_parameter(label_prefix + key, float(value))

def write_dict(writer, d, name):
    if type(d) is not dict:
        d = vars(d)
        
    writer.add_text(
        name,
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{d[key]}|" for key in d])),
    )

def update_stats_from_message(all, success, failure, msg, smoothing):
    def update_stats_helper(stats, msg, smoothing):
        if stats == {}:
            for key in msg:
                stats[key] = msg[key]
        else:
            for key in msg:
                stats[key] = stats[key]*smoothing + (1 - smoothing)*msg[key]  

        if 'ep_count' in stats:
            stats['ep_count'] += 1
        else:
            stats['ep_count'] = 1
    
    if 'id' in msg:
        del msg['id']      
    msg['path_lenght_ratio'] = msg['distance_traveled'] / msg['path_length']
    msg['SPL'] = msg['success'] * (msg['path_length']/max(msg['path_length'], msg['distance_traveled']))
    
    update_stats_helper(all, msg, smoothing)
    if msg['success'] == 1:
        update_stats_helper(success, msg, smoothing)
    else:
        update_stats_helper(failure, msg, smoothing)

def print_update(env_step, total_timesteps, start_time, stats):
    print_text = f"[{env_step}/{total_timesteps}] "
    for s in ['success', 'reward', 'collisions', 'length', 'SPL']:
        print_text += f"|{s}: {stats[s]:.5f}"
    print_text += f'| SPS: {int(env_step / (time.time() - start_time))}'
    print(print_text)
       
def log_stats_to_tensorboard(writer, all, success, failure):   
    for s in all:
        writer.add_scalar("all_ep_stats/" + s, all[s], all['ep_count'])
    for s in success:
        writer.add_scalar("success_ep_stats/" + s, success[s], all['ep_count'])
    for s in failure:
        writer.add_scalar("failure_ep_stats/" + s, failure[s], all['ep_count'])  
                
def log_stats_to_wandb(wandb_run, dicts, labels, x_value):
    for label, d in zip(labels, dicts):
        if label != '':
            to_save = {f'{label}/{k}': v for k, v in d.items()}
            wandb_run.log(to_save, step=x_value)
        else:
            wandb_run.log(d, step=x_value)



####################################################################################################
####################################################################################################

#   ╔═══════════════════════╗
#   ║   Data Manipolation   ║
#   ╚═══════════════════════╝


def organize_observations(raw_observations, num_tags):
    features_per_ray = num_tags + 2
    selected = raw_observations[features_per_ray - 1::features_per_ray]
    return selected

def collect_data_after_step(environment, BEHAVIOUR_NAME, STATE_SIZE):
    decision_steps, terminal_steps = environment.get_steps(BEHAVIOUR_NAME)
    
    obs = {}
    
    for id in decision_steps:
        decision_step = decision_steps[id]
        # agent_id, obs, reward, action, done
        state = np.concatenate([organize_observations(decision_step.obs[0], 2),
                                decision_step.obs[1].reshape(-1, STATE_SIZE + 1)[:,1:].flatten()])

        obs[decision_step.obs[1][0]] = [id,
                   state,
                   decision_step.reward,
                   None,
                   0]
    
    for id in terminal_steps:
        terminal_step = terminal_steps[id]
        # agent_id, obs, reward, action, done
        state = np.concatenate([organize_observations(terminal_step.obs[0], 2),
                                terminal_step.obs[1].reshape(-1, STATE_SIZE + 1)[:,1:].flatten()])
        obs[terminal_step.obs[1][0]] = [id,
                   state,
                   terminal_step.reward,
                   None,
                   1]

    return obs



def organize_observations_batch(raw_observations, num_tags):
    features_per_ray = num_tags + 2
    selected = raw_observations[:,features_per_ray - 1::features_per_ray]
    return selected

def sync_indices(ids1, ids2):

    mask1 = np.isin(ids1, ids2)
    sorter = np.argsort(ids2)

    target_values = ids1[mask1]
    positions_in_sorted = np.searchsorted(ids2, target_values, sorter=sorter)
    indices2 = sorter[positions_in_sorted]
    
    return mask1, indices2

def observe_batch(environment, BEHAVIOUR_NAME, observation_size):
    n_tag = 2
    
    decision_steps, terminal_steps = environment.get_steps(BEHAVIOUR_NAME)
    
    # --- DECISION STEPS ---
    return_decision = []
    
    if len(decision_steps.agent_id) > 0:
        ray_state = organize_observations_batch(decision_steps.obs[0], n_tag)
        internal_state = decision_steps.obs[1][:, 1:] # remove id

        return_decision.append(decision_steps.obs[1][:,0]) # ID speciali
        return_decision.append(np.concatenate([ray_state, internal_state], axis=1)) # Obs Batch
        return_decision.append(decision_steps.reward) # Rewards
        return_decision.append(np.zeros_like(decision_steps.reward)) # Non terminal

    else:
        return_decision = [np.array([]), np.zeros((0, observation_size), dtype=np.float32), np.array([]), np.array([])] 

    # --- TERMINAL STEPS ---
    return_terminal = []
    
    if len(terminal_steps.agent_id) > 0:
        ray_state = organize_observations_batch(terminal_steps.obs[0], n_tag)
        internal_state = terminal_steps.obs[1][:, 1:]

        return_terminal.append(terminal_steps.obs[1][:,0]) # ID speciali
        return_terminal.append(np.concatenate([ray_state, internal_state], axis=1)) # Obs Batch
        return_terminal.append(terminal_steps.reward) # Rewards
        return_terminal.append(~terminal_steps.interrupted) # Terminal
        
    else:
        return_terminal = [np.array([]), np.zeros((0, observation_size), dtype=np.float32), np.array([]), np.array([])]

    # id speciali, obs, reward, terminal
    return return_decision, return_terminal

def observe_batch_stacked(env, BEHAVIOUR_NAME, input_stack, observation_size):    
    decision_obs, terminal_obs = observe_batch(env, BEHAVIOUR_NAME, observation_size)
    
    if not hasattr(observe_batch_stacked, "memory"):
        observe_batch_stacked.memory = [np.array([]), None]
    
    # 1. Inserimento nuovi Agenti
    stored_ids = observe_batch_stacked.memory[0]
    stored_data = observe_batch_stacked.memory[1]
        
    current_ids = decision_obs[0]
    current_data = decision_obs[1] 
    
    is_new = ~np.isin(current_ids, stored_ids)
    
    if np.any(is_new):
        new_ids_to_add = current_ids[is_new]
        new_rows_to_add = current_data[is_new]
        
        # Stack iniziale: ripetiamo il primo frame input_stack volte
        stacked_new_rows = np.tile(new_rows_to_add, input_stack)
        
        if stored_data is None:
            observe_batch_stacked.memory[0] = new_ids_to_add
            observe_batch_stacked.memory[1] = stacked_new_rows
        else:
            observe_batch_stacked.memory[0] = np.concatenate((stored_ids, new_ids_to_add))
            observe_batch_stacked.memory[1] = np.vstack((stored_data, stacked_new_rows))

    # 2. Aggiornamento Stack e Memoria (Decision Steps)
    # Rileggiamo i riferimenti aggiornati
    stored_ids = observe_batch_stacked.memory[0]
    stored_data = observe_batch_stacked.memory[1]

    if len(current_ids) > 0:
        obs_dim = current_data.shape[1]
        
        sorter = np.argsort(stored_ids)
        idxs = sorter[np.searchsorted(stored_ids, current_ids, sorter=sorter)]
        
        prev_stack = stored_data[idxs]
        
        # [Frame_T-2, Frame_T-1, Frame_T] -> [Frame_T-1, Frame_T, New_Frame]
        new_stack = np.concatenate((prev_stack[:, obs_dim:], current_data), axis=1)
        
        observe_batch_stacked.memory[1][idxs] = new_stack
        decision_obs[1] = new_stack

    # 3. Gestione Terminal Steps (Output + Pulizia Memoria)
    term_ids = terminal_obs[0]
    term_data = terminal_obs[1]
    
    if len(term_ids) > 0:
        obs_dim = term_data.shape[1]
        
        # A. Costruiamo l'output per il training (ci serve ancora la memoria vecchia)
        sorter_t = np.argsort(stored_ids)
        idxs_t = sorter_t[np.searchsorted(stored_ids, term_ids, sorter=sorter_t)]
        
        prev_stack_t = stored_data[idxs_t]
        final_stack = np.concatenate((prev_stack_t[:, obs_dim:], term_data), axis=1)
        terminal_obs[1] = final_stack
        

        keep_mask = ~np.isin(stored_ids, term_ids)
        observe_batch_stacked.memory[0] = stored_ids[keep_mask]
        observe_batch_stacked.memory[1] = stored_data[keep_mask]

    return decision_obs, terminal_obs # (id, obs, reward, termination)

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
