import argparse
import sys
import time
import random
import traceback
from collections import deque
from pprint import pprint
from sympy import true
import wandb
import numpy as np

import torch.optim as optim
from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel

from gymnasium import spaces 
from stable_baselines3.common.buffers import ReplayBuffer

from training_utils import *

import itertools
from copy import deepcopy
import random
import math

from types import SimpleNamespace # <--- Aggiungi questo import se manca


def sample_log_uniform(a, b):
    return 10 ** random.uniform(math.log10(a), math.log10(b))

def sample_hparams():
    # Definiamo prima il base_lr per coerenza degli altri parametri
    base_lr = sample_log_uniform(1e-5, 3e-3)

    return {
        "lr": base_lr,
        "tau": random.uniform(0.003, 0.025),
        "target_entropy": random.uniform(-4.0, -1.5),
        "policy_frequency": random.choice([1, 2, 3]),
        "batch_size": random.choice([128, 256, 512]),
        "gamma": random.uniform(0.975, 0.999),

        "safety_threshold": random.uniform(1, 12),
    }

class CostReplayBuffer(ReplayBuffer):
    """
    Estensione del ReplayBuffer di SB3 per gestire i costi (Safety).
    Salva i costi passati tramite 'infos' e li restituisce nel batch.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Inizializziamo il buffer dei costi a zero
        self.costs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

    def add(self, obs, next_obs, action, reward, done, infos):
        # Estraiamo il costo dal dizionario infos. Default a 0.0 se non presente.
        # reshape((self.n_envs,)) serve perché il buffer si aspetta (batch, envs)
        cost_array = np.array([info.get('cost', 0.0) for info in infos])
        self.costs[self.pos] = cost_array.reshape((self.n_envs,))
        
        super().add(obs, next_obs, action, reward, done, infos)

    def _get_samples(self, batch_inds, env=None):
        # 1. Otteniamo i campioni standard (che sono una NamedTuple immutabile)
        samples = super()._get_samples(batch_inds, env)
        
        # 2. Recuperiamo i costi corrispondenti agli indici
        batch_costs = self.costs[batch_inds]
        
        # 3. FIX CRITICO: Convertiamo la NamedTuple in un dizionario per poter aggiungere dati
        data = samples._asdict()
        
        # 4. Aggiungiamo i costi come tensore PyTorch
        data['costs'] = torch.tensor(batch_costs, dtype=torch.float32).to(self.device)
        
        # 5. Restituiamo un SimpleNamespace (che permette l'accesso col punto: data.costs)
        return SimpleNamespace(**data)

def compute_lag_cost(obs, RAYCAST_SIZE, STACKS, d_safe, ray_len):
    
    for id in obs:
        closest_obj_dist = min(obs[id][0][RAYCAST_SIZE * (STACKS - 1): RAYCAST_SIZE*STACKS])
        closest_obj_real_dist = closest_obj_dist*ray_len # 3 meters raycast to 0-1
        
        obs[id].append(max(0, (d_safe - closest_obj_real_dist)/d_safe ))
    
def train(args, agent_config, obstacles_config, other_config):
    
    args.seed = random.randint(0, 2**16)

    print('Training with the following parameters:')
    pprint(vars(args))

    # seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    print(f'Seed: {args.seed}')

    # Start Environment
    env_info = CustomChannel()
    param_channel = EnvironmentParametersChannel()

    print('Applying Unity settings from config...')
    apply_unity_settings(param_channel, agent_config, 'ag_')
    apply_unity_settings(param_channel, obstacles_config, 'obs_')

    if args.test_lib:
        print('Testing Ended')
        exit(0)

    # env setup
    print(f'Starting Unity Environment from build: {args.build_path}')
    env = UnityEnvironment(args.build_path, 
                        seed=args.seed, 
                        side_channels=[env_info, param_channel], 
                        no_graphics=args.headless,
                        worker_id=args.worker_id)
    print('Unity Environment connected.')
    print('Resetting environment...')
    env.reset()

    # Environment Variables and Log
    run_name = f"{args.exp_name}_{int(time.time()) - args.base_time}"
    args.run_name = run_name
    print(f"Run name: {run_name}")

    if args.wandb:
        print('Setting up wandb experiment tracking.')
        wandb_run = wandb.init(
            entity="giacomo-aru",
            project="UARSL_NEXT",
            name=args.run_name,
            config={
                "training": vars(args),
                "agent": agent_config,
                "obstacles": obstacles_config,
                "other": other_config
            }
        )

    BEHAVIOUR_NAME = other_config['behavior_name'] + '?team=' + other_config['team']
    RAY_PER_DIRECTION = other_config['rays_per_direction']
    RAYCAST_MIN = other_config['rays_min_observation']
    RAYCAST_MAX = other_config['rays_max_observation']
    RAYCAST_SIZE = 2*RAY_PER_DIRECTION + 1

    STATE_SIZE = other_config['state_observation_size'] - 1
    STATE_MIN = other_config['state_min_observation']
    STATE_MAX = other_config['state_max_observation']

    ACTION_SIZE = other_config['action_size']
    ACTION_MIN = other_config['min_action']
    ACTION_MAX = other_config['max_action']

    TOTAL_STATE_SIZE = (STATE_SIZE + RAYCAST_SIZE)*args.input_stack

    # --- NETWORKS SETUP ---
    print('Creating actor and critic networks...')
    # Actor (Policy)
    actor = OldDenseActor(TOTAL_STATE_SIZE, ACTION_SIZE, ACTION_MIN, ACTION_MAX, args.actor_network_layers).to(DEVICE)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # Reward Critics (Standard SAC)
    qf_ensemble = [OldDenseSoftQNetwork(TOTAL_STATE_SIZE, ACTION_SIZE, args.q_network_layers).to(DEVICE) for _ in range(args.q_ensemble_n)]
    qf_ensemble_target = [OldDenseSoftQNetwork(TOTAL_STATE_SIZE, ACTION_SIZE, args.q_network_layers).to(DEVICE) for _ in range(args.q_ensemble_n)]
    for q_t, q in zip(qf_ensemble_target, qf_ensemble):
        q_t.load_state_dict(q.state_dict())
    
    qf_optimizer = optim.Adam(itertools.chain(*[q.parameters() for q in qf_ensemble]), lr=args.q_lr)

    # [NEW] Cost Critics (Per stimare i costi futuri - Lagrangian SAC)
    cf_ensemble = [OldDenseSoftQNetwork(TOTAL_STATE_SIZE, ACTION_SIZE, args.q_network_layers).to(DEVICE) for _ in range(args.q_ensemble_n)]
    cf_ensemble_target = [OldDenseSoftQNetwork(TOTAL_STATE_SIZE, ACTION_SIZE, args.q_network_layers).to(DEVICE) for _ in range(args.q_ensemble_n)]
    for c_t, c in zip(cf_ensemble_target, cf_ensemble):
        c_t.load_state_dict(c.state_dict())

    cf_optimizer = optim.Adam(itertools.chain(*[c.parameters() for c in cf_ensemble]), lr=args.q_lr)

    # [NEW] Lagrangian Multiplier (Lambda)
    # Usiamo il logaritmo per garantire che lambda sia sempre > 0
    target_cost = args.safety_threshold
    log_lagrange = torch.tensor(np.log(args.init_lambda), requires_grad=True, device=DEVICE)
    lagrange_optimizer = optim.Adam([log_lagrange], lr=args.lr_lagrange)

    # --- REPLAY BUFFER SETUP ---
    print('Setting up replay buffer...')
    observation_space = spaces.Box(low=min(RAYCAST_MIN, STATE_MIN), high=max(RAYCAST_MAX, STATE_MAX), shape=(TOTAL_STATE_SIZE,), dtype=np.float32)
    action_space = spaces.Box(low=ACTION_MIN, high=ACTION_MAX, shape=(ACTION_SIZE,), dtype=np.float32)

    # [MOD] Usiamo la nostra classe Custom CostReplayBuffer
    rb = CostReplayBuffer(
        buffer_size=args.buffer_size,
        observation_space=observation_space,
        action_space=action_space,
        device=DEVICE,                
        handle_timeout_termination=True,
        n_envs=1 
    )

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = args.target_entropy
        log_alpha = torch.zeros(1, requires_grad=True, device=DEVICE)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.alpha_lr)
        print(f'autotune target_entropy: {target_entropy}')
    else:
        alpha = args.alpha

    # Setup Logging
    save_path = './models/' + run_name
    os.makedirs(save_path, exist_ok=True)
    print('saving to path:', save_path)

    training_stats = {
        "time/python_time": RunningMean(),
        "time/unity_time": RunningMean(),
        "stats/action_saturation": RunningMean(),
        "stats/qf_mean": RunningMean(),
        "stats/qf_std":RunningMean(),
        "stats/actor_entropy": RunningMean(),
        "stats/alpha": RunningMean(),
        "stats/uncertainty": RunningMean(),
        "loss/critic_ens": RunningMean(),
        "loss/actor": RunningMean(),
        "loss/alpha": RunningMean(),
        # [NEW] Stats per Lagrangian
        "stats/cost_mean": RunningMean(),       # Costo medio stimato
        "stats/lambda": RunningMean(),          # Valore del moltiplicatore
        "loss/cost_critic": RunningMean(),      # Loss dei critici del costo
        "loss/lagrange": RunningMean()          # Loss del moltiplicatore
    }

    best_reward = -float('inf')
    best_success = -float('inf')
    episodic_stats, success_stats, failure_stats = {}, {}, {}

    start_time = time.time()
    unity_end_time = -1
    global_step = 0
    print(f'[{global_step}/{args.total_timesteps}] Starting Training - run name: {run_name}')

    try:
        # Initial Collection
        obs = collect_data_after_step(env, BEHAVIOUR_NAME, STATE_SIZE)
        compute_lag_cost(obs, RAYCAST_SIZE, args.input_stack, args.d_safe, other_config['raycast_length'])
        
        while global_step < args.total_timesteps:

            # --- ACTION SELECTION ---
            for id in obs:
                agent_obs = obs[id]
                if agent_obs[3]: continue
                
                if global_step < args.learning_starts * 2:
                    action = get_initial_action(id)
                else:
                    # Se agent_obs[0] è un array numpy (molto comune):
                    obs_tensor = torch.from_numpy(agent_obs[0]).float().unsqueeze(0).to(DEVICE)
                    action, _, _, _, _ = actor.get_action(obs_tensor)
                    action = action[0].detach().cpu().numpy()
                
                agent_obs[2] = action
                a = ActionTuple(continuous=np.array([action]))
                env.set_action_for_agent(BEHAVIOUR_NAME, id, a)
            
            # --- ENVIRONMENT STEP ---
            unity_start_time = time.time()
            if unity_end_time > 0 and global_step > args.learning_starts:
                training_stats['time/python_time'].update(unity_start_time - unity_end_time)
            
            env.step()
            unity_end_time = time.time()
            if global_step > args.learning_starts:
                training_stats['time/unity_time'].update(unity_end_time - unity_start_time)

            next_obs = collect_data_after_step(env, BEHAVIOUR_NAME, STATE_SIZE)
            compute_lag_cost(next_obs, RAYCAST_SIZE, args.input_stack, args.d_safe, other_config['raycast_length'])
            
            # --- LOGGING MESSAGES ---
            while env_info.stop_msg_queue:
                msg = env_info.stop_msg_queue.pop()
                if global_step >= args.learning_starts:
                    update_stats_from_message(episodic_stats, success_stats, failure_stats, msg, args.metrics_smoothing)        
                    if episodic_stats['ep_count'] % args.metrics_log_interval == 0:
                        print_update(global_step, args.total_timesteps, start_time, episodic_stats)
                        if args.wandb:
                            log_stats_to_wandb(wandb_run, [episodic_stats, success_stats, failure_stats], ['all_ep', 'success_ep', 'failure_ep'], global_step)
                            print(f"[{global_step}/{args.total_timesteps}] Logged episodic stats to wandb")
                            
            # --- DATA STORAGE (MODIFICATO PER I COSTI) ---
            for id in obs:
                prev_agent_obs = obs[id]
                if prev_agent_obs[3] or id not in next_obs: continue
                next_agent_obs = next_obs[id]
                
                # [MOD] Estrazione del COSTO
                # Assumiamo che collect_data_after_step ora restituisca il costo all'indice 4
                # Struttura presunta: [obs, reward, action, done, cost]
                current_cost = next_agent_obs[4]

                rb.add(obs = prev_agent_obs[0], 
                    next_obs = next_agent_obs[0],
                    action = np.array(prev_agent_obs[2]), 
                    reward = next_agent_obs[1], 
                    done = next_agent_obs[3],
                    infos = [{"cost": current_cost}]) # [MOD] Passiamo il costo negli info
                
            obs = next_obs
            
            # Save best models
            if episodic_stats != {} and episodic_stats["reward"] > best_reward:
                best_reward = episodic_stats["reward"]
                best_success = episodic_stats["success"]
                save_models(actor, qf_ensemble, qf_ensemble_target, save_path, suffix=f'_best')
                print(f"[{global_step}/{args.total_timesteps}] Models saved, suffix: _best")
                    
            # --- TRAINING LOOP (LAGRANGIAN SAC) ---
            for _ in range(args.update_frequency):
                if global_step > args.learning_starts:

                    # 1. SAMPLE BATCH (Includes Costs)
                    data = rb.sample(args.batch_size)
                    
                    # [NEW] Recuperiamo i costi dal buffer custom
                    # data.costs è stato aggiunto dalla nostra classe CostReplayBuffer
                    costs = data.costs 

                    saturation = data.actions.detach().cpu().numpy()
                    saturation = (np.abs(saturation) > 0.99).mean()
                    training_stats["stats/action_saturation"].update(saturation)
                        
                    with torch.no_grad():
                        next_action, next_log_pi, _, _, _ = actor.get_action(data.next_observations)
                        # if args.noise_clip > 0:
                        #    noise = torch.randn_like(next_action) * args.noise_clip
                        #    next_action = torch.clamp(next_action + noise, -1, 1)

                        # --- TARGET UPDATE PER REWARD (Standard SAC) ---
                        target_q_values = [q_t(data.next_observations, next_action) for q_t in qf_ensemble_target]
                        min_qf_next_target = torch.stack(target_q_values).min(dim=0).values - alpha * next_log_pi
                        next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * min_qf_next_target.view(-1)

                        # --- [NEW] TARGET UPDATE PER COST (Safety) ---
                        target_c_values = [c_t(data.next_observations, next_action) for c_t in cf_ensemble_target]
                        # IMPORTANTE: Per la safety prendiamo il MAX (pessimistico) invece del MIN
                        max_cf_next_target = torch.stack(target_c_values).max(dim=0).values 
                        # Equazione di Bellman per il costo (senza termine entropico)
                        next_c_value = costs.flatten() + (1 - data.dones.flatten()) * args.gamma * max_cf_next_target.view(-1)

                    # Q-function updates (with bootstrapping)
                    q_losses = []
                    q_vals = []
                    batch_size = int(data.actions.shape[0] * args.bootstrap_batch_proportion)
                    for q in qf_ensemble:
                        # Bootstrap indices
                        indices = torch.randint(0, batch_size, (batch_size,), device=data.actions.device)
                        
                        observation = data.observations[indices]
                        actions = data.actions[indices]
                        target = next_q_value[indices]

                        # Compute Q loss
                        q_val = q(observation, actions).view(-1)
                        loss = F.mse_loss(q_val, target)
                        q_losses.append(loss)
                        q_vals.append(q_val)
                        
                    total_q_loss = torch.stack(q_losses).mean()
                    qf_optimizer.zero_grad()
                    torch.nn.utils.clip_grad_norm_(itertools.chain(*[q.parameters() for q in qf_ensemble]), max_norm=1.0)
                    total_q_loss.backward()
                    qf_optimizer.step()
                    
                    # Track Q-value statistics
                    all_q_values = torch.cat(q_vals)
                    training_stats['stats/qf_mean'].update(all_q_values.mean().item())
                    training_stats['stats/qf_std'].update(all_q_values.std().item())
                    training_stats['loss/critic_ens'].update(total_q_loss.item())

                    # --- [NEW] UPDATE COST CRITICS ---
                    # Q-function updates (with bootstrapping)
                    c_losses = []
                    batch_size = int(data.actions.shape[0] * args.bootstrap_batch_proportion)
                    for c in cf_ensemble:
                        # Bootstrap indices
                        indices = torch.randint(0, batch_size, (batch_size,), device=data.actions.device)
                        
                        observation = data.observations[indices]
                        actions = data.actions[indices]
                        target = next_c_value[indices]

                        # Compute Q loss
                        c_val = c(observation, actions).view(-1)
                        loss = F.mse_loss(c_val, target)
                        c_losses.append(loss)
                        
                    total_c_loss = torch.stack(c_losses).mean()
                    cf_optimizer.zero_grad()
                    total_c_loss.backward()
                    torch.nn.utils.clip_grad_norm_(itertools.chain(*[c.parameters() for c in cf_ensemble]), max_norm=1.0)
                    cf_optimizer.step()
                    
                    training_stats['loss/cost_critic'].update(total_c_loss.item())
                    
                    # --- DELAYED ACTOR UPDATE ---
                    if global_step % args.policy_frequency == 0:
                        for _ in range(args.policy_frequency):
                            pi, log_pi, _, _, _ = actor.get_action(data.observations)
                            
                            # 1. Calcolo Q-values per la policy attuale
                            q_pi_vals = [q(data.observations, pi) for q in qf_ensemble]
                            min_qf_pi = torch.min(torch.stack(q_pi_vals), dim=0).values.view(-1)

                            # 2. [NEW] Calcolo Cost-values per la policy attuale
                            c_pi_vals = [c(data.observations, pi) for c in cf_ensemble]
                            max_cf_pi = torch.max(torch.stack(c_pi_vals), dim=0).values.view(-1)

                            # 3. [NEW] Recupero Moltiplicatore Lagrange
                            lagrange_multiplier = log_lagrange.exp()

                            # 4. [MOD] LOSS ATTORE LAGRANGIANA
                            # Loss = Alpha*Entropy - Reward + Lambda * (Cost - Limit)
                            # Se Cost > Limit, Lambda aumenta la penalità.
                            actor_loss = ((alpha * log_pi) - min_qf_pi + lagrange_multiplier.detach() * (max_cf_pi - target_cost)).mean()

                            actor_optimizer.zero_grad()
                            actor_loss.backward()
                            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
                            actor_optimizer.step()

                            # 5. [NEW] UPDATE LAGRANGE MULTIPLIER (LAMBDA)
                            # Vogliamo trovare lambda che massimizza la penalità (Dual Ascent) o minimizza l'errore del vincolo
                            # Loss = - lambda * (Cost - Limit)  => Aggiornamento gradiente per aumentare lambda se Cost > Limit
                            # Nota: usiamo .detach() su max_cf_pi perché non vogliamo aggiornare il critico qui
                            lambda_loss = -(lagrange_multiplier * (max_cf_pi.detach() - target_cost).mean())
                            
                            lagrange_optimizer.zero_grad()
                            lambda_loss.backward()
                            torch.nn.utils.clip_grad_norm_([log_lagrange], max_norm=0.5)
                            lagrange_optimizer.step()
                            
                            with torch.no_grad():
                                log_lagrange.clamp_(min=-12, max=4)

                            # --- STATS LOGGING ---
                            training_stats['loss/actor'].update(actor_loss.item())
                            training_stats['loss/lagrange'].update(lambda_loss.item())
                            training_stats['stats/cost_mean'].update(max_cf_pi.mean().item())
                            training_stats['stats/lambda'].update(lagrange_multiplier.item())
                            training_stats['stats/actor_entropy'].update(-log_pi.mean().item())
                            
                            # Uncertainty metric
                            with torch.no_grad():
                                uncertainty = torch.stack(q_pi_vals).std(dim=0).mean().item()
                            training_stats['stats/uncertainty'].update(uncertainty)

                            # Automatic entropy tuning (Standard SAC)
                            if args.autotune:
                                with torch.no_grad():
                                    _, log_pi, _, _, _ = actor.get_action(data.observations)
                                alpha_loss = (-log_alpha * (log_pi + target_entropy)).mean()
                                a_optimizer.zero_grad()
                                alpha_loss.backward()
                                a_optimizer.step()
                                alpha = log_alpha.exp().item()
                                training_stats['loss/alpha'].update(alpha_loss.item())
                            training_stats['stats/alpha'].update(alpha)
                            
                    # Soft update target networks (Both Reward and Cost)
                    if global_step % args.target_network_update_period == 0:
                        # Update Reward Critics
                        for q, q_t in zip(qf_ensemble, qf_ensemble_target):
                            for param, target_param in zip(q.parameters(), q_t.parameters()):
                                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                        # [NEW] Update Cost Critics
                        for c, c_t in zip(cf_ensemble, cf_ensemble_target):
                            for param, target_param in zip(c.parameters(), c_t.parameters()):
                                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

                    # LOGGING PERIODICO
                    if global_step % args.loss_log_interval == 0:
                        training_stats_divided = extract_and_reset_stats(training_stats, aggregations=['mean'])
                        current_time = time.time()
                        if 'time' not in training_stats_divided: training_stats_divided['time'] = {}
                        training_stats_divided['time']['SPS'] = global_step / (current_time - start_time + 1e-6)
                        
                        if args.wandb:
                            log_stats_to_wandb(wandb_run, list(training_stats_divided.values()), list(training_stats_divided.keys()), global_step)
                            print(f"[{global_step}/{args.total_timesteps}] Logged training stats to wandb")
                                
                elif global_step == args.learning_starts:
                    print("Start Learning")

                global_step += 1
                
    except Exception as e:  
        print(f"[{global_step}/{args.total_timesteps}] An error occurred: {e}")
        traceback.print_exc()

    print("Closing environment")
    env.close()
    
    if args.wandb:
        wandb.log({'best_reward': best_reward, 'best_success': best_success}, step=global_step)              
        print("Closing wandb run")
        wandb.finish()

    save_models(actor, qf_ensemble, qf_ensemble_target, save_path, suffix='_final')
    print(f"[{global_step}/{args.total_timesteps}] Models saved, suffix: _final")
    
args = parse_args()
agent_config = parse_config_file(args.agent_config_path)
obstacles_config = parse_config_file(args.obstacles_config_path)
other_config = parse_config_file(args.other_config_path)

if torch.cuda.is_available() and args.cuda >= 0:
    # F-string per inserire l'indice: diventa "cuda:2"
    device_str = f"cuda:{args.cuda}"
else:
    device_str = "cpu"

DEVICE = torch.device(device_str)
print(f"Using device: {DEVICE}")

run_id = 0
while run_id < 1000:

    combo = sample_hparams()

    print(f"\n=== RUN {run_id} ===")
    print(combo)

    # Parametri standard SAC
    args.policy_lr = combo['lr']
    args.q_lr = combo['lr']
    args.alpha_lr = combo['lr']
    
    args.tau = combo['tau']
    args.policy_frequency = combo['policy_frequency']
    args.target_entropy = combo['target_entropy']
    args.batch_size = combo['batch_size']
    args.gamma = combo['gamma']
    
    # Parametri LAGSAC
    args.safety_threshold = combo['safety_threshold']
    args.lr_lagrange = combo['lr']/3

    # Avvio del processo di training
    train(args, agent_config, obstacles_config, other_config)
    run_id += 1