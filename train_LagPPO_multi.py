import argparse
import sys
import time
import random
import traceback
import math
import itertools
import os
from collections import deque, defaultdict
from pprint import pprint
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

import wandb
from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
from gymnasium import spaces 

# Assumo che training_utils contenga le tue funzioni helper
from training_utils import *


# ==============================================================================
# MAIN TRAIN FUNCTION
# ==============================================================================

def train_ppo(args, agent_config, obstacles_config, other_config):
    
    args.seed = random.randint(0, 2**16)
    print('Training PPO with the following parameters:')
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

    # Env setup
    print(f'Starting Unity Environment from build: {args.build_path}')
    env = UnityEnvironment(args.build_path, 
                           seed=args.seed, 
                           side_channels=[env_info, param_channel], 
                           no_graphics=args.headless,
                           worker_id=args.worker_id)
    print('Unity Environment connected.')
    print('Resetting environment...')
    env.reset()

    # Logging
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

    # Config Extraction
    BEHAVIOUR_NAME = other_config['behavior_name'] + '?team=' + other_config['team']
    RAY_PER_DIRECTION = other_config['rays_per_direction']
    RAYCAST_SIZE = 2*RAY_PER_DIRECTION + 1
    STATE_SIZE = other_config['state_observation_size'] - 1
    ACTION_SIZE = other_config['action_size']
    ACTION_MIN = other_config['min_action']
    ACTION_MAX = other_config['max_action']
    TOTAL_STATE_SIZE = (STATE_SIZE + RAYCAST_SIZE)*args.input_stack

    # Creating PPO Agent
    print('Creating PPO Agent...')
    agent = LagPPOAgent(TOTAL_STATE_SIZE, ACTION_SIZE, ACTION_MIN, ACTION_MAX).to(DEVICE)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ==========================================================================
    # LAGRANGIAN SETUP (SAC STYLE)
    # ==========================================================================
    # Lettura sicura dei parametri (usa default se non presenti in args)
    init_lambda_val = getattr(args, 'init_lambda', 0.01)
    lr_lagrange = getattr(args, 'lambda_lr', 0.05) 
    target_cost = getattr(args, 'cost_limit', 10.0)
    d_safe_val = getattr(args, 'd_safe', 0.3)
    cost_vf_coef = getattr(args, 'cost_vf_coef', 0.5)

    # Parametro ottimizzabile in Log-Space
    log_lagrange = torch.tensor(np.log(init_lambda_val), requires_grad=True, device=DEVICE)
    lagrange_optimizer = optim.Adam([log_lagrange], lr=lr_lagrange)

    # Buffer PPO
    ppo_buffer = LagPPORolloutBuffer(DEVICE, gamma=args.gamma, gae_lambda=args.gae_lambda)

    # Training Loop variables
    save_path = './models/' + run_name
    os.makedirs(save_path, exist_ok=True)
    
    epoch_stats = defaultdict(list)
    recent_rewards = deque(maxlen=50)
    recent_lengths = deque(maxlen=50)

    start_time = time.time()
    best_reward = -float('inf')
    
    global_step = 0
    iteration = 0

    print(f'[{global_step}/{args.total_timesteps}] Starting Training - run name: {run_name}')

    try:
        # PRIMO COLLECT
        obs = collect_data_after_step(env, BEHAVIOUR_NAME, STATE_SIZE)
        compute_lag_cost(obs, RAYCAST_SIZE, args.input_stack, d_safe_val, other_config['raycast_length'])
        
        while global_step < args.total_timesteps:
            
            # ==================================================================
            # 1. DATA COLLECTION PHASE (ROLLOUT)
            # ==================================================================
            step_data_cache = {} 

            # Actions loop
            for id in obs:
                agent_obs = obs[id]
                if agent_obs[3]: continue
                
                with torch.no_grad():
                    obs_tensor = torch.from_numpy(agent_obs[0]).float().unsqueeze(0).to(DEVICE)
                    
                    if torch.isnan(obs_tensor).any() or torch.isinf(obs_tensor).any():
                        print(f"⚠️ [WARNING] NaN/Inf detected for Agent {id}")
                        obs_tensor = torch.nan_to_num(obs_tensor, nan=0.0)
                        
                    raw_action, scaled_action, logprob, _, value = agent.get_action_and_value(obs_tensor)
                    cost_val_est = agent.get_cost_value(obs_tensor) # Stima costo corrente
                
                step_data_cache[id] = {
                    'raw_action': raw_action,     
                    'logprob': logprob,           
                    'value': value,               
                    'cost_value': cost_val_est.item(), 
                    'state': obs_tensor           
                }

                action_np = scaled_action.cpu().numpy()[0]
                agent_obs[2] = action_np
                a = ActionTuple(continuous=np.array([action_np]))
                env.set_action_for_agent(BEHAVIOUR_NAME, id, a)

            # --- ENVIRONMENT STEP ---
            env.step()

            # NEXT OBS
            next_obs = collect_data_after_step(env, BEHAVIOUR_NAME, STATE_SIZE)
            compute_lag_cost(next_obs, RAYCAST_SIZE, args.input_stack, d_safe_val, other_config['raycast_length'])
            
            # Stats processing
            while env_info.stop_msg_queue:
                msg = env_info.stop_msg_queue.pop()
                for key, value in msg.items():
                    epoch_stats[key].append(value)
                    if key == 'reward': recent_rewards.append(value)
                    if key == 'episode_length': recent_lengths.append(value)

            # --- SAVE DATA TO PPO BUFFER ---
            count_new_steps = 0
            for id in obs:
                prev_agent_obs = obs[id]
                if prev_agent_obs[3] or id not in next_obs: continue
                if id not in step_data_cache: continue
                
                cached = step_data_cache[id]
                next_agent_obs = next_obs[id]
                
                # Aggiungiamo al buffer (Notare: usiamo cost_value dalla cache)
                ppo_buffer.add(
                    id=id,
                    state=cached['state'][0],       
                    raw_action=cached['raw_action'][0], 
                    logprob=cached['logprob'],
                    reward=next_agent_obs[1],
                    cost=next_agent_obs[4],     # Costo reale ottenuto
                    done=next_agent_obs[3],
                    value=cached['value'],
                    cost_value=cached['cost_value']
                )
                count_new_steps += 1

            obs = next_obs
            global_step += count_new_steps

            # ==================================================================
            # 2. PPO UPDATE PHASE
            # ==================================================================
            
            if ppo_buffer.total_steps >= args.num_steps:
                iteration += 1
                
                # Annealing LR
                if args.anneal_lr:
                    frac = 1.0 - (global_step - 1.0) / args.total_timesteps
                    lrnow = frac * args.learning_rate
                    optimizer.param_groups[0]["lr"] = lrnow

                # Calcolo Last Values per GAE
                last_values = {}
                last_costs = {} 
                with torch.no_grad():
                    for id in obs:
                        agent_obs = obs[id]
                        obs_tensor = torch.from_numpy(agent_obs[0]).float().unsqueeze(0).to(DEVICE)
                        obs_tensor = torch.nan_to_num(obs_tensor, nan=0.0)
                        
                        last_values[id] = agent.get_value(obs_tensor).item()
                        last_costs[id] = agent.get_cost_value(obs_tensor).item()

                # 1. Prendi TUTTO il batch (Usa la funzione corretta con adv_r/adv_c)
                b_obs, b_raw_actions, b_logprobs, \
                b_adv_r, b_adv_c, b_ret_r, b_ret_c, b_values, b_cost_values = \
                    ppo_buffer.get_full_batch(last_value_estimates=last_values, last_cost_estimates=last_costs)

                # 2. Normalizzazione Vantaggi
                b_adv_r = (b_adv_r - b_adv_r.mean()) / (b_adv_r.std() + 1e-8)
                b_adv_c = (b_adv_c - b_adv_c.mean()) / (b_adv_c.std() + 1e-8)

                # Flatten batch indices
                batch_size_total = b_obs.shape[0]
                b_inds = np.arange(batch_size_total)
                
                clipfracs = []
                
                # PPO Epochs
                for epoch in range(args.update_epochs):
                    np.random.shuffle(b_inds)
                    
                    for start in range(0, batch_size_total, args.batch_size):
                        end = start + args.batch_size
                        mb_inds = b_inds[start:end]
                        if len(mb_inds) < 2: continue

                        # Forward pass
                        _, _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], action=b_raw_actions[mb_inds])
                        new_cost_value = agent.get_cost_value(b_obs[mb_inds]).view(-1)

                        logratio = newlogprob - b_logprobs[mb_inds]
                        ratio = logratio.exp()

                        with torch.no_grad():
                            approx_kl = ((ratio - 1) - logratio).mean()
                            clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                        # --- POLICY LOSS (REWARD) ---
                        mb_adv = b_adv_r[mb_inds]
                        reward_loss1 = -mb_adv * ratio
                        reward_loss2 = -mb_adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                        reward_loss = torch.max(reward_loss1, reward_loss2).mean()

                        # --- COST PENALTY (LAGRANGIAN) ---
                        # Recupera lambda attuale (dal log-space)
                        lagrange_multiplier = log_lagrange.exp()
                        
                        mb_cost_adv = b_adv_c[mb_inds]
                        cost_surr1 = mb_cost_adv * ratio
                        cost_surr2 = mb_cost_adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                        cost_surrogate_loss = torch.max(cost_surr1, cost_surr2).mean()

                        # --- TOTAL POLICY LOSS ---
                        # Usiamo .detach() su lambda perché qui alleniamo solo l'agente
                        pg_loss = reward_loss + lagrange_multiplier.detach() * cost_surrogate_loss

                        # --- VALUE LOSSES ---
                        newvalue = newvalue.view(-1)
                        v_loss = 0.5 * ((newvalue - b_ret_r[mb_inds]) ** 2).mean()

                        cost_v_loss = 0.5 * ((new_cost_value - b_ret_c[mb_inds]) ** 2).mean()
                        entropy_loss = entropy.mean()

                        # --- AGENT UPDATE ---
                        loss = (
                            pg_loss
                            - args.ent_coef * entropy_loss
                            + args.vf_coef * v_loss
                            + cost_vf_coef * cost_v_loss
                        )

                        optimizer.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                        optimizer.step()

                        # --- LAMBDA UPDATE (SAC STYLE) ---
                        # Aggiornamento separato per lambda
                        # Loss = - lambda * (Cost - Limit)
                        # Usiamo il costo medio del minibatch (Return) come proxy
                        mean_batch_cost = b_ret_c[mb_inds].mean()
                        
                        lambda_loss = -(lagrange_multiplier * (mean_batch_cost.detach() - target_cost))
                        
                        lagrange_optimizer.zero_grad()
                        lambda_loss.backward()
                        lagrange_optimizer.step()
                        
                        # Clamp Log-Lambda per stabilità
                        with torch.no_grad():
                            log_lagrange.clamp_(min=-15.0, max=10.0)

                # --- FINE UPDATE LOOP: METRICHE ---
                
                # Calcolo Varianza Spiegata
                y_pred, y_true = b_values.cpu().numpy(), b_ret_r.cpu().numpy()
                var_y = np.var(y_true)
                explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

                training_losses = {
                    'value_loss': v_loss.item(),
                    'policy_loss': pg_loss.item(),
                    'entropy': entropy_loss.item(),
                    'approx_kl': approx_kl.item(),
                    'clipfrac': np.mean(clipfracs),
                    'explained_variance': explained_var,
                    'cost_value_loss': cost_v_loss.item(),
                    # Lag Metrics
                    'lagrangian_lambda': lagrange_multiplier.item(),
                    'lambda_loss': lambda_loss.item(),
                    'log_lagrange': log_lagrange.item()
                }

                # Episodic Metrics
                episodic_metrics = {}
                n_episodes_batch = 0
                if len(epoch_stats) > 0:
                    n_episodes_batch = len(next(iter(epoch_stats.values())))
                    for k, v_list in epoch_stats.items():
                        episodic_metrics[f'batch_{k}'] = np.mean(v_list)
                
                if len(recent_rewards) > 0:
                    episodic_metrics['rolling_reward'] = np.mean(recent_rewards)
                else:
                    episodic_metrics['rolling_reward'] = -float('inf')
                episodic_metrics['episodes_per_batch'] = n_episodes_batch
                
                # Tech Metrics
                tech_metrics = {
                    'learning_rate': optimizer.param_groups[0]["lr"],
                    'SPS': int(global_step / (time.time() - start_time))
                }

                # Console Print
                roll_rew = episodic_metrics['rolling_reward']
                lambda_val = lagrange_multiplier.item()
                print(f"[{global_step}] Loss: {loss.item():.3f} | Roll Rew: {roll_rew:.2f} | Lambda: {lambda_val:.4f}")
                
                # WandB Log
                if args.wandb:
                    log_stats_to_wandb(
                        wandb_run, 
                        [training_losses, tech_metrics, episodic_metrics], 
                        ['losses', 'charts', 'episodic'], 
                        global_step
                    )
                
                # Reset Buffer & Stats
                ppo_buffer.reset()
                epoch_stats = defaultdict(list)

                # Save Checkpoint
                current_reward = episodic_metrics['rolling_reward']
                if current_reward > best_reward and len(recent_rewards) > 10: 
                    best_reward = current_reward
                    print(f"New best reward: {best_reward:.2f}, model saved")
                    save_models_simple(agent, save_path, suffix='_best')
                
                save_models_simple(agent, save_path, suffix='_final')

    except Exception as e:  
        print(f"[{global_step}/{args.total_timesteps}] An error occurred: {e}")
        traceback.print_exc()

    print("Closing environment")
    env.close()
    
    if args.wandb:
        print("Closing wandb run")
        wandb.finish()

    save_models_simple(agent, save_path, suffix='_final')
    print("Training Complete.")
    
    
# ==============================================================================
# ENTRY POINT
# ==============================================================================

args = parse_args()
agent_config = parse_config_file(args.agent_config_path)
obstacles_config = parse_config_file(args.obstacles_config_path)
other_config = parse_config_file(args.other_config_path)

if torch.cuda.is_available() and args.cuda >= 0:
    device_str = f"cuda:{args.cuda}"
else:
    device_str = "cpu"

DEVICE = torch.device(device_str)
print(f"Using device: {DEVICE}")

train_ppo(args, agent_config, obstacles_config, other_config)
