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

    print('agent_config:')
    pprint(agent_config)

    print('obstacles_config:')
    pprint(obstacles_config)

    print('other_config:')
    pprint(other_config)
    
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
    agent = PPOAgent(TOTAL_STATE_SIZE, ACTION_SIZE, ACTION_MIN, ACTION_MAX).to(DEVICE)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Buffer PPO (Sostituisce ReplayBuffer)
    ppo_buffer = PPORolloutBuffer(DEVICE, gamma=args.gamma, gae_lambda=args.gae_lambda)

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
    # ... prima del while loop ...

    print(f'[{global_step}/{args.total_timesteps}] Starting Training - run name: {run_name}')

    try:
        # PRIMO COLLECT (identico al tuo codice)
        obs = collect_data_after_step(env, BEHAVIOUR_NAME, STATE_SIZE)
        
        while global_step < args.total_timesteps:
            
            # ==================================================================
            # 1. DATA COLLECTION PHASE (ROLLOUT)
            # Accumuliamo dati finché ppo_buffer.total_steps < args.num_steps
            # ==================================================================
            
            # Memorizziamo temporaneamente le azioni/logprobs/values calcolati in QUESTO step
            # per salvarli nel buffer DOPO aver fatto env.step() e ottenuto il reward
            step_data_cache = {} 

            # Actions loop
            for id in obs:
                agent_obs = obs[id]
                
                # Terminated agents check
                if agent_obs[3]:
                    continue
                
                # PPO Logic: Get Action + LogProb + Value
                # Non usiamo più get_initial_action randomica, PPO parte subito con la rete
                with torch.no_grad():
                    obs_tensor = torch.from_numpy(agent_obs[0]).float().unsqueeze(0).to(DEVICE)
                    
                    if torch.isnan(obs_tensor).any() or torch.isinf(obs_tensor).any():
                        print(f"⚠️ [WARNING] SPAZZATURA RILEVATA! Agent {id} ha mandato NaN/Inf nell'osservazione.")
                        obs_tensor = torch.nan_to_num(obs_tensor, nan=0.0, posinf=1.0, neginf=-1.0)
                        
                    raw_action, scaled_action, logprob, _, value = agent.get_action_and_value(obs_tensor)
                
                # Salviamo i dati per il buffer (IMPORTANTE: detach o .item())
                step_data_cache[id] = {
                    'raw_action': raw_action,     # Tensor
                    'logprob': logprob,           # Tensor
                    'value': value,               # Tensor
                    'state': obs_tensor           # Tensor
                }

                action_np = scaled_action.cpu().numpy()[0]
                
                # Memorizza azione per il prossimo step (logica tua)
                agent_obs[2] = action_np
                
                a = ActionTuple(continuous=np.array([action_np]))
                env.set_action_for_agent(BEHAVIOUR_NAME, id, a)

            # --- ENVIRONMENT STEP ---
            env.step()

            # NEXT OBS
            next_obs = collect_data_after_step(env, BEHAVIOUR_NAME, STATE_SIZE)
            
            # Stats processing (Side Channel)
            # Stats processing (Side Channel)
            while env_info.stop_msg_queue:
                msg = env_info.stop_msg_queue.pop()
                for key, value in msg.items():
                    epoch_stats[key].append(value)
                    
                    # --- AGGIUNTA: Aggiorna le medie mobili ---
                    if key == 'reward': # Assicurati che la chiave sia quella giusta dal tuo C#
                        recent_rewards.append(value)
                    if key == 'episode_length':
                        recent_lengths.append(value)

            # --- SAVE DATA TO PPO BUFFER ---
            # Qui usiamo ESATTAMENTE la tua logica di confronto prev_obs vs next_obs
            count_new_steps = 0
            for id in obs:
                prev_agent_obs = obs[id]
                
                # Se l'agente non c'è nel next_obs o era già terminato, saltiamo
                if prev_agent_obs[3] or id not in next_obs:
                    continue
                
                # Recuperiamo i dati calcolati PRIMA dello step
                if id not in step_data_cache:
                    continue
                
                cached = step_data_cache[id]
                next_agent_obs = next_obs[id]
                
                # Aggiungiamo al buffer PPO
                # Nota: usiamo i tensori raw, non numpy, per efficienza GPU
                ppo_buffer.add(
                    id=id,
                    state=cached['state'][0],       
                    raw_action=cached['raw_action'][0], # Salviamo azione Raw (Gaussian)
                    logprob=cached['logprob'],
                    reward=next_agent_obs[1],
                    done=next_agent_obs[3],
                    value=cached['value']
                )
                count_new_steps += 1

            # Update obs pointer
            obs = next_obs
            global_step += count_new_steps

            # ==================================================================
            # 2. PPO UPDATE PHASE
            # Se abbiamo raccolto abbastanza dati, facciamo l'update
            # ==================================================================
            
            if ppo_buffer.total_steps >= args.num_steps:
                iteration += 1
                print(f"[{global_step}] PPO Update triggered. Collected {ppo_buffer.total_steps} steps.")

                # Annealing LR
                if args.anneal_lr:
                    frac = 1.0 - (global_step - 1.0) / args.total_timesteps
                    lrnow = frac * args.learning_rate
                    optimizer.param_groups[0]["lr"] = lrnow

                # Calcolo Last Values
                last_values = {}
                with torch.no_grad():
                    for id in obs:
                        agent_obs = obs[id]
                        obs_tensor = torch.from_numpy(agent_obs[0]).float().unsqueeze(0).to(DEVICE)
                        obs_tensor = torch.nan_to_num(obs_tensor, nan=0.0)
                        value = agent.get_value(obs_tensor)
                        last_values[id] = value.item()


                # 1. Prendi TUTTO il batch
                b_obs, b_raw_actions, b_logprobs, b_advantages, b_returns, b_values = ppo_buffer.get_full_batch(last_value_estimates=last_values)

                # 2. Normalizzazione Vantaggi (CRITICO: Farlo qui, una volta sola)
                b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

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

                        # Forward pass su minibatch
                        # IMPORTANTE: Passiamo b_raw_actions (Gaussian) alla rete
                        _, _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], action=b_raw_actions[mb_inds])
                        
                        logratio = newlogprob - b_logprobs[mb_inds]
                        ratio = logratio.exp()

                        with torch.no_grad():
                            old_approx_kl = (-logratio).mean()
                            approx_kl = ((ratio - 1) - logratio).mean()
                            clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                        # Vantaggi già normalizzati
                        mb_advantages = b_advantages[mb_inds]
                        
                        # Policy Loss
                        pg_loss1 = -mb_advantages * ratio
                        pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                        # Value Loss
                        newvalue = newvalue.view(-1)
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                        # Entropy Loss
                        entropy_loss = entropy.mean()

                        # Total Loss
                        loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                        # Optimization
                        optimizer.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                        optimizer.step()
                    
                # --- FINE UPDATE LOOP: CALCOLO METRICHE & LOGGING ---
                # (Questo blocco deve stare FUORI dai cicli for epoch/start, ma DENTRO l'if ppo_buffer...)
                
                # Calcolo Varianza Spiegata su tutto il batch
                y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
                var_y = np.var(y_true)
                explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

                # 1. Losses Dictionary
                training_losses = {
                    'value_loss': v_loss.item(),
                    'policy_loss': pg_loss.item(),
                    'entropy': entropy_loss.item(),
                    'approx_kl': approx_kl.item(),
                    'clipfrac': np.mean(clipfracs),
                    'explained_variance': explained_var
                }

                # 2. Episodic Metrics Dictionary (Dinamico)
                # Calcolo metriche episodiche (Batch corrente)
                episodic_metrics = {}
                n_episodes_batch = 0
                
                # Calcola statistiche del batch corrente (per vedere oscillazioni immediate)
                if len(epoch_stats) > 0:
                    n_episodes_batch = len(next(iter(epoch_stats.values())))
                    for k, v_list in epoch_stats.items():
                        episodic_metrics[f'batch_{k}'] = np.mean(v_list) # Rinomina per chiarezza
                
                # Aggiungi le metriche "Globali" (più stabili)
                if len(recent_rewards) > 0:
                    episodic_metrics['rolling_reward'] = np.mean(recent_rewards)
                else:
                    episodic_metrics['rolling_reward'] = -float('inf')

                episodic_metrics['episodes_per_batch'] = n_episodes_batch
                
                # 3. Technical Metrics Dictionary
                tech_metrics = {
                    'learning_rate': optimizer.param_groups[0]["lr"],
                    'SPS': int(global_step / (time.time() - start_time))
                }

                # Console Print
                roll_rew = episodic_metrics['rolling_reward']
                print(f"[{global_step}] Loss: {loss.item():.4f} | Roll Reward: {roll_rew:.2f} | Eps in Batch: {n_episodes_batch}")
                
                # WandB Log (usando l'helper richiesto)
                if args.wandb:
                    log_stats_to_wandb(
                        wandb_run, 
                        [training_losses, tech_metrics, episodic_metrics], 
                        ['losses', 'charts', 'episodic'], 
                        global_step
                    )
                
                # Reset Buffer e Statistiche
                ppo_buffer.reset()
                epoch_stats = defaultdict(list)

                # Save Checkpoint
                current_reward = episodic_metrics['rolling_reward']
                
                if current_reward > best_reward and len(recent_rewards) > 10: # Aspetta almeno 10 episodi prima di salvare
                    best_reward = current_reward
                    print(f"New best reward: {best_reward:.2f}, model saved")
                    save_models_simple(agent, save_path, suffix='_best')
                
                # 3. Salva il LATEST MODEL (Sovrascrivi sempre lo stesso file per resume)
                save_models_simple(agent, save_path, suffix='_final')

    except Exception as e:  
        print(f"[{global_step}/{args.total_timesteps}] An error occurred: {e}")
        traceback.print_exc()

    # Close Environment
    print("Closing environment")
    env.close()
    
    if args.wandb:
        print("Closing wandb run")
        wandb.finish()

    # Save Final
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

for i in range(5):
    train_ppo(args, agent_config, obstacles_config, other_config)
