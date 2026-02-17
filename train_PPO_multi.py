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
# PPO AGENT & NETWORK
# ==============================================================================

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class PPOAgent(nn.Module):
    def __init__(self, state_dim, action_dim, action_min, action_max, hidden_dim=256):
        super().__init__()
        self.action_min = action_min
        self.action_max = action_max
        
        # Critic (Value Function)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(state_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )
        
        # Actor (Policy Mean)
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(state_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, action_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        
        if action is None:
            action = probs.sample()
            
        # Log probability e entropy per l'update
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

# ==============================================================================
# PPO ROLLOUT BUFFER (Gestisce dizionari di ID come il tuo codice)
# ==============================================================================

class PPORolloutBuffer:
    def __init__(self, device, gamma=0.99, gae_lambda=0.95):
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.reset()

    def reset(self):
        # Memorizziamo le transizioni divise per ID agente per calcolare GAE corretto
        self.memories = defaultdict(lambda: {'states': [], 'actions': [], 'logprobs': [], 'rewards': [], 'values': [], 'dones': []})
        self.total_steps = 0

    def add(self, id, state, action, logprob, reward, done, value):
        mem = self.memories[id]
        mem['states'].append(state)
        mem['actions'].append(action)
        mem['logprobs'].append(logprob)
        mem['rewards'].append(reward)
        mem['dones'].append(done)
        mem['values'].append(value)
        self.total_steps += 1

    def prepare_batch(self, agent_network):
        # Questa funzione viene chiamata quando dobbiamo allenare.
        # Calcola GAE per ogni agente e appiattisce tutto in un unico tensore.
        
        all_states, all_actions, all_logprobs, all_advantages, all_returns, all_values = [], [], [], [], [], []

        for id, mem in self.memories.items():
            if len(mem['states']) == 0: continue
            
            # Convertiamo liste in tensori
            states = torch.stack(mem['states'])
            rewards = torch.tensor(mem['rewards'], dtype=torch.float32).to(self.device)
            dones = torch.tensor(mem['dones'], dtype=torch.float32).to(self.device)
            values = torch.cat(mem['values']).view(-1)
            
            # Bootstrap value: serve il valore dello stato successivo all'ultimo step registrato
            # Se l'agente è morto (done=True), il next_value è 0.
            # Se è vivo, dobbiamo stimarlo con la rete.
            with torch.no_grad():
                # Nota: qui facciamo una stima "sporca" usando l'ultimo stato per semplicità, 
                # oppure 0 se done. Per precisione assoluta servirebbe il 'next_obs' reale dell'ultimo step.
                # Dato che PPO tronca le traiettorie, usiamo value[-1] come stima se non è done.
                last_value = values[-1] if not mem['dones'][-1] else 0.0
            
            # Calcolo GAE
            advantages = torch.zeros_like(rewards).to(self.device)
            lastgaelam = 0
            steps = len(rewards)
            
            for t in reversed(range(steps)):
                if t == steps - 1:
                    nextnonterminal = 1.0 - dones[t]
                    nextvalues = last_value
                else:
                    nextnonterminal = 1.0 - dones[t+1]
                    nextvalues = values[t+1]
                    
                delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
            
            returns = advantages + values
            
            all_states.append(states)
            all_actions.append(torch.stack(mem['actions']))
            all_logprobs.append(torch.stack(mem['logprobs']))
            all_advantages.append(advantages)
            all_returns.append(returns)
            all_values.append(values)

        # Concatena tutto (Flat Batch)
        return (torch.cat(all_states), 
                torch.cat(all_actions), 
                torch.cat(all_logprobs), 
                torch.cat(all_advantages), 
                torch.cat(all_returns), 
                torch.cat(all_values))

# ==============================================================================
# UTILS
# ==============================================================================

def sample_log_uniform(a, b):
    return 10 ** random.uniform(math.log10(a), math.log10(b))

def sample_hparams_ppo():
    lr = sample_log_uniform(2e-4, 4e-4)
    num_steps = random.choice([2048, 4096])
    batch_size = random.choice([256, 512])

    return {
        "lr": lr,
        
        # Buffer
        "num_steps": num_steps,
        "batch_size": batch_size,

        # Iperparametri PPO (Zona Stabile)
        "clip_coef": random.uniform(0.15, 0.25),
        "ent_coef": sample_log_uniform(1e-4, 1e-2), # Unica cosa che lasciamo variare bene per l'esplorazione
    }
    
    
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
                        # Ripariamo i dati per non far crashare la rete
                        obs_tensor = torch.nan_to_num(obs_tensor, nan=0.0, posinf=1.0, neginf=-1.0)
                        
                    action, logprob, _, value = agent.get_action_and_value(obs_tensor)
                
                # Salviamo i tensori per il buffer
                step_data_cache[id] = {
                    'action': action,       # Tensor
                    'logprob': logprob,     # Tensor
                    'value': value,         # Tensor
                    'state': obs_tensor   # Tensor
                }

                # Converti per Unity
                action_np = action.cpu().numpy()[0]
                # Clipping per Unity (fisico)
                action_np = np.clip(action_np, ACTION_MIN, ACTION_MAX)
                
                # Memorizza azione per il prossimo step (logica tua)
                agent_obs[2] = action_np
                
                a = ActionTuple(continuous=np.array([action_np]))
                env.set_action_for_agent(BEHAVIOUR_NAME, id, a)

            # --- ENVIRONMENT STEP ---
            unity_start_time = time.time()
            env.step()
            unity_end_time = time.time()

            # NEXT OBS
            next_obs = collect_data_after_step(env, BEHAVIOUR_NAME, STATE_SIZE)
            
            # Stats processing (Side Channel)
            while env_info.stop_msg_queue:
                msg = env_info.stop_msg_queue.pop()
                # Itera su tutte le chiavi del messaggio (reward, length, success, hit_wall, ecc.)
                for key, value in msg.items():
                    epoch_stats[key].append(value)

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
                    state=cached['state'][0],           # Rimuovi batch dim [1, ...] -> [...]
                    action=cached['action'][0],
                    logprob=cached['logprob'],
                    reward=next_agent_obs[1],           # Reward ottenuto
                    done=next_agent_obs[3],             # Done flag
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

                # Calcolo GAE e preparazione batch
                # Questo gestisce le traiettorie separate per ogni ID
                b_obs, b_actions, b_logprobs, b_advantages, b_returns, b_values = ppo_buffer.prepare_batch(agent)
                
                # Flatten batch size
                batch_size_total = b_obs.shape[0]
                b_inds = np.arange(batch_size_total)
                
                clipfracs = []
                
                # PPO Epochs
                for epoch in range(args.update_epochs):
                    np.random.shuffle(b_inds)
                    
                    for start in range(0, batch_size_total, args.batch_size):
                        end = start + args.batch_size
                        mb_inds = b_inds[start:end]
                        
                        # --- FIX 1: Evita crash su batch residui (es. size=1) ---
                        if len(mb_inds) < 2:
                            continue
                        # --------------------------------------------------------

                        # Forward pass
                        _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                        logratio = newlogprob - b_logprobs[mb_inds]
                        ratio = logratio.exp()

                        with torch.no_grad():
                            old_approx_kl = (-logratio).mean()
                            approx_kl = ((ratio - 1) - logratio).mean()
                            clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                        # --- FIX 2: Normalizzazione Vantaggi Blindata ---
                        mb_advantages = b_advantages[mb_inds]
                        
                        # A. Pulizia preventiva NaN
                        if torch.isnan(mb_advantages).any():
                            mb_advantages = torch.nan_to_num(mb_advantages, 0.0)

                        # B. Calcolo statistiche sicuro
                        adv_mean = mb_advantages.mean()
                        # unbiased=False è cruciale: divide per N invece che N-1 (evita div by zero se N=1)
                        adv_std = mb_advantages.std(unbiased=False) 

                        # C. Normalizzazione condizionale
                        if adv_std.item() < 1e-6:
                            # Se il batch è piatto (tutti uguali), centra solo sulla media
                            mb_advantages = mb_advantages - adv_mean
                        else:
                            mb_advantages = (mb_advantages - adv_mean) / (adv_std + 1e-8)

                        # --- Loss Calculation ---
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

                        # --- FIX 3: Circuit Breaker (Ottimizzazione Sicura) ---
                        optimizer.zero_grad()
                        
                        # Controllo se la loss è esplosa
                        if torch.isnan(loss) or torch.isinf(loss):
                            print(f"⚠️ [SKIP] Loss esplosa (NaN/Inf) al batch {start}. Salto l'update.")
                            continue
                        
                        loss.backward()

                        # Clipping del gradiente
                        nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)

                        # Controllo finale sui gradienti
                        grads_are_safe = True
                        for param in agent.parameters():
                            if param.grad is not None:
                                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                                    grads_are_safe = False
                                    break
                        
                        if grads_are_safe:
                            optimizer.step()
                        else:
                            print(f"⚠️ [SKIP] Gradienti corrotti al batch {start}. Salto l'update.")
                    
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
                episodic_metrics = {}
                n_episodes = 0
                
                if len(epoch_stats) > 0:
                    # Prende la lunghezza della prima lista trovata
                    n_episodes = len(next(iter(epoch_stats.values())))
                    # Calcola media per ogni chiave
                    for k, v_list in epoch_stats.items():
                        episodic_metrics[k] = np.mean(v_list)
                    
                    episodic_metrics['episodes_per_batch'] = n_episodes
                
                # 3. Technical Metrics Dictionary
                tech_metrics = {
                    'learning_rate': optimizer.param_groups[0]["lr"],
                    'SPS': int(global_step / (time.time() - start_time))
                }

                # Console Print
                avg_rew = episodic_metrics.get('reward', float('nan'))
                print(f"[{global_step}] Update. Loss: {loss.item():.4f} | Avg Reward: {avg_rew:.2f} | Eps: {n_episodes} | SPS: {tech_metrics['SPS']}")

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
                # 1. Recupera la reward media corrente (gestendo il caso in cui non ci sono episodi finiti)
                current_reward = episodic_metrics.get('reward', -float('inf'))
                
                # 2. Salva il BEST MODEL (Solo se battiamo il record)
                if current_reward > best_reward:
                    best_reward = current_reward
                    print(f"🔥 NEW BEST MODEL! Reward: {best_reward:.2f}")
                    save_models(agent, None, None, save_path, suffix='_best')
                
                # 3. Salva il LATEST MODEL (Sovrascrivi sempre lo stesso file per resume)
                save_models(agent, None, None, save_path, suffix='_final')

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
    save_models(agent, None, None, save_path, suffix='_final')
    print("Training Complete.")


# ==============================================================================
# ENTRY POINT
# ==============================================================================

def save_models(agent, qf, qf_target, path, suffix=''):
    # Helper adattato per salvare solo l'agente PPO
    torch.save(agent.state_dict(), f"{path}/agent{suffix}.pth")

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

run_id = 0
while run_id < 1:
    combo = sample_hparams_ppo()
    print(f"\n=== RUN {run_id} ===")
    
    # Mapping hparams
    args.learning_rate = combo['lr']
    args.num_steps = combo['num_steps']
    args.batch_size = combo['batch_size']
    args.clip_coef = combo['clip_coef']
    args.ent_coef = combo['ent_coef']


    train_ppo(args, agent_config, obstacles_config, other_config)
    run_id += 1
    