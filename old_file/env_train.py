import argparse
import sys
import time
import random
import traceback
from collections import deque
from pprint import pprint
import wandb
import numpy as np

import torch.optim as optim
from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel

from gymnasium import spaces 
from stable_baselines3.common.buffers import ReplayBuffer

from training_utils import *

args = parse_args()
agent_config = parse_config_file(args.agent_config_path)
obstacles_config = parse_config_file(args.obstacles_config_path)
other_config = parse_config_file(args.other_config_path)

args.seed = random.randint(0, 2**16)
# args.name = generate_funny_name()

print('Training with the following parameters:')
pprint(vars(args))

print('agent_config:')
pprint(agent_config)

print('obstacles_config:')
pprint(obstacles_config)

print('other_config:')
pprint(other_config)

if torch.cuda.is_available() and args.cuda >= 0:
    # F-string per inserire l'indice: diventa "cuda:2"
    device_str = f"cuda:{args.cuda}"
else:
    device_str = "cpu"

DEVICE = torch.device(device_str)
print(f"Using device: {DEVICE}")

# [markdown]
#  Seeding

# seeding
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic
print(f'Seed: {args.seed}')

# [markdown]
#  Start Environment

# Create the channel
env_info = CustomChannel()
param_channel = EnvironmentParametersChannel()

print('Applying Unity settings from config...')
apply_unity_settings(param_channel, agent_config, 'ag_')

curriculum_step = 1
curriculum_last_update = 0
modify_config_for_curriculum(curriculum_step, args.curriculum_steps, obstacles_config)
apply_unity_settings(param_channel, obstacles_config, 'obs_')

if args.test_lib:
    print('Testing Ended')
    exit(0)

# env setup
print(f'Starting Unity Environment from build: {args.build_path}')
# args.build_path
env = UnityEnvironment(args.build_path, 
                       seed=args.seed, 
                       side_channels=[env_info, param_channel], 
                       no_graphics=args.headless,
                       worker_id=args.worker_id)
print('Unity Environment connected.')

print('Resetting environment...')
env.reset()

# [markdown]
#  Environment Variables and Log

run_name = f"{args.exp_name}_{int(time.time()) - args.base_time}"
args.run_name = run_name
print(f"Run name: {run_name}")

# wandb to track experiments
# Start a new wandb run to track this script.
if args.wandb:
    print('Setting up wandb experiment tracking.')
    wandb_run = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="giacomo-aru",
        # Set the wandb project where this run will be logged.
        project="UASRL",
        # force the 
        name=args.run_name,
        # Track hyperparameters and run metadata.
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

# creating the training networks
print('Creating actor and critic networks...')
actor = DenseActor(TOTAL_STATE_SIZE, ACTION_SIZE, ACTION_MIN, ACTION_MAX, args.actor_network_layers).to(DEVICE)
actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

qf_ensemble = [DenseSoftQNetwork(TOTAL_STATE_SIZE, ACTION_SIZE, args.q_network_layers).to(DEVICE) for _ in range(args.q_ensemble_n)]
qf_ensemble_target = [DenseSoftQNetwork(TOTAL_STATE_SIZE, ACTION_SIZE, args.q_network_layers).to(DEVICE) for _ in range(args.q_ensemble_n)]
for q_t, q in zip(qf_ensemble_target, qf_ensemble):
    q_t.load_state_dict(q.state_dict())

par = []
for q in qf_ensemble:
    par += list(q.parameters())
qf_optimizer = torch.optim.Adam(
    par,
    lr=args.q_lr
)

obs_stack = DenseStackedObservations(args.input_stack, 
                                     STATE_SIZE + RAYCAST_SIZE, 
                                     args.n_envs)

# [markdown]
#  Replay Buffer

print('Setting up replay buffer...')
observation_space = spaces.Box(
    low=min(RAYCAST_MIN, STATE_MIN), 
    high=max(RAYCAST_MAX, STATE_MAX), 
    shape=(TOTAL_STATE_SIZE,), 
    dtype=np.float32
)
action_space = spaces.Box(
    low=ACTION_MIN, 
    high=ACTION_MAX, 
    shape=(ACTION_SIZE,), 
    dtype=np.float32
)

rb = ReplayBuffer(
    buffer_size=args.buffer_size,
    observation_space=observation_space,
    action_space=action_space,
    device=DEVICE,                
    handle_timeout_termination=True,
    n_envs=1 # necessario data la natura asincrona del'env   
)

# [markdown]
#  start algorithm

# Automatic entropy tuning
if args.autotune:
    target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(DEVICE)).item()
    log_alpha = torch.tensor([-1.0], requires_grad=True, device=DEVICE)
    alpha = log_alpha.exp().item()
    a_optimizer = optim.Adam([log_alpha], lr=args.alpha_lr)
    print(f'autotune target_entropy: {target_entropy}')
else:
    alpha = args.alpha

# start training
save_path = './models/' + run_name
os.makedirs(save_path, exist_ok=True)
print('saving to path:', save_path)

training_stats = {
    "time/python_time": RunningMean(),
    "time/unity_time": RunningMean(),
    
    "stats/action_saturation": RunningMean(),
    'stats/qf_mean': RunningMean(),
    'stats/qf_std':RunningMean(),
    'stats/actor_entropy': RunningMean(),
    'stats/alpha': RunningMean(),
    'stats/uncertainty': RunningMean(),
    
    'loss/critic_ens': RunningMean(),
    'loss/actor': RunningMean(),
    'loss/alpha': RunningMean(),
}

best_reward = -float('inf')

episodic_stats = {}
success_stats = {}
failure_stats = {}

start_time = time.time()
unity_end_time = -1
unity_start_time = -1

env_step = 0
gradient_step = 0
print(f'[{env_step}/{args.total_timesteps}] Starting Training - run name: {run_name}')

try:
    decision_obs, terminal_obs = observe_batch(env, BEHAVIOUR_NAME, (STATE_SIZE + RAYCAST_SIZE)) 
    obs_stack.add_observation(terminal_obs[1], terminal_obs[0])
    terminal_obs[1] = obs_stack.get_stacked_observations(terminal_obs[0])
    obs_stack.reset(terminal_obs[0]) # reset for the NEXT step
    
    obs_stack.add_observation(decision_obs[1], decision_obs[0])
    decision_obs[1] = obs_stack.get_stacked_observations(decision_obs[0])
    
    # observe_batch_stacked(env, BEHAVIOUR_NAME, args.input_stack, TOTAL_STATE_SIZE)
    while env_step < args.total_timesteps:
        # --- ACTION SELECTION ---
        if env_step < args.learning_starts: # warm-up with random actions
            action = get_initial_action_batch(decision_obs[0])

        else:
            obs_tensor = torch.as_tensor(obs_stack.get_stacked_observations(decision_obs[0]), dtype=torch.float32).to(DEVICE)
            
            actor.eval()
            with torch.no_grad():
                action, _, _ = actor.get_action(obs_tensor)
            action = action.detach().cpu().numpy()
        
        # Action Taken
        # decision_obs.append(action) 
        if len(action) > 0: 
            a = ActionTuple(continuous=action)
            env.set_actions(BEHAVIOUR_NAME, a)
        
        # --- ENVIRONMENT STEP ---
        unity_start_time = time.time()
        if unity_end_time > 0 and env_step > args.learning_starts:
            training_stats['time/python_time'].update(unity_start_time - unity_end_time)
        
        env.step()
        unity_end_time = time.time()
        if env_step > args.learning_starts:
            training_stats['time/unity_time'].update(unity_end_time - unity_start_time)
        
        next_decision_obs, next_terminal_obs = observe_batch(env, BEHAVIOUR_NAME, (STATE_SIZE + RAYCAST_SIZE))
        obs_stack.add_observation(next_terminal_obs[1], next_terminal_obs[0])
        
        dummy_var = obs_stack.get_stacked_observations(next_terminal_obs[0])
        if not np.allclose(dummy_var[:,-(STATE_SIZE + RAYCAST_SIZE):], next_terminal_obs[1], atol=1e-8):
            print(f'Warning: State part of observation changed at step {env_step}. Possible error in obs stacking.')
            print(f'dummy_var state part: {dummy_var}')
            print(f'next_terminal_obs state part: {next_terminal_obs[1]}')
            raise ValueError("State part of observation changed between steps.")
        
        next_terminal_obs[1] = obs_stack.get_stacked_observations(next_terminal_obs[0])
        obs_stack.reset(next_terminal_obs[0]) # reset for the NEXT step
        
        obs_stack.add_observation(next_decision_obs[1], next_decision_obs[0])
        
        dummy_var = obs_stack.get_stacked_observations(next_decision_obs[0])
        if not np.allclose(dummy_var[:,-(STATE_SIZE + RAYCAST_SIZE):], next_decision_obs[1], atol=1e-8):
            print(f'Warning: State part of observation at step {env_step}. Possible error in obs stacking.')
            print(f'dummy_var state part: {dummy_var}')
            print(f'next_decision_obs state part: {next_decision_obs[1]}')
            raise ValueError("State part of observation changed between steps.")
        
        next_decision_obs[1] = obs_stack.get_stacked_observations(next_decision_obs[0])
        

        # --- BUFFER DATA ---
        # for each agent in decision_obs, try to store its transition if it is present in next_decision_obs or next_terminal_obs
        for i, id in enumerate(decision_obs[0]):
            if id in next_terminal_obs[0]:
                new_idx = np.where(next_terminal_obs[0] == id)[0][0]
                
                reward = next_terminal_obs[2][new_idx]
                done = 1 # next_terminal_obs[3][new_idx]
                next_obs = next_terminal_obs[1][new_idx]
            elif id in next_decision_obs[0]:
                new_idx = np.where(next_decision_obs[0] == id)[0][0]
                
                reward = next_decision_obs[2][new_idx]
                done = 0 # next_decision_obs[3][new_idx]
                next_obs = next_decision_obs[1][new_idx]
            else:
                continue
            
            pre_obs = decision_obs[1][i]
            act = action[i]
            
            if not np.allclose(pre_obs[(STATE_SIZE + RAYCAST_SIZE):], next_obs[:-(STATE_SIZE + RAYCAST_SIZE)], atol=1e-8):
                print(f'Warning: State part of observation changed for agent {id} at step {env_step}. Possible error in obs stacking.')
                print(f'pre_obs state part: {pre_obs}')
                print(f'next_obs state part: {next_obs}')
                raise ValueError("State part of observation changed between steps.")
            
            rb.add(
                pre_obs, 
                next_obs, 
                act, 
                reward * args.reward_scale,
                done, 
                [{}]
            )
            

        # update current obs
        decision_obs = next_decision_obs
        terminal_obs = next_terminal_obs
        env_step += 1
        
        # --- STATS UPDATE MIGLIORATO ---
        while env_info.stop_msg_queue:
            msg = env_info.stop_msg_queue.pop()
            
            if env_step >= args.learning_starts:
                update_stats_from_message(episodic_stats, success_stats, failure_stats, msg, args.metrics_smoothing)        
                if episodic_stats['ep_count'] % args.metrics_log_interval == 0:
                    print_update(env_step, args.total_timesteps, start_time, episodic_stats)
                    if args.wandb:
                        log_stats_to_wandb(wandb_run, 
                                        [episodic_stats, success_stats, failure_stats],
                                        ['all_ep', 'success_ep', 'failure_ep'],
                                        env_step)
                        print(f"[{env_step}/{args.total_timesteps}] Logged episodic stats to wandb")

                enough_episodes_passed = curriculum_last_update  > args.min_episodes_per_curriculum
                is_performance_good = episodic_stats['success'] > args.min_success_rate
                not_last_step = curriculum_step < args.curriculum_steps

                if enough_episodes_passed and is_performance_good and not_last_step:    
                    save_models(actor, qf_ensemble, qf_ensemble_target, save_path, suffix=f'_c{curriculum_step}_final')
                    print(f"[{env_step}/{args.total_timesteps}] Models saved, suffix: _c{curriculum_step}_final")
                    
                    curriculum_step += 1
                    curriculum_last_update = episodic_stats['ep_count']
                    best_reward = -float('inf') # to save new best model for new curriculum step
                    modify_config_for_curriculum(curriculum_step, args.curriculum_steps, obstacles_config)
                    apply_unity_settings(param_channel, obstacles_config, 'obs_')

        # Save best models based on reward
        if episodic_stats != {} and episodic_stats["reward"] > best_reward:
            best_reward = episodic_stats["reward"]
            save_models(actor, qf_ensemble, qf_ensemble_target, save_path, suffix=f'_c{curriculum_step}_best')
            print(f"[{env_step}/{args.total_timesteps}] Models saved, suffix: _c{curriculum_step}_best")

        # normalizzazione input reti
        if env_step == args.learning_starts + 1:
            print(f"[{env_step}/{args.total_timesteps}] Computing Z-score normalization parameters...")

            # Recuperiamo tutte le osservazioni raccolte finora nel replay buffer
            current_len = rb.buffer_size if rb.full else rb.pos
            full_batch = rb.sample(current_len)
            observations = full_batch.observations

            # Calcoliamo Media e Deviazione Standard per ogni feature
            obs_mean = torch.mean(observations, dim=0)
            obs_std = torch.std(observations, dim=0)

            # Gestione delle feature costanti (es. sensori che leggono sempre 0)
            # Se la std è 0, la impostiamo a 1.0 per evitare la divisione per zero
            const_mask = (obs_std == 0)
            if const_mask.any():
                print(f"[{env_step}/{args.total_timesteps}] WARNING: {const_mask.sum().item()} constant features detected.")
                obs_std[const_mask] = 1.0

            # --- AGGIORNAMENTO MODELLI ---
            
            # 1. Actor (usa solo osservazioni)
            actor.set_normalization_params(obs_mean.to(DEVICE), obs_std.to(DEVICE))

            # 3. Aggiorna Ensemble corrente
            for q_net in qf_ensemble:
                q_net.set_normalization_params(obs_mean, obs_std)

            # 4. Aggiorna Ensemble Target
            for q_target_net in qf_ensemble_target:
                q_target_net.set_normalization_params(obs_mean, obs_std)
            
            print("Normalization updated: Actor and Q-Nets are now synchronized.")              
        
        # --- DIAGNOSTIC CHECKS ---


        # --- TRAINING LOOP ---       
        if env_step > args.learning_starts:
            
            actor.train()
            for q in qf_ensemble: q.train()
            
            for _ in range(args.update_frequency):
                
                gradient_step += 1
                data = rb.sample(args.batch_size)
                
                # --- CALCOLO SATURAZIONE ---
                saturation = data.actions.detach().cpu().numpy()
                saturation = (np.abs(saturation) > 0.99).mean()
                training_stats["stats/action_saturation"].update(saturation)
                
                # --- 1. CRITIC UPDATE ---
                with torch.no_grad():
                    next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
                    
                    qf_next_target = []
                    for q_target in qf_ensemble_target:
                        q_val = q_target(data.next_observations, next_state_actions)
                        qf_next_target.append(q_val)
                    qf_next_target = torch.stack(qf_next_target)
                    
                    min_qf_next_target = qf_next_target.min(dim=0).values - alpha * next_state_log_pi
                    next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).reshape(-1)

                # Calcolo Loss Critici
                qf_values = []
                qf_losses = []
                for i, q in enumerate(qf_ensemble):
                    current_qf_val = q(data.observations, data.actions).reshape(-1)
                    qf_values.append(current_qf_val)
                    
                    current_loss = F.mse_loss(current_qf_val, next_q_value)
                    qf_losses.append(current_loss)
                qf_loss = torch.stack(qf_losses).mean()
                
                # optimize the model
                qf_optimizer.zero_grad()
                qf_loss.backward()
                qf_optimizer.step()
            
                training_stats['stats/qf_mean'].update(torch.stack(qf_values).mean())
                training_stats['stats/qf_std'].update(torch.stack(qf_values).std())
                training_stats['loss/critic_ens'].update(qf_loss.item())
                
                # --- 2. ACTOR UPDATE (Delayed) ---
                if gradient_step % args.policy_frequency == 0:
                    for _ in range(args.policy_frequency):
                        
                        pi, log_pi, _ = actor.get_action(data.observations)

                        q_pi_vals = [q(data.observations, pi) for q in qf_ensemble]
                        
                        q_pi_vals = torch.stack(q_pi_vals, dim=0)   # [n_q, batch]
                        min_qf_pi = q_pi_vals.min(dim=0).values 

                        actor_loss = ((alpha * log_pi) - min_qf_pi).mean()
                        
                        actor_optimizer.zero_grad()
                        actor_loss.backward()
                        actor_optimizer.step()

                        training_stats['stats/uncertainty'].update((q_pi_vals.std(dim=0)).mean())
                        training_stats['stats/actor_entropy'].update(-log_pi.mean())
                        training_stats['loss/actor'].update(actor_loss.item())
                        
                        # --- 3. ALPHA AUTO-TUNING ---
                        if args.autotune:
                            # data_pi = rb.sample(args.batch_size)
                            with torch.no_grad():
                                _, log_pi_alpha, _ = actor.get_action(data.observations)
                            alpha_loss = (-log_alpha.exp() * (log_pi_alpha + target_entropy)).mean()

                            a_optimizer.zero_grad()
                            alpha_loss.backward()
                            a_optimizer.step()
                            
                            alpha = log_alpha.exp().item()
                            
                            training_stats['loss/alpha'].update(alpha_loss.item())

                        training_stats['stats/alpha'].update(alpha)

                # --- 4. TARGET UPDATE (Soft Update) ---
                if gradient_step % args.target_network_update_period == 0:
                    for q, q_t in zip(qf_ensemble, qf_ensemble_target):
                        for param, target_param in zip(q.parameters(), q_t.parameters()):
                            target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

                # --- 5. LOGGING LOSS (METODO SNAPSHOT/ISTANTANEO) ---
                if gradient_step % args.loss_log_interval == 0:

                    # COSTRUZIONE DIZIONARIO SNAPSHOT
                    training_stats_divided = {}
                    for key in training_stats:
                        splitted = key.split('/')
                        if splitted[0] not in training_stats_divided:
                            training_stats_divided[splitted[0]] = {}
                        training_stats_divided[splitted[0]][splitted[1]] = training_stats[key].mean()
                        
                        # reset
                        training_stats[key].reset()
                        
                    current_time = time.time()
                    training_stats_divided['time']['SPS'] = env_step / (current_time - start_time + 1e-6)
                    
                    # log stats su wandb
                    if args.wandb:
                        log_stats_to_wandb(wandb_run, list(training_stats_divided.values()), list(training_stats_divided.keys()), env_step)
                        print(f"[{env_step}/{args.total_timesteps}] Logged training stats to wandb")
            
            '''if env_step % 100 == 0:  # ogni 100 step
                # 1. Reward batch
                print(f"[Step {env_step}] Reward batch stats: mean={data.rewards.mean():.4f}, std={data.rewards.std():.4f}, min={data.rewards.min():.4f}, max={data.rewards.max():.4f}")

                # 2. Critic target
                print(f"[Step {env_step}] next_q_value stats: mean={next_q_value.mean():.4f}, std={next_q_value.std():.4f}, min={next_q_value.min():.4f}, max={next_q_value.max():.4f}")

                # 3. Critic outputs
                q_outputs = torch.stack([q(data.observations, data.actions).view(-1) for q in qf_ensemble])
                print(f"[Step {env_step}] Q ensemble outputs: mean={q_outputs.mean():.4f}, std={q_outputs.std():.4f}, min={q_outputs.min():.4f}, max={q_outputs.max():.4f}")

                # 4. Actor outputs (mean Q under policy)
                with torch.no_grad():
                    pi, log_pi, _ = actor.get_action(data.observations)
                    q_pi_vals = torch.stack([q(data.observations, pi) for q in qf_ensemble])
                print(f"[Step {env_step}] Q under actor: mean={q_pi_vals.mean():.4f}, std={q_pi_vals.std():.4f}, min={q_pi_vals.min():.4f}, max={q_pi_vals.max():.4f}")
                print(f"[Step {env_step}] Actor log_pi stats: mean={log_pi.mean():.4f}, std={log_pi.std():.4f}")

                # 5. Alpha
                print(f"[Step {env_step}] Alpha: {alpha:.4f}")'''             
                        
except Exception as e:  
    print(f"[{env_step}/{args.total_timesteps}] An error occurred: {e}")
    traceback.print_exc()

# [markdown]
#  Close Environment

print("Closing environment")
env.close()

print("Closing wandb run")
wandb.finish()

# save trained networks, actor and critics
save_models(actor, qf_ensemble, qf_ensemble_target, save_path, suffix='_final')
print(f"[{env_step}/{args.total_timesteps}] Models saved, suffix: _final")

