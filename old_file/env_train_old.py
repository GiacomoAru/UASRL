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

# [markdown]
#  Args


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
apply_unity_settings(param_channel, obstacles_config, 'obs_')

if args.test_lib:
    print('Testing Ended')
    exit(0)

# env setup
print(f'Starting Unity Environment from build: {args.build_path}')
# args.build_path
env = UnityEnvironment(None, 
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
actor = OldDenseActor(TOTAL_STATE_SIZE, ACTION_SIZE, ACTION_MIN, ACTION_MAX, args.actor_network_layers).to(DEVICE)
actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

qf_ensemble = [OldDenseSoftQNetwork(TOTAL_STATE_SIZE, ACTION_SIZE, args.q_network_layers).to(DEVICE) for _ in range(args.q_ensemble_n)]
qf_ensemble_target = [OldDenseSoftQNetwork(TOTAL_STATE_SIZE, ACTION_SIZE, args.q_network_layers).to(DEVICE) for _ in range(args.q_ensemble_n)]
for q_t, q in zip(qf_ensemble_target, qf_ensemble):
    q_t.load_state_dict(q.state_dict())

par = []
for q in qf_ensemble:
    par += list(q.parameters())
qf_optimizer = torch.optim.Adam(
    par,
    lr=args.q_lr
)



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
    target_entropy = args.target_entropy
    log_alpha = torch.zeros(1, requires_grad=True, device=DEVICE)
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

global_step = 0
print(f'[{global_step}/{args.total_timesteps}] Starting Training - run name: {run_name}')

try:
    obs = collect_data_after_step(env, BEHAVIOUR_NAME, STATE_SIZE)
    
    
    while global_step < args.total_timesteps:

        # actions for each agent in the environment
        # dim = (naagents, action_space)
        for id in obs:
            agent_obs = obs[id]
            
            # terminated agents are not considered
            if agent_obs[3]:
                continue
            
            # algo logic
            if global_step < args.learning_starts * 2:
                # change this to use the handcrafted starting policy or a previously trained policy
                
                action = get_initial_action(id)
                # action, _, _ = old_actor.get_action(torch.Tensor([obs[id][0]]), 
                #                                 torch.Tensor([obs[id][1]]),
                #                                 0.5)
                # action = action[0].detach().numpy()
            else:
                # training policy
                action, _, _ = actor.get_action(torch.Tensor([obs[id][0]]).to(DEVICE))
                action = action[0].detach().cpu().numpy()
            
            # memorize the action taken for the next step
            agent_obs[2] = action
            
            # the first dimention of the action is the "number of agent"
            # Always 1 if "set_action_for_agent" is used
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
        
        while env_info.stop_msg_queue:
                msg = env_info.stop_msg_queue.pop()
                
                if global_step >= args.learning_starts:
                    update_stats_from_message(episodic_stats, success_stats, failure_stats, msg, args.metrics_smoothing)        
                    if episodic_stats['ep_count'] % args.metrics_log_interval == 0:
                        print_update(global_step, args.total_timesteps, start_time, episodic_stats)
                        if args.wandb:
                            log_stats_to_wandb(wandb_run, 
                                            [episodic_stats, success_stats, failure_stats],
                                            ['all_ep', 'success_ep', 'failure_ep'],
                                            global_step)
                            print(f"[{global_step}/{args.total_timesteps}] Logged episodic stats to wandb")
                        
        # save data to reply buffer; handle `terminal_observation`
        for id in obs:
            prev_agent_obs = obs[id]
            # consider every agent that in the previous step was not terminated
            # in this way are excluded the agents that are already considered before and don't have a 
            # couple prev_obs - next_obs and a reward
            if prev_agent_obs[3] or id not in next_obs:
                continue
                
            next_agent_obs = next_obs[id]
            
            # add the data to the replay buffer
            rb.add(obs = prev_agent_obs[0], 
                next_obs = next_agent_obs[0],
                action = np.array(prev_agent_obs[2]), 
                reward = next_agent_obs[1], 
                done = next_agent_obs[3],
                infos = [{}])
            
        # crucial step, easy to overlook, update the previous observation
        obs = next_obs
        
        # Save best models based on reward
        if episodic_stats != {} and episodic_stats["reward"] > best_reward:
            best_reward = episodic_stats["reward"]
            save_models(actor, qf_ensemble, qf_ensemble_target, save_path, suffix=f'_best')
            print(f"[{global_step}/{args.total_timesteps}] Models saved, suffix: _best")
                
        # Training loop
        for _ in range(args.update_frequency):

            # Start learning after a warm-up phase
            if global_step > args.learning_starts:

                # Sample a batch from replay buffer
                data = rb.sample(args.batch_size)

                # --- CALCOLO SATURAZIONE ---
                saturation = data.actions.detach().cpu().numpy()
                saturation = (np.abs(saturation) > 0.99).mean()
                training_stats["stats/action_saturation"].update(saturation)
                    
                with torch.no_grad():
                    # Compute target action with exploration noise
                    next_action, next_log_pi, _ = actor.get_action(
                        data.next_observations
                    )

                    if args.noise_clip > 0:
                        noise = torch.randn_like(next_action) * args.noise_clip
                        next_action = torch.clamp(next_action + noise, -1, 1)

                    # Compute target Q-value (min over ensemble)
                    target_q_values = []
                    for q_target in qf_ensemble_target:
                        q_val = q_target(
                            data.next_observations, 
                            next_action
                        )
                        target_q_values.append(q_val)
                    stacked_target_q = torch.stack(target_q_values)
                    min_qf_next_target = stacked_target_q.min(dim=0).values - alpha * next_log_pi
                    next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * min_qf_next_target.view(-1)

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
                total_q_loss.backward()
                qf_optimizer.step()
                
                # Track Q-value statistics
                all_q_values = torch.cat(q_vals)
                training_stats['stats/qf_mean'].update(all_q_values.mean().item())
                training_stats['stats/qf_std'].update(all_q_values.std().item())
                training_stats['loss/critic_ens'].update(total_q_loss.item())
                
                # Delayed policy (actor) update
                if global_step % args.policy_frequency == 0:
                    for _ in range(args.policy_frequency):
                        pi, log_pi, _ = actor.get_action(data.observations)
                        actor_entropy = - (log_pi.exp() * log_pi).sum(dim=-1).mean()

                        q_pi_vals = [q(data.observations, pi) for q in qf_ensemble]
                        min_qf_pi = torch.min(torch.stack(q_pi_vals), dim=0).values.view(-1)

                        actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                        actor_optimizer.zero_grad()
                        actor_loss.backward()
                        actor_optimizer.step()

                        # 1. Calcolo Incertezza (Disaccordo tra i critici)
                        q_pi_stack = torch.stack(q_pi_vals) 
                        with torch.no_grad():
                            uncertainty = q_pi_stack.std(dim=0).mean().item()
                        training_stats['stats/uncertainty'].update(uncertainty)

                        # 2. Log Entropia e Loss Attore
                        training_stats['stats/actor_entropy'].update(-log_pi.mean().item())
                        training_stats['loss/actor'].update(actor_loss.item())
                        
                        # Automatic entropy tuning (if enabled)
                        if args.autotune:
                            with torch.no_grad():
                                _, log_pi, _ = actor.get_action(data.observations)
                            alpha_loss = (-log_alpha * (log_pi + target_entropy)).mean()

                            a_optimizer.zero_grad()
                            alpha_loss.backward()
                            a_optimizer.step()
                            alpha = log_alpha.exp().item()
                            
                            training_stats['loss/alpha'].update(alpha_loss.item())

                        training_stats['stats/alpha'].update(alpha)
                        
                # Soft update target Q-networks
                if global_step % args.target_network_update_period == 0:
                    for q, q_t in zip(qf_ensemble, qf_ensemble_target):
                        for param, target_param in zip(q.parameters(), q_t.parameters()):
                            target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

                # --- 5. LOGGING LOSS (METODO SNAPSHOT/ISTANTANEO) ---
                if global_step % args.loss_log_interval == 0:

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
                    training_stats_divided['time']['SPS'] = global_step / (current_time - start_time + 1e-6)
                    
                    # log stats su wandb
                    if args.wandb:
                        log_stats_to_wandb(wandb_run, list(training_stats_divided.values()), list(training_stats_divided.keys()), global_step)
                        print(f"[{global_step}/{args.total_timesteps}] Logged training stats to wandb")  
                            
            elif global_step == args.learning_starts:
                print("Start Learning")

            # Step counter
            global_step += 1
            
except Exception as e:  
    print(f"[{global_step}/{args.total_timesteps}] An error occurred: {e}")
    traceback.print_exc()

# [markdown]
#  Close Environment

print("Closing environment")
env.close()

print("Closing wandb run")
wandb.finish()

# save trained networks, actor and critics
save_models(actor, qf_ensemble, qf_ensemble_target, save_path, suffix='_final')
print(f"[{global_step}/{args.total_timesteps}] Models saved, suffix: _final")

