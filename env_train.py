import time
import random
from collections import deque
from pprint import pprint

import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
import numpy as np
from gymnasium import spaces 
from stable_baselines3.common.buffers import ReplayBuffer

from utils_policy_train import *

CONFIG_PATH = './train_config/train_std.yaml'
BASE_TIME = 1765283466
print(f'Config path: {CONFIG_PATH}')

# [markdown]
#  Args

args = parse_config(CONFIG_PATH)
args = argparse.Namespace(**args)
agent_config = parse_config(args.agent_config_path)
obstacles_config = parse_config(args.obstacles_config_path)
other_config = parse_config(args.other_config_path)

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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
apply_unity_settings(param_channel, obstacles_config, 'obst_')

# env setup
print(f'Starting Unity Environment from build: {args.build_path}')
'''if args.headless:
    print('Running in headless mode...')
    env = UnityEnvironment(args.build_path, seed=args.seed, side_channels=[env_info, param_channel], no_graphics=True)
else:
    env = UnityEnvironment(args.build_path, seed=args.seed, side_channels=[env_info, param_channel])'''
env = UnityEnvironment(None, seed=args.seed, side_channels=[env_info, param_channel])
print('Unity Environment connected.')

print('Resetting environment...')
env.reset()

# [markdown]
#  Environment Variables and Log

run_name = f"{args.exp_name}_{int(time.time()) - BASE_TIME}"
args.full_name = run_name
print(f"Run name: {run_name}")

# writer to track performance  
print('Setting up TensorBoard writer...')
writer = SummaryWriter(f"train/{run_name}")
write_dict(writer, args, 'config')
write_dict(writer, agent_config, 'agent_config')
write_dict(writer, obstacles_config, 'obstacles_config')
write_dict(writer, other_config, 'other_config')

other_config

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

qf_ensemble = [DenseSoftQNetwork(TOTAL_STATE_SIZE + ACTION_SIZE, args.q_network_layers).to(DEVICE) for _ in range(args.q_ensemble_n)]
qf_ensemble_target = [DenseSoftQNetwork(TOTAL_STATE_SIZE + ACTION_SIZE, args.q_network_layers).to(DEVICE) for _ in range(args.q_ensemble_n)]
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
    low=-1.0, 
    high=1.0, 
    shape=(ACTION_SIZE,), 
    dtype=np.float32
)

rb = ReplayBuffer(
    buffer_size=args.buffer_size,
    observation_space=observation_space,
    action_space=action_space,
    device=DEVICE,                
    handle_timeout_termination=True,
    n_envs=1               
)

# [markdown]
#  start algorithm

# Automatic entropy tuning
if args.autotune:
    target_entropy = -torch.prod(torch.Tensor(ACTION_SIZE).to(DEVICE)).item()
    log_alpha = torch.zeros(1, requires_grad=True, device=DEVICE)
    alpha = log_alpha.exp().clamp(min=1e-4).item()
    a_optimizer = optim.Adam([log_alpha], lr=args.alpha_lr)
else:
    alpha = args.alpha

# start training
save_path = './models/' + run_name
os.makedirs(save_path, exist_ok=True)
print('saving to path:', save_path)

training_stats = {}

keys = (
    [f"loss_ens/qf{i}" for i in range(len(qf_ensemble))] +
    [
        "loss/qf_mean",
        "loss/actor",
        "stats/policy_entropy",
        "stats/qf_mean",
        "stats/qf_std",
        "stats/SPS",
        "stats/action_saturation",
        "time/python_time",
        "time/unity_time"
    ]
)

if args.autotune:
    keys += ["loss/alpha", "stats/alpha"]

for key in keys:
    training_stats[key] = [0, 0]

best_reward = -float('inf')

episodic_stats = {}
success_stats = {}
failure_stats = {}

def update_stats(stats, msg, smoothing):
    if 'id' in msg:
        del msg['id']
        
    if stats == {}:
        for key in msg:
            stats[key] = msg[key]
    else:
        for key in msg:
            stats[key] = stats[key]*smoothing + (1 - smoothing)*msg[key]


print(f'Starting Training - run name: {run_name}')

decision_obs, terminal_obs = observe_batch_stacked(env, BEHAVIOUR_NAME, args.input_stack, TOTAL_STATE_SIZE)

start_time = time.time()
unity_end_time = -1
unity_start_time = -1

env_step = 0
gradient_step = 0

while env_step < args.total_timesteps:

    # --- ACTION SELECTION ---
    if env_step < args.learning_starts * 2:
        action = get_initial_action_batch(decision_obs[0])

    else:
        obs_tensor = torch.as_tensor(decision_obs[1], dtype=torch.float32).to(DEVICE)
        
        actor.eval()
        with torch.no_grad():
            action, _, _, _ = actor.get_action(obs_tensor)
        action = action.cpu().numpy()
    
    # Action Taken
    # decision_obs.append(action) 
    if len(action) > 0: 
        a = ActionTuple(continuous=action)
        env.set_actions(BEHAVIOUR_NAME, a)
    
    # --- ENVIRONMENT STEP ---
    unity_start_time = time.time()
    if unity_end_time > 0:
        training_stats["time/python_time"][0] += (unity_start_time - unity_end_time)
        training_stats["time/python_time"][1] += 1
    
    env.step()
    unity_end_time = time.time()
    training_stats["time/unity_time"][0] += (unity_end_time - unity_start_time)
    training_stats["time/unity_time"][1] += 1
            
    next_decision_obs, next_terminal_obs = observe_batch_stacked(env, BEHAVIOUR_NAME, args.input_stack, TOTAL_STATE_SIZE)
    
    env_step += 1 # Aggiunto incremento step (mancava!)

    # --- STATS UPDATE MIGLIORATO ---
    while env_info.stop_msg_queue:
        msg = env_info.stop_msg_queue.pop()
        
        if env_step >= args.learning_starts:

            msg['path_lenght_ratio'] = msg['distance_traveled'] / msg['path_length']
            msg['SPL'] = msg['success'] * (msg['path_length']/max(msg['path_length'], msg['distance_traveled']))
            
            update_stats(episodic_stats, msg, args.metrics_smoothing)
            if msg['success'] == 1:
                update_stats(success_stats, msg, args.metrics_smoothing)
            else:
                update_stats(failure_stats, msg, args.metrics_smoothing)
                        
    # --- BUFFER DATA PREPARATION ---
    
    # Done flags (inizializzati a 0 / False)
    termination = np.zeros_like(decision_obs[0], dtype=np.float32)
    
    # Next Observations (inizializzati a Nan o Zeros)
    next_obs_clean = np.full_like(decision_obs[1], np.nan)
    
    # --- HANDLING TERMINAL AGENTS ---
    # Cerchiamo quali ID attuali sono finiti in terminal_obs
    terminal_index = np.isin(decision_obs[0], next_terminal_obs[0])
    
    if np.any(terminal_index):
        # Assumiamo align_batch restituisca le osservazioni ordinate per matchare gli ID filtrati
        terminal_aligned_batch = align_batch(next_terminal_obs[1], next_terminal_obs[0], decision_obs[0][terminal_index])
        next_obs_clean[terminal_index] = terminal_aligned_batch
        termination[terminal_index] = 1
    
    # --- HANDLING CONTINUING AGENTS ---
    decision_index = np.isin(decision_obs[0], next_decision_obs[0])
    
    if np.any(decision_index):
        decision_aligned_batch = align_batch(next_decision_obs[1], next_decision_obs[0], decision_obs[0][decision_index])
        next_obs_clean[decision_index] = decision_aligned_batch
    
    # --- ADD TO BUFFER ---
    num_agents = len(decision_obs[0])
    for i in range(num_agents):
        if np.isnan(next_obs_clean[i]).any() or np.isnan(decision_obs[1][i]).any():
            # print(f'ERRORE NAN IN OSSERVAZIONI BUFFER, ID: {decision_obs[0][i]}')
            continue 
        
        scaled_reward = args.reward_scale * decision_obs[2][i]
        rb.add(
            obs      = decision_obs[1][i],      # Prende la singola riga (210,)
            next_obs = next_obs_clean[i],      # Prende la singola riga (210,)
            action   = action[i],      # Prende la singola azione (2,)
            reward   = scaled_reward,      # Scalare o array (1,)
            done     = termination[i],      # Scalare o array (1,)
            infos    = [{}]                     # Lista con un dizionario vuoto per 1 env
        )

    # update current obs
    decision_obs = next_decision_obs
    terminal_obs = next_terminal_obs
    
    # --- 1. LOGGING EPISODICO E SALVATAGGIO (Fuori dal ciclo di update) ---
    # Lo facciamo una sola volta per environment step
    if episodic_stats != {} and env_step % args.metrics_log_interval == 0:
        for s in episodic_stats:
            writer.add_scalar("episodic_stats/" + s, episodic_stats[s], env_step)
        
        print_text = f"[{env_step}/{args.total_timesteps}] "
        for s in ['success', 'reward', 'collisions', 'length', 'SPL']:
            print_text += f"|{s}: {episodic_stats[s]:.5f}"
        print_text += f'| SPS: {int(env_step / (time.time() - start_time))}'
        print(print_text)
        
    if success_stats is not None and env_step % args.metrics_log_interval == 0:   
        for s in success_stats:
            writer.add_scalar("episodic_success_stats/" + s, success_stats[s], env_step)
    if failure_stats is not None and env_step % args.metrics_log_interval == 0:   
        for s in failure_stats:
            writer.add_scalar("episodic_failure_stats/" + s, failure_stats[s], env_step)  

    # Save best models based on reward
    if episodic_stats != {} and episodic_stats["reward"] > best_reward:
        best_reward = episodic_stats["reward"]
        torch.save(actor.state_dict(), os.path.join(save_path, 'actor_best.pth'))
        for i, qf in enumerate(qf_ensemble):
            torch.save(qf.state_dict(), os.path.join(save_path, f'qf{i+1}_best.pth'))
        for i, qft in enumerate(qf_ensemble_target):
            torch.save(qft.state_dict(), os.path.join(save_path, f'qf{i+1}_target_best.pth'))

    # --- 2. TRAINING LOOP (Gradient Updates) ---
    # Inizializziamo a None per evitare crash se il logger parte prima dell'update
    actor_loss = None 
    alpha_loss = None
    actor_entropy = None

    if env_step > args.learning_starts:
        actor.train()
        for q in qf_ensemble: q.train() # Anche i critici!
        for q_t in qf_ensemble_target: q_t.train() # (Opzionale, ma male non fa)
        
        for _ in range(args.update_frequency):
            
            gradient_step += 1
            
            # Sample a batch from replay buffer
            data = rb.sample(args.batch_size)
            if torch.isnan(data.observations).any():
                raise ValueError('ERRORE NAN NEL BATCH DI OSSERVAZIONI DAL REPLAY BUFFER')
            
            training_stats["stats/action_saturation"][0] += data.actions.abs().mean().item()
            training_stats["stats/action_saturation"][1] += 1
                    
            with torch.no_grad():
                # Compute target action with exploration noise
                next_action, next_log_pi, _, _ = actor.get_action(data.next_observations)

                if args.noise_clip > 0:
                    noise = torch.randn_like(next_action) * args.noise_clip
                    next_action = torch.clamp(next_action + noise, ACTION_MIN, ACTION_MAX)

                # Compute target Q-value (min over ensemble)
                target_q_values = []
                for q_target in qf_ensemble_target:
                    q_val = q_target(torch.cat((data.next_observations, next_action), dim=1))
                    target_q_values.append(q_val)
                
                stacked_target_q = torch.stack(target_q_values)
                
                # reshape next_log_pi per sicurezza broadcasting (N, 1)
                min_qf_next_target = stacked_target_q.min(dim=0).values - alpha * next_log_pi
                
                # next_q_value deve essere piatto (N,)
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * min_qf_next_target.view(-1)

            # Q-function updates (with bootstrapping)
            q_losses = []
            q_vals = []
            batch_size = int(data.actions.shape[0] * args.bootstrap_batch_proportion)
            
            for i, q in enumerate(qf_ensemble):
                # Bootstrap indices
                if args.bootstrap:
                    indices = torch.randint(0, batch_size, (batch_size,), device=data.actions.device)
                    
                    # Compute Q loss
                    q_val = q(torch.cat((data.observations, data.actions), dim=1)).view(-1)
                    loss = F.mse_loss(q_val[indices], next_q_value[indices])
                    q_losses.append(loss)
                    q_vals.append(q_val)
                    
                    training_stats[f"loss_ens/qf{i}"][0] += loss.item()
                    training_stats[f"loss_ens/qf{i}"][1] += 1
                        
                else:
                    # Compute Q loss
                    q_val = q(torch.cat((data.observations, data.actions), dim=1)).view(-1)
                    loss = F.mse_loss(q_val, next_q_value)
                    q_losses.append(loss)
                    q_vals.append(q_val)
                    
                    training_stats[f"loss_ens/qf{i}"][0] += loss.item()
                    training_stats[f"loss_ens/qf{i}"][1] += 1
                    
                
            total_q_loss = torch.stack(q_losses).mean()
            qf_optimizer.zero_grad()
            total_q_loss.backward()
            qf_optimizer.step()
            
            training_stats["loss/qf_mean"][0] += total_q_loss.item()
            training_stats["loss/qf_mean"][1] += 1
            
            training_stats["stats/qf_mean"][0] += torch.stack(q_vals).mean().item() 
            training_stats["stats/qf_mean"][1] += 1
            
            training_stats["stats/qf_std"][0] += torch.stack(q_vals).std(dim=0).mean().item()
            training_stats["stats/qf_std"][1] += 1
            
            # Delayed policy (actor) update
            if gradient_step % args.policy_update_period == 0:

                pi, log_pi, _, log_std = actor.get_action(data.observations)
                actor_entropy = - (log_pi.exp() * log_pi).sum(dim=-1).mean()

                q_pi_vals = [q(torch.cat((data.observations, pi), dim=1)) for q in qf_ensemble]
                min_qf_pi = torch.min(torch.stack(q_pi_vals), dim=0).values.view(-1)

                actor_loss = ((alpha * log_pi) - min_qf_pi).mean()
            
                actor_optimizer.zero_grad()
                actor_loss.backward()
                # torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
                actor_optimizer.step()

                training_stats["loss/actor"][0] += actor_loss.item()
                training_stats["loss/actor"][1] += 1

                training_stats["stats/policy_entropy"][0] += actor_entropy.item()
                training_stats["stats/policy_entropy"][1] += 1
                    
                # Automatic entropy tuning (if enabled)
                if args.autotune:
                    with torch.no_grad():
                        _, log_pi, _, _ = actor.get_action(data.observations)
                    
                    # log_pi.sum(dim=-1) se log_pi è per dimensione azione
                    alpha_loss = (-log_alpha * (log_pi.detach() + target_entropy)).mean()

                    a_optimizer.zero_grad()
                    alpha_loss.backward()
                    a_optimizer.step()
                    alpha = log_alpha.exp().item()
                    
                    training_stats["loss/alpha"][0] += alpha_loss.item()
                    training_stats["loss/alpha"][1] += 1
                    
                    training_stats["stats/alpha"][0] += alpha
                    training_stats["stats/alpha"][1] += 1

            # Soft update target Q-networks
            if gradient_step % args.target_network_update_period == 0:
                for q, q_t in zip(qf_ensemble, qf_ensemble_target):
                    for param, target_param in zip(q.parameters(), q_t.parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
            
            # Log training losses and stats
            if gradient_step % args.loss_log_interval == 0:
                for key in training_stats:
                    if training_stats[key][1] > 0:
                        writer.add_scalar(key, training_stats[key][0] / training_stats[key][1], env_step)


    elif env_step == args.learning_starts:
        print(f"Start Learning...")
    
    elif env_step == args.learning_starts*2:
        print(f"Start Using Policy...")

# [markdown]
#  Close Environment

# close environment
env.close()

# save trained networks, actor and critics
torch.save(actor.state_dict(), os.path.join(save_path, 'actor_final.pth'))
for i, qf in enumerate(qf_ensemble):
    torch.save(qf.state_dict(), os.path.join(save_path, f'qf{i+1}_final.pth'))
for i, qft in enumerate(qf_ensemble_target):
    torch.save(qft.state_dict(), os.path.join(save_path, f'qf{i+1}_target_final.pth'))

