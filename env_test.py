import argparse
import sys
import time
import random
import traceback
from collections import deque
from pprint import pprint
import wandb
import numpy as np
from decimal import Decimal

import torch.optim as optim
from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel

from gymnasium import spaces 
from stable_baselines3.common.buffers import ReplayBuffer

from training_utils import *
from testing_utils import *

# [markdown]
#  Testing Function

def test(env, 
         env_info,
         param_channel,
         
         args, 
         agent_config,
         obstacles_config,
         
         actor,
         
         BEHAVIOUR_NAME,
         STATE_SIZE,
         RAYCAST_SIZE,
         STACK_NUMBER,
         DEVICE
        ):

    print('Start testing...')
    
    print('Applying Unity settings from config...')
    apply_unity_settings(param_channel, agent_config, 'ag_')
    apply_unity_settings(param_channel, obstacles_config, 'obs_')

    print('Resetting environment...')
    env.reset() 
    env_info.clear_queue()
    
    print('Sending initial episode seeds...')
    for i in range(args.episode_queue_length):
        env_info.send_episode_seed(i+args.seed) # semplice seeding per ogni episodio
    seed_sent = args.episode_queue_length

    start_time = time.time()
    unity_end_time = -1
    unity_start_time = -1
    testing_stats = {
        "python_time": RunningMean(),
        "unity_time": RunningMean(),
    }
    
    current_episode = 1
    cumulative_obs = {}          # per-agent memory (obs, action, uncertainty info)
    running_episodes = {}        # active episodes data
    terminated_episodes = []    # finished episodes
    
    episodic_stats = {}
    dataset = []                 # collected dataset
        
    while current_episode <= args.total_episodes:
        
        try:  
            # --- ENVIRONMENT STEP ---
            unity_start_time = time.time()
            if unity_end_time > 0:
                testing_stats['python_time'].update(unity_start_time - unity_end_time)
            
            env.step()
            
            unity_end_time = time.time()
            testing_stats['unity_time'].update(unity_end_time - unity_start_time)
            
            obs = collect_data_after_step_id(env, BEHAVIOUR_NAME, STATE_SIZE)
            
            for id in obs:
                agent_obs = obs[id]

                # Handle terminated agents
                if agent_obs[3] == 1:
                    if id in cumulative_obs:
                        # Remove agent from active lists and finalize episode
                        del cumulative_obs[id]
                        terminated_episodes.append((agent_obs[4], running_episodes[id])) # tuple (internal_id, episode data)
                        del running_episodes[id]
                        
                    else:
                        # Agent killed very early
                        terminated_episodes.append((agent_obs[4], []))
                        assert id not in running_episodes and id not in cumulative_obs
                        
                else:
                    actual_obs = agent_obs[0]
                        
                    # Initialize new agent entry
                    if id not in cumulative_obs:
                        cumulative_obs[id] = [
                            args.decision_frame_period, # steps until next decision
                            None,   # last obs
                            None,   # last action taken
                        ]
                        
                    # Time to decide an action
                    if cumulative_obs[id][0] >= args.decision_frame_period:
                        cumulative_obs[id][0] = 0
                        
                        # Update ray observations with frame stacking
                        if cumulative_obs[id][1] is None:
                            cumulative_obs[id][1] = actual_obs
                            corrected_obs = actual_obs
                        else:
                            p1 = cumulative_obs[id][1][RAYCAST_SIZE:RAYCAST_SIZE*STACK_NUMBER]
                            p2 = cumulative_obs[id][1][RAYCAST_SIZE*STACK_NUMBER + STATE_SIZE:RAYCAST_SIZE*STACK_NUMBER + STATE_SIZE*STACK_NUMBER]
                            
                            corrected_obs = np.concatenate(
                                [p1,
                                actual_obs[RAYCAST_SIZE*(STACK_NUMBER - 1):RAYCAST_SIZE*STACK_NUMBER],
                                p2,
                                actual_obs[-STATE_SIZE:]]
                                )
                        
                        # Policy action from actor
                        action, _, _ = actor.get_action(torch.tensor(corrected_obs, dtype=torch.float32, device=DEVICE).unsqueeze(0), args.actor_std)
                        action = action[0].detach().cpu().numpy()
                        
                        
                        # Update agent memory
                        cumulative_obs[id][1] = corrected_obs
                        cumulative_obs[id][2] = action
                        
                        # Start new episode if not already tracked
                        if id not in running_episodes: running_episodes[id] = []
                        running_episodes[id].append({
                            'obs': corrected_obs,
                            # 'u_e': cumulative_obs[id][4],
                            # 'uf_activation': cumulative_obs[id][5],
                            'action': action,
                            'inner_steps': []
                        })

                    # Use last predicted action by default
                    policy_action = cumulative_obs[id][2] 

                    # Final action selection (UF + CBF logic)
                    final_action = policy_action
      
                    # Debug visualization (optional)
                    if args.send_debug_action:
                        env_info.send_agent_action_debug(
                            agent_obs[4],
                            
                            final_action[0], 
                            final_action[1],
                            
                            policy_action[0],
                            policy_action[1], 
                            
                            False, 
                            0.0, 
                            0.0,
                            
                            False,
                            0.0,
                            0.0
                        ) 
                                                            
                    # Apply final action to environment
                    a = ActionTuple(continuous=np.array([final_action]))
                    env.set_action_for_agent(
                        BEHAVIOUR_NAME, id, a
                    )
                    
                    # Increment frame counter
                    cumulative_obs[id][0] += 1
                    
        except Exception as e:
            print('Execution Ended ?!')
            traceback.print_exc() 
            
        new_stop_msgs = []
        for msg in env_info.stop_msg_queue:
            
            t_episode = None
            t_episode_index = -1
            for external_id, (internal_id, episode_data) in enumerate(terminated_episodes):
                if internal_id == msg['id']:
                    t_episode = episode_data
                    t_episode_index = external_id
                    break
            if t_episode is None:
                new_stop_msgs.append(msg)
                continue
            # Process terminated episode
            

            if t_episode == []:
                print(current_episode, '- agent killed too early, step', msg['length'])
                print(msg)
            else:
                update_stats_from_message_rm(episodic_stats, None, None, msg)        
                if current_episode % args.metrics_log_interval == 0:
                    print_update_rm(current_episode, args.total_episodes, start_time, episodic_stats)
                
                current_episode += 1

                if seed_sent < args.total_episodes:
                    env_info.send_episode_seed(current_episode - 1 + args.seed)
                    seed_sent += 1
                    
                # Save data if required
                # saving all obseravtion + action and episode stats
                if args.accumulate_data:
                    dataset.append(([
                        list(element['obs']) + list(element['action'])
                        for element in t_episode
                    ], msg))
            
            del terminated_episodes[int(t_episode_index)]
        env_info.stop_msg_queue = new_stop_msgs
              
        # Safety check: queue should not grow indefinitely
        if len(env_info.stop_msg_queue) > 10:
            print('ERRORE')
            raise AssertionError('Unexpected queue growth')
        
    testing_stats["unity_time"] = testing_stats["unity_time"].mean
    testing_stats["python_time"] = testing_stats["python_time"].mean
    testing_stats["ep_count"] = len(dataset)
    
    return testing_stats, episodic_stats, dataset


# [markdown]
#  Start Testing Code


args = parse_args()
train_config = parse_config_file(args.train_config_path)
other_config = parse_config_file(train_config["other_config_path"])
agent_config = parse_config_file(args.agent_config_path)

# args.seed = random.randint(0, 2**16)
# args.name = generate_funny_name()

print('Testing with the following parameters:')
pprint(vars(args))
print('train_config:')
pprint(train_config)
print('agent_config:')
pprint(agent_config)
print('other_config:')
pprint(other_config)


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
TOTAL_STATE_SIZE = (STATE_SIZE + RAYCAST_SIZE)*train_config['input_stack']
    
    
if torch.cuda.is_available() and args.cuda >= 0:
    # F-string per inserire l'indice: diventa "cuda:2"
    device_str = f"cuda:{args.cuda}"
else:
    device_str = "cpu"
DEVICE = torch.device(device_str)
print(f"Using device: {DEVICE}")

# seeding
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic
print(f'Seed: {args.seed}')

# Create the channel
env_info = CustomChannel()
param_channel = EnvironmentParametersChannel()
    
# env setup
print(f'Starting Unity Environment from build: {args.build_path}')
env = UnityEnvironment(args.build_path,
                    seed=args.seed,
                    side_channels=[env_info, param_channel], 
                    no_graphics=args.headless,
                    worker_id=args.worker_id)

print('Unity Environment connected.')
if args.episode_queue_length > args.total_episodes:
    args.episode_queue_length = args.total_episodes
    
for obs_config_path in args.obstacles_config_paths:
    obstacles_config = parse_config_file(obs_config_path)
    print('obstacles_config:')
    pprint(obstacles_config)
    
    for p_name, p_layers in zip(args.policy_names, args.policy_layers):
        
        additional_number = int(time.time()) - train_config["base_time"]
        test_name = f"{args.test_name}_{additional_number}"
        args.test_full_name = test_name

        print(f"Test name: {args.test_full_name}")
        print(f"Policy name: {p_name}")
        

        summary_save_filepath = args.save_path + args.test_name + ".csv"
        specific_save_filepath = args.save_path + args.test_name + "/" + f"{args.test_full_name}"
        os.makedirs(args.save_path, exist_ok=True)
        os.makedirs(args.save_path + args.test_name, exist_ok=True)
        
        print('Creating and loading actor network...')
        actor = OldDenseActor(
            TOTAL_STATE_SIZE,
            ACTION_SIZE,
            ACTION_MIN,
            ACTION_MAX,
            p_layers
        ).to(DEVICE)
        load_models(actor, save_path='./models/' + p_name, suffix='_best', DEVICE=DEVICE)

        other_stats, episodic_stats, dataset = test(env, 
                env_info,
                param_channel,
                
                args,
                agent_config,
                obstacles_config,
                
                actor, 
                
                BEHAVIOUR_NAME,
                STATE_SIZE,
                RAYCAST_SIZE,
                train_config['input_stack'],
                DEVICE)

        other_stats['env_name'] = obs_config_path.split('/')[-1].split('.')[0]
        other_stats['policy_name'] = p_name
        other_stats['test_name'] = args.test_full_name
        
        print(f'Saving summary data to: {specific_save_filepath}')
        save_stats_to_csv(other_stats, episodic_stats, summary_save_filepath)
        
        # Save dataset to JSON if accumulation is enabled
        print(f'Saving accumulated dataset to path: {specific_save_filepath}')
        if args.accumulate_data: 
            
            d1 = {
                'metadata': {
                    'test_config': vars(args),
                    'train_config': train_config,
                    'agent_config': agent_config,
                    'obstacles_config': obstacles_config,
                    'other_config': other_config
                },
                'data': [x[1] for x in dataset]
            }
            d2 = [x[0] for x in dataset]
            
            # Recursive helper to convert all numbers into float (JSON safe)
            def convert_all_to_float(obj):
                if isinstance(obj, dict):
                    return {k: convert_all_to_float(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [convert_all_to_float(item) for item in obj]
                elif isinstance(obj, (np.floating, Decimal)):
                    return float(obj)
                else:
                    return obj
                
            # Save dataset with timestamp in filename
            with open(specific_save_filepath + '_info.json', 'w+') as file:
                file.write(json.dumps(convert_all_to_float(d1)))
            with open(specific_save_filepath + '_transitions.json', 'w+') as file:
                file.write(json.dumps(convert_all_to_float(d2)))
env.close()

