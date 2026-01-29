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
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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
                            corrected_obs = cumulative_obs[id][1][RAYCAST_SIZE + STATE_SIZE:] 
                            corrected_obs = np.concatenate([corrected_obs, actual_obs[-RAYCAST_SIZE - STATE_SIZE:]])

                        # Policy action from actor
                        action, _, _ = actor.get_action(torch.tensor(corrected_obs, dtype=torch.float32, device=DEVICE).unsqueeze(0))
                        action = action[0].detach().cpu().numpy()
                        
                        
                        '''# Uncertainty filter (optional)
                        if CONFIG_DICT['uncertainty_filter']['enabled']: 
                            uncertanty_estimate = filter_methods[CONFIG_DICT['uncertainty_filter']['method']](
                                corrected_obs, 
                                action
                            )
                            cumulative_obs[id][4] = uncertanty_estimate
                            cumulative_obs[id][5] = uncertanty_estimate > CONFIG_DICT['uncertainty_filter']['threshold']'''
                        
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
                    
                    # Control Barrier Function (CBF) correction
                    '''
                    cbf_action = np.zeros(2)
                    if CONFIG_DICT['cbf']['enabled']:
                        if CONFIG_DICT['uncertainty_filter']['application'] != 'dynamic':
                            cbf_action = CBF_from_obs(
                                actual_ray_obs[-1], policy_action, env_info,
                                CONFIG_DICT['cbf']['d_safe'],
                                CONFIG_DICT['cbf']['alpha'],
                                CONFIG_DICT['cbf']['d_safe_mul'],
                                angoli_radianti_precalcolati
                            )

                        else:
                            cbf_action = CBF_from_obs(
                                actual_ray_obs[-1], policy_action, env_info,
                                CONFIG_DICT['cbf']['d_safe'] * min(cumulative_obs[id][4]/CONFIG_DICT['uncertainty_filter']['threshold'], 1),
                                CONFIG_DICT['cbf']['alpha'],
                                CONFIG_DICT['cbf']['d_safe_mul'],
                                angoli_radianti_precalcolati
                            )
                            
                        # Ensure minimum forward velocity
                        if policy_action[0] > CONFIG_DICT['cbf']['min_forward']:
                            cbf_action[0] = max(CONFIG_DICT['cbf']['min_forward'], cbf_action[0])
                        else:
                            cbf_action[0] = max(policy_action[0], cbf_action[0])'''
                                
                    # Check if CBF activated
                    # cbf_activation = CONFIG_DICT['cbf']['enabled'] and np.linalg.norm(cbf_action - policy_action) > 0.0001
                    # running_episodes[id][-1]['inner_steps'].append([np.linalg.norm(cbf_action - policy_action), cbf_activation])
                    
                    # Final action selection (UF + CBF logic)
                    final_action = policy_action
                    '''if CONFIG_DICT['uncertainty_filter']['application'] == 'interpolation':
                        interpolation_coeff = min(cumulative_obs[id][4]/CONFIG_DICT['uncertainty_filter']['threshold'], 1)
                        final_action = cbf_action * interpolation_coeff + ( 1- interpolation_coeff) * policy_action
                    else:  
                        if cumulative_obs[id][5] and cbf_activation:
                            final_action = cbf_action'''
                    
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
                    print_update(current_episode, args.total_episodes, start_time, episodic_stats)
                
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
    obstacles_config = parse_config_file(train_config["obstacles_config_path"])
    print('obstacles_config:')
    pprint(other_config)
    
    for p_name, p_layers in zip(args.policy_names, args.policy_layers):
        
        additional_number = int(time.time()) - train_config["base_time"]
        test_name = f"{args.test_name}_{additional_number}"
        args.test_full_name = test_name

        print(f"Test name: {args.test_full_name}")
        print(f"Policy name: {p_name}")
        

        summary_save_filepath = args.save_path + args.test_name + ".csv"
        specific_save_filepath = args.save_path + args.test_name + "/" + f"{additional_number}" + ".json"
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
        load_models(actor, save_path='./models/' + p_name, suffix='_best')


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
                DEVICE)

        other_stats['env_name'] = obs_config_path.split('/')[-1].split('.')[0]
        other_stats['policy_name'] = p_name
        
        print(f'Saving summary data to: {specific_save_filepath}')
        save_stats_to_csv(other_stats, episodic_stats, summary_save_filepath)
        
        # Save dataset to JSON if accumulation is enabled
        print(f'Saving accumulated dataset to path: {specific_save_filepath}')
        if args.accumulate_data: 
            
            dataset = {
                'metadata': {
                    'test_config': vars(args),
                    'train_config': train_config,
                    'agent_config': agent_config,
                    'obstacles_config': obstacles_config,
                    'other_config': other_config
                },
                'data': dataset
            }
            
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
            with open(specific_save_filepath, 'w+') as file:
                file.write(json.dumps(convert_all_to_float(dataset)))
    
env.close()

