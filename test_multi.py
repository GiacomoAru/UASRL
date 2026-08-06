import argparse
import sys
import time
import random
import traceback
from collections import deque
from pprint import pprint
from sympy import Q
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
from uncertainty_utils import *
from atom_cbf import ATOMCBFController

# [markdown]
#  Testing Function

def test(env, 
         env_info,
         param_channel,
         
         args, 
         agent_config,
         obstacles_config,
         other_config,
         
         actor,
         unc_ensamble,
         unc_enamble_norm_stats,
         atom_controller,
         
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

    env_info.clear_queue()

    print('Sending initial episode seeds...')
    for i in range(args.episode_queue_length):
        env_info.send_episode_seed(i + args.seed) # semplice seeding per ogni episodio
    seed_sent = args.episode_queue_length
    replacement_seeds_sent = 0

    # Side-channel messages queued before reset are delivered with the reset
    # command, so the first evaluated episodes use the requested seeds too.
    print('Resetting environment...')
    env.reset()

    if args.uf:
        idc_unc_percentile = int(torch.argmin(torch.abs(unc_enamble_norm_stats['percentile_levels'] - args.ut)))
        print(f"Using {unc_enamble_norm_stats['percentile_levels'][idc_unc_percentile]} Percentile for Uncertainty")
        print(f"\treal value: {unc_enamble_norm_stats['epistemic']['percentiles'][idc_unc_percentile]}")
        percentile_real_value = float(
            unc_enamble_norm_stats['epistemic']['percentiles'][idc_unc_percentile]
        )
    else:
        percentile_real_value = 0.0
        
    start_time = time.time()
    prev_time = -1
    
    testing_stats = {
        "cbf_time": RunningMean(),
        "atom_cbf_time": RunningMean(),
        "atom_cbf_activation": RunningMean(),
        "atom_cbf_uncertainty": RunningMean(),
        "atom_cbf_epsilon": RunningMean(),
        "atom_cbf_slack": RunningMean(),
        "atom_cbf_solver_failure": RunningMean(),
        "atom_cbf_emergency": RunningMean(),
        "uf_time": RunningMean(),
        "policy_time": RunningMean(),
        "unity_time": RunningMean(),
    }
    
    current_episode = 1
    cumulative_obs = {}          # per-agent memory (obs, action, uncertainty info)
    running_episodes = {}        # active episodes data
    terminated_episodes = []    # finished episodes
    ray_max_degrees = other_config.get('raycast_max_degrees', 90)
    angoli_radianti_precalcolati = generate_angles_rad(
        other_config['rays_per_direction'], ray_max_degrees
    )
    
    episodic_stats = {}
    dataset = []                 # collected dataset
        
    while current_episode <= args.total_episodes:
        

        obs = collect_data_after_step_id(env, BEHAVIOUR_NAME, STATE_SIZE)
        env_info.collision_msg_queue = attach_collision_events_to_running_transitions(
            env_info.collision_msg_queue,
            obs,
            running_episodes,
        )
        
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
                        0.0,    # last epistemic estimate
                        True,   # last UF activation
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
                    prev_time = time.time()
                    action_torch, _, _, action_mean, action_std = actor.get_action(torch.tensor(corrected_obs, dtype=torch.float32, device=DEVICE).unsqueeze(0), args.actor_std)
                    action = action_torch[0].detach().cpu().numpy()
                    testing_stats['policy_time'].update(time.time() - prev_time)
                    
                    if args.uf: 
                        prev_time = time.time()
                        obs_tensor = torch.tensor(corrected_obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                        if args.ue_action_type == 'distribution':
                            ue_input = torch.cat((obs_tensor, action_mean, action_std), dim=1).to(dtype=torch.float32, device=DEVICE)
                            # ue_input = torch.tensor(ue_input, dtype=torch.float32, device=DEVICE)
                        else:
                            ue_input = torch.cat((obs_tensor, action_torch), dim=1).to(dtype=torch.float32, device=DEVICE)
                            # ue_input = torch.tensor(ue_input, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                        
                        aleatoric_unc, epistemic_unc = predict_uncertainty(unc_ensamble, ue_input)
                        
                        # Calcolo vettorizzato su GPU
                        # z_score_tensor = (epistemic_unc - unc_enamble_norm_stats['epistemic']['mean']) / unc_enamble_norm_stats['epistemic']['std']
                        
                        # 5. Salvataggio
                        epistemic_unc = epistemic_unc.item() # Estrae il float dal tensore (1,)
                        
                        cumulative_obs[id][3] = epistemic_unc
                        cumulative_obs[id][4] = epistemic_unc > percentile_real_value
                        testing_stats['uf_time'].update(time.time() - prev_time)
                        
                    # Update agent memory
                    cumulative_obs[id][1] = corrected_obs
                    cumulative_obs[id][2] = action
                    
                    # Start new episode if not already tracked
                    if id not in running_episodes: running_episodes[id] = []
                    running_episodes[id].append({
                        'obs': corrected_obs,
                        'u_e': cumulative_obs[id][3],
                        'uf_activation': cumulative_obs[id][4],
                        'action': action,
                        'collision': False,
                        'collision_count': 0,
                        'collision_events': [],
                        'inner_steps': []
                    })

                # Use last predicted action by default
                policy_action = cumulative_obs[id][2] 

                # Control Barrier Function (CBF) correction
                cbf_action = np.zeros(2)
                atom_info = None
                if args.atom_cbf:
                    prev_time = time.time()
                    atom_result = atom_controller.filter(
                        actual_obs[
                            RAYCAST_SIZE*(STACK_NUMBER - 1):RAYCAST_SIZE*STACK_NUMBER
                        ],
                        policy_action,
                    )
                    cbf_action = atom_result.action
                    atom_info = atom_result.info
                    testing_stats['atom_cbf_time'].update(time.time() - prev_time)
                    testing_stats['atom_cbf_activation'].update(
                        float(atom_info['activated'])
                    )
                    testing_stats['atom_cbf_uncertainty'].update(
                        atom_info['uncertainty']
                    )
                    testing_stats['atom_cbf_epsilon'].update(
                        atom_info['epsilon_adapt']
                    )
                    testing_stats['atom_cbf_slack'].update(atom_info['slack'])
                    testing_stats['atom_cbf_solver_failure'].update(
                        float(atom_info['status'] == 'solver_failure')
                    )
                    testing_stats['atom_cbf_emergency'].update(
                        float(atom_info['status'] == 'emergency_inside_margin')
                    )
                elif args.cbf:
                    prev_time = time.time()
                    cbf_action = CBF_from_obs(
                        actual_obs[RAYCAST_SIZE*(STACK_NUMBER - 1):RAYCAST_SIZE*STACK_NUMBER], 
                        policy_action,
                        
                        other_config['raycast_length'],
                        agent_config['max_movement_speed'],
                        agent_config['max_turn_speed'],
                        
                        args.d_safe,
                        args.alpha,
                        args.d_safe_mul,
                        
                        angoli_radianti_precalcolati
                    )
                    testing_stats['cbf_time'].update(time.time() - prev_time)
                        
                    # Ensure minimum forward velocity
                    if policy_action[0] > args.cbf_min_forward_velocity:
                        cbf_action[0] = max(args.cbf_min_forward_velocity, cbf_action[0])
                    else:
                        cbf_action[0] = max(policy_action[0], cbf_action[0])
                            
                # Check if CBF activated
                cbf_activation = (
                    (args.cbf or args.atom_cbf)
                    and np.linalg.norm(cbf_action - policy_action) > 1e-06
                )
                # Final action selection (UF + CBF logic)
                final_action = policy_action
                if args.atom_cbf:
                    final_action = cbf_action
                elif cumulative_obs[id][4] and cbf_activation:
                    final_action = cbf_action

                running_episodes[id][-1]['inner_steps'].append({
                    'inner_step_index': len(running_episodes[id][-1]['inner_steps']),
                    'policy_action': policy_action.copy(),
                    'cbf_action': cbf_action.copy(),
                    'executed_action': final_action.copy(),
                    'cbf_action_delta': np.linalg.norm(cbf_action - policy_action),
                    'cbf_activation': cbf_activation,
                    'uf_activation': cumulative_obs[id][4],
                    'u_e': cumulative_obs[id][3],
                    'safety_filter': 'atom_cbf' if args.atom_cbf else (
                        'cbf' if args.cbf else 'none'
                    ),
                    'atom_cbf': atom_info,
                    'collision': False,
                    'collision_events': [],
                })
                        
                # Debug visualization (optional)
                if args.send_debug_action:
                    env_info.send_agent_action_debug(
                        agent_obs[4],
                        
                        final_action[0], 
                        final_action[1],
                        
                        policy_action[0],
                        policy_action[1], 
                        
                        cbf_activation, 
                        cbf_action[0], 
                        cbf_action[1],
                        
                        cumulative_obs[id][4]*100,
                        percentile_real_value,
                        cumulative_obs[id][3]*100
                    ) 
                                                        
                # Apply final action to environment
                a = ActionTuple(continuous=np.array([final_action]))
                env.set_action_for_agent(
                    BEHAVIOUR_NAME, id, a
                )
                
                # Increment frame counter
                cumulative_obs[id][0] += 1
                    
            
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

                # L'episodio non viene contato nelle metriche: rimpiazza il seed
                # scartato senza sottrarre uno dei seed originali 0..N-1.
                replacement_seed = args.seed + args.total_episodes + replacement_seeds_sent
                env_info.send_episode_seed(replacement_seed)
                replacement_seeds_sent += 1
            else:
                update_stats_from_message_rm(episodic_stats, None, None, msg)        
                if current_episode % args.metrics_log_interval == 0:
                    print_update_rm(current_episode, args.total_episodes, start_time, episodic_stats)
                
                current_episode += 1

                # Cerca questa parte nel tuo blocco if seed_sent < args.total_episodes:
                if seed_sent < args.total_episodes:
                    # CORRETTO: Usa seed_sent come offset, non current_episode
                    env_info.send_episode_seed(seed_sent + args.seed) 
                    seed_sent += 1
                    
                # Save data if required
                # saving all obseravtion + action and episode stats
                if args.accumulate_data:
                    dataset.append((t_episode, msg))
            
            del terminated_episodes[int(t_episode_index)]
        env_info.stop_msg_queue = new_stop_msgs
        
        # --- ENVIRONMENT STEP ---
        prev_time = time.time()
        env.step()
        testing_stats['unity_time'].update(time.time() - prev_time)
            
        # Safety check: queue should not grow indefinitely
        if len(env_info.stop_msg_queue) > 10:
            print('ERRORE')
            raise AssertionError('Unexpected queue growth')
    
    for key in testing_stats:
        testing_stats[key] = testing_stats[key].mean
    testing_stats["ep_count"] = current_episode - 1
    
    return testing_stats, episodic_stats, dataset


# [markdown]
#  Start Testing Code


args = parse_args()
if not hasattr(args, 'atom_cbf'):
    args.atom_cbf = False
if not hasattr(args, 'atom_checkpoint_path'):
    args.atom_checkpoint_path = None
if not hasattr(args, 'atom_solver'):
    args.atom_solver = 'CLARABEL'
if args.atom_cbf and args.cbf:
    raise ValueError('ATOM-CBF and the existing CBF cannot both control the robot')
if args.atom_cbf and args.uf:
    raise ValueError('The standalone ATOM-CBF baseline requires uf=false')
if args.atom_cbf and not args.atom_checkpoint_path:
    raise ValueError('atom_checkpoint_path is required when atom_cbf=true')
train_config = parse_config_file(args.train_config_path)
other_config = parse_config_file(train_config["other_config_path"])
agent_config = parse_config_file(args.agent_config_path)

# args.seed = random.randint(0, 2**16)
# args.name = generate_funny_name()

print('Testing with the following parameters:')
pprint(vars(args))
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

if type(args.build_path) == str:
    args.build_path = [args.build_path]
for b_path in args.build_path:
    
    # Create the channel
    env_info = CustomChannel(capture_collision_steps=True)
    param_channel = EnvironmentParametersChannel()
        
    # env setup
    print(f'Starting Unity Environment from build: {b_path}')
    env = UnityEnvironment(b_path,
                        seed=args.seed,
                        side_channels=[env_info, param_channel], 
                        no_graphics=args.headless,
                        worker_id=args.worker_id)

    print('Unity Environment connected.')
    if args.episode_queue_length > args.total_episodes:
        args.episode_queue_length = args.total_episodes

    if type(args.obstacles_config_path) == str:
        args.obstacles_config_path = [args.obstacles_config_path]
    for obs_config_path in args.obstacles_config_path:
        
        obstacles_config = parse_config_file(obs_config_path)
        print('obstacles_config:')
        pprint(obstacles_config)

        if args.atom_cbf:
            print('Loading calibrated ATOM-CBF baseline...')
            atom_controller = ATOMCBFController.from_checkpoint(
                args.atom_checkpoint_path,
                max_movement_speed=agent_config['max_movement_speed'],
                max_turn_speed_degrees=agent_config['max_turn_speed'],
                device=DEVICE,
                expected_ray_length=other_config['raycast_length'],
                expected_ray_angles=generate_angles_rad(
                    RAY_PER_DIRECTION,
                    other_config.get('raycast_max_degrees', 90),
                ),
                expected_d_safe=args.d_safe,
                expected_d_safe_multiplier=args.d_safe_mul,
                expected_cbf_gain=args.alpha,
                solver=args.atom_solver,
            )
        else:
            atom_controller = None
        
        if type(args.policy_names) == str:
            args.policy_names = [args.policy_names]
        for p_name in args.policy_names:
            
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
            if 'LAGPPO' in p_name:
                actor = LagPPOAgent(TOTAL_STATE_SIZE,
                                    ACTION_SIZE,
                                    ACTION_MIN,
                                    ACTION_MAX,
                                    256,
                ).to(DEVICE)
            elif 'PPO' in p_name:
                actor = PPOAgent(TOTAL_STATE_SIZE,
                                    ACTION_SIZE,
                                    ACTION_MIN,
                                    ACTION_MAX,
                                    256
                ).to(DEVICE)
            else:
                actor = OldDenseActor(
                    TOTAL_STATE_SIZE,
                    ACTION_SIZE,
                    ACTION_MIN,
                    ACTION_MAX,
                    [256, 256, 256]
                ).to(DEVICE)
            load_models(actor, save_path='./models/' + p_name, suffix='_best', DEVICE=DEVICE)
            
            if args.uf:
                if args.ue_action_type == 'distribution':
                    ens_input_dim = (21 + 7)*4 + 4
                else:
                    ens_input_dim = (21 + 7)*4 + 2
                
                ue = load_trained_ensemble(args.ue_path + 'unc_' + p_name, ens_input_dim, (21 + 7), DEVICE)[0]
                ue_norm = torch.load(args.ue_path + 'unc_' + p_name + '/norm.pth', map_location=DEVICE)
            else:
                ue = None
                ue_norm = None
                
            try:
                other_stats, episodic_stats, dataset = test(env, 
                    env_info,
                    param_channel,
                    
                    args,
                    agent_config,
                    obstacles_config,
                    other_config,
                    
                    actor, 
                    ue,
                    ue_norm,
                    atom_controller,
                    
                    BEHAVIOUR_NAME,
                    STATE_SIZE,
                    RAYCAST_SIZE,
                    train_config['input_stack'],
                    DEVICE)
                
            except Exception as e:
                # 1. Messaggio semplice
                print(f"Si è verificato un errore durante l'esecuzione: {e}")
                
                # 2. (Opzionale) Stampa il percorso completo dell'errore (Traceback)
                # Questo ti dice anche la riga esatta del file dove è successo il problema
                traceback.print_exc()
                
                # Chiudiamo l'ambiente e usciamo
                env.close()
                exit(1)
                
            other_stats['env_name'] = obs_config_path.split('/')[-1].split('.')[0]
            other_stats['policy_name'] = p_name
            other_stats['test_name'] = args.test_full_name
            
            other_stats['ut'] = args.ut
            other_stats['ue_action_type'] = args.ue_action_type
            
            print(f'Saving summary data to: {specific_save_filepath}')
            save_stats_to_csv(other_stats, episodic_stats, summary_save_filepath)
            
            # Save dataset to JSON if accumulation is enabled
            print(f'Saving accumulated dataset to path: {specific_save_filepath}')
            if args.accumulate_data: 
                
                d1 = {
                    'metadata': {
                        'test_config': vars(args),
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
                    elif isinstance(obj, np.ndarray):
                        return [convert_all_to_float(item) for item in obj.tolist()]
                    elif isinstance(obj, (np.floating, Decimal)):
                        return float(obj)
                    elif isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.bool_):
                        return bool(obj)
                    else:
                        return obj
                    
                # Save dataset with timestamp in filename
                with open(specific_save_filepath + '_info.json', 'w+') as file:
                    file.write(json.dumps(convert_all_to_float(d1)))
                with open(specific_save_filepath + '_transitions.json', 'w+') as file:
                    file.write(json.dumps(convert_all_to_float(d2)))

    env.close()
