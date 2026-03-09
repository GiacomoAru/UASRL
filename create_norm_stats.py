import os
import torch
from testing_utils import *
from training_utils import *
from uncertainty_utils import *


base_path = 'DATA_NORM_PPO' #ATTENZIONE ORA é UNA STRINGA E NON UNA LISA ATTENZIONE!!!!!
p_names = ["PPOWP_7167668", "PPO_RETRAIN_7154081"]
numbers = ['7254512','7256469']
raws = []
infos = []
actors = []
ens = []
stats = []

for i, p in enumerate(p_names):
    specific = './results/' + base_path + '/' + base_path + '_' + numbers[i]
    with open(specific + '_transitions.json', 'r') as f:
        raws.append(json.load(f))
    with open(specific + '_info.json', 'r') as f:
        infos.append(json.load(f))
    
    RAY_PER_DIRECTION = infos[-1]['metadata']['other_config']['rays_per_direction']
    RAYCAST_SIZE = 2*RAY_PER_DIRECTION + 1
    STATE_SIZE = infos[-1]['metadata']['other_config']['state_observation_size'] - 1

    ACTION_SIZE = infos[-1]['metadata']['other_config']['action_size']
    ACTION_MIN = infos[-1]['metadata']['other_config']['min_action']
    ACTION_MAX = infos[-1]['metadata']['other_config']['max_action']

    INPUT_STACK = 4
    TOTAL_STATE_SIZE = (STATE_SIZE + RAYCAST_SIZE)*INPUT_STACK

    print(f"Loading actor network")
    if 'LAGPPO' in p:
        actor = LagPPOAgent(TOTAL_STATE_SIZE,
                            ACTION_SIZE,
                            ACTION_MIN,
                            ACTION_MAX,
                            256,
        ).to('cuda:0')
    elif 'PPO' in p:
        actor = PPOAgent(TOTAL_STATE_SIZE,
                            ACTION_SIZE,
                            ACTION_MIN,
                            ACTION_MAX,
                            256
        ).to('cuda:0')
    else:
        actor = OldDenseActor(
            TOTAL_STATE_SIZE,
            ACTION_SIZE,
            ACTION_MIN,
            ACTION_MAX,
            [256, 256, 256]
        ).to('cuda:0')
    load_models(actor, save_path='./models/' + p, suffix='_best', DEVICE='cuda:0')
    actors.append(actor)
    
    ens.append(load_trained_ensemble('./UE/unc_' + p_names[i], (21+7)*4+2, (21+7), 'cuda:0')[0])
    
    stats.append(generate_uncertainty_stats(
            raw_data=raws[i],
            actor_model=actors[i],
            ensemble_models=ens[i],
            RAYCASY_SIZE=RAYCAST_SIZE,
            INPUT_STACK=INPUT_STACK,
            DEVICE='cuda:0',
            explicit_transition=True,  # Se False, includerà l'output dell'attore nelle statistiche
            save_path='./UE/unc_' + p_names[i] + '/norm.pth'
        ))