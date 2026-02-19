import os
import torch
from testing_utils import *
from training_utils import *
from uncertainty_utils import *


base_path = 'UNC_NORM_DATA_TEST' #ATTENZIONE ORA é UNA STRINGA E NON UNA LISA ATTENZIONE!!!!!
p_names = ["NEW_TR_BASE_5746072", "NEW_TR_SIMPLE_5770545", "NEW_TR_SIMPLEWP_5819553", "NEW_TR_SIMPLE_EASY_5804798", "NEW_TR_SIMPLEWP_EASY_5841772"]
numbers = ['5944593','5944604','5947648', '5944599', '5946653']
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

    INPUT_STACK = infos[-1]['metadata']['train_config']['input_stack']
    TOTAL_STATE_SIZE = (STATE_SIZE + RAYCAST_SIZE)*INPUT_STACK

    print(f"Loading actor network")
    actors.append(OldDenseActor(
        TOTAL_STATE_SIZE,
        ACTION_SIZE,
        ACTION_MIN,
        ACTION_MAX,
        [256, 256, 256]
    ).to('cuda:0'))
    load_models(actors[-1], save_path='./models/' + p_names[i], suffix='_best', DEVICE='cuda:0')
    
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