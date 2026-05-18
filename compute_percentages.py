# --- Librerie Standard e Utilità ---
import os
import json
import random
import numpy as np
from tqdm import tqdm

# --- Machine Learning e Processamento Dati ---
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

# --- Moduli Personalizzati ---
from training_utils import *
from testing_utils import *
from plotting_utils import *
from uncertainty_utils import *

# SAC, SACP, PPO, PPOP
OOD_1 = ['OOD_UECBF_TH090_2', 'OOD_UECBF_TH090_2', 'UECBF_TH080_PPO', 'UECBF_TH080_PPO']
OOD_3 = ['OOD_UECBF_TH090_2', 'OOD_UECBF_TH090_2', 'UECBF_TH085_PPO', 'UECBF_TH085_PPO']

ST_1 = ['ST_UECBF_TH095', 'ST_UECBF_TH095', 'ST_UECBF_TH080_PPO', 'ST_UECBF_TH080_PPO']
ST_2 = ['ST_UECBF_TH090', 'ST_UECBF_TH090', 'ST_UECBF_TH090_PPO', 'ST_UECBF_TH085_PPO']

SA = ['SA_UECBF_TH090_2', 'SA_UECBF_TH090_2', 'SA_UECBF_TH085_PPO', 'SA_UECBF_TH085_PPO']

quantili = [[90, 90, 80, 80], [90,90, 85, 85], [95, 95, 80, 80], [90, 90, 90, 85], [90, 90, 85, 85]]
labels = ['OOD_1', 'OOD_3', 'ST_1', 'ST_2', 'SA']
t = [OOD_1, OOD_3, ST_1, ST_2, SA]
lambdas = [ lambda x: x['collisions'] < 80, 
           lambda x: x['collisions'] < 80, 
           
           lambda x: x['collisions'] < 80 and x['global_avg_dist_obstacle'] == 0 and x['length'] > 35, 
           lambda x: x['collisions'] < 80 and x['global_avg_dist_obstacle'] == 2 and x['length'] > 35,
           
           lambda x: x['collisions'] < 80 and x['length'] > 35
           ]
env_names = ['obstacles_ood1', 'obstacles_ood3', None, None, None]


for dummy in t:
    dummy = [f'./remote_results/{x}.csv' for x in dummy]

metrics = ['success_nc','collision_rate', 'success', 'collisions', 'reward', 'length', 'velocity', 'distance_traveled', 'stuck_rate', 'vel_success', 'length_success']#  'weighted_success', 'SPL'
simple_metrics = ['success_nc_mean', 'success_nc_std', 'collision_rate_mean', 'collision_rate_std', 'stuck_rate_mean', 'stuck_rate_std',  'length_success_mean', 'length_success_std']

p_names_1 = [
 'NEW_TR_SIMPLE_EASY_5804798',
 'NEW_TR_SIMPLEWP_EASY_5841772',
 ]
p_names_2 = [
 'PPO_RETRAIN_7154081',
 'PPOWP_7167668'
 ]

sac_ue = load_trained_ensemble('./unc_models/' + 'unc_' + 'NEW_TR_SIMPLE_EASY_5804798', (21 + 7)*4 + 2, (21 + 7), 'cuda')[0]
sac_ue_norm = torch.load('./unc_models/' + 'unc_' + 'NEW_TR_SIMPLE_EASY_5804798' + '/norm.pth', map_location='cuda')

sacp_ue = load_trained_ensemble('./unc_models/' + 'unc_' + 'NEW_TR_SIMPLEWP_EASY_5841772', (21 + 7)*4 + 2, (21 + 7), 'cuda')[0]
sacp_ue_norm = torch.load('./unc_models/' + 'unc_' + 'NEW_TR_SIMPLEWP_EASY_5841772' + '/norm.pth', map_location='cuda')

ppo_ue = load_trained_ensemble('./unc_models/' + 'unc_' + 'PPO_RETRAIN_7154081', (21 + 7)*4 + 2, (21 + 7), 'cuda')[0]
ppo_ue_norm = torch.load('./unc_models/' + 'unc_' + 'PPO_RETRAIN_7154081' + '/norm.pth', map_location='cuda')

ppop_ue = load_trained_ensemble('./unc_models/' + 'unc_' + 'PPOWP_7167668', (21 + 7)*4 + 2, (21 + 7), 'cuda')[0]
ppop_ue_norm = torch.load('./unc_models/' + 'unc_' + 'PPOWP_7167668' + '/norm.pth', map_location='cuda')

def obtain_thresholds(dati, ue, norm, q):
    idc_unc_percentile = int(torch.argmin(torch.abs(norm['percentile_levels'] - q)))
    percentile_real_value = norm['epistemic']['percentiles'][idc_unc_percentile]
    
    ue_input = torch.tensor([riga for blocco in dati[0]['NEW_TR_SIMPLE_EASY_5804798'][1] for riga in blocco]).to('cuda')
    _, epistemic_unc = predict_uncertainty(ue, ue_input)
    
    perc = epistemic_unc > percentile_real_value
    return 1 - perc.sum() / len(epistemic_unc)
 
new_thresholds = {}
for macro_test, filtering, env_name, l, q in zip(t, lambdas, env_names, labels, quantili):
    sac = [load_test_from_csv(x, filtering, ['NEW_TR_SIMPLE_EASY_5804798'], env_name=env_name, transitions=True) for x in macro_test[0]]
    sacp = [load_test_from_csv(x, filtering, ['NEW_TR_SIMPLEWP_EASY_5841772'], env_name=env_name, transitions=True) for x in macro_test[1]]
    ppo = [load_test_from_csv(x, filtering, ['PPO_RETRAIN_7154081'], env_name=env_name, transitions=True) for x in macro_test[2]]
    ppop = [load_test_from_csv(x, filtering, ['PPOWP_7167668'], env_name=env_name, transitions=True) for x in macro_test[3]]

    new_thresholds[l] = {'sac': obtain_thresholds(sac, sac_ue, sac_ue_norm, q[0]),
                        'sacp': obtain_thresholds(sacp, sacp_ue, sacp_ue_norm, q[1]),
                        'ppo': obtain_thresholds(ppo, ppo_ue, ppo_ue_norm, q[2]),
                        'ppop': obtain_thresholds(ppop, ppop_ue, ppop_ue_norm, q[3])
                       }

with open('new_thresholds.json', 'w') as f:
    json.dump(new_thresholds, f, indent=4)