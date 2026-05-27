# --- Librerie Standard e Utilità ---
import json
import pandas as pd

# --- Machine Learning e Processamento Dati ---
import torch
from sklearn.model_selection import train_test_split

# --- Moduli Personalizzati ---
from uncertainty_utils import *

def filter_and_enance_data(ep_list, filtering_function=lambda x: True, transitions=None):
    filtered_ep_list = []
    filtered_t_list = []
    for i, ep in enumerate(ep_list):
        if filtering_function(ep):
            if ep['length'] == 1999:
                ep['success'] = 0
            if ep['length'] == 0:
                ep['success'] = 0
                  
            # Calcolo metriche aggiuntive
            ep_ext = ep.copy()
            ep_ext['velocity'] = ep['distance_traveled'] / ep['length'] if ep['length'] > 0 else 0
            
            ep_ext['weighted_success'] = ep['success'] * ep['path_tortuosity']
            ep_ext['SPL'] = ep['success'] * (ep['path_length'] / max(ep['distance_traveled'], ep['path_length']))
            ep_ext['SPL2'] = ep['success'] * (ep['path_length'] / ep['distance_traveled']) if ep['distance_traveled'] > 0 else 0
            
            
            ep_ext['success_nc'] = ep['success'] if ep['collisions'] == 0 else 0
            ep_ext['stuck_rate'] = 1 if ep['success'] == 0 and ep['collisions'] == 0 else 0
            ep_ext['collision_rate'] = 1 if ep['collisions'] > 0 else 0
            
            ep_ext['vel_success'] = ep_ext['velocity'] if ep['success'] == 1 else None
            ep_ext['length_success'] = ep['length'] if ep['success'] == 1 else None
            
            filtered_ep_list.append(ep_ext)
            if transitions is not None:
                filtered_t_list.append(transitions[i])
    if transitions is not None:
        return filtered_ep_list, filtered_t_list
    else:
        return filtered_ep_list


def load_test_from_csv(csv_path, 
                       
                       filtering_function = lambda x: True, 
                       policy_order=None, 
                       env_name=None,
                       transitions=False):
    # 1. Caricamento del DataFrame
    control_df = pd.read_csv(csv_path)
    data = {}
    print(f'Loading data from {csv_path}')
    
    # --- Ordinamento personalizzato ---
    if policy_order is not None:
        # Diciamo a Pandas qual è l'ordine ufficiale per 'policy_name'
        control_df['policy_name'] = pd.Categorical(
            control_df['policy_name'], 
            categories=policy_order, 
            ordered=True
        )
        # Ordiniamo il dataframe in base a questa nuova categoria
        control_df = control_df.sort_values('policy_name')
        
        # Sicurezza: rimuoviamo le policy del CSV che non erano presenti nella tua lista
        # (altrimenti avrebbero valore NaN e farebbero crashare il ciclo)
        control_df = control_df.dropna(subset=['policy_name'])
    # ----------------------------------------

    # Ora il ciclo seguirà esattamente l'ordine che hai imposto!
    for p_name in control_df['policy_name']:
        if env_name is not None:
            control_row = control_df.query(f"policy_name == '{p_name}' & env_name == '{env_name}'")
        else:
            control_row = control_df.query(f"policy_name == '{p_name}'")
            
        specific_test_name = control_row['test_name'].values[0]
        json_path = csv_path.rsplit('.', 1)[0] + '/' + specific_test_name + '_info.json'
        with open(json_path, 'r') as f:
            specific_test_data = json.load(f)
            
        if transitions:
            json_t_path = csv_path.rsplit('.', 1)[0] + '/' + specific_test_name + '_transitions.json'
            with open(json_t_path, 'r') as f:
                specific_test_t_data = json.load(f)

            ep_data, t_data = filter_and_enance_data(specific_test_data['data'], filtering_function, specific_test_t_data)
            data[p_name] = (ep_data, t_data)
            
        else:
            ep_data = filter_and_enance_data(specific_test_data['data'], filtering_function)
            data[p_name] = ep_data
        
        print(f'\t{len(ep_data)} data for {p_name}')
    
    return data
 
 
 
# SAC, SACP, PPO, PPOP
OOD_1 = ['OOD_UECBF_TH090_2', 'OOD_UECBF_TH090_2', 'UECBF_TH080_PPO', 'UECBF_TH080_PPO'] # d_safe 0.25
OOD_3 = ['OOD_UECBF_TH090_2', 'OOD_UECBF_TH090_2', 'UECBF_TH085_PPO', 'UECBF_TH085_PPO'] # d_safe 0.25

ST_1 = ['ST_UECBF_TH095', 'ST_UECBF_TH095', 'ST_UECBF_TH080_PPO', 'ST_UECBF_TH080_PPO'] # d_safe 0.25
ST_2 = ['ST_UECBF_TH090', 'ST_UECBF_TH090', 'ST_UECBF_TH090_PPO', 'ST_UECBF_TH085_PPO'] # d_safe 0.25

SA = ['SA_UECBF_TH090_2', 'SA_UECBF_TH090_2', 'SA_UECBF_TH085_PPO', 'SA_UECBF_TH085_PPO'] # d_safe 0.2

quantili = [[0.90, 0.90, 0.80, 0.80], [0.90, 0.90, 0.85, 0.85], [0.95, 0.95, 0.80, 0.80], [0.90, 0.90, 0.90, 0.85], [0.90, 0.90, 0.85, 0.85]]
labels = ['OOD_1', 'OOD_3', 'ST_1', 'ST_2', 'SA']
t = [OOD_1, OOD_3, ST_1, ST_2, SA]
t = [[f'./results/{x}.csv' for x in gruppo] for gruppo in t]
lambdas = [ lambda x: x['collisions'] < 80, 
           lambda x: x['collisions'] < 80, 
           
           lambda x: x['collisions'] < 80 and x['global_avg_dist_obstacle'] == 0 and x['length'] > 35, 
           lambda x: x['collisions'] < 80 and x['global_avg_dist_obstacle'] == 2 and x['length'] > 35,
           
           lambda x: x['collisions'] < 80 and x['length'] > 35
           ]
env_names = ['obstacles_ood1', 'obstacles_ood3', None, None, None]

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

sac_ue = load_trained_ensemble('./UE/' + 'unc_' + 'NEW_TR_SIMPLE_EASY_5804798', (21 + 7)*4 + 2, (21 + 7), 'cuda')[0]
sac_ue_norm = torch.load('./UE/' + 'unc_' + 'NEW_TR_SIMPLE_EASY_5804798' + '/norm.pth', map_location='cuda')

sacp_ue = load_trained_ensemble('./UE/' + 'unc_' + 'NEW_TR_SIMPLEWP_EASY_5841772', (21 + 7)*4 + 2, (21 + 7), 'cuda')[0]
sacp_ue_norm = torch.load('./UE/' + 'unc_' + 'NEW_TR_SIMPLEWP_EASY_5841772' + '/norm.pth', map_location='cuda')

ppo_ue = load_trained_ensemble('./UE/' + 'unc_' + 'PPO_RETRAIN_7154081', (21 + 7)*4 + 2, (21 + 7), 'cuda')[0]
ppo_ue_norm = torch.load('./UE/' + 'unc_' + 'PPO_RETRAIN_7154081' + '/norm.pth', map_location='cuda')

ppop_ue = load_trained_ensemble('./UE/' + 'unc_' + 'PPOWP_7167668', (21 + 7)*4 + 2, (21 + 7), 'cuda')[0]
ppop_ue_norm = torch.load('./UE/' + 'unc_' + 'PPOWP_7167668' + '/norm.pth', map_location='cuda')

def obtain_thresholds(dati, ue, norm, q, p_name):
    idc_unc_percentile = int(torch.argmin(torch.abs(norm['percentile_levels'] - q)))
    percentile_real_value = norm['epistemic']['percentiles'][idc_unc_percentile]
    
    ue_input = torch.tensor([riga for blocco in dati[0][p_name][1] for riga in blocco]).to('cuda')
    _, epistemic_unc = predict_uncertainty(ue, ue_input)
    
    perc = epistemic_unc > percentile_real_value
    return 1 - (perc.sum() / len(epistemic_unc)).item()
 

new_thresholds = {}
for macro_test, filtering, env_name, l, q in zip(t, lambdas, env_names, labels, quantili):
    print(macro_test)
    new_thresholds[l] = {}
    
    sac = [load_test_from_csv(x, filtering, ['NEW_TR_SIMPLE_EASY_5804798'], env_name=env_name, transitions=True) for x in [macro_test[0]]]
    new_thresholds[l]['sac'] = obtain_thresholds(sac, sac_ue, sac_ue_norm, q[0], 'NEW_TR_SIMPLE_EASY_5804798')
    del sac
    
    sacp = [load_test_from_csv(x, filtering, ['NEW_TR_SIMPLEWP_EASY_5841772'], env_name=env_name, transitions=True) for x in [macro_test[1]]]
    new_thresholds[l]['sacp'] = obtain_thresholds(sacp, sacp_ue, sacp_ue_norm, q[1], 'NEW_TR_SIMPLEWP_EASY_5841772')
    del sacp
    
    ppo = [load_test_from_csv(x, filtering, ['PPO_RETRAIN_7154081'], env_name=env_name, transitions=True) for x in [macro_test[2]]]
    new_thresholds[l]['ppo'] = obtain_thresholds(ppo, ppo_ue, ppo_ue_norm, q[2], 'PPO_RETRAIN_7154081')
    del ppo
    
    ppop = [load_test_from_csv(x, filtering, ['PPOWP_7167668'], env_name=env_name, transitions=True) for x in [macro_test[3]]]
    new_thresholds[l]['ppop'] = obtain_thresholds(ppop, ppop_ue, ppop_ue_norm, q[3], 'PPOWP_7167668')
    del ppop
    

with open('new_thresholds.json', 'w') as f:
    json.dump(new_thresholds, f, indent=4)