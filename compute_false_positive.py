import argparse
import sys
import time
import os
import random
import traceback
from collections import deque
from pprint import pprint
from sympy import Q
import wandb
import pandas as pd
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


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--policy",
        type=str,
        required=True,
        choices=["sac", "sacp", "ppo", "ppop", "all"],
        help="Policy da valutare: sac, sacp, ppo, ppop oppure all."
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=2048,
        help="Batch size per calcolo azioni e uncertainty."
    )

    parser.add_argument(
        "--max-episodes-k",
        type=int,
        default=100,
        help="Numero massimo di episodi da campionare per test."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=69,
        help="Seed per il campionamento degli episodi."
    )

    parser.add_argument(
        "--output",
        type=str,
        default="stats_reduced.json",
        help="File JSON di output."
    )

    parser.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="Salva un checkpoint ogni N policy completate. Default: 1."
    )
    
    return parser.parse_args()

def save_stats_checkpoint(stats, output_path):
    tmp_path = output_path + ".tmp"

    with open(tmp_path, "w") as f:
        json.dump(stats, f, separators=(",", ":"))

    os.replace(tmp_path, output_path)
    print(f"Checkpoint saved to {output_path}")
    
def compute_uf_cbf_activation_for_dataset(
    dataset,
    *,
    actor,
    unc_ensemble,
    unc_norm_stats,
    ut,
    max_episodes_k=None,
    seed=None,
    device=DEVICE,
    batch_size=2048,
    raycast_size=21,
    state_size=7,
    stack_number=4,
    actor_std=0.95,
    ue_action_type="sample",
    cbf_enabled=True,
    uf_enabled=True,
    d_safe=0.5,
    alpha=1.0,
    d_safe_mul=1.0,
    cbf_min_forward_velocity=0.0,
):
    results = []

    actor.eval()
    actor.to(device)

    if unc_ensemble is not None:
        for model in unc_ensemble:
            model.eval()
            model.to(device)

    obs_size = (raycast_size + state_size) * stack_number

    valid_episodes = [
        (episode_idx, episode)
        for episode_idx, episode in enumerate(dataset)
        if len(episode) > 4
    ]

    if max_episodes_k is not None and len(valid_episodes) > max_episodes_k:
        rng = random.Random(seed)
        valid_episodes = rng.sample(valid_episodes, max_episodes_k)

    flat_obs = []
    episode_lengths = []

    for _, episode in valid_episodes:
        episode_obs = []

        for transition in episode[4:]:
            obs = np.asarray(
                transition[:obs_size],
                dtype=np.float32,
            )
            episode_obs.append(obs)

        episode_lengths.append(len(episode_obs))

        if len(episode_obs) > 0:
            flat_obs.extend(episode_obs)

    total_transitions = len(flat_obs)

    if total_transitions == 0:
        return [[] for _ in valid_episodes]

    flat_obs_np = np.asarray(flat_obs, dtype=np.float32)

    percentile_real_value = None

    if uf_enabled:
        if unc_ensemble is None or unc_norm_stats is None:
            raise ValueError(
                "Per calcolare UF servono unc_ensemble e unc_norm_stats."
            )

        percentile_levels = torch.as_tensor(
            unc_norm_stats["percentile_levels"],
            dtype=torch.float32,
            device=device,
        )

        percentiles = torch.as_tensor(
            unc_norm_stats["epistemic"]["percentiles"],
            dtype=torch.float32,
            device=device,
        )

        idc_unc_percentile = int(
            torch.argmin(torch.abs(percentile_levels - ut)).item()
        )

        percentile_real_value = float(percentiles[idc_unc_percentile].item())

    angles_rad = None
    if cbf_enabled:
        angles_rad = generate_angles_rad(10, 90)

    all_policy_actions = np.zeros((total_transitions, 2), dtype=np.float32)
    all_uf_activation = np.zeros(total_transitions, dtype=np.int32)
    all_epistemic_unc = np.full(total_transitions, np.nan, dtype=np.float32)

    processed = 0

    with torch.inference_mode():
        for start in range(0, total_transitions, batch_size):
            end = min(start + batch_size, total_transitions)

            obs_batch = torch.as_tensor(
                flat_obs_np[start:end],
                dtype=torch.float32,
                device=device,
            )

            action_torch, _, _, action_mean, action_std = actor.get_action(
                obs_batch,
                actor_std,
            )

            all_policy_actions[start:end] = action_torch.detach().cpu().numpy()

            if uf_enabled:
                if ue_action_type == "distribution":
                    ue_input = torch.cat(
                        (obs_batch, action_mean, action_std),
                        dim=1,
                    ).to(dtype=torch.float32, device=device)
                else:
                    ue_input = torch.cat(
                        (obs_batch, action_torch),
                        dim=1,
                    ).to(dtype=torch.float32, device=device)

                _, epistemic_unc = predict_uncertainty(
                    unc_ensemble,
                    ue_input,
                )

                epistemic_unc = epistemic_unc.reshape(-1)

                epistemic_unc_np = epistemic_unc.detach().cpu().numpy()
                all_epistemic_unc[start:end] = epistemic_unc_np
                all_uf_activation[start:end] = (
                    epistemic_unc_np > percentile_real_value
                ).astype(np.int32)

            processed += end - start

            if processed % 10000 == 0 or processed == total_transitions:
                print(f"Computed NN batches: {processed}/{total_transitions} transitions")

    flat_results = []

    for idx in range(total_transitions):
        policy_action = all_policy_actions[idx]

        cbf_activation = False

        if cbf_enabled:
            obs = flat_obs_np[idx]

            last_raycast_obs = obs[
                raycast_size * (stack_number - 1):
                raycast_size * stack_number
            ]

            cbf_action = CBF_from_obs(
                last_raycast_obs,
                policy_action,
                3,
                1,
                90,
                d_safe,
                alpha,
                d_safe_mul,
                angles_rad,
            )

            if policy_action[0] > cbf_min_forward_velocity:
                cbf_action[0] = max(
                    cbf_min_forward_velocity,
                    cbf_action[0],
                )
            else:
                cbf_action[0] = max(
                    policy_action[0],
                    cbf_action[0],
                )

            cbf_delta = float(
                np.linalg.norm(cbf_action - policy_action)
            )

            cbf_activation = cbf_delta > 1e-6

        epistemic_unc_value = None

        if uf_enabled:
            epistemic_unc_value = round(float(all_epistemic_unc[idx]), 6)

        flat_results.append([
            int(all_uf_activation[idx]),
            int(cbf_activation),
            epistemic_unc_value,
        ])

        if (idx + 1) % 10000 == 0 or idx + 1 == total_transitions:
            print(f"Computed CBF: {idx + 1}/{total_transitions} transitions")

    cursor = 0

    for episode_len in episode_lengths:
        episode_results = flat_results[cursor:cursor + episode_len]
        results.append(episode_results)
        cursor += episode_len

    return results



def filter_and_enance_data(ep_list, filtering_function=lambda x: True, transitions=None):
    filtered_ep_list = []
    filtered_t_list = []
    for i, ep in enumerate(ep_list):
        if filtering_function(ep):

            filtered_ep_list.append(ep)
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
 

args = parse_args()
 
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

    
sac_actor = OldDenseActor(
    28*4,
    2,
    -1.0,
    1.0,
    [256, 256, 256]
).to(DEVICE)
load_models(sac_actor, save_path='./models/' + 'NEW_TR_SIMPLE_EASY_5804798', suffix='_best', DEVICE=DEVICE)
sac_ue = load_trained_ensemble('./UE/' + 'unc_' + 'NEW_TR_SIMPLE_EASY_5804798', (21 + 7)*4 + 2, (21 + 7), DEVICE)[0]
sac_ue_norm = torch.load('./UE/' + 'unc_' + 'NEW_TR_SIMPLE_EASY_5804798' + '/norm.pth', map_location=DEVICE)

sacp_actor = OldDenseActor(
    28*4,
    2,
    -1.0,
    1.0,
    [256, 256, 256]
).to(DEVICE)
load_models(sacp_actor, save_path='./models/' + 'NEW_TR_SIMPLEWP_EASY_5841772', suffix='_best', DEVICE=DEVICE)
sacp_ue = load_trained_ensemble('./UE/' + 'unc_' + 'NEW_TR_SIMPLEWP_EASY_5841772', (21 + 7)*4 + 2, (21 + 7), DEVICE)[0]
sacp_ue_norm = torch.load('./UE/' + 'unc_' + 'NEW_TR_SIMPLEWP_EASY_5841772' + '/norm.pth', map_location=DEVICE)

ppo_actor = PPOAgent(
    28*4,
    2,
    -1.0,
    1.0,
    256
).to(DEVICE)
load_models(ppo_actor, save_path='./models/' + 'PPO_RETRAIN_7154081', suffix='_best', DEVICE=DEVICE)
ppo_ue = load_trained_ensemble('./UE/' + 'unc_' + 'PPO_RETRAIN_7154081', (21 + 7)*4 + 2, (21 + 7), DEVICE)[0]
ppo_ue_norm = torch.load('./UE/' + 'unc_' + 'PPO_RETRAIN_7154081' + '/norm.pth', map_location=DEVICE)


ppop_actor = PPOAgent(
    28*4,
    2,
    -1.0,
    1.0,
    256
    ).to(DEVICE)
load_models(ppop_actor, save_path='./models/' + 'PPOWP_7167668', suffix='_best', DEVICE=DEVICE)
ppop_ue = load_trained_ensemble('./UE/' + 'unc_' + 'PPOWP_7167668', (21 + 7)*4 + 2, (21 + 7), DEVICE)[0]
ppop_ue_norm = torch.load('./UE/' + 'unc_' + 'PPOWP_7167668' + '/norm.pth', map_location=DEVICE)


policy_configs = {
    "sac": {
        "policy_name": "NEW_TR_SIMPLE_EASY_5804798",
        "actor": sac_actor,
        "ue": sac_ue,
        "ue_norm": sac_ue_norm,
        "macro_idx": 0,
    },
    "sacp": {
        "policy_name": "NEW_TR_SIMPLEWP_EASY_5841772",
        "actor": sacp_actor,
        "ue": sacp_ue,
        "ue_norm": sacp_ue_norm,
        "macro_idx": 1,
    },
    "ppo": {
        "policy_name": "PPO_RETRAIN_7154081",
        "actor": ppo_actor,
        "ue": ppo_ue,
        "ue_norm": ppo_ue_norm,
        "macro_idx": 2,
    },
    "ppop": {
        "policy_name": "PPOWP_7167668",
        "actor": ppop_actor,
        "ue": ppop_ue,
        "ue_norm": ppop_ue_norm,
        "macro_idx": 3,
    },
}



if args.policy == "all":
    selected_policies = ["sac", "sacp", "ppo", "ppop"]
else:
    selected_policies = [args.policy]


stats = {}

for macro_test, filtering, env_name, label, q in zip(
    t,
    lambdas,
    env_names,
    labels,
    quantili,
):
    print(macro_test)
    stats[label] = {}

    for policy_key in selected_policies:
        cfg = policy_configs[policy_key]

        policy_name = cfg["policy_name"]
        macro_idx = cfg["macro_idx"]

        print(f"Running {label} / {policy_key} / {policy_name}")

        policy_data = load_test_from_csv(
            macro_test[macro_idx],
            filtering,
            [policy_name],
            env_name=env_name,
            transitions=True,
        )

        policy_transitions = policy_data[policy_name][1]

        stats[label][policy_key] = compute_uf_cbf_activation_for_dataset(
            policy_transitions,
            actor=cfg["actor"],
            unc_ensemble=cfg["ue"],
            unc_norm_stats=cfg["ue_norm"],
            ut=q[macro_idx],
            max_episodes_k=args.max_episodes_k,
            seed=args.seed,
            batch_size=args.batch_size,
            device=DEVICE,
        )

        del policy_data, policy_transitions

        if DEVICE == "cuda":
            torch.cuda.empty_cache()


with open(args.output, "w") as f:
    json.dump(stats, f, separators=(",", ":"))
    

with open('stats_reduced.json', 'w') as f:
    json.dump(stats, f, separators=(',', ':'))