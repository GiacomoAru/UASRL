# --- Librerie Standard e Utilità ---
import os
import json
import random
import numpy as np
from tqdm import tqdm

# --- Machine Learning e Processamento Dati ---
import torch

import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

# --- Ottimizzazione e Monitoraggio ---
import optuna
import wandb

# --- Moduli Personalizzati ---
from training_utils import *
from testing_utils import *
from uncertainty_utils import *


def objective(trial, train_data, val_data, input_dim, output_dim, args, DEVICE):
    # Parametri Optuna
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    hidden_size = trial.suggest_categorical("hidden_size", [128, 256, 512])
    batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024])
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    
    model = ProbabilisticNetwork(input_dim, output_dim, hidden_size).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.GaussianNLLLoss() # Non serve .to(DEVICE) per le loss function semplici
    
    # I DATI SONO GIA' SU GPU (dalla funzione load_and_split modificata)
    X_train, y_train = train_data
    X_val, y_val = val_data
    
    num_samples = X_train.size(0)

    for epoch in range(args.hpo_epochs):
        model.train()
        
        # Shuffle manuale veloce su GPU
        permutation = torch.randperm(num_samples, device=DEVICE)
        
        epoch_loss = 0
        num_batches = 0
        
        # Loop manuale sui batch (più veloce di DataLoader per tensori GPU)
        for i in range(0, num_samples, batch_size):
            indices = permutation[i : i + batch_size]
            batch_x = X_train[indices]
            batch_y = y_train[indices]
            
            optimizer.zero_grad()
            mu, logvar = model(batch_x)
            loss = loss_fn(mu, batch_y, torch.exp(logvar) + 1e-6)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
        # Validation
        model.eval()
        with torch.no_grad():
            v_mu, v_logvar = model(X_val)
            val_loss = loss_fn(v_mu, y_val, torch.exp(v_logvar) + 1e-6).item()
        
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
    return val_loss

def train_ensemble(train_data, val_data, input_dim, output_dim, config, DEVICE):
    print(f"\n" + "="*40)
    print(f" FASE 2: TRAINING ENSEMBLE ({config['k_models_total']} MODELLI) - GPU OPTIMIZED")
    print("="*40)

    # 1. Preparazione Dati (Assumiamo siano già Tensori su GPU dalla funzione load_and_split)
    X_train, y_train = train_data 
    X_val, y_val = val_data      
    
    # Calcoliamo la dimensione del dataset di training per il bootstrap
    num_samples = X_train.size(0)

    loss_fn = nn.GaussianNLLLoss()
    mse_fn = nn.MSELoss() 
    trained_model_infos = [] 
    
    os.makedirs("models", exist_ok=True)

    # --- BARRA ESTERNA (Loop sui Modelli) ---
    pbar_ensemble = tqdm(range(config['k_models_total']), desc="Ensemble Progress", unit="model")

    for i in pbar_ensemble:
        
        # --- IMPLEMENTAZIONE BOOTSTRAPPING (GPU) ---
        # Generiamo indici casuali direttamente su GPU (sostituisce il TensorDataset)
        boot_indices = torch.randint(0, num_samples, (num_samples,), device=DEVICE)
        
        # Creiamo le view per il training corrente (Indexing su GPU è immediato)
        X_train_boot = X_train[boot_indices]
        y_train_boot = y_train[boot_indices]
        
        # Inizializza modello e optimizer
        model = ProbabilisticNetwork(input_dim, output_dim, config['hidden_size']).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config.get('weight_decay', 0))
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

        # PRESERVATO: WandB Watch sul primo modello
        if i == 0: 
            wandb.watch(model, log="gradients", log_freq=100)
            
        # Setup Early Stopping
        save_dir = f"{config['save_path']}unc_{config['p_name']}"
        os.makedirs(save_dir, exist_ok=True) 
        save_path = f"{save_dir}/{i}_best.pth"
        
        stopper = EarlyStopping(patience=config['patience'], save_path=save_path)
        
        # --- BARRA INTERNA (Loop Epoche) ---
        pbar_epochs = tqdm(range(config['final_epochs']), 
                           desc=f"Model {i+1}/{config['k_models_total']}", 
                           leave=False,
                           unit="epoch")
        
        batch_size = config['batch_size']

        for epoch in pbar_epochs:
            
            model.train()
            epoch_nll_acc = 0.0
            epoch_mse_acc = 0.0 
            num_batches = 0
            
            # Shuffle per ogni epoca direttamente su GPU
            epoch_perm = torch.randperm(num_samples, device=DEVICE)
            
            # --- TRAINING LOOP MANUALE (NO DATALOADER) ---
            # Sostituisce: for batch_x, batch_y in train_loader
            for start_idx in range(0, num_samples, batch_size):
                # Prendi gli indici del batch corrente
                idx = epoch_perm[start_idx : start_idx + batch_size]
                
                # Slicing diretto su GPU (Zero Copy tra CPU e GPU)
                batch_x = X_train_boot[idx]
                batch_y = y_train_boot[idx]
                
                optimizer.zero_grad()
                mu, logvar = model(batch_x)
                
                # Calcolo Loss
                loss = loss_fn(mu, batch_y, torch.exp(logvar) + 1e-6)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                # --- CALCOLI PER LOGGING ---
                epoch_nll_acc += loss.item()
                with torch.no_grad():
                    batch_mse = mse_fn(mu, batch_y)
                    epoch_mse_acc += batch_mse.item() 
                
                num_batches += 1
            
            # Medie per epoca
            avg_train_nll = epoch_nll_acc / num_batches
            avg_train_mse = epoch_mse_acc / num_batches
            
            # --- VALIDATION (Full Batch su GPU) ---
            model.eval()
            with torch.no_grad():
                # X_val è già su DEVICE, passiamo tutto insieme (molto più veloce)
                v_mu, v_logvar = model(X_val) 
                v_var = torch.exp(v_logvar) 
                
                val_nll = loss_fn(v_mu, y_val, v_var + 1e-6).item()
                val_mse = mse_fn(v_mu, y_val).item() 
            
            scheduler.step(val_nll)

            # --- AGGIORNAMENTO BARRA TQDM ---
            pbar_epochs.set_postfix({
                "T_NLL": f"{avg_train_nll:.3f}", 
                "V_NLL": f"{val_nll:.3f}", 
                "Best": f"{stopper.best_loss:.3f}"
            })

            # --- PRESERVATO: WANDB LOGGING COMPLETO ---
            metrics = {
                f"ensemble/train_nll": avg_train_nll,
                f"ensemble/train_mse": avg_train_mse,
                f"ensemble/val_nll": val_nll,
                f"ensemble/val_mse": val_mse,
                # Log parametri interni per monitorare il collasso della varianza
                f"ensemble/max_logvar": model.max_logvar.mean().item(),
                f"ensemble/min_logvar": model.min_logvar.mean().item(),
                f"ensemble/predicted_var_mean": v_var.mean().item(),
                # Log Learning Rate corrente
                f"ensemble/lr": optimizer.param_groups[0]['lr'],
                "epoch": epoch
            }
            wandb.log(metrics)

            # Check Early Stopping
            stopper(val_nll, model)
            
            if stopper.early_stop:
                break
        
        # 4. Fine training modello corrente
        # Ricarichiamo i pesi migliori salvati dall'EarlyStopping
        model.load_state_dict(torch.load(save_path))
        
        trained_model_infos.append({
            "id": i,
            "best_val_loss": stopper.best_loss,
            "model": model,    
            "path": save_path
        })
        
    return trained_model_infos

args = parse_args()

if torch.cuda.is_available() and args.cuda >= 0:
    # F-string per inserire l'indice: diventa "cuda:2"
    device_str = f"cuda:{args.cuda}"
else:
    device_str = "cpu"
DEVICE = torch.device(device_str)
print(f"Using device: {DEVICE}")

splitted = args.data_test_name.rsplit('_', 1)
full_data_path = args.data_path + splitted[0] + '/' + args.data_test_name
print(f"Loading data from {full_data_path}...")
with open(full_data_path + '_transitions.json', 'r') as f:
    data = json.load(f)
with open(full_data_path + '_info.json', 'r') as f:
    info_test = json.load(f)

RAY_PER_DIRECTION = info_test['metadata']['other_config']['rays_per_direction']
RAYCAST_SIZE = 2*RAY_PER_DIRECTION + 1
STATE_SIZE = info_test['metadata']['other_config']['state_observation_size'] - 1

ACTION_SIZE = info_test['metadata']['other_config']['action_size']
ACTION_MIN = info_test['metadata']['other_config']['min_action']
ACTION_MAX = info_test['metadata']['other_config']['max_action']

INPUT_STACK = info_test['metadata']['train_config']['input_stack']
TOTAL_STATE_SIZE = (STATE_SIZE + RAYCAST_SIZE)*INPUT_STACK

print(f"Loading actor network")
actor = OldDenseActor(
    TOTAL_STATE_SIZE,
    ACTION_SIZE,
    ACTION_MIN,
    ACTION_MAX,
    info_test['metadata']['test_config']['policy_layers'][info_test['metadata']['test_config']['policy_names'].index(args.p_name)]
).to(DEVICE)
load_models(actor, save_path='./models/' + args.p_name, suffix='_best', DEVICE=DEVICE)

# 1. Dati
train_data, val_data, test_data, input_dim, output_dim = load_and_split_data(
                                                    data,
                                                    actor,
                                                    RAYCAST_SIZE,
                                                    INPUT_STACK,
                                                    STATE_SIZE,
                                                    DEVICE
                                                )

# ---------------------------------------------------------
# FASE 1: Hyperparameter Optimization (HPO)
# ---------------------------------------------------------
print("\n" + "="*40)
print(" FASE 1: RICERCA IPERPARAMETRI (Optuna)")
print("="*40)

study = optuna.create_study(direction="minimize")
study.optimize(lambda t: objective(t, train_data, val_data, input_dim, output_dim, args, DEVICE), 
                n_trials=args.hpo_trials)

best_params = study.best_params
print(f"\n>>> Migliori Parametri Trovati: {best_params}")

# Uniamo la config globale con i parametri ottimizzati
FINAL_CONFIG = vars(args).copy()
FINAL_CONFIG.update(best_params)

wandb.init(
    project=args.project_name,
    config=FINAL_CONFIG,
)

# ---------------------------------------------------------
# FASE 2: Training Ensemble Completo
# ---------------------------------------------------------
all_models_info = train_ensemble(train_data, val_data, input_dim, output_dim, FINAL_CONFIG, DEVICE)

# 4. SELEZIONE ELITE
print("\n" + "="*40)
print(" FASE 3: SELEZIONE ELITE")
# Ordiniamo in base alla validation loss ritornata dalla funzione
all_models_info.sort(key=lambda x: x["best_val_loss"])

# Prendiamo i primi N
elites_info = all_models_info[:FINAL_CONFIG["n_elites"]]
elite_indices = [m["id"] for m in elites_info]
elite_models = [m["model"] for m in elites_info]

print(f"Migliori modelli selezionati (ID): {elite_indices}")


# ---------------------------------------------------------
# FASE 3: Selezione Elite
# ---------------------------------------------------------
print("\n" + "="*40)
print(" FASE 3: SELEZIONE ELITE")
print("="*40)

# Ordina modelli per validation loss
all_models_info.sort(key=lambda x: x["best_val_loss"])

# Prendi i primi N
elites_info = all_models_info[:FINAL_CONFIG["n_elites"]]
elite_indices = [m["id"] for m in elites_info]
elite_models = [m["model"] for m in elites_info]

print(f"Migliori modelli selezionati (ID): {elite_indices}")
# wandb.log({"elite_indices": elite_indices})

# ---------------------------------------------------------
# FASE 4: Test e Incertezza
# ---------------------------------------------------------
print("\n" + "="*40)
print(" FASE 4: TEST SET & METRICHE INCERTEZZA")
print("="*40)

X_test, y_test = test_data
X_test = X_test.to(DEVICE)
y_test = y_test.to(DEVICE)

# Liste per raccogliere predizioni di tutti gli elite
mus_list = []
vars_list = []

with torch.no_grad():
    for model in elite_models:
        model.eval()
        mu, logvar = model(X_test)
        mus_list.append(mu.unsqueeze(0))         # [1, N_data, Dim]
        vars_list.append(torch.exp(logvar).unsqueeze(0))
        
# Stack: [N_Elites, N_data, Dim]
ensemble_mus = torch.cat(mus_list, dim=0)
ensemble_vars = torch.cat(vars_list, dim=0)

# Calcoli Mixture of Gaussians
# 1. Predizione finale (Media delle medie)
final_mean = torch.mean(ensemble_mus, dim=0)

# 2. Incertezza Aleatoria (Media delle varianze)
aleatoric = torch.mean(ensemble_vars, dim=0)

# 3. Incertezza Epistemica (Varianza delle medie)
epistemic = torch.var(ensemble_mus, dim=0, unbiased=False)

# 4. Errore MSE
mse = nn.MSELoss()(final_mean, y_test)

print(f"TEST MSE: {mse.item():.5f}")
print(f"Mean Aleatoric Unc: {aleatoric.mean().item():.5f}")
print(f"Mean Epistemic Unc: {epistemic.mean().item():.5f}")

# Log metriche finali
wandb.log({
    "test_mse": mse.item(),
    "aleatoric_uncertainty_mean": aleatoric.mean().item(),
    "epistemic_uncertainty_mean": epistemic.mean().item()
})

args.save_path

# ---------------------------------------------------------
# FASE 5: Salvataggio Finale
# ---------------------------------------------------------
print("\nSalvataggio Checkpoint Finale...")

checkpoint = {
    "config": FINAL_CONFIG,
    "elite_indices": elite_indices,
    "best_params": best_params,
    "test_metrics": {
        "mse": mse.item()
    }
}
torch.save(checkpoint, f"{args.save_path}unc_{args.p_name}/info.pth")

wandb.finish()
print("PIPELINE COMPLETATA CORRETTAMENTE.")

