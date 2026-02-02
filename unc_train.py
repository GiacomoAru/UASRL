import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import numpy as np
import os
import optuna
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader # Assicurati di importarlo
import json

from training_utils import *
from testing_utils import *

import argparse
import sys

def parse_args(default_config_path="./config/uncertainty_debug.yaml"):
    """
    Parse arguments from CLI or notebook.
    - In notebook: usa il default se non passato
    - In CLI: permette override dei parametri nel config
    """
    # --- Gestione notebook: evita crash su ipykernel args ---
    argv = sys.argv[1:]
    # Se siamo in notebook o non è passato il config_path, inseriamo il default
    if len(argv) == 0 or "--f=" in " ".join(argv):
        argv = [default_config_path]

    # --- Pre-parser per leggere il config_path ---
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "config_path",
        type=str,
        nargs="?",
        default=default_config_path,
        help="Main config file path"
    )
    initial_args, remaining_argv = pre_parser.parse_known_args(argv)
    CONFIG_PATH = initial_args.config_path
    print(f"Config path: {CONFIG_PATH}")

    # --- Legge parametri dal file di config ---
    file_config_dict = parse_config_file(CONFIG_PATH)

    # --- Parser principale ---
    parser = argparse.ArgumentParser(description="Training Script")
    parser.add_argument(
        "config_path",
        type=str,
        nargs="?",
        default=CONFIG_PATH,
        help="Main config file path"
    )

    # Aggiunge parametri dal config file, con tipi corretti
    for key, value in file_config_dict.items():
        if isinstance(value, bool):
            parser.add_argument(f"--{key}", type=str2bool, default=value)
        elif value is None:
            parser.add_argument(f"--{key}", type=str, default=value)
        else:
            parser.add_argument(f"--{key}", type=type(value), default=value)

    # --- Parse finale con remaining_argv per ignorare args extra Jupyter ---
    args, unknown = parser.parse_known_args(remaining_argv)
    if unknown:
        print("Ignored unknown args:", unknown)
    return args


# --- 1. IL MODELLO ---
class ProbabilisticNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), 
            nn.ReLU()
        )
        self.mu_head = nn.Linear(hidden_size, output_dim)
        self.logvar_head = nn.Linear(hidden_size, output_dim)
        
        # Limiti per stabilità numerica (Softplus)
        self.max_logvar = nn.Parameter(torch.ones(1, output_dim) / 2.0)
        self.min_logvar = nn.Parameter(-torch.ones(1, output_dim) * 10.0)

    def forward(self, x):
        features = self.network(x)
        mu = self.mu_head(features)
        logvar = self.logvar_head(features)
        
        # Clamping morbido
        logvar = self.max_logvar - nn.functional.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + nn.functional.softplus(logvar - self.min_logvar)
        return mu, logvar

# --- 2. EARLY STOPPING ---
class EarlyStopping:
    def __init__(self, patience=5, save_path=None):
        self.patience = patience
        self.counter = 0
        self.best_loss = np.Inf
        self.early_stop = False
        self.save_path = save_path
        
        if self.save_path:
            # os.path.dirname estrae la cartella dal path completo (es: "models/test.pth" -> "models")
            dir_name = os.path.dirname(self.save_path)
            
            # Creiamo la cartella solo se dir_name non è vuoto
            if dir_name and not os.path.exists(dir_name):
                print(f" Creazione cartella: {dir_name}")
                os.makedirs(dir_name, exist_ok=True) # exist_ok evita errori se la cartella appare nel mentre

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            if self.save_path:
                torch.save(model.state_dict(), self.save_path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def load_and_split_data(raw_data, 
                        actor_model, 
                        
                        RAYCASY_SIZE,
                        INPUT_STACK,
                        STATE_SIZE, 
                        DEVICE,
                        
                        shuffle=True):
    print(">>> Caricamento e Processamento Dati...")
    
    
    inputs_list = []
    outputs_list = []
    
    # Assicuriamoci che l'actor sia in modalità valutazione e non salvi i gradienti qui
    # Questo risparmia moltissima memoria e velocizza il caricamento
    actor_model.eval()
    
    print(f"Processing {len(raw_data)} episodes...")
    
    with torch.no_grad(): # DISATTIVA GRADIENTI PER VELOCITÀ
        for episode in raw_data:
            # episode[0] sono le osservazioni, episode[1] le info (che ignoriamo per ora)
            all_observations = episode[0]

            for t in range(len(all_observations) - 1):
                # 1. Recupera Input Corrente
                actual_obs_and_action = all_observations[t]
                
                # 2. Calcola Next Observation (Logica custom tua mantenuta)
                # Assumiamo che INPUT_STACK, RAYCASY_SIZE, STATE_SIZE siano costanti globali o in config
                # Se sono in config, usa config['input_stack'] etc.
                next_obs = all_observations[t + 1][(INPUT_STACK - 1)*RAYCASY_SIZE: (INPUT_STACK)*RAYCASY_SIZE] + all_observations[t + 1][-STATE_SIZE - 2:-2]
                
                # 3. Processamento Actor
                # Convertiamo l'input dell'actor in tensore (su DEVICE per l'inferenza veloce)
                obs_tensor = torch.FloatTensor(actual_obs_and_action[:-2])
                
                # L'actor restituisce una lista/tupla? La concateniamo.
                actor_distrib = actor_model(obs_tensor.to(DEVICE))
                
                # IMPORTANTE: .cpu() qui! Riportiamo il risultato in RAM per non intasare la GPU
                if isinstance(actor_distrib, (tuple, list)):
                    actor_distrib = torch.cat(actor_distrib).detach().cpu()
                else:
                    actor_distrib = actor_distrib.detach().cpu()
                
                # 4. Creazione Input Finale
                # Uniamo osservazione (convertita in tensor CPU) + distribuzione actor
                input_tensor = torch.cat([torch.FloatTensor(actual_obs_and_action), actor_distrib])
                output_tensor = torch.FloatTensor(next_obs)
                
                inputs_list.append(input_tensor)
                outputs_list.append(output_tensor)
    
    # --- 5. STACKING E DATASET CREATION ---
    print("Stacking dei tensori...")
    # Convertiamo la lista di tensori in UN unico tensore gigante [N_samples, Input_Dim]
    # Restiamo su CPU (.float() per precisione standard)
    X = torch.stack(inputs_list).float()
    y = torch.stack(outputs_list).float()
    
    input_dim = X.shape[1]
    output_dim = y.shape[1]
    print(f"Dataset Shape -> X: {X.shape}, y: {y.shape}")

    # --- 6. SPLITTING ---
    # Usiamo scikit-learn sui tensori CPU (funziona benissimo)
    # Split: 80% Train, 20% Temp
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=shuffle)
    # Split Temp: 10% Val, 10% Test (metà del 20%)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, shuffle=shuffle)
    
    print(f"Dataset Split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    # Ritorniamo i Tensori PURI (non DataLoader). 
    # I DataLoader li creiamo dentro il training loop (vedi sotto perché).
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), input_dim, output_dim

# --- 4. OPTIMIZATION LOOP (OPTUNA) ---
def objective(trial, train_data, val_data, input_dim, output_dim, args, DEVICE):
    # Suggerisci parametri
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    hidden_size = trial.suggest_categorical("hidden_size", [128, 256, 512])
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    
    model = ProbabilisticNetwork(input_dim, output_dim, hidden_size).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.GaussianNLLLoss().to(DEVICE)
    
    X_train, y_train = train_data
    X_val, y_val = val_data
    
    y_val = y_val.to(DEVICE)
    X_val = X_val.to(DEVICE)
    
    # Training Loop Breve
    for epoch in range(args.hpo_epochs):
        model.train()
        # Batching semplificato per HPO
        permutation = torch.randperm(X_train.size(0))
        batch_size = args.batch_size
        
        epoch_loss = 0
        for i in range(0, X_train.size(0), batch_size):
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = X_train[indices], y_train[indices]
            
            optimizer.zero_grad()
            mu, logvar = model(batch_x.to(DEVICE))
            loss = loss_fn(mu, batch_y.to(DEVICE), torch.exp(logvar))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        # Validation
        model.eval()
        with torch.no_grad():
            v_mu, v_logvar = model(X_val.to(DEVICE))
            val_loss = loss_fn(v_mu, y_val, torch.exp(v_logvar)).item()
        
        # Pruning (Optuna ferma i trial che vanno male subito)
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
    return val_loss

def train_ensemble(train_data, val_data, input_dim, output_dim, config, DEVICE):
    print(f"\n" + "="*40)
    print(f" FASE 2: TRAINING ENSEMBLE ({config['k_models_total']} MODELLI)")
    print("="*40)

    # 1. Preparazione Dati
    X_train, y_train = train_data 
    X_val, y_val = val_data      
    
    # Per la validazione usiamo tutto il set su GPU (se entra in memoria)
    X_val_gpu = X_val.to(DEVICE)
    y_val_gpu = y_val.to(DEVICE)

    loss_fn = nn.GaussianNLLLoss()
    mse_fn = nn.MSELoss() ### NEW: Serve per calcolare l'errore puro
    trained_model_infos = [] 
    
    os.makedirs("models", exist_ok=True)

    # 2. Loop sui K modelli dell'Ensemble
    for i in range(config['k_models_total']):
        
        print(f"\n--- Sampling Training Data Modello {i+1} ---")
        
        # --- IMPLEMENTAZIONE BOOTSTRAPPING ---
        num_samples = len(X_train)
        indices = torch.randint(0, num_samples, (num_samples,))
        
        X_train_boot = X_train[indices]
        y_train_boot = y_train[indices]
        
        train_dataset = TensorDataset(X_train_boot, y_train_boot)
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        # -------------------------------------
        
        # Inizializza modello e optimizer
        model = ProbabilisticNetwork(input_dim, output_dim, config['hidden_size']).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config.get('weight_decay', 0))
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

        if i == 0: 
            wandb.watch(model, log="gradients", log_freq=100)
            
        # Setup Early Stopping
        save_path = f"{config['save_path']}unc_{config['p_name']}_{i}_best.pth"
        stopper = EarlyStopping(patience=config['patience'], save_path=save_path)
        
        # 3. Training Loop (Epoche)
        for epoch in range(config['final_epochs']):
            model.train()
            epoch_nll_acc = 0.0
            epoch_mse_acc = 0.0 ### NEW: Accumulatore per MSE
            num_batches = 0
            
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(DEVICE)
                batch_y = batch_y.to(DEVICE)
                
                optimizer.zero_grad()
                mu, logvar = model(batch_x)
                
                # Calcolo Loss (Gaussian NLL) -> Per l'ottimizzazione
                loss = loss_fn(mu, batch_y, torch.exp(logvar))
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                # --- CALCOLI PER LOGGING ---
                # Usiamo no_grad per risparmiare memoria, calcoliamo MSE puro
                with torch.no_grad():
                    batch_mse = mse_fn(mu, batch_y)
                    epoch_mse_acc += batch_mse.item() ### NEW: Aggiornamento accumulatore
                
                epoch_nll_acc += loss.item()
                num_batches += 1
            
            # Medie per epoca
            avg_train_nll = epoch_nll_acc / num_batches
            avg_train_mse = epoch_mse_acc / num_batches
            
            # --- VALIDATION ---
            model.eval()
            with torch.no_grad():
                v_mu, v_logvar = model(X_val_gpu)
                v_var = torch.exp(v_logvar) ### NEW: Calcolo esplicito varianza per log
                
                val_nll = loss_fn(v_mu, y_val_gpu, v_var).item()
                val_mse = mse_fn(v_mu, y_val_gpu).item() ### NEW: Calcolo MSE validation
            
            scheduler.step(val_nll)

            # --- WANDB LOGGING AVANZATO ---
            metrics = {
                f"model_{i}/train_nll": avg_train_nll,
                f"model_{i}/train_mse": avg_train_mse,
                f"model_{i}/val_nll": val_nll,
                f"model_{i}/val_mse": val_mse,
                # Monitoraggio parametri interni
                f"model_{i}/max_logvar": model.max_logvar.mean().item(),
                f"model_{i}/min_logvar": model.min_logvar.mean().item(),
                # Media varianza predetta
                f"model_{i}/predicted_var_mean": v_var.mean().item(),
                # Monitoraggio Learning Rate
                f"model_{i}/lr": optimizer.param_groups[0]['lr'],
                "epoch": epoch
            }
            wandb.log(metrics)

            # Check Early Stopping
            stopper(val_nll, model)
            
            if stopper.early_stop:
                print(f"  -> Early stopping all'epoca {epoch}. Best Val NLL: {stopper.best_loss:.4f}")
                break
        
        # 4. Fine training modello corrente
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

with open(args.data_path, 'r') as f:
    data = json.load(f)

RAY_PER_DIRECTION = data['metadata']['other_config']['rays_per_direction']
RAYCAST_SIZE = 2*RAY_PER_DIRECTION + 1
STATE_SIZE = data['metadata']['other_config']['state_observation_size'] - 1

ACTION_SIZE = data['metadata']['other_config']['action_size']
ACTION_MIN = data['metadata']['other_config']['min_action']
ACTION_MAX = data['metadata']['other_config']['max_action']

INPUT_STACK = data['metadata']['train_config']['input_stack']
TOTAL_STATE_SIZE = (STATE_SIZE + RAYCAST_SIZE)*INPUT_STACK

actor = OldDenseActor(
    TOTAL_STATE_SIZE,
    ACTION_SIZE,
    ACTION_MIN,
    ACTION_MAX,
    data['metadata']['test_config']['policy_layers'][data['metadata']['test_config']['policy_names'].index(args.p_name)]
).to(DEVICE)

# 1. Dati
train_data, val_data, test_data, input_dim, output_dim = load_and_split_data(
                                                    data['data'],
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
torch.save(checkpoint, "final_pipeline_metadata.pt")


# wandb.finish()
print("PIPELINE COMPLETATA CORRETTAMENTE.")

