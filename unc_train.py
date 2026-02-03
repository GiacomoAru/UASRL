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

# --- Ottimizzazione e Monitoraggio ---
import optuna
import wandb

# --- Moduli Personalizzati ---
from training_utils import *
from testing_utils import *

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
    print(">>> Caricamento e Processamento Dati (Split per Episodi)...")
    
    # 1. SPLIT DEGLI EPISODI (PRIMA DI TUTTO)
    # Copiamo raw_data per non modificare la lista originale fuori dalla funzione
    all_episodes = list(raw_data) 
    
    if shuffle:
        print("Shuffling degli episodi...")
        random.shuffle(all_episodes)

    total_episodes = len(all_episodes)
    n_train = int(total_episodes * 0.8) # 80%
    n_val = int(total_episodes * 0.1)   # 10%
    # Il restante 10% va al test

    train_episodes = all_episodes[:n_train]
    val_episodes = all_episodes[n_train : n_train + n_val]
    test_episodes = all_episodes[n_train + n_val:]

    print(f"Split Episodi -> Train: {len(train_episodes)}, Val: {len(val_episodes)}, Test: {len(test_episodes)}")

    # Assicuriamoci che l'actor sia in eval
    actor_model.eval()

    # --- FUNZIONE INTERNA PER PROCESSARE UNA LISTA DI EPISODI ---
    def process_dataset_subset(episodes_subset, subset_name):
        if not episodes_subset:
            print(f"Warning: {subset_name} set is empty!")
            return torch.tensor([]), torch.tensor([])

        inputs_list = []
        outputs_list = []
        
        print(f"Processing {subset_name} ({len(episodes_subset)} episodes)...")

        with torch.no_grad():
            for all_observations in episodes_subset:

                # Saltiamo episodi troppo corti se necessario, o gestiamo l'errore
                if len(all_observations) < 2:
                    continue

                for t in range(len(all_observations) - 1):
                    # --- A. Recupera Input Corrente ---
                    actual_obs = all_observations[t][:-2]
                    
                    # --- B. Calcola Next Observation (Logica Custom) ---
                    # Nota: Qui assumiamo che la struttura di episode[0] supporti questo slicing
                    next_obs = all_observations[t + 1][(INPUT_STACK - 1)*RAYCASY_SIZE: (INPUT_STACK)*RAYCASY_SIZE] + all_observations[t + 1][-STATE_SIZE - 2: - 2]
                    
                    # --- C. Processamento Actor ---
                    obs_tensor = torch.FloatTensor(actual_obs).to(DEVICE)
                    
                    actor_distrib = actor_model(obs_tensor)
                    actor_distrib = torch.cat(actor_distrib).detach().cpu()
                    
                    # --- D. Creazione Input Finale ---
                    # Riportiamo obs su CPU per unirlo
                    input_tensor = torch.cat([obs_tensor.cpu(), actor_distrib])
                    output_tensor = torch.FloatTensor(next_obs)
                    
                    inputs_list.append(input_tensor)
                    outputs_list.append(output_tensor)
        
        # Stacking finale per questo subset
        if len(inputs_list) > 0:
            X = torch.stack(inputs_list).float()
            y = torch.stack(outputs_list).float()
            return X, y
        else:
            return torch.tensor([]), torch.tensor([])

    # 2. ESEGUIAMO IL PROCESSAMENTO SUI 3 GRUPPI SEPARATI
    X_train, y_train = process_dataset_subset(train_episodes, "Train")
    X_val, y_val = process_dataset_subset(val_episodes, "Validation")
    X_test, y_test = process_dataset_subset(test_episodes, "Test")

    input_dim = X_train.shape[1] if len(X_train) > 0 else 0
    output_dim = y_train.shape[1] if len(y_train) > 0 else 0
    
    print(f"Final Dataset Shapes:")
    print(f"Train: X={X_train.shape}, y={y_train.shape}")
    print(f"Val:   X={X_val.shape}, y={y_val.shape}")
    print(f"Test:  X={X_test.shape}, y={y_test.shape}")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), input_dim, output_dim

# --- 4. OPTIMIZATION LOOP (OPTUNA) ---
def objective(trial, train_data, val_data, input_dim, output_dim, args, DEVICE):
    # Suggerisci parametri
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    hidden_size = trial.suggest_categorical("hidden_size", [128, 256, 512])
    batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024])
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
        batch_size = batch_size
        
        epoch_loss = 0
        for i in range(0, X_train.size(0), batch_size):
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = X_train[indices], y_train[indices]
            
            optimizer.zero_grad()
            mu, logvar = model(batch_x.to(DEVICE))
            loss = loss_fn(mu, batch_y.to(DEVICE), torch.exp(logvar) + 1e-6) # epsilon to avoid instability
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        # Validation
        model.eval()
        with torch.no_grad():
            v_mu, v_logvar = model(X_val.to(DEVICE))
            val_loss = loss_fn(v_mu, y_val, torch.exp(v_logvar) + 1e-6).item()
        
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
    mse_fn = nn.MSELoss() 
    trained_model_infos = [] 
    
    os.makedirs("models", exist_ok=True)

    # --- BARRA ESTERNA (Loop sui Modelli) ---
    # Monitora il progresso totale (es. 1/5, 2/5...)
    pbar_ensemble = tqdm(range(config['k_models_total']), desc="Ensemble Progress", unit="model")

    for i in pbar_ensemble:
        
        # --- IMPLEMENTAZIONE BOOTSTRAPPING ---
        num_samples = len(X_train)
        indices = torch.randint(0, num_samples, (num_samples,))
        
        X_train_boot = X_train[indices]
        y_train_boot = y_train[indices]
        
        train_dataset = TensorDataset(X_train_boot, y_train_boot)
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
        # -------------------------------------
        
        # Inizializza modello e optimizer
        model = ProbabilisticNetwork(input_dim, output_dim, config['hidden_size']).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config.get('weight_decay', 0))
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

        if i == 0: 
            wandb.watch(model, log="gradients", log_freq=100)
            
        # Setup Early Stopping
        # Correzione path: meglio assicurarsi che la cartella esista
        save_dir = f"{config['save_path']}unc_{config['p_name']}"
        os.makedirs(save_dir, exist_ok=True) 
        save_path = f"{save_dir}/{i}_best.pth"
        
        stopper = EarlyStopping(patience=config['patience'], save_path=save_path)
        
        # --- BARRA INTERNA (Loop Epoche) ---
        # leave=False fa sparire la barra quando il modello finisce
        pbar_epochs = tqdm(range(config['final_epochs']), 
                           desc=f"Model {i+1}/{config['k_models_total']}", 
                           leave=False,
                           unit="epoch")
        
        for epoch in pbar_epochs:
            
            model.train()
            epoch_nll_acc = 0.0
            epoch_mse_acc = 0.0 
            num_batches = 0
            
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(DEVICE)
                batch_y = batch_y.to(DEVICE)
                
                optimizer.zero_grad()
                mu, logvar = model(batch_x)
                
                # Calcolo Loss (Gaussian NLL) -> Per l'ottimizzazione
                loss = loss_fn(mu, batch_y, torch.exp(logvar) + 1e-6)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                # --- CALCOLI PER LOGGING ---
                with torch.no_grad():
                    batch_mse = mse_fn(mu, batch_y)
                    epoch_mse_acc += batch_mse.item() 
                
                epoch_nll_acc += loss.item()
                num_batches += 1
            
            # Medie per epoca
            avg_train_nll = epoch_nll_acc / num_batches
            avg_train_mse = epoch_mse_acc / num_batches
            
            # --- VALIDATION ---
            model.eval()
            with torch.no_grad():
                v_mu, v_logvar = model(X_val_gpu)
                v_var = torch.exp(v_logvar) 
                
                val_nll = loss_fn(v_mu, y_val_gpu, v_var + 1e-6).item()
                val_mse = mse_fn(v_mu, y_val_gpu).item() 
            
            scheduler.step(val_nll)

            # --- AGGIORNAMENTO BARRA TQDM ---
            # Questo mostra i numeri direttamente sulla barra di caricamento!
            pbar_epochs.set_postfix({
                "T_NLL": f"{avg_train_nll:.3f}", 
                "V_NLL": f"{val_nll:.3f}", 
                "Best": f"{stopper.best_loss:.3f}"
            })

            # --- WANDB LOGGING ---
            # Ho corretto 'ensamble' in 'ensemble' (typo comune)
            metrics = {
                f"ensemble/train_nll": avg_train_nll,
                f"ensemble/train_mse": avg_train_mse,
                f"ensemble/val_nll": val_nll,
                f"ensemble/val_mse": val_mse,
                f"ensemble/max_logvar": model.max_logvar.mean().item(),
                f"ensemble/min_logvar": model.min_logvar.mean().item(),
                f"ensemble/predicted_var_mean": v_var.mean().item(),
                f"ensemble/lr": optimizer.param_groups[0]['lr'],
                "epoch": epoch
            }
            wandb.log(metrics)

            # Check Early Stopping
            stopper(val_nll, model)
            
            if stopper.early_stop:
                # Opzionale: stampa se vuoi evidenziare lo stop
                # tqdm.write(f"-> Early stop Model {i+1} at epoch {epoch}")
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

