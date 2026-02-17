import torch.nn as nn
import torch
import os
import numpy as np
import random

import matplotlib.pyplot as plt
from matplotlib.colors import (
    LinearSegmentedColormap,
    ListedColormap,
    LogNorm,
    PowerNorm,
)

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
                        shuffle=False):
    print(">>> Caricamento e Processamento Dati (Vettorializzato)...")
    
    # 1. SPLIT EPISODI
    all_episodes = list(raw_data) 
    if shuffle:
        print("Shuffling degli episodi...")
        random.shuffle(all_episodes)

    total_episodes = len(all_episodes)
    n_train = int(total_episodes * 0.8)
    n_val = int(total_episodes * 0.1)
    
    train_episodes = all_episodes[:n_train]
    val_episodes = all_episodes[n_train : n_train + n_val]
    test_episodes = all_episodes[n_train + n_val:]

    print(f"Split Episodi -> Train: {len(train_episodes)}, Val: {len(val_episodes)}, Test: {len(test_episodes)}")

    actor_model.eval()

    # --- FUNZIONE DI PROCESSAMENTO VETTORIALIZZATA ---
    def process_dataset_subset(episodes_subset, subset_name, explicit_transition=True):
        if not episodes_subset:
            return torch.tensor([]).to(DEVICE), torch.tensor([]).to(DEVICE)

        print(f"Processing {subset_name} ({len(episodes_subset)} episodes)...")

        # Liste temporanee per accumulare dati numpy (molto veloci su CPU)
        raw_obs_list = []
        raw_next_obs_list = []

        # Fase 1: Estrazione dati puri (tutto su CPU per ora)
        for all_observations in episodes_subset:
            if len(all_observations) < 2:
                continue
            
            # Trasformiamo l'intero episodio in array numpy
            obs_arr = np.array(all_observations)
            
            # Input attuali (tutti tranne l'ultimo)
            if explicit_transition:
                curr_obs = obs_arr[:-1, :]
            else:
                curr_obs = obs_arr[:-1, :-2]
                
            # Next observation (logica di slicing complessa preservata)
            idx_start = (INPUT_STACK - 1) * RAYCASY_SIZE
            idx_end = INPUT_STACK * RAYCASY_SIZE
            
            part1 = obs_arr[1:, idx_start:idx_end]
            part2 = obs_arr[1:, -STATE_SIZE - 2 : -2]
            
            next_obs = np.hstack([part1, part2])

            raw_obs_list.append(curr_obs)
            raw_next_obs_list.append(next_obs)

        # Concatenazione finale
        if not raw_obs_list:
             return torch.tensor([]).to(DEVICE), torch.tensor([]).to(DEVICE)

        X_raw = np.concatenate(raw_obs_list, axis=0)
        y_raw = np.concatenate(raw_next_obs_list, axis=0)

        # Fase 2: Spostamento su GPU
        X_tensor = torch.tensor(X_raw, dtype=torch.float32, device=DEVICE)
        y_tensor = torch.tensor(y_raw, dtype=torch.float32, device=DEVICE)

        if not explicit_transition:
            # Fase 3: Inferenza Actor (BATCHED su GPU)
            with torch.no_grad():
                # actor_distrib_tuple è una tupla: (mu, std) o simile
                actor_distrib_tuple = actor_model(X_tensor)
                
                # --- CORREZIONE QUI ---
                # Uniamo gli elementi della tupla (es. mu e std) in un unico tensore largo
                # dim=1 significa che li affianchiamo orizzontalmente
                actor_distrib_tensor = torch.cat(actor_distrib_tuple, dim=1)
                
            # Concatenazione finale su GPU: [Obs, Actor_Output_Unito]
            X_final = torch.cat([X_tensor, actor_distrib_tensor], dim=1)
        else:
            X_final = X_tensor
            
        return X_final, y_tensor

    # Eseguiamo il processamento
    X_train, y_train = process_dataset_subset(train_episodes, "Train")
    X_val, y_val = process_dataset_subset(val_episodes, "Validation")
    X_test, y_test = process_dataset_subset(test_episodes, "Test")

    input_dim = X_train.shape[1] if len(X_train) > 0 else 0
    output_dim = y_train.shape[1] if len(y_train) > 0 else 0
    
    print(f"Final Dataset Shapes (On {DEVICE}):")
    print(f"Train: X={X_train.shape}, y={y_train.shape}")
    print(f"Val:   X={X_val.shape}, y={y_val.shape}")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), input_dim, output_dim


def load_trained_ensemble(checkpoint_dir, input_dim, output_dim, DEVICE):
    """
    Carica i modelli elite in una lista pronta all'uso.
    """
    # Carica le info generali
    info_path = os.path.join(checkpoint_dir, "info.pth")
    if not os.path.exists(info_path):
        raise FileNotFoundError(f"Info file non trovato in {checkpoint_dir}")
    
    checkpoint_info = torch.load(info_path, map_location=DEVICE)
    config = checkpoint_info['config']
    elite_indices = checkpoint_info['elite_indices']
    
    loaded_elites = []
    
    print(f"Caricamento di {len(elite_indices)} modelli elite...")
    for idx in elite_indices:
        # Inizializza l'architettura (stessi parametri del training)
        model = ProbabilisticNetwork(input_dim, output_dim, config['hidden_size']).to(DEVICE)
        
        # Carica i pesi
        weight_path = os.path.join(checkpoint_dir, f"{idx}_best.pth")
        model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
        model.eval()
        loaded_elites.append(model)
        
    return loaded_elites, config     
 
def predict_uncertainty(ensemble, x_input):
    """
    Calcola mu, aleatoric e epistemic uncertainty.
    x_input: [Batch, Input_Dim]
    """
    mus = []
    variances = []
    
    with torch.no_grad():
        for model in ensemble:
            mu, logvar = model(x_input)
            mus.append(mu.unsqueeze(0))
            variances.append(torch.exp(logvar).unsqueeze(0))
    
    # Stack: [N_Models, Batch, Output_Dim]
    mus = torch.cat(mus, dim=0)
    variances = torch.cat(variances, dim=0)
    
    # 2. Incertezza Aleatoria (Rumore intrinseco nei dati)
    # formula: 1/M * sum(sigma_i^2)
    aleatoric_unc = torch.mean(variances, dim=0)
    
    # 3. Incertezza Epistemica (Disaccordo tra i modelli)
    # formula: 1/M * sum(mu_i^2) - (1/M * sum(mu_i))^2
    epistemic_unc = torch.var(mus, dim=0, unbiased=False)
    
    return aleatoric_unc.mean(dim=1), epistemic_unc.mean(dim=1)


# --- 2. Funzione Principale per le Statistiche ---
def generate_uncertainty_stats(raw_data, 
                               actor_model, 
                               ensemble_models,  # Lista dei modelli elite caricati
                               RAYCASY_SIZE, 
                               INPUT_STACK, 
                               DEVICE,
                               explicit_transition=True, 
                               batch_size=1024, # Fondamentale per non esplodere la VRAM
                               save_path="uncertainty_stats.pt"):
    
    print(">>> Calcolo Statistiche Incertezza (Aleatorica ed Epistemica)...")
    
    actor_model.eval()
    for model in ensemble_models:
        model.eval() # Assicuriamoci che l'ensemble sia in eval mode

    # Liste per accumulare Input (X)
    all_inputs_list = []  

    print(f"1. Preparazione Dataset ({len(raw_data)} episodi)...")

    # --- FASE 1: Preparazione Input (X) ---
    # Identica a prima, ci serve solo X per darlo in pasto all'ensemble
    for all_observations in raw_data:
        if len(all_observations) < 2:
            continue
        
        obs_arr = np.array(all_observations)
        
        # Slicing Input
        if explicit_transition:
            curr_obs = obs_arr[:-1, :]
        else:
            curr_obs = obs_arr[:-1, :-2]
        
        all_inputs_list.append(curr_obs)

    if not all_inputs_list:
        print("ERRORE: Nessun dato valido.")
        return None

    # Concatenazione numpy -> Tensor
    X_raw = np.concatenate(all_inputs_list, axis=0)
    X_tensor = torch.tensor(X_raw, dtype=torch.float32, device=DEVICE)

    # Integrazione Actor (Se necessaria)
    if not explicit_transition:
        print("   Integrazione azioni Actor...")
        # Anche qui usiamo il batching per sicurezza se il dataset è enorme
        X_final_parts = []
        with torch.no_grad():
            for i in range(0, len(X_tensor), batch_size):
                batch_x = X_tensor[i : i + batch_size]
                dist_tuple = actor_model(batch_x)
                dist_tensor = torch.cat(dist_tuple, dim=1)
                X_final_parts.append(torch.cat([batch_x, dist_tensor], dim=1))
        X_final = torch.cat(X_final_parts, dim=0)
    else:
        X_final = X_tensor

    print(f"   Dataset Input pronto: {X_final.shape}")

    # --- FASE 2: Inferenza Ensemble (Batched) ---
    print(f"2. Calcolo Incertezze tramite Ensemble (Batch Size: {batch_size})...")
    
    aleatoric_values = []
    epistemic_values = []
    
    total_samples = X_final.shape[0]
    
    with torch.no_grad():
        for i in range(0, total_samples, batch_size):
            # Prendiamo il batch corrente
            batch_x = X_final[i : i + batch_size]
            
            # Usiamo la funzione definita sopra
            ale_batch, epi_batch = predict_uncertainty(ensemble_models, batch_x)
            
            aleatoric_values.append(ale_batch)
            epistemic_values.append(epi_batch)
            
            if i % (batch_size * 50) == 0 and i > 0:
                print(f"   Processati {i}/{total_samples} campioni...")

    # Concateniamo tutti i risultati
    all_aleatoric = torch.cat(aleatoric_values, dim=0) # Shape: [N_samples]
    all_epistemic = torch.cat(epistemic_values, dim=0) # Shape: [N_samples]

    # --- FASE 3: Calcolo Statistiche e Percentili ---
    print("3. Calcolo Percentili e Statistiche...")
    
    # Percentili richiesti:
    low_extreme = np.linspace(0.001, 0.01, num=10, endpoint=False)
    middle = np.arange(0.01, 0.99, 0.01)
    high_extreme = np.linspace(0.99, 0.999, num=10)

    percentili_completi = np.unique(np.concatenate([low_extreme, middle, high_extreme]))
    percentili_finali = [round(p, 3) for p in percentili_completi]
    q_levels = torch.tensor(percentili_finali, device=DEVICE, dtype=torch.float32)
    
    stats = {
        "percentile_levels": torch.cat([torch.tensor([0.00], dtype=torch.float32, device=DEVICE), q_levels, torch.tensor([1.00], dtype=torch.float32, device=DEVICE)])
    }

    def calc_metrics(data_tensor, name, q_levels):
        # Assicuriamoci che q_levels sia tra 0.0 e 1.0 (escludendo 0 e 1)
        # Calcoliamo i quantili reali sui dati presenti
        mean_val = torch.mean(data_tensor)
        std_val = torch.std(data_tensor)
        
        # Calcolo quantili (es. dallo 0.1% al 99.9%)
        actual_quantiles = torch.quantile(data_tensor, q_levels)
        
        # AGGIUNTA: "Incolliamo" -inf e +inf agli estremi del risultato
        # Creiamo un nuovo tensore che ha: [-inf, ...quantili_reali..., inf]
        quantiles_with_limits = torch.cat([
            torch.tensor([-float('inf')], device=data_tensor.device),
            actual_quantiles,
            torch.tensor([float('inf')], device=data_tensor.device)
        ])
        
        print(f"   {name} -> Mean: {mean_val:.4f}, Std: {std_val:.4f}, Max: {data_tensor.max():.4f}")
        
        # Restituiamo la versione con i limiti
        return mean_val, std_val, quantiles_with_limits

    # Calcolo su Aleatoric
    al_mean, al_std, al_quant = calc_metrics(all_aleatoric, "Aleatoric", q_levels)
    stats["aleatoric"] = {
        "mean": al_mean,
        "std": al_std,
        "percentiles": al_quant
    }

    # Calcolo su Epistemic
    ep_mean, ep_std, ep_quant = calc_metrics(all_epistemic, "Epistemic", q_levels)
    stats["epistemic"] = {
        "mean": ep_mean,
        "std": ep_std,
        "percentiles": ep_quant
    }

    # --- SALVATAGGIO ---
    if save_path:
        torch.save(stats, save_path)
        print(f"Statistiche salvate in: {save_path}")

    return stats


def make_cmap_with_white(base_cmap='viridis', white_cut=0.05, cmap_range=(0.0, 1.0)):
    """
    Create a modified colormap with white replacing the lowest values.

    This function samples a base matplotlib colormap over a specified range 
    and replaces a fraction of the lowest values with pure white. It is useful 
    for visualizations where very low values should appear as background or 
    highlight regions.

    Parameters
    ----------
    base_cmap : str or Colormap, optional
        Name of the base colormap or a Colormap object (default is "viridis").
    white_cut : float, optional
        Fraction of the colormap (from the bottom) to replace with white, 
        between 0 and 1 (default is 0.05).
    cmap_range : tuple of float, optional
        Range of the base colormap to sample, as (min, max) values in [0, 1] 
        (default is (0.0, 1.0)).

    Returns
    -------
    matplotlib.colors.ListedColormap
        A modified colormap with the lowest values replaced by white.
    """

    # Get the base colormap
    base = plt.get_cmap(base_cmap)

    # Sample colors in the given range
    colors = base(np.linspace(cmap_range[0], cmap_range[1], 256))

    # Number of entries to replace with white
    n_white = int(256 * white_cut)
    if n_white > 0:
        colors[:n_white, :] = [1, 1, 1, 1]

    # Return modified colormap
    return ListedColormap(colors)


