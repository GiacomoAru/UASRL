import torch.nn as nn
import torch
import os
import numpy as np

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