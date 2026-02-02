import os
import random
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.metrics import roc_curve, roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from training_utils import *


####################################################################################################
####################################################################################################

#   ╔══════════════════════╗
#   ║   Training Classes   ║
#   ╚══════════════════════╝


class SupervisedDataset(Dataset):
    """
    Custom PyTorch dataset for supervised learning from state vectors.

    This dataset splits raw state tensors into input features `x` and 
    target outputs `y`, based on predefined index ranges. It is designed 
    to train models that learn to predict specific parts of the state 
    from others.

    Parameters
    ----------
    states : torch.Tensor
        Raw state tensor of shape (N, D), where N is the number of samples 
        and D is the dimensionality of the state vector.

    Attributes
    ----------
    states_raw : torch.Tensor
        Original unmodified input tensor of shape (N, D).
    x : torch.Tensor
        Input features constructed by concatenating:
            - first 17*3 values,
            - 7*3 values after the ray section,
            - last 2 values.
    y : torch.Tensor
        Target outputs constructed by concatenating:
            - one block of 17 values,
            - one block of 7 values.

    Methods
    -------
    __len__()
        Return the number of samples in the dataset.
    __getitem__(idx)
        Return the (input, target) pair corresponding to the given index.
    """

    def __init__(self, states):
        # Save original raw states (Tensor of shape [N, D])
        self.states_raw = states

        # Input features: concatenate selected subsets of state vector
        # - first 17*3 values
        # - 7*3 values after the ray section
        # - last 2 values (actions or extra state info)
        self.x = torch.cat([
            states[:, :17*3],
            states[:, 17*4 : 17*4 + 7*3],
            states[:, -2:]
        ], dim=1)

        # Target output: concatenate other parts of the state vector
        # - one block of 17 values
        # - one block of 7 values
        self.y = torch.cat([
            states[:, 17*3:17*4],
            states[:, 17*4 + 7*3: 17*4 + 7*4]
        ], dim=1)

    def __len__(self):
        # Return number of samples
        return self.states_raw.shape[0]

    def __getitem__(self, idx):
        # Return (input, target) pair for given index
        return self.x[idx], self.y[idx]

class FlatDataset(Dataset):
    """
    A simple dataset wrapper for flat state tensors with optional normalization.

    This dataset stores raw state vectors and can normalize them using a provided
    mean and standard deviation. It supports retrieval of samples in normalized
    or raw form, as well as denormalization of normalized inputs.

    Parameters
    ----------
    states : torch.Tensor
        Tensor of shape [N, D] containing the dataset samples.
    mean : torch.Tensor or None, optional
        Mean values for normalization. If None, no normalization is applied.
    std : torch.Tensor or None, optional
        Standard deviation values for normalization. If None, no normalization is applied.

    Methods
    -------
    __len__():
        Return the number of samples in the dataset.
    __getitem__(idx):
        Return the normalized (or raw) sample at the given index.
    denormalize(x_norm):
        Convert a normalized sample back to the original scale.
    """

    def __init__(self, states, mean=None, std=None):
        # Store raw data
        self.states_raw = states
        
        # Normalize if mean and std are provided
        if mean is not None and std is not None:
            self.mean, self.std = mean, std
            self.states = (states - mean) / std
        else:
            self.mean, self.std = None, None
            self.states = states  # keep raw data

    def __len__(self):
        # Number of samples in the dataset
        return self.states.size(0)

    def __getitem__(self, idx):
        # Get one normalized (or raw) sample
        return self.states[idx]

    def denormalize(self, x_norm):
        # Convert a normalized sample back to original scale
        if self.mean is not None and self.std is not None:
            return x_norm * self.std + self.mean
        return x_norm

   
   
class ProbabilisticWorldModel(nn.Module):
    """
    Probabilistic neural network for world modeling.

    This model predicts both the mean and variance of the next state (or 
    target output) given an input state, enabling uncertainty-aware 
    predictions. It consists of a stack of fully connected hidden layers, 
    followed by two separate heads for mean and variance estimation.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input features.
    output_dim : int
        Dimensionality of the predicted output.
    hidden_layers : list of int, optional
        Sizes of the hidden fully connected layers (default is [128, 128]).
    stable_variant : bool, optional
        If True, adds a small epsilon (1e-6) to the variance output for 
        numerical stability (default is False).

    Attributes
    ----------
    stable_variant : bool
        Whether to apply extra stabilization to the variance.
    hidden_layers : nn.ModuleList
        Sequence of fully connected layers with ReLU activations.
    mean_head : nn.Linear
        Final linear layer producing the mean of the predictive distribution.
    logvar_head : nn.Linear
        Final linear layer producing the raw log-variance before transformation.

    Methods
    -------
    forward(x)
        Compute the forward pass. Returns the predicted mean and variance 
        (variance obtained by applying a softplus transformation to the 
        log-variance output).
    """

    def __init__(self, input_dim, output_dim, hidden_layers=[128, 128], stable_variant=False):
        super().__init__()

        self.stable_variant = stable_variant

        # Define fully connected hidden layers
        dims = [input_dim] + hidden_layers
        self.hidden_layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.hidden_layers.append(nn.Linear(dims[i], dims[i + 1]))

        # Two separate heads: one for mean, one for log-variance
        self.mean_head = nn.Linear(dims[-1], output_dim)
        self.logvar_head = nn.Linear(dims[-1], output_dim)

    def forward(self, x):
        # Pass input through hidden layers with ReLU activations
        for layer in self.hidden_layers:
            x = F.relu(layer(x))

        # Predict mean of the output distribution
        mean = self.mean_head(x)

        # Predict log-variance (raw) and transform with softplus for positivity
        raw_var = self.logvar_head(x)
        if self.stable_variant:
            # Extra epsilon added for numerical stability
            var = F.softplus(raw_var) + 1e-6
        else:
            var = F.softplus(raw_var)

        return mean, var
  
class MCDropoutWorldModel(nn.Module):
    """
    Monte Carlo Dropout world model for uncertainty estimation.

    This model uses dropout at both training and test time to produce 
    stochastic predictions. By performing multiple forward passes with 
    dropout enabled, it estimates both the predictive mean and variance 
    of the output distribution.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input features.
    output_dim : int
        Dimensionality of the predicted output.
    hidden_layers : list of int, optional
        Sizes of the hidden fully connected layers (default is [256, 128]).
    dropout_prob : float, optional
        Dropout probability applied after each hidden layer (default is 0.1).

    Attributes
    ----------
    layers : nn.ModuleList
        Sequence of fully connected layers with ReLU activations.
    dropouts : nn.ModuleList
        Sequence of Dropout layers applied after each hidden layer.
    output_layer : nn.Linear
        Final linear layer producing the output prediction.

    Methods
    -------
    forward(x)
        Compute the forward pass with ReLU activations and dropout. 
        Dropout remains active also during test time.
    predict(x, n_samples=10)
        Perform `n_samples` stochastic forward passes with dropout to 
        estimate predictive mean, variance, and return all predictions.
        Returns (mean, var, preds).
    """

    def __init__(self, input_dim, output_dim, hidden_layers=[256, 128], dropout_prob=0.1):
        super().__init__()

        # Define hidden layers and corresponding dropout layers
        dims = [input_dim] + hidden_layers
        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))
            self.dropouts.append(nn.Dropout(p=dropout_prob))

        # Final linear layer for output prediction
        self.output_layer = nn.Linear(dims[-1], output_dim)

    def forward(self, x):
        # Forward pass with ReLU and Dropout (dropout also active at test time)
        for layer, dropout in zip(self.layers, self.dropouts):
            x = F.relu(layer(x))
            x = dropout(x)
        return self.output_layer(x)

    def predict(self, x, n_samples=10):
        # Keep dropout active by setting model in train mode
        self.train()

        preds = []
        # Disable gradient tracking since only uncertainty estimation is needed
        with torch.no_grad():
            for _ in range(n_samples):
                preds.append(self.forward(x).unsqueeze(0))  # collect multiple stochastic passes
        preds = torch.cat(preds, dim=0)

        # Monte Carlo estimate of mean and variance
        mean = preds.mean(dim=0)
        var = preds.var(dim=0, unbiased=False)

        return mean, var, preds

class RNDNetwork(nn.Module):
    """
    Random Network Distillation (RND) neural network.

    This feedforward network is used in the RND framework for 
    novelty or uncertainty estimation. It produces feature embeddings from 
    input states that can be compared between a fixed target network and a 
    trainable predictor network.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input features.
    hidden_layers : list of int, optional
        Sizes of the hidden fully connected layers (default is [256, 128]).

    Attributes
    ----------
    layers : nn.ModuleList
        Sequence of fully connected layers. ReLU activations are applied 
        after all but the last layer.

    Methods
    -------
    forward(x)
        Compute the forward pass. Applies ReLU activation to all hidden 
        layers except the last one, which remains linear to produce the 
        final feature embedding.
    """

    def __init__(self, input_dim, hidden_layers=[256, 128]):
        super().__init__()

        # Define fully connected layers
        dims = [input_dim] + hidden_layers
        self.layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))

    def forward(self, x):
        # Apply ReLU to all hidden layers except the last one
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        # Last layer without activation (linear output)
        x = self.layers[-1](x)
        return x

     

####################################################################################################
####################################################################################################

#   ╔═══════════════════════╗
#   ║   Data Manipolation   ║
#   ╚═══════════════════════╝

def split_dataset(data, seed=2205, shuffle=True):
    """
    Split a dataset into train, validation, calibration, and test sets.

    The dataset is divided into four subsets with fixed proportions:
    - Train: 50%
    - Validation: 15%
    - Calibration: 25%
    - Test: 10%

    A fixed random seed ensures reproducibility of the splits.

    Parameters
    ----------
    data : torch.Tensor
        Input dataset as a tensor of shape (N, ...), where N is the number of samples.
    seed : int, optional
        Random seed used to generate reproducible splits (default is 2205).
    shuffle : bool, optional
        If True, shuffle the dataset before splitting (default is True).

    Returns
    -------
    list of torch.Tensor
        A list containing four tensors:
        [train_data, val_data, calib_data, test_data].
    """

    # Get total number of samples
    data_len = data.size(0)

    # Optionally shuffle the entire dataset before splitting
    if shuffle:
        data = data[torch.randperm(data_len)]

    # Compute sizes for train (50%), validation (15%), calibration (25%), and test (10%)
    train_size = int(data_len * 0.5)
    val_size = int(data_len * 0.15)
    calib_size = int(data_len * 0.25)
    test_size = int(data_len * 0.1)

    # Adjust train size to fix rounding mismatches
    diff = data_len - (train_size + val_size + calib_size + test_size)
    train_size += diff

    # Generate deterministic random indices using fixed seed
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(data_len, generator=generator)

    # Slice indices for each split
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    calib_indices = indices[train_size + val_size:train_size + val_size + calib_size]
    test_indices = indices[train_size + val_size + calib_size:]

    # Create dataset splits by indexing into original data
    train_data = data[train_indices]
    val_data = data[val_indices]
    calib_data = data[calib_indices]
    test_data = data[test_indices]

    return [train_data, val_data, calib_data, test_data]


####################################################################################################
####################################################################################################

#   ╔════════════════════╗
#   ║   Training Utils   ║
#   ╚════════════════════╝


def calibration_and_sharpness(mu, var, y, conf_levels=None, device='cpu'):
    """
    Compute calibration error and sharpness for probabilistic predictions.

    Calibration measures how well predicted distributions match observed 
    frequencies at different confidence levels. Sharpness quantifies the 
    concentration of the predictive distributions (average variance).

    Parameters
    ----------
    mu : torch.Tensor
        Predicted means of shape (N,).
    var : torch.Tensor
        Predicted variances of shape (N,).
    y : torch.Tensor
        Ground-truth targets of shape (N,).
    conf_levels : torch.Tensor or None, optional
        Confidence levels in [0, 1] at which to evaluate calibration. 
        If None, defaults to 19 levels uniformly spaced between 0.05 and 0.95.
    device : str, optional
        Device on which computations are performed (default is "cpu").

    Returns
    -------
    calib_error : float
        Weighted squared error between desired and empirical coverage frequencies.
    sharpness : float
        Average predicted variance across all samples.
    """

    # Define default confidence levels if none are provided
    if conf_levels is None:
        conf_levels = torch.linspace(0.05, 0.95, 19, device=device)

    # Convert variance to standard deviation, avoid zeros
    std = torch.clamp(torch.sqrt(var), min=1e-6)

    # Standardize residuals
    z = (y - mu) / std

    # Compute CDF values under a standard normal
    cdf_vals = 0.5 * (1 + torch.erf(z / np.sqrt(2)))

    calib_freqs = []  # empirical coverage frequencies
    weights = []      # number of samples per confidence level

    # Compute empirical frequencies at each confidence level
    for p in conf_levels:
        mask = (cdf_vals <= p)
        count = mask.sum().item()
        freq = mask.float().mean().item()
        calib_freqs.append(freq)
        weights.append(count)

    # Convert to tensors and normalize weights
    calib_freqs = torch.tensor(calib_freqs, device=device)
    weights = torch.tensor(weights, dtype=torch.float32, device=device)
    weights /= weights.sum()

    # Weighted squared error between desired and empirical frequencies
    calib_error = (weights * (conf_levels - calib_freqs) ** 2).sum().item()

    # Sharpness = average predicted variance
    sharpness = var.mean().item()

    return calib_error, sharpness


def train_world_model_prob(prefix_name, train_loader, val_loader, device, hidden_layers=[128, 128],
                           stable_variant=True, max_epochs=100, learning_rate=1e-3,
                           weight_decay=1e-5, patience=5):
    """
    Train a probabilistic world model with Gaussian NLL loss and early stopping.

    The model is a `ProbabilisticWorldModel` that predicts both mean and 
    variance for uncertainty-aware regression. Training logs include NLL, MSE, 
    calibration error, and sharpness, stored in TensorBoard. Early stopping 
    is applied based on validation NLL.

    Parameters
    ----------
    prefix_name : str
        Prefix used to generate a unique run name for logs and saved checkpoints.
    train_loader : torch.utils.data.DataLoader
        Dataloader providing training batches (inputs, targets).
    val_loader : torch.utils.data.DataLoader
        Dataloader providing validation batches (inputs, targets).
    device : str or torch.device
        Device used for training (e.g., "cpu" or "cuda").
    hidden_layers : list of int, optional
        Sizes of hidden layers in the world model (default is [128, 128]).
    stable_variant : bool, optional
        If True, applies an epsilon to the variance for numerical stability 
        (default is True).
    max_epochs : int, optional
        Maximum number of training epochs (default is 100).
    learning_rate : float, optional
        Learning rate for the Adam optimizer (default is 1e-3).
    weight_decay : float, optional
        Weight decay (L2 regularization) for the optimizer (default is 1e-5).
    patience : int, optional
        Number of epochs without improvement before early stopping (default is 5).

    Returns
    -------
    None
        The trained model is saved to disk, and training/validation metrics 
        are logged to TensorBoard. Best metrics are also recorded in the logger.
    """

    # Create unique run name and paths for logging and saving
    run_name = f"{prefix_name}_{int(time.time())}_{generate_funny_name()}"
    writer = SummaryWriter(log_dir=f"./u_e_test/unc_train/{run_name}")
    save_path = f"./u_e_test/unc_models/{run_name}.pth"

    # Infer input and output dimensions from dataloader
    input_dim = next(iter(train_loader))[0].shape[1]
    output_dim = next(iter(train_loader))[1].shape[1]

    # Initialize model, optimizer, and Gaussian NLL loss
    model = ProbabilisticWorldModel(input_dim, output_dim, hidden_layers, stable_variant).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = torch.nn.GaussianNLLLoss(full=True)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_metrics = None

    # Print header for logging
    print(f"{'Ep':>3} | {'Train NLL':>10} ± {'Std':>5} | {'MSE':>8} | {'Calib':>8} | {'Sharp':>8} | "
          f"{'Val NLL':>10} ± {'Std':>5} | {'MSE':>8} | {'Calib':>8} | {'Sharp':>8} | {'Time(s)':>7}")

    for epoch in range(max_epochs):
        start = time.time()

        # ===== TRAINING PHASE =====
        model.train()
        train_nlls, train_mse = [], []
        mu_all, var_all, y_all = [], [], []

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            mu, var = model(x)
            loss = loss_fn(mu, y, var)
            loss.backward()
            optimizer.step()

            train_nlls.append(loss.item())
            train_mse.extend(((mu - y) ** 2).mean(dim=1).detach().cpu().numpy())
            mu_all.append(mu)
            var_all.append(var)
            y_all.append(y)

        # Aggregate metrics
        mu_cat, var_cat, y_cat = torch.cat(mu_all), torch.cat(var_all), torch.cat(y_all)
        train_nll = np.mean(train_nlls); train_nll_std = np.std(train_nlls)
        train_mse = np.mean(train_mse)
        train_calib, train_sharp = calibration_and_sharpness(mu_cat, var_cat, y_cat, device=device)

        # ===== VALIDATION PHASE =====
        model.eval()
        val_nlls, val_mse = [], []
        mu_all, var_all, y_all = [], [], []

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                mu, var = model(x)
                loss = loss_fn(mu, y, var)
                val_nlls.append(loss.item())
                val_mse.extend(((mu - y) ** 2).mean(dim=1).detach().cpu().numpy())
                mu_all.append(mu)
                var_all.append(var)
                y_all.append(y)

        # Aggregate metrics
        mu_cat, var_cat, y_cat = torch.cat(mu_all), torch.cat(var_all), torch.cat(y_all)
        val_nll = np.mean(val_nlls); val_nll_std = np.std(val_nlls)
        val_mse = np.mean(val_mse)
        val_calib, val_sharp = calibration_and_sharpness(mu_cat, var_cat, y_cat, device=device)

        duration = time.time() - start

        # ===== LOGGING =====
        writer.add_scalar("Train/NLL", train_nll, epoch)
        writer.add_scalar("Train/NLL_std", train_nll_std, epoch)
        writer.add_scalar("Train/MSE", train_mse, epoch)
        writer.add_scalar("Train/Calibration", train_calib, epoch)
        writer.add_scalar("Train/Sharpness", train_sharp, epoch)

        writer.add_scalar("Val/NLL", val_nll, epoch)
        writer.add_scalar("Val/NLL_std", val_nll_std, epoch)
        writer.add_scalar("Val/MSE", val_mse, epoch)
        writer.add_scalar("Val/Calibration", val_calib, epoch)
        writer.add_scalar("Val/Sharpness", val_sharp, epoch)

        writer.add_scalar("Time/epoch_seconds", duration, epoch)

        # ===== EARLY STOPPING =====
        if val_nll < best_val_loss:
            best_val_loss = val_nll
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
            status = "saved"

            # Save metrics for best checkpoint
            best_metrics = {
                "Val_NLL": val_nll, "Val_NLL_std": val_nll_std, "Val_MSE": val_mse,
                "Val_Calibration": val_calib, "Val_Sharpness": val_sharp,
                "Train_NLL": train_nll, "Train_NLL_std": train_nll_std, "Train_MSE": train_mse,
                "Train_Calibration": train_calib, "Train_Sharpness": train_sharp,
            }
        else:
            epochs_no_improve += 1
            status = f"no imp. ({epochs_no_improve})"
            if patience and epochs_no_improve >= patience:
                print("Early stopping.")
                break

        # Epoch summary print
        print(f"{epoch+1:3} | {train_nll:10.4f} ± {train_nll_std:5.3f} | {train_mse:8.4f} | "
              f"{train_calib:8.4f} | {train_sharp:8.4f} | "
              f"{val_nll:10.4f} ± {val_nll_std:5.3f} | {val_mse:8.4f} | "
              f"{val_calib:8.4f} | {val_sharp:8.4f} | {duration:7.2f} [{status}]")

    # ===== LOG BEST METRICS =====
    if best_metrics:
        writer.add_scalar("Best_Val/NLL", best_metrics["Val_NLL"], epoch)
        writer.add_scalar("Best_Val/NLL_std", best_metrics["Val_NLL_std"], epoch)
        writer.add_scalar("Best_Val/MSE", best_metrics["Val_MSE"], epoch)
        writer.add_scalar("Best_Val/Calibration", best_metrics["Val_Calibration"], epoch)
        writer.add_scalar("Best_Val/Sharpness", best_metrics["Val_Sharpness"], epoch)

        writer.add_scalar("Best_Train/NLL", best_metrics["Train_NLL"], epoch)
        writer.add_scalar("Best_Train/NLL_std", best_metrics["Train_NLL_std"], epoch)
        writer.add_scalar("Best_Train/MSE", best_metrics["Train_MSE"], epoch)
        writer.add_scalar("Best_Train/Calibration", best_metrics["Train_Calibration"], epoch)
        writer.add_scalar("Best_Train/Sharpness", best_metrics["Train_Sharpness"], epoch)

def train_mc_dropout_world_model(prefix_name, train_loader, val_loader, device,
                                 hidden_layers=[128, 128], dropout_prob=0.1,
                                 max_epochs=100, learning_rate=1e-3,
                                 weight_decay=1e-5, patience=5, n_mc_samples=10):
    """
    Train a Monte Carlo Dropout world model with early stopping.

    The model is an `MCDropoutWorldModel` trained with MSE loss. Dropout is 
    active at both training and test time, enabling uncertainty estimation 
    through repeated stochastic forward passes. Calibration error and 
    sharpness are computed using MC dropout predictions. Training logs 
    are saved to TensorBoard.

    Parameters
    ----------
    prefix_name : str
        Prefix used to generate a unique run name for logs and saved checkpoints.
    train_loader : torch.utils.data.DataLoader
        Dataloader providing training batches (inputs, targets).
    val_loader : torch.utils.data.DataLoader
        Dataloader providing validation batches (inputs, targets).
    device : str or torch.device
        Device used for training (e.g., "cpu" or "cuda").
    hidden_layers : list of int, optional
        Sizes of hidden layers in the MC dropout model (default is [128, 128]).
    dropout_prob : float, optional
        Dropout probability applied after each hidden layer (default is 0.1).
    max_epochs : int, optional
        Maximum number of training epochs (default is 100).
    learning_rate : float, optional
        Learning rate for the Adam optimizer (default is 1e-3).
    weight_decay : float, optional
        Weight decay (L2 regularization) for the optimizer (default is 1e-5).
    patience : int, optional
        Number of epochs without improvement before early stopping (default is 5).
    n_mc_samples : int, optional
        Number of stochastic forward passes used for MC dropout prediction 
        when estimating calibration and sharpness (default is 10).

    Returns
    -------
    None
        The trained model is saved to disk, and training/validation metrics 
        are logged to TensorBoard. Best metrics are also recorded in the logger.
    """


    # Create unique run name and setup logging/saving
    run_name = f"{prefix_name}_{int(time.time())}_{generate_funny_name()}"
    writer = SummaryWriter(log_dir=f"./u_e_test/unc_train/{run_name}")
    save_path = f"./u_e_test/unc_models/{run_name}.pth"

    # Infer input/output dimensions from dataloader
    input_dim = next(iter(train_loader))[0].shape[1]
    output_dim = next(iter(train_loader))[1].shape[1]

    # Initialize MC Dropout model and optimizer
    model = MCDropoutWorldModel(input_dim, output_dim, hidden_layers, dropout_prob).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = torch.nn.MSELoss()

    best_val_loss = float('inf')
    epochs_no_improve = 0

    # Print header for log table
    print(f"{'Ep':>3} | {'Train MSE':>10} ± {'Std':>5} | {'Calib':>8} | {'Sharp':>8} | "
          f"{'Val MSE':>10} ± {'Std':>5} | {'Calib':>8} | {'Sharp':>8} | {'Time(s)':>7}")

    for epoch in range(max_epochs):
        start = time.time()
        model.train()

        # ===== TRAINING =====
        train_losses = []
        mu_all, var_all, y_all = [], [], []

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            # Forward pass with dropout (training mode)
            output = model(x)
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

            # Collect MC dropout predictions for calibration metrics
            with torch.no_grad():
                mu, var, _ = model.predict(x, n_samples=n_mc_samples)
                mu_all.append(mu); var_all.append(var); y_all.append(y)

        # Aggregate results for training
        mu_cat, var_cat, y_cat = torch.cat(mu_all), torch.cat(var_all), torch.cat(y_all)
        train_mse = np.mean(train_losses); train_mse_std = np.std(train_losses)
        train_calib, train_sharp = calibration_and_sharpness(mu_cat, var_cat, y_cat, device=device)

        # ===== VALIDATION =====
        model.eval()

        # Compute plain MSE using deterministic forward (no MC dropout)
        val_losses = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                output = model(x)
                val_losses.append(loss_fn(output, y).item())

        val_mse = np.mean(val_losses); val_mse_std = np.std(val_losses)

        # For calibration/sharpness, evaluate with MC dropout predictions
        mu_all, var_all, y_all = [], [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                mu, var, _ = model.predict(x, n_samples=n_mc_samples)
                mu_all.append(mu); var_all.append(var); y_all.append(y)

        mu_cat, var_cat, y_cat = torch.cat(mu_all), torch.cat(var_all), torch.cat(y_all)
        val_calib, val_sharp = calibration_and_sharpness(mu_cat, var_cat, y_cat, device=device)

        duration = time.time() - start

        # ===== LOGGING =====
        writer.add_scalar("Train/MSE", train_mse, epoch)
        writer.add_scalar("Train/MSE_std", train_mse_std, epoch)
        writer.add_scalar("Train/Calibration", train_calib, epoch)
        writer.add_scalar("Train/Sharpness", train_sharp, epoch)

        writer.add_scalar("Val/MSE", val_mse, epoch)
        writer.add_scalar("Val/MSE_std", val_mse_std, epoch)
        writer.add_scalar("Val/Calibration", val_calib, epoch)
        writer.add_scalar("Val/Sharpness", val_sharp, epoch)

        writer.add_scalar("Time/epoch_seconds", duration, epoch)

        # ===== EARLY STOPPING =====
        if val_mse < best_val_loss:
            best_val_loss = val_mse
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
            status = "saved"

            best_metrics = {
                "Val_MSE": val_mse, "Val_MSE_std": val_mse_std,
                "Val_Calibration": val_calib, "Val_Sharpness": val_sharp,
                "Train_MSE": train_mse, "Train_MSE_std": train_mse_std,
                "Train_Calibration": train_calib, "Train_Sharpness": train_sharp,
            }
        else:
            epochs_no_improve += 1
            status = f"no imp. ({epochs_no_improve})"
            if patience and epochs_no_improve >= patience:
                print("Early stopping.")
                break

        # Print epoch summary
        print(f"{epoch+1:3} | {train_mse:10.4f} ± {train_mse_std:5.3f} | {train_calib:8.4f} | {train_sharp:8.4f} | "
              f"{val_mse:10.4f} ± {val_mse_std:5.3f} | {val_calib:8.4f} | {val_sharp:8.4f} | {duration:7.2f} [{status}]")

    # ===== LOG BEST METRICS =====
    if best_metrics:
        writer.add_scalar("Best_Val/MSE", best_metrics["Val_MSE"], epoch)
        writer.add_scalar("Best_Val/MSE_std", best_metrics["Val_MSE_std"], epoch)
        writer.add_scalar("Best_Val/Calibration", best_metrics["Val_Calibration"], epoch)
        writer.add_scalar("Best_Val/Sharpness", best_metrics["Val_Sharpness"], epoch)

        writer.add_scalar("Best_Train/MSE", best_metrics["Train_MSE"], epoch)
        writer.add_scalar("Best_Train/MSE_std", best_metrics["Train_MSE_std"], epoch)
        writer.add_scalar("Best_Train/Calibration", best_metrics["Train_Calibration"], epoch)
        writer.add_scalar("Best_Train/Sharpness", best_metrics["Train_Sharpness"], epoch)

def train_rnd(prefix_name,
              train_loader,
              val_loader,
              device,
              hidden_layers=[128, 128, 128],
              max_epochs=100,
              learning_rate=1e-3,
              weight_decay=1e-5,
              patience=5):
    """
    Train a Random Network Distillation (RND) model.

    The method trains a predictor network to mimic the output of a fixed,
    randomly initialized target network. The difference between the two
    networks' outputs can later be used as an uncertainty or novelty score.
    Early stopping is applied based on validation loss.

    Parameters
    ----------
    prefix_name : str
        Name prefix for the experiment run (used in logging and saving).
    train_loader : DataLoader
        DataLoader providing the training dataset.
    val_loader : DataLoader
        DataLoader providing the validation dataset.
    device : torch.device or str
        Device on which to run training ("cpu" or "cuda").
    hidden_layers : list of int, optional
        Sizes of hidden layers for both target and predictor networks.
        Default is [128, 128, 128].
    max_epochs : int, optional
        Maximum number of training epochs. Default is 100.
    learning_rate : float, optional
        Learning rate for the Adam optimizer. Default is 1e-3.
    weight_decay : float, optional
        Weight decay (L2 regularization) for the optimizer. Default is 1e-5.
    patience : int, optional
        Number of epochs without improvement in validation loss before
        early stopping is triggered. Default is 5.

    Returns
    -------
    None
        The best predictor network weights (together with the fixed random
        network) are saved to disk in the `./u_e_test/ood_models/` folder.
        Training progress and losses are logged in TensorBoard under
        `./u_e_test/ood_train/`.
    """


    # Metadata and run identifiers
    hyperparams = {}
    test_name = generate_funny_name()
    seed = int(time.time()) - 1750154400
    run_name = f'{prefix_name}_{seed:07}_{test_name}'

    # Logging and saving paths
    writer = SummaryWriter(log_dir="./u_e_test/ood_train/" + run_name)
    save_path = "./u_e_test/ood_models/" + run_name + '.pth'

    # Input dimensionality from training samples
    input_dim = next(iter(train_loader)).shape[1]

    # Random target network (fixed parameters)
    random_net = RNDNetwork(input_dim, hidden_layers).to(device)
    for param in random_net.parameters():
        param.requires_grad = False

    # Predictor network (trained to mimic random_net)
    predictor_net = RNDNetwork(input_dim, hidden_layers).to(device)

    # Optimizer for predictor network
    optimizer = torch.optim.Adam(
        predictor_net.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    # Store hyperparameters for logging
    hyperparams.update({
        'prefix_name': prefix_name,
        'test_name': test_name,
        'hidden_layers': hidden_layers,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'max_epochs': max_epochs,
        'patience': patience,
        'seed': seed,
    })

    writer.add_text(
        'hyperparams',
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{k}|{v}|" for k, v in hyperparams.items()])),
    )

    print(f'Run name: {run_name}')
    print('Training started...\n')
    print(f"\n{'Epoch':>5} | {'Train Loss':>10} | {'Train Std':>10} | {'Val Loss':>10} | {'Val Std':>10} | {'Status':>25}")
    print("-" * 90)

    best_val_loss = float('inf')
    epochs_no_improve = 0

    # Training loop
    for epoch in range(max_epochs):
        predictor_net.train()
        train_losses = []

        # Training step
        for x in train_loader:
            x = x.to(device)
            optimizer.zero_grad()

            with torch.no_grad():
                target_output = random_net(x)

            predictor_output = predictor_net(x)

            mse_per_sample = F.mse_loss(
                predictor_output, target_output,
                reduction='none'
            ).mean(dim=1)
            loss = mse_per_sample.sum()

            loss.backward()
            optimizer.step()

            train_losses.extend(mse_per_sample.detach().cpu().numpy())

        avg_train_loss = np.mean(train_losses)
        std_train_loss = np.std(train_losses)

        # Validation step
        predictor_net.eval()
        val_losses = []

        with torch.no_grad():
            for x in val_loader:
                x = x.to(device)
                target_output = random_net(x)
                predictor_output = predictor_net(x)
                mse_per_sample = F.mse_loss(
                    predictor_output, target_output,
                    reduction='none'
                ).mean(dim=1)
                val_losses.extend(mse_per_sample.detach().cpu().numpy())

        avg_val_loss = np.mean(val_losses)
        std_val_loss = np.std(val_losses)

        # Log losses to TensorBoard
        writer.add_scalar("Loss/train_loss", avg_train_loss, epoch)
        writer.add_scalar("Loss/train_std", std_train_loss, epoch)
        writer.add_scalar("Loss/val_loss", avg_val_loss, epoch)
        writer.add_scalar("Loss/val_std", std_val_loss, epoch)

        # Check improvements and save best model
        status = ""
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save({
                'predictor_state_dict': predictor_net.state_dict(),
                'random_state_dict': random_net.state_dict(),
            }, save_path)
            status = "🟩 best so far (saved)"
        else:
            epochs_no_improve += 1
            if patience and epochs_no_improve >= patience:
                status = "🟥      early stopping"
                print(f"{epoch + 1:5d} | {avg_train_loss:.6f} | {std_train_loss:.6f} | {avg_val_loss:.6f} | {std_val_loss:.6f} | {status:>25}")
                break
            else:
                status = f"🟥  NO improvement ({epochs_no_improve})"

        # Progress update
        print(f"{epoch + 1:5d} | {avg_train_loss:.6f} | {std_train_loss:.6f} | {avg_val_loss:.6f} | {std_val_loss:.6f} | {status:>25}")


####################################################################################################
####################################################################################################

#   ╔═════════════════════════╗
#   ║   OOD Detection Utils   ║
#   ╚═════════════════════════╝


def compute_uncertainties_prob(world_model, data_loader):
    """
    Compute predictive uncertainties from a probabilistic world model.

    This function performs a forward pass through a trained 
    `ProbabilisticWorldModel` on all batches from a dataloader, 
    collects the predicted variances, and aggregates them across 
    output dimensions.

    Parameters
    ----------
    world_model : ProbabilisticWorldModel
        Trained probabilistic world model that outputs (mean, variance).
    data_loader : torch.utils.data.DataLoader
        Dataloader providing input batches for evaluation.

    Returns
    -------
    numpy.ndarray
        Array of shape (N,) containing aggregated predictive variances 
        for each sample, where N is the total number of samples in the dataloader.
    """

    all_vars = []
    with torch.no_grad():
        for x, _ in data_loader:
            # Forward pass -> probabilistic world model returns (mu, var)
            _, var = world_model(x)
            # Collect variances for each batch (move to CPU)
            all_vars.append(var.detach().cpu())

    # Concatenate all batches
    all_vars = torch.cat(all_vars, dim=0)

    # Aggregate variance across output dimensions (sum or mean)
    agg_vars = all_vars.sum(dim=1)  # alternative: .mean(dim=1)

    return agg_vars.numpy()

def compute_uncertainties_mcd(world_model, data_loader, n_samples=20):
    """
    Compute predictive uncertainties from a Monte Carlo Dropout world model.

    This function performs multiple stochastic forward passes with dropout 
    enabled, estimates predictive variances for each sample, and aggregates 
    them across output dimensions.

    Parameters
    ----------
    world_model : MCDropoutWorldModel
        Trained Monte Carlo Dropout model with a `predict` method returning 
        (mean, variance, predictions).
    data_loader : torch.utils.data.DataLoader
        Dataloader providing input batches for evaluation.
    n_samples : int, optional
        Number of stochastic forward passes to perform for each batch 
        (default is 20).

    Returns
    -------
    numpy.ndarray
        Array of shape (N,) containing aggregated predictive variances 
        for each sample, where N is the total number of samples in the dataloader.
    """

    all_vars = []
    with torch.no_grad():
        for x, _ in data_loader:
            # multiple dropout predictions, returns mean, var, samples
            _, var, _ = world_model.predict(x, n_samples=n_samples)
            # store variances on cpu
            all_vars.append(var.detach().cpu())

    # concatenate batches
    all_vars = torch.cat(all_vars, dim=0)

    # aggregate variance across output dims
    agg_vars = all_vars.sum(dim=1)

    return agg_vars.numpy()

def compute_uncertainties_QNet(Q_ensemble, data_loader):
    """
    Compute ensemble-based uncertainties from a Q-network ensemble.

    This function evaluates multiple Q-networks on batches of input data, 
    computes the variance of their predictions, and uses it as a measure 
    of epistemic uncertainty.

    Parameters
    ----------
    Q_ensemble : list of nn.Module
        List of trained Q-network models forming the ensemble.
    data_loader : torch.utils.data.DataLoader
        Dataloader providing input batches. Each batch is split into:
            - part1 : lidar-like features reshaped to (batch_size, 4, 17),
            - part2 : remaining state features,
            - action : last 2 elements of the input.

    Returns
    -------
    numpy.ndarray
        Array of shape (N,) containing predictive uncertainties estimated 
        as the variance across ensemble predictions for each sample, where 
        N is the total number of samples in the dataloader.
    """

    all_std = []

    with torch.no_grad():
        for x, y in data_loader:
            # split input into lidar part (reshaped) and state part
            part1 = torch.cat([x[:, :17*3], y[:, :17]], dim=1).reshape(-1, 4, 17)  
            part2 = torch.cat([x[:, 17*3:-2], y[:, 17:]], dim=1)                
            action = x[:, -2:]                                              

            # compute Q-values for each model in the ensemble
            q_vals = torch.stack([
                q(part1, part2, action) for q in Q_ensemble
            ])  # shape: [ensemble_size, batch_size, 1]

            # compute variance across models (ensemble uncertainty)
            std_q = torch.var(q_vals, dim=0).squeeze(-1)  # shape: [batch_size]

            all_std.append(std_q.cpu())

    # concatenate results across all batches
    total_unc = torch.cat(all_std, dim=0)  # shape: [num_samples]

    return total_unc.numpy()

def compute_uncertainties_rnd(source, predictor, data_loader):
    """
    Compute uncertainties using Random Network Distillation (RND).

    This function evaluates a fixed target network (`source`) and a trainable 
    predictor network on the same input and computes the squared difference 
    between their outputs. The discrepancy is used as a measure of novelty 
    or epistemic uncertainty.

    Parameters
    ----------
    source : nn.Module
        Fixed, randomly initialized target network in the RND framework.
    predictor : nn.Module
        Trainable predictor network that attempts to mimic the target network.
    data_loader : torch.utils.data.DataLoader
        Dataloader providing input batches. Each batch provides two tensors:
            - x : input state features,
            - y : target features. Both are concatenated before passing through the networks.

    Returns
    -------
    numpy.ndarray
        Array of shape (N,) containing per-sample uncertainty values, 
        computed as the sum of squared differences between predictor and target outputs, 
        scaled by a factor of 100.
    """

    all_diffs = []
    source.eval()
    predictor.eval()
    
    with torch.no_grad():
        for x, y in data_loader:
            # concatenate state and target as input
            input_x = torch.cat([x[:, :-2], y], dim=1)
            
            # forward pass through target (fixed) and predictor networks
            pred_source = source(input_x)
            pred_predictor = predictor(input_x)
            
            # squared difference between networks
            diff = (pred_source - pred_predictor) ** 2
            # aggregate across feature dimension → per-sample uncertainty
            diff = diff.sum(dim=1) * 100
            all_diffs.append(diff.detach().cpu())
    
    # concatenate results across all batches
    all_diffs = torch.cat(all_diffs, dim=0)
    return all_diffs.numpy()


def generate_ood_data_from_loader(loader, noise_std=3.0, seed=42):
    """
    Generate out-of-distribution (OOD) data by corrupting features with noise.

    This function takes a dataset loader and produces a corrupted version 
    of the data by applying different types of noise perturbations at the 
    sample level. Each sample is modified with one randomly chosen noise 
    strategy.

    Parameters
    ----------
    loader : torch.utils.data.DataLoader
        Dataloader yielding (x, y) pairs to be corrupted.
    noise_std : float, optional
        Standard deviation or scale factor for noise (default is 3.0).
    seed : int, optional
        Random seed for reproducibility (default is 42).

    Returns
    -------
    torch.utils.data.TensorDataset
        A dataset containing the corrupted input-target pairs (x_ood, y_ood).

    Notes
    -----
    The following noise types are applied randomly per sample:
        - "gaussian" : Add Gaussian noise with std = noise_std.
        - "uniform" : Add uniform noise in [-noise_std, noise_std].
        - "sparse" : Add Gaussian noise only to a random 10% of features.
        - "feature" : Corrupt entire features with probability 0.2.
        - "permutation" : Randomly permute the order of features.
        - "extreme_values" : Replace 10% of features with extreme values 
          scaled by 10 × noise_std.
        - "corrupted_chunk" : Replace a contiguous random block of features 
          with Gaussian noise.
    """

    torch.manual_seed(seed)
    random.seed(seed)

    # collect all data from the loader
    x_all, y_all = [], []
    for x, y in loader:
        x_all.append(x)
        y_all.append(y)
    x_all = torch.cat(x_all, dim=0)
    y_all = torch.cat(y_all, dim=0)

    N = x_all.shape[0]
    x_ood = torch.empty_like(x_all)
    y_ood = torch.empty_like(y_all)

    for i in range(N):
        # randomly choose a noise type for each sample
        noise_type = random.choice([
            "gaussian", "uniform", "sparse", "feature", 
            "permutation", "extreme_values", "corrupted_chunk"
        ])
        
        def add_noise(tensor):
            if noise_type == "gaussian":
                # add Gaussian noise
                return tensor + torch.randn_like(tensor) * noise_std
            elif noise_type == "uniform":
                # add uniform noise in [-noise_std, noise_std]
                return tensor + (torch.rand_like(tensor) - 0.5) * 2 * noise_std
            elif noise_type == "sparse":
                # add noise only on a fraction of features
                mask = torch.rand_like(tensor) < 0.1
                noise = torch.randn_like(tensor) * noise_std
                return tensor + noise * mask
            elif noise_type == "feature":
                # corrupt entire features across the sample
                mask = (torch.rand(tensor.shape[0]) < 0.2).float()
                noise = torch.randn_like(tensor) * noise_std
                return tensor + noise * mask
            elif noise_type == "permutation":
                # permute feature order
                return tensor[torch.randperm(tensor.shape[0])]
            elif noise_type == "extreme_values":
                # replace some features with extreme large/small values
                mask = torch.rand_like(tensor) < 0.1
                extreme = torch.randn_like(tensor) * noise_std * 10
                return tensor * (~mask) + extreme * mask
            elif noise_type == "corrupted_chunk":
                # replace a contiguous block with random noise
                length = random.randint(1, max(1, tensor.shape[0] // 4))
                start = random.randint(0, tensor.shape[0] - length)
                corrupted = tensor.clone()
                corrupted[start:start+length] = torch.randn_like(corrupted[start:start+length]) * noise_std
                return corrupted
            else:
                return tensor

        # apply noise to both x and y
        x_ood[i] = add_noise(x_all[i])
        y_ood[i] = add_noise(y_all[i])

    return TensorDataset(x_ood, y_ood)

def evaluate_ood_detection_multiple(unc_functions, funcs_names, id_loaders, fontsize=14):
    """
    Evaluate multiple uncertainty functions for out-of-distribution (OOD) detection.

    Each uncertainty estimator is paired with its own in-distribution (ID) data loader, 
    from which a corresponding OOD dataset is generated. The function computes 
    ROC curves, AUC scores, and selects the best decision threshold using Youden's J statistic.

    Parameters
    ----------
    unc_functions : list of callable
        A list of uncertainty functions, each mapping a DataLoader to an array of 
        uncertainty scores.
    funcs_names : list of str
        Names corresponding to each uncertainty function, used for reporting and plotting.
    id_loaders : list of torch.utils.data.DataLoader
        A list of in-distribution data loaders, one per uncertainty function.
    fontsize : int, optional (default=14)
        Font size for axis labels, title, legend, and ticks in the ROC plot.

    Returns
    -------
    list of float
        Best thresholds for each uncertainty function, selected according to 
        Youden's J statistic (maximizing sensitivity + specificity - 1).

    Notes
    -----
    For each estimator:
        - Uncertainty is computed on its ID loader.
        - An OOD dataset is generated from the same loader using 
          `generate_ood_data_from_loader`.
        - Uncertainty is computed on OOD samples.
        - Scores are concatenated (ID=0, OOD=1).
        - ROC curve, AUC, and optimal threshold are computed.
    The function prints AUC and best threshold values for each method, 
    and plots ROC curves in a single figure for comparison.
    """

    # compute uncertainty on ID data
    vars_id = [f(l) for f, l in zip(unc_functions, id_loaders)]
    
    # generate OOD datasets from ID loaders
    ood_datasets = [generate_ood_data_from_loader(l, noise_std=3.0) for l in id_loaders]
    ood_loaders = [DataLoader(d, batch_size=64) for d in ood_datasets]

    # compute uncertainty on OOD data
    vars_ood = [f(l) for f, l in zip(unc_functions, ood_loaders)]

    # concatenate ID and OOD scores
    scores = [np.concatenate([a, b]) for a, b in zip(vars_id, vars_ood)]
    labels = [np.concatenate([np.zeros_like(a), np.ones_like(b)]) for a, b in zip(vars_id, vars_ood)]

    # ROC, AUC and thresholds
    results = []
    plt.figure(figsize=(8, 6))
    for i, (s, y, name) in enumerate(zip(scores, labels, funcs_names)):
        fpr, tpr, thresh = roc_curve(y, s)
        auc = roc_auc_score(y, s)

        # Youden index
        j = tpr - fpr
        best_idx = np.argmax(j)
        best_thresh = thresh[best_idx]

        results.append(best_thresh)
        print(f"[{name}] AUC: {auc:.4f} | Best threshold (Youden): {best_thresh:.4f}")

        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

    # diagonal line for random baseline
    plt.plot([0, 1], [0, 1], 'k--')

    # axes and labels
    plt.xlabel("False Positive Rate", fontsize=fontsize)
    plt.ylabel("True Positive Rate", fontsize=fontsize)
    plt.title("ROC Curve - OOD Detection via Uncertainty", fontsize=fontsize)
    plt.legend(fontsize=fontsize - 2)
    plt.grid(True)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.show()

    return results


####################################################################################################
####################################################################################################

#   ╔═══════════════════════╗
#   ║   Model Calibration   ║
#   ╚═══════════════════════╝



def evaluate_calibration(model, data_loader, model_type="prob", n_samples=20):
    """
    Compute predictions, predictive uncertainties, and ground truth for calibration analysis.

    This function runs the given model on a dataset and collects predicted means, 
    predictive standard deviations, and true target values. It supports both 
    probabilistic world models and MC Dropout world models.

    Parameters
    ----------
    model : torch.nn.Module
        The trained model to evaluate. Must implement either:
        - forward(x) → (mu, var) for model_type="prob"
        - predict(x, n_samples) → (mu, var, preds) for model_type="mcd"
    data_loader : torch.utils.data.DataLoader
        DataLoader providing input samples (x) and ground-truth targets (y).
    model_type : {"prob", "mcd"}, optional (default="prob")
        Type of model:
        - "prob": probabilistic world model that directly outputs (mu, var).
        - "mcd": MC Dropout model that outputs mean and variance from multiple stochastic passes.
    n_samples : int, optional (default=20)
        Number of stochastic forward passes when using "mcd".

    Returns
    -------
    mu : numpy.ndarray, shape (N, D)
        Predicted means for all samples.
    sigma : numpy.ndarray, shape (N, D)
        Predicted standard deviations (uncertainties) for all samples.
    y_true : numpy.ndarray, shape (N, D)
        Ground-truth target values.
    """

    all_mu, all_sigma, all_y = [], [], []

    with torch.no_grad():
        for x, y in data_loader:
            if model_type == "prob":
                mu, var = model(x)                         # Probabilistic world model
            elif model_type == "mcd":
                mu, var, _ = model.predict(x, n_samples=n_samples)  # MC Dropout with sampling
            else:
                raise ValueError("model_type must be 'prob' or 'mcd'")

            sigma = torch.sqrt(var)  # standard deviation
            all_mu.append(mu.cpu())
            all_sigma.append(sigma.cpu())
            all_y.append(y.cpu())

    # Concatenate results
    mu = torch.cat(all_mu, dim=0).numpy()
    sigma = torch.cat(all_sigma, dim=0).numpy()
    y_true = torch.cat(all_y, dim=0).numpy()

    return mu, sigma, y_true


def plot_calibration_regression(mu, sigma, y_true, model_name="Model", out_dir="./calibration_plots", fontsize=14):
    """
    Generate and save calibration plots for regression models.

    The function produces three plots:
    - QQ plot of normalized prediction errors
    - Histogram of normalized errors compared to N(0,1)
    - Prediction Interval Coverage Probability (PICP) curve

    Parameters
    ----------
    mu : np.ndarray
        Predicted means from the model.
    sigma : np.ndarray
        Predicted standard deviations corresponding to mu.
    y_true : np.ndarray
        Ground truth target values.
    model_name : str, optional
        Name used in plot titles and filenames (default: "Model").
    out_dir : str, optional
        Directory where plots are saved (default: "./calibration_plots").
    fontsize : int, optional
        Font size for plot labels and legends (default: 14).

    Returns
    -------
    None
        The function saves three plots (QQ, histogram, PICP) as PNG files.
    """

    os.makedirs(out_dir, exist_ok=True)

    # Normalized errors
    norm_err = (y_true - mu) / sigma
    norm_err = norm_err.flatten()

    # 1. QQ-plot
    fig, ax = plt.subplots(figsize=(6, 4))
    stats.probplot(norm_err, dist="norm", plot=ax)
    ax.tick_params(axis="both", labelsize=fontsize)
    fig.savefig(os.path.join(out_dir, f"{model_name}_qqplot.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 2. Normalized error histogram
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(norm_err, bins=30, density=True, alpha=0.6, color="C0")
    x_axis = np.linspace(-4, 4, 200)
    ax.plot(x_axis, stats.norm.pdf(x_axis), 'r--', label="N(0,1)")
    ax.set_xlabel("Normalized error", fontsize=fontsize)
    ax.set_ylabel("Density", fontsize=fontsize)
    ax.tick_params(axis="both", labelsize=fontsize)
    ax.legend(fontsize=fontsize)
    fig.savefig(os.path.join(out_dir, f"{model_name}_hist.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 3. PICP (Prediction Interval Coverage Probability)
    z_values = np.linspace(0.1, 3.0, 20)  # multiples of sigma
    observed_cov, expected_cov = [], []
    for z in z_values:
        inside = np.abs(norm_err) <= z
        coverage_obs = inside.mean()
        coverage_exp = stats.norm.cdf(z) - stats.norm.cdf(-z)
        observed_cov.append(coverage_obs)
        expected_cov.append(coverage_exp)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(expected_cov, observed_cov, 'o-', label=model_name)
    ax.plot([0,1], [0,1], 'r--', label="Perfect calibration")
    ax.set_xlabel("Expected coverage", fontsize=fontsize)
    ax.set_ylabel("Observed coverage", fontsize=fontsize)
    ax.tick_params(axis="both", labelsize=fontsize)
    ax.legend(fontsize=fontsize)
    fig.savefig(os.path.join(out_dir, f"{model_name}_picp.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_all_comparisons(
    run1_name,
    run2_name,
    metrics,
    max_step=40000,
    folder="./",
    fontsize=14
):
    """
    Compare two training runs across multiple metrics and plot them side by side.

    The function loads CSV log files for two runs, aligns them by training step,
    and plots the metric values up to a maximum step for direct comparison.

    Parameters
    ----------
    run1_name : str
        Base name of the first run (prefix of CSV files).
    run2_name : str
        Base name of the second run (prefix of CSV files).
    metrics : list of str
        List of metric names to compare (each must correspond to a CSV file).
    max_step : int, optional
        Maximum training step to display on the x-axis (default: 40000).
    folder : str, optional
        Directory containing the CSV log files (default: "./").
    fontsize : int, optional
        Font size for axis labels and legend (default: 14).

    Returns
    -------
    None
        Saves a PNG file with the side-by-side comparison plots and displays the figure.
    """


    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 4), sharex=True)

    # Ensure axes is iterable
    if len(metrics) == 1:
        axes = [axes]

    for i, metric in enumerate(metrics):
        # Load CSV files
        df1 = pd.read_csv(f"{folder}/{run1_name}_{metric}.csv")
        df2 = pd.read_csv(f"{folder}/{run2_name}_{metric}.csv")

        # Truncate at max_step
        df1 = df1[df1["Step"] <= max_step]
        df2 = df2[df2["Step"] <= max_step]

        # Plot
        axes[i].plot(df1["Step"], df1["Value"], label="5Q-net")
        axes[i].plot(df2["Step"], df2["Value"], label="2Q-net")
        axes[i].set_xlabel("Step", fontsize=fontsize)
        axes[i].set_ylabel(metric.capitalize(), fontsize=fontsize)
        axes[i].tick_params(axis="both", labelsize=fontsize)
        axes[i].ticklabel_format(axis="x", style="sci", scilimits=(0,0))
        axes[i].grid(True, alpha=0.3)

    # Legend only once
    axes[0].legend(loc="best", fontsize=fontsize - 2)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs("./img_2/others", exist_ok=True)
    fig.savefig(f"./img_2/others/{'_'.join(metrics)}.png", dpi=300, bbox_inches="tight")
    plt.show()
