import os
import json
import math
from collections import defaultdict

from ipykernel import control
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.cm as cm
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plotta_confronto_csv_final(lista_paths, colonna_ordinamento, colonne_da_plottare=None):
    """
    Legge, elabora e confronta i dati da diversi file CSV, generando dei grafici a barre.
    Gestisce automaticamente file con lunghezze diverse e aggiunge i valori sulle barre.
    """
    
    dataframes = []
    nomi_file = []
    
    # --- 1. CARICAMENTO E PREPARAZIONE ASSE X GLOBALE ---
    print(f"--- Inizio elaborazione ---")
    
    tutti_i_valori_x = set() 

    for path in lista_paths:
        try:
            df = pd.read_csv(path, sep=None, engine='python')
            df.columns = df.columns.str.strip()

            if colonna_ordinamento not in df.columns:
                print(f"ATTENZIONE: '{colonna_ordinamento}' non trovata in {os.path.basename(path)}. Salto questo file.")
                continue

            df = df.drop_duplicates(subset=[colonna_ordinamento]).set_index(colonna_ordinamento)
            tutti_i_valori_x.update(df.index.tolist())
            
            dataframes.append(df)
            nomi_file.append(os.path.basename(path))
            
        except Exception as e:
            print(f"Errore lettura {path}: {e}")
            continue

    if not dataframes:
        print("Nessun dato valido caricato.")
        return

    asse_x_globale = sorted(list(tutti_i_valori_x))
    indici_x = np.arange(len(asse_x_globale))
    
    # --- 2. IDENTIFICAZIONE E FILTRAGGIO METRICHE ---
    colonne_csv = dataframes[0].columns
    metriche_map = {}
    colonne_processate = set()
    
    target_cols = set(colonne_da_plottare) if colonne_da_plottare else None

    for col in colonne_csv:
        if col in colonne_processate: continue
        
        is_mean = col.endswith('_mean')
        base_name = col.replace('_mean', '') if is_mean else col
        
        if target_cols is not None:
            if (base_name not in target_cols) and (col not in target_cols):
                continue

        if is_mean:
            std_col = base_name + '_std'
            if std_col in colonne_csv:
                metriche_map[base_name] = {'val': col, 'std': std_col}
                colonne_processate.update([col, std_col])
            else:
                metriche_map[col] = {'val': col, 'std': None}
                colonne_processate.add(col)
        elif col.endswith('_std'):
             pass 
        else:
            metriche_map[col] = {'val': col, 'std': None}
            colonne_processate.add(col)

    # --- 3. PLOTTING ---
    n_metriche = len(metriche_map)
    if n_metriche == 0:
        print("Nessuna metrica trovata. Controlla i nomi richiesti.")
        return

    # Ho aumentato un pochino l'altezza della figura per fare spazio ai numeri
    fig, axes = plt.subplots(nrows=n_metriche, ncols=1, figsize=(12, 5 * n_metriche), sharex=True)
    if n_metriche == 1: axes = [axes]
    
    n_files = len(dataframes)
    larghezza_totale = 0.8
    larghezza_barra = larghezza_totale / n_files
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, n_files))

    for ax, (nome_metrica, cols) in zip(axes, metriche_map.items()):
        val_col = cols['val']
        std_col = cols['std']
        
        # ... [il codice precedente rimane uguale fino al ciclo delle metriche] ...

    for ax, (nome_metrica, cols) in zip(axes, metriche_map.items()):
        val_col = cols['val']
        std_col = cols['std']
        
        for i, df in enumerate(dataframes):
            df_allineato = df.reindex(asse_x_globale)
            
            offset = (i - n_files/2) * larghezza_barra + (larghezza_barra/2)
            pos = indici_x + offset
            
            # --- MODIFICA QUI: Forziamo i valori ad essere NUMERI ---
            if val_col in df_allineato.columns:
                # errors='coerce' trasforma eventuali stringhe spurie in NaN
                valori = pd.to_numeric(df_allineato[val_col], errors='coerce')
            else:
                valori = pd.Series([np.nan] * len(asse_x_globale))
                
            if std_col and std_col in df_allineato.columns:
                errori = pd.to_numeric(df_allineato[std_col], errors='coerce')
            else:
                errori = None
            # --------------------------------------------------------
            
            barre_disegnate = ax.bar(pos, valori, 
                   width=larghezza_barra, 
                   yerr=errori,
                   capsize=4,
                   label=nomi_file[i] if ax == axes[0] else "", 
                   color=colors[i],
                   alpha=0.85,
                   error_kw={'ecolor': 'black', 'elinewidth': 1.5})
            
            # --- Creiamo le etichette manualmente (come fatto in precedenza) ---
            etichette_barre = []
            for v in valori:
                if pd.isna(v): 
                    etichette_barre.append("")
                else:          
                    etichette_barre.append(f"{v:.2f}")

            ax.bar_label(barre_disegnate, 
                         labels=etichette_barre,
                         padding=4,         
                         fontsize=8,        
                         color='black')   
        
        ax.set_ylabel(nome_metrica, fontsize=10, fontweight='bold')
        ax.set_title(f"Metrica: {nome_metrica}", fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        
        # Aumentiamo un po' il limite superiore dell'asse Y per non tagliare i numeri
        ax.margins(y=0.15) 
        
    # --- 4. FORMATTAZIONE ---
    etichette = [str(x) for x in asse_x_globale]
    axes[-1].set_xlabel(colonna_ordinamento, fontsize=12)
    axes[-1].set_xticks(indici_x)
    axes[-1].set_xticklabels(etichette, rotation=45, ha='right')
    
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.01), ncol=4, frameon=False, fontsize=11)
    plt.tight_layout()
    plt.show()


def plot_metric_series_on_ax(
    ax,             # <--- NUOVO: L'asse su cui disegnare
    data_list_x,
    data_list_y,
    metric_x,
    metric_y,
    labels,
    metric_labels=None, 
    cmap_name="plasma",
    cmap_col="percentile", 
    upper_bound_x=None,
    upper_bound_y=None,
    lower_bound_x=None,
    lower_bound_y=None,
    title=None,
    fontsize=14
):
    """
    Disegna le traiettorie delle metriche su un asse Matplotlib specifico (ax).
    """

    # --- 1. Helper per convertire i dati (Invariato) ---
    def process_data_input(data_list, metric_name, col_name_tau):
        processed_dfs = []
        for experiment_data in data_list:
            rows = []
            if not experiment_data:
                processed_dfs.append(pd.DataFrame(columns=[metric_name, col_name_tau]))
                continue

            sorted_taus = sorted(experiment_data.keys())
            for tau in sorted_taus:
                episode_list = experiment_data[tau]
                values = [ep[metric_name] for ep in episode_list if metric_name in ep]
                if values:
                    mean_val = np.mean(values)
                    rows.append({metric_name: mean_val, col_name_tau: tau})
            
            if rows:
                processed_dfs.append(pd.DataFrame(rows))
            else:
                processed_dfs.append(pd.DataFrame(columns=[metric_name, col_name_tau]))
        return processed_dfs

    # --- 2. Preparazione Dati ---
    data_x = process_data_input(data_list_x, metric_x, cmap_col)
    data_y = process_data_input(data_list_y, metric_y, cmap_col)

    if metric_labels is None:
        metric_labels = {metric_x: metric_x, metric_y: metric_y}

    # --- 3. Logica Baseline ---
    # Se c'è più di 1 esperimento nella lista, l'ultimo è la baseline
    if len(data_x) > 1:
        num_experiments = len(data_x) - 1
        has_baseline = True
    else:
        num_experiments = len(data_x)
        has_baseline = False

    # --- 4. Setup Colormap ---
    def truncate_cmap(cmap, vmin=0.1, vmax=0.9, n=256):
        return LinearSegmentedColormap.from_list(
            f"trunc({cmap.name},{vmin:.2f},{vmax:.2f})",
            cmap(np.linspace(vmin, vmax, n))
        )
    cmap = truncate_cmap(plt.cm.get_cmap(cmap_name), 0.1, 0.85)

    # --- 5. Loop di Disegno (Tutti sullo stesso ax) ---
    sc = None # Placeholder per lo scatter object (per la colorbar)

    for i in range(num_experiments):
        df_x, df_y = data_x[i], data_y[i]
        
        if df_x.empty or df_y.empty:
            continue

        # Allineamento
        min_len = min(len(df_x), len(df_y))
        df_x, df_y = df_x.iloc[:min_len], df_y.iloc[:min_len]

        # Valori
        x_vals = df_x[metric_x].values
        y_vals = df_y[metric_y].values
        perc_vals = df_x[cmap_col].values

        # LineCollection (Linea sfumata)
        points = np.column_stack([x_vals, y_vals]).reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        seg_vals = (perc_vals[:-1] + perc_vals[1:]) / 2

        lc = LineCollection(segments, array=seg_vals, cmap=cmap, linewidths=2, alpha=0.8, zorder=1)
        ax.add_collection(lc)

        # Scatter (Punti)
        sc = ax.scatter(
            x_vals, y_vals, c=perc_vals, cmap=cmap,
            s=50, edgecolors="black", linewidths=0.4, zorder=2
        )

    # --- 6. Disegno Baseline (Grigio) ---
    base_line_handle = None
    if has_baseline:
        baseline_x = data_x[-1]
        baseline_y = data_y[-1]
        
        if not baseline_x.empty:
            base_line_handle, = ax.plot(
                baseline_x[metric_x], baseline_y[metric_y],
                color="0.6", linestyle="-", linewidth=1, alpha=0.9, zorder=0, label="random_UE"
            )
            ax.scatter(
                baseline_x[metric_x], baseline_y[metric_y],
                color="0.5", s=50, edgecolors="black", linewidths=0.5, zorder=1
            )

    # --- 7. Bounds (Target Policy) ---
    if upper_bound_x and upper_bound_y:
        if metric_x in upper_bound_x and metric_y in upper_bound_y:
            ax.plot(
                [upper_bound_x[metric_x], lower_bound_x[metric_x]],
                [upper_bound_y[metric_y], lower_bound_y[metric_y]],
                color="0.6", linestyle="--", linewidth=1, alpha=0.9, zorder=0
            )
            ax.scatter(upper_bound_x[metric_x], upper_bound_y[metric_y], color="blue", marker="x", s=80, linewidths=2, zorder=2)
            ax.scatter(lower_bound_x[metric_x], lower_bound_y[metric_y], color="green", marker="x", s=80, edgecolors="black", linewidths=2, zorder=2)

    # --- 8. Legenda ---
    handles = []
    # Ricreiamo le handle per la legenda in base agli esperimenti plottati
    for i in range(num_experiments):
        # Cerchiamo di prendere un colore rappresentativo (medio)
        df_temp = data_x[i]
        if not df_temp.empty:
            perc_vals = df_temp[cmap_col].values
            norm_val = (np.median(perc_vals) - perc_vals.min()) / (perc_vals.max() - perc_vals.min() + 1e-9)
            median_color = cmap(norm_val)
        else:
            median_color = "black"
        
        lbl = labels[i] if i < len(labels) else f"Exp {i}"
        handles.append(Line2D([0], [0], color=median_color, linewidth=2, label=lbl))

    if base_line_handle:
        handles.append(base_line_handle)

    # Aggiungiamo le legende dei bounds se presenti
    if upper_bound_x and upper_bound_y:
         if metric_x in upper_bound_x and metric_y in upper_bound_y:
            handles.append(Line2D([0], [0], color="0.6", linestyle="--", linewidth=2, label="policy → policy+cbf"))
            handles.append(Line2D([0], [0], color="blue", marker="x", linestyle="None", markersize=8, label="policy"))
            handles.append(Line2D([0], [0], color="green", marker="x", linestyle="None", markersize=8, label="policy+cbf"))

    ax.legend(handles=handles, loc="best", fontsize=fontsize - 2)

    # --- 9. Etichette e Titoli ---
    ax.set_xlabel(metric_labels[metric_x], fontsize=fontsize)
    ax.set_ylabel(metric_labels[metric_y], fontsize=fontsize)
    if title:
        ax.set_title(title, fontsize=fontsize)
    
    ax.tick_params(axis="both", which="major", labelsize=fontsize)
    ax.autoscale_view() # Importante per LineCollection

    # --- 10. Colorbar (Attaccata alla figura dell'ax) ---
    # Aggiungiamo la colorbar solo se abbiamo disegnato qualcosa (sc esiste)
    if sc:
        # Usiamo ax.figure per ottenere la figura padre
        cbar = ax.figure.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_ticks([]) # Rimuoviamo i numeri se vogliamo solo il gradiente visivo
        cbar.set_label('tau', fontsize=fontsize)
        cbar.ax.tick_params(labelsize=fontsize)

    return ax


def filter_and_enance_data(ep_list, filtering_function=lambda x: True):
    filtered_ep_list = []
    for ep in ep_list:
        if filtering_function(ep):
            # Calcolo metriche aggiuntive
            ep_ext = ep.copy()
            ep_ext['velocity'] = ep['distance_traveled'] / ep['length'] if ep['length'] > 0 else 0
            ep_ext['weighted_success'] = ep['success'] * ep['path_tortuosity']
            ep_ext['SPL'] = ep['success'] * (ep['path_length'] / max(ep['distance_traveled'], ep['path_length']))
            ep_ext['SPL2'] = ep['success'] * (ep['path_length'] / ep['distance_traveled']) if ep['distance_traveled'] > 0 else 0
            
            ep_ext['success_nc'] = ep['success'] if ep['collisions'] == 0 else 0
            filtered_ep_list.append(ep_ext)
            
    return filtered_ep_list

def get_outlier_indices(episodes, metric, percentage=0.05):
    """
    Restituisce gli indici degli outlier rimuovendo
    una percentuale simmetrica dai valori più bassi e più alti.

    percentage = frazione totale da rimuovere (es. 0.05 = 5%)
    La rimozione è simmetrica: metà sotto, metà sopra.
    """

    n = len(episodes)
    if n == 0:
        return set()

    if not (0 <= percentage < 1):
        raise ValueError("percentage deve essere tra 0 e 1")

    # Numero totale di punti da rimuovere
    total_to_remove = int(math.floor(n * percentage))

    if total_to_remove == 0:
        return set()

    # Divisione simmetrica
    per_side = total_to_remove // 2

    if per_side == 0:
        return set()

    # Costruzione lista (valore, indice)
    indexed_values = [
        (ep[metric], i)
        for i, ep in enumerate(episodes)
        if metric in ep
    ]

    if len(indexed_values) < 2 * per_side:
        return set()

    indexed_values.sort(key=lambda x: x[0])

    low_indices = {idx for _, idx in indexed_values[:per_side]}
    high_indices = {idx for _, idx in indexed_values[-per_side:]}

    return low_indices | high_indices
                  
def load_test_from_csv(csv_path, filtering_function = lambda x: True):
    control_df = pd.read_csv(csv_path)
    data = {}
    print(f'loafing data from {csv_path}')
    for p_name in control_df['policy_name']:
        control_row = control_df.query(f"policy_name == '{p_name}'")
        
        specific_test_name = control_row['test_name'].values[0]
        json_path = csv_path.rsplit('/', 1)[0] + '/' + specific_test_name + '_info.json'
        
        with open(json_path, 'r') as f:
            specific_test_data = json.load(f)
        
        ep_data = filter_and_enance_data(specific_test_data['data'], filtering_function)
        print(f'\t{len(ep_data)} data for {p_name}')
        # 2. Elaborazione e calcolo metriche personalizzate
        data[p_name] = ep_data
    
    return data
    
def load_test_data(csv_list, filtering_function = lambda x: True):
    data_liste = []
    labels = []

    # Lista delle policy da analizzare
    policies = ['basic_1_4205364', 'simple_0_4164735', 'simple_wp_1_4599899', 'complex_1_4165576', 'complex_wp_1_4611744']
    
    for p in policies:
        raw_data = {}
        th_list = [0.01, 0.25, 0.5, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
        for env in ['obstacles_simple', 'obstacles_complex']:
            
            for i, th_path in enumerate(csv_list):
                # 1. Caricamento dati
                control_df = pd.read_csv(th_path)
                control_row = control_df.query(f"policy_name == '{p}' and env_name == '{env}'")
                
                if control_row.empty:
                    continue
                    
                specific_test_name = control_row['test_name'].values[0]
                json_path = th_path.rsplit('/', 1)[0] + '/' + specific_test_name + '_info.json'
                
                with open(json_path, 'r') as f:
                    specific_test_data = json.load(f)
                
                # 2. Elaborazione e calcolo metriche personalizzate
                current_episodes = filter_and_enance_data(specific_test_data['data'], filtering_function)

                # 4. Aggregazione nel dizionario raw_data
                if th_list[i] in raw_data:
                    raw_data[th_list[i]].extend(current_episodes)
                else:
                    raw_data[th_list[i]] = current_episodes
                        
        data_liste.append(raw_data)
        labels.append(p)

    return data_liste, labels


def multy_plot_filtered(csv_list, m1, m2, policies, ncols=2, filtering_function=lambda x: True, outlier_percentage=2):
    data_liste = []
    metriche_liste = []
    labels = []
    
    # Lista delle policy da analizzare
    for p in policies:
        raw_data = {}
        th_list = [0.01, 0.25, 0.5, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
        
        for i, th_path in enumerate(csv_list):
            # 1. Caricamento dati
            control_df = pd.read_csv(th_path)
            control_row = control_df.query(f"policy_name == '{p}'")
            
            if control_row.empty:
                continue
                
            specific_test_name = control_row['test_name'].values[0]
            json_path = th_path.rsplit('/', 1)[0] + '/' + specific_test_name + '_info.json'
            
            with open(json_path, 'r') as f:
                specific_test_data = json.load(f)
            
            
            # 2. Elaborazione e calcolo metriche personalizzate
            current_episodes = filter_and_enance_data(specific_test_data['data'], filtering_function)
            
            # 4. Aggregazione nel dizionario raw_data
            if th_list[i] in raw_data:
                raw_data[th_list[i]].extend(current_episodes)
            else:
                raw_data[th_list[i]] = current_episodes

        # --- All'interno della tua funzione, dopo aver accumulato tutti gli episodi ---

        for th, episodes in raw_data.items():
            if outlier_percentage > 0 :
                # 1. Identifichiamo gli indici degli outliers per m1, m2 (e m3 se vuoi)
                outliers_to_remove = set()
                outliers_to_remove |= get_outlier_indices(episodes, m1, outlier_percentage)
                outliers_to_remove |= get_outlier_indices(episodes, m2, outlier_percentage)

                cleaned_episodes = [
                    ep for i, ep in enumerate(episodes) 
                    if i not in outliers_to_remove
                ]
                
                raw_data[th] = cleaned_episodes
            else:
                raw_data[th] = episodes
                        
        data_liste.append([raw_data])
        metriche_liste.append((m1, m2))
        labels.append([p])

    # --- Parte di Visualizzazione ---
    nrows = math.ceil(len(data_liste) / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6*ncols, 4*nrows))
    axes_flat = np.atleast_1d(axes).flatten()

    for i in range(len(data_liste)):
        if i < len(axes_flat):
            plot_metric_series_on_ax(
                ax=axes_flat[i],
                data_list_x=data_liste[i],
                data_list_y=data_liste[i],
                metric_x=metriche_liste[i][0],
                metric_y=metriche_liste[i][1],
                labels=labels[i]
            )
            # Stampa di debug per vedere quanti episodi restano dopo la pulizia
            remaining_data_count = np.array([len(data_liste[i][0][key]) for key in data_liste[i][0]] if len(data_liste[i][0]) > 0 else [0]).mean()
            print(f'Episodi per {labels[i]}: {remaining_data_count}')

    plt.tight_layout()
    plt.show()  


def calculate_stats_from_list(data_list, percentiles=[1] + list(range(0,100,5)) + [99]):
    """
    Calcola statistiche robuste gestendo anche Booleani e NaN.
    """
    grouped_values = defaultdict(list)
    
    # 1. Raccogliamo i dati puliti
    for entry in data_list:
        if not isinstance(entry, dict):
            continue
            
        for key, value in entry.items():
            # Nota: bool è sottoclasse di int in Python, quindi passa questo check.
            # Accettiamo bool, int, float, numpy types
            if isinstance(value, (int, float, bool, np.number)):
                # Escludiamo NaN se è un float, ma lasciamo passare i bool
                if isinstance(value, float) and np.isnan(value):
                    continue
                grouped_values[key].append(value)

    stats_results = {}
    
    # 2. Calcoliamo le statistiche
    for key, values in grouped_values.items():
        if not values:
            continue
            
        # Creiamo l'array numpy
        arr = np.array(values)
        
        # --- FIX PER L'ERRORE ---
        # Se l'array è di tipo booleano, lo convertiamo in float (0.0 e 1.0)
        # Questo permette di calcolare media (tasso successo) e percentili senza errori.
        if arr.dtype == bool:
            arr = arr.astype(float)
        # ------------------------
        
        # Calcolo statistiche base
        key_stats = {
            "mean": np.mean(arr),
            "std": np.std(arr),
            "min": np.min(arr),
            "max": np.max(arr),
            "count": len(arr)
        }
        
        # Calcolo percentili
        for p in percentiles:
            # np.percentile vuole valori 0-100
            key_stats[f"p{p}"] = np.percentile(arr, p)
            
        stats_results[key] = key_stats
        
    return stats_results
    
    
    