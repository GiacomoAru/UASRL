import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_confronto_metriche(lista_episodi, chiave_x, chiave_y):
    
    # 2. Estrazione delle metriche
    # Creiamo una lista di valori per X e Y prendendoli da ogni episodio
    valori_x = [ep[chiave_x] for ep in lista_episodi if chiave_x in ep]
    valori_y = [ep[chiave_y] for ep in lista_episodi if chiave_y in ep]
    
    # Creiamo un DataFrame Pandas (molto più comodo per Seaborn)
    df = pd.DataFrame({
        chiave_x: valori_x,
        chiave_y: valori_y
    })

    # 3. Creazione del Joint Plot
    # kind="reg" aggiunge anche una linea di regressione per vedere il trend
    g = sns.jointplot(data=df, x=chiave_x, y=chiave_y, kind="reg", 
                      marginal_kws=dict(bins=20, fill=True))

    # Aggiungiamo un titolo e sistemiamo lo spazio
    g.fig.suptitle(f"Confronto tra {chiave_x} e {chiave_y}", y=1.02)
    
    plt.show()

# Esempio di utilizzo:
# plot_confronto_metriche('risultati.json', 'reward', 'loss')