import papermill as pm
import os

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="papermill")

list_i = [4]
list_pressure = [1]
list_water = [0.15]

notebook_in = 'auto.ipynb'
notebook_analisi = 'analisi.ipynb'
is_first = True

for i in list_i:
    for p in list_pressure:
        for w in list_water:
            print(f"\n \n Avvio simulazione: moltiplicatore={i}, pressione={p}, perc water = {w} ...")

            pm.execute_notebook(
                notebook_in,
                os.devnull,  # Redirige e scarta il file .ipynb in uscita
                parameters=dict(moltiplicatore=i, pressione=p, water = w , is_first_run=is_first)
            )

            # Esecuzione del notebook di analisi per aggiornare i risultati (k_max)
            print(f"Aggiornamento analisi...")
            pm.execute_notebook(
                notebook_analisi,
                os.devnull
            )

            is_first = False
