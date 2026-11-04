import os
import shutil
import papermill as pm

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="papermill")

list_i = [1.75]
list_pressure = [1]
list_water = [0.15]

notebook_in = 'auto/auto.ipynb'
is_first = True

for i in list_i:
    for p in list_pressure:
        for w in list_water:
            print(f"\n \n Avvio simulazione: moltiplicatore={i}, pressione={p}, perc water = {w} ...")

            pm.execute_notebook(
                   input_path=notebook_in,
                   parameters={'moltiplicatore': i, 'pressione': p, 'water_perc': w, 'is_first_run': is_first},
                   output_path=os.devnull  # Redirige e scarta il file .ipynb in uscita
               )

            is_first = False

