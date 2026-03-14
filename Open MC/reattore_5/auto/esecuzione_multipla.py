import papermill as pm
import os

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="papermill")

list_i = [1.045]
list_pressure = [100 ]

notebook_in = 'auto.ipynb'
is_first = True

for i in list_i:
    for p in list_pressure:
        print(f"\n \n Avvio simulazione: moltiplicatore={i}, pressione={p} ...")
        
        pm.execute_notebook(
            notebook_in,
            os.devnull,  # Redirige e scarta il file .ipynb in uscita
            parameters=dict(moltiplicatore=i, pressione=p, is_first_run=is_first)
        )
        
        is_first = False