import os
import shutil
import papermill as pm


# --- Configurazione path e input ---
lista_file_parametri = ['../parametri_7.txt']  # Lista dei file parametri da processare
notebook_path = os.path.abspath('auto.ipynb')

for file_param in lista_file_parametri:
    # Parsing del nome file (es. 'parametri_1')
    nome_run = os.path.splitext(os.path.basename(file_param))[0]
    out_dir = os.path.abspath(f"./output_{nome_run}")
    
    # 1. Creazione della directory dedicata
    os.makedirs(out_dir, exist_ok=True)
    
    # Path assoluto del file parametri da iniettare nel notebook
    param_file_abs = os.path.abspath(file_param)
    out_nb_path = os.path.join(out_dir, f"fixed_out_{nome_run}.ipynb")
    
    print(f"Avvio simulazione: {nome_run} -> {out_dir}")
    
    # 3. Esecuzione del notebook confinata in out_dir
    pm.execute_notebook(
        input_path=notebook_path,
        output_path=out_nb_path,
        parameters={'param_file': param_file_abs},
        cwd=out_dir
    )
