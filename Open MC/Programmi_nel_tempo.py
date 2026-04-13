#import subprocess

#print("Inizio esecuzione sequenziale...")
#
#print("Esecuzione di circolarita_k_max.py in reattore_8 a <= 50PCM...")
#subprocess.run(["python", "circolarita_k_max.py"], cwd="reattore_8/50PCM/circolarita_k_max", check=True)
#
#print("Esecuzione di run_pot.py in reattore_8 a <= 50PCM...")
#subprocess.run(["python", "run_pot.py"], cwd="reattore_8/50PCM", check=True)
#
#print("Tutte le simulazioni sono terminate con successo.")

import subprocess
import time

attesa_secondi =  3600 * 3  # 50 ore in secondi
print("Attesa di 24 ore...")
time.sleep(attesa_secondi)

print("Avvio run_pot.py in reattore_8...")
# Popen avvia il processo in background permettendo allo script di proseguire con il timer
subprocess.Popen(["python", "analisi.ipynb"], cwd="reattore_9")

print("Avvio run_pot.py in reattore_8...")
# Popen avvia il processo in background permettendo allo script di proseguire con il timer
subprocess.Popen(["python", "auto_save.py"], cwd="..")