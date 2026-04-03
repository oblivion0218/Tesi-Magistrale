import subprocess

print("Inizio esecuzione sequenziale...")

print("Esecuzione di circolarita_k_max.py in reattore_8 a <= 50PCM...")
subprocess.run(["python", "circolarita_k_max.py"], cwd="reattore_8/50PCM/circolarita_k_max", check=True)

print("Esecuzione di run_pot.py in reattore_8 a <= 50PCM...")
subprocess.run(["python", "run_pot.py"], cwd="reattore_8/50PCM", check=True)

print("Tutte le simulazioni sono terminate con successo.")

'''
import subprocess
import time

attesa_secondi = 24 * 3600  # 50 ore in secondi

print("Avvio run_pot.py in reattore_8...")
# Popen avvia il processo in background permettendo allo script di proseguire con il timer
subprocess.Popen(["python", "run_pot.py"], cwd="reattore_8")

print("Attesa di 24 ore...")
time.sleep(attesa_secondi)

print("Avvio run_max.py in reattore_8...")
subprocess.Popen(["python", "run_max.py"], cwd="reattore_8")

print("Attesa di 24 ore...")
time.sleep(attesa_secondi)

print("Avvio auto.py in reattore_9...")
subprocess.Popen(["python", "auto.py"], cwd="reattore_9")

print("Tutti i processi di avvio sono stati schedulati.")

'''