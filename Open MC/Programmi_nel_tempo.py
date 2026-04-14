#import subprocess

#print("Inizio esecuzione sequenziale...")
#
#print("Esecuzione di circolarita_k_max.py in reattore_8 a <= 50PCM...")
#subprocess.run(["python", "circolarita_k_max.py"], cwd="reattore_8/50PCM/circolarita_k_max", check=True)
#
#print("Esecuzione di run_pot.py in reattore_8 a <= 50PCM...")
#subprocess.run(["python", "run_pot.py"], cwd="reattore_8/50PCM", check=True)
#


#import subprocess
#import time

#attesa_secondi =  3600 * 3  # 50 ore in secondi
#print("Attesa di 24 ore...")
#time.sleep(attesa_secondi)

#print("Avvio run_pot.py in reattore_8...")
# Popen avvia il processo in background permettendo allo script di proseguire con il timer
#subprocess.Popen(["python", "arricchimenti_VS_pitch.py"], cwd="reattore_9", check=True)

#print("Avvio run_pot.py in reattore_8...")
# Popen avvia il processo in background permettendo allo script di proseguire con il timer
#subprocess.Popen(["python", "run_pot.py"], cwd="reattore_9", check=True)

#print("Tutte le simulazioni sono terminate con successo.")

import subprocess
import time

# Imposta il tempo di attesa (es. 3 ore)
attesa_secondi =  3*3600 
print(f"Inizio conteggio: attesa di {attesa_secondi/3600} ore...")
time.sleep(attesa_secondi)

print("--- FASE 1: Avvio arricchimenti_VS_pitch.py ---")
# Usiamo subprocess.run senza check=True così continua anche se il primo script fallisce
subprocess.run(["python", "arricchimenti_VS_pitch.py"], cwd="reattore_9")

print("--- FASE 1 COMPLETATA (o terminata con errore) ---")

print("--- FASE 2: Avvio run_pot.py ---")
# Ora parte il secondo
subprocess.run(["python", "run_pot.py"], cwd="reattore_9")

print("Tutte le simulazioni programmate sono state eseguite.")
