import subprocess
import time
from datetime import datetime

def git_auto_sync():
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Inizio sincronizzazione...")
    try:
        # 1. Aggiunge tutti i file modificati
        subprocess.run(["git", "add", "."], check=True)
        
        # 2. Esegue il commit. check=False perché se non ci sono modifiche, 
        # git restituisce un codice di errore (exit status 1) e bloccherebbe lo script.
        commit_result = subprocess.run(["git", "commit", "-m", "."], capture_output=True, text=True)
        
        if "nothing to commit" in commit_result.stdout:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Nessuna modifica da committare.")
        else:
            # 3. Esegue il push solo se c'è stato un commit effettivo
            subprocess.run(["git", "push"], check=True)
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Push completato con successo.")
            
    except subprocess.CalledProcessError as e:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Errore critico in esecuzione Git: {e}")

if __name__ == "__main__":
    ore_intervallo = 1
    secondi_intervallo = ore_intervallo * 36
    
    print(f"Script di auto-sync avviato. Frequenza: {ore_intervallo} ore.")
    
    while True:
        git_auto_sync()
        time.sleep(secondi_intervallo)