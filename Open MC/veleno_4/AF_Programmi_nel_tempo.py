import subprocess
print("Inizio esecuzione sequenziale...")
print("Esecuzione di poison_multipli.py in veleno_4 ...")
subprocess.run(["python", "poison_multipli_AF.py"],  check=True)
print("Esecuzione di k_trans in veleno_4 ")
subprocess.run(["python", "k_transitorio_AF.py"], check=True)