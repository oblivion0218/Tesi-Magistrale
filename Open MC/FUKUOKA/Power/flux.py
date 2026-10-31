import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve, eigs

# --- Parametri Geometrici (da parametri.txt) ---
R_TARGET = 2.0
R_PIPE = 5.0
R_MID = 22.5     # Raggio fittizio separazione anelli (metà core)
R_CORE = 40.0
R_REF = 50.0     # Riflettore
N = 400
r = np.linspace(1e-3, R_REF, N)
dr = r[1] - r[0]

# --- Configurazioni a pari k=0.975 (da result_final.txt) ---
configs = [
    {"name": "Config A (Alto e_int, Basso e_ext)", "e_int": 15.87, "e_ext": 0.000},
    {"name": "Config B (Basso e_int, Alto e_ext)", "e_int": 2.38,  "e_ext": 10.000}
]

fig, axs = plt.subplots(1, 2, figsize=(16, 7))

for cfg in configs:
    e_int = cfg["e_int"]
    e_ext = cfg["e_ext"]
    
    # Sezioni d'Urto Sintetiche a 2 Gruppi
    D1, D2 = np.ones(N)*1.5, np.ones(N)*0.4
    Sa1, Sa2 = np.ones(N)*0.01, np.ones(N)*0.03
    Ss12 = np.ones(N)*0.02
    vSf1, vSf2 = np.zeros(N), np.zeros(N)
    
    for i, x in enumerate(r):
        if x <= R_TARGET:          # Target 5% U
            vSf2[i], Sa2[i] = 0.05, 0.06
        elif x <= R_PIPE:          # Acqua 15% D2O
            Sa2[i], Ss12[i] = 0.015, 0.05
        elif x <= R_MID:           # Anello Interno
            vSf2[i] = 0.008 * e_int
            Sa2[i] = 0.02 + 0.001 * e_int
        elif x <= R_CORE:          # Anello Esterno
            vSf2[i] = 0.008 * e_ext
            Sa2[i] = 0.02 + 0.001 * e_ext
        else:                      # Riflettore Grafite
            Sa2[i], Ss12[i], D1[i], D2[i] = 0.0002, 0.01, 1.0, 0.8

    # Costruzione Matrici (Differenze Finite, 1D Cilindrica)
    A = lil_matrix((2*N, 2*N))
    F = lil_matrix((2*N, 2*N))
    S = np.zeros(2*N)
    S_adj = np.zeros(2*N)
    
    for i in range(N):
        if r[i] <= R_TARGET: S[i] = 1.0 
        S_adj[i], S_adj[N+i] = vSf1[i], vSf2[i]
        F[i, i], F[i, N+i] = vSf1[i], vSf2[i]
        A[i, i] = Sa1[i] + Ss12[i]
        A[N+i, N+i] = Sa2[i]
        A[N+i, i] = -Ss12[i]
        
        # Operatore di Diffusione 1/r d/dr (r D d/dr)
        if 0 < i < N-1:
            rp, rm = r[i] + dr/2, r[i] - dr/2
            cp, cm = D1[i]*rp/(r[i]*dr**2), D1[i]*rm/(r[i]*dr**2)
            A[i, i] += cp + cm; A[i, i+1] = -cp; A[i, i-1] = -cm
            cp2, cm2 = D2[i]*rp/(r[i]*dr**2), D2[i]*rm/(r[i]*dr**2)
            A[N+i, N+i] += cp2 + cm2; A[N+i, N+i+1] = -cp2; A[N+i, N+i-1] = -cm2
        elif i == 0:
            rp = r[i] + dr/2
            cp, cp2 = D1[i]*rp/(r[i]*dr**2), D2[i]*rp/(r[i]*dr**2)
            A[i, i] += cp; A[i, i+1] = -cp
            A[N+i, N+i] += cp2; A[N+i, N+i+1] = -cp2
        elif i == N-1:
            rm = r[i] - dr/2
            cm, cm2 = D1[i]*rm/(r[i]*dr**2), D2[i]*rm/(r[i]*dr**2)
            A[i, i] += cm; A[i, i-1] = -cm
            A[N+i, N+i] += cm2; A[N+i, N+i-1] = -cm2

    A_csc, F_csc = A.tocsc(), F.tocsc()
    
    # Ricalibrazione subcritica a k=0.975 e soluzione
    k_eff = np.real(eigs(F_csc, k=1, M=A_csc, sigma=1.0)[0][0])
    F_sub = F_csc * (0.975 / k_eff)
    
    phi_thermal = spsolve(A_csc - F_sub, S)[N:]
    phi_adj_thermal = spsolve(A_csc.T - F_sub.T, S_adj)[N:]
    
