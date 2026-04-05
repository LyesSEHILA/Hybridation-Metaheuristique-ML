import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tsp_data import TSPInstance
from aco_solver import ACOSolver
from hybrid_solver import HybridACOSolver

def run_mass_benchmark():
    SIZES = [20, 50, 100]
    N_RUNS = 10
    N_ANTS = 20
    N_ITER = 100

    results = []

    print("=== Démarrage de la campagne de tests massifs ===")

    for size in SIZES:
        print(f"\n--- Instance de taille {size} ---")
        instance = TSPInstance(name=f"Test_{size}")
        # Seed fixe pour que l'instance soit identique sur les N_RUNS exécutions
        instance.generate_random(n_cities=size, seed=size * 10)

        for run in range(N_RUNS):
            print(f"  > Taille {size} | Run {run+1}/{N_RUNS}...")

            # ACO classique
            t_start = time.time()
            classic_solver = ACOSolver(instance, n_ants=N_ANTS, n_iter=N_ITER, alpha=1, beta=2, decay=0.95)
            _, c_score, _ = classic_solver.solve()
            t_classic = time.time() - t_start

            # ACO hybride (Q-Learning)
            t_start = time.time()
            hybrid_solver = HybridACOSolver(instance, n_ants=N_ANTS, n_iter=N_ITER)
            hybrid_solver.agent.epsilon = 0.2
            _, h_score, _, _ = hybrid_solver.solve()
            t_hybrid = time.time() - t_start

            results.append({
                "Taille (Villes)": size,
                "Run ID": run + 1,
                "Algorithme": "Classique",
                "Score (Distance)": c_score,
                "Temps (s)": t_classic
            })
            results.append({
                "Taille (Villes)": size,
                "Run ID": run + 1,
                "Algorithme": "Hybride (IA)",
                "Score (Distance)": h_score,
                "Temps (s)": t_hybrid
            })

    df = pd.DataFrame(results)
    df.to_csv("resultats_massifs.csv", index=False)
    print("\n=== Tests terminés. Données sauvegardées dans 'resultats_massifs.csv' ===")

    generate_analysis_plots(df)


def generate_analysis_plots(df):
    """Génère des graphiques statistiques (boxplots) pour analyser la stabilité et la qualité."""
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    sns.boxplot(data=df, x="Taille (Villes)", y="Score (Distance)", hue="Algorithme", palette="Set1")
    plt.title("Qualité des solutions et stabilité (10 runs)")
    plt.ylabel("Distance Totale (plus bas = meilleur)")

    plt.subplot(1, 2, 2)
    sns.barplot(data=df, x="Taille (Villes)", y="Temps (s)", hue="Algorithme", palette="Set1", errorbar='sd')
    plt.title("Temps de calcul moyen")
    plt.ylabel("Temps (secondes)")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_mass_benchmark()
