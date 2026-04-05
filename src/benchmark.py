import numpy as np
import matplotlib.pyplot as plt
from tsp_data import TSPInstance
from aco_solver import ACOSolver
from hybrid_solver import HybridACOSolver

def run_comparative_benchmark():
    print("--- Génération de l'instance de test (50 villes) ---")
    instance = TSPInstance()
    instance.generate_random(n_cities=50, seed=123)

    N_ANTS = 30
    N_ITER = 150
    N_BEST = 5

    # ACO Classique
    print(f"\n[1/2] Lancement ACO classique...")
    classic_solver = ACOSolver(instance, n_ants=N_ANTS, n_iter=N_ITER,
                               alpha=1, beta=2, decay=0.95, n_best=N_BEST)
    c_path, c_score, c_metrics = classic_solver.solve()
    print(f"   -> Score classique final : {c_score:.2f}")

    # ACO Hybride
    print(f"\n[2/2] Lancement ACO hybride (Q-Learning)...")
    hybrid_solver = HybridACOSolver(instance, n_ants=N_ANTS, n_iter=N_ITER, n_best=N_BEST)
    hybrid_solver.agent.epsilon = 0.2
    h_path, h_score, h_metrics, h_actions = hybrid_solver.solve()
    print(f"   -> Score hybride final : {h_score:.2f}")

    # Visualisation
    print("\n--- Génération des graphiques ---")
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))
    iterations = range(N_ITER)

    # Convergence
    ax1.plot(iterations, c_metrics.best_scores_history, 'b--', label=f'Classique (Fin: {c_score:.1f})', alpha=0.7)
    ax1.plot(iterations, h_metrics.best_scores_history, 'r-', label=f'Hybride (Fin: {h_score:.1f})', linewidth=2)
    ax1.set_title("Convergence : Classique vs Hybride")
    ax1.set_ylabel("Distance Totale")
    ax1.legend()
    ax1.grid(True)

    # Diversité
    ax2.plot(iterations, c_metrics.diversity_history, 'b--', label='Diversité Classique', alpha=0.5)
    ax2.plot(iterations, h_metrics.diversity_history, 'r-', label='Diversité Hybride')
    ax2.set_title("Comparaison de la Diversité (Écart-Type des fourmis)")
    ax2.set_ylabel("Diversité")
    ax2.legend()
    ax2.grid(True)

    # Actions de l'IA
    ax3.plot(h_actions, 'o-', color='purple', markersize=4, label='Choix IA')
    ax3.set_title("Stratégie dynamique de l'IA au cours du temps")
    ax3.set_ylabel("Action")
    ax3.set_xlabel("Itérations")
    ax3.set_yticks([0, 1, 2])
    ax3.set_yticklabels(["Exploration\n(Alpha=1, Rho=0.5)", "Normal\n(Alpha=1, Rho=0.95)", "Intensification\n(Alpha=5, Rho=0.99)"])
    ax3.grid(True, axis='y')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_comparative_benchmark()
