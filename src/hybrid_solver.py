import numpy as np
from aco_solver import ACOSolver
from ml_agent import QLearningAgent

class HybridACOSolver(ACOSolver):
    """
    Version améliorée de l'ACO qui utilise un agent Q-Learning pour
    adapter dynamiquement alpha, beta et decay en cours de route.
    """
    def __init__(self, instance, n_ants=20, n_iter=100, n_best=5):
        super().__init__(instance, n_ants, n_iter, alpha=1, beta=2, decay=0.95, n_best=n_best)

        # Tactiques disponibles pour l'agent : (alpha, beta, decay)
        self.tactics = [
            (1, 5, 0.5),   # 0: Exploration forte (oublie vite, suit la carte)
            (1, 2, 0.95),  # 1: Equilibré (classique)
            (5, 1, 0.99)   # 2: Intensification (suit la mémoire, évapore peu)
        ]

        self.agent = QLearningAgent(self.tactics)
        self.history_actions = []

    def solve(self):
        """Surcharge de solve pour inclure le cycle ML."""
        best_path = None
        best_score = np.inf
        stagnation_counter = 0

        print(f"[HYBRID] Démarrage IA sur '{self.instance.name}'...")

        for i in range(self.n_iter):
            # L'agent observe la stagnation et choisit les paramètres
            params = self.agent.choose_action(stagnation_counter)
            self.alpha, self.beta, self.decay = params
            self.history_actions.append(self.agent.last_action_idx)

            # L'ACO construit et évalue les solutions
            all_paths = self._construct_solutions()
            self._update_pheromones(all_paths)
            iter_best_path, iter_best_score = min(all_paths, key=lambda x: x[1])

            # Calcul de la récompense
            if iter_best_score < best_score:
                best_score = iter_best_score
                best_path = iter_best_path
                stagnation_counter = 0
                reward = 50
                print(f"  > Iter {i+1} [Action {self.agent.last_action_idx}]: nouveau record = {best_score:.2f}")
            else:
                stagnation_counter += 1
                reward = -1

            self.metrics.update(all_paths, best_score)

            # L'agent apprend
            self.agent.learn(stagnation_counter, reward)

        return best_path, best_score, self.metrics, self.history_actions


if __name__ == "__main__":
    from tsp_data import TSPInstance
    import matplotlib.pyplot as plt

    instance = TSPInstance()
    instance.generate_random(n_cities=20, seed=42)

    solver = HybridACOSolver(instance, n_ants=20, n_iter=100)
    best_path, best_score, metrics, actions = solver.solve()

    print(f"\n[RÉSULTAT HYBRIDE] Score: {best_score:.2f}")

    metrics.plot_metrics()

    plt.figure(figsize=(10, 3))
    plt.plot(actions, 'o-', color='purple')
    plt.title("Choix de l'IA au fil du temps (0=Explore, 1=Normal, 2=Intensifie)")
    plt.xlabel("Itérations")
    plt.ylabel("Action")
    plt.yticks([0, 1, 2], ["Exploration", "Normal", "Intensification"])
    plt.grid(True)
    plt.show()
