import numpy as np
import matplotlib.pyplot as plt

class ACOMetrics:
    """
    Classe utilitaire pour enregistrer et analyser les performances de l'ACO.
    """
    def __init__(self):
        self.best_scores_history = []
        self.diversity_history = []
        self.stagnation_history = []
        self.stagnation_counter = 0
        self.last_best_score = np.inf

    def update(self, all_paths_scores, current_global_best_score):
        """
        Met à jour les métriques à chaque itération de l'algorithme.
        """
        # Calcul de la stagnation
        if current_global_best_score < self.last_best_score:
            self.stagnation_counter = 0
            self.last_best_score = current_global_best_score
        else:
            self.stagnation_counter += 1
        
        # Calcul de la diversité (écart-type des scores de la population)
        scores = [score for _, score in all_paths_scores]
        diversity = np.std(scores)

        # Enregistrement des historiques
        self.best_scores_history.append(current_global_best_score)
        self.diversity_history.append(diversity)
        self.stagnation_history.append(self.stagnation_counter)

        return self.stagnation_counter, diversity

    def plot_metrics(self):
        """Génère les graphiques d'analyse pour le rapport."""
        iterations = range(len(self.best_scores_history))
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        
        # Graphique 1 : Convergence
        ax1.plot(iterations, self.best_scores_history, 'b-', label='Meilleur Score')
        ax1.set_ylabel('Distance')
        ax1.set_title('Convergence')
        ax1.grid(True)
        ax1.legend()

        # Graphique 2 : Diversité
        ax2.plot(iterations, self.diversity_history, 'g-', label='Diversité (Std Dev)')
        ax2.set_ylabel('Écart-Type')
        ax2.set_title('Diversité de la population')
        ax2.grid(True)
        ax2.legend()

        # Graphique 3 : Stagnation
        ax3.plot(iterations, self.stagnation_history, 'r-', label='Compteur Stagnation')
        ax3.set_ylabel('Tours sans amélioration')
        ax3.set_title('Stagnation')
        ax3.set_xlabel('Itérations')
        ax3.grid(True)
        ax3.legend()

        plt.tight_layout()
        plt.show()