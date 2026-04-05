import numpy as np
import random
from tsp_data import TSPInstance
from aco_metrics import ACOMetrics

class ACOSolver:
    """
    Solveur ACO (Ant Colony Optimization) Classique.
    Implémente la stratégie 'Ant System' avec une touche d'élitisme (n_best).
    """
    def __init__(self, instance: TSPInstance, n_ants=20, n_iter=100, alpha=1, beta=2, decay=0.95, n_best=5):
        self.instance = instance
        self.dist_matrix = instance.dist_matrix
        self.n_cities = len(instance.cities)

        self.n_ants = n_ants
        self.n_iter = n_iter
        self.alpha = alpha      # Poids de la phéromone (mémoire)
        self.beta = beta        # Poids de l'heuristique (visibilité)
        self.decay = decay      # Taux d'évaporation (rho)
        self.n_best = n_best    # Nombre de fourmis qui déposent (élitisme)

        # Initialisation des phéromones (tau) : petite valeur uniforme
        self.pheromone = np.ones((self.n_cities, self.n_cities)) / self.n_cities

        self.metrics = ACOMetrics()

    def solve(self):
        """Exécute l'algorithme et retourne le meilleur chemin et l'historique."""
        best_path = None
        best_score = np.inf

        print(f"[ACO] Démarrage sur '{self.instance.name}' ({self.n_iter} itérations)...")

        for i in range(self.n_iter):
            all_paths = self._construct_solutions()
            self._update_pheromones(all_paths)

            iter_best_path, iter_best_score = min(all_paths, key=lambda x: x[1])

            if iter_best_score < best_score:
                best_score = iter_best_score
                best_path = iter_best_path

            self.metrics.update(all_paths, best_score)

            if (i + 1) % 10 == 0:
                print(f"  > Iter {i+1}/{self.n_iter} : Best = {best_score:.2f}")

        return best_path, best_score, self.metrics

    def _construct_solutions(self):
        """Chaque fourmi construit un chemin complet."""
        all_paths = []
        for _ in range(self.n_ants):
            path = self._build_one_path()
            score = self._calculate_score(path)
            all_paths.append((path, score))
        return all_paths

    def _build_one_path(self):
        """Une fourmi parcourt les villes."""
        start_node = random.randint(0, self.n_cities - 1)
        path = [start_node]
        visited = {start_node}

        current = start_node
        for _ in range(self.n_cities - 1):
            next_node = self._select_next_city(current, visited)
            path.append(next_node)
            visited.add(next_node)
            current = next_node

        return path

    def _select_next_city(self, current, visited):
        """Règle de transition probabiliste (Roulette Wheel)."""
        probs = []
        candidates = []

        for city in range(self.n_cities):
            if city in visited:
                continue

            tau = self.pheromone[current][city]
            dist = self.dist_matrix[current][city]
            eta = 1.0 / (dist + 1e-10)

            prob = (tau ** self.alpha) * (eta ** self.beta)
            probs.append(prob)
            candidates.append(city)

        probs = np.array(probs)
        total = probs.sum()
        if total == 0:
            return random.choice(candidates)

        probs = probs / total
        return np.random.choice(candidates, p=probs)

    def _calculate_score(self, path):
        dist = 0
        for i in range(len(path) - 1):
            dist += self.dist_matrix[path[i]][path[i+1]]
        dist += self.dist_matrix[path[-1]][path[0]]
        return dist

    def _update_pheromones(self, all_paths):
        # Évaporation
        self.pheromone *= self.decay

        # Renforcement élitiste
        sorted_paths = sorted(all_paths, key=lambda x: x[1])

        for path, score in sorted_paths[:self.n_best]:
            deposit = 1.0 / score
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                self.pheromone[u][v] += deposit
                self.pheromone[v][u] += deposit

            u, v = path[-1], path[0]
            self.pheromone[u][v] += deposit
            self.pheromone[v][u] += deposit


if __name__ == "__main__":
    instance = TSPInstance()
    instance.generate_random(n_cities=20, seed=42)

    solver = ACOSolver(instance, n_ants=20, n_iter=50, alpha=1, beta=5)
    best_path, best_score, metrics = solver.solve()

    print(f"\n[RÉSULTAT] Distance optimale trouvée : {best_score:.2f}")
    metrics.plot_metrics()
