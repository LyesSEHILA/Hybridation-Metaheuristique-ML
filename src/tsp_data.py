import numpy as np
import matplotlib.pyplot as plt
import json
import os

class TSPInstance:
    """
    Classe représentant une instance du problème TSP (Voyageur de Commerce).
    Gère la génération, le calcul des distances, la sauvegarde et le chargement.
    """
    def __init__(self, name="Instance", cities=None):
        self.name = name
        self.cities = cities  # Liste de tuples (x, y)
        self.dist_matrix = None
        self.optimal_score = None  # Pour stocker le best known si on le connait

        if self.cities is not None:
            self._compute_distance_matrix()

    def generate_random(self, n_cities, width=100, height=100, seed=None):
        """Génère une distribution uniforme de villes."""
        if seed is not None:
            np.random.seed(seed)

        raw_cities = np.random.rand(n_cities, 2)
        # Mise à l'échelle
        self.cities = list(map(tuple, raw_cities * [width, height]))
        self.name = f"Random_{n_cities}_S{seed if seed else 'X'}"
        self._compute_distance_matrix()
        print(f"[DATA] Instance '{self.name}' générée avec {n_cities} villes.")

    def _compute_distance_matrix(self):
        """Calcule la matrice des distances euclidiennes."""
        n = len(self.cities)
        self.dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    p1 = np.array(self.cities[i])
                    p2 = np.array(self.cities[j])
                    self.dist_matrix[i][j] = np.linalg.norm(p1 - p2)
                else:
                    self.dist_matrix[i][j] = np.inf

    def save_to_json(self, folder="datasets"):
        """Sauvegarde l'instance sur le disque pour réutilisation."""
        os.makedirs(folder, exist_ok=True)

        filename = f"{folder}/{self.name}.json"
        data = {
            "name": self.name,
            "n_cities": len(self.cities),
            "cities": self.cities
        }
        with open(filename, 'w') as f:
            json.dump(data, f)
        print(f"[DATA] Instance sauvegardée : {filename}")

    @staticmethod
    def load_from_json(filepath):
        """Charge une instance depuis un fichier JSON."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        instance = TSPInstance(name=data["name"], cities=[tuple(c) for c in data["cities"]])
        print(f"[DATA] Instance chargée : {data['name']} ({len(instance.cities)} villes)")
        return instance

    def plot(self, path=None, title_suffix=""):
        """Affiche les villes et un chemin optionnel."""
        cities_arr = np.array(self.cities)
        plt.figure(figsize=(8, 6))
        plt.scatter(cities_arr[:, 0], cities_arr[:, 1], c='red', zorder=2)

        # Annoter les villes
        for i, (x, y) in enumerate(self.cities):
            plt.text(x, y, str(i), fontsize=9)

        if path:
            # Tracer le chemin
            for k in range(len(path) - 1):
                i, j = path[k], path[k+1]
                p1, p2 = self.cities[i], self.cities[j]
                plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'b-', alpha=0.6, zorder=1)
            # Fermer la boucle
            p1, p2 = self.cities[path[-1]], self.cities[path[0]]
            plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'b-', alpha=0.6, zorder=1)

        plt.title(f"Instance: {self.name} {title_suffix}")
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    dataset = TSPInstance()
    dataset.generate_random(n_cities=20, seed=42)
    dataset.save_to_json()
    dataset.plot()
