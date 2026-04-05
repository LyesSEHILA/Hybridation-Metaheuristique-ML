# Hybridation Métaheuristique + Machine Learning — Problème du Voyageur de Commerce (TSP)

Projet d'optimisation combinatoire combinant un algorithme de colonies de fourmis (ACO) avec un agent de Q-Learning pour adapter dynamiquement les hyperparamètres en cours d'exécution.

---

## Problème traité

Le **Problème du Voyageur de Commerce (TSP)** consiste à trouver le chemin le plus court passant par un ensemble de villes, en visitant chaque ville exactement une fois avant de revenir au point de départ. C'est un problème NP-difficile classique en optimisation combinatoire.

---

## Architecture du projet

```
├── src/
│   ├── tsp_data.py          # Génération et chargement des instances TSP
│   ├── aco_metrics.py       # Suivi des métriques (convergence, diversité, stagnation)
│   ├── aco_solver.py        # Solveur ACO classique (Ant System avec élitisme)
│   ├── ml_agent.py          # Agent Q-Learning (gestion de la Q-Table)
│   ├── hybrid_solver.py     # Solveur hybride ACO + Q-Learning
│   ├── benchmark.py         # Benchmark comparatif sur une instance (50 villes)
│   └── mass_benchmark.py    # Campagne de tests sur plusieurs tailles d'instances
├── datasets/
│   └── Random_20_S42.json   # Instance de test sauvegardée
├── tests/
│   └── graphs/              # Graphiques générés lors des expérimentations
├── requirements.txt
└── README.md
```

---

## Approche hybride

### ACO classique
L'algorithme ACO simule le comportement de fourmis qui déposent des phéromones sur les chemins parcourus. À chaque itération, les fourmis construisent des solutions en favorisant les arêtes à forte concentration de phéromones et à courte distance (compromis mémoire/visibilité contrôlé par `alpha` et `beta`). Un mécanisme d'évaporation (`decay`) évite la convergence prématurée.

### Agent Q-Learning
L'agent observe l'état de **stagnation** de l'algorithme (nombre d'itérations sans amélioration) et choisit parmi trois tactiques :

| Tactique | Alpha | Beta | Decay | Comportement |
|----------|-------|------|-------|--------------|
| 0 — Exploration | 1 | 5 | 0.50 | Oublie vite, suit la distance |
| 1 — Equilibré | 1 | 2 | 0.95 | Comportement classique |
| 2 — Intensification | 5 | 1 | 0.99 | Suit la mémoire, évapore peu |

L'agent apprend via l'**équation de Bellman** et une stratégie **epsilon-greedy** : il reçoit une récompense de +50 à chaque nouveau record, et -1 par itération sans amélioration.

---

## Résultats expérimentaux

Campagne de tests sur 10 exécutions indépendantes par taille d'instance :

| Taille | ACO Classique (moy.) | ACO Hybride (moy.) | Gain |
|--------|---------------------|--------------------|------|
| 20 villes | ~360 | ~361 | — |
| 50 villes | ~591 | ~569 | +3.7% |
| 100 villes | ~912 | ~850 | +6.8% |

**Observations :**
- Sur les petites instances, les deux algorithmes convergent vers le même optimum — le problème est trop simple pour que le Q-Learning apporte une valeur ajoutée.
- L'avantage de l'hybride s'accentue avec la taille du problème : +7% sur 100 villes avec un temps de calcul identique.
- L'hybride converge significativement plus vite (score optimal atteint dès l'itération ~20 contre ~120 pour le classique sur 50 villes).
- La diversité de la population chute plus vite dans l'hybride : l'agent bascule rapidement en intensification, ce qui peut être un risque sur des paysages très multimodaux.
- La Q-table n'atteint pas une politique stable en 150 itérations ; un pré-entraînement ou un budget plus élevé améliorerait la cohérence des choix de l'IA.

---

## Installation

```bash
git clone <url-du-repo>
cd Hybridation-Metaheuristique-ML
pip install -r requirements.txt
```

---

## Utilisation

```bash
cd src

# Benchmark comparatif (1 run, 50 villes, graphiques)
python benchmark.py

# Campagne de tests massifs (10 runs, 20/50/100 villes, export CSV)
python mass_benchmark.py

# Test rapide du solveur hybride seul
python hybrid_solver.py
```

---

## Dépendances

- `numpy` — calculs matriciels
- `matplotlib` — visualisation
- `pandas` — analyse des résultats
- `seaborn` — graphiques statistiques
