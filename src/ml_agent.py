import numpy as np
import random
import os
import json

class QLearningAgent:
    """
    Agent générique de Q-Learning.
    Gère une Q-Table pour mapper (État) -> (Meilleure Action).
    """
    def __init__(self, actions, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        self.actions = actions  # Liste des paramètres possibles [(alpha, beta, rho), ...]
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        
        # Q-Table : Lignes = États de stagnation (0 à 20), Colonnes = Actions
        # On limite la stagnation vue par l'IA à 20 (si >20, on considère que c'est pareil)
        self.n_states = 21 
        self.n_actions = len(actions)
        self.q_table = np.zeros((self.n_states, self.n_actions))
        
        self.last_state = 0
        self.last_action_idx = 0

    def get_state_index(self, stagnation_counter):
        """Borne l'état entre 0 et 20."""
        return min(stagnation_counter, self.n_states - 1)

    def choose_action(self, stagnation_counter):
        """Choisit une action selon la stratégie Epsilon-Greedy."""
        state = self.get_state_index(stagnation_counter)
        self.last_state = state

        # Exploration : Parfois on teste au hasard
        if np.random.uniform(0, 1) < self.epsilon:
            action_idx = np.random.choice(self.n_actions)
        # Exploitation : Sinon on prend la meilleure action connue
        else:
            action_idx = np.argmax(self.q_table[state])
        
        self.last_action_idx = action_idx
        return self.actions[action_idx]

    def learn(self, current_stagnation, reward):
        """Met à jour la Q-Table avec l'équation de Bellman."""
        next_state = self.get_state_index(current_stagnation)
        
        # Valeur actuelle
        current_q = self.q_table[self.last_state, self.last_action_idx]
        
        # Valeur cible (Reward + meilleure valeur future)
        max_future_q = np.max(self.q_table[next_state])
        new_q = current_q + self.lr * (reward + self.gamma * max_future_q - current_q)
        
        self.q_table[self.last_state, self.last_action_idx] = new_q

    def save_brain(self, filename="q_table.npy"):
        np.save(filename, self.q_table)

    def load_brain(self, filename="q_table.npy"):
        if os.path.exists(filename):
            self.q_table = np.load(filename)
            print(f"[ML] Cerveau chargé depuis {filename}")