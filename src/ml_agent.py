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
        self.n_states = 21 
        self.n_actions = len(actions)
        self.q_table = np.zeros((self.n_states, self.n_actions))
        
        self.last_state = 0
        self.last_action_idx = 0

    def get_state_index(self, stagnation_counter):
        """Borne l'état entre 0 et 20."""
        return min(stagnation_counter, self.n_states - 1)

    def choose_action(self, stagnation_counter):
        self.last_state = self.get_state_index(stagnation_counter)
        # Pour le moment, on renvoie toujours la première action par défaut
        self.last_action_idx = 0
        return self.actions[self.last_action_idx]

    def learn(self, current_stagnation, reward):
        # Implémenter la mise à jour des valeurs Q avec l'équation de Bellman
        pass

    def save_brain(self, filename="q_table.npy"):
        np.save(filename, self.q_table)

    def load_brain(self, filename="q_table.npy"):
        if os.path.exists(filename):
            self.q_table = np.load(filename)
            print(f"[ML] Cerveau chargé depuis {filename}")