import numpy as np


class QLearningAgent:
    def __init__(self, state_size, action_size, alpha=0.1, gamma=0.95,
                 epsilon_start=1.0, epsilon_decay=0.9995, epsilon_min=0.0):
        self.q_table = np.zeros((state_size, action_size))
        self.alpha = alpha  # Taux d'apprentissage
        self.gamma = gamma  # Facteur d'actualisation
        self.epsilon = epsilon_start  # Taux d'exploration initial
        self.epsilon_decay = epsilon_decay  # Taux de décroissance
        self.epsilon_min = epsilon_min  # Exploration minimale

    def choose_action(self, state):
        """Choix ε-greedy avec décroissance exponentielle"""
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.q_table.shape[1])  # Exploration
        return np.argmax(self.q_table[state])  # Exploitation

    def decay_epsilon(self):
        """Décroissance exponentielle de l'exploration"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update(self, state, action, reward, next_state):
        """Mise à jour Q-learning avec experience replay"""
        old_value = self.q_table[state, action]
        next_max = np.max(self.q_table[next_state])

        # Formule de mise à jour Q-learning
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        self.q_table[state, action] = new_value

        return new_value - old_value  # Retourne le delta pour monitoring

