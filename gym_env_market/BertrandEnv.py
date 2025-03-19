import gym
from gym import spaces
import numpy as np


class BertrandEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, n_firms=2, m=15, mu=0.25, a=1, a0=0, c=1, xi=0.1):
        super(BertrandEnv, self).__init__()

        # Paramètres économiques
        self.n_firms = n_firms
        self.mu = mu
        self.a = a
        self.a0 = a0
        self.c = c
        self.xi = xi

        # Calcul des prix de référence
        self.pN = self.calculate_nash_price()
        self.pM = self.calculate_monopoly_price()

        # Espace d'action (prix discrets)
        self.m = m
        self.action_space = spaces.Discrete(m)

        # Espace d'observation (prix précédents)
        self.observation_space = spaces.Box(
            low=np.array([self.pN - self.xi * (self.pM - self.pN)] * n_firms),
            high=np.array([self.pM + self.xi * (self.pM - self.pN)] * n_firms),
            dtype=np.float32
        )

        # Initialisation de l'état
        self.state = None
        self.reset()

    def calculate_nash_price(self):
        # Calcul simplifié du prix de Nash (à adapter selon le modèle exact)
        return self.c + 0.1  # Exemple

    def calculate_monopoly_price(self):
        # Calcul simplifié du prix de monopole
        return self.c + 0.5  # Exemple

    def logit_demand(self, prices):
        exponents = [(self.a - p) / self.mu for p in prices]
        denominator = sum(np.exp(e) for e in exponents) + np.exp(self.a0 / self.mu)
        return [np.exp(e) / denominator for e in exponents]

    def step(self, actions):
        # Conversion des actions discrètes en prix
        min_price = self.pN - self.xi * (self.pM - self.pN)
        max_price = self.pM + self.xi * (self.pM - self.pN)
        prices = [min_price + (max_price - min_price) * (a / (self.m - 1)) for a in actions]

        # Calcul de la demande et des profits
        shares = self.logit_demand(prices)
        profits = [(p - self.c) * s for p, s in zip(prices, shares)]

        # Mise à jour de l'état
        self.state = np.array(prices)

        # Récompenses, done, info
        return self.state, profits, False, {}

    def reset(self):
        # Initialisation aléatoire des prix
        min_price = self.pN - self.xi * (self.pM - self.pN)
        max_price = self.pM + self.xi * (self.pM - self.pN)
        self.state = np.random.uniform(low=min_price, high=max_price, size=self.n_firms)
        return self.state

    def render(self, mode='human'):
        print(f"Current prices: {self.state}")

    def close(self):
        pass

