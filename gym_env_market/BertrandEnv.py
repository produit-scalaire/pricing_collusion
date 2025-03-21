import gym
from gym import spaces
import numpy as np
from scipy.optimize import root_scalar
from collections import deque

class BertrandEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, n_firms=2, m=200, mu=0.25, k=1, a=1, a0=0, c=1, xi=0.1):
        super(BertrandEnv, self).__init__()

        self.k = k  # Longueur mémoire
        self.price_history = deque(maxlen=k)

        # Paramètres économiques
        self.n_firms = n_firms
        self.mu = mu
        self.a = a
        self.a0 = a0
        self.c = c
        self.xi = xi
        self.prices = np.ones(n_firms)

        # Calcul des prix de référence
        self.pN = self.calculate_nash_price()
        self.pM = self.calculate_monopoly_price()

        # Intervalle de prix avec xi=0.1
        self.min_price = self.pN - self.xi * (self.pM - self.pN)
        self.max_price = self.pM + self.xi * (self.pM - self.pN)
        print(self.min_price, self.max_price)
        # Espace d'action (prix discrets)
        self.m = m
        self.action_space = spaces.Discrete(m)

        # Espace d'observation (prix précédents)
        self.observation_space = spaces.MultiDiscrete([m]*n_firms*k)
        self.price_grid = np.linspace(self.min_price, self.max_price, m)
        print(self.price_grid)


        # Initialisation de l'état
        self.state = None
        self.reset()

    def calculate_nash_price(self):
        def f(p):
            exp_term = np.exp((self.a - p) / self.mu)
            denominator = self.n_firms * exp_term + np.exp(self.a0 / self.mu)
            return p - self.c - self.mu * (self.n_firms * exp_term + np.exp(self.a0 / self.mu)) / (
                    exp_term + np.exp(self.a0 / self.mu))

        # Intervalle de recherche réaliste pour pN
        result = root_scalar(f, bracket=[self.c + 0.1, self.c + 2 * self.mu], method='brentq')
        return result.root

    def calculate_monopoly_price(self):
        def f(p):
            exp_term = np.exp((self.a - p) / self.mu)
            total_demand = (self.n_firms * exp_term) / (self.n_firms * exp_term + np.exp(self.a0 / self.mu))
            derivative = (total_demand * (1 - total_demand)) / self.mu  # Dérivée correcte
            return (p - self.c) * derivative - total_demand

        # Intervalle de recherche réaliste pour pM
        result = root_scalar(f, bracket=[self.c + self.mu, self.c + 5 * self.mu], method='brentq')
        return result.root

    def logit_demand(self, prices):
        exponents = [(self.a - p) / self.mu for p in prices]
        denominator = sum(np.exp(e) for e in exponents) + np.exp(self.a0 / self.mu)
        return [np.exp(e) / denominator for e in exponents]

    def step(self, actions):
        # Conversion des actions discrètes en prix
        prices = [self.price_grid[a] for a in actions]

        self.prices = prices
        self.price_history.append(actions)
        # Calcul de la demande et des profits
        shares = self.logit_demand(prices)
        profits = [(p - self.c) * s for p, s in zip(prices, shares)]

        # Mise à jour de l'état
        self.state = np.array(prices)

        self.state = np.array([
            int((price - self.min_price) / (self.max_price - self.min_price) * (self.m - 1))
            for price in np.array(prices)
        ])

        # Récompenses, done, info
        return self.state, profits, False, {}

    def reset(self):
        self.price_history.clear()
        # Initialiser avec k périodes de prix aléatoires
        for _ in range(self.k):
            self.price_history.append(np.random.choice(self.action_space.n, self.n_firms))
        return self._get_state()

    def _get_state(self):
        # État = concaténation des k derniers prix (ex: k=1 -> [p1_t-1, p2_t-1])
        return np.array(self.price_history).flatten()

    def render(self, mode='human'):
        print(f"Current prices: {self.state}")

    def close(self):
        pass

