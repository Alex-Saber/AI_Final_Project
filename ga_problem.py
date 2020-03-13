from abc import ABC, abstractmethod

class ga_problems(ABC):
    def __init__(self, sta_dim, act_dim, goal):
        self.dim = sta_dim
        self.act_dim = act_dim
        self.goal = goal
        super().__init__()

    @abstractmethod
    def fit_func(self, sol):
        pass