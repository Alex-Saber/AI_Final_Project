from abc import ABC, abstractmethod

class ga_problems(ABC):
    def __init__(self, sta_dim, act_dim, goal):
        self.dim = sta_dim # state dimension
        self.act_dim = act_dim # action dimension
        self.goal = goal # goal score
        super().__init__()

    @abstractmethod
    def fit_func(self, sol):
        ''' a fit_func calculate what score a solution 'sol' gets
            depends on different problem, this calculation will be different
        '''
        pass