''' CS 551 - AI, Winter 2020, PSU
    Dawei Zhang (dz4@pdx.edu)
    Feb 18, 2020
'''
import numpy as np
import matplotlib.pyplot as plt
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

class eight_queen(ga_problems):
    def fit_func(self, sol):
        assert len(sol) == self.dim, "wrong dimension solution node!"
        score = 0 # pair-wise non-attacking queens, max = 8*7/2
        for i in range(self.dim):
            for j in range(i+1, self.dim):
                ri, rj = sol[i] // self.dim, sol[j] // self.dim
                ci, cj = sol[i] % self.dim, sol[j] % self.dim
                d_attack = abs(ri - rj) == abs(ci - cj) # diagnal attack
                score += 0 if (ri==rj or ci==cj or d_attack) else 1 
        return score # mutual attacking score 2x->x

class genetic_agent:
    def __init__(self, problem, size, mut_pct, num_iter):
        self.size = size; self.dim = problem.dim; self.goal = problem.goal
        self.arr = np.zeros((size, self.dim))
        for i in range(size):
            self.arr[i,:] = np.random.choice(problem.act_dim, self.dim, replace = False)
        self.fit_func = problem.fit_func # fitness function
        self.mutp = mut_pct; self.niter = num_iter
        self.scores = []

    def next_gen(self):
        fit_score = np.apply_along_axis(self.fit_func, 1, self.arr)
        test = np.argmax(fit_score >= self.goal)
        self.scores.append(fit_score.max())
        if test: return self.arr[test]
        # create lottery poll for selection
        lottery_poll = []
        for i, s in enumerate(fit_score): 
            lottery_poll.extend([i for j in range(s)])
        # select nodes to a new population based on probabilities
        new_arr = np.zeros(self.arr.shape)
        for i in range(self.size):
            if len(lottery_poll):
                dice = np.random.randint(0, len(lottery_poll))
                new_arr[i,:] = self.arr[lottery_poll[dice],:]
            else:
                new_arr[i,:] = self.arr[0,:]
        self.arr[:,:] = new_arr[:,:]
        # cross-over
        assert self.size % 2 == 0, "population size needs to be even"
        for i in range(self.size // 2):
            dice = np.random.randint(1, self.dim)
            self.arr[i*2,dice:] = new_arr[i*2+1, dice:]
            self.arr[i*2+1,dice:] = new_arr[i*2, dice:]
        # mutation
        for i in range(self.size):
            dice = np.random.rand() <= self.mutp
            if not dice: continue # lucky, no mutation
            dice = np.random.randint(0, self.dim**2)
            m, n, k = dice//self.dim, dice, (dice+1)
            self.arr[i,m] = k if self.arr[i,m] == n else n
            # self.arr[i,m] = np.random.randint(0, self.dim)
        return []

    def evolve(self):
        for i in range(self.niter):
            result = self.next_gen()
            if len(result): return i, result
        return None, None

    
if __name__ == '__main__':
    PopulationSize = [4, 20, 150, 500, 800]
    MutationPct = [0.4, 0.5, 0.9]
    NumIterations = [200, 1000, 4000]
    # PopulationSize = [100, 1000]
    # MutationPct = [0.5]
    # NumIterations = [200, 1000]
    dim = 8
    problem = eight_queen(dim, dim**2, (dim-1)*dim//2)
    # for lunar lander
    # problem = lunar_lander(38400, 4, 280) # string length (state space), action range, target score (total rewards)
    for psz in PopulationSize:
        for mpt in MutationPct:
            for nit in NumIterations:
                print("==== population_size [{0}], mutation_pct [{1}], num_iters [{2}]".format(psz, mpt, nit))
                agent = genetic_agent(problem, psz, mpt, nit)
                print(agent.arr[0])
                idx, sol = agent.evolve()
                #print("    scores = {0}".format(agent.scores))
                if idx:
                    print("    at iter [{0}] found solution {1}".format(idx, sol))
                    print("    scores = {0}".format(agent.scores))
                    plt.plot(agent.scores)
                    plt.show()