''' CS 551 - AI, Winter 2020, PSU
    Alex Saber, Armando Lajara, Dawei Zhang
    March 12, 2020
'''
import numpy as np
import matplotlib.pyplot as plt
from ga_problem import ga_problems
from lunar_lander import ga_lunar_lander_problem

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
        self.act_dim = problem.act_dim
        self.arr = np.zeros((size, self.dim), dtype=np.int)
        for i in range(size):
            self.arr[i,:] = np.random.choice(problem.act_dim, self.dim)
        self.fit_func = problem.fit_func # fitness function
        self.mutp = mut_pct; self.niter = num_iter
        self.scores = []

    def next_gen(self):
        fit_score = np.apply_along_axis(self.fit_func, 1, self.arr)
        test = np.argmax(fit_score >= self.goal)
        self.scores.append(fit_score.max())
        if test: return self.arr[test]

        # # create lottery poll for selection
        # lottery_poll = []
        # for i, s in enumerate(fit_score): 
        #     lottery_poll.extend([i for j in range(s)])
        # # select nodes to a new population based on probabilities
        # new_arr = np.zeros(self.arr.shape)
        # for i in range(self.size):
        #     if len(lottery_poll):
        #         dice = np.random.randint(0, len(lottery_poll))
        #         new_arr[i,:] = self.arr[lottery_poll[dice],:]
        #     else:
        #         new_arr[i,:] = self.arr[0,:]
        # self.arr[:,:] = new_arr[:,:]

        # select nodes to a new population based on fitness
        top_n = self.size // 2 # TODO: better citizen didn't get more breed chance
        top_n = top_n + 1 if top_n % 2 else top_n
        order = np.argsort(fit_score)
        new_arr = np.zeros(self.arr.shape)
        for i, p in enumerate(reversed(order)):
            new_arr[i,:] = self.arr[p,:]
            if i >= top_n: break
        new_arr[top_n:,:] = new_arr[:top_n,:]
        np.random.shuffle(new_arr)
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
            dice1 = np.random.randint(0, self.dim)
            dice2 = np.random.randint(0, self.act_dim)
            n, k = dice2, (dice2+1)%self.act_dim
            self.arr[i,dice1] = k if self.arr[i,dice1] == n else n
        return []

    def evolve(self):
        last_p = 0
        for i in range(self.niter):
            result = self.next_gen()
            if len(result): return i, result
            p = i*100//self.niter
            if p != last_p:
                if p % 10 == 0: print(f"{p}", end='')
                else: print(".", end='')
                last_p = p
        print("")
        return None, None

    
if __name__ == '__main__':
    PopulationSize = [12, 20, 150, 500, 800]
    MutationPct = [0.4, 0.5, 0.9]
    NumIterations = [200, 800]
    # PopulationSize = [100, 1000]
    # MutationPct = [0.5]
    # NumIterations = [200, 1000]
    dim = 8
    #problem = eight_queen(dim, dim**2, (dim-1)*dim//2)
    # for lunar lander
    problem = ga_lunar_lander_problem(38400, 4, 250) # string length (state space), action range, target score (total rewards)
    for psz in PopulationSize:
        for mpt in MutationPct:
            for nit in NumIterations:
                print("==== population_size [{0}], mutation_pct [{1}], num_iters [{2}]".format(psz, mpt, nit))
                agent = genetic_agent(problem, psz, mpt, nit)
                print(agent.arr[0])
                idx, sol = agent.evolve()
                print("    scores = {0}".format(max(agent.scores)))
                if idx:
                    print("    at iter [{0}] found solution {1}".format(idx, sol))
                    print("    scores = {0}".format(agent.scores))
                    plt.plot(agent.scores)
                    plt.show()
                    problem.fit_func(sol, True)