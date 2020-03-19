''' CS 551 - AI, Winter 2020, PSU
    Alex Saber, Armando Lajara, Dawei Zhang
    March 12, 2020
'''
import numpy as np
import matplotlib.pyplot as plt
from ga_problem import ga_problems
from lunar_lander import ga_lunar_lander_problem

class genetic_agent:
    def __init__(self, problem, size, mut_pct, num_iter):
        self.size = size; self.dim = problem.dim; self.goal = problem.goal
        self.act_dim = problem.act_dim
        self.arr = np.zeros((size, self.dim), dtype=np.int)
        for i in range(size): # a random population
            self.arr[i,:] = np.random.choice(self.act_dim, self.dim)
        self.fit_func = problem.fit_func # fitness function
        self.mutp = mut_pct; self.niter = num_iter
        self.scores = [] # to track the scores

    def next_gen(self):
        # check the score, see if anyone hit the goal
        fit_score = np.apply_along_axis(self.fit_func, 1, self.arr)
        test = np.argmax(fit_score >= self.goal)
        self.scores.append(fit_score.max())
        best = (np.copy(self.arr[test]), self.scores[-1])
        # normalize the score into range (0 - 100)
        mn, mx = fit_score.min(), fit_score.max()
        fit_score = [0] if mx == mn else ((fit_score - mn) * 100 / (mx - mn)).astype(np.int)
        # create lottery poll for selection
        lottery_poll = []
        for i, s in enumerate(fit_score): 
            lottery_poll.extend([i for j in range(s)])
        # select nodes to a new population based on probabilities
        new_arr = np.zeros(self.arr.shape)
        for i in range(self.size):
            if len(lottery_poll): # we have a distribution pool
                dice = np.random.randint(0, len(lottery_poll))
                new_arr[i,:] = self.arr[lottery_poll[dice],:]
            else: # only one choice left
                new_arr[i,:] = self.arr[0,:]
        self.arr[:,:] = new_arr[:,:]
        # cross-over
        assert self.size % 2 == 0, "population size needs to be even"
        for i in range(self.size // 2):
            dice = np.random.randint(1, self.dim) # choose a random position to cross
            self.arr[i*2,dice:] = new_arr[i*2+1, dice:] # second half substitute
            self.arr[i*2+1,dice:] = new_arr[i*2, dice:] # first half substitute
        # mutation
        for i in range(self.size):
            # dice = np.random.rand() <= self.mutp
            # if not dice: continue # lucky, no mutation
            prob = np.random.binomial(1, self.mutp, self.dim) # create distribution based on mutp
            mut_arr = np.random.choice(self.act_dim, self.dim) # prepare a random array for mutation
            self.arr[i,:] = np.where(prob == 1, mut_arr, self.arr[i,:]) # mutate the actions where prob == 1
        return best

    def evolve(self):
        last_p = 0
        the_sol, the_i, the_max = None, 0, -np.inf
        for i in range(self.niter):
            sol, score = self.next_gen() # the only real work
            if score > the_max: the_sol, the_i, the_max = sol, i, score
            p = i*100//self.niter
            if p != last_p:
                if p % 10 == 0: print(f"{p}pct({max(self.scores):.1f})", end='')
                else: print(".", end='')
                last_p = p
        print("")
        return the_i, the_sol, the_max
