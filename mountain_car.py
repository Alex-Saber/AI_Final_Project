"""
http://incompleteideas.net/sutton/MountainCar/MountainCar1.cp
permalink: https://perma.cc/6Z2N-PFWC
"""

import math

import numpy as np

import gym
from gym import spaces
from gym.utils import seeding


class MountainCarEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, goal_velocity=0):
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.5
        self.goal_velocity = goal_velocity

        self.closest_reach = -1.2

        self.lowest_point = 1.0
        self.prev_velocity = 0.0

        self.force = 0.001
        self.gravity = 0.0025

        self.low = np.array([self.min_position, -self.max_speed], dtype=np.float32)
        self.high = np.array([self.max_position, self.max_speed], dtype=np.float32)

        self.viewer = None

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        position, velocity = self.state
        velocity += (action - 1) * self.force + math.cos(3 * position) * (-self.gravity)
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity
        position = np.clip(position, self.min_position, self.max_position)
        if (position == self.min_position and velocity < 0): velocity = 0

        done = bool(position >= self.goal_position and velocity >= self.goal_velocity)

        reward = 0

        if position == -1.2:
            if abs(velocity) - abs(self.prev_velocity) > 0.0001:
                reward += +1.0

            # print("reached")

            self.prev_velocity = velocity

        if position > 0.4 or position < -1.0:
            if position > self.closest_reach:
                reward += 1
                self.closest_reach = position

        # if position < self.lowest_point:
        #     self.lowest_point = position
        #     print(position)

        self.state = (position, velocity)
        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = np.array([self.np_random.uniform(low=-0.6, high=-0.4), 0])
        self.closest_reach = -1.2
        self.lowest_point = 1
        self.prev_velocity = 0
        return np.array(self.state)

    def _height(self, xs):
        return np.sin(3 * xs) * .45 + .55

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.max_position - self.min_position
        scale = screen_width / world_width
        carwidth = 40
        carheight = 20

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            xs = np.linspace(self.min_position, self.max_position, 100)
            ys = self._height(xs)
            xys = list(zip((xs - self.min_position) * scale, ys * scale))

            self.track = rendering.make_polyline(xys)
            self.track.set_linewidth(4)
            self.viewer.add_geom(self.track)

            clearance = 10

            l, r, t, b = -carwidth / 2, carwidth / 2, carheight, 0
            car = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            car.add_attr(rendering.Transform(translation=(0, clearance)))
            self.cartrans = rendering.Transform()
            car.add_attr(self.cartrans)
            self.viewer.add_geom(car)
            frontwheel = rendering.make_circle(carheight / 2.5)
            frontwheel.set_color(.5, .5, .5)
            frontwheel.add_attr(rendering.Transform(translation=(carwidth / 4, clearance)))
            frontwheel.add_attr(self.cartrans)
            self.viewer.add_geom(frontwheel)
            backwheel = rendering.make_circle(carheight / 2.5)
            backwheel.add_attr(rendering.Transform(translation=(-carwidth / 4, clearance)))
            backwheel.add_attr(self.cartrans)
            backwheel.set_color(.5, .5, .5)
            self.viewer.add_geom(backwheel)
            flagx = (self.goal_position - self.min_position) * scale
            flagy1 = self._height(self.goal_position) * scale
            flagy2 = flagy1 + 50
            flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
            self.viewer.add_geom(flagpole)
            flag = rendering.FilledPolygon([(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)])
            flag.set_color(.8, .8, 0)
            self.viewer.add_geom(flag)

        pos = self.state[0]
        self.cartrans.set_translation((pos - self.min_position) * scale, self._height(pos) * scale)
        self.cartrans.set_rotation(math.cos(3 * pos))

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def get_keys_to_action(self):
        return {(): 1, (276,): 0, (275,): 2, (275, 276): 1}  # control with left and right arrow keys

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


import ga
from ga_problem import ga_problems
import math

class ga_mountain_car_problem(ga_problems):
    def __init__(self, sta_dim, act_dim, goal):
        self.car = MountainCarEnv()
        # states division: 181, 141
        self.resolution = [19, 15]
        self.codec_base = [0, 1] # for later translation 'states <-> index'
        dim_chk = self.resolution[-1]
        for i in range(len(self.resolution)-2, -1, -1): # reversed order
            self.codec_base[i] = self.codec_base[i+1] * self.resolution[i+1]
            dim_chk *= self.resolution[i]
        assert dim_chk == sta_dim, "resolution not match dimension"
        super().__init__(sta_dim, act_dim, goal)

    def _state2index(self, s): # translate state "s" s[0-7] to ga index
        sd = [0] * 2
        sd[0] = int(s[0] * 10 + 12)
        sd[1] = int(s[1] * 100 + 7)

        for i in range(len(sd)):
            assert sd[i] >= 0 and sd[i] < self.resolution[i], f"cannot encode val[{i}]={s[i]}"

        s_int = 0 # translate from digitized state sequence to genetic string index
        for i in range(len(sd)):
            s_int += sd[i] * self.codec_base[i]
        return int(s_int)

    def _index2state(self, idx):
        pass # implement when you need it

    def ga_solution(self, env, s, sol):
        idx = self._state2index(s) # translate from state to index
        return sol[idx] # get action based on genetic string citizen

    def fit_func(self, sol, render=False): # calculate the score for one solution candidate (one citizen)
        assert len(sol) == self.dim, "wrong dimension solution node!"
        self.car.seed(55)
        total_reward = 0; steps = 0
        s = self.car.reset()
        while True:
            a = self.ga_solution(self.car, s, sol)
            s, r, done, info = self.car.step(a)
            total_reward += r

            if render:
                still_open = self.car.render()
                if still_open == False: break

            # if steps % 20 == 0 or done:
            #     print("observations:", " ".join(["{:+0.2f}".format(x) for x in s]))
            #     print("step {} total_reward {:+0.2f}".format(steps, total_reward))
            steps += 1
            if done or steps > 2000: break
        if render:
            self.car.close()
            print(f"step = {steps}, reward = {total_reward}")
        return total_reward

if __name__ == "__main__":

    import random
    import matplotlib.pyplot as plt

    # car.reset()
    # for iteration in range(0, 300):
    #     car.step(2)
    #     print(car.state)
    #     car.render()

    PopulationSize = [10, 20, 30]
    MutationPct = [0.1, 0.2]
    NumIterations = [30, 50, 70] # , 1200, 2000]

    # for mountain car
    problem = ga_mountain_car_problem(285, 3, 100)  # string length (state space), action range, target score (total rewards)
    for psz in PopulationSize:
        for mpt in MutationPct:
            for nit in NumIterations:
                print("==== population_size [{0}], mutation_pct [{1}], num_iters [{2}]".format(psz, mpt, nit))
                agent = ga.genetic_agent(problem, psz, mpt, nit)
                print(agent.arr[0])
                idx, sol, score = agent.evolve()
                print(f"max score this time = {score}")
                print(f"solution this time = {sol}")
                plt.clf()
                plt.title(f"PopulationSize={psz},MutationPct={mpt},NumIterations={nit}")
                plt.ylabel("Scores")
                plt.xlabel("Generations")
                plt.plot(agent.scores)
                plt.savefig(f"PopulationSize={psz},MutationPct={mpt},NumIterations={nit}.png") # plot to png file
                # plt.show()
                if score >= problem.goal:
                    print("    at iter [{0}] found solution {1}".format(idx, sol))
                    problem.fit_func(sol, True)
