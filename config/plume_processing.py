"""
Code for creating plume objects.
"""
from utilities import get_hit_rate
import numpy as np
import config


class IdealInfotaxisPlume(object):
    def __init__(self, src_pos):
        self.src_pos = np.array(src_pos)
        self.x_bounds = tuple(config.x_bounds)
        self.y_bounds = tuple(config.y_bounds)

    def sample(self, pos, t):
        xs_src = np.array([self.src_pos[0]])
        ys_src = np.array([self.src_pos[1]])

        hit_rate = get_hit_rate(xs_src, ys_src, pos=pos)[0, 0]

        mean_hits = hit_rate * config.dt
        sample = int(np.random.poisson(mean_hits) > 0)
        return sample

    def get_profile(self, grid):
        xs = np.linspace(*self.x_bounds, num=grid[0])
        ys = np.linspace(*self.y_bounds, num=grid[1])
        xs_src = np.array([self.src_pos[0]])
        ys_src = np.array([self.src_pos[1]])

        conc = np.nan * np.zeros((len(xs), len(ys)))
        for x_ctr, x in enumerate(xs):
            for y_ctr, y in enumerate(ys):
                hit_rate = get_hit_rate(xs_src, ys_src, pos=(x, y))[0, 0]
                conc[x_ctr, y_ctr] = hit_rate

        dx = np.mean(np.diff(xs))
        dy = np.mean(np.diff(ys))
        x_lim = [xs[0] - dx/2, xs[-1] + dx/2]
        y_lim = [ys[0] - dy/2, ys[-1] + dy/2]
        extent = x_lim + y_lim
        return conc, extent