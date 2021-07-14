import config
import numpy as np
from copy import copy
from scipy.special import k0
from scipy.stats import entropy as entropy_


"""
Construct a log-probability distribution from a prior type specifier.
Units are probability per area. (Not cell area but map area)    
"""
def build_log_src_prior(prior_type, xs, ys):
    shape = (len(xs), len(ys))
    dx = np.mean(np.diff(xs))
    dy = np.mean(np.diff(ys))

    if prior_type == 'uniform':
        log_p_unnormalized = np.ones(shape)
        norm_factor = np.sum(np.exp(log_p_unnormalized) * dx * dy)  # eS
        log_norm_factor = np.log(norm_factor)                       # 1+logS
        log_src_prior = log_p_unnormalized - log_norm_factor        # -logS
    else:
        raise NotImplementedError
    return log_src_prior                                            # log(1/area) unnormalized


"""
Return the turbulence length constant: sqrt( (d*tau) / (1 + (tau * w**2)/(4d) ) )
:param d: diffusivity coefficient (m^2/s)
:param w: wind speed (m/s)
:param tau: particle lifetime (s)
:return: length constant (m)
"""
def get_length_constant():
    d = config.diffusivity
    tau = config.p_lifetime
    num = d * tau
    denom = 1 + (tau * config.wind**2) / (4 * d)
    return np.sqrt(num / denom)


"""
Logarithm of modified bessel function of the second kind of order 0.
Infinite values may still be returned if argument is too close to
zero.
"""
def log_k0(x):
    y = k0(x)
    # if array
    try:
        logy = np.zeros(x.shape, dtype=float)
        # attempt to calculate bessel function for all elements in x
        logy[y!=0] = np.log(y[y!=0])
        # for zero-valued elements use approximation
        logy[y==0] = -x[y==0] - np.log(x[y==0]) + np.log(np.sqrt(np.pi/2))
        return logy
    except:
        if y == 0:
            return -x - np.log(x) + np.log(np.sqrt(np.pi/2))
        else:
            return np.log(y)


"""
Calculate hit rate at specified position.
This is given by Eq. 7 in the infotaxis paper supplementary materials:
    http://www.nature.com/nature/journal/v445/n7126/extref/nature05464-s1.pdf
:param xs: 1-D array of x-positions of source (m)
:param ys: 1-D array of y-positions of source (m)
:param pos: position where hit rate is calculated (m)
:return: grid of hit rates, with one value per source location
"""
def get_hit_rate(xs, ys, pos, resolution=0.000001):
    xs_src_, ys_src_ = np.meshgrid(xs, ys, indexing='ij')

    dx = pos[0] - xs_src_    #distance to x,y matrix of cells to point
    dy = pos[1] - ys_src_

    # round dx's and dy's less than resolution down to zero
    dx[np.abs(dx) < resolution] = 0
    dy[np.abs(dy) < resolution] = 0

    lam = get_length_constant()     # calc lambda
    scale_factor = config.emission_rate / np.log(lam/config.searcher_size)  # calc scale factor
    exp_term = np.exp((config.wind/(2*config.diffusivity))*dx)  # calc exponential term

    # calc bessel term
    abs_dist = (dx**2 + dy**2) ** 0.5
    bessel_term = np.exp(log_k0(abs_dist/lam))

    # calc final hit rate
    hit_rate = scale_factor * exp_term * bessel_term
    return hit_rate


"""
Update the log posterior over the src given sample h at position pos.
    :param pos: position
    :param xs: x-coords over which src prob is defined
    :param ys: y-coords over which src prob is defined
    :param h: sample value (0 for no hit, 1 for hit)
    :param log_p_src: previous estimate of log src posterior
    :return: new (unnormalized) log src posterior
"""
def update_log_p_src(pos, xs, ys, h, log_p_src):
    mean_hits = config.dt * get_hit_rate(xs, ys, pos) # get mean number of hits at pos given different src positions
    if h == 0:           # calculate log-likelihood (prob of h at pos given src position [Poisson])
        log_like = -mean_hits
    else:
        log_like = np.log(1 - np.exp(-mean_hits))

    # compute the new log src posterior
    log_p_src = log_like + log_p_src

    # set log prob to -inf everywhere within src_radius of pos
    # probability = 0
    xs_, ys_ = np.meshgrid(xs, ys, indexing='ij')
    mask = ((pos[0] - xs_)**2 + (pos[1] - ys_)**2 < config.src_radius**2)
    log_p_src[mask] = -np.inf

    # if we've exhausted the search space start over
    if np.all(np.isinf(log_p_src)):
        log_p_src = np.ones(log_p_src.shape)
    return log_p_src


"""
Wrapper around scipy.stats entropy function that takes in a 2D
log probability distribution.
:param log_p_src: 2D array of log probabilities
"""
def entropy(log_p_src):
    p_src = np.exp(log_p_src)           # convert to non-log probability distribution
    p_src /= p_src.sum()                # normalizes to 1
    return entropy_(p_src.flatten())    # calculate entropy


"""
Get the 5 possible moves from a position given a constant speed.
(left, right, forward, back, stay still)
:param pos:
:param step:
:return:
"""
def get_moves(pos, step):    
    moves = []
    # ["stop", "forward", "backward", "left", "right"]
    for dx, dy in [(0, 0), (step, 0), (-step, 0), (0, -step), (0, step)]:
        x = pos[0] + dx
        y = pos[1] + dy
        moves.append((x, y))
    return moves

"""
Return the probability that a position is within source "radius" given the source probability distribution.
:param pos: position to calc prob that you are close to source
:param xs: x-coords over which source prob distribution is defined
:param ys: y-coords over which source prob distribution is defined
:param log_p_src: log probability distribution over source position
:return: probability
"""
def get_p_src_found(pos, xs, ys, log_p_src):
    
    # get mask containing only points within radius of pos
    xs_, ys_ = np.meshgrid(xs, ys, indexing='ij')               #coordination matrix x,y of cells
    dxs = pos[0] - xs_                                          #distance to x,y matrix of cells to point
    dys = pos[1] - ys_

    mask = (dxs**2 + dys**2 < config.src_radius**2)             #cell has distance inside the radius is set true

    # sum probabilities contained in mask
    p_src = np.exp(log_p_src)                                   #return back to the probability map before logarith
    p_src /= p_src.sum()                                        #? chia cho 1?
    p_src_found = p_src[mask].sum()                             #total probabity the soucre inside the radius area

    return p_src_found


def get_p_sample(pos, xs, ys, h, log_p_src):
    """
    Get the probability of sampling h at position pos.
    :param pos: position
    :param xs: x-coords over which source prob distribution is defined
    :param ys: y-coords over which source prob distribution is defined
    :param h: sample value
    :param log_p_src: log probability distribution over source position
    :return: probability
    """
    # poisson probability of no hit given mean hit rate
    hit_rate = get_hit_rate(xs, ys, pos)     # hit rate map
    p_no_hits = np.exp(-config.dt * hit_rate)

    if h == 0:
        p_samples = p_no_hits
    elif h == 1:
        p_samples = 1 - p_no_hits
    else:
        raise Exception('h must be either 0 (no hit) or 1 (hit)')

    # get source distribution
    p_src = np.exp(log_p_src)
    p_src /= p_src.sum()            #normalize the distribution

    # make sure p_src being 0 wins over p_sample being nan/inf
    p_samples[p_src == 0] = 0

    # average over all source positions
    p_sample = np.sum(p_samples * p_src)
    return p_sample