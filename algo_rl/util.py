import uuid
import copy
import torch
from collections import namedtuple
from numpy import linalg as LA


CacheItem = namedtuple('CacheItem', ['exp_rtn', 'policy', 'uuid', 'exp_TRAJS'])


def init_cache(rl_oracle_generator=None, args=None):
    cache_size = args.cache_size
    cache = []

    for _ in range(cache_size):
        rl_oracle = rl_oracle_generator()
        with torch.no_grad():
            [exp_rtn, TRAJ_LIST, paramm] = rl_oracle.learn_policy(n_traj=args.rl_traj,
                                                                  n_iter=args.rl_iter,
                                                                  update=False,
                                                                  cost=True,
                                                                  TRACK=[],
                                                                  entropy_check_number=args.entropy_check_number,
                                                                  entropy_coef=args.entropy_coef,
                                                                  entropy_threshod=args.entropy_threshold,
                                                                  entropy_increase=args.entropy_increase)
        cache.append(CacheItem(copy.deepcopy(exp_rtn),
                               rl_oracle.net.state_dict(),
                               uuid.uuid1(),
                               copy.deepcopy(TRAJ_LIST)))

    return cache




def calc_dist_1(x, bins):
    dist = []
    for i in range(len(bins)):
        dist.append(x[i]-bins[i])
    return dist

def calc_dist_2(x, bins):
    return LA.norm(x - bins, ord=2)

