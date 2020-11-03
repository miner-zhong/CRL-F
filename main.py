

'''

Implementation of CRL-F
Written by Miner Zhong.


first sepcify input data in args.py

    For Nguyen-Dupuis network:
    there are 4 observation scenarios for:
        all of these scenarios share the same set of observed link flow (30% of the entire link set)
        for observed trajectories:
        -- s1: All path sampling rate = 100%, and observed trajectories cover all possible transitions in the network
        -- s2: All path sampling rate = 100%
        -- s3: Among the entire path set, sampling rate range from 20%-40%
        -- s4: Among the entire path set, sampling rate range from 5%-20%

    For Berlin network:
    there are 4 observation scenarios for:
        all of these scenarios share the same set of observed link flow (30% of the entire link set)
        for observed trajectories:
        -- s1: Trajectory sampling rate =20%-40% + use real road features to define state features
        -- s2: Trajectory sampling rate =5%-20% + use real road features to define state features
        -- s3: Trajectory sampling rate =20%-40% + use one-hot vector of the state index to define state features)
        -- s4: Trajectory sampling rate =5%-20% + use one-hot vector of the state index to define state features)

then run this program

when finished, the output file contains state visit frequency & link flow estimation results

'''







import random
import torch
import numpy as np
np.set_printoptions(suppress=True)

from env_GridRoad import RoadEnv
from algo_rl.projection_oracle import ProjectionOracle
from algo_rl.rl_oracle_actor_critic import RL_Oracle as RL_Oracle_Actor_Critic
from algo_rl.nets import MLP
import algo_rl.solver as solver
import algo_rl.util as util
from get_results import *

import argparse
from args import appropo_args
parser = argparse.ArgumentParser()
appropo_args(parser)
args = parser.parse_args()

RANDOM_SEED = random.randint(0,1e+5)
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)





def proj_basic(p, args):
    p = np.ndarray.copy(p)

    # constraint-1: reward limit (reward limit for all OD pairs)
    bins1 = [args.maxlen * args.maxlen_increase * (-1)]
    threshold = bins1[0]
    if p[0] < threshold: p[0] = threshold


    # constraint-2: state visit expectation -- traffic count
    bins2 = np.array(args.obs_state_prob)
    ratio = args.obs_state_threshold
    d = p[1:1+len(args.obs_states)]-bins2
    if np.linalg.norm(d) > ratio:
        p[1:1+len(args.obs_states)] = (d/np.linalg.norm(d))*ratio+bins2


    # constraint-3: feature expectation matching -- trajectories
    bins3 = np.array(args.obs_feature_expectation)
    ratio = args.feature_expectation_threshold
    d = p[1+len(args.obs_states):]-bins3
    if np.linalg.norm(d) > ratio:
        p[1+len(args.obs_states):] = (d/np.linalg.norm(d))*ratio+bins3

    return p






def main():

    # initialization
    theta = np.zeros(1 + len(args.obs_states) + args.feature_length)

    env = RoadEnv(args)
    env.reset()

    net = MLP(env)
    if not (args.pretrained_net=='n/a'):
        net.load_state_dict(torch.load(args.pretrained_net))

    net = net.to(args.device)

    rl_oracle_generator = lambda: RL_Oracle_Actor_Critic(env=env, net=net, args=args, theta=theta)

    proj_oracle = ProjectionOracle(dim=theta.size, proj=proj_basic, args=args)


    # run the algorithm
    bins1 = [args.maxlen * args.maxlen_increase * (-1)]
    bins1 = np.array(bins1)
    bins2 = np.array(args.obs_state_prob)
    bins3 = np.array(args.obs_feature_expectation)

    best_policy = solver.run(target1=bins1,
                             target2=bins2,
                             target3=bins3,
                             proj_oracle=proj_oracle,
                             rl_oracle_generator=rl_oracle_generator,
                             args=args)

    solver.print_results(bins1, bins2, bins3, best_policy, args=args)

    solver.record_trajectories(policy=best_policy, n_traj = args.rl_traj,
                               target1=bins1, target2=bins2, target3=bins3,
                               args=args, path = args.output_1)


    # get svf results from recorded trajectories
    svf_true, svf_estimated = get_svf(trajs = args.trajectories, all_trajs_number = args.trajs_number_truth,
                                      nbstates = args.nb_state, nb_traj_each_policy = args.rl_traj,
                                      path_1 = args.output_1)
    svf_result = {}
    for i in range(args.nb_state):
        svf_result['s' + str(i)] = {'svf_true':svf_true[i], 'svf_estimated':svf_estimated[i]}


    # find scalar
    ratio = []
    for k in args.link_flow_obs.keys():
        state = 's' + k
        if (svf_result[state]['svf_estimated'] > args.scalar_tolerance):
            if (args.link_flow_obs[k] > 0):
                ratio.append(args.link_flow_obs[k] / svf_result[state]['svf_estimated'])
    scalar = sum(ratio) / len(ratio)


    # get estimation result
    estimation_result = {}
    for k in svf_result.keys():
        estimation_result[k] = svf_result[k]['svf_estimated'] * scalar


    # outputs
    output_svf(svf_true, svf_estimated, args)
    output_estimation(estimation_result, args)

    print('done')



if __name__ == "__main__":
    main()

