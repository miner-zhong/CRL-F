from util import *


path = '.'
path_1 = './data/berlin'
path_2 = './data/berlin/s1'

f_network_info = path_1 + '/network_info.csv'
f_transition_info = path_1 + '/transition_info.csv'
f_od_info = path_1 + '/origin_destination_info.csv'
f_nn_distance = path_1 + '/node_to_node_distance_info.csv'
f_link_true = path_1 + '/link_flow_true.csv'

f_trajectories = path_2 + '/INPUT/trajectory.csv'
f_link_obs = path_2 + '/INPUT/link_flow_obs.csv'
f_features = path_2 + '/INPUT/state_feature.csv'

f_output_1 = path_2 + '/RESULT/recorded_trajectories.txt'
f_output_2 = path_2 + '/RESULT/svf_result.csv'
f_output_3 = path_2 + '/RESULT/estimation_result.csv'





def appropo_args(parser):
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument("--num_epochs", type=int, default=10)     # number of itertation

    # training parameters -- Solver
    parser.add_argument("--cache_size", type=int, default=5)       # number of policies revisited in each interation
    parser.add_argument('--atol_cache', type=float, default=0.0)   # tolerance for 'accured rewards of policy trained by RL in each interation must be at least equal to zero ('positive-response oracle')'
    parser.add_argument('--rtol_cache', type=float, default=1e-05)

    # training parameters -- Projection Oracle
    parser.add_argument("--olo_optim", choices=['adam', 'sgd'], default='sgd')    # gradient-decend method
    parser.add_argument("--proj_lr", type=float, default=1)          # learning rate
    parser.add_argument('--mx_size', type=int, default=20)           # value for transforming a convex set into a convex cone
    parser.add_argument("--pretrained_theta", type=str, default='n/a')

    # training parameters -- RL Oracle
    parser.add_argument("--action_masking", type=str, default='YES')  # or 'NO'
    parser.add_argument("--rl_traj", type=int, default=60)         # number of trajectories
    parser.add_argument("--rl_iter", type=int, default=20)         # number of steps
    parser.add_argument("--rl_lr", type=float, default=1e-3)       # learning rate
    parser.add_argument("--entropy_coef", type=float, default=0.001)        # weight of entropy in RL objective
    parser.add_argument("--entropy_check_number", type=float, default=2)    # to incereace exploration when 'stuck' (i.e., small changes in average-measurement between two policies)
    parser.add_argument("--entropy_threshold", type=float, default=0.0001)
    parser.add_argument("--entropy_increase", type=float, default=0.001)
    parser.add_argument("--value_coef", type=float, default=1.0)            # weight of value in RL objection
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G')   # discount factor
    parser.add_argument("--pretrained_net", type=str, default='n/a')


    # training parameters -- constraints
    parser.add_argument("--maxlen_increase", type=float, default=1.2)            # for constraint-1 (i.e., the maximum length of generated trajetcories should be less than 1.2 * max_len of observed trajectories )
    parser.add_argument("--obs_state_threshold", type=float, default=0.01)       # for constraint-2 (i.e., the distance between observed and estimated state visit frequency (calculated from link flow observations) should be less than 0.01)
    parser.add_argument("--feature_expectation_threshold", type=float, default=0.01)  # for constraint-3 for constraint-2 (i.e., the distance between observed and estimated state visit frequency (calculated from trajectory observations) should be less than 0.01)

    # scaling parameter
    parser.add_argument('--scalar_tolerance', type=float, default=1e-04)     # tolerance for calculating scaling factor



    # network initilization
    nb_state, nb_action, nb_feature = load_network_properties(path=f_network_info)

    parser.add_argument("--nb_state", type=int, default=nb_state)
    parser.add_argument("--nb_action", type=int, default=nb_action)

    tran_info = get_transition_info(path=f_transition_info)
    parser.add_argument('--tran_info', type=dict, default=tran_info)

    distance_info = get_distance_info(path=f_nn_distance)
    parser.add_argument('--distance_info', type=dict, default=distance_info)

    origin_info, destination_info = load_origin_dest_states_info(path=f_od_info)
    trajectories, trajs_number, trajs_number_truth = load_trajectories(path=f_trajectories)

    ls_origin = get_ls_origin(trajectories)
    parser.add_argument("--ls_origin", type=list, default=ls_origin)

    isd = get_isd(trajectories, trajs_number, ls_origin)
    parser.add_argument("--isd", type=dict, default=isd)

    ls_destination = get_ls_destination(trajectories)
    parser.add_argument("--ls_destination", type=list, default=ls_destination)

    dest_absorb_info = get_dest_absorb_info(ls_destination, nb_state)
    parser.add_argument("--dest_absorb_info", type=dict, default=dest_absorb_info)

    parser.add_argument("--origin_info", type=dict, default=origin_info)
    parser.add_argument("--destination_info", type=dict, default=destination_info)


    # observed trajectories
    parser.add_argument("--trajectories", type=dict, default=trajectories)
    parser.add_argument("--trajs_number_truth", type=dict, default=trajs_number_truth)

    # observed traffic counts
    counts_obs = load_traffic_count_1(path=f_link_obs)
    counts_true = load_traffic_count_2(path=f_link_true)
    obs_states, obs_state_prob = get_obs_state_prob(counts_obs)
    parser.add_argument("--obs_states", type=list, default=obs_states)
    parser.add_argument('--link_flow_obs', type=dict, default=counts_obs)
    parser.add_argument('--link_flow_true', type=dict, default=counts_true)


    # (c1) reward limit for each OD
    od_pairs = get_od_pairs(trajectories, origin_info, destination_info)
    maxlen = get_maxlen(trajectories, od_pairs, origin_info, destination_info)
    parser.add_argument("--maxlen", type=list, default=maxlen)


    # (c2) observed state visit probability
    parser.add_argument("--obs_state_prob", type=list, default=obs_state_prob)


    # (c3) observed state-action visit probability
    feature_length, state_feature = get_state_feature(nb_state, nb_feature, ls_destination, path=f_features)
    parser.add_argument("--feature_length", type=list, default=feature_length)
    parser.add_argument("--state_feature", type=list, default=state_feature)
    obs_feature_expectation = get_feature_expectation(trajectories, trajs_number, state_feature, feature_length)
    parser.add_argument("--obs_feature_expectation", type=list, default=obs_feature_expectation)


    # output
    parser.add_argument("--output_1", type=str, default=f_output_1)
    parser.add_argument("--output_2", type=str, default=f_output_2)
    parser.add_argument('--output_3', type=str, default=f_output_3)






