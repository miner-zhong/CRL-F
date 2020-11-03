import numpy as np
from algo_rl.util import calc_dist_1, calc_dist_2


class MixturePolicy:
    def __init__(self):
        self.loss_vec = []
        self.stats = []

        self.ls_trajectories = []

        self.ls_policies = {}


    def add_response(self, best_exp_rtn=None, TRAJ_LIST=None, policy_state_dict=None, POLICY_ID=None,
                     bins1=None, bins2=None, bins3=None, args=None, stats=None):

        self.loss_vec.append(best_exp_rtn)
        self.exp_rtn_of_avg_policy = np.average(np.stack(self.loss_vec, axis=0), axis=0)


        self.stats.append(np.hstack([stats['episode'],
                                     stats['num_trajs'],

                                     str(self.exp_rtn_of_avg_policy[:1].tolist()),
                                     calc_dist_1(self.exp_rtn_of_avg_policy[:1], bins1),

                                     str(self.exp_rtn_of_avg_policy[1:1+len(args.obs_states)].tolist()),
                                     calc_dist_2(self.exp_rtn_of_avg_policy[1:1+len(args.obs_states)], bins2),

                                     str(self.exp_rtn_of_avg_policy[1+len(args.obs_states):].tolist()),
                                     calc_dist_2(self.exp_rtn_of_avg_policy[1+len(args.obs_states):], bins3),

                                     0,
                                     stats['oracle_calls'],
                                     stats['cache_calls']]))

        self.ls_trajectories.append({'policy_id': POLICY_ID, 'traj_list': TRAJ_LIST, 'avg_measurement': best_exp_rtn})

        self.ls_policies['policy-' + str(POLICY_ID)] = {}
        self.ls_policies['policy-' + str(POLICY_ID)]['state_dic'] = policy_state_dict






