import numpy as np
from numpy import linalg as LA
from collections import defaultdict, namedtuple

import torch
import torch.nn as nn
import torch.optim
import torch.nn.functional
import torch.nn.functional as F
from torch.distributions import Categorical



SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


class RL_Oracle:
    def __init__(self, env=None, theta=None, net=None, args=None):
        super(RL_Oracle, self).__init__()
        self.env = env
        self.gamma = args.gamma
        self.lr = args.rl_lr
        self.saved_actions = []
        self.rewards = []
        self.entropies = []
        self.theta = theta
        self.eps = np.finfo(np.float32).eps.item()
        self.device = args.device

        self.net = net
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        # self.optimizer = torch.optim.RMSprop(self.net.parameters(), lr=self.lr, eps=1e-5, alpha=0.99)
        self.entropy_coef = args.entropy_coef
        self.value_coef = args.value_coef

        self.args=args


    def reset(self, normalize_theta=True):
        del self.saved_actions[:]
        del self.rewards[:]
        del self.entropies[:]



    def select_action(self, state, forbid_action):
        state = torch.from_numpy(state).float().to(self.device)
        action_scores, state_value = self.net(state)

        # action-masking
        if(len(forbid_action)>0):
            for a in forbid_action:
                action_scores[:, a] = 9999999999
        m = Categorical(logits=-action_scores)
        action = m.sample()

        self.saved_actions.append(SavedAction(m.log_prob(action), state_value))
        self.entropies.append(m.entropy())
        return action.item()




    def finish_episode(self, normalize_theta=None):
        R = 0
        saved_actions = self.saved_actions
        policy_losses = []
        value_losses = []
        returns = []
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        #returns = (returns - returns.mean()) / (returns.std() + self.eps)

        for (log_prob, value), R in zip(saved_actions, returns):
            advantage = R - value.item()
            policy_losses.append(-log_prob * advantage)
            R = torch.tensor([R]).to(self.device)
            value_losses.append(F.smooth_l1_loss(value, R.reshape(-1,1)))

        self.optimizer.zero_grad()

        loss = torch.stack(policy_losses).mean()\
             + (self.value_coef * torch.stack(value_losses).mean())\
             - (self.entropy_coef * torch.stack(self.entropies).mean())

        loss.backward()

        nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)

        self.optimizer.step()

        self.reset(normalize_theta=normalize_theta)



    def find_action(self, from_state, to_state):
        ls_action = []
        for a in range(self.env.nA):
            if(self.env.P[from_state][a][0][1] == to_state):
                ls_action.append(a)
        return ls_action


    def learn_policy(self, n_traj, n_iter, update=True, normalize_theta=True, cost=False, TRACK = None,
                     entropy_check_number=None, entropy_coef=None, entropy_threshod=None, entropy_increase=None):

        self.reset(normalize_theta=normalize_theta)
        sum_measurements = np.zeros(np.shape(self.theta))

        TRAJ_LIST = {}
        for n in range(n_traj):
            TRAJ_LIST['traj'+str(n)] = []

        for ind in range(n_traj):
            obs = self.env.reset()
            done = False
            traj_measurements = np.zeros(self.theta.size)
            MEASURE_LIST = []          # measurements

            for i in range(n_iter):
                current_state_location = self.env.state_from_repr_to_idx(obs)

                invalid_action = []      # if a non-absorbing state returns to itself, the action will be mask!!
                ls_absorbing = []
                for k in self.args.dest_absorb_info.keys():
                    ls_absorbing.append(self.args.dest_absorb_info[k])
                if not (current_state_location in ls_absorbing):
                    invalid_action = self.find_action(current_state_location, current_state_location)

                if(self.args.action_masking == 'YES'):
                    action = self.select_action(obs, forbid_action=invalid_action)
                else:
                    action = self.select_action(obs, forbid_action=[])

                obs, done, env_reward = self.env.step(action)

                next_state_location = self.env.state_from_repr_to_idx(obs)

                measurements = np.zeros(1 + len(self.args.obs_states) + self.args.feature_length)

                measurements[0] = env_reward              # measurement -- single reward

                for i in range(len(self.args.obs_states)):                  # measurement -- states visit
                    if(current_state_location == self.args.obs_states[i]):
                        measurements[1 + i] = 1

                measurements[1 + len(self.args.obs_states):] += self.args.state_feature[str(current_state_location)]     # measurement -- feature expectation

                MEASURE_LIST.append(measurements)

                traj_measurements = traj_measurements + measurements

                TRAJ_LIST['traj' + str(ind)].append(current_state_location)

                if done:
                    TRAJ_LIST['traj' + str(ind)].append(next_state_location)
                    break


            for m in range(len(MEASURE_LIST)):
                current_measure = MEASURE_LIST[m]
                current_theta = self.theta
                reward = np.dot(current_theta, current_measure)
                if cost:
                    reward = -reward
                self.rewards.append(reward)

            # given rl trainng reward, update policy
            if update:
                self.finish_episode(normalize_theta=normalize_theta)

            sum_measurements = sum_measurements+traj_measurements



        avg_measurements = np.zeros(self.theta.size)

        avg_measurements[0] = sum_measurements[0] / n_traj
        avg_measurements[1+len(self.args.obs_states):] = sum_measurements[1+len(self.args.obs_states):] / n_traj

        info = list(sum_measurements[1:1+len(self.args.obs_states)] / n_traj)
        info_normalized = []
        for a in range(len(info)):
            summ = sum(info) + 1e-10
            info_normalized.append(info[a] / summ)
        info_normalized = np.array(info_normalized)
        avg_measurements[1:1+len(self.args.obs_states)] = info_normalized


        # check the performance of current policy, if avg_measurement is simialr to several other previous policies, increase exploration
        nb = entropy_check_number
        if not (len(TRACK)==0):
            ls_check = []
            for i in range(nb):
                ls_check.append(False)
            for i in range(nb):
                if (len(TRACK) > i):
                    # compare Euclidean distance between avg_measurements
                    ls_check[i] = (LA.norm(TRACK[-1 * (i + 1)]-avg_measurements, ord=2) < entropy_threshod)
            check = True
            for i in range(nb):
                if (ls_check[i] == False):
                    check = False
            if (check):
                print('current net generate SIMILAR trajectories as the previous -- increase entropy')
                self.entropy_coef = self.entropy_coef + entropy_increase
            else:
                self.entropy_coef = entropy_coef
        # print("next entropy coef: " + str(self.entropy_coef))

        # state_dict of current policy
        new_state_dict = {}
        for key in self.net.state_dict():
            new_state_dict[key] = self.net.state_dict()[key].clone()

        return (avg_measurements, TRAJ_LIST, new_state_dict)


