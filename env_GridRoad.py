import numpy as np
from gym.utils import seeding



def categorical_sample(prob_n, np_random):
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return (csprob_n > np_random.rand()).argmax()


class DiscreteEnv():
    def __init__(self, nS, nA, P, isd):
        self.P = P
        self.isd = isd
        self.nS = nS
        self.nA = nA

        self.seed()
        self.s = categorical_sample(self.isd, self.np_random)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.s = categorical_sample(self.isd, self.np_random)
        return self.s

    def step(self, a):
        transitions = self.P[self.s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, d= transitions[i]
        self.s = s
        return (s, r, d, {"prob" : p})




class RoadEnv(DiscreteEnv):

    def __init__(self, args):

        self.nS = args.nb_state + len(args.ls_destination)
        self.nA = args.nb_action + 1


        isd = []
        for s in range(self.nS):
            isd.append(0)
        for origin in args.ls_origin:
            isd[origin] = args.isd[str(origin)]
        isd=np.array(isd)




        tran_info = args.tran_info

        ls_destinations = args.ls_destination
        ls_absorbing = []
        for key in args.dest_absorb_info.keys():
            ls_absorbing.append(args.dest_absorb_info[key])

        def find_next(current_state, action, is_dest):
            next = None
            FLAG = False
            if(is_dest):
                if(action == args.nb_action):
                    next = args.dest_absorb_info[str(current_state)]
                else:
                    info = tran_info[str(current_state)]
                    for pair in info['action_next_state_pairs']:
                        if(action == pair[0]):
                            next = pair[1]
                            FLAG = True
                    if not (FLAG):
                        next = current_state
            else:
                if(action == args.nb_action):
                    next = current_state
                else:
                    if (str(current_state) in list(tran_info.keys())):
                        info = tran_info[str(current_state)]
                        for pair in info['action_next_state_pairs']:
                            if(action == pair[0]):
                                next = pair[1]
                                FLAG = True
                        if not (FLAG):
                            next = current_state
                    else:
                        next = current_state
            return next


        # P[s][a] == [(probability, nextstate, reward, done), ...]
        P = {s: {a: [] for a in range(self.nA)} for s in range(self.nS)}

        # reward is -1 for each state (excluding origin & absorbing states)
        for s in range(self.nS):
            for a in range(self.nA):
                li = P[s][a]

                if (s in ls_absorbing):
                    next_s = s
                    li.append((1.0, next_s, 0, True))
                else:
                    if (s in ls_destinations):
                        next_s = find_next(current_state=s, action=a, is_dest=True)
                        li.append((1.0, next_s, -1, False))
                    else:
                        next_s = find_next(current_state=s, action=a, is_dest=False)
                        li.append((1.0, next_s, -1, False))

        self.transition=P

        super(RoadEnv, self).__init__(self.nS, self.nA, P, isd)




    def _state_repr(self, location):
        vec = np.zeros(self.nS)
        vec[location] = 1.0
        return np.array(vec)


    def state_from_repr_to_idx(self, repr):
        location = None
        size = len(repr)
        for i in range(size):
            if(repr[i]==1.0):
                location = i
        return location




    def step(self, action):
        state_idx, env_reward, done, info = DiscreteEnv.step(self, action)
        state = self._state_repr(state_idx)
        return state, done, env_reward



    def reset(self):
        state = DiscreteEnv.reset(self)
        return self._state_repr(state)




